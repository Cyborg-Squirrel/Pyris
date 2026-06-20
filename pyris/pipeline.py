"""The core orchestration — ties providers, ffmpeg, sampling, and the LLMs
together. This is the only stateful object a library user constructs.

Flow of ``analyze`` / ``analyze_async`` (same logic; the async path offloads
ffmpeg to a thread and awaits the async LLM methods):

    1. provider.probe()            -> MediaInfo
    2. resolve_mode(requested)     -> deterministic, no LLM
    3. provider.fetch(range)       -> RawMedia (local artifact)
    4. STT branch:  ffmpeg.extract_audio -> stt.transcribe -> Transcript
    5. vision branch: sampler.sample     -> budgeted Frames
    6. build_timeline(frames, transcript) -> interleaved PromptParts
    7. vision answer (batched map-reduce if over the image limit)
    8. RawMedia.close()            -> cleanup temps
"""
from __future__ import annotations

import asyncio
import mimetypes
from collections.abc import Sequence
from importlib import resources

from .config import Config
from .errors import ConfigError, UnsupportedMediaError
from .ffmpeg import Ffmpeg, SubprocessFfmpeg
from .llm import (
    ImagePart,
    LLMResponse,
    OpenAICompatibleSttClient,
    OpenAICompatibleVisionClient,
    PromptPart,
    SttClient,
    TextPart,
    VisionClient,
)
from .provider import MediaProvider
from .sampling import FrameSampler
from .types import (
    AnalysisResult,
    Frame,
    MediaInfo,
    MediaType,
    RawMedia,
    RequestMode,
    TimeRange,
    Transcript,
    Usage,
)

_MAP_TMPL = (
    "The following are ordered excerpts (frames and/or transcript) from one "
    'segment of a longer media file. Note everything relevant to answering: '
    '"{prompt}". Be concise and factual; do not speculate beyond what is shown.'
)
_REDUCE_TMPL = (
    "Below are ordered notes taken from consecutive segments of a single media "
    'file. Using only these notes, answer: "{prompt}"\n\nNotes:\n{notes}'
)


def resolve_mode(requested: RequestMode, info: MediaInfo) -> RequestMode:
    """Deterministic routing — no LLM round-trip.

    - IMAGE                -> VISION
    - AUDIO                -> STT
    - VIDEO + AUTO         -> VISION_AND_STT if it has an audio track, else VISION

    An explicit mode incompatible with the media (STT on an image, VISION on
    audio, STT on a silent video) raises ``UnsupportedMediaError`` rather than
    being silently changed.
    """
    if requested is RequestMode.AUTO:
        if info.media_type is MediaType.IMAGE:
            return RequestMode.VISION
        if info.media_type is MediaType.AUDIO:
            return RequestMode.STT
        return RequestMode.VISION_AND_STT if info.has_audio else RequestMode.VISION

    if requested is RequestMode.VISION:
        if not info.has_video:
            raise UnsupportedMediaError("VISION requested but media has no visual stream")
        return RequestMode.VISION
    if requested is RequestMode.STT:
        if not info.has_audio:
            raise UnsupportedMediaError("STT requested but media has no audio track")
        return RequestMode.STT
    if requested is RequestMode.VISION_AND_STT:
        if not info.has_video:
            raise UnsupportedMediaError("VISION_AND_STT requested but media has no visual stream")
        if not info.has_audio:
            raise UnsupportedMediaError("VISION_AND_STT requested but media has no audio track")
        return RequestMode.VISION_AND_STT
    raise UnsupportedMediaError(f"unknown request mode: {requested!r}")


def build_timeline(
    frames: Sequence[Frame], transcript: Transcript | None
) -> list[PromptPart]:
    """Merge frames and transcript segments into one timestamp-ordered list of
    prompt parts. This interleaving is what lets the model reason temporally —
    each frame is labelled with its timestamp and sits between the transcript
    segments that bracket it. On ties, frames sort before segments.
    """
    events: list[tuple[float, int, object]] = []
    for frame in frames:
        events.append((frame.timestamp, 0, frame))
    if transcript is not None:
        for seg in transcript.segments:
            events.append((seg.start, 1, seg))
    events.sort(key=lambda e: (e[0], e[1]))

    parts: list[PromptPart] = []
    for ts, kind, payload in events:
        if kind == 0:  # frame
            parts.append(TextPart(f"[frame @ {ts:.2f}s]"))
            parts.append(ImagePart(payload))  # type: ignore[arg-type]
        else:  # transcript segment
            parts.append(
                TextPart(f"[{payload.start:.2f}-{payload.end:.2f}s] {payload.text.strip()}")  # type: ignore[attr-defined]
            )
    return parts


class Pyris:
    def __init__(
        self,
        config: Config,
        *,
        ffmpeg: Ffmpeg,
        vision: VisionClient,
        stt: SttClient | None = None,
    ) -> None:
        self._config = config
        self._ffmpeg = ffmpeg
        self._vision = vision
        self._stt = stt
        self._sampler = FrameSampler(ffmpeg, config.sampling)

    @classmethod
    def from_config(cls, config: Config) -> "Pyris":
        """Build a Pyris with the bundled subprocess-ffmpeg + OpenAI-compatible
        clients. Validates config up front (fail fast)."""
        config.validate()
        ffmpeg = SubprocessFfmpeg(config)
        vision = OpenAICompatibleVisionClient(config.vision).with_timeout(
            config.request_timeout
        )
        stt: SttClient | None = None
        if config.stt is not None:
            stt = OpenAICompatibleSttClient(config.stt).with_timeout(
                config.request_timeout
            )
        return cls(config, ffmpeg=ffmpeg, vision=vision, stt=stt)

    # -- public API ----------------------------------------------------------

    def analyze(
        self,
        provider: MediaProvider,
        prompt: str,
        *,
        mode: RequestMode = RequestMode.AUTO,
        time_range: TimeRange | None = None,
        fps: float | None = None,
    ) -> AnalysisResult:
        """Run the full pipeline and return a structured result.

        ``prompt`` may be empty for pure-STT 'just transcribe' calls, in which
        case the synthesis LLM step is skipped and the transcript is returned
        as-is.
        """
        info = provider.probe()
        resolved = resolve_mode(mode, info)
        raw = provider.fetch(time_range)
        core_range = None if provider.supports_range_fetch else time_range
        try:
            transcript: Transcript | None = None
            frames: list[Frame] = []

            if resolved in (RequestMode.STT, RequestMode.VISION_AND_STT):
                transcript = self._transcribe(raw, core_range)
            if resolved in (RequestMode.VISION, RequestMode.VISION_AND_STT):
                frames = self._collect_frames(raw, info, core_range, fps)

            if resolved is RequestMode.STT and not prompt.strip():
                return self._transcript_only_result(transcript)

            parts = build_timeline(frames, transcript)
            system = self._load_system_prompt()
            resp = self._vision_answer(system, prompt, parts)
            return self._result(resolved, resp, transcript, len(frames))
        finally:
            raw.close()

    async def analyze_async(
        self,
        provider: MediaProvider,
        prompt: str,
        *,
        mode: RequestMode = RequestMode.AUTO,
        time_range: TimeRange | None = None,
        fps: float | None = None,
    ) -> AnalysisResult:
        """Async counterpart of :meth:`analyze`. Blocking ffmpeg work runs in a
        worker thread; LLM calls use the clients' async methods. Useful when a
        caller is fanning out many image/audio jobs on one event loop."""
        info = await asyncio.to_thread(provider.probe)
        resolved = resolve_mode(mode, info)
        raw = await asyncio.to_thread(provider.fetch, time_range)
        core_range = None if provider.supports_range_fetch else time_range
        try:
            transcript: Transcript | None = None
            frames: list[Frame] = []

            if resolved in (RequestMode.STT, RequestMode.VISION_AND_STT):
                transcript = await self._atranscribe(raw, core_range)
            if resolved in (RequestMode.VISION, RequestMode.VISION_AND_STT):
                frames = await asyncio.to_thread(
                    self._collect_frames, raw, info, core_range, fps
                )

            if resolved is RequestMode.STT and not prompt.strip():
                return self._transcript_only_result(transcript)

            parts = build_timeline(frames, transcript)
            system = self._load_system_prompt()
            resp = await self._avision_answer(system, prompt, parts)
            return self._result(resolved, resp, transcript, len(frames))
        finally:
            raw.close()

    # -- STT helpers ---------------------------------------------------------

    def _require_stt(self) -> SttClient:
        if self._stt is None or self._config.stt is None:
            raise ConfigError("STT requested but no STT client/config is set")
        return self._stt

    def _transcribe(self, raw: RawMedia, time_range: TimeRange | None) -> Transcript:
        stt = self._require_stt()
        audio = self._ffmpeg.extract_audio(raw.path, time_range=time_range)
        try:
            return stt.transcribe(
                audio,
                model=self._config.stt.model,  # type: ignore[union-attr]
                language=self._config.stt.language,  # type: ignore[union-attr]
            )
        finally:
            _silent_unlink(audio)

    async def _atranscribe(
        self, raw: RawMedia, time_range: TimeRange | None
    ) -> Transcript:
        stt = self._require_stt()
        audio = await asyncio.to_thread(
            self._ffmpeg.extract_audio, raw.path, time_range=time_range
        )
        try:
            return await stt.transcribe_async(
                audio,
                model=self._config.stt.model,  # type: ignore[union-attr]
                language=self._config.stt.language,  # type: ignore[union-attr]
            )
        finally:
            _silent_unlink(audio)

    # -- vision helpers ------------------------------------------------------

    def _collect_frames(
        self,
        raw: RawMedia,
        info: MediaInfo,
        time_range: TimeRange | None,
        fps: float | None,
    ) -> list[Frame]:
        if info.media_type is MediaType.IMAGE:
            return [self._load_single_image(raw)]
        return list(
            self._sampler.sample(
                raw.path,
                info,
                time_range=time_range,
                fps=fps,
                max_dim=self._config.vision.image_max_dim,
            )
        )

    def _load_single_image(self, raw: RawMedia) -> Frame:
        data = raw.path.read_bytes()
        return Frame(
            timestamp=0.0,
            image=data,
            mime=_image_mime(raw),
            width=raw.info.width or 0,
            height=raw.info.height or 0,
        )

    def _vision_answer(
        self, system: str, prompt: str, parts: Sequence[PromptPart]
    ) -> LLMResponse:
        model = self._config.vision.resolved_vision_model()
        batches = _batch_parts(parts, self._config.vision.max_images_per_request)
        image_total = sum(1 for p in parts if isinstance(p, ImagePart))

        if len(batches) <= 1:
            resp = self._vision.complete(
                system=system, parts=_with_prompt(prompt, parts), model=model
            )
            resp.usage.frames_sent = image_total
            return resp

        usage = Usage()
        partials: list[str] = []
        for batch in batches:
            r = self._vision.complete(
                system=system,
                parts=[TextPart(_MAP_TMPL.format(prompt=prompt)), *batch],
                model=model,
            )
            partials.append(r.text)
            usage = _add_usage(usage, r.usage)
        reduce = self._vision.complete(
            system=system, parts=[TextPart(_reduce_text(prompt, partials))], model=model
        )
        usage = _add_usage(usage, reduce.usage)
        usage.frames_sent = image_total
        return LLMResponse(text=reduce.text, usage=usage)

    async def _avision_answer(
        self, system: str, prompt: str, parts: Sequence[PromptPart]
    ) -> LLMResponse:
        model = self._config.vision.resolved_vision_model()
        batches = _batch_parts(parts, self._config.vision.max_images_per_request)
        image_total = sum(1 for p in parts if isinstance(p, ImagePart))

        if len(batches) <= 1:
            resp = await self._vision.complete_async(
                system=system, parts=_with_prompt(prompt, parts), model=model
            )
            resp.usage.frames_sent = image_total
            return resp

        # Map step runs the batches concurrently on the event loop.
        results = await asyncio.gather(
            *(
                self._vision.complete_async(
                    system=system,
                    parts=[TextPart(_MAP_TMPL.format(prompt=prompt)), *batch],
                    model=model,
                )
                for batch in batches
            )
        )
        usage = Usage()
        for r in results:
            usage = _add_usage(usage, r.usage)
        reduce = await self._vision.complete_async(
            system=system,
            parts=[TextPart(_reduce_text(prompt, [r.text for r in results]))],
            model=model,
        )
        usage = _add_usage(usage, reduce.usage)
        usage.frames_sent = image_total
        return LLMResponse(text=reduce.text, usage=usage)

    # -- result assembly -----------------------------------------------------

    def _result(
        self,
        mode: RequestMode,
        resp: LLMResponse,
        transcript: Transcript | None,
        frames_used: int,
    ) -> AnalysisResult:
        usage = resp.usage
        if transcript is not None:
            usage.audio_seconds = _transcript_seconds(transcript)
        return AnalysisResult(
            text=resp.text,
            mode=mode,
            model=self._config.vision.resolved_vision_model(),
            usage=usage,
            transcript=transcript,
            frames_used=frames_used,
        )

    def _transcript_only_result(self, transcript: Transcript | None) -> AnalysisResult:
        transcript = transcript or Transcript(segments=[])
        return AnalysisResult(
            text=transcript.full_text,
            mode=RequestMode.STT,
            model=self._config.stt.model if self._config.stt else "",
            usage=Usage(audio_seconds=_transcript_seconds(transcript)),
            transcript=transcript,
        )

    def _load_system_prompt(self) -> str:
        if self._config.system_prompt_path is not None:
            return self._config.system_prompt_path.read_text(encoding="utf-8")
        return resources.files("pyris.prompts").joinpath("default.md").read_text(
            encoding="utf-8"
        )


# -- module-level helpers ----------------------------------------------------


def _with_prompt(prompt: str, parts: Sequence[PromptPart]) -> list[PromptPart]:
    """Lead the message with the user's prompt, then the media timeline."""
    if prompt.strip():
        return [TextPart(prompt), *parts]
    return list(parts)


def _batch_parts(
    parts: Sequence[PromptPart], max_images: int
) -> list[list[PromptPart]]:
    """Split into batches each holding at most ``max_images`` images, preserving
    order. Text parts attach to the current batch so each frame keeps its label."""
    batches: list[list[PromptPart]] = []
    current: list[PromptPart] = []
    count = 0
    for part in parts:
        if isinstance(part, ImagePart):
            if count >= max_images:
                batches.append(current)
                current = []
                count = 0
            current.append(part)
            count += 1
        else:
            current.append(part)
    if current:
        batches.append(current)
    return batches or [[]]


def _reduce_text(prompt: str, partials: Sequence[str]) -> str:
    notes = "\n".join(f"{i + 1}. {t.strip()}" for i, t in enumerate(partials))
    return _REDUCE_TMPL.format(prompt=prompt, notes=notes)


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        frames_sent=a.frames_sent + b.frames_sent,
        audio_seconds=a.audio_seconds + b.audio_seconds,
    )


def _transcript_seconds(transcript: Transcript) -> float:
    return max((seg.end for seg in transcript.segments), default=0.0)


def _image_mime(raw: RawMedia) -> str:
    mime = raw.info.mime
    if mime and mime != "image/*":
        return mime
    guessed = mimetypes.guess_type(raw.path.name)[0]
    return guessed or "image/jpeg"


def _silent_unlink(path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
