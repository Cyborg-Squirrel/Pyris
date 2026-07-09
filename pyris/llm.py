"""LLM client interfaces: the vision/reasoning model and the STT model.

Both are Protocols so you can drop in any OpenAI-compatible backend (the
default), a local faster-whisper, the Anthropic API, etc. The pipeline never
imports a concrete client directly.

Design decision (resolved from the design doc): the "driver" model is a
*reasoner*, not a *router*. Routing to vision-vs-STT is done deterministically
from media type (``pipeline.resolve_mode``), with no LLM round-trip. The model's
only job is to answer the prompt over the assembled content — which for combined
requests is a timestamp-interleaved transcript + frames. That's why there's no
tool-use/agent machinery here.

Transport is the standard library (``urllib``) so the package has zero runtime
dependencies. The async methods offload that blocking I/O to a worker thread via
``asyncio.to_thread`` — so ``analyze_async`` never blocks the event loop — which
is plenty for the single-shot image/audio/video calls Pyris makes.
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import SttConfig, VisionConfig
from .errors import TranscriptionError, VisionError
from .types import Frame, Transcript, TranscriptEvent, TranscriptSegment, Usage


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    frame: Frame  # carries its own timestamp for interleaving/labelling


PromptPart = TextPart | ImagePart


@dataclass
class LLMResponse:
    text: str
    usage: Usage


class VisionClient(Protocol):
    def complete(
        self, *, system: str, parts: Sequence[PromptPart], model: str
    ) -> LLMResponse:
        """Answer over an ordered sequence of text + image parts. The pipeline
        builds ``parts`` as an interleaved timeline so 'what is shown when X is
        said' is answerable. Batching across the model's image limit and any
        map-reduce live in the pipeline, not here.
        """
        ...

    async def complete_async(
        self, *, system: str, parts: Sequence[PromptPart], model: str
    ) -> LLMResponse: ...


class SttClient(Protocol):
    def transcribe(
        self, audio: Path, *, model: str, language: str | None = None
    ) -> Transcript:
        """Transcribe an audio file to timestamped segments."""
        ...

    async def transcribe_async(
        self, audio: Path, *, model: str, language: str | None = None
    ) -> Transcript: ...


class StreamingSttClient(Protocol):
    def stream_transcribe(
        self, audio: AsyncIterator[bytes], *, model: str, language: str | None = None
    ) -> AsyncIterator[TranscriptEvent]:
        """Consume a live stream of audio blobs and yield transcript events as
        they are recognized. ``Pyris.stream_stt`` drives the returned iterator
        and forwards each event to the caller's sink. Implementations are async
        generators (``async def`` + ``yield``); batching/windowing, if any, lives
        here, not in the pipeline.
        """
        ...


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _http_post_json(
    url: str, payload: dict, headers: dict[str, str], timeout: float
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={**headers, "Content-Type": "application/json"}
    )
    return _send(req, timeout)


def _send(req: urllib.request.Request, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise _HttpFailure(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise _HttpFailure(f"request failed: {exc.reason}") from exc


class _HttpFailure(Exception):
    """Internal: re-wrapped into Vision/Transcription errors by callers."""


class OpenAICompatibleVisionClient:
    """Targets the OpenAI ``/chat/completions`` shape (text + image_url parts)."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._url = _join_url(config.base_url, "chat/completions")
        self._timeout = 120.0  # overridden via with_timeout()

    def with_timeout(self, timeout: float) -> "OpenAICompatibleVisionClient":
        self._timeout = timeout
        return self

    def complete(
        self, *, system: str, parts: Sequence[PromptPart], model: str
    ) -> LLMResponse:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [_render_part(p) for p in parts]},
            ],
        }
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        try:
            data = _http_post_json(self._url, payload, headers, self._timeout)
        except _HttpFailure as exc:
            raise VisionError(str(exc)) from exc
        return _parse_chat_response(data)

    async def complete_async(
        self, *, system: str, parts: Sequence[PromptPart], model: str
    ) -> LLMResponse:
        return await asyncio.to_thread(
            self.complete, system=system, parts=parts, model=model
        )


class OpenAICompatibleSttClient:
    """Targets the OpenAI ``/audio/transcriptions`` shape, which faster-whisper
    and whisper.cpp servers both expose. Requests ``verbose_json`` to get
    timestamped segments and the detected language.
    """

    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._url = _join_url(config.base_url, "audio/transcriptions")
        self._timeout = 120.0

    def with_timeout(self, timeout: float) -> "OpenAICompatibleSttClient":
        self._timeout = timeout
        return self

    def transcribe(
        self, audio: Path, *, model: str, language: str | None = None
    ) -> Transcript:
        return self._post_transcription(audio.name, audio.read_bytes(), model, language)

    async def transcribe_async(
        self, audio: Path, *, model: str, language: str | None = None
    ) -> Transcript:
        return await asyncio.to_thread(
            self.transcribe, audio, model=model, language=language
        )

    async def stream_transcribe(
        self, audio: AsyncIterator[bytes], *, model: str, language: str | None = None
    ) -> AsyncIterator[TranscriptEvent]:
        """POST each incoming blob to ``/audio/transcriptions`` and emit its
        segments as they come back, shifted onto a running timeline so timestamps
        stay absolute across blobs. Each blob is transcribed on its own, so it
        must be an independently-decodable audio file (see
        :class:`~pyris.provider.AudioStreamProvider`).

        This endpoint is request/response, not wire-streaming, so every event is
        final; a websocket ASR backend implementing
        :class:`StreamingSttClient` would additionally emit partials. Each blob's
        last segment is marked ``end_of_batch`` so a consumer can tell blobs
        apart. The blocking POST is offloaded per blob so the event loop stays
        free to keep pulling audio and pushing results to the sink.
        """
        offset = 0.0
        async for blob in audio:
            transcript = await asyncio.to_thread(
                self._post_transcription, "audio.wav", blob, model, language
            )
            last = len(transcript.segments) - 1
            for index, seg in enumerate(transcript.segments):
                yield TranscriptEvent(
                    segment=TranscriptSegment(
                        start=seg.start + offset,
                        end=seg.end + offset,
                        text=seg.text,
                    ),
                    end_of_batch=index == last,
                )
            offset += max((s.end for s in transcript.segments), default=0.0)

    def _post_transcription(
        self, filename: str, data: bytes, model: str, language: str | None
    ) -> Transcript:
        fields = {"model": model, "response_format": "verbose_json"}
        if language:
            fields["language"] = language
        body, content_type = _encode_multipart_bytes(fields, filename, data)
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": content_type,
        }
        req = urllib.request.Request(self._url, data=body, headers=headers)
        try:
            resp = _send(req, self._timeout)
        except _HttpFailure as exc:
            raise TranscriptionError(str(exc)) from exc
        return _parse_transcription(resp)


# -- response/payload helpers -----------------------------------------------


def _render_part(part: PromptPart) -> dict:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        b64 = base64.b64encode(part.frame.image).decode("ascii")
        url = f"data:{part.frame.mime};base64,{b64}"
        return {"type": "image_url", "image_url": {"url": url}}
    raise VisionError(f"unsupported prompt part: {type(part).__name__}")


def _parse_chat_response(data: dict) -> LLMResponse:
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionError(f"unexpected chat response shape: {data}") from exc
    usage = data.get("usage") or {}
    return LLMResponse(
        text=text,
        usage=Usage(
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
        ),
    )


def _parse_transcription(data: dict) -> Transcript:
    segments = [
        TranscriptSegment(
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            text=str(s.get("text", "")),
        )
        for s in data.get("segments", [])
    ]
    if not segments and data.get("text"):
        # Servers returning plain text (no segments) still give us the words.
        segments = [TranscriptSegment(start=0.0, end=0.0, text=str(data["text"]))]
    return Transcript(segments=segments, language=data.get("language"))


def _encode_multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    return _encode_multipart_bytes(fields, file_path.name, file_path.read_bytes())


def _encode_multipart_bytes(
    fields: dict[str, str], filename: str, data: bytes
) -> tuple[bytes, str]:
    """Build a ``multipart/form-data`` body from in-memory bytes. Shared by the
    file path (``transcribe``) and the streaming path (``stream_transcribe``),
    which has only a blob and no file on disk."""
    boundary = uuid.uuid4().hex
    crlf = b"\r\n"
    out: list[bytes] = []
    for name, value in fields.items():
        out += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            value.encode("utf-8"),
        ]
    file_mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    out += [
        f"--{boundary}".encode(),
        (
            'Content-Disposition: form-data; name="file"; '
            f'filename="{filename}"'
        ).encode(),
        f"Content-Type: {file_mime}".encode(),
        b"",
        data,
        f"--{boundary}--".encode(),
        b"",
    ]
    return crlf.join(out), f"multipart/form-data; boundary={boundary}"
