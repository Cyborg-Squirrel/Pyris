"""Shared test fakes so the core runs without ffmpeg or network."""
from __future__ import annotations

import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

from pyris.config import Config, SttConfig, VisionConfig
from pyris.llm import ImagePart, LLMResponse, PromptPart
from pyris.provider import MediaProvider
from pyris.types import (
    Frame,
    MediaInfo,
    MediaType,
    RawMedia,
    TimeRange,
    Transcript,
    TranscriptSegment,
    Usage,
)


def make_config(**overrides) -> Config:
    cfg = Config(
        vision=VisionConfig(
            base_url="http://vision.test/v1",
            api_key="k",
            model="driver",
            vision_model="vmodel",
        ),
        stt=SttConfig(base_url="http://stt.test/v1", api_key="k", model="whisper"),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_frame(ts: float) -> Frame:
    return Frame(timestamp=ts, image=b"\xff\xd8jpeg", mime="image/jpeg", width=4, height=4)


class FakeFfmpeg:
    """Implements the Ffmpeg protocol with no subprocess."""

    def __init__(self, *, frame_count: int = 3, info: MediaInfo | None = None):
        self.frame_count = frame_count
        self._info = info
        self.extract_calls: list[dict] = []

    def probe(self, path: Path) -> MediaInfo:
        return self._info or MediaInfo(
            MediaType.VIDEO, 10.0, True, True, 100, 100, "video/*"
        )

    def extract_frames(
        self, path, *, time_range, fps, scene_threshold, max_dim
    ) -> Iterator[Frame]:
        self.extract_calls.append(
            {"fps": fps, "scene_threshold": scene_threshold, "max_dim": max_dim}
        )
        for i in range(self.frame_count):
            yield make_frame(float(i))

    def extract_audio(self, path, *, time_range) -> Path:
        fd, name = tempfile.mkstemp(suffix=".wav")
        Path(name).write_bytes(b"RIFFfake")
        import os

        os.close(fd)
        return Path(name)


class FakeVision:
    def __init__(self, reply: str = "answer"):
        self.reply = reply
        self.calls: list[Sequence[PromptPart]] = []

    def complete(self, *, system, parts, model) -> LLMResponse:
        self.calls.append(list(parts))
        images = sum(1 for p in parts if isinstance(p, ImagePart))
        return LLMResponse(
            text=f"{self.reply}({images}img)",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

    async def complete_async(self, *, system, parts, model) -> LLMResponse:
        return self.complete(system=system, parts=parts, model=model)


class FakeStt:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, *, model, language=None) -> Transcript:
        self.calls += 1
        return Transcript(
            segments=[
                TranscriptSegment(0.0, 1.5, "hello"),
                TranscriptSegment(1.5, 3.0, "world"),
            ],
            language="en",
        )

    async def transcribe_async(self, audio, *, model, language=None) -> Transcript:
        return self.transcribe(audio, model=model, language=language)


class FakeProvider(MediaProvider):
    def __init__(self, info: MediaInfo, path: Path | None = None, *, range_fetch=False):
        self._info = info
        self._path = path or Path("/tmp/fake-media")
        self._range_fetch = range_fetch
        self.fetched_range: TimeRange | None = None

    @property
    def supports_range_fetch(self) -> bool:
        return self._range_fetch

    def probe(self) -> MediaInfo:
        return self._info

    def fetch(self, time_range: TimeRange | None = None) -> RawMedia:
        self.fetched_range = time_range
        return RawMedia(path=self._path, info=self._info, time_range=time_range)
