"""Core data types shared across Pyris.

These are deliberately transport-agnostic: providers, the ffmpeg layer, and the
LLM clients all speak in terms of these types, so each piece stays swappable and
unit-testable in isolation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MediaType(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class RequestMode(str, Enum):
    """What the caller wants done with the media.

    AUTO lets Pyris pick deterministically from the probed media type
    (see ``pyris.pipeline.resolve_mode``). The explicit modes override that and
    are validated against the media rather than silently re-routed.
    """

    AUTO = "auto"
    VISION = "vision"
    STT = "stt"
    VISION_AND_STT = "vision_and_stt"


@dataclass(frozen=True)
class TimeRange:
    """Half-open ``[start, end)`` in seconds. ``end=None`` means 'to the end'."""

    start: float = 0.0
    end: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"TimeRange.start must be >= 0, got {self.start}")
        if self.end is not None and self.end <= self.start:
            raise ValueError(
                f"TimeRange.end ({self.end}) must be > start ({self.start})"
            )

    @property
    def duration(self) -> float | None:
        """Length in seconds, or None when the range is open-ended."""
        if self.end is None:
            return None
        return self.end - self.start


@dataclass
class MediaInfo:
    """Result of probing media (ffprobe). Drives routing and validation."""

    media_type: MediaType
    duration: float | None  # seconds; None for still images
    has_audio: bool
    has_video: bool
    width: int | None
    height: int | None
    mime: str | None


@dataclass
class RawMedia:
    """What a provider hands back: a locally-readable artifact plus probe info.

    Design note (the provider/core boundary): a provider's job *ends here*. It
    fetches bytes and makes them available as a local path; it does NOT sample
    frames or extract audio. Frame sampling and audio extraction are the core's
    job (``pyris.ffmpeg`` + ``pyris.sampling``), so a custom provider — YouTube,
    remote URL, S3 — never reimplements ffmpeg logic. It only answers "give me
    the bytes for this time range."
    """

    path: Path
    info: MediaInfo
    time_range: TimeRange | None  # the range this artifact already represents
    _cleanup: Callable[[], None] | None = None  # e.g. delete a downloaded temp

    def close(self) -> None:
        """Release any temp resources. The core calls this when done."""
        if self._cleanup is not None:
            self._cleanup()


@dataclass
class Frame:
    """A single sampled video frame. Frames flow through an iterator and are
    consumed (encoded + sent, then dropped) so we never hold the whole video in
    memory — see ``pyris.sampling.FrameSampler``.
    """

    timestamp: float  # seconds into the original media
    image: bytes  # encoded image payload (e.g. JPEG)
    mime: str
    width: int
    height: int


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[TranscriptSegment]
    language: str | None = None

    @property
    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments).strip()


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    frames_sent: int = 0
    audio_seconds: float = 0.0


@dataclass
class AnalysisResult:
    """The single thing ``Pyris.analyze`` returns. Structured (not just a
    string) so callers get traceability: which frames/segments backed the answer.
    """

    text: str  # the synthesized answer to the caller's prompt
    mode: RequestMode  # the mode actually run (after AUTO resolution)
    model: str
    usage: Usage
    transcript: Transcript | None = None  # present for STT / VISION_AND_STT
    frames_used: int = 0
