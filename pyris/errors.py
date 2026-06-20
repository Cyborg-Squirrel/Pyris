"""Pyris exception hierarchy. Catch ``PyrisError`` to catch everything."""
from __future__ import annotations


class PyrisError(Exception):
    """Base class for all Pyris errors."""


class ConfigError(PyrisError):
    """Invalid or incomplete configuration (raised at startup, fail-fast)."""


class ProviderError(PyrisError):
    """A media provider failed to fetch or probe its media."""


class FfmpegError(PyrisError):
    """ffmpeg/ffprobe is missing, the wrong version, or exited non-zero."""


class UnsupportedMediaError(PyrisError):
    """The requested mode is incompatible with the media.

    e.g. STT requested on a silent image, or VISION on an audio-only file.
    """


class VisionError(PyrisError):
    """The vision LLM request failed."""


class TranscriptionError(PyrisError):
    """The STT/Whisper request failed."""
