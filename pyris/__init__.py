"""Pyris — a Python vision + STT library powered by vision LLMs and Whisper."""
from __future__ import annotations

from .config import Config, SamplingConfig, SttConfig, VisionConfig
from .errors import (
    ConfigError,
    FfmpegError,
    ProviderError,
    PyrisError,
    TranscriptionError,
    UnsupportedMediaError,
    VisionError,
)
from .ffmpeg import SubprocessFfmpeg
from .llm import OpenAICompatibleSttClient, OpenAICompatibleVisionClient
from .pipeline import Pyris, build_timeline, resolve_mode
from .provider import MediaProvider
from .providers import FileProvider
from .types import (
    AnalysisResult,
    Frame,
    MediaInfo,
    MediaType,
    RawMedia,
    RequestMode,
    TimeRange,
    Transcript,
    TranscriptSegment,
    Usage,
)

__all__ = [
    "Pyris",
    "resolve_mode",
    "build_timeline",
    "Config",
    "VisionConfig",
    "SttConfig",
    "SamplingConfig",
    "MediaProvider",
    "FileProvider",
    "SubprocessFfmpeg",
    "OpenAICompatibleVisionClient",
    "OpenAICompatibleSttClient",
    "RequestMode",
    "MediaType",
    "TimeRange",
    "MediaInfo",
    "RawMedia",
    "Frame",
    "Transcript",
    "TranscriptSegment",
    "Usage",
    "AnalysisResult",
    "PyrisError",
    "ConfigError",
    "ProviderError",
    "FfmpegError",
    "UnsupportedMediaError",
    "VisionError",
    "TranscriptionError",
]
