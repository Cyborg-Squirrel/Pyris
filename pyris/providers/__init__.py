"""Bundled media providers."""
from __future__ import annotations

from .file import FileProvider
from .streaming import CallbackTranscriptSink, WavFileAudioStreamProvider

__all__ = [
    "FileProvider",
    "WavFileAudioStreamProvider",
    "CallbackTranscriptSink",
]
