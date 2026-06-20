"""The media provider interface — Pyris's main extension point.

Implement this to teach Pyris a new source (YouTube, remote URL, S3, a frame
buffer from a camera, ...). The bundled one is ``providers.file.FileProvider``.

Contract recap (see ``RawMedia`` for the rationale): a provider FETCHES and
PROBES. It does not sample frames or extract audio — that's the core's job, so
you never reimplement ffmpeg here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .types import MediaInfo, RawMedia, TimeRange


class MediaProvider(ABC):
    @property
    def supports_range_fetch(self) -> bool:
        """If True, ``fetch(time_range=...)`` returns *only* that range and the
        core skips its own crop. A YouTube provider that can request byte/time
        ranges sets this True; a plain local-file provider leaves it False and
        lets the core crop with ffmpeg. This is the one capability flag that
        lets remote providers avoid downloading whole files.
        """
        return False

    @abstractmethod
    def probe(self) -> MediaInfo:
        """Cheap metadata lookup used for routing, ideally without a full fetch
        (e.g. an HTTP HEAD or a YouTube metadata call).
        """
        ...

    @abstractmethod
    def fetch(self, time_range: TimeRange | None = None) -> RawMedia:
        """Make the media available as a local artifact the core can run ffmpeg
        against. May download to a temp file; set ``RawMedia._cleanup`` so the
        core can release it. ``time_range`` is a hint honored only when
        ``supports_range_fetch`` is True.
        """
        ...
