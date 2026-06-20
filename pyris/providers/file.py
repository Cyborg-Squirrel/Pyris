"""Local-file provider — the bundled default.

The simplest possible provider: the bytes are already on disk, so ``fetch``
just probes and wraps the existing path. No range fetch (the core crops with
ffmpeg), no temp cleanup (we didn't create the file).
"""
from __future__ import annotations

from pathlib import Path

from ..errors import ProviderError
from ..ffmpeg import MediaProbe
from ..provider import MediaProvider
from ..types import MediaInfo, RawMedia, TimeRange


class FileProvider(MediaProvider):
    def __init__(self, path: str | Path, probe: MediaProbe) -> None:
        self._path = Path(path)
        self._probe = probe  # injected so the provider doesn't own ffmpeg either
        self._info: MediaInfo | None = None

    def probe(self) -> MediaInfo:
        if self._info is None:
            if not self._path.is_file():
                raise ProviderError(f"file not found: {self._path}")
            self._info = self._probe.probe(self._path)
        return self._info

    def fetch(self, time_range: TimeRange | None = None) -> RawMedia:
        # supports_range_fetch is False, so time_range is recorded but not
        # applied here; the core crops with ffmpeg. We hand back the path + info.
        info = self.probe()
        return RawMedia(path=self._path, info=info, time_range=time_range)
