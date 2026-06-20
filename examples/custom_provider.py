"""A fully-worked custom provider: fetch media from an HTTP(S) URL.

Shows the provider contract end-to-end. A provider only *fetches* and *probes* —
frame sampling and audio extraction stay in the core, so there's no ffmpeg here.
We download to a temp file and hand the core a cleanup callback via RawMedia.

    python examples/custom_provider.py https://example.com/clip.mp4 "What is this?"
"""
from __future__ import annotations

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from pyris import (
    FileProvider,  # noqa: F401  (imported for reference / comparison)
    MediaInfo,
    MediaProvider,
    Pyris,
    RawMedia,
    SubprocessFfmpeg,
    TimeRange,
)
from pyris.ffmpeg import MediaProbe

from _shared import config_from_env


class RemoteUrlProvider(MediaProvider):
    """Downloads a URL once, caches the temp file, and probes it locally."""

    def __init__(self, url: str, probe: MediaProbe) -> None:
        self._url = url
        self._probe = probe  # injected ffmpeg prober — provider stays ffmpeg-free
        self._local: Path | None = None
        self._info: MediaInfo | None = None

    # We can't serve arbitrary byte ranges from a plain URL, so let the core crop.
    @property
    def supports_range_fetch(self) -> bool:
        return False

    def _ensure_downloaded(self) -> Path:
        if self._local is None:
            suffix = os.path.splitext(self._url)[1] or ".bin"
            fd, name = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            urllib.request.urlretrieve(self._url, name)
            self._local = Path(name)
        return self._local

    def probe(self) -> MediaInfo:
        if self._info is None:
            self._info = self._probe.probe(self._ensure_downloaded())
        return self._info

    def fetch(self, time_range: TimeRange | None = None) -> RawMedia:
        path = self._ensure_downloaded()
        info = self.probe()

        def cleanup() -> None:
            try:
                path.unlink()
            except OSError:
                pass
            self._local = None

        return RawMedia(
            path=path, info=info, time_range=time_range, _cleanup=cleanup
        )


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com/clip.mp4"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this media."

    config = config_from_env()
    pyris = Pyris.from_config(config)
    provider = RemoteUrlProvider(url, probe=SubprocessFfmpeg(config))

    result = pyris.analyze(provider, prompt)  # core calls RawMedia.close() -> cleanup
    print(result.text)


if __name__ == "__main__":
    main()
