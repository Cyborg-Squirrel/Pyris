"""The media provider interface — Pyris's main extension point.

Implement this to teach Pyris a new source (YouTube, remote URL, S3, a frame
buffer from a camera, ...). The bundled one is ``providers.file.FileProvider``.

Contract recap (see ``RawMedia`` for the rationale): a provider FETCHES and
PROBES. It does not sample frames or extract audio — that's the core's job, so
you never reimplement ffmpeg here.

For live audio there are two more extension points, used by ``Pyris.stream_stt``:
:class:`AudioStreamProvider` (the *input* side — yields audio as it arrives) and
:class:`TranscriptSink` (the *output* side — receives transcript events as they
are produced). Bundled implementations live in ``providers.streaming``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .types import MediaInfo, RawMedia, TimeRange, TranscriptEvent


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


class AudioStreamProvider(ABC):
    """The streaming, audio-only counterpart to :class:`MediaProvider`.

    Where ``MediaProvider`` fetches a whole file up front, this yields audio *as
    it arrives* — a microphone, a phone call, a websocket, an ffmpeg pipe — which
    ``Pyris.stream_stt`` pipes straight to the STT host. There is deliberately no
    ``probe``: a live stream has no known duration, and the mode is fixed (STT).

    Contract: each yielded blob must be **independently decodable** by the STT
    host (e.g. a self-contained WAV/Opus fragment your capture pipeline cuts on a
    container boundary), because the bundled client transcribes one blob at a
    time. A bidirectional ASR backend that accepts raw frames can pair with its
    own :class:`~pyris.llm.StreamingSttClient` and relax this.
    """

    @abstractmethod
    def stream(self) -> AsyncIterator[bytes]:
        """Return an async iterator over audio blobs. Implemented as an ``async
        def`` generator that ``yield``s ``bytes``; it is iterated exactly once."""
        ...


class TranscriptSink(ABC):
    """The output side of streaming STT — the "second provider".

    Receives every :class:`~pyris.types.TranscriptEvent` the STT host produces,
    in order, the moment it is produced. This is where live results go: a
    websocket back to a browser, a growing subtitle file, stdout. Kept separate
    from the input so the same source can drive different sinks.
    """

    @abstractmethod
    async def emit(self, event: TranscriptEvent) -> None:
        """Handle a single event. Called sequentially, so ordering is preserved
        and a slow sink naturally backpressures the stream."""
        ...

    async def aclose(self) -> None:
        """Called once after the last event, including if the stream errors.
        Override to flush or close an underlying resource; the default no-ops."""
        return None
