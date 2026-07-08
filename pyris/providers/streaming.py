"""Bundled streaming helpers for ``Pyris.stream_stt``.

- :class:`WavFileAudioStreamProvider` — a reference *input* provider that replays
  a WAV file as a live stream of standalone WAV windows. It exists so the
  streaming path is runnable end to end without a microphone; a real app swaps in
  a mic/socket/ffmpeg-pipe provider.
- :class:`CallbackTranscriptSink` — a reference *output* sink that forwards each
  event to a callback (sync or async): the simplest way to wire results into an
  app (print captions, push to a websocket, append to an .srt).

Both use only the standard library (``wave``), keeping Pyris dependency-free.
"""
from __future__ import annotations

import asyncio
import inspect
import io
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from ..errors import ProviderError
from ..provider import AudioStreamProvider, TranscriptSink
from ..types import TranscriptEvent


class WavFileAudioStreamProvider(AudioStreamProvider):
    """Replay a WAV file as a stream of fixed-length windows.

    Each window is re-wrapped as a *complete, standalone WAV* (header included),
    so it satisfies the ``AudioStreamProvider`` contract that every blob be
    independently decodable by the STT host. With ``realtime=True`` the stream is
    paced to wall-clock — each window is held back for roughly its own duration —
    which simulates live capture (useful for demos and backpressure testing).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        window_seconds: float = 5.0,
        realtime: bool = False,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        self._path = Path(path)
        self._window_seconds = window_seconds
        self._realtime = realtime

    async def stream(self) -> AsyncIterator[bytes]:
        if not self._path.is_file():
            raise ProviderError(f"file not found: {self._path}")
        try:
            with wave.open(str(self._path), "rb") as wav:
                nchannels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                frames_per_window = max(1, int(framerate * self._window_seconds))
                while True:
                    frames = wav.readframes(frames_per_window)
                    if not frames:
                        break
                    yield _wrap_wav(nchannels, sampwidth, framerate, frames)
                    if self._realtime:
                        nframes = len(frames) // (sampwidth * nchannels)
                        await asyncio.sleep(nframes / framerate)
        except wave.Error as exc:
            raise ProviderError(f"not a readable WAV file: {self._path} ({exc})") from exc


def _wrap_wav(nchannels: int, sampwidth: int, framerate: int, frames: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(nchannels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.writeframes(frames)
    return buf.getvalue()


class CallbackTranscriptSink(TranscriptSink):
    """Forward each event to ``on_event`` (and call ``on_close`` at the end).

    Both callbacks may be plain or ``async`` — an awaitable return value is
    awaited — so the same sink works from sync glue code or an event loop.
    """

    def __init__(
        self,
        on_event: Callable[[TranscriptEvent], Awaitable[None] | None],
        *,
        on_close: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        self._on_event = on_event
        self._on_close = on_close

    async def emit(self, event: TranscriptEvent) -> None:
        await _maybe_await(self._on_event(event))

    async def aclose(self) -> None:
        if self._on_close is not None:
            await _maybe_await(self._on_close())


async def _maybe_await(result: Awaitable[None] | None) -> None:
    if inspect.isawaitable(result):
        await result
