"""Streaming STT: pipe a live-ish audio stream to the STT host and print each
transcript event as it arrives — instead of waiting for the whole file.

The input side is any ``AudioStreamProvider`` (here the bundled WAV replayer,
paced to real time so it behaves like a mic); the output side is any
``TranscriptSink`` (here a callback that prints each segment). Swap either for
your own — a websocket source, a subtitle-file sink — without touching the core.

    python examples/stream_transcribe.py path/to/podcast.wav

Note: the bundled provider yields WAV windows, so give it a .wav (convert with
`ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav`).
"""
from __future__ import annotations

import asyncio
import sys

from pyris import CallbackTranscriptSink, Pyris, WavFileAudioStreamProvider

from _shared import config_from_env


async def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "podcast.wav"

    config = config_from_env()
    if config.stt is None:
        raise SystemExit("Set PYRIS_STT_* env vars to use audio.")

    pyris = Pyris.from_config(config)
    source = WavFileAudioStreamProvider(path, window_seconds=5.0, realtime=True)

    def show(event) -> None:
        seg = event.segment
        marker = " " if event.is_final else "~"  # ~ = partial (revisable)
        print(f"[{seg.start:7.2f}-{seg.end:7.2f}]{marker}{seg.text.strip()}", flush=True)

    sink = CallbackTranscriptSink(show)

    transcript = await pyris.stream_stt(source, sink)

    print("\nFull transcript:\n", transcript.full_text)


if __name__ == "__main__":
    asyncio.run(main())
