"""Two ways to use audio:

  1. Pure transcription — pass an empty prompt; the LLM step is skipped and you
     get the raw transcript back.
  2. Ask a question about the audio — pass a prompt; Pyris transcribes, then has
     the driver model answer over the transcript.

    python examples/transcribe_audio.py path/to/podcast.mp3
    python examples/transcribe_audio.py path/to/podcast.mp3 "What are the 3 key takeaways?"
"""
from __future__ import annotations

import sys

from pyris import FileProvider, Pyris, SubprocessFfmpeg

from _shared import config_from_env


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "podcast.mp3"
    prompt = sys.argv[2] if len(sys.argv) > 2 else ""  # empty -> pure transcription

    config = config_from_env()
    if config.stt is None:
        raise SystemExit("Set PYRIS_STT_* env vars to use audio.")

    pyris = Pyris.from_config(config)
    provider = FileProvider(path, probe=SubprocessFfmpeg(config))

    result = pyris.analyze(provider, prompt)

    if prompt:
        print("Answer:\n", result.text, "\n")
    print("Transcript:\n", result.transcript.full_text if result.transcript else "")


if __name__ == "__main__":
    main()
