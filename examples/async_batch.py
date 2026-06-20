"""Fan out over many files concurrently with the async API.

Each analyze_async offloads its ffmpeg work to a thread and awaits the LLM
calls, so a batch overlaps I/O instead of running strictly one-at-a-time.

    python examples/async_batch.py "Describe this" a.mp4 b.jpg c.mp3
"""
from __future__ import annotations

import asyncio
import sys

from pyris import FileProvider, Pyris, SubprocessFfmpeg

from _shared import config_from_env


async def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Describe this media."
    paths = sys.argv[2:] or ["clip.mp4"]

    config = config_from_env()
    pyris = Pyris.from_config(config)
    probe = SubprocessFfmpeg(config)

    async def run(path: str):
        provider = FileProvider(path, probe=probe)
        return path, await pyris.analyze_async(provider, prompt)

    for path, result in await asyncio.gather(*(run(p) for p in paths)):
        print(f"\n### {path}  [{result.mode.value}]")
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
