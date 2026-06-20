"""Shared config builder for the examples — reads endpoints/keys from the env.

Set these before running any example:

    export PYRIS_VISION_BASE_URL="https://api.openai.com/v1"
    export PYRIS_VISION_API_KEY="sk-..."
    export PYRIS_VISION_MODEL="gpt-4o"
    # optional, only needed for audio / video-with-audio:
    export PYRIS_STT_BASE_URL="http://localhost:8000/v1"
    export PYRIS_STT_API_KEY="sk-..."
    export PYRIS_STT_MODEL="whisper-1"
"""
from __future__ import annotations

import os

from pyris import Config, SttConfig, VisionConfig


def config_from_env() -> Config:
    vision = VisionConfig(
        base_url=os.environ["PYRIS_VISION_BASE_URL"],
        api_key=os.environ["PYRIS_VISION_API_KEY"],
        model=os.environ.get("PYRIS_VISION_MODEL", "gemma4:26b"),
    )
    stt = None
    if os.environ.get("PYRIS_STT_BASE_URL"):
        stt = SttConfig(
            base_url=os.environ["PYRIS_STT_BASE_URL"],
            api_key=os.environ.get("PYRIS_STT_API_KEY", ""),
            model=os.environ.get("PYRIS_STT_MODEL", "whisper-1"),
        )
    return Config(vision=vision, stt=stt)
