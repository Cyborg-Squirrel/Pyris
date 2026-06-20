"""Configuration for Pyris.

Plain dataclasses here to keep the sketch dependency-light; in the real thing
these are good candidates for Pydantic models so you get validation + env-var
loading for free. Note vision and STT carry *separate* base URLs and keys —
they're frequently different providers (e.g. an OpenAI-compatible vision
endpoint plus a self-hosted faster-whisper server).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError


@dataclass
class VisionConfig:
    base_url: str
    api_key: str
    model: str  # the "driver" / reasoning model; must support vision...
    vision_model: str | None = None  # ...unless a dedicated vision model is set
    max_images_per_request: int = 20  # respect model image-count limits
    image_max_dim: int = 1024  # downscale frames before sending (cost/latency)

    def resolved_vision_model(self) -> str:
        """The model actually used for image calls (falls back to ``model``)."""
        return self.vision_model or self.model


@dataclass
class SttConfig:
    base_url: str
    api_key: str
    model: str  # e.g. "whisper-1" / the served whisper model name
    language: str | None = None  # None = autodetect


@dataclass
class SamplingConfig:
    """Frame-budget strategy. This is the knob that keeps long videos from
    blowing past context windows and cost — see ``pyris.sampling``.
    """

    default_fps: float = 1.0
    max_frames: int = 60  # hard cap regardless of fps/duration
    scene_detection: bool = True  # prefer scene-change frames over fixed cadence
    scene_threshold: float = 0.3
    dedup_near_identical: bool = True


@dataclass
class Config:
    vision: VisionConfig
    stt: SttConfig | None = None  # optional at startup, required at STT runtime
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    system_prompt_path: Path | None = None  # falls back to bundled default
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    temp_dir: Path | None = None  # None = system default
    request_timeout: float = 120.0
    max_retries: int = 2

    def validate(self) -> None:
        """Fail fast. Raises ``ConfigError`` on the first problem found."""
        if not self.vision.base_url:
            raise ConfigError("vision.base_url is required")
        if not self.vision.api_key:
            raise ConfigError("vision.api_key is required")
        if not self.vision.model:
            raise ConfigError("vision.model is required")
        if self.vision.max_images_per_request < 1:
            raise ConfigError("vision.max_images_per_request must be >= 1")
        if self.vision.image_max_dim < 1:
            raise ConfigError("vision.image_max_dim must be >= 1")

        if self.stt is not None:
            if not self.stt.base_url:
                raise ConfigError("stt.base_url is required when stt is set")
            if not self.stt.api_key:
                raise ConfigError("stt.api_key is required when stt is set")
            if not self.stt.model:
                raise ConfigError("stt.model is required when stt is set")

        if self.sampling.default_fps <= 0:
            raise ConfigError("sampling.default_fps must be > 0")
        if self.sampling.max_frames < 1:
            raise ConfigError("sampling.max_frames must be >= 1")

        if self.system_prompt_path is not None and not Path(
            self.system_prompt_path
        ).is_file():
            raise ConfigError(
                f"system_prompt_path does not exist: {self.system_prompt_path}"
            )
