"""Frame-budget strategy — the layer that makes long videos tractable.

A 10-minute video at 1 fps is 600 frames: too many for any vision model's
image-count limit, too expensive, too much context. The sampler sits between
raw ffmpeg extraction and the vision client and enforces a budget:

  1. extract candidate frames (scene-change if enabled, else fixed fps; dedup
     of near-identical neighbours is delegated to ffmpeg's mpdecimate filter)
  2. if the result exceeds ``max_frames``, thin uniformly across the timeline so
     coverage stays even (don't just truncate the tail)

For fixed-cadence sampling it first lowers fps via ``_estimate_frame_count`` so
ffmpeg doesn't decode thousands of frames only for us to drop most. Scene-change
counts are unpredictable, so there we extract then thin.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .config import SamplingConfig
from .ffmpeg import FrameExtractor
from .types import Frame, MediaInfo, TimeRange


class FrameSampler:
    def __init__(self, ffmpeg: FrameExtractor, config: SamplingConfig) -> None:
        self._ffmpeg = ffmpeg
        self._config = config

    def sample(
        self,
        path: Path,
        info: MediaInfo,
        *,
        time_range: TimeRange | None,
        fps: float | None,
        max_dim: int,
    ) -> Iterator[Frame]:
        """Yield budgeted frames in timestamp order. ``fps=None`` uses the
        config default. ``max_dim`` (a vision-side setting) is forwarded to the
        extractor for downscaling. Honors ``max_frames`` as a hard cap.
        """
        cfg = self._config
        scene_threshold = cfg.scene_threshold if cfg.scene_detection else None

        effective_fps = fps if fps is not None else cfg.default_fps
        if scene_threshold is None:
            # Pre-shrink fps so a long clip doesn't over-extract.
            estimate = self._estimate_frame_count(info, time_range, effective_fps)
            if estimate > cfg.max_frames:
                span = self._span_seconds(info, time_range)
                if span and span > 0:
                    effective_fps = min(effective_fps, cfg.max_frames / span)

        frames = list(
            self._ffmpeg.extract_frames(
                path,
                time_range=time_range,
                fps=effective_fps,
                scene_threshold=scene_threshold,
                max_dim=max_dim,
            )
        )
        yield from _thin_uniform(frames, cfg.max_frames)

    def _estimate_frame_count(
        self, info: MediaInfo, time_range: TimeRange | None, fps: float
    ) -> int:
        """Upper-bound the fixed-cadence frame count so we can lower fps before
        extraction. Returns ``max_frames`` as a conservative guess when the span
        is unknown (e.g. a stream with no duration)."""
        span = self._span_seconds(info, time_range)
        if span is None:
            return self._config.max_frames
        return int(span * fps) + 1

    @staticmethod
    def _span_seconds(
        info: MediaInfo, time_range: TimeRange | None
    ) -> float | None:
        if time_range is not None and time_range.duration is not None:
            return time_range.duration
        if info.duration is None:
            return None
        start = time_range.start if time_range else 0.0
        return max(0.0, info.duration - start)


def _thin_uniform(frames: list[Frame], max_frames: int) -> Iterator[Frame]:
    """Keep at most ``max_frames``, spaced evenly across the list so we don't
    bias toward the start or end of the timeline."""
    n = len(frames)
    if n <= max_frames:
        yield from frames
        return
    if max_frames == 1:
        yield frames[n // 2]
        return
    # indices round(i * (n-1) / (max_frames-1)) give even coverage incl. ends.
    step = (n - 1) / (max_frames - 1)
    seen: set[int] = set()
    for i in range(max_frames):
        idx = round(i * step)
        if idx not in seen:
            seen.add(idx)
            yield frames[idx]
