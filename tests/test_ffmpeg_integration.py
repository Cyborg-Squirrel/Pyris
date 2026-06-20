"""Real ffmpeg/ffprobe round-trips. Skipped when the binaries aren't present."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from pyris.config import Config, VisionConfig
from pyris.ffmpeg import SubprocessFfmpeg
from pyris.types import MediaType, TimeRange

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not installed",
)


def _config() -> Config:
    return Config(vision=VisionConfig(base_url="u", api_key="k", model="m"))


def _make_video(path, *, seconds=3, with_audio=True):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=10",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _make_image(path):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=160x120", "-frames:v", "1", str(path)],
        check=True, capture_output=True,
    )


def test_probe_video_with_audio(tmp_path):
    v = tmp_path / "v.mp4"
    _make_video(v, with_audio=True)
    info = SubprocessFfmpeg(_config()).probe(v)
    assert info.media_type is MediaType.VIDEO
    assert info.has_audio and info.has_video
    assert info.width == 160 and info.height == 120
    assert info.duration and info.duration > 2.5


def test_probe_silent_video(tmp_path):
    v = tmp_path / "s.mp4"
    _make_video(v, with_audio=False)
    info = SubprocessFfmpeg(_config()).probe(v)
    assert info.media_type is MediaType.VIDEO
    assert not info.has_audio


def test_probe_image(tmp_path):
    img = tmp_path / "i.png"
    _make_image(img)
    info = SubprocessFfmpeg(_config()).probe(img)
    assert info.media_type is MediaType.IMAGE
    assert not info.has_audio


def test_extract_frames_fixed_fps(tmp_path):
    v = tmp_path / "v.mp4"
    _make_video(v, seconds=3, with_audio=False)
    ff = SubprocessFfmpeg(_config())
    frames = list(
        ff.extract_frames(v, time_range=None, fps=2.0, scene_threshold=None, max_dim=80)
    )
    assert len(frames) >= 4  # ~2fps over 3s
    assert all(f.image[:2] == b"\xff\xd8" for f in frames)  # JPEG SOI
    assert all(f.width <= 80 for f in frames)
    assert frames == sorted(frames, key=lambda f: f.timestamp)


def test_extract_frames_with_time_range_offsets_timestamps(tmp_path):
    v = tmp_path / "v.mp4"
    _make_video(v, seconds=5, with_audio=False)
    ff = SubprocessFfmpeg(_config())
    frames = list(
        ff.extract_frames(
            v, time_range=TimeRange(2, 4), fps=2.0, scene_threshold=None, max_dim=80
        )
    )
    assert frames
    # timestamps are absolute (offset by the range start)
    assert all(f.timestamp >= 2.0 - 0.1 for f in frames)


def test_extract_audio(tmp_path):
    v = tmp_path / "v.mp4"
    _make_video(v, with_audio=True)
    ff = SubprocessFfmpeg(_config())
    out = ff.extract_audio(v, time_range=None)
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        out.unlink(missing_ok=True)
