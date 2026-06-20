from pyris.config import SamplingConfig
from pyris.sampling import FrameSampler, _thin_uniform
from pyris.types import MediaInfo, MediaType
from tests.conftest import FakeFfmpeg, make_frame


def test_thin_uniform_keeps_endpoints():
    frames = [make_frame(float(i)) for i in range(10)]
    kept = list(_thin_uniform(frames, 4))
    assert len(kept) == 4
    assert kept[0].timestamp == 0.0
    assert kept[-1].timestamp == 9.0


def test_thin_uniform_under_budget_passthrough():
    frames = [make_frame(float(i)) for i in range(3)]
    assert list(_thin_uniform(frames, 10)) == frames


def test_thin_uniform_single():
    frames = [make_frame(float(i)) for i in range(9)]
    kept = list(_thin_uniform(frames, 1))
    assert len(kept) == 1


def test_sampler_lowers_fps_for_long_media():
    # 100s of media, default 1fps would be ~100 frames; max_frames=10 -> fps<=0.1
    ff = FakeFfmpeg(frame_count=5)
    sampler = FrameSampler(ff, SamplingConfig(default_fps=1.0, max_frames=10, scene_detection=False))
    info = MediaInfo(MediaType.VIDEO, 100.0, False, True, 10, 10, None)
    list(sampler.sample(None, info, time_range=None, fps=None, max_dim=512))
    assert ff.extract_calls[0]["fps"] <= 0.1 + 1e-9
    assert ff.extract_calls[0]["max_dim"] == 512


def test_sampler_scene_detection_passes_threshold():
    ff = FakeFfmpeg(frame_count=2)
    sampler = FrameSampler(ff, SamplingConfig(scene_detection=True, scene_threshold=0.4))
    info = MediaInfo(MediaType.VIDEO, 10.0, False, True, 10, 10, None)
    list(sampler.sample(None, info, time_range=None, fps=None, max_dim=256))
    assert ff.extract_calls[0]["scene_threshold"] == 0.4


def test_sampler_enforces_max_frames():
    ff = FakeFfmpeg(frame_count=50)
    sampler = FrameSampler(ff, SamplingConfig(max_frames=8, scene_detection=True))
    info = MediaInfo(MediaType.VIDEO, 10.0, False, True, 10, 10, None)
    out = list(sampler.sample(None, info, time_range=None, fps=None, max_dim=256))
    assert len(out) == 8
