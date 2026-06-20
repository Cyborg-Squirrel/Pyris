import asyncio

import pytest

from pyris.errors import ConfigError
from pyris.llm import ImagePart
from pyris.pipeline import Pyris
from pyris.types import MediaInfo, MediaType, RequestMode, TimeRange
from tests.conftest import (
    FakeFfmpeg,
    FakeProvider,
    FakeStt,
    FakeVision,
    make_config,
)


def build(info, *, frames=3, stt=True, **cfg_over):
    ff = FakeFfmpeg(frame_count=frames, info=info)
    vision = FakeVision()
    stt_client = FakeStt() if stt else None
    pyris = Pyris(make_config(**cfg_over), ffmpeg=ff, vision=vision, stt=stt_client)
    return pyris, ff, vision, stt_client


VIDEO = MediaInfo(MediaType.VIDEO, 10.0, True, True, 100, 100, "video/*")
SILENT_VIDEO = MediaInfo(MediaType.VIDEO, 10.0, False, True, 100, 100, "video/*")
AUDIO = MediaInfo(MediaType.AUDIO, 10.0, True, False, None, None, "audio/*")
IMAGE = MediaInfo(MediaType.IMAGE, None, False, True, 100, 100, "image/jpeg")


def test_vision_and_stt_flow():
    pyris, ff, vision, stt = build(VIDEO, frames=3)
    res = pyris.analyze(FakeProvider(VIDEO), "what happens?")
    assert res.mode is RequestMode.VISION_AND_STT
    assert res.frames_used == 3
    assert res.transcript is not None and res.transcript.language == "en"
    # one vision call carrying 3 images + interleaved transcript text
    assert len(vision.calls) == 1
    assert sum(isinstance(p, ImagePart) for p in vision.calls[0]) == 3
    assert res.usage.frames_sent == 3
    assert res.usage.audio_seconds == 3.0


def test_vision_only_silent_video():
    pyris, ff, vision, stt = build(SILENT_VIDEO, frames=2)
    res = pyris.analyze(FakeProvider(SILENT_VIDEO), "describe")
    assert res.mode is RequestMode.VISION
    assert res.transcript is None
    assert stt.calls == 0


def test_stt_only_transcription_no_prompt():
    pyris, ff, vision, stt = build(AUDIO)
    res = pyris.analyze(FakeProvider(AUDIO), "")
    assert res.mode is RequestMode.STT
    assert res.text == "hello world"  # transcript.full_text, no vision call
    assert vision.calls == []


def test_stt_only_with_prompt_uses_driver():
    pyris, ff, vision, stt = build(AUDIO)
    res = pyris.analyze(FakeProvider(AUDIO), "summarize")
    assert res.mode is RequestMode.STT
    assert vision.calls and sum(isinstance(p, ImagePart) for p in vision.calls[0]) == 0
    assert res.transcript is not None


def test_image_loads_single_frame(tmp_path):
    img_path = tmp_path / "pic.jpg"
    img_path.write_bytes(b"\xff\xd8realjpeg\xff\xd9")
    pyris, ff, vision, stt = build(IMAGE)
    res = pyris.analyze(FakeProvider(IMAGE, img_path), "what is this?")
    assert res.mode is RequestMode.VISION
    assert res.frames_used == 1
    # the single image bytes are read straight off disk, not via ffmpeg
    img_parts = [p for p in vision.calls[0] if isinstance(p, ImagePart)]
    assert img_parts[0].frame.image == b"\xff\xd8realjpeg\xff\xd9"


def test_stt_without_client_raises():
    pyris, ff, vision, _ = build(AUDIO, stt=False)
    with pytest.raises(ConfigError):
        pyris.analyze(FakeProvider(AUDIO), "x")


def test_batching_triggers_map_reduce():
    # 10 frames, limit 4 -> ceil(10/4)=3 map calls + 1 reduce = 4 vision calls
    pyris, ff, vision, stt = build(SILENT_VIDEO, frames=10)
    pyris._config.vision.max_images_per_request = 4
    pyris._config.sampling.max_frames = 100  # don't thin below 10
    res = pyris.analyze(FakeProvider(SILENT_VIDEO), "describe")
    assert len(vision.calls) == 4
    # reduce call is text-only
    assert sum(isinstance(p, ImagePart) for p in vision.calls[-1]) == 0
    # usage aggregates all 4 calls; frames_sent reflects the 10 images
    assert res.usage.input_tokens == 40
    assert res.usage.frames_sent == 10


def test_range_fetch_skips_core_crop():
    # provider already returns the range, so core passes time_range=None to ffmpeg
    info = SILENT_VIDEO
    ff = FakeFfmpeg(frame_count=2, info=info)
    pyris = Pyris(make_config(), ffmpeg=ff, vision=FakeVision(), stt=FakeStt())
    provider = FakeProvider(info, range_fetch=True)
    pyris.analyze(provider, "x", time_range=TimeRange(1, 5))
    assert provider.fetched_range == TimeRange(1, 5)


def test_async_matches_sync():
    pyris, ff, vision, stt = build(VIDEO, frames=2)
    res = asyncio.run(pyris.analyze_async(FakeProvider(VIDEO), "what?"))
    assert res.mode is RequestMode.VISION_AND_STT
    assert res.frames_used == 2
    assert res.transcript is not None


def test_async_batching_concurrent_map():
    pyris, ff, vision, stt = build(SILENT_VIDEO, frames=10)
    pyris._config.vision.max_images_per_request = 4
    pyris._config.sampling.max_frames = 100
    res = asyncio.run(pyris.analyze_async(FakeProvider(SILENT_VIDEO), "describe"))
    assert len(vision.calls) == 4
    assert res.usage.frames_sent == 10
