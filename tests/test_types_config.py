import pytest

from pyris.config import Config, SamplingConfig, VisionConfig
from pyris.errors import ConfigError
from pyris.types import TimeRange, Transcript, TranscriptSegment
from tests.conftest import make_config


def test_timerange_validation():
    with pytest.raises(ValueError):
        TimeRange(start=-1)
    with pytest.raises(ValueError):
        TimeRange(start=5, end=5)
    with pytest.raises(ValueError):
        TimeRange(start=5, end=2)


def test_timerange_duration():
    assert TimeRange(2, 7).duration == 5
    assert TimeRange(2).duration is None


def test_transcript_full_text():
    t = Transcript(
        segments=[TranscriptSegment(0, 1, " hello "), TranscriptSegment(1, 2, "world")]
    )
    assert t.full_text == "hello world"
    assert Transcript(segments=[]).full_text == ""


def test_config_validate_ok():
    make_config().validate()  # should not raise


def test_config_validate_missing_fields():
    cfg = make_config()
    cfg.vision = VisionConfig(base_url="", api_key="k", model="m")
    with pytest.raises(ConfigError):
        cfg.validate()


def test_config_validate_bad_sampling():
    cfg = make_config(sampling=SamplingConfig(default_fps=0))
    with pytest.raises(ConfigError):
        cfg.validate()


def test_resolved_vision_model_fallback():
    v = VisionConfig(base_url="u", api_key="k", model="driver")
    assert v.resolved_vision_model() == "driver"
    v.vision_model = "vm"
    assert v.resolved_vision_model() == "vm"
