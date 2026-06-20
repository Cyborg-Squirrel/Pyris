import pytest

from pyris.errors import UnsupportedMediaError
from pyris.llm import ImagePart, TextPart
from pyris.pipeline import build_timeline, resolve_mode
from pyris.types import (
    MediaInfo,
    MediaType,
    RequestMode,
    Transcript,
    TranscriptSegment,
)
from tests.conftest import make_frame


def info(media_type, *, has_audio, has_video):
    return MediaInfo(media_type, 10.0, has_audio, has_video, 100, 100, None)


def test_resolve_auto():
    img = info(MediaType.IMAGE, has_audio=False, has_video=True)
    aud = info(MediaType.AUDIO, has_audio=True, has_video=False)
    vid_a = info(MediaType.VIDEO, has_audio=True, has_video=True)
    vid_silent = info(MediaType.VIDEO, has_audio=False, has_video=True)

    assert resolve_mode(RequestMode.AUTO, img) is RequestMode.VISION
    assert resolve_mode(RequestMode.AUTO, aud) is RequestMode.STT
    assert resolve_mode(RequestMode.AUTO, vid_a) is RequestMode.VISION_AND_STT
    assert resolve_mode(RequestMode.AUTO, vid_silent) is RequestMode.VISION


def test_resolve_explicit_conflicts():
    aud = info(MediaType.AUDIO, has_audio=True, has_video=False)
    img = info(MediaType.IMAGE, has_audio=False, has_video=True)
    silent_vid = info(MediaType.VIDEO, has_audio=False, has_video=True)

    with pytest.raises(UnsupportedMediaError):
        resolve_mode(RequestMode.VISION, aud)
    with pytest.raises(UnsupportedMediaError):
        resolve_mode(RequestMode.STT, img)
    with pytest.raises(UnsupportedMediaError):
        resolve_mode(RequestMode.VISION_AND_STT, silent_vid)


def test_build_timeline_interleaves_by_timestamp():
    frames = [make_frame(0.0), make_frame(2.0)]
    transcript = Transcript(
        segments=[TranscriptSegment(0.5, 1.0, "a"), TranscriptSegment(1.5, 2.5, "b")]
    )
    parts = build_timeline(frames, transcript)

    # Expected order: frame@0, seg(0.5), seg(1.5), frame@2
    kinds = [
        ("img", p.frame.timestamp) if isinstance(p, ImagePart) else ("txt", p.text)
        for p in parts
    ]
    assert kinds[0] == ("txt", "[frame @ 0.00s]")
    assert kinds[1] == ("img", 0.0)
    assert kinds[2][0] == "txt" and "a" in kinds[2][1]
    assert kinds[3][0] == "txt" and "b" in kinds[3][1]
    assert kinds[4] == ("txt", "[frame @ 2.00s]")
    assert kinds[5] == ("img", 2.0)


def test_build_timeline_frames_only():
    parts = build_timeline([make_frame(1.0)], None)
    assert any(isinstance(p, ImagePart) for p in parts)
    assert all(not isinstance(p, TextPart) or "frame" in p.text for p in parts)
