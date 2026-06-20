"""Unit tests for the ffmpeg-only probe fallback parser (no binary needed)."""
from pyris.ffmpeg import _parse_ffmpeg_identify
from pyris.types import MediaType

VIDEO_WITH_AUDIO = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:00:12.50, start: 0.000000, bitrate: 1300 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv), 1920x1080 [SAR 1:1 DAR 16:9], 1200 kb/s, 30 fps
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 128 kb/s
"""

SILENT_VIDEO = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 's.mp4':
  Duration: 00:00:05.00, start: 0.000000, bitrate: 800 kb/s
  Stream #0:0[0x1]: Video: h264 (High), yuv420p, 640x480, 30 fps
"""

JPEG = """\
Input #0, image2, from 'pic.jpg':
  Duration: 00:00:00.04, start: 0.000000, bitrate: 1000 kb/s
  Stream #0:0: Video: mjpeg (Baseline), yuvj420p(pc), 160x120, 25 fps
"""

PNG = """\
Input #0, png_pipe, from 'i.png':
  Duration: N/A, bitrate: N/A
  Stream #0:0: Video: png, rgb24(pc), 320x240, 25 fps
"""

MP3 = """\
Input #0, mp3, from 'podcast.mp3':
  Duration: 00:01:03.20, start: 0.000000, bitrate: 128 kb/s
  Stream #0:0: Audio: mp3 (mp3float), 44100 Hz, stereo, fltp, 128 kb/s
"""


def test_parse_video_with_audio():
    info = _parse_ffmpeg_identify(VIDEO_WITH_AUDIO)
    assert info.media_type is MediaType.VIDEO
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (1920, 1080)
    assert abs(info.duration - 12.5) < 1e-6


def test_parse_silent_video():
    info = _parse_ffmpeg_identify(SILENT_VIDEO)
    assert info.media_type is MediaType.VIDEO
    assert info.has_video and not info.has_audio
    assert (info.width, info.height) == (640, 480)


def test_parse_jpeg_is_image():
    info = _parse_ffmpeg_identify(JPEG)
    assert info.media_type is MediaType.IMAGE
    assert not info.has_audio
    assert info.duration is None
    assert info.mime == "image/jpeg"


def test_parse_png_is_image():
    info = _parse_ffmpeg_identify(PNG)
    assert info.media_type is MediaType.IMAGE
    assert info.mime == "image/png"


def test_parse_audio_only():
    info = _parse_ffmpeg_identify(MP3)
    assert info.media_type is MediaType.AUDIO
    assert info.has_audio and not info.has_video


def test_parse_garbage_returns_none():
    assert _parse_ffmpeg_identify("not ffmpeg output") is None
