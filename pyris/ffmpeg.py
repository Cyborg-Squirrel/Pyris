"""The ffmpeg/ffprobe layer.

This is the *only* place that knows about ffmpeg. It's a Protocol so the
pipeline depends on the interface, not the subprocess — which makes the whole
core mockable in tests without a real ffmpeg install.

Security note: every value that reaches the command line (paths, timestamps,
fps) is passed as a discrete ``subprocess`` arg list — never an interpolated
shell string (``shell=False``) — and numeric args are validated as numbers.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from .config import Config
from .errors import FfmpegError
from .types import Frame, MediaInfo, MediaType, TimeRange

# ffprobe reports container/codec names; these mark "this is a still image, not
# a 1-frame video" so routing can forbid STT on it.
_IMAGE_FORMATS = {
    "image2",
    "png_pipe",
    "jpeg_pipe",
    "mjpeg",
    "webp_pipe",
    "gif",
    "bmp_pipe",
    "tiff_pipe",
}
_IMAGE_CODECS = {"mjpeg", "png", "bmp", "gif", "webp", "tiff", "apng"}

# showinfo prints one line per frame to stderr; we read timestamp + dimensions
# from it and rely on its ordering matching the written frame files.
_PTS_RE = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")
_SIZE_RE = re.compile(r"\bs:(\d+)x(\d+)")


class MediaProbe(Protocol):
    def probe(self, path: Path) -> MediaInfo:
        """Run ffprobe and report type, duration, and which streams exist.

        ``has_audio`` is what lets us decide whether a video even *can* be
        transcribed before we try.
        """
        ...


class FrameExtractor(Protocol):
    def extract_frames(
        self,
        path: Path,
        *,
        time_range: TimeRange | None,
        fps: float,
        scene_threshold: float | None,
        max_dim: int,
    ) -> Iterator[Frame]:
        """Yield encoded frames lazily.

        If ``scene_threshold`` is set, select scene-change frames instead of a
        fixed cadence. Frames are downscaled to ``max_dim`` here so the rest of
        the pipeline never handles full-res images. Yielding (not returning a
        list) keeps memory flat for long videos.
        """
        ...


class AudioExtractor(Protocol):
    def extract_audio(
        self, path: Path, *, time_range: TimeRange | None
    ) -> Path:
        """Extract the audio track to a temp file suitable for Whisper
        (mono, 16 kHz). Returns the temp path; caller owns cleanup.
        """
        ...


class Ffmpeg(MediaProbe, FrameExtractor, AudioExtractor, Protocol):
    """The full surface the core depends on."""


def _to_number(value: float, name: str) -> str:
    """Render a numeric arg, refusing anything non-finite/negative so nothing
    weird can slip into the command line."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FfmpegError(f"{name} must be a number, got {value!r}")
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        raise FfmpegError(f"{name} must be finite and >= 0, got {value!r}")
    return repr(float(value))


class SubprocessFfmpeg:
    """Concrete implementation that shells out to ffmpeg/ffprobe.

    ``ffmpeg`` is required and verified at startup. ``ffprobe`` is preferred for
    probing (clean JSON) but optional: when it's absent we fall back to parsing
    ``ffmpeg -i`` output, so ffmpeg-only environments still work.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._ffmpeg = config.ffmpeg_path
        self._ffprobe = config.ffprobe_path
        self._temp_dir = str(config.temp_dir) if config.temp_dir else None
        self._require_binary(self._ffmpeg)
        self._has_ffprobe = self._binary_available(self._ffprobe)

    @staticmethod
    def _binary_available(binary: str) -> bool:
        try:
            subprocess.run([binary, "-version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _require_binary(self, binary: str) -> None:
        if not self._binary_available(binary):
            raise FfmpegError(
                f"{binary!r} not found on PATH; ffmpeg is required at runtime"
            )

    # -- probe ---------------------------------------------------------------

    def probe(self, path: Path) -> MediaInfo:
        if not self._has_ffprobe:
            return self._probe_via_ffmpeg(path)
        cmd = [
            self._ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise FfmpegError(
                f"ffprobe failed for {path}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise FfmpegError(f"could not parse ffprobe output for {path}") from exc
        return self._info_from_probe(data)

    def _info_from_probe(self, data: dict) -> MediaInfo:
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration = _opt_float(fmt.get("duration"))
        if duration is None and video is not None:
            duration = _opt_float(video.get("duration"))

        width = _opt_int(video.get("width")) if video else None
        height = _opt_int(video.get("height")) if video else None

        format_names = set((fmt.get("format_name") or "").split(","))
        codec = (video or {}).get("codec_name", "")
        nb_frames = _opt_int((video or {}).get("nb_frames"))

        is_image = bool(video) and (
            bool(format_names & _IMAGE_FORMATS)
            or codec in _IMAGE_CODECS
            and (nb_frames in (None, 1))
        )

        if is_image:
            media_type = MediaType.IMAGE
        elif video is not None:
            media_type = MediaType.VIDEO
        elif audio is not None:
            media_type = MediaType.AUDIO
        else:
            raise FfmpegError("media has neither a video nor an audio stream")

        return MediaInfo(
            media_type=media_type,
            duration=None if media_type is MediaType.IMAGE else duration,
            has_audio=audio is not None,
            has_video=video is not None,
            width=width,
            height=height,
            mime=_guess_mime(media_type, codec, format_names),
        )

    def _probe_via_ffmpeg(self, path: Path) -> MediaInfo:
        """Fallback when ffprobe is unavailable: ``ffmpeg -i`` writes the input's
        format/stream summary to stderr and exits non-zero (no output file). We
        parse that text instead of ffprobe's JSON."""
        proc = subprocess.run(
            [self._ffmpeg, "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
        )
        info = _parse_ffmpeg_identify(proc.stderr)
        if info is None:
            raise FfmpegError(
                f"could not identify media from ffmpeg output for {path}: "
                f"{proc.stderr.strip()[:300]}"
            )
        return info

    # -- frames --------------------------------------------------------------

    def extract_frames(
        self,
        path: Path,
        *,
        time_range: TimeRange | None,
        fps: float,
        scene_threshold: float | None,
        max_dim: int,
    ) -> Iterator[Frame]:
        if max_dim < 1:
            raise FfmpegError(f"max_dim must be >= 1, got {max_dim}")

        filters: list[str] = []
        if scene_threshold is not None:
            # comma inside gt() must be escaped so the filtergraph parser does
            # not read it as a filter separator.
            filters.append(f"select=gt(scene\\,{_to_number(scene_threshold, 'scene_threshold')})")
        else:
            filters.append(f"fps={_to_number(fps, 'fps')}")
        if self._config.sampling.dedup_near_identical:
            # dedup happens in ffmpeg (mpdecimate) rather than Python so we never
            # decode frames just to compare them.
            filters.append("mpdecimate")
        filters.append(f"scale=w='min({int(max_dim)}\\,iw)':h=-2")
        filters.append("showinfo")
        vf = ",".join(filters)

        offset = time_range.start if time_range else 0.0
        with tempfile.TemporaryDirectory(dir=self._temp_dir) as td:
            out_pattern = os.path.join(td, "frame_%06d.jpg")
            cmd = [self._ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info"]
            cmd += self._seek_args(time_range)
            cmd += ["-i", str(path)]
            cmd += ["-vf", vf, "-vsync", "vfr", "-q:v", "3", out_pattern]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise FfmpegError(
                    f"frame extraction failed for {path}: {proc.stderr.strip()}"
                )

            timestamps = [float(m) for m in _PTS_RE.findall(proc.stderr)]
            sizes = [(int(w), int(h)) for w, h in _SIZE_RE.findall(proc.stderr)]
            files = sorted(Path(td).glob("frame_*.jpg"))
            for i, fp in enumerate(files):
                ts = offset + (timestamps[i] if i < len(timestamps) else 0.0)
                width, height = sizes[i] if i < len(sizes) else (0, 0)
                yield Frame(
                    timestamp=ts,
                    image=fp.read_bytes(),
                    mime="image/jpeg",
                    width=width,
                    height=height,
                )

    # -- audio ---------------------------------------------------------------

    def extract_audio(self, path: Path, *, time_range: TimeRange | None) -> Path:
        fd, out = tempfile.mkstemp(suffix=".wav", dir=self._temp_dir)
        os.close(fd)
        cmd = [self._ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error"]
        cmd += self._seek_args(time_range)
        cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-y", out]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            _silent_unlink(out)
            raise FfmpegError(
                f"audio extraction failed for {path}: {proc.stderr.strip()}"
            )
        return Path(out)

    def _seek_args(self, time_range: TimeRange | None) -> list[str]:
        """Input-side seeking: ``-ss``/``-t`` before ``-i``. Frame/segment
        timestamps then start near 0, so callers add ``time_range.start`` back to
        recover absolute positions (done in ``extract_frames``)."""
        if time_range is None:
            return []
        args = ["-ss", _to_number(time_range.start, "time_range.start")]
        if time_range.duration is not None:
            args += ["-t", _to_number(time_range.duration, "time_range.duration")]
        return args


def _opt_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def _opt_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _guess_mime(media_type: MediaType, codec: str, format_names: set[str]) -> str | None:
    if media_type is MediaType.IMAGE:
        if codec in ("mjpeg",) or "jpeg_pipe" in format_names:
            return "image/jpeg"
        if codec == "png" or "png_pipe" in format_names:
            return "image/png"
        if codec == "webp" or "webp_pipe" in format_names:
            return "image/webp"
        if codec == "gif" or "gif" in format_names:
            return "image/gif"
        return "image/*"
    if media_type is MediaType.AUDIO:
        return "audio/*"
    return "video/*"


_INPUT_RE = re.compile(r"^Input #\d+,\s*(.+?),\s*from ", re.MULTILINE)
_DURATION_RE = re.compile(r"Duration:\s*(N/A|\d+:\d+:\d+\.\d+)")
_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?:\s*(Video|Audio):\s*([A-Za-z0-9_]+)")
_DIM_RE = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")


def _parse_ffmpeg_identify(stderr: str) -> MediaInfo | None:
    """Parse the format/stream summary ``ffmpeg -i`` prints to stderr. Pure
    function (no subprocess) so it's unit-testable without the binary."""
    input_match = _INPUT_RE.search(stderr)
    if input_match is None:
        return None
    format_names = {f.strip() for f in input_match.group(1).split(",")}

    duration: float | None = None
    dur_match = _DURATION_RE.search(stderr)
    if dur_match and dur_match.group(1) != "N/A":
        h, m, s = dur_match.group(1).split(":")
        duration = int(h) * 3600 + int(m) * 60 + float(s)

    has_video = has_audio = False
    codec = ""
    width = height = None
    for line in stderr.splitlines():
        sm = _STREAM_RE.search(line)
        if sm is None:
            continue
        kind, stream_codec = sm.group(1), sm.group(2)
        if kind == "Video" and not has_video:
            has_video = True
            codec = stream_codec
            dim = _DIM_RE.search(line)
            if dim:
                width, height = int(dim.group(1)), int(dim.group(2))
        elif kind == "Audio":
            has_audio = True

    is_image = has_video and (
        bool(format_names & _IMAGE_FORMATS)
        or (codec in _IMAGE_CODECS and (duration is None or duration < 1.0))
    )
    if is_image:
        media_type = MediaType.IMAGE
    elif has_video:
        media_type = MediaType.VIDEO
    elif has_audio:
        media_type = MediaType.AUDIO
    else:
        return None

    return MediaInfo(
        media_type=media_type,
        duration=None if media_type is MediaType.IMAGE else duration,
        has_audio=has_audio,
        has_video=has_video,
        width=width,
        height=height,
        mime=_guess_mime(media_type, codec, format_names),
    )


def _silent_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
