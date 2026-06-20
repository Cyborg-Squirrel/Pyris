"""Analyze a media file (frames + transcript) and inspect the full result.

    # whole file, auto-routed, scene-detection sampling
    python examples/analyze_video.py clip.mp4 "What is being demonstrated?"

    # second half at 1 fps
    python examples/analyze_video.py clip.mp4 "Describe it" --start 15.53 --fps 1

    # a 30s–90s window, forced mode
    python examples/analyze_video.py clip.mp4 "Summarize" --start 30 --end 90 --mode vision
"""
from __future__ import annotations

import argparse

from _shared import config_from_env

from pyris import FileProvider, Pyris, RequestMode, SubprocessFfmpeg, TimeRange


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a media file with Pyris.")
    p.add_argument("path", nargs="?", default="clip.mp4", help="media file path")
    p.add_argument(
        "prompt",
        nargs="?",
        default="Describe what happens in this video.",
        help="question to ask about the media",
    )
    p.add_argument("--start", type=float, help="clip start in seconds")
    p.add_argument("--end", type=float, help="clip end in seconds")
    p.add_argument(
        "--fps",
        type=float,
        help="sample frames at this rate (implies fixed cadence, not scene detection)",
    )
    p.add_argument(
        "--mode",
        choices=[m.value for m in RequestMode],
        help="force a mode; default auto-routes (vision-only if no STT configured)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        help="max images per LLM call for map-reduce (default: 20)",
    )
    p.add_argument(
        "--timeout", type=float, default=600.0, help="per-request timeout (seconds)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = config_from_env()
    config.request_timeout = args.timeout
    if args.batch_size is not None:
        config.vision.max_images_per_request = args.batch_size

    if args.fps is not None:
        # fps only takes effect with fixed-cadence sampling; scene detection
        # picks its own frames and ignores the rate.
        config.sampling.scene_detection = False
        config.sampling.max_frames = 10_000  # fps × duration is the only cap
    else:
        config.sampling.max_frames = 8  # sensible default for scene-detection mode

    pyris = Pyris.from_config(config)
    provider = FileProvider(args.path, probe=SubprocessFfmpeg(config))

    # --mode wins; otherwise auto-route, but fall back to vision-only when there
    # is no STT endpoint (AUTO would pick VISION_AND_STT for videos with audio).
    if args.mode is not None:
        mode = RequestMode(args.mode)
    else:
        mode = RequestMode.AUTO if config.stt is not None else RequestMode.VISION

    time_range = None
    if args.start is not None or args.end is not None:
        time_range = TimeRange(start=args.start or 0.0, end=args.end)

    result = pyris.analyze(
        provider,
        args.prompt,
        mode=mode,
        time_range=time_range,
        fps=args.fps,
    )

    print(f"=== mode: {result.mode.value} | model: {result.model} ===\n")
    print(result.text)

    if result.transcript is not None:
        print(f"\n--- transcript ({result.transcript.language}) ---")
        for seg in result.transcript.segments:
            print(f"[{seg.start:6.2f}-{seg.end:6.2f}] {seg.text.strip()}")

    u = result.usage
    print(
        f"\nframes sent: {u.frames_sent} | frames used: {result.frames_used} | "
        f"audio: {u.audio_seconds:.1f}s | "
        f"tokens: {u.input_tokens} in / {u.output_tokens} out"
    )


if __name__ == "__main__":
    main()
