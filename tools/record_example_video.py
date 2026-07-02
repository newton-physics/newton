# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a Newton example to an MP4, headless.

Drives a registered example with an offscreen OpenGL viewer, grabs each
rendered frame with :meth:`ViewerGL.get_frame`, and pipes raw RGB straight
into the system ``ffmpeg`` binary. No extra Python dependencies — just
``ffmpeg`` on ``PATH``.

    uv run python tools/record_example_video.py cloth_twist
    uv run python tools/record_example_video.py ik_franka -o franka.mp4 --num-frames 200
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", help="Registered example name, e.g. 'cloth_twist'.")
    parser.add_argument("-o", "--output", help="Output .mp4 path (default: <example>.mp4).")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH (e.g. `apt install ffmpeg`).")

    import newton.examples as nex  # noqa: PLC0415
    from newton.viewer import ViewerGL  # noqa: PLC0415

    examples = nex.get_examples()
    if args.example not in examples:
        sys.exit(f"Unknown example {args.example!r}; see `python -m newton.examples --list`.")
    example_cls = importlib.import_module(examples[args.example]).Example

    # Reuse the example's own parser for its defaults, then point it at our
    # offscreen viewer. Keep headless False: some examples swap in a non-rendering
    # ViewerNull when args.headless is True, which would discard our GL viewer.
    ex_parser = example_cls.create_parser() if hasattr(example_cls, "create_parser") else nex.create_parser()
    ex_args = ex_parser.parse_args([])
    ex_args.viewer, ex_args.headless, ex_args.num_frames = "gl", False, args.num_frames

    viewer = ViewerGL(width=args.width, height=args.height, headless=True)
    example = example_cls(viewer, ex_args)

    output = args.output or f"{args.example}.mp4"
    ffmpeg = None
    first_frame = blank = None
    try:
        for _ in range(args.num_frames):
            if viewer.should_step():
                example.step()
            example.render()
            frame = viewer.get_frame().numpy()  # (h, w, 3) uint8, top-left origin

            if ffmpeg is None:
                h, w = frame.shape[:2]
                ffmpeg = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
                     "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output],
                    stdin=subprocess.PIPE,
                )  # fmt: skip
                first_frame, blank = frame, True
            elif blank and (frame != first_frame).any():
                blank = False
            ffmpeg.stdin.write(frame.tobytes())
    finally:
        if ffmpeg is not None:
            ffmpeg.stdin.close()
            ffmpeg.wait()
        viewer.close()

    if blank:
        print("warning: every frame is identical — the scene may be empty or out of frame")
    print(f"Wrote {output} ({args.num_frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
