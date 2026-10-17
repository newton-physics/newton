# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture the desktop or send a smooth, window-relative drag."""

import argparse
import subprocess
import time
from pathlib import Path

import mss
import mss.tools

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="action", required=True)
s = sub.add_parser("shot")
s.add_argument("path")
d = sub.add_parser("drag")
d.add_argument("window")
d.add_argument("button", type=int)
d.add_argument("coords", type=int, nargs=4)
d.add_argument("--modifier", choices=["Shift_L", "Control_L"])
a = p.parse_args()


def xdo(*args):
    subprocess.run(["xdotool", *map(str, args)], check=True)


if a.action == "shot":
    Path(a.path).parent.mkdir(parents=True, exist_ok=True)
    with mss.MSS(backend="xlib") as screen:
        shot = screen.grab(screen.monitors[0])
        mss.tools.to_png(shot.rgb, shot.size, output=a.path)
else:
    x1, y1, x2, y2 = a.coords
    xdo("windowactivate", "--sync", a.window)
    xdo("mousemove", "--window", a.window, x1, y1)
    try:
        if a.modifier:
            xdo("keydown", a.modifier)
        xdo("mousedown", a.button)
        for i in range(1, 21):
            xdo("mousemove", "--window", a.window, round(x1 + (x2 - x1) * i / 20), round(y1 + (y2 - y1) * i / 20))
            time.sleep(0.025)
    finally:
        try:
            xdo("mouseup", a.button)
        finally:
            if a.modifier:
                xdo("keyup", a.modifier)
