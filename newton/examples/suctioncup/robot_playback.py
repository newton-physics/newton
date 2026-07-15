# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Playback of a recorded FANUC palletizer cycle (``robot_recording.jsonl``).

Simulator-agnostic: this module only loads the recording and steps through it. It has no Newton (or
any other framework) dependency -- callers read each :class:`Frame` and apply it to their own robot
(joint targets, gripper) via callbacks. Used by ``example_suction_cup_isaac_sim_repro``.

Recording format -- JSON Lines, one object per frame (~24 Hz)::

    {"ts": "2026-06-05T16:40:10.191", "joints": [<9 floats, deg>],
     "ro": [<4 bools>], "do": {"301": 0, ...}}

``joints`` indices 0-5 are the arm joints J1-J6 (degrees); 6-8 are always 0 (finger DOFs, excluded
from arm playback). J3 is stored relative to J2 -- the real J3 command is ``joints[2] + joints[1]``,
a coupling the caller applies at drive time. ``ro[0]`` is the gripper signal (rising edge = close,
falling edge = open); ``do`` is observational digital I/O.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Frame:
    """One recorded frame, as stored (no coupling or unit conversion applied)."""

    joints_deg: list[float]  # 9 raw joint angles [deg]; indices 0-5 are J1-J6, 6-8 are unused fingers
    ro: list[bool]  # 4 robot-output booleans; ro[0] is the gripper signal
    do: dict[int, int]  # 40 digital outputs keyed by integer signal id (301-340), values 0/1
    ts: str | None = None  # wall-clock timestamp string, for time-accurate replay


def load_recording(path: str | Path) -> list[Frame]:
    """Read a ``.jsonl`` recording into a flat list of :class:`Frame` (blank lines skipped).

    Raises:
        FileNotFoundError: if ``path`` does not exist (with a clear message rather than a deep parse
            error).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"robot recording not found: {path}")
    frames: list[Frame] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        frames.append(
            Frame(
                joints_deg=[float(x) for x in obj["joints"]],
                ro=[bool(x) for x in obj["ro"]],
                do={int(k): int(v) for k, v in obj["do"].items()},
                ts=obj.get("ts"),
            )
        )
    return frames


def recorded_times(frames: list[Frame]) -> list[float]:
    """Seconds of each frame relative to the first, parsed from the ``ts`` timestamps.

    Used for time-accurate playback -- interpolating targets between frames by wall-clock time rather
    than stepping one frame per simulation step. Requires every frame to carry a ``ts``.
    """
    if not frames:
        return []
    t0 = datetime.fromisoformat(frames[0].ts)
    times = [0.0] * len(frames)
    for i in range(len(frames)):
        times[i] = (datetime.fromisoformat(frames[i].ts) - t0).total_seconds()
    return times


class GripperEdgeDetector:
    """Fires ``on_close`` / ``on_open`` on rising / falling edges of the gripper signal.

    Feed the current ``ro[0]`` each step via :meth:`update`. A rising edge (``False -> True``) closes
    the gripper, a falling edge (``True -> False``) opens it. Each callback can be delayed by a
    configurable number of frames (``delay``) after its edge -- e.g. to model a valve/actuation lag.
    """

    def __init__(
        self,
        on_close: Callable[[], None] | None = None,
        on_open: Callable[[], None] | None = None,
        delay: int = 0,
    ):
        self.on_close = on_close
        self.on_open = on_open
        self.delay = delay
        self._prev: bool | None = None
        # queue of (callback, frames_left) so each edge fires after its own delay, independently
        self._pending: list[tuple[Callable[[], None] | None, int]] = []

    def reset(self) -> None:
        """Forget the previous signal and any pending delayed callbacks."""
        self._prev = None
        self._pending = []

    def update(self, signal: bool) -> None:
        """Advance one frame: fire any pending callbacks that come due, then detect a new edge."""
        # count down scheduled (delayed) callbacks and fire those that reach zero this frame
        if self._pending:
            still_pending = []
            for callback, frames_left in self._pending:
                frames_left -= 1
                if frames_left <= 0:
                    if callback is not None:
                        callback()
                else:
                    still_pending.append((callback, frames_left))
            self._pending = still_pending

        # detect an edge against the previous frame's signal
        if self._prev is not None and signal != self._prev:
            callback = self.on_close if signal else self.on_open  # rising = close, falling = open
            if self.delay <= 0:
                if callback is not None:
                    callback()
            else:
                self._pending.append((callback, self.delay))

        self._prev = signal
