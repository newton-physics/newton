# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Map physical time to an authored command schedule without slowing motion."""

import copy
import math


class Timeline:
    """Extend a stationary hold while keeping every physics step unchanged."""

    def __init__(self, case: dict, *, hold_seconds: float = 60.0):
        self.protocol = copy.deepcopy(case)
        self.frame_dt = float(case["frame_dt_s"])
        self.substeps = int(case["substeps"])
        if not 1 <= self.substeps <= 10 or self.substeps != case["substeps"]:
            raise ValueError("Substeps must be an integer between one and ten")
        if not math.isfinite(self.frame_dt) or self.frame_dt <= 0:
            raise ValueError("Frame interval must be finite and positive")
        if not math.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("Hold duration must be finite and positive")
        self.dt = self.frame_dt / self.substeps
        original_duration = float(case["duration_s"])
        if not math.isfinite(original_duration) or original_duration <= 0:
            raise ValueError("Scene duration must be finite and positive")
        start, end = case.get("hold_start"), case.get("hold_end")
        self.pause = None
        self.extra_hold = 0.0
        if (start is None) != (end is None):
            raise ValueError("Both hold boundaries must be provided")
        if start is not None:
            if not 0 <= start < end <= original_duration or end - start <= 2 * self.frame_dt:
                raise ValueError("Hold must contain an interior stationary frame")
            if hold_seconds < end - start:
                raise ValueError("The full profile cannot shorten the authored hold")
            self.pause = 0.5 * (start + end)
            self.extra_hold = hold_seconds - (end - start)
        self.duration = original_duration + self.extra_hold
        self.steps = math.ceil(self.duration / self.dt - 1e-9)
        intervals = []
        for a, b in case.get("stationary_windows", []):
            if not 0 <= a < b <= original_duration:
                raise ValueError("Stationary interval is outside the authored timeline")
            physical_start, physical_end = self.physical_time(a), self.physical_time(b)
            intervals.append(
                {
                    "start_s": math.ceil(physical_start / self.dt - 1e-9) * self.dt,
                    "end_s": math.floor(physical_end / self.dt + 1e-9) * self.dt,
                    "requested_start_s": physical_start,
                    "requested_end_s": physical_end,
                    "phase": "hold" if start is not None else "steady observation",
                }
            )
        self.protocol.update(
            duration_s=self.duration,
            dt_s=self.dt,
            window_s=0.5,
            hold_start=start,
            hold_end=None if end is None else self.physical_time(end),
            extra_hold_s=self.extra_hold,
            dynamic_window=[0.0, self.duration],
            stationary_intervals=intervals,
            stationary_windows=[[i["start_s"], i["end_s"]] for i in intervals],
        )
        for source, destination in (
            ("transport_windows", "transport_intervals"),
            ("place_release_windows", "place_release_intervals"),
        ):
            self.protocol[destination] = [
                {"start_s": self.physical_time(a), "end_s": self.physical_time(b)} for a, b in case.get(source, [])
            ]
        if case.get("applicability", {}).get("analytic_freefall"):
            self.protocol["gravity_only_interval"] = {
                "start_s": 0.0,
                "end_s": self.duration,
                "gravity_m_s2": [0.0, 0.0, -9.81],
            }

    def command_time(self, physical_time: float) -> float:
        """Return the command time at the start of a physical substep."""
        if self.pause is None or physical_time <= self.pause:
            return float(physical_time)
        return float(max(self.pause, physical_time - self.extra_hold))

    def physical_time(self, command_time: float) -> float:
        """Map an authored stage boundary to elapsed simulation time."""
        extra = self.extra_hold if self.pause is not None and command_time > self.pause else 0.0
        return float(command_time + extra)
