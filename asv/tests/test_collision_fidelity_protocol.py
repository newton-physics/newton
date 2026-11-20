# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check physical time and measurement intervals independently of the solver."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from collision_fidelity.protocol import Timeline


class TestFidelityTimeline(unittest.TestCase):
    def test_hold_extension_preserves_command_speed(self):
        """Extend only the stationary plateau and retain physical step spacing."""
        case = {
            "frame_dt_s": 0.1,
            "substeps": 10,
            "duration_s": 10.0,
            "hold_start": 4.0,
            "hold_end": 6.0,
            "stationary_windows": [[4.0, 6.0]],
            "transport_windows": [[2.0, 4.0]],
            "place_release_windows": [[6.0, 10.0]],
            "applicability": {},
        }
        schedule = Timeline(case, hold_seconds=60.0)
        self.assertEqual(schedule.duration, 68.0)
        self.assertEqual(schedule.steps, 6800)
        self.assertEqual(schedule.command_time(4.9), 4.9)
        self.assertEqual(schedule.command_time(5.0), 5.0)
        self.assertEqual(schedule.command_time(62.9), 5.0)
        self.assertAlmostEqual(schedule.command_time(63.01), 5.01)
        self.assertEqual(schedule.command_time(68.0), 10.0)
        self.assertEqual(schedule.protocol["stationary_windows"], [[4.0, 64.0]])
        self.assertEqual(schedule.protocol["place_release_intervals"], [{"start_s": 64.0, "end_s": 68.0}])
        self.assertEqual(case["stationary_windows"], [[4.0, 6.0]])

    def test_substep_alignment_and_passive_schedule(self):
        """Exclude partial boundary steps without extending passive scenes."""
        case = {
            "frame_dt_s": 0.1,
            "substeps": 2,
            "duration_s": 2.0,
            "hold_start": None,
            "hold_end": None,
            "stationary_windows": [[0.123, 1.987]],
            "applicability": {},
        }
        schedule = Timeline(case)
        self.assertEqual(schedule.duration, 2.0)
        self.assertEqual(schedule.steps, 40)
        self.assertAlmostEqual(schedule.protocol["stationary_windows"][0][0], 0.15)
        self.assertAlmostEqual(schedule.protocol["stationary_windows"][0][1], 1.95)
        self.assertEqual(schedule.command_time(1.5), 1.5)

    def test_invalid_timing_is_rejected(self):
        """Reject timing that would change or reverse the authored operation."""
        case = {
            "frame_dt_s": 0.1,
            "substeps": 2,
            "duration_s": 3.0,
            "hold_start": 1.0,
            "hold_end": 2.0,
            "stationary_windows": [[1.0, 2.0]],
            "applicability": {},
        }
        for seconds in (0.0, 0.5, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                Timeline(case, hold_seconds=seconds)
        with self.assertRaises(ValueError):
            Timeline({**case, "substeps": 11})


if __name__ == "__main__":
    unittest.main()
