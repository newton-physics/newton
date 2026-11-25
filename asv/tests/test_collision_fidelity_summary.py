# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check denominators and coverage in multi-case result summaries."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from collision_fidelity.runner import summarize_results


class TestFidelitySummary(unittest.TestCase):
    def test_missing_outcomes_do_not_become_passes(self):
        """Separate evaluated criteria, missing evidence and unsupported scenes."""
        results = [
            {"case_id": "a", "execution": {"status": "completed"}, "criteria": {"overall": {"status": "pass"}}},
            {"case_id": "b", "execution": {"status": "completed"}, "criteria": {"overall": {"status": "fail"}}},
            {"case_id": "c", "execution": {"status": "completed"}, "criteria": {"overall": {"status": "incomplete"}}},
            {"case_id": "d", "execution": {"status": "unsupported"}, "criteria": {}},
        ]
        summary = summarize_results(results)
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["criteria"]["overall"]["failure_rate"], 0.5)
        self.assertEqual(summary["criteria"]["overall"]["evaluated"], 2)
        self.assertEqual(summary["criteria"]["overall"]["applicable"], 3)
        self.assertEqual(summary["criteria"]["overall"]["unsupported"], 1)
        self.assertEqual(summary["criteria"]["overall"]["unevaluated_fraction"], 1 / 3)
        self.assertIsNone(summary["means"]["jitter_mm"]["value"])

    def test_unknown_outcomes_do_not_become_failures(self):
        """Keep an empty failure-rate denominator unavailable, with missing coverage visible."""
        summary = summarize_results(
            [
                {
                    "case_id": "a",
                    "execution": {"status": "completed"},
                    "criteria": {"overall": {"status": "incomplete"}},
                },
                {
                    "case_id": "b",
                    "execution": {"status": "completed"},
                    "criteria": {"overall": {"status": "not_applicable"}},
                },
            ]
        )
        overall = summary["criteria"]["overall"]
        self.assertIsNone(overall["failure_rate"])
        self.assertEqual(overall["unevaluated_fraction"], 1)
        self.assertEqual(overall["fail"], 0)

    def test_partial_hold_is_excluded_from_full_hold_mean(self):
        """Average per-case maxima only when the required windows are complete."""
        results = []
        for case_id, value, complete in (("a", 0.002, 2), ("b", 0.004, 2), ("partial", 1.0, 1)):
            results.append(
                {
                    "case_id": case_id,
                    "execution": {"status": "completed"},
                    "measurements": {
                        "stability": {
                            "jitter_m": value,
                            "rms_speed_m_s": value * 10,
                            "complete_window_count": complete,
                            "declared_complete_window_count": 2,
                        }
                    },
                }
            )
        summary = summarize_results(results)
        self.assertAlmostEqual(summary["means"]["jitter_mm"]["value"], 3.0)
        self.assertEqual(summary["means"]["jitter_mm"]["cases"], ["a", "b"])
        self.assertAlmostEqual(summary["means"]["rms_speed_m_s"]["value"], 0.03)

    def test_nonfinite_trajectory_is_excluded_from_penetration_mean(self):
        """Reject a completion flag that conflicts with nonfinite state evidence."""
        summary = summarize_results(
            [
                {
                    "case_id": "invalid",
                    "execution": {"status": "completed", "full_timeline_completed": True},
                    "measurements": {
                        "finite": {"all_samples_finite": False},
                        "penetration": {"rigid_soft_penetration_m": 0.001},
                    },
                },
            ]
        )
        self.assertEqual(summary["means"]["rigid_soft_penetration_mm"]["count"], 0)
        self.assertIsNone(summary["means"]["rigid_soft_penetration_mm"]["value"])


if __name__ == "__main__":
    unittest.main()
