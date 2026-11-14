# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only mathematical checks for the experimental comparison observer."""

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from collision_fidelity.metrics import FidelityObserver


def scene(particle_count=2):
    return SimpleNamespace(
        model=SimpleNamespace(particle_mass=np.ones(particle_count), particle_radius=np.zeros(particle_count)),
        state=state(np.zeros((particle_count, 3)), np.zeros((particle_count, 3))),
        time=0.0,
    )


def state(q, qd):
    return SimpleNamespace(particle_q=np.asarray(q, dtype=float), particle_qd=np.asarray(qd, dtype=float))


class TestFidelityMetrics(unittest.TestCase):
    def test_linear_drift(self):
        """Remove linear position drift without removing relative velocity."""
        observer = FidelityObserver(scene().model, {"stationary_intervals": [{"start_s": 0, "end_s": 1}]})
        for i in range(1, 101):
            t = i / 100
            observer.observe(state([[1000 + 2 * t, 0, 0], [0, 0, 0]], [[2, 0, 0], [0, 0, 0]]), t, 0.01, phase="hold")
        result = observer.result()["stability"]
        self.assertEqual(result["complete_window_count"], 2)
        self.assertLess(result["jitter_m"], 1e-07)
        self.assertAlmostEqual(result["rms_speed_m_s"], 2)
        self.assertEqual(result["worst_velocity_sample"]["particle_index"], 0)

    def test_sinusoid_and_worst_local_point(self):
        """Retain local oscillation instead of averaging it over quiet vertices."""
        observer = FidelityObserver(scene(100).model, {"stationary_phases": ["hold"]})
        for i in range(1, 1001):
            t = i / 2000
            angle = 4 * math.pi * (t - 0.00025)
            (q, v) = (np.zeros((100, 3)), np.zeros((100, 3)))
            (q[99, 0], v[99, 0]) = (0.01 * math.cos(angle), -0.04 * math.pi * math.sin(angle))
            observer.observe(state(q, v), t, 0.0005, phase="hold")
        result = observer.result()["stability"]
        self.assertAlmostEqual(result["jitter_m"], 0.01 / math.sqrt(2), places=10)
        self.assertAlmostEqual(result["rms_speed_m_s"], 0.04 * math.pi / math.sqrt(2), places=10)
        self.assertEqual(result["worst_position_sample"]["particle_index"], 99)

    def test_prescribed_rotating_translating_frame(self):
        """Remove only motion from the declared frame, including angular velocity."""

        def frame(t):
            return {
                "position_m": [t, 0, 0],
                "rotation_xyzw": [0, 0, math.sin(t / 2), math.cos(t / 2)],
                "linear_velocity_m_s": [1, 0, 0],
                "angular_velocity_rad_s": [0, 0, 1],
            }

        observer = FidelityObserver(scene(1).model, {"stationary_phases": ["hold"], "frame_at_time": frame})
        for i in range(1, 51):
            t = i / 100
            observer.observe(
                state([[t + math.cos(t), math.sin(t), 0]], [[1 - math.sin(t), math.cos(t), 0]]), t, 0.01, phase="hold"
            )
        result = observer.result()["stability"]
        self.assertLess(result["jitter_m"], 1e-14)
        self.assertLess(result["rms_speed_m_s"], 1e-14)

    def test_incomplete_and_skipped_windows(self):
        """Reject short windows and a missing physical substep without shifting the grid."""
        observer = FidelityObserver(scene().model, {"stationary_intervals": [{"start_s": 0, "end_s": 1.2}]})
        for i in range(1, 121):
            if i != 10:
                observer.observe(scene().state, i / 100, 0.01, phase="hold")
        result = observer.result()["stability"]
        self.assertEqual(result["complete_window_count"], 1)
        self.assertEqual(result["discarded_window_count"], 2)
        self.assertEqual(result["windows"][0]["start_s"], 0.5)
        short = FidelityObserver(scene().model, {"stationary_phases": ["rest"]})
        short.observe(scene().state, 0.1, 0.1, phase="rest")
        self.assertIsNone(short.result()["stability"]["jitter_m"])

    def test_phase_boundaries_do_not_complete_a_window(self):
        """Keep short stationary phases separate even when their total exceeds one window."""
        observer = FidelityObserver(scene().model, {"stationary_phases": ["hold", "rest"]})
        for i in range(1, 61):
            observer.observe(scene().state, i / 100, 0.01, phase="hold" if i <= 30 else "rest")
        self.assertEqual(observer.result()["stability"]["complete_window_count"], 0)

    def test_nonfinite_is_not_silently_dropped(self):
        """Invalidate a contaminated window and emit JSON without NaN values."""
        observer = FidelityObserver(scene(1).model, {"stationary_phases": ["hold"]})
        for i in range(1, 51):
            observer.observe(state([[float("nan") if i == 10 else 0, 0, 0]], [[0, 0, 0]]), i / 100, 0.01, phase="hold")
        result = observer.result()
        self.assertFalse(result["finite"]["all_samples_finite"])
        self.assertEqual(result["finite"]["first_nonfinite_time_s"], 0.1)
        self.assertEqual(result["stability"]["complete_window_count"], 0)
        json.dumps(result, allow_nan=False)

    def test_mass_weighted_freefall(self):
        """Use the complete system momentum instead of averaging object velocities."""
        current = scene()
        current.model.particle_mass = np.array([1.0, 3.0])
        observer = FidelityObserver(
            current.model, {"gravity_only_interval": {"start_s": 0, "end_s": 0.5, "gravity_m_s2": [0, 0, -10]}}
        )
        observer.frame(current.state, 0.0)
        for i in range(1, 51):
            t = i / 100
            observer.observe(
                state([[0, 0, 1], [0, 0, 1]], [[3, 0, -10 * t], [-1, 0, -10 * t]]), t, 0.01, phase="free fall"
            )
        result = observer.result()["freefall"]
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["max_mass_weighted_velocity_error_m_s"], 0)
        self.assertEqual(result["total_mass_kg"], 4)

    def test_freefall_interval_is_normalized_without_changing_input(self):
        """Accept array-valued gravity and serialize the validated interval."""
        current = scene(1)
        interval = {
            "start_s": "0",
            "end_s": "0.5",
            "gravity_m_s2": np.array([0, 0, -10]),
            "body_indices": np.array([], dtype=np.int32),
        }
        observer = FidelityObserver(current.model, {"gravity_only_interval": interval})
        observer.frame(current.state, 0.0)
        for i in range(1, 51):
            t = i / 100
            observer.observe(state([[0, 0, 0]], [[0, 0, -10 * t]]), t, 0.01)
        result = observer.result()
        self.assertEqual(result["freefall"]["status"], "measured")
        self.assertEqual(
            json.loads(json.dumps(result, allow_nan=False))["freefall"]["interval"],
            {"start_s": 0.0, "end_s": 0.5, "gravity_m_s2": [0.0, 0.0, -10.0], "body_indices": []},
        )
        self.assertEqual(interval["start_s"], "0")
        self.assertIsInstance(interval["gravity_m_s2"], np.ndarray)

    def test_penetration_uses_finite_box_and_particle_shell(self):
        """Measure a physical box gap without extending the tabletop beyond its edge."""
        current = scene()
        current.model.particle_radius = np.array([0.1, 0.1])
        observer = FidelityObserver(
            current.model,
            {"penetration_surfaces": [{"kind": "box", "center_m": [0, 0, 0], "half_extents_m": [1, 1, 1]}]},
        )
        observer.observe(state([[0, 0, 1.05], [2, 0, 0]], [[0, 0, 0], [0, 0, 0]]), 0.1, 0.1, phase="hold")
        result = observer.result()["penetration"]
        self.assertAlmostEqual(result["minimum_sampled_particle_shell_gap_m"], -0.05)
        self.assertFalse(result["swept_crossings_checked"])

    def test_cable_surface_detects_twist(self):
        """Observe off-axis cable material speed using COM-based rigid velocity."""
        current = scene(0)
        current.cable_bodies = [0]
        current.model.body_com = np.zeros((1, 3))
        current.model.shape_body = np.array([0])
        current.model.shape_type = np.array([4])
        current.model.shape_scale = np.array([[0.1, 0.5, 0]])
        current.model.shape_transform = np.array([[0, 0, 0, 0, 0, 0, 1]])
        observer = FidelityObserver(current.model, {"stationary_phases": ["hold"], "cable_body_indices": [0]})
        for i in range(1, 51):
            t = i / 100
            s = state(np.empty((0, 3)), np.empty((0, 3)))
            s.body_q = np.array([[0, 0, 0, 0, 0, math.sin(t / 2), math.cos(t / 2)]])
            s.body_qd = np.array([[0, 0, 0, 0, 0, 1]])
            observer.observe(s, t, 0.01, phase="hold")
        result = observer.result()
        self.assertEqual(result["sampling"]["cable_surface_sample_count"], 14)
        self.assertAlmostEqual(result["stability"]["rms_speed_m_s"], 0.1)
        self.assertGreater(result["stability"]["jitter_m"], 0)

    def test_rotated_box_and_separate_numerical_margin(self):
        """Rotate the physical support and report its margin without hiding nominal gaps."""
        observer = FidelityObserver(
            scene(1).model,
            {
                "penetration_surfaces": [
                    {
                        "kind": "box",
                        "center_m": [0, 0, 0],
                        "half_extents_m": [1, 0.1, 0.1],
                        "rotation_xyzw": [0, 0, math.sqrt(0.5), math.sqrt(0.5)],
                        "shape_margin_m": 0.02,
                    }
                ]
            },
        )
        observer.observe(state([[0.11, 0, 0]], [[0, 0, 0]]), 0.1, 0.1, phase="hold")
        result = observer.result()["penetration"]
        self.assertAlmostEqual(result["minimum_sampled_particle_shell_gap_m"], 0.01)
        self.assertAlmostEqual(result["minimum_sampled_zero_force_gap_m"], -0.01)

    def test_initial_observe_and_missing_freefall_baseline(self):
        """Allow a zero-time initial sample while rejecting an invented free-fall baseline."""
        observer = FidelityObserver(
            scene().model, {"gravity_only_interval": {"start_s": 1, "end_s": 1.5, "gravity_m_s2": [0, 0, -10]}}
        )
        observer.observe(scene().state, 0, 0, phase="rest")
        observer.observe(scene().state, 1.1, 0.1, phase="free fall")
        result = observer.result()
        self.assertEqual(result["sampling"]["physical_substep_samples"], 1)
        self.assertEqual(result["freefall"]["status"], "incomplete")
        self.assertIsNone(result["freefall"]["max_mass_weighted_velocity_error_m_s"])

    def test_unobserved_entire_window_is_visible(self):
        """Report missing declared windows even if an entire window has no calls."""
        observer = FidelityObserver(scene().model, {"stationary_intervals": [{"start_s": 0, "end_s": 1}]})
        for i in range(51, 101):
            observer.observe(scene().state, i / 100, 0.01, phase="hold")
        result = observer.result()["stability"]
        self.assertEqual(result["declared_complete_window_count"], 2)
        self.assertEqual(result["missing_or_invalid_declared_window_count"], 1)

    def test_cable_offset_com_and_shape(self):
        """Use original shape placement and subtract world COM in surface velocity."""
        current = scene(0)
        current.cable_bodies = [0]
        current.model.body_com = np.array([[1, 0, 0]])
        current.model.shape_body = np.array([0])
        current.model.shape_type = np.array([4])
        current.model.shape_scale = np.array([[0.1, 0.5, 0]])
        current.model.shape_transform = np.array([[1, 0, 0, 0, 0, 0, 1]])
        observer = FidelityObserver(current.model, {"stationary_phases": ["hold"], "cable_body_indices": [0]})
        for i in range(1, 51):
            t = i / 100
            s = state(np.empty((0, 3)), np.empty((0, 3)))
            s.body_q = np.array([[0, 0, 0, 0, 0, math.sin(t / 2), math.cos(t / 2)]])
            s.body_qd = np.array([[-math.sin(t), math.cos(t), 0, 0, 0, 1]])
            observer.observe(s, t, 0.01, phase="hold")
        self.assertAlmostEqual(observer.result()["stability"]["rms_speed_m_s"], 1.1)


if __name__ == "__main__":
    unittest.main()
