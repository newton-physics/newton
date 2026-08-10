# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase-2 dynamic Digital Instron replay through SolverMuJoCo."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_instron_v2 import phase1, phase2, workflow
from projects.digital_instron_v2.core import predict
from projects.digital_instron_v2.geometry import build_column_grid, load_mesh
from projects.digital_instron_v2.phase2 import DynamicReplayConfig, DynamicReplayResult

MANIFEST = "DigitalInstron/manifest_v2.json"
_HAS_MUJOCO = importlib.util.find_spec("mujoco_warp") is not None


def _synthetic_result(force: np.ndarray, phase: np.ndarray, **overrides) -> DynamicReplayResult:
    """Build a DynamicReplayResult with the given force/phase and neutral defaults."""
    n = len(force)
    fields = {
        "fixture": "fullfoot_last",
        "drive": "kinematic",
        "carrier_body": 0,
        "dt_s": 5.0e-4,
        "period_s": 0.5,
        "phase": phase,
        "commanded_depth_m": np.linspace(0.0, 0.018, n),
        "achieved_depth_m": np.linspace(0.0, 0.018, n),
        "force_n": force,
        "cop_x_m": np.zeros(n),
        "cop_y_m": np.zeros(n),
        "active_columns": np.full(n, 600.0),
        "wrench_fz_n": force,
        "moment_mx_nm": np.zeros(n),
        "moment_my_nm": np.zeros(n),
        "max_compression_m": np.full(n, 0.018),
        "column_count": 600,
        "column_area_m2": 2.5e-5,
    }
    fields.update(overrides)
    return DynamicReplayResult(**fields)


class TestPhase2Helpers(unittest.TestCase):
    def test_signed_loop_area_unit_square(self):
        """Recover the exact area of a unit square traversed counter-clockwise."""
        x = np.array([0.0, 1.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(abs(phase2._signed_loop_area(x, y)), 1.0, places=12)

    def test_resample_force_recovers_periodic_triangle(self):
        """Interpolate a closed (periodic) force loop onto a finer phase grid without distortion."""
        phase = np.linspace(0.0, 1.0, 11)
        triangle = 200.0 * np.minimum(phase, 1.0 - phase)  # 0 at both ends, peak at phase 0.5
        result = _synthetic_result(triangle, phase)
        grid = np.linspace(0.0, 1.0, 21)
        out = phase2._resample_force(result, grid)
        expected = 200.0 * np.minimum(grid % 1.0, 1.0 - (grid % 1.0))
        self.assertTrue(np.allclose(out, expected, atol=1.0e-9))

    def test_diagnostics_reports_peak_cop_and_continuity(self):
        """Summarize peak COP at the force peak, max jump, area, and finiteness."""
        phase = np.linspace(0.0, 1.0, 5)
        force = np.array([0.0, 10.0, 40.0, 10.0, 0.0])
        result = _synthetic_result(
            force, phase, cop_x_m=np.array([0.0, 0.01, 0.02, 0.01, 0.0]), active_columns=np.array([0.0, 3, 5, 3, 0.0])
        )
        diag = phase2._diagnostics(result)
        self.assertEqual(diag["peak_active_cells"], 5)
        self.assertAlmostEqual(diag["peak_cop_m"][0], 0.02)
        self.assertAlmostEqual(diag["max_force_jump_n"], 30.0)
        self.assertAlmostEqual(diag["peak_active_area_m2"], 5 * 2.5e-5)
        self.assertTrue(diag["force_finite"])
        self.assertAlmostEqual(diag["tracking_error_max_m"], 0.0)

    def test_servo_config_mirrors_kinematic_settings(self):
        """Derive a servo-drive twin that keeps every timestep and gain from the kinematic config."""
        base = DynamicReplayConfig(drive="kinematic", substeps=24, warmup_cycles=2, servo_ke=3.0e7)
        servo = phase2._servo_config(base)
        self.assertEqual(servo.drive, "servo")
        self.assertEqual(servo.substeps, 24)
        self.assertEqual(servo.warmup_cycles, 2)
        self.assertEqual(servo.servo_ke, 3.0e7)

    def test_default_drive_is_kinematic(self):
        """Default the replay to the faithful position-controlled crosshead."""
        self.assertEqual(DynamicReplayConfig().drive, "kinematic")

    def test_load_validation_trace_subtracts_force_baseline(self):
        """Load a held-out trace and shift force so its inactive baseline is zero."""
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "validate.csv"
            csv.write_text(
                "phase,time_s,displacement_m,force_n\n0.0,0.0,0.0,20.0\n0.5,0.25,0.018,1200.0\n1.0,0.5,0.0,20.0\n"
            )
            trace = phase2._load_validation_trace({"validate": {"probe": str(csv)}}, "probe")
        self.assertAlmostEqual(float(trace["force_n"].min()), 0.0)
        self.assertAlmostEqual(float(trace["force_n"].max()), 1180.0)
        self.assertTrue(np.all(trace["displacement_m"] >= 0.0))


@unittest.skipUnless(wp.is_cuda_available() and _HAS_MUJOCO, "requires a CUDA device and mujoco_warp")
class TestPhase2DynamicReplay(unittest.TestCase):
    def test_kinematic_replay_reproduces_static_force(self):
        """Reproduce the Phase-1 static held-out force with the coupled Warp/MuJoCo kernel.

        A kinematic crosshead prescribing the exact measured pose must yield the same
        force history (and hence peak/RMSE/hysteresis) as the analytic
        ``core.predict``, proving the dynamic implementation of the calibrated law is
        faithful, phase aligned, and stable.
        """
        path = Path(MANIFEST).resolve()
        config = json.loads(path.read_text())
        base = path.parent
        material, _, split = phase1.fit_train_material(path)
        midsole = load_mesh(base / config["midsole_mesh"], 0.001)
        grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
        validate, _, _ = workflow.prepare_trials(base, config, grid, midsole, trace_paths=split["validate"])
        trial = next(t for t in validate if t.name == "fullfoot_185ms")
        static = predict(trial, material)
        static = static - static.min()

        trace = phase2._load_validation_trace(split, "fullfoot_185ms")
        cfg = DynamicReplayConfig(drive="kinematic", substeps=16, warmup_cycles=2)
        result = phase2.run_dynamic_replay(path, "fullfoot_last", material, trace, cfg)
        simulated = phase2._resample_force(result, trace["phase"])
        simulated = simulated - simulated.min()

        self.assertTrue(np.all(np.isfinite(result.force_n)))
        self.assertLess(float(np.max(np.abs(result.commanded_depth_m - result.achieved_depth_m))), 1.0e-6)
        peak_rel = abs(simulated.max() - static.max()) / static.max()
        self.assertLess(peak_rel, 0.02)
        self.assertLess(float(np.sqrt(np.mean((simulated - static) ** 2)) / static.max()), 0.03)

    def test_servo_drive_tracks_and_stays_finite(self):
        """Track the crosshead trajectory in closed loop within tolerance and stay finite."""
        path = Path(MANIFEST).resolve()
        material, _, split = phase1.fit_train_material(path)
        trace = phase2._load_validation_trace(split, "rearfoot_140ms")
        cfg = DynamicReplayConfig(drive="servo", substeps=32, warmup_cycles=1)
        result = phase2.run_dynamic_replay(path, "rearfoot_punch", material, trace, cfg)
        self.assertTrue(np.all(np.isfinite(result.force_n)))
        track = float(np.max(np.abs(result.commanded_depth_m - result.achieved_depth_m)))
        self.assertLess(track, 1.0e-3)


if __name__ == "__main__":
    unittest.main()
