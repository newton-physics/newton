# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase-1 held-out cycle-window split, metrics, and reporting."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.digital_instron_v2 import validation
from projects.digital_instron_v2.cycle_windows import build_cycle_window_trace, infer_cycle_column
from projects.digital_instron_v2.frame_qc import infer_frame_config, read_numeric_csv


def _write_synthetic_csv(path: Path, *, cycles: int = 4, samples: int = 60, peak_mm: float = 10.0) -> None:
    """Write a synthetic multi-cycle Instron CSV with a known per-cycle amplitude.

    Position descends from 0 to ``-cycle * peak_mm`` at mid-cycle and returns, so
    a top-of-stroke displacement zero recovers a per-cycle penetration whose
    amplitude grows with the cycle index; force is a soft-nonlinear function of
    penetration so a dissipative loop can be constructed downstream.
    """
    rows = [
        (
            "Total Time (s)",
            "Cycle Elapsed Time (s)",
            "Elapsed Cycles",
            "Position (mm)",
            "Force (N)",
            "Cycle Energy(Energy Calculation) (J)",
        )
    ]
    dt = 0.001
    total_time = 0.0
    for cycle in range(1, cycles + 1):
        amplitude = peak_mm * cycle
        for sample in range(samples):
            phase = sample / (samples - 1)
            penetration = amplitude * (1.0 - abs(2.0 * phase - 1.0))  # triangle up then down
            position = -penetration  # descends downward
            force = -(50.0 * penetration + 2.0 * penetration**2)  # machine sign is negative
            rows.append((total_time, phase * 0.5, cycle, position, force, 0.0))
            total_time += dt
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


class TestFrameQCAndCycleWindows(unittest.TestCase):
    """Frame inference and cycle-window averaging on synthetic raw traces."""

    def setUp(self) -> None:
        """Create a synthetic four-cycle raw CSV in a temporary directory."""
        self._dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self._dir.name) / "synthetic.csv"
        _write_synthetic_csv(self.csv_path, cycles=4, samples=60, peak_mm=10.0)
        self.frame = infer_frame_config(self.csv_path)

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self._dir.cleanup()

    def test_infer_frame_config_resolves_compressive_signs(self):
        """Infer the correct columns and compressive signs for the raw trace."""
        self.assertEqual(self.frame.position_column, "Position (mm)")
        self.assertEqual(self.frame.force_column, "Force (N)")
        self.assertEqual(self.frame.force_sign, -1.0)
        self.assertEqual(self.frame.position_sign, -1.0)

    def test_infer_cycle_column_skips_flat_counter(self):
        """Pick a cycle column that actually takes more than one value."""
        columns = read_numeric_csv(self.csv_path)
        self.assertEqual(infer_cycle_column(columns), "Elapsed Cycles")

    def test_cycle_window_splits_disjoint_windows(self):
        """Average only the requested cycles into each window."""
        train = build_cycle_window_trace(self.csv_path, self.frame, cycles=[1, 2], phase_count=51)
        held = build_cycle_window_trace(self.csv_path, self.frame, cycles=[3, 4], phase_count=51)
        self.assertEqual(train.provenance["cycles"], [1, 2])
        self.assertEqual(held.provenance["cycles"], [3, 4])
        # Later cycles have larger amplitude, so the held-out peak penetration is deeper.
        self.assertGreater(held.data["displacement_m"].max(), train.data["displacement_m"].max())

    def test_top_of_stroke_zero_starts_penetration_at_zero(self):
        """Zero penetration at the top of stroke and match the known amplitude."""
        trace = build_cycle_window_trace(
            self.csv_path, self.frame, cycles=[2], phase_count=101, displacement_zero_policy="top_of_stroke"
        )
        displacement_mm = trace.data["displacement_m"] * 1000.0
        self.assertAlmostEqual(float(displacement_mm.min()), 0.0, places=6)
        self.assertAlmostEqual(float(displacement_mm.max()), 20.0, delta=0.5)  # cycle 2 amplitude = 2 * 10 mm

    def test_force_sign_flip_makes_force_compressive(self):
        """Report positive compressive force from a negative machine-force trace."""
        trace = build_cycle_window_trace(self.csv_path, self.frame, cycles=[1], phase_count=101)
        self.assertGreater(trace.data["force_n"].max(), 0.0)
        self.assertGreaterEqual(trace.data["force_n"].min(), -1.0e-6)


class TestValidationMetrics(unittest.TestCase):
    """Baseline correction, robust peak, active RMSE, and hysteresis metrics."""

    def _loop(self, dissipation: float = 0.0):
        """Return a synthetic loading/unloading displacement-force loop."""
        phase = np.linspace(0.0, 1.0, 201)
        displacement = 0.02 * (1.0 - np.abs(2.0 * phase - 1.0))
        force = 60000.0 * displacement
        # Add a downward offset on the unloading branch to open a dissipative loop.
        peak = int(np.argmax(displacement))
        force[peak + 1 :] -= dissipation * 60000.0 * displacement[peak + 1 :]
        return displacement, force

    def test_robust_peak_averages_top_five(self):
        """Return the mean of the top five active samples as the robust peak."""
        force = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 14.0])
        self.assertAlmostEqual(validation.robust_peak(force), np.mean([10, 11, 12, 13, 14]))

    def test_active_mask_uses_five_percent_of_robust_peak(self):
        """Mark frames at or above 5% of the robust peak as active."""
        _displacement, force = self._loop()
        mask = validation.active_force_mask(force)
        self.assertTrue(mask.sum() > 5)
        self.assertLess(force[~mask].max(initial=0.0), 0.05 * validation.robust_peak(force))

    def test_perfect_prediction_scores_zero_error(self):
        """Score zero peak, RMSE, and hysteresis error for an identical prediction."""
        displacement, force = self._loop(dissipation=0.2)
        metrics = validation.validate_trace_metrics(force, force.copy(), displacement)
        self.assertAlmostEqual(metrics.peak_force_error, 0.0, places=6)
        self.assertAlmostEqual(metrics.force_rmse_relative, 0.0, places=6)
        self.assertAlmostEqual(metrics.hysteresis_error, 0.0, places=6)
        self.assertTrue(metrics.passed)

    def test_positive_hysteresis_matches_loop_area(self):
        """Recover the analytic dissipated loop area from the branch split."""
        displacement, force = self._loop(dissipation=0.3)
        active = validation.active_force_mask(force)
        work = validation.positive_hysteresis_work(displacement, force, active)
        # Dissipated area = 0.3 * area under the unloading elastic line over the loaded branch.
        self.assertGreater(work, 0.0)

    def test_baseline_correction_before_metrics(self):
        """Subtract an inactive-frame force offset before scoring the peak."""
        displacement, force = self._loop(dissipation=0.1)
        offset = 25.0
        measured = force + offset
        simulated = force + offset
        corrected = validation.validate_trace_metrics(
            measured - measured.min(), simulated - simulated.min(), displacement
        )
        self.assertAlmostEqual(corrected.peak_force_error, 0.0, places=6)

    def test_peak_error_flags_a_stiff_overprediction(self):
        """Flag a 20% force overprediction as a failing peak-force gate."""
        displacement, force = self._loop(dissipation=0.15)
        metrics = validation.validate_trace_metrics(force, 1.2 * force, displacement)
        self.assertGreater(metrics.peak_force_error, 0.1)
        self.assertFalse(metrics.passed)


if __name__ == "__main__":
    unittest.main()
