# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the Levenberg-Marquardt runner fit and its residual definition."""

import itertools
import math
import tempfile
import unittest

import numpy as np
import warp as wp

from newton.tests.test_impedance_hogan import _chain, _tiny_shoe
from projects.impedance_instron.hogan import least_squares
from projects.impedance_instron.hogan.identify import Parameterization, Trial, predict, score
from projects.impedance_instron.hogan.runner import RolloutConfig, Runner, State, Task, simulate


def _initial():
    return State(
        np.array([0.01, 0.8 * math.cos(0.12) + 0.101, math.pi / 2, 0.12, -0.24, 0.12]),
        np.array([0.12, -0.5, 0.08, 0.3, -0.25, 0.1]),
        phase_rad=2 * math.pi - 0.03,
    )


class TestLeastSquares(unittest.TestCase):
    """Keep LM residuals consistent with the CEM score and fit on training data."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.shoe = _tiny_shoe(directory.name)
        self.config = RolloutConfig(dt_s=2.5e-4, contact_threshold_n=0.01)
        self.baseline = Runner.seed(reference_speed_m_s=0.3)
        self.parameters = Parameterization(self.baseline, [0.3])

    def trial(self, model=None, split="train", duration=0.008):
        """Build a trial whose targets are an exact rollout of ``model`` on irregular clocks."""
        initial = _initial()
        time = duration * np.array([0.0, 0.127, 0.43, 0.79, 1.0])
        force_time = duration * np.array([0.0, 0.11, 0.39, 0.64, 0.96, 1.0])
        q, grf = np.tile(initial.q, (5, 1)), np.zeros((6, 2))
        if model is not None:
            trace, _ = simulate(model, _chain(), self.shoe, initial, Task(0.3), duration_s=duration, config=self.config)
            q = np.column_stack([np.interp(time, trace["time_s"], trace["state"][:, c]) for c in range(6)])
            grf = np.column_stack([np.interp(force_time, trace["time_s"][:-1], trace["grf_n"][:, c]) for c in range(2)])
        return Trial(
            split,
            split,
            _chain(),
            self.shoe,
            Task(0.3),
            initial,
            time,
            q,
            force_time,
            grf,
            {"compatibility": {"passed": True}},
        )

    def test_residuals_match_score_sample_terms(self):
        """Sum squared residuals to the score's coordinate and GRF mean-square terms."""
        trial = self.trial()
        trial.q[1:] += np.array([0.003, -0.002, 0.01, -0.02, 0.015, 0.005])
        trial.grf_n[:, 1] += 30.0
        trace, summary = predict(self.baseline, trial, self.config)
        result = score(trace, summary, trial, self.baseline)
        tracking, force = np.asarray(result["tracking_rmse"]), np.asarray(result["grf_rmse_n"])
        expected = np.mean((tracking[:2] / 0.02) ** 2) + np.mean((tracking[2:] / 0.05) ** 2)
        expected += np.mean((force / 100.0) ** 2)
        r = least_squares.residuals(trace, summary, trial)
        self.assertAlmostEqual(float(r @ r), float(expected), places=10)
        self.assertIsNone(least_squares.residuals(trace, summary | {"status": "failed"}, trial))

    def test_lm_reduces_cost_toward_known_truth(self):
        """Decrease the training cost monotonically when fitting an exact truth rollout."""
        truth = self.parameters.model(np.random.default_rng(6).normal(0, 0.05, self.parameters.size))
        search = least_squares.LMConfig(iterations=3, regularization=0.0)
        learned, report = least_squares.fit_lm(self.baseline, [self.trial(truth)], config=self.config, search=search)
        costs = [report["initial_cost"]] + [row["cost"] for row in report["history"]]
        self.assertTrue(all(b <= a for a, b in itertools.pairwise(costs)))
        self.assertLess(report["final_cost"], 0.5 * report["initial_cost"])
        self.assertTrue(any(row["accepted"] for row in report["history"]))
        self.assertEqual(report["selection_split"], "train")
        self.assertIsInstance(learned, Runner)

    def test_incompatible_data_is_not_silently_fitted(self):
        """Refuse LM fitting when input compatibility fails without an explicit override."""
        trial = self.trial()
        trial.provenance["compatibility"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "compatibility"):
            least_squares.fit_lm(self.baseline, [trial], config=self.config)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_gpu_rollouts_match_cpu_residuals(self):
        """Match CPU residual vectors from persistent padded CUDA batches."""
        trials = [self.trial(), self.trial(split="train")]
        offsets = np.random.default_rng(7).normal(0, 0.02, (3, self.parameters.size))
        cpu = least_squares._Rollouts(self.parameters, trials, self.config, "cpu")(offsets, 2)
        gpu = least_squares._Rollouts(self.parameters, trials, self.config, "cuda:0")(offsets, 2)
        for a, b in zip(cpu, gpu, strict=True):
            # The unchanged shoe law runs in float32 on both backends.
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
