# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify synthetic parameter recovery and local identifiability analysis."""

import json
import math
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from newton.tests.test_impedance_hogan import _chain, _tiny_shoe
from projects.impedance_instron.hogan import recovery
from projects.impedance_instron.hogan.identify import FitConfig, Parameterization, Trial, predict_many
from projects.impedance_instron.hogan.runner import RolloutConfig, Runner, State, Task, simulate


def _initial():
    return State(
        np.array([0.01, 0.8 * math.cos(0.12) + 0.101, math.pi / 2, 0.12, -0.24, 0.12]),
        np.array([0.12, -0.5, 0.08, 0.3, -0.25, 0.1]),
        phase_rad=2 * math.pi - 0.03,
    )


class TestRecovery(unittest.TestCase):
    """Keep synthetic targets self-consistent and measured targets unused."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.shoe = _tiny_shoe(directory.name)
        self.chain = _chain()
        self.config = RolloutConfig(dt_s=2.5e-4, contact_threshold_n=0.01)
        self.baseline = Runner.seed(reference_speed_m_s=0.3)
        self.parameters = Parameterization(self.baseline, [0.3])

    def trial(self, split="train", duration=0.008):
        initial = _initial()
        time = duration * np.array([0.0, 0.127, 0.43, 0.79, 1.0])
        force_time = duration * np.array([-0.1, 0.11, 0.39, 0.64, 0.96, 1.2])
        return Trial(
            split,
            split,
            self.chain,
            self.shoe,
            Task(0.3),
            initial,
            time,
            initial.q + time[:, None] * initial.v,
            force_time,
            np.full((6, 2), 777.0),
            {"compatibility": {"passed": False}, "shoe_artifact": "unused"},
        )

    def test_offsets_round_trip_and_names(self):
        """Invert the parameterization and name every offset uniquely."""
        x = np.random.default_rng(1).normal(0, 0.2, self.parameters.size)
        np.testing.assert_allclose(recovery.offsets_of(self.parameters, self.parameters.model(x)), x, atol=1e-12)
        names = recovery.parameter_names(self.parameters)
        self.assertEqual(len(names), self.parameters.size)
        self.assertEqual(len(set(names)), len(names))
        self.assertIn("stiffness.knee.load_sin_phase", names)
        self.assertNotIn("equilibrium.hip.task_speed_offset", names)

    def test_synthesize_replaces_targets_only(self):
        """Sample truth predictions on trial clocks without changing predictive inputs."""
        trial = self.trial()
        truth = self.parameters.model(np.full(self.parameters.size, 0.05))
        predictions = predict_many([truth], [trial], self.config, device="cpu")[0]
        exact = recovery.synthesize([trial], predictions, recovery.Noise(0, 0, 0), np.random.default_rng(0))[0]
        trace = predictions[0][0]
        np.testing.assert_allclose(exact.q[-1], trace["state"][-1], atol=1e-12)
        np.testing.assert_array_equal(exact.q[0], trial.initial.q)
        self.assertEqual((exact.force_time_s[0], exact.force_time_s[-1]), (0.0, trial.duration_s))
        self.assertFalse(np.any(exact.grf_n == 777.0))
        self.assertIs(exact.initial, trial.initial)
        self.assertIs(exact.shoe, trial.shoe)
        self.assertTrue(exact.provenance["compatibility"]["passed"])
        self.assertFalse(exact.provenance["source_compatibility"]["passed"])
        self.assertFalse(trial.provenance["compatibility"]["passed"])
        noisy = recovery.synthesize([trial], predictions, recovery.Noise(), np.random.default_rng(0))[0]
        self.assertGreater(np.max(np.abs(noisy.grf_n - exact.grf_n)), 1.0)
        self.assertGreater(np.max(np.abs(noisy.q[:, 3:] - exact.q[:, 3:])), 1e-3)

    def test_verdict_classification(self):
        """Separate optimizer failure from equally good but different impedance."""
        criteria = recovery.Criteria()
        self.assertEqual(recovery.verdict(1.0, 2.0, 0.0, 1.0, criteria), "inconclusive_search")
        self.assertEqual(recovery.verdict(1.0, 1.05, 0.01, 0.2, criteria), "recovered")
        self.assertEqual(recovery.verdict(1.0, 0.9, 0.2, 0.3, criteria), "not_identifiable")

    def test_identifiability_flags_redundant_and_unobservable_parameters(self):
        """Report rank loss and large Cramér-Rao bounds for degenerate columns."""
        rng = np.random.default_rng(2)
        strong = rng.normal(size=50) * 100
        other = rng.normal(size=50)
        jacobian = np.column_stack((strong, other, other, np.zeros(50), np.full(50, np.nan)))
        names = ["strong", "a", "b", "zero", "failed"]
        result = recovery.identifiability(jacobian, names, recovery.Criteria())
        self.assertEqual(result["failed_columns"], ["failed"])
        self.assertEqual(result["parameters"], 4)
        self.assertEqual(result["rank_relative_1e-6"], 2)
        self.assertIn("strong", result["well_determined"])
        self.assertTrue({"a", "b", "zero"} <= set(result["poorly_determined"]))
        self.assertEqual(len(result["weakest_directions"]), 3)
        json.dumps(result, allow_nan=False)

    def test_sensitivity_matches_manual_central_difference(self):
        """Whiten sampled observation derivatives by the declared noise levels."""
        trial, noise, step = self.trial(), recovery.Noise(), 0.01
        index = recovery.parameter_names(self.parameters).index("stiffness.knee.bias")
        center = np.zeros(self.parameters.size)

        def model(x):
            full = center.copy()
            full[index] = x[0]
            return self.parameters.model(full)

        reduced = SimpleNamespace(size=1, model=model)
        jacobian = recovery.sensitivity(reduced, np.zeros(1), [trial], self.config, noise, step=step)
        sampled = []
        for sign in (1, -1):
            trace, _ = simulate(
                model([sign * step]),
                trial.chain,
                trial.shoe,
                trial.initial,
                trial.task,
                duration_s=trial.duration_s,
                config=self.config,
            )
            q, grf = recovery._sample(trace, trial.time_s[1:], trial.force_time_s)
            sampled.append(np.concatenate(((q / noise.q).ravel(), (grf / noise.force_n).ravel())))
        np.testing.assert_allclose(jacobian[:, 0], (sampled[0] - sampled[1]) / (2 * step), rtol=0, atol=1e-9)
        self.assertGreater(np.max(np.abs(jacobian)), 0)

    def test_recover_never_reads_measured_targets(self):
        """Produce identical recovery results after corrupting all measured targets."""
        trials = [self.trial(), self.trial("eval")]
        kwargs = {
            "config": self.config,
            "search": FitConfig(population=4, generations=1, seed=5),
            "truth_scale": 0.1,
            "truth_attempts": 2,
            "seed": 3,
            "sensitivity_step": None,
        }
        truth, learned, report = recovery.recover(trials, **kwargs)
        for trial in trials:
            trial.q[:] = 50.0
            trial.grf_n[:] = -5000.0
        other_truth, other_learned, other = recovery.recover(trials, **kwargs)
        np.testing.assert_array_equal(truth.weights, other_truth.weights)
        np.testing.assert_array_equal(learned.weights, other_learned.weights)
        self.assertEqual(report["offsets"], other["offsets"])
        self.assertFalse(report["measured_targets_used"])
        self.assertFalse(report["validated"])
        self.assertIn(report["verdict"], ("recovered", "not_identifiable", "inconclusive_search"))
        self.assertEqual(set(report["splits"]), {"train", "eval"})
        self.assertGreater(report["offsets"]["truth_distance_from_seed"], 0)
        self.assertGreater(report["splits"]["train"]["seed_impedance_error"]["mean_normalized"], 0)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
