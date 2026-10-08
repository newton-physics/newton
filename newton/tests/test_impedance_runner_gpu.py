# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify target-free CUDA candidate rollouts against the CPU runner."""

import math
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.test_impedance_hogan import _chain, _tiny_shoe
from projects.digital_shoe.runtime import MidsoleFoundation, SurroundConfig
from projects.impedance_instron.cartesian.shoe import Shoe
from projects.impedance_instron.hogan.gpu_runner import GpuBatch
from projects.impedance_instron.hogan.identify import Trial, score
from projects.impedance_instron.hogan.runner import Bounds, RolloutConfig, Runner, State, Task, simulate


def _initial():
    return State(
        np.array([0.01, 0.8 * math.cos(0.12) + 0.101, math.pi / 2, 0.12, -0.24, 0.12]),
        np.array([0.12, -0.5, 0.08, 0.3, -0.25, 0.1]),
        phase_rad=2 * math.pi - 0.03,
        normal_load_bw=0.25,
        torque_nm=np.array([0.5, -0.75, 0.25]),
    )


def _models():
    first = Runner.seed(reference_speed_m_s=0.2)
    weights = first.weights + np.random.default_rng(8).normal(0, 0.2, first.weights.shape)
    second = Runner(
        weights,
        bounds=Bounds(
            equilibrium_lower_rad=(-0.3, -0.9, -0.2),
            equilibrium_upper_rad=(0.5, -0.05, 0.35),
            stiffness_max_nm_rad=(800, 500, 200),
            damping_max_nms_rad=(20, 30, 10),
            torque_max_nm=(3, 4, 2),
            torque_rate_max_nm_s=(40, 60, 30),
        ),
        frequency_hz=2.3,
        response_time_s=0.014,
        reference_speed_m_s=0.1,
        speed_scale_m_s=0.4,
        cadence_speed_gain=0.4,
        load_time_s=0.007,
        phase_feedback=0.65,
    )
    return [first, second]


@unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for runner GPU tests")
class TestRunnerGpu(unittest.TestCase):
    """Compare full interval/state traces, screens, and independent graph resets."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.shoe = _tiny_shoe(cls.directory.name)
        cls.chain = _chain()
        cls.cfg = RolloutConfig(dt_s=1e-4, contact_threshold_n=0.01)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def assertParity(self, actual, expected):
        trace, summary = actual
        reference, info = expected
        self.assertEqual(set(trace), set(reference))
        self.assertEqual(set(summary), set(info))
        for name in trace:
            with self.subTest(field=name):
                self.assertEqual(trace[name].shape, reference[name].shape)
                if name == "torque_saturated":
                    np.testing.assert_array_equal(trace[name], reference[name])
                else:
                    # Identical Shoe.apply poses already differ by ~5e-5 N on
                    # CPU/CUDA for the pitched legacy fixture (float32 law).
                    tolerance = 1e-4 if name == "grf_n" else 2e-5
                    np.testing.assert_allclose(trace[name], reference[name], rtol=2e-5, atol=tolerance)
        for name, value in info.items():
            with self.subTest(summary=name):
                if value is None or isinstance(value, (str, bool)):
                    self.assertEqual(summary[name], value)
                else:
                    tolerance = 1e-4 if name == "peak_grf_n" else 2e-5
                    np.testing.assert_allclose(summary[name], value, rtol=2e-5, atol=tolerance)

    def batch(self, initials, durations, models, *, shoes=None, tasks=None, config=None, **kwargs):
        count = len(initials)
        return GpuBatch(
            [self.chain] * count,
            shoes or [self.shoe] * count,
            initials,
            tasks or [Task(0.3)] * count,
            durations,
            candidates=len(models),
            config=config or self.cfg,
            chunk_steps=13,
            **kwargs,
        )

    def test_moving_contact_phase_and_candidate_parity(self):
        """Match CPU contact, variable phase, bounds, activation, and all summaries."""
        models = _models()
        initial = _initial()
        other = initial.copy()
        other.phase_rad = -0.15
        other.normal_load_bw = 1.3
        other.v[0] = -0.1
        initials, tasks = [initial, other], [Task(0.3), Task(0.7)]
        before = [s.copy() for s in initials]
        batch = self.batch(initials, [0.008, 0.008], models, tasks=tasks)
        self.assertEqual(batch.group_trial_indices, ((0, 1),))
        result = batch.evaluate(models)
        for c, model in enumerate(models):
            for s, state in enumerate(initials):
                expected = simulate(model, self.chain, self.shoe, state, tasks[s], duration_s=0.008, config=self.cfg)
                self.assertParity(result[c][s], expected)
                trace, summary = result[c][s]
                self.assertEqual(summary["status"], "completed")
                self.assertGreater(np.ptp(trace["grf_n"][:, 1]), 0.01)
                self.assertGreater(summary["contact_duration_s"], 0)
                self.assertLess(summary["contact_duration_s"], 0.008)
                self.assertGreater(np.ptp(trace["phase_rad"]), 0.01)
                self.assertGreater(np.ptp(trace["equilibrium_rad"][:, 0]), 1e-5)
                np.testing.assert_array_equal(trace["load"][:, :3], 0)
                torque = np.vstack((state.torque_nm, trace["load"][:, 3:]))
                self.assertTrue(np.all(np.abs(torque) <= model.bounds.torque_max_nm))
                self.assertTrue(
                    np.all(
                        np.abs(np.diff(torque, axis=0))
                        <= np.asarray(model.bounds.torque_rate_max_nm_s) * summary["dt_s"] + 1e-12
                    )
                )
        for state, saved in zip(initials, before, strict=True):
            np.testing.assert_array_equal(state.q, saved.q)
            np.testing.assert_array_equal(state.v, saved.v)
            np.testing.assert_array_equal(state.torque_nm, saved.torque_nm)

    def test_reset_candidate_and_trial_permutation(self):
        """Reset every recurrent state and preserve outputs under world permutation."""
        models = _models()
        first, second = _initial(), _initial()
        second.v[0] *= -1
        second.phase_rad = 1.2
        batch = self.batch([first, second], [0.008, 0.008], models)
        expected = batch.evaluate(models)
        reverse_models = batch.evaluate(models[::-1])
        repeated = batch.evaluate(models)
        permuted = self.batch([second, first], [0.008, 0.008], models).evaluate(models)
        for c in range(2):
            for s in range(2):
                for actual in (reverse_models[1 - c][s], repeated[c][s], permuted[c][1 - s]):
                    for name in expected[c][s][0]:
                        np.testing.assert_array_equal(actual[0][name], expected[c][s][0][name])
                    self.assertEqual(actual[1], expected[c][s][1])

    def test_intrinsic_damping_parity(self):
        """Match CPU traces when lag-free joint damping is nonzero, including its torque cap."""
        models = [
            Runner.from_dict({**model.to_dict(), "intrinsic_damping_nms_rad": damping})
            for model, damping in zip(_models(), ([3.0, 2.0, 1.0], [40.0, 0.0, 25.0]), strict=True)
        ]
        initial = _initial()
        batch = self.batch([initial], [0.008], models)
        for c, model in enumerate(models):
            expected = simulate(model, self.chain, self.shoe, initial, Task(0.3), duration_s=0.008, config=self.cfg)
            self.assertParity(batch.evaluate(models)[c][0], expected)
            self.assertTrue(np.all(np.abs(expected[0]["load"][:, 3:]) <= np.asarray(model.bounds.torque_max_nm)))
        self.assertTrue(np.any(np.abs(expected[0]["load"][:, 3:]) == np.asarray(models[1].bounds.torque_max_nm)))

    def test_mixed_exact_timesteps_durations_and_shoes(self):
        """Use each adjusted timestep for both the shoe and chain without rounding."""
        models = _models()
        shifted = Shoe(self.shoe.artifact_path, [0.005, 0, 0.102], 0.01, friction_model="legacy")
        initials = [_initial() for _ in range(4)]
        durations = [0.00305, 0.0031, 0.0062, 0.00305]
        shoes = [self.shoe, self.shoe, self.shoe, shifted]
        batch = self.batch(initials, durations, models, shoes=shoes)
        expected_steps = [math.ceil(t / self.cfg.dt_s) for t in durations]
        np.testing.assert_array_equal(batch.steps, expected_steps)
        np.testing.assert_array_equal(batch.dts, np.asarray(durations) / expected_steps)
        self.assertNotEqual(batch.dts[0], batch.dts[1])
        self.assertEqual(batch.group_trial_indices, ((0,), (1, 2), (3,)))
        results = batch.evaluate(models)
        for c, model in enumerate(models):
            for s, duration in enumerate(durations):
                expected = simulate(
                    model, self.chain, shoes[s], initials[s], Task(0.3), duration_s=duration, config=self.cfg
                )
                self.assertParity(results[c][s], expected)
                self.assertEqual(len(results[c][s][0]["load"]), expected_steps[s])
                self.assertEqual(results[c][s][1]["dt_s"], duration / expected_steps[s])

    def test_failure_screens_and_accepted_interval_semantics(self):
        """Match immediate and post-step rejection without recording rejected steps."""
        models = [Runner.seed()]
        low, fast, falling, valid = (_initial() for _ in range(4))
        low.q[1] = 0.1
        fast.v[0] = 110
        falling.q[1] = 0.91001
        falling.v[1] = -1
        cases = [
            (low, self.cfg, "Hip height screen exceeded"),
            (fast, self.cfg, "Numerical speed screen exceeded"),
            (falling, replace(self.cfg, minimum_hip_height_m=0.91), "Integrated state exceeded height/speed screen"),
            (valid, replace(self.cfg, maximum_force_n=0.001), "Contact force screen exceeded"),
            (valid, replace(self.cfg, compression_limit=0.001), "Driven shoe compression screen exceeded"),
        ]
        for initial, cfg, reason in cases:
            with self.subTest(reason=reason):
                batch = self.batch([initial], [0.008], models, config=cfg)
                actual = batch.evaluate(models)[0][0]
                expected = simulate(models[0], self.chain, self.shoe, initial, Task(0.3), duration_s=0.008, config=cfg)
                self.assertParity(actual, expected)
                self.assertEqual(actual[1]["failure"], reason)
                self.assertEqual(len(actual[0]["state"]), len(actual[0]["load"]) + 1)
                self.assertParity(batch.evaluate(models)[0][0], expected)
        batch = self.batch([low, valid], [0.008, 0.008], models)
        result = batch.evaluate(models)
        self.assertEqual(result[0][0][1]["status"], "failed")
        self.assertEqual(result[0][1][1]["status"], "completed")

    def test_offline_score_and_future_target_isolation(self):
        """Consume GPU results in offline scoring without uploading future targets."""
        model = _models()[0]
        initial = _initial()
        clock = np.array([0, 0.003, 0.008])
        trial = Trial(
            "synthetic",
            "train",
            self.chain,
            self.shoe,
            Task(0.3),
            initial,
            clock,
            np.tile(initial.q, (3, 1)),
            clock,
            np.zeros((3, 2)),
            {},
        )
        batch = self.batch([trial.initial], [trial.duration_s], [model])
        trace, summary = batch.evaluate([model])[0][0]
        cpu = simulate(model, self.chain, self.shoe, initial, trial.task, duration_s=trial.duration_s, config=self.cfg)
        first = score(trace, summary, trial, model)
        reference = score(*cpu, trial, model)
        self.assertAlmostEqual(first["loss"], reference["loss"], delta=1e-4)
        trial.q[1:] += 5
        trial.grf_n[:] = 1000
        self.assertGreater(score(trace, summary, trial, model)["loss"], first["loss"])
        repeated = batch.evaluate([model])[0][0]
        for name in trace:
            np.testing.assert_array_equal(repeated[0][name], trace[name])

    def test_passive_shoe_history_and_runtime_material(self):
        """Preserve passive sweeps, shoe settings, runtime material, and reset isolation."""
        shoe = Shoe(self.shoe.artifact_path, [0, 0, 0.1], 0, friction_model="legacy")
        bed = shoe.shoe.column_bed
        config = replace(shoe.foundation.config, normal_damping=0.2, mu=0.4)
        shoe.foundation = MidsoleFoundation(
            shoe.anchor_local_m,
            np.zeros(2),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
            shoe.shoe.material,
            0,
            shoe.model.body_com,
            config,
            shoe.device,
            SurroundConfig(driven=np.array([1, 0]), carrier_bond=True, sweeps=3, relaxation_time_s=0.002),
        )
        material = deepcopy(shoe.shoe.material)
        material = replace(material, maxwell_relaxation_time_s=material.maxwell_relaxation_time_s * 1.5)
        shoe.foundation.set_world_material(0, material)
        models, initial = _models(), _initial()
        batch = self.batch([initial], [0.008], models, shoes=[shoe])
        before = shoe.foundation.q_state.numpy().copy()
        first = batch.evaluate(models)
        np.testing.assert_array_equal(shoe.foundation.q_state.numpy(), before)
        self.assertGreater(float(batch._groups[0].foundation.surround_compression.numpy().max()), 0)
        for c, model in enumerate(models):
            expected = simulate(model, self.chain, shoe, initial, Task(0.3), duration_s=0.008, config=self.cfg)
            self.assertParity(first[c][0], expected)
        # A warm graph must neither download device arrays nor explicitly synchronize.
        with (
            patch.object(wp.array, "numpy", side_effect=AssertionError("Host read during rollout")),
            patch.object(wp, "synchronize_device", side_effect=AssertionError("Host sync during rollout")),
        ):
            batch._groups[0].launch(models)
        repeated = batch._groups[0].collect()
        for c in range(len(models)):
            for name in first[c][0][0]:
                np.testing.assert_array_equal(first[c][0][0][name], repeated[c][0][0][name])

    def test_validation_and_group_bound(self):
        """Reject invalid batches and initial torques before launching candidates."""
        initial, models = _initial(), _models()
        for kwargs in (
            {"candidates": 0},
            {"candidates": True},
            {"chunk_steps": 0},
            {"max_trials_per_group": 0},
            {"device": "cpu"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                arguments = {"candidates": 2, "config": self.cfg, **kwargs}
                GpuBatch([self.chain], [self.shoe], [initial], [Task(0.3)], [0.002], **arguments)
        for durations in ([0], [float("nan")], [], [0.001, 0.002]):
            with self.assertRaises(ValueError):
                self.batch([initial], durations, models)
        batch = self.batch([initial] * 3, [0.002] * 3, models, max_trials_per_group=2)
        self.assertEqual(batch.group_trial_indices, ((0, 1), (2,)))
        with self.assertRaises(ValueError):
            batch.evaluate(models[:1])
        initial.torque_nm[0] = 10
        bad = self.batch([initial], [0.002], models)
        with self.assertRaisesRegex(ValueError, "Initial torque"):
            bad.evaluate(models)


if __name__ == "__main__":
    unittest.main(verbosity=2)
