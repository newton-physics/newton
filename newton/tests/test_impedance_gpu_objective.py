# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify resident and streaming CUDA scoring against the CPU objective."""

import gc
import math
import tempfile
import unittest
import weakref
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.test_impedance_hogan import _chain, _tiny_shoe
from newton.tests.test_impedance_runner_gpu import _initial, _models
from projects.impedance_instron.hogan import gpu_objective, identify
from projects.impedance_instron.hogan.gpu_objective import GpuEvaluator
from projects.impedance_instron.hogan.gpu_runner import GpuBatch, _Group, _model_params
from projects.impedance_instron.hogan.identify import Trial, score
from projects.impedance_instron.hogan.runner import RolloutConfig, Task, simulate


@unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for GPU objective tests")
class TestGpuObjective(unittest.TestCase):
    """Keep target clocks, failure truncation, reset, and host transfers faithful."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.shoe = _tiny_shoe(cls.directory.name)
        cls.chain = _chain()
        cls.config = RolloutConfig(dt_s=1e-4, contact_threshold_n=0.01)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def trial(self, name="trial", duration=0.008, initial=None):
        initial = initial or _initial()
        time = duration * np.array([0.0, 0.127, 0.43, 0.79, 1.0])
        q = initial.q + time[:, None] * initial.v
        q[:, 3:] += np.sin(time[:, None] * np.array([160.0, 80.0, 200.0])) * 0.01
        # The zero-time observation must not influence tracking loss.
        q[0] += 20.0
        force_time = duration * np.array([-0.1, 0.11, 0.39, 0.64, 0.96, 1.2])
        grf = np.array([[2, -1], [-4, 0], [8, 40], [-1, 80], [4, 10], [600, 5000]], dtype=float)
        return Trial(name, "train", self.chain, self.shoe, Task(0.3), initial, time, q, force_time, grf, {})

    def assertScores(self, evaluator, trials, models, *, cpu=False):
        result = evaluator.evaluate(models)
        expected = [[] for _ in models]
        offset = 0
        for batch in evaluator._batches:
            for group in batch._groups:
                traces = group.collect()
                for c, model in enumerate(models):
                    for s, index in enumerate(group.indices):
                        trial = trials[offset + index]
                        value = score(*traces[c][s], trial, model)
                        expected[c].append(value)
                        if cpu:
                            prediction = simulate(
                                model,
                                trial.chain,
                                trial.shoe,
                                trial.initial,
                                trial.task,
                                duration_s=trial.duration_s,
                                config=evaluator.config,
                            )
                            reference = score(*prediction, trial, model)
                            self.assertEqual(value["status"], reference["status"])
                            self.assertAlmostEqual(value["loss"], reference["loss"], delta=1e-4)
            offset += batch.trial_count
        for c, rows in enumerate(expected):
            self.assertEqual(set(result[c]), {"mean_loss", "failed"})
            self.assertEqual(result[c]["failed"], sum(row["status"] != "completed" for row in rows))
            self.assertAlmostEqual(result[c]["mean_loss"], np.mean([row["loss"] for row in rows]), delta=1e-9)
        return result, expected

    def test_moving_contact_native_clocks_and_cpu_parity(self):
        """Match CPU scores for moving contact on distinct force and motion clocks."""
        second = _initial()
        second.phase_rad = 1.2
        second.v[0] = -0.1
        trials = [self.trial("first"), self.trial("second", initial=second)]
        models = _models()
        evaluator = GpuEvaluator(trials, candidates=2, config=self.config)
        result, rows = self.assertScores(evaluator, trials, models, cpu=True)
        self.assertEqual([row["failed"] for row in result], [0, 0])
        for candidate in rows:
            for row in candidate:
                self.assertGreater(row["contact_duration_s"], 0)
                self.assertLess(row["contact_duration_s"], trials[0].duration_s)
                self.assertGreater(row["peak_grf_n"][1], 0.01)
        self.assertNotEqual(result[0]["mean_loss"], result[1]["mean_loss"])

    def test_terminal_clock_roundoff_holds_endpoint(self):
        """Clamp native endpoint interpolation when the simulation clock rounds down."""
        duration = next(
            float(t)
            for t in np.linspace(0.001, 0.008, 1000)
            if (float(t) / math.ceil(float(t) / self.config.dt_s)) * math.ceil(float(t) / self.config.dt_s) < t
        )
        trial = self.trial(duration=duration)
        evaluator = GpuEvaluator([trial], candidates=1, config=self.config)
        fractions = evaluator._scorers[0][1].fraction.numpy()
        self.assertEqual(fractions[-1], 1.0)
        self.assertScores(evaluator, [trial], _models()[:1], cpu=True)

    def test_heterogeneous_timesteps_chunking_and_trial_order(self):
        """Preserve exact adjusted timesteps and aggregate all uneven trial chunks."""
        durations = [0.00305, 0.0031, 0.0062, 0.00305, 0.00413]
        trials = [self.trial(str(i), duration) for i, duration in enumerate(durations)]
        models = _models()
        evaluator = GpuEvaluator(trials, candidates=2, config=self.config, max_trials_per_batch=2)
        self.assertEqual([batch.trial_count for batch in evaluator._batches], [2, 2, 1])
        dts = np.concatenate([batch.dts for batch in evaluator._batches])
        np.testing.assert_array_equal(dts, [t / math.ceil(t / self.config.dt_s) for t in durations])
        self.assertNotEqual(dts[0], dts[1])
        first, _ = self.assertScores(evaluator, trials, models, cpu=True)
        permuted = GpuEvaluator(trials[::-1], candidates=2, config=self.config, max_trials_per_batch=3)
        second = permuted.evaluate(models)
        np.testing.assert_allclose([r["mean_loss"] for r in first], [r["mean_loss"] for r in second], atol=1e-10)
        self.assertEqual([r["failed"] for r in first], [r["failed"] for r in second])

    def test_failure_screens_accepted_intervals_and_full_target_horizon(self):
        """Score immediate and late failures using only accepted simulation intervals."""
        low, fast, falling = _initial(), _initial(), _initial()
        low.q[1] = 0.1
        fast.v[0] = 110
        falling.q[1] = 0.91001
        falling.v[1] = -1
        cases = [
            (low, self.config, "Hip height screen exceeded"),
            (fast, self.config, "Numerical speed screen exceeded"),
            (falling, replace(self.config, minimum_hip_height_m=0.91), "Integrated state exceeded height/speed screen"),
            (_initial(), replace(self.config, maximum_force_n=0.001), "Contact force screen exceeded"),
            (_initial(), replace(self.config, compression_limit=0.001), "Driven shoe compression screen exceeded"),
        ]
        for initial, config, reason in cases:
            with self.subTest(reason=reason):
                trial = self.trial(initial=initial)
                evaluator = GpuEvaluator([trial], candidates=1, config=config)
                result, rows = self.assertScores(evaluator, [trial], _models()[:1], cpu=True)
                self.assertEqual(result[0]["failed"], 1)
                self.assertEqual(rows[0][0]["failure"], reason)
                self.assertGreaterEqual(result[0]["mean_loss"], 1000)
                if "Contact force" in reason or "compression" in reason:
                    self.assertGreater(rows[0][0]["integrated_duration_s"], 0)
        trials = [self.trial("bad", initial=low), self.trial("good")]
        evaluator = GpuEvaluator(trials, candidates=2, config=self.config)
        result, _ = self.assertScores(evaluator, trials, _models())
        self.assertEqual([row["failed"] for row in result], [1, 1])

    def test_warm_evaluation_downloads_only_candidate_aggregates(self):
        """Forbid trajectory reads, target preprocessing, host scores, and recapture when warm."""
        trials = [self.trial(str(i), duration) for i, duration in enumerate([0.008, 0.00305, 0.0062])]
        evaluator = GpuEvaluator(trials, candidates=2, config=self.config, max_trials_per_batch=2)
        models = _models()
        expected = evaluator.evaluate(models)
        original_numpy = wp.array.numpy
        downloads = []

        def guarded_numpy(array):
            self.assertIs(array, evaluator._aggregates, "Only candidate aggregates may be downloaded")
            downloads.append(array.shape)
            return original_numpy(array)

        with (
            patch.object(wp.array, "numpy", guarded_numpy),
            patch.object(wp, "synchronize_device", side_effect=AssertionError("Warm explicit synchronization")),
            patch.object(_Group, "collect", side_effect=AssertionError("Trajectory collection")),
            patch.object(GpuBatch, "__init__", side_effect=AssertionError("Batch reconstruction")),
            patch.object(gpu_objective, "_target_arrays", side_effect=AssertionError("Target preprocessing")),
            patch.object(np, "interp", side_effect=AssertionError("Host interpolation")),
            patch.object(identify, "score", side_effect=AssertionError("CPU scoring")),
            patch.object(identify, "evaluate", side_effect=AssertionError("CPU evaluation")),
            patch.object(identify, "predict", side_effect=AssertionError("CPU prediction")),
        ):
            reversed_result = evaluator.evaluate(models[::-1])
            repeated = evaluator.evaluate(models)
        self.assertEqual(reversed_result, expected[::-1])
        self.assertEqual(repeated, expected)
        self.assertEqual(downloads, [(2,), (2,)])

    def test_target_snapshot_and_dynamics_isolation(self):
        """Keep mutated future targets out of dynamics and frozen scoring snapshots."""
        trial = self.trial()
        models = _models()
        evaluator = GpuEvaluator([trial], candidates=2, config=self.config)
        expected = evaluator.evaluate(models)
        group = evaluator._batches[0]._groups[0]
        before = group.collect()
        trial.q[:] += 5
        trial.grf_n[:] += 1000
        self.assertEqual(evaluator.evaluate(models), expected)
        changed = GpuEvaluator([trial], candidates=2, config=self.config)
        scores = changed.evaluate(models)
        after = changed._batches[0]._groups[0].collect()
        self.assertGreater(scores[0]["mean_loss"], expected[0]["mean_loss"])
        for c in range(2):
            for field in before[c][0][0]:
                np.testing.assert_array_equal(before[c][0][0][field], after[c][0][0][field])
        self.assertFalse(hasattr(group, "targets"))
        self.assertFalse(hasattr(group.data, "targets"))

    def test_controlled_trace_boundary_tolerance_stale_tail_and_effort(self):
        """Match interpolation edges, negative peaks, threshold equality and candidate-specific effort."""
        trials = [self.trial(str(i), 0.0004) for i in range(3)]
        for trial in trials:
            trial.time_s = np.array([0, 1e-13, 0.000075, 0.0002, 0.0002 + 5e-13, 0.0002 + 2e-12, 0.0004])
            trial.q = np.tile(trial.initial.q, (7, 1))
            trial.q[0] = 1e6
        evaluator = GpuEvaluator(trials, candidates=2, config=self.config)
        group = evaluator._batches[0]._groups[0]
        models = _models()
        group.models.assign([_model_params(model, group.dt) for model in models])
        counts = np.array([0, 2, 4, 0, 2, 4], dtype=np.int32)
        codes = np.array([2, 5, 1, 2, 5, 1], dtype=np.int32)
        states = np.full((5, 6, 6), np.nan)
        outputs = np.zeros((4, 6), dtype=np.dtype(group.data.outputs.dtype.numpy_dtype()))
        for name in ("load", "grf_n"):
            outputs[name][:] = np.nan
        expected = []
        for w, count in enumerate(counts):
            states[: count + 1, w] = trials[w % 3].initial.q + np.arange(count + 1)[:, None] * 0.001
            # A completed world has contact exactly at and just above threshold;
            # a failed world's accepted peak is negative (not clamped to zero).
            force = np.array([[1, -1e-7], [-3, -2e-7], [5, 0.01], [2, 0.010001]])[:count]
            outputs["grf_n"][:count, w] = force
            outputs["load"][:count, w] = np.tile([0, 0, 0, 2.0, -3.0, 1.0], (count, 1))
            trace = {
                "time_s": np.arange(count + 1) * group.dt,
                "state": states[: count + 1, w],
                "grf_n": force,
                "load": outputs["load"][:count, w],
            }
            summary = {
                "status": "completed" if codes[w] == 1 else "failed",
                "peak_grf_n": force.max(0) if count else [0, 0],
                "grf_impulse_ns": force.sum(0) * group.dt,
                "contact_threshold_n": self.config.contact_threshold_n,
                "contact_duration_s": np.count_nonzero(force[:, 1] > self.config.contact_threshold_n) * group.dt,
            }
            expected.append(score(trace, summary, trials[w % 3], models[w // 3])["loss"])
        group.data.states.assign(states)
        group.data.outputs.assign(outputs)
        group.data.recorded.assign(counts)
        group.data.status.assign(codes)
        with patch.object(group, "launch"):
            result = evaluator.evaluate(models)
        np.testing.assert_allclose(evaluator._scorers[0][2].numpy(), expected, rtol=1e-13, atol=1e-10)
        for c, row in enumerate(result):
            self.assertEqual(row["failed"], 2)
            self.assertAlmostEqual(row["mean_loss"], np.mean(expected[c * 3 : (c + 1) * 3]), delta=1e-10)
        self.assertGreater(result[1]["mean_loss"], result[0]["mean_loss"])

    def test_memory_budget_plans_full_dataset_before_allocating(self):
        """Stream oversized datasets but reject any oversized trial before allocation."""
        trials = [self.trial(str(i), 0.3) for i in range(100)]
        with (
            patch.object(GpuBatch, "__init__", side_effect=AssertionError("Allocated before budget check")),
            patch.object(wp, "zeros", side_effect=AssertionError("Allocated before budget check")),
        ):
            evaluator = GpuEvaluator(trials, candidates=8, config=self.config)
            self.assertTrue(evaluator.streaming)
            self.assertGreater(evaluator.estimated_static_bytes, evaluator.memory_budget_bytes)
            self.assertLessEqual(evaluator.estimated_resident_bytes, evaluator.memory_budget_bytes)
            self.assertEqual([len(chunk) for chunk in evaluator._streaming_chunks], [8] * 12 + [4])
            oversized = self.trial("oversized", duration=30)
            with self.assertRaisesRegex(ValueError, "Single-trial.*oversized.*estimate.*budget"):
                GpuEvaluator([*trials, oversized], candidates=8, config=self.config)
            huge = self.trial(duration=1e8)
            with self.assertRaisesRegex(ValueError, "clock range"):
                GpuEvaluator([huge], candidates=1, config=self.config)
        evaluator = GpuEvaluator(trials[:1], candidates=1, config=self.config)
        self.assertFalse(evaluator.streaming)
        self.assertLessEqual(evaluator.estimated_resident_bytes, evaluator.memory_budget_bytes)
        self.assertEqual(evaluator.memory_budget_bytes, 512 * 1024**2)

    def test_streaming_adaptive_chunks_static_and_cpu_parity(self):
        """Match static and CPU losses with uneven adaptive chunks and mixed failures."""
        trials = [self.trial(str(i), 0.001) for i in range(5)]
        trials[-1].initial.q[1] = 0.1
        models = _models()
        budget = gpu_objective._estimate_bytes([trials[:2]], 2, self.config)
        static = GpuEvaluator(trials, candidates=2, config=self.config)
        expected, _ = self.assertScores(static, trials, models, cpu=True)
        cpu = [identify.evaluate(model, trials, self.config) for model in models]
        with patch.object(gpu_objective, "_MEMORY_BUDGET_BYTES", budget):
            evaluator = GpuEvaluator(trials, candidates=2, config=self.config)
            # An exact-budget request keeps the reusable resident path.
            exact = GpuEvaluator(trials[:2], candidates=2, config=self.config)
            self.assertFalse(exact.streaming)
            self.assertEqual(exact.estimated_resident_bytes, budget)
        self.assertTrue(evaluator.streaming)
        self.assertEqual([len(chunk) for chunk in evaluator._streaming_chunks], [2, 1, 2])
        self.assertEqual(evaluator._batches, [])
        self.assertEqual(evaluator._scorers, [])
        self.assertIsNone(evaluator._aggregates)
        for chunk in evaluator._streaming_chunks:
            self.assertLessEqual(gpu_objective._estimate_bytes([chunk], 2, self.config), budget)
        self.assertEqual(evaluator.estimated_resident_bytes, budget)
        original_init = GpuBatch.__init__
        sizes = []

        def checked_init(batch, *args, **kwargs):
            sizes.append(len(args[0]))
            self.assertLessEqual(sizes[-1], 2)
            original_init(batch, *args, **kwargs)

        # Changing the global default must not change the admitted subchunk cap.
        with (
            patch.object(gpu_objective, "_MEMORY_BUDGET_BYTES", 1),
            patch.object(GpuBatch, "__init__", checked_init),
        ):
            result = evaluator.evaluate(models)
        self.assertEqual(sizes, [2, 1, 2])
        for actual, resident, reference in zip(result, expected, cpu, strict=True):
            self.assertEqual(set(actual), {"mean_loss", "failed"})
            self.assertEqual(actual["failed"], 1)
            self.assertEqual(actual["failed"], reference["failed"])
            self.assertAlmostEqual(actual["mean_loss"], resident["mean_loss"], delta=1e-9)
            self.assertAlmostEqual(actual["mean_loss"], reference["mean_loss"], delta=1e-4)

    def test_streaming_snapshot_transfers_and_chunk_lifetimes(self):
        """Freeze targets and release chunks without reading traces or retaining graphs."""
        trials = [self.trial(str(i), t) for i, t in enumerate([0.0004, 0.00051, 0.0007])]
        models = _models()
        expected = GpuEvaluator(trials, candidates=2, config=self.config).evaluate(models)
        budget = max(gpu_objective._estimate_bytes([[trial]], 2, self.config) for trial in trials)
        with patch.object(gpu_objective, "_MEMORY_BUDGET_BYTES", budget):
            evaluator = GpuEvaluator(trials, candidates=2, config=self.config, max_trials_per_batch=2)
        self.assertTrue(evaluator.streaming)
        for trial in trials:
            trial.q[:] += 5
            trial.grf_n[:] += 1000
            trial.time_s[:] *= 2
            trial.force_time_s[:] *= 2
            trial.initial.q[1] = 0.1
        original_init, original_numpy, original_launch = GpuBatch.__init__, wp.array.numpy, _Group.launch
        refs, downloads = [], []
        constructing = False

        def checked_init(batch, *args, **kwargs):
            nonlocal constructing
            self.assertTrue(all(ref() is None for ref in refs), "Previous chunk still owns GPU resources")
            constructing = True
            try:
                original_init(batch, *args, **kwargs)
            finally:
                constructing = False
            refs.append(weakref.ref(batch))
            for group in batch._groups:
                refs.extend(weakref.ref(obj) for obj in (group, group.foundation, group.data.states))
                if group.foundation.friction_solver is not None:
                    refs.append(weakref.ref(group.foundation.friction_solver))

        def checked_launch(group, candidate_models):
            original_launch(group, candidate_models)
            refs.append(weakref.ref(group.graph))

        def guarded_numpy(array):
            # Reconstruction may read fixed shoe geometry/config, never rollout traces.
            if not constructing:
                self.assertIs(array.dtype, gpu_objective._Aggregate)
                self.assertEqual(array.shape, (2,))
                downloads.append(array.shape)
                refs.append(weakref.ref(array))
            return original_numpy(array)

        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with (
                patch.object(GpuBatch, "__init__", checked_init),
                patch.object(_Group, "launch", checked_launch),
                patch.object(wp.array, "numpy", guarded_numpy),
                patch.object(_Group, "collect", side_effect=AssertionError("Trajectory collection")),
                patch.object(identify, "score", side_effect=AssertionError("CPU scoring")),
                patch.object(identify, "evaluate", side_effect=AssertionError("CPU evaluation")),
                patch.object(identify, "predict", side_effect=AssertionError("CPU prediction")),
            ):
                first = evaluator.evaluate(models)
                reverse = evaluator.evaluate(models[::-1])
                repeated = evaluator.evaluate(models)
            self.assertTrue(all(ref() is None for ref in refs))
        finally:
            if was_enabled:
                gc.enable()
        self.assertEqual(first, repeated)
        self.assertEqual(reverse, first[::-1])
        self.assertEqual(downloads, [(2,)] * 9)
        for actual, resident in zip(first, expected, strict=True):
            self.assertEqual(actual["failed"], resident["failed"])
            self.assertAlmostEqual(actual["mean_loss"], resident["mean_loss"], delta=1e-9)
        changed = GpuEvaluator(trials, candidates=2, config=self.config).evaluate(models)
        self.assertNotEqual(changed, first)

    def test_streaming_heterogeneous_groups_and_trial_order(self):
        """Preserve heterogeneous shoe/dt groups and trial weights across chunk orders."""
        trials = [self.trial(str(i), t) for i, t in enumerate([0.0004, 0.00051, 0.0007, 0.0004, 0.00051])]
        with tempfile.TemporaryDirectory() as directory:
            trials[2].shoe = _tiny_shoe(directory)
            models = _models()
            expected = [identify.evaluate(model, trials, self.config) for model in models]
            static = GpuEvaluator(trials, candidates=2, config=self.config).evaluate(models)
            # Admit two different groups, but not all three initial chunks.
            budget = 2 * max(gpu_objective._estimate_bytes([[trial]], 2, self.config) for trial in trials)
            for ordered in (trials, trials[::-1]):
                evaluator = GpuEvaluator(
                    ordered, candidates=2, config=self.config, max_trials_per_batch=2, memory_budget_bytes=budget
                )
                self.assertTrue(evaluator.streaming)
                self.assertEqual([len(chunk) for chunk in evaluator._streaming_chunks], [2, 2, 1])
                for chunk in evaluator._streaming_chunks:
                    self.assertLessEqual(gpu_objective._estimate_bytes([chunk], 2, self.config), budget)
                for actual, resident, reference in zip(evaluator.evaluate(models), static, expected, strict=True):
                    self.assertEqual(actual["failed"], reference["failed"])
                    self.assertAlmostEqual(actual["mean_loss"], resident["mean_loss"], delta=1e-9)
                    self.assertAlmostEqual(actual["mean_loss"], reference["mean_loss"], delta=1e-4)

    def test_streaming_error_cleanup_and_retry(self):
        """Release a launched chunk after scoring errors and allow a clean retry."""
        trials = [self.trial(str(i), 0.0004) for i in range(3)]
        models = _models()
        budget = gpu_objective._estimate_bytes([trials[:1]], 2, self.config)
        evaluator = GpuEvaluator(trials, candidates=2, config=self.config, memory_budget_bytes=budget)
        expected = evaluator.evaluate(models)
        original_launch, original_numpy = _Group.launch, wp.array.numpy
        refs = []

        def checked_launch(group, candidate_models):
            original_launch(group, candidate_models)
            refs.extend(weakref.ref(obj) for obj in (group, group.graph, group.foundation, group.data.states))
            refs.append(weakref.ref(group.foundation.friction_solver))

        def interrupted_numpy(array):
            if array.dtype is gpu_objective._Aggregate:
                raise RuntimeError("Interrupted aggregate download")
            return original_numpy(array)

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with (
                patch.object(_Group, "launch", checked_launch),
                patch.object(wp.array, "numpy", interrupted_numpy),
                self.assertRaisesRegex(RuntimeError, "Interrupted aggregate download"),
            ):
                evaluator.evaluate(models)
            self.assertTrue(refs)
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertEqual(evaluator._batches, [])
            self.assertIsNone(evaluator._aggregates)
        finally:
            if was_enabled:
                gc.enable()
        self.assertEqual(evaluator.evaluate(models), expected)
        with patch.object(GpuBatch, "__init__", side_effect=AssertionError("Allocated before model validation")):
            with self.assertRaisesRegex(ValueError, "exactly 2"):
                evaluator.evaluate(models[:1])
            with self.assertRaises(TypeError):
                evaluator.evaluate([None, None])

    def test_validation(self):
        """Reject empty trials, invalid sizes, CPU devices and incompatible models."""
        trial = self.trial()
        with self.assertRaisesRegex(ValueError, "at least one trial"):
            GpuEvaluator([], candidates=1)
        for kwargs in (
            {"candidates": 0},
            {"candidates": True},
            {"candidates": 1.5},
            {"max_trials_per_batch": 0},
            {"max_trials_per_batch": True},
            {"max_trials_per_batch": 1.5},
            {"memory_budget_bytes": 0},
            {"memory_budget_bytes": -1},
            {"memory_budget_bytes": True},
            {"memory_budget_bytes": 1.5},
            {"memory_budget_bytes": float("inf")},
            {"device": "cpu"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GpuEvaluator([trial], **({"candidates": 2, **kwargs}))
        evaluator = GpuEvaluator(
            [trial], candidates=np.int64(2), config=self.config, memory_budget_bytes=np.int64(512 * 1024**2)
        )
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            evaluator.evaluate(_models()[:1])
        with self.assertRaises(TypeError):
            evaluator.evaluate([None, None])
        trial.initial.torque_nm[0] = 10
        evaluator = GpuEvaluator([trial], candidates=2, config=self.config)
        with self.assertRaisesRegex(ValueError, "Initial torque"):
            evaluator.evaluate(_models())


if __name__ == "__main__":
    unittest.main(verbosity=2)
