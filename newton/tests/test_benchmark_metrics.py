# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from newton.utils import run_benchmark

BENCHMARK_DIR = Path(__file__).parents[2] / "asv" / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_metrics import (  # noqa: E402
    collect_simulation_metrics,
    compute_simulation_metrics,
    validate_simulation_state,
)


class TestBenchmarkMetrics(unittest.TestCase):
    def test_compute_simulation_metrics(self):
        """Verify derived simulation metrics and their units."""
        metrics = compute_simulation_metrics(
            frame_times=[0.1, 0.2, 0.3, 0.4],
            sim_dt=0.002,
            sim_substeps=5,
            world_count=10,
            gpu_memory_bytes=10 * 1024**2,
        )

        self.assertAlmostEqual(metrics.mean_world_step_time_ms, 5.0)
        self.assertAlmostEqual(metrics.world_steps_per_second, 200.0)
        self.assertAlmostEqual(metrics.real_time_factor, 0.4)
        self.assertAlmostEqual(metrics.p95_frame_time_ms, 385.0)
        self.assertAlmostEqual(metrics.gpu_memory_mib, 10.0)
        self.assertEqual(metrics.sim_dt, 0.002)
        self.assertEqual(metrics.sim_substeps, 5)

    def test_collect_simulation_metrics(self):
        """Verify internal timing, validation, and memory collection."""
        workloads = []
        events = []
        timer_values = iter((0.0, 0.02, 0.02, 0.06, 0.06, 0.08, 0.08, 0.12))

        class FakeDevice:
            free_memory_values = iter((20 * 1024**2, 12 * 1024**2))

            @property
            def free_memory(self):
                if workloads:
                    self_test.assertEqual(workloads[0].step_count, 2)
                return next(self.free_memory_values)

        self_test = self

        class FakeWorkload:
            sim_dt = 0.01
            sim_substeps = 2

            def __init__(self):
                self.benchmark_time = 0.0
                self.step_count = 0

            def step(self):
                self.benchmark_time += (0.01, 0.02)[self.step_count]
                self.step_count += 1

        def create_workload():
            workload = FakeWorkload()
            workloads.append(workload)
            return workload

        def validate(workload):
            events.append(("validate", workload))

        with (
            patch("benchmark_metrics.wp.get_device", return_value=FakeDevice()),
            patch("benchmark_metrics.wp.synchronize_device") as synchronize_device,
        ):
            metrics = collect_simulation_metrics(
                create_workload=create_workload,
                world_count=4,
                num_frames=2,
                samples=2,
                validate=validate,
                timer=lambda: next(timer_values),
            )

        self.assertEqual(len(workloads), 2)
        self.assertEqual(events, [("validate", workloads[0]), ("validate", workloads[1])])
        self.assertEqual(synchronize_device.call_count, 2)
        self.assertAlmostEqual(metrics.mean_world_step_time_ms, 1.875)
        self.assertAlmostEqual(metrics.world_steps_per_second, 32 / 0.12)
        self.assertAlmostEqual(metrics.real_time_factor, 32 * 0.01 / 0.12)
        self.assertAlmostEqual(metrics.p95_frame_time_ms, 40.0)
        self.assertAlmostEqual(metrics.gpu_memory_mib, 8.0)

    def test_collect_simulation_metrics_with_synchronization(self):
        """Verify synchronized wall timing drives collected metrics."""
        workloads = []
        events = []
        sync_calls = []
        timer_values = iter((0.0, 0.01, 0.01, 0.03))

        class FakeDevice:
            free_memory_values = iter((16 * 1024**2, 8 * 1024**2))

            @property
            def free_memory(self):
                return next(self.free_memory_values)

        class FakeWorkload:
            sim_dt = 0.01
            sim_substeps = 2

            def __init__(self):
                self.step_count = 0

            def step(self):
                self.step_count += 1

        def create_workload():
            workload = FakeWorkload()
            workloads.append(workload)
            return workload

        def validate(workload):
            events.append(("validate", workload))

        with (
            patch("benchmark_metrics.wp.get_device", return_value=FakeDevice()),
            patch("benchmark_metrics.wp.synchronize_device") as synchronize_device,
        ):
            metrics = collect_simulation_metrics(
                create_workload=create_workload,
                world_count=4,
                num_frames=2,
                samples=1,
                synchronize=lambda: sync_calls.append(None),
                timer=lambda: next(timer_values),
                validate=validate,
            )

        self.assertEqual(len(sync_calls), 3)
        self.assertEqual(events, [("validate", workloads[0])])
        self.assertEqual(synchronize_device.call_count, 2)
        self.assertAlmostEqual(metrics.mean_world_step_time_ms, 1.875)
        self.assertAlmostEqual(metrics.world_steps_per_second, 16 / 0.03)
        self.assertAlmostEqual(metrics.real_time_factor, 16 * 0.01 / 0.03)
        self.assertAlmostEqual(metrics.gpu_memory_mib, 8.0)

    def test_collect_simulation_metrics_rejects_increased_free_memory(self):
        """Reject an invalid increase in measured free GPU memory."""

        class FakeDevice:
            free_memory_values = iter((1000, 1100))

            @property
            def free_memory(self):
                return next(self.free_memory_values)

        class FakeWorkload:
            sim_dt = 0.01
            sim_substeps = 1
            benchmark_time = 0.0

            def step(self):
                self.benchmark_time += 0.01

        with (
            patch("benchmark_metrics.wp.get_device", return_value=FakeDevice()),
            patch("benchmark_metrics.wp.synchronize_device"),
            self.assertRaisesRegex(RuntimeError, "increased"),
        ):
            collect_simulation_metrics(
                create_workload=FakeWorkload,
                world_count=1,
                num_frames=1,
                samples=1,
                timer=iter((0.0, 0.01)).__next__,
            )

    def test_validate_simulation_state(self):
        """Validate finite states, unit quaternions, and bounded speeds."""

        class FakeArray:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)

            def numpy(self):
                return self.values

        class FakeState:
            joint_q = FakeArray([0.0])
            joint_qd = FakeArray([0.0])
            body_q = FakeArray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
            body_qd = FakeArray([[1.0, 0.0, 0.0, 0.0, 0.0, 2.0]])

        validate_simulation_state(FakeState(), max_linear_speed=10.0, max_angular_speed=10.0)

        FakeState.body_qd = FakeArray([[11.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
        with self.assertRaisesRegex(RuntimeError, "linear speed"):
            validate_simulation_state(FakeState(), max_linear_speed=10.0, max_angular_speed=10.0)

        FakeState.body_qd = FakeArray([[1.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
        FakeState.body_q = FakeArray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
        with self.assertRaisesRegex(RuntimeError, "quaternion"):
            validate_simulation_state(FakeState(), max_linear_speed=10.0, max_angular_speed=10.0)

        FakeState.body_q = FakeArray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        FakeState.joint_qd = FakeArray([np.nan])
        with self.assertRaisesRegex(RuntimeError, "state.joint_qd"):
            validate_simulation_state(FakeState(), max_linear_speed=10.0, max_angular_speed=10.0)

    def test_run_benchmark_with_setup_cache(self):
        """Pass one setup cache through the full benchmark lifecycle."""
        cache_events = []

        class CachedBenchmark:
            params: ClassVar = [[2, 3]]
            setup_cache_calls = 0
            cache_value: ClassVar = {"base": 10}

            def setup_cache(self):
                type(self).setup_cache_calls += 1
                return self.cache_value

            def setup(self, cache, value):
                cache_events.append(("setup", cache, value))

            def time_value(self, cache, value):
                cache_events.append(("time", cache, value))

            def track_value(self, cache, value):
                cache_events.append(("track", cache, value))
                return cache["base"] + value

            def teardown(self, cache, value):
                cache_events.append(("teardown", cache, value))

        results = run_benchmark(CachedBenchmark, print_results=False)

        self.assertEqual(CachedBenchmark.setup_cache_calls, 1)
        self.assertTrue(all(cache is CachedBenchmark.cache_value for _, cache, _ in cache_events))
        self.assertEqual([event for event, _, _ in cache_events].count("setup"), 2)
        self.assertEqual([event for event, _, _ in cache_events].count("time"), 4)
        self.assertEqual([event for event, _, _ in cache_events].count("track"), 2)
        self.assertEqual([event for event, _, _ in cache_events].count("teardown"), 2)
        self.assertEqual({value for _, _, value in cache_events}, {2, 3})
        self.assertEqual(results[("track_value", (2,))], 12)
        self.assertEqual(results[("track_value", (3,))], 13)


if __name__ == "__main__":
    unittest.main()
