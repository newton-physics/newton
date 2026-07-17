# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np

from newton.utils import run_benchmark

BENCHMARK_DIR = Path(__file__).parents[2] / "asv" / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_metrics import (  # noqa: E402
    collect_simulation_metrics,
    collect_simulation_metrics_synchronized,
    compute_gpu_memory_usage,
    compute_simulation_metrics,
    validate_simulation_state,
    validate_simulation_workload,
)


class TestBenchmarkMetrics(unittest.TestCase):
    def test_compute_simulation_metrics(self):
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
        workloads = []
        events = []
        timer_values = iter((0.0, 0.02, 0.02, 0.06, 0.06, 0.08, 0.08, 0.12))

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

        def memory_usage_bytes(workload):
            self.assertIs(workload, workloads[0])
            self.assertEqual(workload.step_count, 2)
            events.append(("memory", workload))
            return 8 * 1024**2

        def validate(workload):
            events.append(("validate", workload))

        metrics = collect_simulation_metrics(
            create_workload=create_workload,
            world_count=4,
            num_frames=2,
            samples=2,
            memory_usage_bytes=memory_usage_bytes,
            validate=validate,
            timer=lambda: next(timer_values),
        )

        self.assertEqual(len(workloads), 2)
        self.assertEqual(events, [("memory", workloads[0]), ("validate", workloads[0]), ("validate", workloads[1])])
        self.assertAlmostEqual(metrics.mean_world_step_time_ms, 1.875)
        self.assertAlmostEqual(metrics.world_steps_per_second, 32 / 0.12)
        self.assertAlmostEqual(metrics.real_time_factor, 32 * 0.01 / 0.12)
        self.assertAlmostEqual(metrics.p95_frame_time_ms, 40.0)
        self.assertAlmostEqual(metrics.gpu_memory_mib, 8.0)

    def test_collect_simulation_metrics_synchronized(self):
        workloads = []
        events = []
        sync_calls = []
        timer_values = iter((0.0, 0.01, 0.01, 0.03))

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

        def memory_usage_bytes(workload):
            events.append(("memory", workload))
            return 8 * 1024**2

        def validate(workload):
            events.append(("validate", workload))

        metrics = collect_simulation_metrics_synchronized(
            create_workload=create_workload,
            world_count=4,
            num_frames=2,
            samples=1,
            synchronize=lambda: sync_calls.append(None),
            timer=lambda: next(timer_values),
            memory_usage_bytes=memory_usage_bytes,
            validate=validate,
        )

        self.assertEqual(len(sync_calls), 3)
        self.assertEqual(events, [("memory", workloads[0]), ("validate", workloads[0])])
        self.assertAlmostEqual(metrics.mean_world_step_time_ms, 1.875)
        self.assertAlmostEqual(metrics.world_steps_per_second, 16 / 0.03)
        self.assertAlmostEqual(metrics.real_time_factor, 16 * 0.01 / 0.03)

    def test_compute_gpu_memory_usage(self):
        class FakeDevice:
            free_memory = 700

        self.assertEqual(compute_gpu_memory_usage(FakeDevice(), free_memory_before=1000), 300)

        FakeDevice.free_memory = 1100
        with self.assertRaisesRegex(RuntimeError, "increased"):
            compute_gpu_memory_usage(FakeDevice(), free_memory_before=1000)

    def test_validate_simulation_state(self):
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

        specialized_checks = []
        FakeState.body_q = FakeArray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        FakeState.joint_qd = FakeArray([np.nan])
        with self.assertRaisesRegex(RuntimeError, "state.joint_qd"):
            validate_simulation_state(FakeState(), max_linear_speed=10.0, max_angular_speed=10.0)

        FakeState.joint_qd = FakeArray([0.0])

        class FakeWorkload:
            state_0 = FakeState()

            def test_final(self):
                specialized_checks.append(self)

        workload = FakeWorkload()
        validate_simulation_workload(workload, max_linear_speed=10.0, max_angular_speed=10.0)
        self.assertEqual(specialized_checks, [workload])

    def test_run_benchmark_with_setup_cache(self):
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
