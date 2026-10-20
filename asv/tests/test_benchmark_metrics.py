# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np

BENCHMARK_DIR = Path(__file__).parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_metrics import (  # noqa: E402
    collect_simulation_metrics,
    collect_startup_metrics,
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

    def test_collect_startup_metrics(self):
        """Average phase and total times, freeing each workload before the next one."""
        clock = [0.0]
        workloads = []
        events = []

        class FakeWorkload:
            def __init__(self, startup_phase_times, scale):
                self.cycle = self  # Only garbage collection frees a reference cycle.
                startup_phase_times.update(
                    model=0.4 * scale, replication=0.1 * scale, finalize=0.2 * scale, solver=0.1 * scale
                )
                clock[0] += 0.5 * scale

            def step(self):
                events.append("step")
                clock[0] += 0.1

        def create_workload(startup_phase_times):
            self.assertTrue(all(workload() is None for workload in workloads), "previous workload is still alive")
            workload = FakeWorkload(startup_phase_times, len(workloads) + 1)
            workloads.append(weakref.ref(workload))
            return workload

        with patch("benchmark_metrics.wp.synchronize_device", side_effect=lambda: events.append("synchronize")):
            metrics = collect_startup_metrics(create_workload, samples=2, timer=lambda: clock[0])

        self.assertEqual(events, ["step", "synchronize"] * 2)
        self.assertAlmostEqual(metrics.model_time, 0.6)
        self.assertAlmostEqual(metrics.replication_time, 0.15)
        self.assertAlmostEqual(metrics.finalize_time, 0.3)
        self.assertAlmostEqual(metrics.solver_time, 0.15)
        self.assertAlmostEqual(metrics.total_time, 0.85)

    def test_collect_startup_metrics_rejects_missing_phases(self):
        """Reject incomplete instrumentation in any startup sample."""
        for missing_phase in ("model", "replication", "finalize", "solver"):
            for incomplete_sample in (0, 1):
                with self.subTest(phase=missing_phase, sample=incomplete_sample):
                    sample_index = 0

                    def create_workload(phase_times, missing_phase=missing_phase, incomplete_sample=incomplete_sample):
                        nonlocal sample_index
                        phase_times.update(model=0.4, replication=0.1, finalize=0.2, solver=0.1)
                        if sample_index == incomplete_sample:
                            del phase_times[missing_phase]
                        sample_index += 1
                        return type("Workload", (), {"step": lambda self: None})()

                    with (
                        patch("benchmark_metrics.wp.synchronize_device"),
                        self.assertRaisesRegex(RuntimeError, f"Missing startup phases:.*{missing_phase}"),
                    ):
                        collect_startup_metrics(create_workload, samples=2)

    def test_collect_startup_metrics_rejects_nonpositive_samples(self):
        """Reject an empty sample set before constructing a workload."""
        for samples in (0, -1):
            with self.subTest(samples=samples), self.assertRaisesRegex(ValueError, "samples must be positive"):
                collect_startup_metrics(lambda phases: self.fail("unexpected workload construction"), samples=samples)

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


if __name__ == "__main__":
    unittest.main()
