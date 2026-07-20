# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

BENCHMARK_DIR = Path(__file__).parents[2] / "asv" / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

_WARP_CONFIG_FIELDS = ("enable_backward", "log_level")
_WARP_CONFIG_BEFORE_BENCHMARK_IMPORTS = {name: getattr(wp.config, name) for name in _WARP_CONFIG_FIELDS}

try:
    from benchmark_metrics import SimulationMetrics
    from simulation import bench_anymal, bench_kamino, bench_mujoco, bench_quadruped_xpbd
finally:
    for _name, _value in _WARP_CONFIG_BEFORE_BENCHMARK_IMPORTS.items():
        setattr(wp.config, _name, _value)


class TestSimulationBenchmarks(unittest.TestCase):
    class _FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def numpy(self):
            return self.values

    def _make_anymal_workload(self, root_y, root_z):
        state = SimpleNamespace(
            joint_q=self._FakeArray([0.0, root_y, root_z, 0.0, 0.0, 0.0, 1.0]),
            joint_qd=self._FakeArray([0.0] * 6),
            body_q=self._FakeArray([[0.0, root_y, root_z, 0.0, 0.0, 0.0, 1.0]]),
            body_qd=self._FakeArray([[0.0] * 6]),
        )
        return SimpleNamespace(state_0=state)

    def test_benchmark_imports_preserve_warp_config(self):
        self.assertEqual(
            {name: getattr(wp.config, name) for name in _WARP_CONFIG_FIELDS},
            _WARP_CONFIG_BEFORE_BENCHMARK_IMPORTS,
        )

    def test_fast_kitchen_g1_validates_kitchen_body_count(self):
        benchmark = bench_mujoco.FastKitchenG1()
        world_count = benchmark.params[0][0]
        kitchen_workload = SimpleNamespace(
            model=SimpleNamespace(body_count=benchmark.expected_bodies_per_world * world_count),
            test_final=Mock(),
        )
        benchmark._validate_workload(kitchen_workload, world_count)
        kitchen_workload.test_final.assert_called_once_with()

        incomplete_kitchen_workload = SimpleNamespace(
            model=SimpleNamespace(body_count=(benchmark.expected_bodies_per_world - 1) * world_count),
            test_final=Mock(),
        )
        with self.assertRaisesRegex(RuntimeError, "bodies per world for kitchen"):
            benchmark._validate_workload(incomplete_kitchen_workload, world_count)

    def test_mujoco_step_falls_back_when_cuda_graph_is_unavailable(self):
        example = bench_mujoco.Example.__new__(bench_mujoco.Example)
        example.actuation = "None"
        example.use_cuda_graph = True
        example.graph = None
        example.simulate = Mock()
        example.benchmark_time = 0.0
        example.sim_time = 0.0
        example.frame_dt = 0.01

        with (
            patch("benchmark_mujoco.time.perf_counter", side_effect=(1.0, 1.25)),
            patch.object(bench_mujoco.wp, "synchronize_device"),
            patch.object(bench_mujoco.wp, "capture_launch") as capture_launch,
        ):
            example.step()

        example.simulate.assert_called_once_with()
        capture_launch.assert_not_called()
        self.assertEqual(example.benchmark_time, 0.25)
        self.assertEqual(example.sim_time, 0.01)

    def test_mujoco_kpi_requires_cuda_graph(self):
        benchmark = bench_mujoco.FastCartpole()
        with (
            patch.object(bench_mujoco, "Example", return_value=SimpleNamespace(graph=None)),
            self.assertRaisesRegex(RuntimeError, "requires CUDA graph capture"),
        ):
            benchmark._create_workload(Mock(), world_count=1)

    def test_mujoco_metrics_include_solver_iterations(self):
        benchmark = bench_mujoco.FastCartpole()
        workloads = []

        class FakeArray:
            def __init__(self, values):
                self.values = np.asarray(values)

            def numpy(self):
                return self.values

        def collect_metrics(**kwargs):
            for values in ([2, 4], [1, 5]):
                workload = SimpleNamespace(
                    solver=SimpleNamespace(mjw_data=SimpleNamespace(solver_niter=FakeArray(values))),
                    test_final=Mock(),
                )
                workloads.append(workload)
                kwargs["validate"](workload)
            return SimulationMetrics(1.0, 2.0, 3.0, 4.0, 5.0, 0.01, 2)

        with (
            patch.object(bench_mujoco.wp, "get_cuda_device_count", return_value=1),
            patch.object(bench_mujoco.Example, "create_model_builder", return_value=Mock()),
            patch.object(bench_mujoco, "collect_simulation_metrics", side_effect=collect_metrics),
        ):
            metrics = benchmark._collect_metrics()[8192]

        self.assertTrue(all(workload.test_final.call_count == 1 for workload in workloads))
        self.assertEqual(metrics.solver_niter_mean, 3.0)
        self.assertEqual(metrics.solver_niter_max, 5.0)

    def test_metric_setup_caches_skip_without_cuda(self):
        with (
            patch.object(bench_mujoco.wp, "get_cuda_device_count", return_value=0),
            patch.object(bench_mujoco.Example, "create_model_builder") as create_mujoco_builder,
            patch.object(bench_kamino.DRLegsBenchmarkWorkload, "create_model_builder") as create_kamino_builder,
            patch.object(bench_anymal, "_create_example") as create_anymal,
            patch.object(bench_quadruped_xpbd, "_create_example") as create_quadruped,
        ):
            self.assertIsNone(bench_mujoco.FastCartpole().setup_cache())
            self.assertIsNone(bench_kamino.KpiDRLegs().setup_cache())
            self.assertIsNone(bench_anymal.FastMetricsExampleAnymalPretrained().setup_cache())
            self.assertIsNone(bench_quadruped_xpbd.FastMetricsExampleQuadrupedXPBD().setup_cache())

        create_mujoco_builder.assert_not_called()
        create_kamino_builder.assert_not_called()
        create_anymal.assert_not_called()
        create_quadruped.assert_not_called()

    def test_kpi_dr_legs_setup_cache_timeout_exceeds_default(self):
        config = json.loads((BENCHMARK_DIR.parents[1] / "asv.conf.json").read_text(encoding="utf-8"))
        self.assertGreater(bench_kamino.KpiDRLegs.setup_cache.timeout, config["default_benchmark_timeout"])

    def test_anymal_short_horizon_validation(self):
        bench_anymal._validate_workload(self._make_anymal_workload(root_y=0.719, root_z=0.530))

        with self.assertRaisesRegex(RuntimeError, "forward progress"):
            bench_anymal._validate_workload(self._make_anymal_workload(root_y=0.0, root_z=0.530))

        with self.assertRaisesRegex(RuntimeError, "base height"):
            bench_anymal._validate_workload(self._make_anymal_workload(root_y=0.719, root_z=0.200))


if __name__ == "__main__":
    unittest.main()
