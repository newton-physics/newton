# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import numpy as np
import warp as wp

BENCHMARK_DIR = Path(__file__).parents[2] / "asv" / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

_WARP_CONFIG_FIELDS = ("enable_backward", "log_level")
_WARP_CONFIG_BEFORE_BENCHMARK_IMPORTS = {name: getattr(wp.config, name) for name in _WARP_CONFIG_FIELDS}

try:
    from simulation import bench_anymal, bench_kamino, bench_mujoco
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

    def test_fast_kitchen_g1_builds_kitchen_environment(self):
        class FakeDevice:
            free_memory = 1024

        with (
            patch.object(bench_mujoco.wp, "synchronize_device"),
            patch.object(bench_mujoco.wp, "get_device", return_value=FakeDevice()),
            patch.object(
                bench_mujoco.Example,
                "create_model_builder",
                return_value=sentinel.builder,
            ) as create_model_builder,
            patch.object(
                bench_mujoco,
                "collect_simulation_metrics",
                return_value=sentinel.metrics,
            ),
        ):
            metrics = bench_mujoco.FastKitchenG1()._collect_metrics()

        create_model_builder.assert_called_once_with(
            "g1",
            128,
            environment="kitchen",
            randomize=True,
            seed=123,
        )
        self.assertEqual(metrics, {128: sentinel.metrics})
        self.assertEqual(bench_mujoco.FastKitchenG1.version, "2")

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

    def test_kpi_dr_legs_has_setup_cache_timeout(self):
        self.assertEqual(bench_kamino.KpiDRLegs.setup_cache.timeout, 1200)

    def test_anymal_short_horizon_validation(self):
        bench_anymal._validate_workload(self._make_anymal_workload(root_y=0.719, root_z=0.530))

        with self.assertRaisesRegex(RuntimeError, "forward progress"):
            bench_anymal._validate_workload(self._make_anymal_workload(root_y=0.0, root_z=0.530))

        with self.assertRaisesRegex(RuntimeError, "base height"):
            bench_anymal._validate_workload(self._make_anymal_workload(root_y=0.719, root_z=0.200))


if __name__ == "__main__":
    unittest.main()
