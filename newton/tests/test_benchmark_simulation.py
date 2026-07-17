# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import numpy as np

BENCHMARK_DIR = Path(__file__).parents[2] / "asv" / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

from simulation import bench_anymal, bench_kamino, bench_mujoco  # noqa: E402


class _FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self.values


def _make_anymal_workload(root_y, root_z):
    state = SimpleNamespace(
        joint_q=_FakeArray([0.0, root_y, root_z, 0.0, 0.0, 0.0, 1.0]),
        joint_qd=_FakeArray([0.0] * 6),
        body_q=_FakeArray([[0.0, root_y, root_z, 0.0, 0.0, 0.0, 1.0]]),
        body_qd=_FakeArray([[0.0] * 6]),
    )
    return SimpleNamespace(state_0=state)


class TestSimulationBenchmarks(unittest.TestCase):
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
            512,
            environment="kitchen",
            randomize=True,
            seed=123,
        )
        self.assertEqual(metrics, {512: sentinel.metrics})
        self.assertEqual(bench_mujoco.FastKitchenG1.version, "2")

    def test_kpi_dr_legs_has_setup_cache_timeout(self):
        self.assertEqual(bench_kamino.KpiDRLegs.setup_cache.timeout, 1200)

    def test_anymal_short_horizon_validation(self):
        bench_anymal._validate_workload(_make_anymal_workload(root_y=0.719, root_z=0.530))

        with self.assertRaisesRegex(RuntimeError, "forward progress"):
            bench_anymal._validate_workload(_make_anymal_workload(root_y=0.0, root_z=0.530))

        with self.assertRaisesRegex(RuntimeError, "base height"):
            bench_anymal._validate_workload(_make_anymal_workload(root_y=0.719, root_z=0.200))


if __name__ == "__main__":
    unittest.main()
