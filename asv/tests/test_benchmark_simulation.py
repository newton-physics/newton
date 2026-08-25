# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

ASV_DIR = Path(__file__).parents[1]
ROOT = ASV_DIR.parent
BENCHMARK_DIR = ASV_DIR / "benchmarks"
sys.path.insert(0, str(ASV_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))


_WARP_CONFIG_FIELDS = ("enable_backward", "log_level")
_WARP_CONFIG_BEFORE_BENCHMARK_IMPORTS = {name: getattr(wp.config, name) for name in _WARP_CONFIG_FIELDS}
_DEFERRED_WORKLOAD_MODULES = (
    "benchmark_kamino",
    "benchmark_mujoco",
    "newton.examples.basic.example_basic_urdf",
    "newton.examples.robot.example_robot_anymal_c_walk",
)
_DEFERRED_WORKLOAD_MODULES_BEFORE_IMPORT = {name: name in sys.modules for name in _DEFERRED_WORKLOAD_MODULES}

try:
    from benchmark_config import pr_gate_repeat
    from benchmark_metrics import SimulationMetrics
    from run_pr_benchmarks import build_pr_config, load_benchmark_patterns
    from simulation import (
        bench_anymal,
        bench_contacts,
        bench_kamino,
        bench_mujoco,
        bench_quadruped_xpbd,
        bench_selection,
        bench_sensor_tiled_camera,
    )

    _DEFERRED_WORKLOAD_MODULES_AFTER_METRIC_IMPORT = {name: name in sys.modules for name in _DEFERRED_WORKLOAD_MODULES}

    from benchmark_kamino import DRLegsBenchmarkWorkload
    from benchmark_mujoco import Example as MuJoCoExample
finally:
    for _name, _value in _WARP_CONFIG_BEFORE_BENCHMARK_IMPORTS.items():
        setattr(wp.config, _name, _value)


class TestSimulationBenchmarks(unittest.TestCase):
    _discovery_cache: ClassVar[dict[bool, list[dict]]] = {}

    @classmethod
    def _discover_benchmarks(cls, *, pr_gate: bool) -> list[dict]:
        """Discover the ASV inventory in an isolated process."""
        if pr_gate not in cls._discovery_cache:
            script = (
                "import json; "
                "import warp as wp; "
                "wp.get_cuda_device_count = lambda: 1; "
                "from asv_runner.discovery import disc_benchmarks, update_sys_path; "
                f"root = {str(BENCHMARK_DIR)!r}; "
                "update_sys_path(root); "
                "benchmarks = disc_benchmarks(root); "
                "data = [{key: getattr(benchmark, key, None) "
                "for key in ('name', 'params', 'repeat', 'rounds')} "
                "for benchmark in benchmarks]; "
                "print('ASV_INVENTORY=' + json.dumps(data))"
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            if pr_gate:
                env["NEWTON_ASV_PR_GATE"] = "1"
            else:
                env.pop("NEWTON_ASV_PR_GATE", None)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise AssertionError(result.stderr)
            payload = next(
                line.removeprefix("ASV_INVENTORY=")
                for line in result.stdout.splitlines()
                if line.startswith("ASV_INVENTORY=")
            )
            cls._discovery_cache[pr_gate] = json.loads(payload)
        return cls._discovery_cache[pr_gate]

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

    def test_asv_runner_resolves_benchmark_support_modules(self):
        """Resolve shared modules through ASV's isolated import path."""
        script = (
            "import importlib; "
            "from asv_runner.discovery import update_sys_path; "
            f"update_sys_path({str(BENCHMARK_DIR)!r}); "
            "importlib.import_module('benchmarks.simulation.bench_cloth'); "
            "importlib.import_module('benchmarks.simulation.bench_contacts'); "
            "importlib.import_module('benchmarks.simulation.bench_selection')"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["NEWTON_ASV_PR_GATE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=BENCHMARK_DIR.parents[1],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_benchmark_imports_preserve_warp_config(self):
        """Preserve Warp global configuration across benchmark imports."""
        self.assertEqual(
            {name: getattr(wp.config, name) for name in _WARP_CONFIG_FIELDS},
            _WARP_CONFIG_BEFORE_BENCHMARK_IMPORTS,
        )

    def test_benchmark_modules_defer_workload_imports(self):
        """Defer workload-only imports until benchmark setup."""
        for name in _DEFERRED_WORKLOAD_MODULES:
            if not _DEFERRED_WORKLOAD_MODULES_BEFORE_IMPORT[name]:
                self.assertFalse(_DEFERRED_WORKLOAD_MODULES_AFTER_METRIC_IMPORT[name], name)

        self.assertFalse(hasattr(bench_anymal, "Example"))
        self.assertFalse(hasattr(bench_anymal, "newton"))
        self.assertFalse(hasattr(bench_kamino, "DRLegsBenchmarkWorkload"))
        self.assertFalse(hasattr(bench_kamino, "newton"))
        self.assertFalse(hasattr(bench_mujoco, "EventTracer"))
        self.assertFalse(hasattr(bench_mujoco, "Example"))
        self.assertFalse(hasattr(bench_quadruped_xpbd, "Example"))
        self.assertFalse(hasattr(bench_quadruped_xpbd, "newton"))

    def test_convex_benchmark_covers_types_and_scales(self):
        """Cover selector crossovers, duplicate-heavy hulls, and every convex type."""
        self.assertEqual(
            tuple(bench_contacts.FastConvexCollision.params[0]),
            (("hulls", 56), ("hulls_duplicate", 192), ("mixed", 191)),
        )
        self.assertEqual(
            {shape for pair in bench_contacts.MIXED_CONVEX_PAIR_TYPES for shape in pair},
            {"sphere", "box", "capsule", "ellipsoid", "cylinder", "cone", "hull"},
        )

        for vertex_count in bench_contacts.IRREGULAR_ROCK_VERTEX_COUNTS:
            with self.subTest(vertex_count=vertex_count):
                mesh = bench_contacts._make_irregular_rock(vertex_count, seed=100)
                self.assertEqual(len(mesh.vertices), vertex_count)
                self.assertEqual(len(mesh.indices), 6 * (vertex_count - 2))

        duplicate_mesh = bench_contacts._make_irregular_rock(10, seed=100, triangle_local_vertices=True)
        self.assertEqual(len(duplicate_mesh.vertices), len(duplicate_mesh.indices))
        self.assertEqual(len(np.unique(duplicate_mesh.vertices, axis=0)), 10)

    def test_fast_kitchen_g1_validates_kitchen_body_count(self):
        """Validate the configured kitchen body count at runtime."""
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
        """Fall back to eager MuJoCo stepping without a captured graph."""
        example = MuJoCoExample.__new__(MuJoCoExample)
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
        """Reject KPI workloads that fail CUDA graph capture."""
        benchmark = bench_mujoco.FastCartpole()
        with (
            patch("benchmark_mujoco.Example", return_value=SimpleNamespace(graph=None)),
            self.assertRaisesRegex(RuntimeError, "requires CUDA graph capture"),
        ):
            benchmark._create_workload(Mock(), world_count=1)

    def test_mujoco_metrics_include_solver_iterations(self):
        """Publish mean and maximum MuJoCo solver iterations."""
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
            patch.object(MuJoCoExample, "create_model_builder", return_value=Mock()),
            patch.object(bench_mujoco, "collect_simulation_metrics", side_effect=collect_metrics),
        ):
            metrics = benchmark._collect_metrics()[8192]

        self.assertTrue(all(workload.test_final.call_count == 1 for workload in workloads))
        self.assertEqual(metrics.solver_niter_mean, 3.0)
        self.assertEqual(metrics.solver_niter_max, 5.0)

    def test_metric_setup_caches_skip_without_cuda(self):
        """Skip metric caches without constructing CPU workloads."""
        with (
            patch.object(bench_mujoco.wp, "get_cuda_device_count", return_value=0),
            patch.object(MuJoCoExample, "create_model_builder") as create_mujoco_builder,
            patch.object(DRLegsBenchmarkWorkload, "create_model_builder") as create_kamino_builder,
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
        """Give the DR Legs cache longer than ASV's default timeout."""
        config = json.loads((BENCHMARK_DIR.parents[1] / "asv.conf.json").read_text(encoding="utf-8"))
        self.assertGreater(bench_kamino.KpiDRLegs.setup_cache.timeout, config["default_benchmark_timeout"])

    def test_aws_benchmark_comparison_gates_only_runtime_metrics(self):
        """Gate discovered PR runtimes while retaining dashboard-only metrics."""
        workflow_path = ROOT / ".github" / "workflows" / "aws_gpu_benchmarks.yml"
        workflow = workflow_path.read_text(encoding="utf-8")
        self.assertIn("python asv/run_pr_benchmarks.py", workflow)
        patterns = tuple(re.compile(selection) for selection in load_benchmark_patterns())
        inventory = {benchmark["name"]: benchmark for benchmark in self._discover_benchmarks(pr_gate=False)}

        blocking_benchmarks = (
            "simulation.bench_mujoco.FastG1.track_simulate",
            "simulation.bench_mujoco.FastG1.track_p95_step_time",
            "simulation.bench_anymal.FastMetricsExampleAnymalPretrained.track_mean_world_step_time",
            "simulation.bench_teleop_mujoco.FastTeleopMuJoCo.time_teleop_loop",
            "simulation.bench_teleop_mujoco.FastTeleopMuJoCo.track_p95_loop_ms",
            "simulation.bench_kamino.FastDRLegs.time_simulate",
            "simulation.bench_viewer.FastViewerGL.time_rendering_frame",
            "simulation.bench_mujoco.FastAllegro.track_p95_step_time",
            "simulation.bench_mujoco.FastAllegro.track_simulate",
            "simulation.bench_sensor_tiled_camera.FastSensorTiledCamera.time_render_color_depth",
            "simulation.bench_sensor_tiled_camera.FastSensorTiledCamera.time_render_depth_only",
            "simulation.bench_sensor_tiled_camera.FastSensorTiledCameraPixel.time_render_color_depth",
            "simulation.bench_inverse_dynamics.FastInverseDynamics.time_eval_inverse_dynamics_force",
        )
        dashboard_benchmarks = (
            "simulation.bench_mujoco.FastG1.track_solver_niter_mean",
            "simulation.bench_mujoco.FastG1.track_solver_niter_max",
            "simulation.bench_mujoco.FastG1.track_simulation_steps_per_second",
            "simulation.bench_mujoco.FastG1.track_real_time_factor",
            "simulation.bench_mujoco.FastG1.track_steady_state_gpu_memory",
            "simulation.bench_mujoco.FastG1.track_sim_dt",
            "simulation.bench_mujoco.FastG1.track_sim_substeps",
            "simulation.bench_mujoco.FastNewtonOverheadG1.track_simulate",
            "simulation.bench_teleop_mujoco.TeleopMuJoCo.track_frame_overrun_pct",
            "simulation.bench_teleop_mujoco.FastTeleopMuJoCo.track_mean_loop_ms",
            "simulation.bench_sensor_tiled_camera.FastSensorTiledCamera.time_render_color_only",
            "simulation.bench_sensor_tiled_camera.FastSensorTiledCameraPixel.time_render_color_only",
            "simulation.bench_sensor_tiled_camera.FastSensorTiledCameraPixel.time_render_depth_only",
        )

        for benchmark in blocking_benchmarks + dashboard_benchmarks:
            self.assertIn(benchmark, inventory)
        self.assertEqual(inventory["simulation.bench_mujoco.FastG1.track_simulate"]["params"], [["8192"]])
        for benchmark in blocking_benchmarks:
            with self.subTest(benchmark=benchmark):
                self.assertTrue(any(pattern.search(benchmark) for pattern in patterns), benchmark)
        for benchmark in dashboard_benchmarks:
            with self.subTest(benchmark=benchmark):
                self.assertFalse(any(pattern.search(benchmark) for pattern in patterns), benchmark)

    def test_pr_gate_caps_repeats_without_dropping_cases(self):
        """Keep full workloads with three timing samples in the PR gate."""
        cases = (
            (bench_quadruped_xpbd.FastExampleQuadrupedXPBD, 10),
            (bench_selection.FastExampleSelectionCartpoleMuJoCo, 10),
            (bench_contacts.FastConvexCollision, 5),
        )
        for benchmark_cls, full_repeat in cases:
            with self.subTest(benchmark=benchmark_cls.__name__):
                self.assertEqual(benchmark_cls.repeat, pr_gate_repeat(full_repeat))
                with patch.dict("os.environ", {"NEWTON_ASV_PR_GATE": "1"}):
                    self.assertEqual(pr_gate_repeat(full_repeat), 3)
                with patch.dict("os.environ", {}, clear=True):
                    self.assertEqual(pr_gate_repeat(full_repeat), full_repeat)

    def test_fast_allegro_uses_representative_pr_workload(self):
        """Keep Allegro coverage while reducing duplicated PR sampling."""
        benchmark = bench_mujoco.FastAllegro()
        self.assertEqual((benchmark.num_frames, benchmark.samples), (300, 2))

        with (
            patch.dict("os.environ", {"NEWTON_ASV_PR_GATE": "1"}),
            patch.object(benchmark, "_collect_metrics", return_value={}) as collect_metrics,
        ):
            self.assertEqual(benchmark.setup_cache(), {})

        self.assertEqual((benchmark.num_frames, benchmark.samples), (200, 1))
        collect_metrics.assert_called_once_with()

    def test_pr_asv_config_only_omits_torch(self):
        """Derive the PR environment from the full ASV dependency pins."""
        full_config = json.loads((ROOT / "asv.conf.json").read_text(encoding="utf-8"))
        pr_config = build_pr_config()

        for config in (full_config, pr_config):
            commands = config["install_command"]
            pinned_stack_index = next(i for i, command in enumerate(commands) if "warp-lang==" in command)
            newton_index = next(i for i, command in enumerate(commands) if "{wheel_file}" in command)
            self.assertLess(pinned_stack_index, newton_index)
            self.assertIn("[examples]", commands[newton_index])
            self.assertNotIn("[dev]", commands[newton_index])

        expected_pr_config = json.loads(json.dumps(full_config))
        expected_pr_config["env_dir"] = "asv/pr-env"
        torch_commands = [command for command in expected_pr_config["install_command"] if "torch==" in command]
        self.assertEqual(len(torch_commands), 1)
        expected_pr_config["install_command"].remove(torch_commands[0])

        self.assertEqual(pr_config, expected_pr_config)
        self.assertFalse((ROOT / "asv-pr.conf.json").exists())

    def test_pr_camera_warmup_matches_selected_outputs(self):
        """Limit camera outputs and fixed repeats to the PR gate."""
        prefix = "simulation.bench_sensor_tiled_camera.FastSensorTiledCamera"
        full = {
            item["name"]: item for item in self._discover_benchmarks(pr_gate=False) if item["name"].startswith(prefix)
        }
        gated = {
            item["name"]: item for item in self._discover_benchmarks(pr_gate=True) if item["name"].startswith(prefix)
        }
        self.assertEqual(len(full), 6)
        self.assertEqual(
            set(gated),
            {
                f"{prefix}.time_render_color_depth",
                f"{prefix}.time_render_depth_only",
                f"{prefix}Pixel.time_render_color_depth",
            },
        )
        self.assertEqual({item["repeat"] for item in full.values()}, {0})
        self.assertEqual({item["repeat"] for item in gated.values()}, {1})

        with patch.dict("os.environ", {"NEWTON_ASV_PR_GATE": "1"}):
            self.assertFalse(bench_sensor_tiled_camera._pr_gate_skips_output(True, True, ((True, True),)))
            self.assertTrue(bench_sensor_tiled_camera._pr_gate_skips_output(True, False, ((True, True),)))
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(bench_sensor_tiled_camera._pr_gate_skips_output(True, False, ((True, True),)))

        cases = (
            (bench_sensor_tiled_camera.FastSensorTiledCamera, ((True, True), (False, True))),
            (bench_sensor_tiled_camera.FastSensorTiledCameraPixel, ((True, True),)),
        )
        for benchmark_cls, expected_modes in cases:
            with (
                self.subTest(benchmark=benchmark_cls.__name__),
                patch.dict("os.environ", {"NEWTON_ASV_PR_GATE": "1"}),
                patch.object(bench_sensor_tiled_camera, "_TiledCameraSceneRig") as rig_cls,
                patch.object(bench_sensor_tiled_camera.wp, "synchronize"),
            ):
                rig = rig_cls.return_value
                benchmark_cls().setup(64, 4096, 50)

            self.assertEqual(
                tuple((call.kwargs["color"], call.kwargs["depth"]) for call in rig.render.call_args_list),
                expected_modes,
            )

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(bench_sensor_tiled_camera, "_TiledCameraSceneRig") as rig_cls,
            patch.object(bench_sensor_tiled_camera.wp, "synchronize"),
        ):
            rig = rig_cls.return_value
            bench_sensor_tiled_camera.FastSensorTiledCameraPixel().setup(64, 4096, 50)

        self.assertEqual(
            tuple((call.kwargs["color"], call.kwargs["depth"]) for call in rig.render.call_args_list),
            ((True, True), (True, False), (False, True)),
        )

    def test_anymal_short_horizon_validation(self):
        """Validate short-horizon ANYmal posture and forward progress."""
        bench_anymal._validate_workload(self._make_anymal_workload(root_y=0.719, root_z=0.530))

        with self.assertRaisesRegex(RuntimeError, "forward progress"):
            bench_anymal._validate_workload(self._make_anymal_workload(root_y=0.0, root_z=0.530))

        with self.assertRaisesRegex(RuntimeError, "base height"):
            bench_anymal._validate_workload(self._make_anymal_workload(root_y=0.719, root_z=0.200))


if __name__ == "__main__":
    unittest.main()
