# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from projects.digital_instron import core, objectives


class TestDigitalInstronProject(unittest.TestCase):
    def test_cli_accepts_subcommand_style(self):
        args = core._apply_mode_defaults(
            core.build_arg_parser().parse_args(core._normalize_cli_argv(["calibrate", "--viewer", "null"]))
        )

        self.assertEqual(args.mode, "calibrate")
        self.assertEqual(args.viewer, "null")
        self.assertEqual(args.compression_relax_steps, 4)
        self.assertEqual(args.fixture_solver, "mujoco")

    def test_view_defaults_to_dynamic_fixture_replay(self):
        args = core._apply_mode_defaults(
            core.build_arg_parser().parse_args(core._normalize_cli_argv(["view", "--test-case", "fullfoot"]))
        )

        self.assertEqual(args.compression_relax_steps, 4)

    def test_mujoco_fixture_solver_flag_is_available(self):
        args = core._apply_mode_defaults(
            core.build_arg_parser().parse_args(core._normalize_cli_argv(["run", "--fixture-solver", "mujoco"]))
        )

        self.assertEqual(args.fixture_solver, "mujoco")
        self.assertEqual(args.fixture_mujoco_nconmax, 4096)
        self.assertEqual(args.fixture_mujoco_njmax, 16384)
        self.assertFalse(args.no_contact_reduction)
        self.assertTrue(args.fail_on_fixture_contact_overflow)

    def test_fixture_contact_capacity_fails_before_mujoco_truncates(self):
        scene: Any = argparse.Namespace(
            test_case="fullfoot",
            contacts=SimpleNamespace(rigid_contact_count=SimpleNamespace(numpy=lambda: core.np.asarray([4097]))),
        )
        args = argparse.Namespace(
            fixture_solver="mujoco",
            fixture_mujoco_cpu=False,
            fixture_mujoco_nconmax=4096,
            fail_on_fixture_contact_overflow=True,
            no_contact_reduction=True,
        )

        with self.assertRaisesRegex(RuntimeError, "truncat"):
            core._validate_fixture_contact_capacity(scene, args)

    def test_solver_divergence_raises_runtime_error_immediately(self):
        state_q = core.np.zeros((1, 7), dtype=core.np.float64)
        state_next_q = core.np.zeros((1, 7), dtype=core.np.float64)

        def state_numpy():
            return state_q

        def state_next_numpy():
            return state_next_q

        def solver_step(*_args):
            state_next_q[0, 0] = core.np.nan

        scene: Any = SimpleNamespace(
            test_case="fullfoot",
            material_kh=1.0,
            state=SimpleNamespace(body_q=SimpleNamespace(numpy=state_numpy), clear_forces=mock.Mock()),
            state_next=SimpleNamespace(body_q=SimpleNamespace(numpy=state_next_numpy)),
            pipeline=SimpleNamespace(collide=mock.Mock()),
            contacts=SimpleNamespace(),
            solver=SimpleNamespace(step=solver_step),
            control=SimpleNamespace(),
            indenter_body=0,
            indenter_stop_z=0.0,
        )
        args = SimpleNamespace()

        with (
            mock.patch.object(core, "_stabilize_shoe_fixture"),
            mock.patch.object(core, "_set_body_z_velocity"),
            mock.patch.object(core, "_validate_fixture_contact_capacity"),
            self.assertRaisesRegex(RuntimeError, "Solver diverged"),
        ):
            core._fixture_step(scene, args, indenter_z=0.0, indenter_velocity_z=0.0, dt=0.01, substeps=1)

    def test_force_fit_objective_is_stride_invariant(self):
        displacement_m = core.np.linspace(0.0, 0.04, 9)
        measured_force_n = 100.0 + 2500.0 * displacement_m
        sim_force_n = measured_force_n + 5.0

        dense_objective = objectives.force_fit_objective(displacement_m, measured_force_n, sim_force_n)
        sparse_objective = objectives.force_fit_objective(
            displacement_m[::2],
            measured_force_n[::2],
            sim_force_n[::2],
        )

        self.assertAlmostEqual(dense_objective[0], sparse_objective[0])
        self.assertAlmostEqual(dense_objective[1], sparse_objective[1])
        self.assertAlmostEqual(dense_objective[2], sparse_objective[2])
        self.assertAlmostEqual(dense_objective[2], 5.0)

    def test_measure_trial_force_uses_solver_body_force(self):
        body_f = core.np.asarray([[0.0, 0.0, 2.25], [0.0, 0.0, 7.5]], dtype=core.np.float64)
        scene: Any = SimpleNamespace(
            state=SimpleNamespace(body_f=SimpleNamespace(numpy=lambda: body_f)),
            pipeline=SimpleNamespace(collide=mock.Mock()),
            contacts=SimpleNamespace(),
            indenter_body=1,
            shoe_body=0,
            indenter_shape=3,
            midsole_shape=4,
            base_shape=5,
            model=SimpleNamespace(),
        )

        with mock.patch.object(core, "_pair_contact_force_z", return_value=(99.0, {"contact_pairs": 3})) as pair_force:
            force, stats = core._measure_trial_force(scene)

        self.assertEqual(force, 7.5)
        self.assertEqual(stats["top_contact_force_n"], 7.5)
        self.assertEqual(stats["bottom_contact_force_n"], 2.25)
        pair_force.assert_not_called()

    def test_measure_trial_force_falls_back_when_solver_force_is_missing(self):
        body_f = core.np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=core.np.float64)
        scene: Any = SimpleNamespace(
            state=SimpleNamespace(body_f=SimpleNamespace(numpy=lambda: body_f)),
            contacts=SimpleNamespace(),
            indenter_body=1,
            shoe_body=0,
            indenter_shape=3,
            midsole_shape=4,
            base_shape=5,
            model=SimpleNamespace(),
        )

        with mock.patch.object(core, "_pair_contact_force_z", side_effect=[(12.0, {"contact_pairs": 2}), (4.0, {})]):
            force, stats = core._measure_trial_force(scene)

        self.assertEqual(force, 12.0)
        self.assertEqual(stats["top_contact_force_n"], 12.0)
        self.assertEqual(stats["top_solver_force_n"], 0.0)

    def test_find_trial_contact_offset_stops_at_contact_threshold(self):
        args = argparse.Namespace(
            fullfoot_auto_contact_offset=True,
            fullfoot_contact_search_max=0.2,
            fullfoot_start_clearance=0.0,
            fullfoot_z_offset=0.0,
            fullfoot_min_start_clearance=0.0,
            fullfoot_initial_force_max=25.0,
            fullfoot_contact_search_steps=3,
            fullfoot_contact_search_tolerance=0.02,
            fullfoot_stop_search_after_target=False,
        )
        state = SimpleNamespace()
        displacement_m = core.np.asarray([0.0, 0.1, 0.2], dtype=core.np.float64)
        measured_force_n = core.np.asarray([0.0, 0.0, 0.0], dtype=core.np.float64)
        indenter_rest_z = 0.5

        cases = [
            (
                [0.5, 1.2],
                0.1,
                2,
                1.2,
                [indenter_rest_z, indenter_rest_z - 0.1],
            ),
            (
                [1.0],
                0.0,
                1,
                1.0,
                [indenter_rest_z],
            ),
        ]

        for forces, expected_offset, expected_calls, expected_initial_force, expected_zs in cases:
            with self.subTest(forces=forces):
                force_iter = iter(forces)

                def fake_pair_contact_force_z(*_args):
                    return next(force_iter), {"contact_pairs": 1}

                with (
                    mock.patch.object(core, "_set_body_z") as set_body_z,
                    mock.patch.object(
                        core, "_pair_contact_force_z", side_effect=fake_pair_contact_force_z
                    ) as pair_force,
                ):
                    offset, stats = core._find_trial_contact_offset(
                        args,
                        test_case="fullfoot",
                        pipeline=SimpleNamespace(collide=mock.Mock()),
                        contacts=SimpleNamespace(),
                        model=SimpleNamespace(),
                        state=state,
                        indenter_body=0,
                        indenter_shape=1,
                        midsole_shape=2,
                        indenter_rest_z=indenter_rest_z,
                        displacement_m=displacement_m,
                        measured_force_n=measured_force_n,
                    )

                self.assertEqual(offset, expected_offset)
                self.assertEqual(stats["fullfoot_auto_contact_threshold_n"], 1.0)
                self.assertEqual(stats["fullfoot_auto_initial_contact_force_n"], expected_initial_force)
                self.assertEqual(pair_force.call_count, expected_calls)
                self.assertEqual(
                    [call.args[2] for call in set_body_z.call_args_list[:-1]],
                    expected_zs,
                )

    def test_read_body_force_z_returns_positive_force_during_compression(self):
        def make_state(body_force_z: float) -> SimpleNamespace:
            return SimpleNamespace(
                body_f=SimpleNamespace(
                    numpy=lambda body_force_z=body_force_z: core.np.asarray([[0.0, 0.0, body_force_z]], dtype=float)
                )
            )

        loading_cases = [
            (0.0, -5.0),
            (0.5, -7.5),
            (1.0, -10.0),
        ]
        measured_forces: list[float] = []

        for displacement_m, body_force_z in loading_cases:
            scene: Any = SimpleNamespace(indenter_body=0, displacement_m=displacement_m, state=make_state(body_force_z))
            measured_force = core._read_body_force_z(scene.state, scene.indenter_body)
            self.assertGreater(measured_force, 0.0)
            measured_forces.append(measured_force)

        self.assertTrue(all(next_force > force for force, next_force in zip(measured_forces, measured_forces[1:])))

    def test_fixture_step_many_preserves_solver_body_f_after_swap(self):
        """Regression: Newton writes body_f to state_in, not state_out.

        After the ping-pong swap inside _fixture_step_many, scene.state points
        to state_out which has an empty body_f buffer. The function must copy
        the computed forces back so _measure_trial_force reads valid values.
        """
        expected_force = 42.0

        def make_body_f(value: float):
            arr = core.np.zeros((2, 6), dtype=core.np.float32)
            arr[1, 2] = value
            return SimpleNamespace(numpy=lambda: arr, assign=lambda a: None)

        state_in = SimpleNamespace(
            body_count=2,
            body_q=SimpleNamespace(numpy=lambda: core.np.zeros((2, 7))),
            body_qd=SimpleNamespace(numpy=lambda: core.np.zeros((2, 6))),
            body_f=make_body_f(expected_force),
            clear_forces=lambda: None,
        )
        state_out = SimpleNamespace(
            body_count=2,
            body_q=SimpleNamespace(numpy=lambda: core.np.zeros((2, 7))),
            body_qd=SimpleNamespace(numpy=lambda: core.np.zeros((2, 6))),
            body_f=make_body_f(0.0),
            clear_forces=lambda: None,
        )

        assigned = []

        def capture_assign(arr):
            assigned.append(arr.copy())

        state_out.body_f.assign = capture_assign

        scene: Any = SimpleNamespace(
            indenter_body=1,
            indenter_stop_z=-1.0,
            shoe_body=0,
            shoe_anchor_xy=(0.0, 0.0),
            shoe_anchor_quat=(0.0, 0.0, 0.0, 1.0),
            settled_rotation_locked=False,
            state=state_in,
            state_next=state_out,
            pipeline=SimpleNamespace(
                collide=lambda _s, _c: None,
                contacts=lambda: SimpleNamespace(),
            ),
            contacts=SimpleNamespace(),
            solver=SimpleNamespace(step=lambda _a, _b, _c, _d, _e: None),
            control=SimpleNamespace(),
        )

        args = SimpleNamespace(
            fixture_lock_shoe_horizontal=False,
            fixture_solver="semi-implicit",
            fail_on_fixture_contact_overflow=False,
        )

        with mock.patch.object(core, "_set_body_z_velocity"):
            core._fixture_step_many(
                [scene],
                args,
                indenter_z=[0.5],
                indenter_velocity_z=[0.0],
                dt=0.01,
                substeps=1,
            )

        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0][1, 2], expected_force)

    def test_bundle_schema_includes_memoryless_state_model(self):
        args = core.build_arg_parser().parse_args(core._normalize_cli_argv(["run"]))
        args.midsole_mesh = "DigitalInstron/midsole.obj"
        args.indenter_mesh = "DigitalInstron/indenter.stl"
        args.mesh_scale = 0.001
        args.fixture_gravity = True
        args.fixture_dynamic_shoe = True
        args.fixture_lock_shoe_horizontal = True
        args.fixture_free_rotation_axis = "none"
        args.fixture_settle_duration = 0.5
        args.fixture_settle_velocity_tol = 1.0e-3
        args.fixture_settle_indenter_clearance = 0.0
        args.fixture_platen_stop_clearance = 1.0e-4
        args.fixture_bottom_platen_overlap = 0.0
        args.compression_relax_steps = 4
        args.fullfoot_rotation_deg = (90.0, 0.0, 0.0)
        args.fullfoot_indenter_crop_fixture = True
        args.fullfoot_indenter_fixture_crop_z = 0.08
        args.fullfoot_start_clearance = 0.0
        args.fullfoot_min_start_clearance = 0.0
        args.punch_radius = 0.022
        args.punch_half_height = 0.025
        args.heel_side = "min"
        args.fullfoot_auto_contact_offset = True
        args.fullfoot_contact_search_max = 0.08
        args.fullfoot_contact_search_steps = 41
        args.fullfoot_initial_force_max = 25.0
        args.fullfoot_contact_search_tolerance = 0.02
        args.pressure_profile = "layer"
        args.layer_eta_lock = 0.65
        args.layer_densification_power = 0.5
        args.layer_cubic = 0.0
        args.layer_quintic = 0.0
        args.pressure_sine_amplitude = (0.0, 0.0, 0.0)
        args.pressure_sine_cycles = (1.0, 1.0, 1.0)
        args.pressure_sine_phase = (0.0, 0.0, 0.0)
        material_payload = {
            "material_model": "digital_instron_pressure_field_v1",
            "source_midsole_mesh": "DigitalInstron/midsole.obj",
            "source_indenter_mesh": "DigitalInstron/indenter.stl",
            "parameters": {
                "kh": 1.0,
                "pressure_profile": "layer",
                "layer_eta_lock": 0.65,
                "layer_densification_power": 0.5,
            },
            "calibration": {"combined_objective": 12.0},
        }

        bundle = core._bundle_payload_from_material(args, material_payload=material_payload)

        self.assertEqual(bundle["asset_type"], "shoe_pressure_field_bundle")
        self.assertEqual(bundle["state_model"]["type"], "memoryless_v1")
        self.assertEqual(bundle["fixture"]["shoe_body"], "dynamic_rigid_body_with_compliant_midsole_mesh")
        self.assertTrue(bundle["fixture"]["shoe_horizontal_lock"])
        self.assertEqual(bundle["fixture"]["shoe_free_rotation_axis"], "none")
        self.assertEqual(bundle["fixture"]["reaction_force_source"], "top_indenter")
        self.assertEqual(bundle["fixture"]["platen_stop_clearance_m"], 1.0e-4)
        self.assertEqual(bundle["fixture"]["bottom_platen_overlap_m"], 0.0)
        self.assertEqual(bundle["fixture"]["compression_replay"]["substeps_per_sample"], 4)
        self.assertTrue(bundle["fixture"]["settling"]["implemented"])
        self.assertEqual(bundle["fixture"]["auto_contact_search"]["pre_embedding"], "disallowed")
        self.assertTrue(bundle["geometry"]["fullfoot_indenter_crop_fixture"])
        self.assertEqual(bundle["geometry"]["fullfoot_indenter_fixture_crop_z_m"], 0.08)
        self.assertEqual(bundle["calibration"]["combined_objective"], 12.0)

    def test_pressure_memory_state_model_records_parameters(self):
        args = core.build_arg_parser().parse_args(
            core._normalize_cli_argv(
                [
                    "run",
                    "--pressure-memory",
                    "--pressure-memory-grid",
                    "12",
                    "6",
                    "2",
                    "--pressure-memory-unloading-loss",
                    "0.35",
                    "--pressure-memory-recovery-tau-s",
                    "0.5",
                    "--pressure-memory-dt-s",
                    "0.01",
                ]
            )
        )

        state_model = core._pressure_memory_state_model(args)

        self.assertEqual(state_model["type"], "max_compression_memory_v1")
        self.assertEqual(state_model["parameters"]["grid"], [12, 6, 2])
        self.assertEqual(state_model["parameters"]["unloading_loss"], 0.35)
        self.assertEqual(state_model["parameters"]["recovery_tau_s"], 0.5)
        self.assertEqual(state_model["parameters"]["dt_s"], 0.01)

    def test_apply_material_parameters_uses_direct_kh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            material_path = Path(tmpdir) / "material.json"
            material_path.write_text(
                json.dumps(
                    {
                        "parameters": {
                            "kh": 1.0,
                            "kd": 1.5,
                            "pressure_profile": "layer",
                            "layer_eta_lock": 0.7,
                            "layer_densification_power": 0.2,
                            "sdf_resolution": 48,
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                load_material=True,
                material_output_json=str(material_path),
                kh=1.0,
                kd=0.0,
                pressure_profile="poisson",
                layer_eta_lock=0.65,
                layer_densification_power=0.0,
                layer_cubic=0.0,
                layer_quintic=0.0,
                pressure_sine_amplitude=(0.0, 0.0, 0.0),
                pressure_sine_cycles=(1.0, 1.0, 1.0),
                pressure_sine_phase=(0.0, 0.0, 0.0),
                narrow_band=0.01,
                sdf_resolution=64,
            )

            loaded = core._apply_material_parameters(args)

        self.assertEqual(args.kh, 1.0)
        self.assertEqual(args.kd, 1.5)
        self.assertEqual(args.pressure_profile, "layer")
        self.assertEqual(args.layer_eta_lock, 0.7)
        self.assertEqual(args.layer_densification_power, 0.2)
        self.assertEqual(args.sdf_resolution, 48)
        assert loaded is not None
        self.assertEqual(loaded["kh"], 1.0)
        self.assertEqual(loaded["kd"], 1.5)

    def test_shared_material_parameters_report_physics_values(self):
        args = argparse.Namespace(
            kh=42.0,
            kd=2.0,
        )
        response = core.TrialResponse(
            test_case="case",
            trace_csv="trace.csv",
            output_csv="out.csv",
            trace={"phase": core.np.asarray([0.0, 1.0])},
            displacement_m=core.np.asarray([0.0, 1.0]),
            time_s=core.np.asarray([0.0, 1.0]),
            displacement_velocity_m_s=core.np.asarray([0.0, 1.0]),
            measured_force_n=core.np.asarray([0.0, 10.0]),
            raw_contact_force_n=core.np.asarray([0.0, 10.0]),
            contact_stats=[],
            midsole_stats={},
        )

        material = core._fit_shared_material_parameters([response], args)

        self.assertEqual(material["kh"], 42.0)
        self.assertEqual(material["kd"], 2.0)
        self.assertEqual(material["objective_components"]["mode"], "force")

    def test_material_json_reports_contact_nuisance_metadata(self):
        args = argparse.Namespace(
            kh=42.0,
            kd=2.0,
            layer_eta_lock=0.65,
            layer_densification_power=0.0,
            midsole_mesh="midsole.obj",
            indenter_mesh="indenter.stl",
            pressure_profile="layer",
            layer_cubic=0.0,
            layer_quintic=0.0,
            pressure_sine_amplitude=(0.0, 0.0, 0.0),
            pressure_sine_cycles=(1.0, 1.0, 1.0),
            pressure_sine_phase=(0.0, 0.0, 0.0),
            narrow_band=0.01,
            sdf_resolution=64,
        )
        response = core.TrialResponse(
            test_case="fullfoot",
            trace_csv="trace.csv",
            output_csv="out.csv",
            trace={"phase": core.np.asarray([0.0, 1.0])},
            displacement_m=core.np.asarray([0.0, 1.0]),
            time_s=core.np.asarray([0.0, 1.0]),
            displacement_velocity_m_s=core.np.asarray([0.0, 1.0]),
            measured_force_n=core.np.asarray([0.0, 10.0]),
            raw_contact_force_n=core.np.asarray([0.0, 10.0]),
            contact_stats=[],
            midsole_stats={
                "fullfoot_auto_contact_offset_m": 0.0125,
                "fullfoot_auto_contact_threshold_n": 1.0,
            },
        )
        material = core._fit_shared_material_parameters([response], args)

        with tempfile.TemporaryDirectory() as tmpdir:
            material_path = Path(tmpdir) / "material.json"
            trial_summary = {
                "trace_csv": "trace.csv",
                "output_csv": "out.csv",
                "measured_peak_force_n": 10.0,
                "sim_peak_force_n": 9.5,
                "raw_contact_peak_force_n": 9.0,
                "raw_contact_peak_relative_error": 0.1,
                "peak_relative_error": 0.05,
                "loop_rmse_n": 0.2,
                "force_r2": 0.99,
                "max_abs_shoe_vertical_velocity_m_s": 0.0,
                "rms_shoe_vertical_velocity_m_s": 0.0,
                "max_abs_shoe_free_axis_angular_velocity_rad_s": 0.0,
                "rms_shoe_free_axis_angular_velocity_rad_s": 0.0,
                "max_adjacent_force_jump_n": 0.0,
                "force_jump_limit_n": 200.0,
                "force_jump_violation_count": 0,
                "metrics": {},
            }
            core._write_material_json(
                material_path,
                args,
                material=material,
                summaries={"fullfoot": trial_summary},
            )
            payload = json.loads(material_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["contact_offset_m"], 0.0125)
        self.assertEqual(payload["contact_threshold_n"], 1.0)
        self.assertIn("geometric nuisance parameter", payload["description"])
        self.assertNotIn("contact_offset_m", payload["parameters"])

    def test_goal_objective_penalizes_residual_shoe_motion(self):
        args = argparse.Namespace(
            kh=42.0,
            kd=2.0,
            calibration_objective="goal",
            goal_rmse_weight=1.0,
            goal_r2_weight=1.0,
            goal_hysteresis_weight=0.0,
            goal_shoe_vz_weight=1.0,
            goal_shoe_omega_weight=1.0,
            goal_shoe_vz_target=0.001,
            goal_shoe_omega_target=0.1,
        )
        response = core.TrialResponse(
            test_case="case",
            trace_csv="trace.csv",
            output_csv="out.csv",
            trace={"phase": core.np.asarray([0.0, 0.5, 1.0])},
            displacement_m=core.np.asarray([0.0, 0.01, 0.0]),
            time_s=core.np.asarray([0.0, 0.1, 0.2]),
            displacement_velocity_m_s=core.np.asarray([0.0, 0.1, -0.1]),
            measured_force_n=core.np.asarray([0.0, 100.0, 0.0]),
            raw_contact_force_n=core.np.asarray([0.0, 90.0, 10.0]),
            contact_stats=[
                {
                    "shoe_vertical_velocity_m_s": 0.0,
                    "shoe_free_axis_angular_velocity_rad_s": 0.0,
                    "shoe_angular_velocity_norm_rad_s": 0.0,
                },
                {
                    "shoe_vertical_velocity_m_s": -0.003,
                    "shoe_free_axis_angular_velocity_rad_s": 0.2,
                    "shoe_angular_velocity_norm_rad_s": 0.25,
                },
                {
                    "shoe_vertical_velocity_m_s": 0.001,
                    "shoe_free_axis_angular_velocity_rad_s": -0.1,
                    "shoe_angular_velocity_norm_rad_s": 0.1,
                },
            ],
            midsole_stats={},
        )

        material = core._fit_shared_material_parameters([response], args)
        components = material["objective_components"]

        self.assertEqual(components["mode"], "goal")
        self.assertEqual(components["selected_component"], "goal_objective")
        self.assertEqual(material["combined_objective"], components["combined_objective"])
        self.assertGreater(components["mean"]["shoe_vz_relative"], 1.0)
        self.assertGreater(components["mean"]["shoe_omega_relative"], 1.0)
        self.assertGreater(material["combined_objective"], components["mean"]["force_objective"])

    def test_force_jump_metrics_detect_adjacent_spikes(self):
        metrics = core._force_jump_metrics(core.np.asarray([0.0, 100.0, 350.0, 340.0]), 200.0)

        self.assertEqual(metrics["force_jump_limit_n"], 200.0)
        self.assertEqual(metrics["max_adjacent_force_jump_n"], 250.0)
        self.assertEqual(metrics["max_adjacent_force_jump_index"], 2)
        self.assertEqual(metrics["force_jump_violation_count"], 1)
        self.assertEqual(metrics["force_jump_violation"], 1)

    def test_loop_metrics_reports_force_r2(self):
        metrics = core._loop_metrics(
            core.np.asarray([0.0, 0.01, 0.0]),
            core.np.asarray([0.0, 100.0, 0.0]),
            core.np.asarray([0.0, 90.0, 10.0]),
        )

        self.assertIn("force_r2", metrics)
        self.assertLess(metrics["force_r2"], 1.0)
        self.assertGreater(metrics["force_r2"], 0.9)

    def test_loop_metrics_uses_velocity_split_for_plateau(self):
        metrics = core._loop_metrics(
            core.np.asarray([0.0, 0.01, 0.01, 0.005]),
            core.np.asarray([0.0, 1.0, 1.0, 0.0]),
            core.np.asarray([0.0, 1.0, 2.0, 0.0]),
        )

        self.assertAlmostEqual(metrics["loading_rmse_n"], 1.0 / 3.0**0.5)
        self.assertAlmostEqual(metrics["unloading_rmse_n"], 1.0 / 2.0**0.5)

    def test_residual_motion_metrics_summarize_shoe_bounce(self):
        metrics = core._residual_motion_metrics(
            [
                {
                    "shoe_vertical_velocity_m_s": 0.0,
                    "shoe_free_axis_angular_velocity_rad_s": 0.0,
                    "shoe_angular_velocity_norm_rad_s": 0.0,
                },
                {
                    "shoe_vertical_velocity_m_s": -0.003,
                    "shoe_free_axis_angular_velocity_rad_s": 0.2,
                    "shoe_angular_velocity_norm_rad_s": 0.25,
                },
            ]
        )

        self.assertAlmostEqual(metrics["max_abs_shoe_vertical_velocity_m_s"], 0.003)
        self.assertAlmostEqual(metrics["rms_shoe_vertical_velocity_m_s"], (0.003 * 0.003 / 2.0) ** 0.5)
        self.assertAlmostEqual(metrics["max_abs_shoe_free_axis_angular_velocity_rad_s"], 0.2)
        self.assertAlmostEqual(metrics["max_shoe_angular_velocity_norm_rad_s"], 0.25)

    def test_force_jump_failure_is_opt_in(self):
        args = core.build_arg_parser().parse_args(core._normalize_cli_argv(["run"]))

        self.assertFalse(args.fail_on_force_jump)

    def test_goal_objective_cli_flags_are_available(self):
        args = core.build_arg_parser().parse_args(
            core._normalize_cli_argv(
                [
                    "calibrate",
                    "--calibration-objective",
                    "goal",
                    "--goal-shoe-vz-target",
                    "0.002",
                    "--goal-shoe-omega-target",
                    "0.03",
                ]
            )
        )

        self.assertEqual(args.calibration_objective, "goal")
        self.assertEqual(args.goal_shoe_vz_target, 0.002)
        self.assertEqual(args.goal_shoe_omega_target, 0.03)

    def test_calibration_early_out_reports_unstable_force_jump(self):
        args = argparse.Namespace(
            calibration_early_out=True,
            calibration_early_out_force_multiplier=0.0,
            calibration_early_out_force_jump_limit=50.0,
            calibration_early_out_max_shoe_vz=0.0,
            calibration_early_out_max_shoe_omega=0.0,
        )

        reason = core._calibration_early_out_reason(
            args,
            test_case="rearfoot",
            measured_force_n=core.np.asarray([0.0, 100.0, 0.0]),
            contact_force_n=core.np.asarray([0.0, 120.0, 0.0]),
            contact_stats=[{}, {}],
            sample_index=1,
        )

        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("candidate became unstable", reason)
        self.assertIn("force jump", reason)

    def test_coordinate_material_search_walks_toward_better_candidate(self):
        args = argparse.Namespace(
            fit_kh=True,
            fit_kh_min=1.0,
            fit_kh_max=9.0,
            fit_kh_steps=3,
            kh=5.0,
            fit_kd=False,
            fit_kd_min=0.0,
            fit_kd_max=10.0,
            fit_kd_steps=2,
            kd=4.0,
            layer_eta_lock=0.5,
            layer_densification_power=0.25,
            fit_layer_eta_lock=False,
            fit_layer_eta_lock_min=0.45,
            fit_layer_eta_lock_max=0.85,
            fit_layer_eta_lock_steps=3,
            fit_layer_densification_power=False,
            fit_layer_densification_power_min=0.0,
            fit_layer_densification_power_max=2.0,
            fit_layer_densification_power_steps=3,
            calibration_kh_spacing="linear",
            calibration_search_method="coordinate",
            calibration_batch_size=8,
            calibration_search_sample_stride=4,
            calibration_optimizer_iterations=3,
            calibration_optimizer_shrink=0.5,
        )

        original_evaluate = core._evaluate_material_candidate

        def fake_evaluate(_args, candidate):
            objective = abs(candidate.kh - 7.0)
            material = {
                "kh": candidate.kh,
                "kd": candidate.kd,
                "layer_eta_lock": candidate.layer_eta_lock,
                "layer_densification_power": candidate.layer_densification_power,
                "combined_objective": float(objective),
            }
            return material, []

        core._evaluate_material_candidate = fake_evaluate
        try:
            result = core._search_material_candidates(args, workers=1)
        finally:
            core._evaluate_material_candidate = original_evaluate

        self.assertAlmostEqual(result.material["kh"], 7.0)
        self.assertLess(result.candidate_count, 9)
        self.assertEqual(result.history[0]["method"], "coordinate")

    def test_rearfoot_candidate_batches_use_multiworld_dispatch(self):
        args = argparse.Namespace(
            calibration_cases="rearfoot",
            calibration_workers=1,
            calibration_batch_size=2,
            calibration_search_sample_stride=4,
        )
        candidates = [
            core.MaterialCandidate(kh=1.0, kd=2.0, layer_eta_lock=0.65, layer_densification_power=0.0),
            core.MaterialCandidate(kh=3.0, kd=4.0, layer_eta_lock=0.70, layer_densification_power=0.5),
        ]
        self.assertTrue(core._can_evaluate_batch_multiworld(args, candidates))

        original_batch = core._evaluate_material_candidate_batch_multiworld
        original_single = core._evaluate_material_candidate
        calls = {"batch": 0, "single": 0}

        def fake_batch(_args, batch):
            calls["batch"] += 1
            return [
                (
                    {
                        "kh": candidate.kh,
                        "kd": candidate.kd,
                        "layer_eta_lock": candidate.layer_eta_lock,
                        "layer_densification_power": candidate.layer_densification_power,
                        "combined_objective": float(index),
                    },
                    [],
                )
                for index, candidate in enumerate(batch)
            ]

        def fake_single(_args, _candidate):
            calls["single"] += 1
            return {"combined_objective": 0.0}, []

        core._evaluate_material_candidate_batch_multiworld = fake_batch
        core._evaluate_material_candidate = fake_single
        try:
            results = core._evaluate_candidate_batches(args, candidates, workers=1)
        finally:
            core._evaluate_material_candidate_batch_multiworld = original_batch
            core._evaluate_material_candidate = original_single

        self.assertEqual(len(results), 2)
        self.assertEqual(calls["batch"], 1)
        self.assertEqual(calls["single"], 0)

    def test_multiworld_batching_supports_rearfoot_and_fullfoot(self):
        args = argparse.Namespace(calibration_cases="rearfoot,fullfoot", calibration_workers=1)
        candidates = [
            core.MaterialCandidate(kh=1.0, kd=2.0, layer_eta_lock=0.65, layer_densification_power=0.0),
            core.MaterialCandidate(kh=3.0, kd=4.0, layer_eta_lock=0.70, layer_densification_power=0.5),
        ]

        self.assertTrue(core._can_evaluate_batch_multiworld(args, candidates))

    def test_multiworld_batching_rejects_unknown_cases(self):
        args = argparse.Namespace(calibration_cases="rearfoot,toebox", calibration_workers=1)
        candidates = [
            core.MaterialCandidate(kh=1.0, kd=2.0, layer_eta_lock=0.65, layer_densification_power=0.0),
            core.MaterialCandidate(kh=3.0, kd=4.0, layer_eta_lock=0.70, layer_densification_power=0.5),
        ]

        self.assertFalse(core._can_evaluate_batch_multiworld(args, candidates))

    def test_trial_evaluation_reports_r2_and_residual_motion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            response = core.TrialResponse(
                test_case="rearfoot",
                trace_csv="trace.csv",
                output_csv=str(Path(tmpdir) / "out.csv"),
                trace={"phase": core.np.asarray([0.0, 0.5, 1.0])},
                displacement_m=core.np.asarray([0.0, 0.01, 0.0]),
                time_s=core.np.asarray([0.0, 0.1, 0.2]),
                displacement_velocity_m_s=core.np.asarray([0.0, 0.1, -0.1]),
                measured_force_n=core.np.asarray([0.0, 100.0, 0.0]),
                raw_contact_force_n=core.np.asarray([0.0, 90.0, 10.0]),
                contact_stats=[
                    {
                        "rigid_contact_count": 1,
                        "pair_contact_count": 1,
                        "active_pair_contact_count": 1,
                        "top_contact_force_n": 0.0,
                        "bottom_contact_force_n": 0.0,
                        "bottom_active_pair_contact_count": 0,
                        "pair_depth_min_m": 0.0,
                        "pair_stiffness_sum": 0.0,
                        "shoe_vertical_velocity_m_s": 0.0,
                        "shoe_free_axis_angular_velocity_rad_s": 0.0,
                        "shoe_angular_velocity_norm_rad_s": 0.0,
                    },
                    {
                        "rigid_contact_count": 1,
                        "pair_contact_count": 1,
                        "active_pair_contact_count": 1,
                        "top_contact_force_n": 90.0,
                        "bottom_contact_force_n": 0.0,
                        "bottom_active_pair_contact_count": 0,
                        "pair_depth_min_m": -0.001,
                        "pair_stiffness_sum": 1.0,
                        "shoe_vertical_velocity_m_s": -0.003,
                        "shoe_free_axis_angular_velocity_rad_s": 0.2,
                        "shoe_angular_velocity_norm_rad_s": 0.25,
                    },
                    {
                        "rigid_contact_count": 1,
                        "pair_contact_count": 1,
                        "active_pair_contact_count": 1,
                        "top_contact_force_n": 10.0,
                        "bottom_contact_force_n": 0.0,
                        "bottom_active_pair_contact_count": 0,
                        "pair_depth_min_m": -0.001,
                        "pair_stiffness_sum": 1.0,
                        "shoe_vertical_velocity_m_s": 0.001,
                        "shoe_free_axis_angular_velocity_rad_s": -0.1,
                        "shoe_angular_velocity_norm_rad_s": 0.1,
                    },
                ],
                midsole_stats={},
            )
            args = argparse.Namespace(
                kh=42.0,
                kd=2.0,
                force_jump_limit=200.0,
                fail_on_force_jump=False,
            )

            summary = core._evaluate_trial_response(response, args)

        self.assertIn("force_r2", summary)
        self.assertAlmostEqual(summary["max_abs_shoe_vertical_velocity_m_s"], 0.003)
        self.assertAlmostEqual(summary["max_abs_shoe_free_axis_angular_velocity_rad_s"], 0.2)
        self.assertAlmostEqual(summary["max_shoe_angular_velocity_norm_rad_s"], 0.25)

    def test_trial_evaluation_fails_on_large_force_jump(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            response = core.TrialResponse(
                test_case="fullfoot",
                trace_csv="trace.csv",
                output_csv=str(Path(tmpdir) / "out.csv"),
                trace={"phase": core.np.asarray([0.0, 0.5, 1.0])},
                displacement_m=core.np.asarray([0.0, 0.01, 0.02]),
                time_s=core.np.asarray([0.0, 0.1, 0.2]),
                displacement_velocity_m_s=core.np.asarray([0.0, 0.1, 0.1]),
                measured_force_n=core.np.asarray([0.0, 100.0, 200.0]),
                raw_contact_force_n=core.np.asarray([0.0, 50.0, 350.0]),
                contact_stats=[
                    {
                        "rigid_contact_count": 1,
                        "pair_contact_count": 1,
                        "active_pair_contact_count": 1,
                        "top_contact_force_n": 0.0,
                        "bottom_contact_force_n": 0.0,
                        "bottom_active_pair_contact_count": 0,
                        "pair_depth_min_m": 0.0,
                        "pair_stiffness_sum": 0.0,
                    },
                    {
                        "rigid_contact_count": 1,
                        "pair_contact_count": 1,
                        "active_pair_contact_count": 1,
                        "top_contact_force_n": 50.0,
                        "bottom_contact_force_n": 0.0,
                        "bottom_active_pair_contact_count": 0,
                        "pair_depth_min_m": -0.001,
                        "pair_stiffness_sum": 1.0,
                    },
                    {
                        "rigid_contact_count": 1,
                        "pair_contact_count": 1,
                        "active_pair_contact_count": 1,
                        "top_contact_force_n": 350.0,
                        "bottom_contact_force_n": 0.0,
                        "bottom_active_pair_contact_count": 0,
                        "pair_depth_min_m": -0.002,
                        "pair_stiffness_sum": 1.0,
                    },
                ],
                midsole_stats={},
            )
            args = argparse.Namespace(
                kh=42.0,
                kd=2.0,
                force_jump_limit=200.0,
                fail_on_force_jump=True,
            )

            with self.assertRaisesRegex(RuntimeError, "adjacent force jumps"):
                core._evaluate_trial_response(response, args)

    def test_material_candidate_force_jump_is_invalid_candidate(self):
        args = argparse.Namespace(
            calibration_search_sample_stride=4,
        )
        candidate = core.MaterialCandidate(kh=1.0, kd=2.0, layer_eta_lock=0.65, layer_densification_power=0.0)
        original_simulate = core._simulate_calibration_responses

        def fake_simulate(_args, *, sample_stride):
            raise RuntimeError("fullfoot generated 1 adjacent force jumps above 200 N.")

        core._simulate_calibration_responses = fake_simulate
        try:
            material, responses = core._evaluate_material_candidate(args, candidate)
        finally:
            core._simulate_calibration_responses = original_simulate

        self.assertEqual(responses, [])
        self.assertEqual(material["kh"], 1.0)
        self.assertEqual(material["kd"], 2.0)
        self.assertEqual(material["combined_objective"], float("inf"))
        invalid_reason = material["invalid_reason"]
        assert isinstance(invalid_reason, str)
        self.assertIn("adjacent force jumps", invalid_reason)

    def test_material_candidate_grid_batches(self):
        args = argparse.Namespace(
            fit_kh=True,
            fit_kh_min=1.0,
            fit_kh_max=3.0,
            fit_kh_steps=3,
            kh=9.0,
            fit_kd=True,
            fit_kd_min=0.0,
            fit_kd_max=10.0,
            fit_kd_steps=2,
            kd=4.0,
            layer_eta_lock=0.5,
            layer_densification_power=0.25,
        )

        candidates = core._material_candidate_grid(args)
        batches = core._iter_candidate_batches(candidates, 4)

        self.assertEqual(len(candidates), 6)
        self.assertEqual(
            candidates[0],
            core.MaterialCandidate(kh=1.0, kd=0.0, layer_eta_lock=0.5, layer_densification_power=0.25),
        )
        self.assertEqual(
            candidates[-1],
            core.MaterialCandidate(kh=3.0, kd=10.0, layer_eta_lock=0.5, layer_densification_power=0.25),
        )
        self.assertEqual([len(batch) for batch in batches], [4, 2])

    def test_material_candidate_grid_can_use_log_kh(self):
        args = argparse.Namespace(
            fit_kh=True,
            fit_kh_min=1.0,
            fit_kh_max=100.0,
            fit_kh_steps=3,
            kh=9.0,
            fit_kd=False,
            fit_kd_min=0.0,
            fit_kd_max=10.0,
            fit_kd_steps=2,
            kd=4.0,
            layer_eta_lock=0.5,
            layer_densification_power=0.25,
            calibration_kh_spacing="log",
        )

        candidates = core._material_candidate_grid(args)

        self.assertEqual([candidate.kh for candidate in candidates], [1.0, 10.0, 100.0])
        self.assertEqual([candidate.kd for candidate in candidates], [4.0, 4.0, 4.0])

    def test_adaptive_material_search_refines_around_best_candidate(self):
        args = argparse.Namespace(
            fit_kh=True,
            fit_kh_min=1.0,
            fit_kh_max=100.0,
            fit_kh_steps=3,
            kh=9.0,
            fit_kd=False,
            fit_kd_min=0.0,
            fit_kd_max=10.0,
            fit_kd_steps=2,
            kd=4.0,
            layer_eta_lock=0.5,
            layer_densification_power=0.25,
            calibration_kh_spacing="log",
            calibration_search_method="adaptive",
            calibration_refine_stages=2,
            calibration_refine_radius=0.5,
            calibration_batch_size=8,
            calibration_search_sample_stride=4,
        )

        original_evaluate = core._evaluate_material_candidate

        def fake_evaluate(_args, candidate):
            objective = abs(core.np.log(candidate.kh) - core.np.log(10.0))
            material = {
                "kh": candidate.kh,
                "kd": candidate.kd,
                "combined_objective": float(objective),
            }
            return material, []

        core._evaluate_material_candidate = fake_evaluate
        try:
            result = core._search_material_candidates(args, workers=1)
        finally:
            core._evaluate_material_candidate = original_evaluate

        self.assertAlmostEqual(result.material["kh"], 10.0)
        self.assertEqual(result.candidate_count, 5)
        self.assertEqual([stage["candidate_count"] for stage in result.history], [3, 2])

    def test_material_search_reports_zero_contact_reason_when_all_candidates_invalid(self):
        args = argparse.Namespace(
            fit_kh=True,
            fit_kh_min=1.0,
            fit_kh_max=10.0,
            fit_kh_steps=2,
            kh=1.0,
            fit_kd=False,
            fit_kd_min=0.0,
            fit_kd_max=10.0,
            fit_kd_steps=2,
            kd=4.0,
            layer_eta_lock=0.5,
            layer_densification_power=0.25,
            calibration_kh_spacing="linear",
            calibration_search_method="grid",
            calibration_refine_stages=1,
            calibration_refine_radius=0.5,
            calibration_batch_size=8,
            calibration_search_sample_stride=4,
        )

        original_evaluate = core._evaluate_material_candidate

        def fake_evaluate(_args, candidate):
            return {
                "kh": candidate.kh,
                "kd": candidate.kd,
                "combined_objective": float("inf"),
                "invalid_reason": "fullfoot generated no raw pressure-field contact force.",
            }, []

        core._evaluate_material_candidate = fake_evaluate
        try:
            with self.assertRaisesRegex(RuntimeError, "fullfoot generated no raw pressure-field contact force"):
                core._search_material_candidates(args, workers=1)
        finally:
            core._evaluate_material_candidate = original_evaluate

    def test_trace_stride_keeps_endpoints_and_peak(self):
        trace = {
            "displacement_m": core.np.asarray([0.0, 0.1, 0.3, 0.2, 0.0]),
            "time_s": core.np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
            "force_n": core.np.asarray([0.0, 10.0, 30.0, 20.0, 0.0]),
        }

        strided, indices = core._stride_trace(trace, 4)

        self.assertEqual(indices.tolist(), [0, 2, 4])
        self.assertEqual(strided["displacement_m"].tolist(), [0.0, 0.3, 0.0])

    def test_fullfoot_auto_contact_search_uses_requested_range(self):
        args = argparse.Namespace(
            fullfoot_contact_search_max=0.20,
            fullfoot_start_clearance=0.002,
            fullfoot_z_offset=0.0,
        )

        search_max, rest_clearance = core._fullfoot_auto_contact_search_limit(args)

        self.assertEqual(rest_clearance, 0.002)
        self.assertEqual(search_max, 0.20)

    def test_default_fullfoot_start_clearance_places_indenter_at_contact(self):
        args = core.build_arg_parser().parse_args(core._normalize_cli_argv(["view", "--test-case", "fullfoot"]))

        self.assertEqual(args.fullfoot_start_clearance, 0.0)

    def test_indenter_rest_height_tracks_settled_shoe_height(self):
        args = argparse.Namespace(
            fullfoot_start_clearance=0.0,
            fullfoot_z_offset=0.0,
            punch_half_height=0.025,
        )

        self.assertAlmostEqual(core._fullfoot_indenter_rest_z(args, shoe_z=-0.01, midsole_top=0.07), 0.06)
        self.assertAlmostEqual(core._rearfoot_indenter_rest_z(args, shoe_z=-0.01, local_top_z=0.05), 0.065)

    def test_platen_stop_accounts_for_indenter_reference_frame(self):
        args = argparse.Namespace(
            fixture_platen_stop_clearance=1.0e-4,
            punch_half_height=0.025,
        )

        self.assertAlmostEqual(
            core._indenter_platen_stop_z(args, test_case="fullfoot", base_top_z=0.0),
            1.0e-4,
        )
        self.assertAlmostEqual(
            core._indenter_platen_stop_z(args, test_case="rearfoot", base_top_z=0.0),
            0.0251,
        )

    def test_indenter_command_clamps_at_platen_stop(self):
        z, vz, active = core._clamp_indenter_to_platen_stop(-0.01, -2.0, 1.0e-4)

        self.assertEqual(active, 1)
        self.assertEqual(z, 1.0e-4)
        self.assertEqual(vz, 0.0)

    def test_instron_fixture_defaults_to_dynamic_level_specimen(self):
        args = core.build_arg_parser().parse_args(core._normalize_cli_argv(["view-calibration"]))

        self.assertTrue(args.fixture_dynamic_shoe)
        self.assertEqual(args.fixture_free_rotation_axis, "none")
        self.assertEqual(args.fixture_bottom_platen_overlap, 0.0)

    def test_indenter_substep_commands_ramp_between_samples(self):
        commands = core._indenter_substep_commands(
            start_z=0.04,
            target_z=0.0,
            trace_velocity_z=-1.0,
            dt=0.2,
            substeps=4,
            stop_z=-1.0,
        )

        self.assertEqual([round(command[0], 6) for command in commands], [0.03, 0.02, 0.01, 0.0])
        self.assertEqual([round(command[1], 6) for command in commands], [-0.2, -0.2, -0.2, -0.2])

    def test_indenter_substep_commands_respect_platen_stop(self):
        commands = core._indenter_substep_commands(
            start_z=0.04,
            target_z=-0.04,
            trace_velocity_z=-1.0,
            dt=0.2,
            substeps=4,
            stop_z=0.01,
        )

        self.assertEqual(commands[-1], (0.01, 0.0, 1))

    def test_fixture_crop_removes_high_indenter_post_triangles(self):
        vertices = core.np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.02],
                [0.0, 0.0, 0.12],
                [1.0, 0.0, 0.12],
                [0.0, 1.0, 0.13],
            ],
            dtype=core.np.float32,
        )
        faces = core.np.asarray([[0, 1, 2], [3, 4, 5]], dtype=core.np.int32)

        cropped_vertices, cropped_faces, stats = core._crop_mesh_by_vertex_z_max(vertices, faces, 0.08)

        self.assertEqual(cropped_faces.shape[0], 1)
        self.assertEqual(stats["crop_removed_triangles"], 1)
        self.assertLessEqual(float(core.np.max(cropped_vertices[:, 2])), 0.08)

    def test_calibration_cli_report_highlights_final_material(self):
        report = core._format_cli_report(
            {
                "mode": "calibrate",
                "material_json": "material.json",
                "bundle_json": "bundle.json",
                "plot_output": "fit.png",
                "candidate_count": 12,
                "calibration_cases": ["rearfoot", "fullfoot"],
                "calibration_search_sample_stride": 4,
                "calibration_final_sample_stride": 1,
                "calibration_search_method": "adaptive",
                "calibration_kh_spacing": "log",
                "calibration_refine_stages": 2,
                "calibration_search_history": [
                    {
                        "stage": 1,
                        "best_kh": 5.0e6,
                        "best_kd": 0.0,
                        "best_objective": 55.0,
                        "candidate_count": 15,
                    }
                ],
                "calibration_batch_size": 4,
                "calibration_workers": 1,
                "parameters": {
                    "kh": 6.085e6,
                    "kd": 12.5,
                    "combined_objective": 34.0,
                },
                "trials": {
                    "fullfoot": {
                        "loop_rmse_n": 120.0,
                        "force_r2": 0.95,
                        "peak_relative_error": 0.05,
                        "sim_peak_force_n": 1900.0,
                    }
                },
            }
        )

        self.assertIn("final kh: 6.085e+06 N/m", report)
        self.assertIn("final kd: 12.5", report)
        self.assertIn("cases: rearfoot, fullfoot", report)
        self.assertIn("search: adaptive (log kh, 2 stages)", report)
        self.assertIn("1: kh 5e+06, kd 0, objective 55", report)
        self.assertIn("sample stride: search 4, final 1", report)
        self.assertIn("fullfoot: RMSE 120 N", report)
        self.assertIn("R2 0.95", report)
        self.assertIn("material: material.json", report)

    def test_report_json_flag_is_available(self):
        args = core.build_arg_parser().parse_args(core._normalize_cli_argv(["calibrate", "--report-json"]))

        self.assertTrue(args.report_json)


if __name__ == "__main__":
    unittest.main(verbosity=2)
