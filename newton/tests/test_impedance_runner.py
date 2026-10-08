# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify causal generative impedance, conservation, and shared identification."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from newton.tests.test_impedance_hogan import _chain, _reference, _tiny_shoe
from projects.impedance_instron.cartesian.prepare_dataset import _flight_velocity
from projects.impedance_instron.cartesian.profile import validate
from projects.impedance_instron.hogan.generate import main as generate_main
from projects.impedance_instron.hogan.identify import (
    FitConfig,
    Parameterization,
    Trial,
    coordinates,
    evaluate,
    fit,
    initialize,
    load_trials,
    main,
    predict,
    score,
)
from projects.impedance_instron.hogan.runner import (
    _SCHEMA_1_FEATURES,
    FEATURE_NAMES,
    Bounds,
    RolloutConfig,
    Runner,
    State,
    Task,
    simulate,
)


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value.copy()


class _Shoe:
    """Supply a controlled external wrench for dynamics tests, not a shoe fit."""

    def __init__(self, force=(0.0, 0.0, 0.0)):
        self.force = np.asarray(force)
        self.resets = 0
        self.foundation = SimpleNamespace(driven=_Array([1]), compression=_Array([0.0]), reset=self.reset)
        self.shoe = SimpleNamespace(column_bed=SimpleNamespace(rest_length_m=np.array([0.02])))

    def reset(self):
        self.resets += 1

    def apply(self, ankle_m, velocity_m_s, pitch_rad, angular_velocity_rad_s, dt):
        return self.force.copy(), 0.0


def _initial():
    return State(np.array([0.1, 1.1, 1.4, 0.3, -0.5, 0.2]), np.array([3.0, -0.1, 0.1, 0.2, -0.3, 0.1]))


def _trial(split="train", speed=3.0):
    initial = _initial()
    t = np.linspace(0, 0.003, 4)
    return Trial(
        split,
        split,
        _chain(),
        _Shoe(),
        Task(speed),
        initial,
        t,
        np.tile(initial.q, (4, 1)),
        t.copy(),
        np.zeros((4, 2)),
        {"compatibility": {"passed": True}},
    )


class TestGenerativeRunner(unittest.TestCase):
    """Check the physical and causal boundary without motion datasets."""

    def test_joint_only_bounded_actuation(self):
        """Keep base actuation zero and joint torque and slew within bounds."""
        runner = Runner.seed()
        state = _initial()
        state.q[3:] = [8, -8, 8]
        state.v[3:] = [100, -100, 100]
        before = state.copy()
        load, info = runner.actuate(state, _chain(), Task(3.0), 0.001)
        np.testing.assert_array_equal(load[:3], 0)
        self.assertTrue(info["torque_saturated"].all())
        self.assertTrue(np.all(abs(load[3:]) <= runner.bounds.torque_max_nm))
        self.assertTrue(np.all(abs(load[3:]) <= np.array(runner.bounds.torque_rate_max_nm_s) * 0.001))
        np.testing.assert_array_equal(state.q, before.q)
        np.testing.assert_array_equal(state.v, before.v)

    def test_intrinsic_damping_bypasses_torque_lag(self):
        """Apply lag-free damping in the first step without changing the activation state."""
        damping = np.array([2.0, 3.0, 1.0])
        plain = Runner.seed()
        damped = Runner.from_dict({**plain.to_dict(), "intrinsic_damping_nms_rad": damping.tolist()})
        np.testing.assert_array_equal(plain.intrinsic_damping_nms_rad, 0)
        first, second = _initial(), _initial()
        base, _ = plain.actuate(first, _chain(), Task(3.0), 1e-4)
        load, _ = damped.actuate(second, _chain(), Task(3.0), 1e-4)
        np.testing.assert_allclose(load[3:], base[3:] - damping * second.v[3:], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(first.torque_nm, second.torque_nm)
        np.testing.assert_array_equal(load[:3], 0)
        second.v[3:] = [1e4, -1e4, 1e4]
        load, _ = damped.actuate(second, _chain(), Task(3.0), 1e-4)
        np.testing.assert_allclose(np.abs(load[3:]), damped.bounds.torque_max_nm)
        self.assertEqual(Runner.from_dict(damped.to_dict()).to_dict(), damped.to_dict())
        with self.assertRaises(ValueError):
            Runner(plain.weights, intrinsic_damping_nms_rad=[-1.0, 0.0, 0.0])

    def test_variable_impedance_responds_to_simulated_state(self):
        """Change impedance with internal phase and load rather than a reference."""
        runner = Runner.seed()
        state = _initial()
        first = runner.impedance(state, _chain(), Task(3))
        state.phase_rad = 1.0
        state.normal_load_bw = 2.0
        second = runner.impedance(state, _chain(), Task(3))
        self.assertGreater(np.max(abs(first[0] - second[0])), 0.01)
        self.assertGreater(np.max(abs(first[1] - second[1])), 1.0)
        for eq, stiffness, damping in (first, second):
            self.assertTrue(np.all(eq >= runner.bounds.equilibrium_lower_rad))
            self.assertTrue(np.all(eq <= runner.bounds.equilibrium_upper_rad))
            self.assertTrue(np.all(stiffness > 0))
            self.assertTrue(np.all(damping > 0))

    def test_continuous_phase_and_finite_activation_step(self):
        """Bound oscillator increments through abrupt changes of simulated load."""
        runner = Runner.seed()
        state = _initial()
        dt = 0.001
        for force in (0, 1400, 0, 2800, 0):
            old = state.phase_rad
            runner.advance_sensors(state, force, 700, Task(3.7), dt)
            delta = (state.phase_rad - old) % (2 * math.pi)
            self.assertGreater(delta, 0)
            self.assertLessEqual(delta, 2 * math.pi * runner.frequency_hz * 1.2 * dt + 1e-12)
        load, _ = runner.actuate(state, _chain(), Task(3.7), 1.0)
        self.assertTrue(np.isfinite(load).all())

    def test_internal_actuation_conserves_external_momentum_balance(self):
        """Preserve COM acceleration and angular momentum under internal torques."""
        chain, state = _chain(), _initial()
        load, _ = Runner.seed().actuate(state, chain, Task(3.0), 0.001)
        mass, bias = chain.dynamics(state.q, state.v)
        acceleration = np.linalg.solve(mass, load - bias)
        com_acceleration = np.zeros(2)
        for body, m in enumerate(chain.masses_kg):
            _, jacobian, centripetal = chain.point(state.q, body, chain.com_local_m[body], state.v)
            com_acceleration += m * (jacobian @ acceleration + centripetal)
        np.testing.assert_allclose(com_acceleration / chain.total_mass_kg, [0, -9.81], atol=1e-10)

        def angular_momentum(q, v):
            com = chain.com(q)
            result = 0.0
            for body, m in enumerate(chain.masses_kg):
                point, jacobian, _ = chain.point(q, body, chain.com_local_m[body])
                r, speed = point - com, jacobian @ v
                result += m * (r[0] * speed[1] - r[1] * speed[0])
                result += chain.inertias_kg_m2[body] * (chain.angular_jacobian(body) @ v)
            return result

        eps = 1e-6
        rate = (
            angular_momentum(state.q + eps * state.v, state.v + eps * acceleration)
            - angular_momentum(state.q - eps * state.v, state.v - eps * acceleration)
        ) / (2 * eps)
        self.assertAlmostEqual(rate, 0.0, delta=1e-7)

    def test_rollout_reset_and_input_isolation(self):
        """Repeat free predictions exactly without mutating initial state or model."""
        runner, initial, shoe = Runner.seed(), _initial(), _Shoe()
        before = initial.copy()
        first, summary = simulate(runner, _chain(), shoe, initial, Task(3), duration_s=0.01)
        second, _ = simulate(runner, _chain(), shoe, initial, Task(3), duration_s=0.01)
        self.assertEqual(summary["status"], "completed")
        self.assertFalse(summary["reference_inputs_used"])
        self.assertEqual(shoe.resets, 4)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(initial.q, before.q)
        np.testing.assert_array_equal(initial.torque_nm, before.torque_nm)
        np.testing.assert_array_equal(first["load"][:, :3], 0)
        self.assertEqual(len(first["time_s"]), len(first["grf_n"]) + 1)

    def test_future_targets_cannot_change_prediction(self):
        """Keep predictions identical after arbitrary changes to future target data."""
        trial = _trial()
        runner, cfg = Runner.seed(), RolloutConfig()
        first, _ = predict(runner, trial, cfg)
        # Keep horizon and predictive inputs fixed, corrupt all fitting targets.
        trial.q[:] = 1000
        trial.grf_n[:] = 9000
        trial.force_time_s[:] = -100
        trial.time_s[1:-1] = 0.00001
        second, _ = predict(runner, trial, cfg)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_failure_is_explicit_and_reset(self):
        """Report rejected contact without fabricating a completed interval."""
        shoe = _Shoe((0, -1, 0))
        trace, summary = simulate(Runner.seed(), _chain(), shoe, _initial(), Task(3), duration_s=0.01)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(trace["load"].shape, (0, 6))
        self.assertEqual(summary["integrated_duration_s"], 0)
        self.assertEqual(shoe.resets, 2)

    def test_real_shoe_runs_without_plan(self):
        """Integrate the existing shoe interface without inverse dynamics or targets."""
        with tempfile.TemporaryDirectory() as directory:
            shoe = _tiny_shoe(directory)
            trace, summary = simulate(Runner.seed(), _chain(), shoe, _initial(), Task(3), duration_s=0.005)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(np.isfinite(trace["state"]).all())
        self.assertFalse(summary["validated"])

    def test_portable_round_trip_and_validation(self):
        """Persist model semantics and reject nonfinite or incompatible inputs."""
        runner = Runner.seed()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runner.json"
            runner.save(path)
            loaded = Runner.load(path)
        self.assertEqual(loaded.to_dict(), runner.to_dict())
        for build in (
            lambda: Task(float("nan")),
            lambda: RolloutConfig(dt_s=0),
            lambda: Bounds(torque_max_nm=(1, -1, 1)),
            lambda: Runner(np.zeros((3, 3, 7))),
        ):
            with self.assertRaises(ValueError):
                build()
        data = runner.to_dict()
        data["features"][0] = "future_force"
        with self.assertRaises(ValueError):
            Runner.from_dict(data)

    def test_schema_1_model_loads_with_identical_impedance(self):
        """Load an eight-feature model exactly by zeroing the added feature weights."""
        runner = Runner(Runner.seed().weights + np.random.default_rng(4).normal(0, 0.3, (3, 3, len(FEATURE_NAMES))))
        data = runner.to_dict()
        old = [FEATURE_NAMES.index(name) for name in _SCHEMA_1_FEATURES]
        added = [i for i in range(len(FEATURE_NAMES)) if i not in old]
        data["weights"] = np.asarray(data["weights"])[:, :, old].tolist()
        data["schema"], data["features"] = "generative_runner_1", list(_SCHEMA_1_FEATURES)
        loaded = Runner.from_dict(data)
        np.testing.assert_array_equal(loaded.weights[:, :, added], 0)
        state = _initial()
        state.phase_rad, state.normal_load_bw = 2.3, 1.4
        features = loaded.features(state, _chain(), Task(3))
        # Schema-1 logits used exactly the retained features with the stored weights.
        np.testing.assert_allclose(
            loaded.weights @ features, np.asarray(data["weights"]) @ features[old], rtol=0, atol=1e-14
        )

    def test_harmonic_and_load_phase_features(self):
        """Evaluate higher phase harmonics and load-gated phase from internal state only."""
        state = _initial()
        state.phase_rad, state.normal_load_bw = 0.7, 1.5
        features = dict(zip(FEATURE_NAMES, Runner.seed().features(state, _chain(), Task(3)), strict=True))
        for k in (2, 3):
            self.assertAlmostEqual(features[f"sin_{k}phase"], math.sin(k * 0.7))
            self.assertAlmostEqual(features[f"cos_{k}phase"], math.cos(k * 0.7))
        self.assertAlmostEqual(features["load_sin_phase"], 1.5 * math.sin(0.7))
        self.assertAlmostEqual(features["load_cos_phase"], 1.5 * math.cos(0.7))
        state.normal_load_bw = 9.0
        features = dict(zip(FEATURE_NAMES, Runner.seed().features(state, _chain(), Task(3)), strict=True))
        self.assertAlmostEqual(features["load_cos_phase"], 4.0 * math.cos(0.7))


class TestRunnerIdentification(unittest.TestCase):
    """Keep observations outside prediction and held-out data outside selection."""

    def test_backward_prefix_initialization(self):
        """Differentiate a quadratic at the prefix endpoint using only observed frames."""
        t = np.array([0.0, 0.004, 0.011])
        q = np.tile(_initial().q, (3, 1)) + 2 * t[:, None] + 3 * t[:, None] ** 2
        state = initialize(t, q)
        np.testing.assert_allclose(state.v, 2 + 6 * t[-1], atol=1e-10)
        np.testing.assert_array_equal(state.q, q[-1])
        with self.assertRaises(ValueError):
            initialize(t[:2], q[:2])

    def test_flight_com_velocity_sets_model_com(self):
        """Choose the hip velocity so the model COM moves at the supplied flight velocity."""
        t = np.array([0.0, 0.005, 0.010])
        q = np.tile(_initial().q, (3, 1)) + 2 * t[:, None]
        chain = _chain()
        state = initialize(t, q, com_velocity_m_s=[3.6, 0.05], chain=chain)
        np.testing.assert_allclose(chain.com_jacobian(state.q) @ state.v, [3.6, 0.05], atol=1e-10)
        np.testing.assert_allclose(state.v[2:], 2.0, atol=1e-10)
        step = 1e-6
        moved = chain.com(state.q + step * state.v) - chain.com(state.q - step * state.v)
        np.testing.assert_allclose(moved / (2 * step), [3.6, 0.05], atol=1e-6)
        with self.assertRaises(ValueError):
            initialize(t, q, com_velocity_m_s=[3.6, 0.05])
        with self.assertRaises(ValueError):
            initialize(t, q, com_velocity_m_s=[3.6], chain=chain)

    def test_flight_velocity_from_preceding_stride(self):
        """Recover a known belt-frame hip velocity from plate force and earlier hip positions only."""
        mass, belt, drift, rise = 60.0, 3.65, 0.02, 0.3
        step, contact = 0.33, 0.2
        fine = np.arange(0.0, 2.0, 1e-5)
        local = np.mod(fine, step)
        peak = mass * 9.81 * step * math.pi / (2 * contact)
        force_z = np.where(local < contact, peak * np.sin(math.pi * local / contact), 0.0)
        acceleration = force_z / mass - 9.81
        velocity_z = rise + np.concatenate(([0.0], np.cumsum(0.5 * (acceleration[1:] + acceleration[:-1]) * 1e-5)))
        height = 1.0 + np.concatenate(([0.0], np.cumsum(0.5 * (velocity_z[1:] + velocity_z[:-1]) * 1e-5)))
        marker_time = np.arange(0.0, 2.0, 0.005)
        analog_time = np.arange(0.0, 2.0, 0.001)
        trial = SimpleNamespace(
            marker_time_s=marker_time,
            analog_time_s=analog_time,
            force_n=np.column_stack(
                (np.zeros_like(analog_time), np.zeros_like(analog_time), np.interp(analog_time, fine, force_z))
            ),
        )
        hip = np.column_stack((drift * marker_time, np.interp(marker_time, fine, height)))
        origin = int(np.searchsorted(marker_time, 4 * step - 0.05))
        estimate = _flight_velocity(trial, hip, origin, mass, belt)
        expected_z = np.interp(marker_time[origin], fine, velocity_z)
        np.testing.assert_allclose(estimate, [belt + drift, expected_z], atol=5e-3)
        self.assertIsNone(_flight_velocity(trial, hip, int(np.searchsorted(marker_time, step)), mass, belt))

    def test_coordinate_conversion_ignores_grf_and_exported_velocity(self):
        """Retain measured geometry rather than adapting it to measured force."""
        ref = _reference()
        ref["pelvis_target_rad"] = np.full(len(ref["time_s"]), math.pi / 2)
        first = coordinates(ref)
        ref["grf_target_n"][:] = 9999
        ref["velocity"][:] = 9999
        np.testing.assert_array_equal(coordinates(ref), first)
        del ref["pelvis_target_rad"]
        with self.assertRaises(ValueError):
            coordinates(ref)

    def test_coordinates_reach_measured_ankle(self):
        """Re-solve hip and knee so fixed-length FK lands on the measured ankle without turning the foot."""
        ref = _reference(frames=4)
        ref["pelvis_target_rad"] = np.full(4, math.pi / 2)
        ref["state"][:, 3] = -0.4
        chain = _chain()
        angles = coordinates(ref)
        # A measured leg shorter than the angles imply mimics the F01 joint-center mismatch.
        ref["ankle_target_m"] = np.array([chain.point(row, 3, np.zeros(2))[0] for row in angles]) + np.array(
            [0.01, 0.02]
        )
        q = coordinates(ref, chain, height_offset_m=0.003)
        lift = np.array([0.0, 0.003])
        for row, angle_row, ankle in zip(q, angles, ref["ankle_target_m"], strict=True):
            np.testing.assert_allclose(chain.point(row, 3, np.zeros(2))[0], ankle + lift, atol=1e-12)
            self.assertAlmostEqual(chain.angle(row, 3), chain.angle(angle_row, 3), places=12)
        np.testing.assert_array_equal(q[:, :3], angles[:, :3] + [0, 0.003, 0])
        ref["grf_target_n"][:] = 9999
        np.testing.assert_array_equal(coordinates(ref, chain, height_offset_m=0.003), q)

    def test_single_speed_freezes_unidentified_speed_parameters(self):
        """Freeze task-speed coefficients until multiple training speeds are present."""
        baseline = Runner.seed()
        fixed = Parameterization(baseline, [3.7, 3.7])
        changed = fixed.model(np.full(fixed.size, 0.1))
        np.testing.assert_array_equal(changed.weights[:, :, -1], baseline.weights[:, :, -1])
        self.assertEqual(changed.cadence_speed_gain, baseline.cadence_speed_gain)
        variable = Parameterization(baseline, [3.0, 4.0])
        changed = variable.model(np.full(variable.size, 0.1))
        self.assertTrue(np.all(changed.weights[:, :, -1] != baseline.weights[:, :, -1]))
        self.assertGreater(variable.size, fixed.size)

    def test_measured_targets_change_loss_not_dynamics(self):
        """Use future motion and force only in the offline score."""
        trial, runner, cfg = _trial(), Runner.seed(), RolloutConfig()
        trace, summary = predict(runner, trial, cfg)
        first = score(trace, summary, trial, runner)
        trial.q[1:, 3] += 0.3
        trial.grf_n[:, 1] += 1000
        second = score(trace, summary, trial, runner)
        self.assertGreater(second["loss"], first["loss"])

    def test_shared_fit_keeps_best_and_excludes_evaluation_selection(self):
        """Select a shared model on train alone regardless of held-out targets."""
        baseline = Runner.seed()
        training, held_out = _trial(), _trial("eval")
        search = FitConfig(population=6, generations=3, seed=3)

        def objective(model, trials, config, **kwargs):
            target = 0.2 if trials[0].split == "train" else float(trials[0].q[0, 0])
            error = model.weights[0, 0, 0] - baseline.weights[0, 0, 0] - target
            return {"mean_loss": float(error**2), "failed": 0, "trials": []}

        with patch("projects.impedance_instron.hogan.identify.evaluate", side_effect=objective):
            learned, report = fit(baseline, [training, held_out], search=search)
            held_out.q[:] = 9000
            other, _ = fit(baseline, [training, held_out], search=search)
        np.testing.assert_array_equal(learned.weights, other.weights)
        self.assertLess(report["splits"]["train"]["learned"]["mean_loss"], 0.04)
        self.assertEqual(report["selection_split"], "train")

    def test_incompatible_data_is_not_silently_fitted(self):
        """Refuse fitting when input compatibility fails without an explicit override."""
        trial = _trial()
        trial.provenance["compatibility"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "compatibility"):
            fit(Runner.seed(), [trial])

    def test_short_end_to_end_fit(self):
        """Run forward identification and held-out evaluation without a tracking plan."""
        trials = [_trial(), _trial("eval")]
        _, report = fit(Runner.seed(), trials, search=FitConfig(population=4, generations=1))
        self.assertFalse(report["reference_inputs_used"])
        self.assertFalse(report["validated"])
        self.assertEqual(report["splits"]["train"]["learned"]["failed"], 0)
        self.assertEqual(len(report["splits"]["eval"]["learned"]["trials"]), 1)
        json.dumps(report, allow_nan=False)

    def test_failed_rollout_has_unfinished_penalty(self):
        """Penalize an immediate failure rather than dropping its missing samples."""
        trial = _trial()
        trial.shoe = _Shoe((0, -10, 0))
        result = evaluate(Runner.seed(), [trial], RolloutConfig())
        self.assertEqual(result["failed"], 1)
        self.assertGreaterEqual(result["mean_loss"], 2000)

    def test_multicondition_loader_and_cli(self):
        """Load member-specific shoes and speeds and preserve causal initialization."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tiny_shoe(directory)
            # This test owns a full minimal profile rather than relying on local output artifacts.
            profile = {
                "schema": "cartesian_single_leg_1",
                "masses_kg": [8.66, 3.45, 1.46],
                "com_local_m": [[0.17, 0], [0.17, 0], [0.05, 0]],
                "inertias_kg_m2": [0.15, 0.05, 0.008],
                "hip_stiffness_n_m": [1, 1],
                "hip_damping_ns_m": [1, 1],
                "joint_stiffness_nm_rad": [1, 1],
                "joint_damping_nms_rad": [1, 1],
                "joint_lower_rad": [-3, -2],
                "joint_upper_rad": [0, 2],
                "equilibrium_lower": [-5, -5, -3, -2],
                "equilibrium_upper": [5, 5, 0, 2],
                "equilibrium_rate_limit": [10] * 4,
                "equilibrium_acceleration_limit": [100] * 4,
                "provenance": {"inertial": "synthetic", "impedance": "synthetic", "limits": "synthetic"},
            }
            validate(profile)
            (root / "profile.json").write_text(json.dumps(profile))
            ref = _reference(hip_z=1.2, frames=6, duration_s=0.005)
            ref["pelvis_target_rad"] = np.full(6, math.pi / 2)
            np.savez(root / "reference.npz", **ref)
            manifest = {
                "schema": "generative_runner_dataset_1",
                "subject_id": "synthetic",
                "members": [
                    {
                        "id": split,
                        "split": split,
                        "reference": "reference.npz",
                        "profile": "profile.json",
                        "shoe_artifact": "shoe.json",
                        "mount_m": [0, 0, 0.1],
                        "pitch_rad": 0,
                        "speed_m_s": speed,
                    }
                    for split, speed in (("train", 3), ("eval", 4))
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            trials = load_trials(root)
            self.assertEqual([t.task.speed_m_s for t in trials], [3, 4])
            initial = trials[0].initial.copy()
            ref["state"][3:, 0] += 100
            ref["grf_target_n"][:] = 999
            np.savez(root / "reference.npz", **ref)
            changed = load_trials(root)
            np.testing.assert_array_equal(changed[0].initial.q, initial.q)
            np.testing.assert_array_equal(changed[0].initial.v, initial.v)
            output = root / "inspection"
            main(["inspect", "--dataset", str(root), "--output", str(output)])
            self.assertTrue((output / "summary.json").is_file())
            model_path = root / "seed.json"
            Runner.seed().save(model_path)
            output = root / "evaluation"
            main(
                [
                    "evaluate",
                    "--dataset",
                    str(root),
                    "--model",
                    str(model_path),
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                ]
            )
            self.assertTrue((output / "runner.json").is_file())
            self.assertTrue((output / "trace_000.npz").is_file())
            report = json.loads((output / "summary.json").read_text())
            self.assertIn("impedance_instron/hogan/runner.py", {p.replace("\\", "/") for p in report["source_sha256"]})
            # Make the measurements unavailable: the standalone generator must not read them.
            (root / "reference.npz").unlink()
            (root / "manifest.json").unlink()
            generated = root / "generated"
            generate_main(
                [
                    "--model",
                    str(output / "runner.json"),
                    "--scenario",
                    str(output / "scenario_000.json"),
                    "--output",
                    str(generated),
                    "--device",
                    "cpu",
                ]
            )
            with np.load(output / "trace_000.npz") as expected, np.load(generated / "trace.npz") as actual:
                np.testing.assert_array_equal(expected["state"], actual["state"])
                np.testing.assert_array_equal(expected["load"], actual["load"])

    def test_synthetic_forward_targets_improve_without_feedforward(self):
        """Improve a synthetic free-trajectory fit using shared impedance parameters."""
        baseline = Runner.seed(reference_speed_m_s=3.0)
        weights = baseline.weights.copy()
        weights[0, :, 0] += 0.15
        target_model = Runner(weights, reference_speed_m_s=3.0)
        cfg = RolloutConfig(dt_s=0.0005)
        template = _trial()
        trace, summary = simulate(
            target_model, template.chain, template.shoe, template.initial, template.task, duration_s=0.02, config=cfg
        )
        self.assertEqual(summary["status"], "completed")
        trial = Trial(
            "synthetic",
            "train",
            template.chain,
            template.shoe,
            template.task,
            template.initial,
            trace["time_s"],
            trace["state"],
            trace["time_s"],
            np.zeros((len(trace["time_s"]), 2)),
            {"compatibility": {"passed": True}},
        )
        _, report = fit(baseline, [trial], config=cfg, search=FitConfig(population=8, generations=3, seed=8))
        results = report["splits"]["train"]
        self.assertLess(results["learned"]["mean_loss"], results["baseline"]["mean_loss"])

    def test_zero_offset_preserves_seed_and_invalid_baseline_is_rejected(self):
        """Preserve the baseline exactly and reject unsupported identification ranges."""
        baseline = Runner.seed()
        parameters = Parameterization(baseline, [3.7])
        self.assertEqual(parameters.model(np.zeros(parameters.size)).to_dict(), baseline.to_dict())
        with self.assertRaises(ValueError):
            Parameterization(Runner(baseline.weights, frequency_hz=10.0), [3.7])
        with self.assertRaises(ValueError):
            FitConfig(population=4.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
