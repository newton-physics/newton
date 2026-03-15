# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import warp as wp

import newton
from newton import BodyFlags, ModelBuilder
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestKinematicLinks(unittest.TestCase):
    """Tests for kinematic body flag handling."""

    def test_body_flags_persist_through_finalize(self):
        """body_flags array on the finalized Model has correct length and values."""
        builder = ModelBuilder()
        builder.add_body(mass=1.0)
        builder.add_body(mass=0.0, is_kinematic=True)
        builder.add_body(mass=2.0)

        model = builder.finalize()
        flags = model.body_flags.numpy()

        self.assertEqual(len(flags), 3)
        self.assertEqual(flags[0], BodyFlags.DYNAMIC)
        self.assertTrue(flags[1] & BodyFlags.KINEMATIC)
        self.assertEqual(flags[2], BodyFlags.DYNAMIC)

    def test_invalid_body_flag_raises_during_finalize(self):
        """finalize() rejects stored body flags that are not a single body state."""
        builder = ModelBuilder()
        builder.add_body(mass=1.0, label="invalid_body")
        builder.body_flags[0] = int(BodyFlags.ALL)

        with self.assertRaises(ValueError) as exc_info:
            builder.finalize()

        self.assertIn("invalid_body", str(exc_info.exception))
        self.assertIn("BodyFlags.DYNAMIC", str(exc_info.exception))
        self.assertIn("BodyFlags.KINEMATIC", str(exc_info.exception))

    def test_body_flags_survive_collapse_fixed_joints(self):
        """Kinematic flags remain attached to retained bodies after collapse."""
        builder = ModelBuilder()
        root = builder.add_link(mass=1.0, is_kinematic=True, label="kinematic_root")
        child = builder.add_link(mass=2.0, label="fixed_child")

        j0 = builder.add_joint_free(parent=-1, child=root)
        j1 = builder.add_joint_fixed(parent=root, child=child)
        builder.add_articulation([j0, j1])

        builder.collapse_fixed_joints()

        self.assertEqual(builder.body_count, 1)
        self.assertEqual(builder.body_flags[0], BodyFlags.KINEMATIC)

        model = builder.finalize()
        np.testing.assert_array_equal(model.body_flags.numpy(), np.array([BodyFlags.KINEMATIC], dtype=np.int32))

    def test_kinematic_root_link_in_articulation(self):
        """A kinematic root link with dynamic children should be valid."""
        builder = ModelBuilder()
        root = builder.add_link(mass=0.0, is_kinematic=True, label="root")
        child = builder.add_link(mass=1.0, label="child")

        j0 = builder.add_joint_fixed(parent=-1, child=root)
        j1 = builder.add_joint_revolute(
            parent=root,
            child=child,
            axis=(0.0, 0.0, 1.0),
        )
        builder.add_articulation([j0, j1])

        model = builder.finalize()
        flags = model.body_flags.numpy()
        self.assertTrue(flags[root] & BodyFlags.KINEMATIC)
        self.assertEqual(flags[child], BodyFlags.DYNAMIC)

    def test_kinematic_non_root_link_raises(self):
        """A kinematic link attached to a non-world parent must raise ValueError."""
        builder = ModelBuilder()
        root = builder.add_link(mass=1.0, label="root")
        child = builder.add_link(mass=0.0, is_kinematic=True, label="child")

        j0 = builder.add_joint_free(parent=-1, child=root)
        j1 = builder.add_joint_revolute(
            parent=root,
            child=child,
            axis=(0.0, 0.0, 1.0),
        )

        with self.assertRaises(ValueError, msg="Only root bodies"):
            builder.add_articulation([j0, j1])

    def test_kinematic_middle_link_raises(self):
        """A kinematic link in the middle of a chain must raise ValueError."""
        builder = ModelBuilder()
        b0 = builder.add_link(mass=1.0, label="b0")
        b1 = builder.add_link(mass=1.0, is_kinematic=True, label="b1")
        b2 = builder.add_link(mass=1.0, label="b2")

        j0 = builder.add_joint_free(parent=-1, child=b0)
        j1 = builder.add_joint_revolute(parent=b0, child=b1, axis=(0.0, 0.0, 1.0))
        j2 = builder.add_joint_revolute(parent=b1, child=b2, axis=(0.0, 0.0, 1.0))

        with self.assertRaises(ValueError, msg="Only root bodies"):
            builder.add_articulation([j0, j1, j2])

    def test_imported_kinematic_root_attached_to_parent_raises(self):
        """Sequential articulation composition preserves the root-only kinematic rule."""
        builder = ModelBuilder()
        parent = builder.add_body(mass=1.0, label="parent")
        imported_root = builder.add_link(mass=1.0, is_kinematic=True, label="imported_root")
        imported_joint = builder.add_joint_fixed(parent=parent, child=imported_root)

        with self.assertRaises(ValueError, msg="Only root bodies"):
            builder._finalize_imported_articulation([imported_joint], parent_body=parent)

    def test_featherstone_rebuilds_mass_matrix_after_kinematic_toggle(self):
        """notify_model_changed() should force a Featherstone mass-matrix rebuild."""
        sim_dt = 1.0 / 60.0
        applied_wrench = np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        builder = ModelBuilder(gravity=0.0)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            is_kinematic=True,
            label="toggle_body",
        )
        builder.add_shape_sphere(body, radius=0.1)

        model = builder.finalize(requires_grad=False)
        solver = newton.solvers.SolverFeatherstone(
            model,
            angular_damping=0.0,
            update_mass_matrix_interval=100,
        )

        state_0, state_1 = model.state(), model.state()
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

        flags = model.body_flags.numpy()
        flags[body] = int(BodyFlags.DYNAMIC)
        model.body_flags.assign(flags)
        solver.notify_model_changed(newton.solvers.SolverNotifyFlags.BODY_PROPERTIES)

        state_0.clear_forces()
        _set_body_wrench(state_0, body, applied_wrench)
        solver.step(state_0, state_1, None, None, sim_dt)

        pos_after_toggle = state_1.body_q.numpy()[body, :3]
        self.assertGreater(
            pos_after_toggle[0],
            1.0e-2,
            "Dynamic body should move on the first step after a kinematic toggle.",
        )


class TestKinematicLinksCanonical(unittest.TestCase):
    pass


KINEMATIC_TEST_WRENCH = np.array([20.0, -15.0, 10.0, 0.5, -0.4, 0.3], dtype=np.float32)


def _uses_maximal_coordinates(solver) -> bool:
    return isinstance(solver, newton.solvers.SolverXPBD | newton.solvers.SolverSemiImplicit | newton.solvers.SolverVBD)


def _create_contacts(model: newton.Model, solver):
    return model.contacts() if not isinstance(solver, newton.solvers.SolverMuJoCo) else None


def _find_joint_for_child(model: newton.Model, child: int) -> int:
    joint_child = model.joint_child.numpy()
    indices = np.where(joint_child == child)[0]
    if len(indices) != 1:
        raise AssertionError(f"Expected exactly one joint for child body {child}, found {len(indices)}")
    return int(indices[0])


def _set_body_wrench(state: newton.State, body_index: int, wrench: np.ndarray) -> None:
    body_f = state.body_f.numpy()
    body_f[body_index] = wrench
    state.body_f.assign(body_f)


def _assert_quat_close(test: unittest.TestCase, qa: np.ndarray, qb: np.ndarray, min_dot: float = 0.999) -> None:
    dot = abs(float(np.dot(qa, qb)))
    test.assertGreater(dot, min_dot, f"Quaternion mismatch: |dot|={dot:.6f} <= {min_dot}")


def _configure_contact_defaults(builder: ModelBuilder) -> None:
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 500.0
    builder.default_shape_cfg.kf = 0.5


def _build_free_root_scene(device):
    builder = ModelBuilder(gravity=0.0)
    _configure_contact_defaults(builder)

    kinematic_body = builder.add_body(
        xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        is_kinematic=True,
        label="kinematic_free",
    )
    builder.add_shape_box(kinematic_body, hx=0.25, hy=0.15, hz=0.15)

    probe_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.45, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        label="probe",
    )
    builder.add_shape_sphere(probe_body, radius=0.1)

    builder.color()
    model = builder.finalize(device=device)
    kinematic_joint = _find_joint_for_child(model, kinematic_body)
    return model, kinematic_body, probe_body, kinematic_joint


def _build_revolute_root_pendulum_scene(device):
    builder = ModelBuilder(gravity=0.0)
    _configure_contact_defaults(builder)

    root = builder.add_link(mass=1.0, is_kinematic=True, label="kinematic_root")
    pendulum = builder.add_link(mass=1.0, label="pendulum")

    root_joint = builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Y, label="kinematic_root_joint")
    pendulum_joint = builder.add_joint_revolute(
        parent=root,
        child=pendulum,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(0.45, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        label="pendulum_joint",
    )
    builder.add_articulation([root_joint, pendulum_joint])

    # Offset kinematic geometry creates a sweeping contact path during root rotation.
    builder.add_shape_box(
        root,
        xform=wp.transform(wp.vec3(0.6, 0.0, 0.0), wp.quat_identity()),
        hx=0.15,
        hy=0.08,
        hz=0.08,
    )
    builder.add_shape_sphere(
        pendulum,
        xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
        radius=0.1,
    )

    probe_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.72, 0.0, 0.0), wp.quat_identity()),
        mass=0.6,
        label="probe",
    )
    builder.add_shape_sphere(probe_body, radius=0.1)

    builder.color()
    model = builder.finalize(device=device)
    return model, root, pendulum, probe_body, root_joint


def _build_fixed_root_scene(device):
    builder = ModelBuilder(gravity=0.0)
    _configure_contact_defaults(builder)

    static_body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        is_kinematic=True,
        label="static_root",
    )
    static_joint = builder.add_joint_fixed(parent=-1, child=static_body, label="static_root_joint")
    builder.add_articulation([static_joint])
    builder.add_shape_box(static_body, hx=0.25, hy=0.25, hz=0.25)

    probe_body = builder.add_body(
        xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        label="probe",
    )
    builder.add_shape_sphere(probe_body, radius=0.12)

    builder.color()
    model = builder.finalize(device=device)
    probe_joint = _find_joint_for_child(model, probe_body)
    return model, static_body, probe_body, probe_joint


def test_kinematic_free_base_prescribed_motion(
    test: TestKinematicLinksCanonical,
    device,
    solver_fn,
):
    sim_dt = 1.0 / 240.0
    steps = 100
    x0 = -0.3
    vx = 1.0

    def run_once(apply_force: bool):
        model, kinematic_body, probe_body, kinematic_joint = _build_free_root_scene(device)
        solver = solver_fn(model)
        contacts = _create_contacts(model, solver)
        state_0, state_1 = model.state(), model.state()

        q_start = int(model.joint_q_start.numpy()[kinematic_joint])
        qd_start = int(model.joint_qd_start.numpy()[kinematic_joint])

        max_probe_speed = 0.0
        initial_probe_pos = state_0.body_q.numpy()[probe_body, :3].copy()

        for step_idx in range(steps):
            t = step_idx * sim_dt

            joint_q = state_0.joint_q.numpy()
            joint_qd = state_0.joint_qd.numpy()
            joint_q[q_start : q_start + 7] = np.array([x0 + vx * t, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            joint_qd[qd_start : qd_start + 6] = np.array([vx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            state_0.joint_q.assign(joint_q)
            state_0.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

            state_0.clear_forces()
            if apply_force:
                _set_body_wrench(state_0, kinematic_body, KINEMATIC_TEST_WRENCH)

            if contacts is not None:
                model.collide(state_0, contacts)

            solver.step(state_0, state_1, None, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

            probe_speed = float(np.linalg.norm(state_0.body_qd.numpy()[probe_body, :3]))
            max_probe_speed = max(max_probe_speed, probe_speed)

        body_q = state_0.body_q.numpy()
        body_qd = state_0.body_qd.numpy()
        return {
            "kin_pos": body_q[kinematic_body, :3].copy(),
            "kin_quat": body_q[kinematic_body, 3:].copy(),
            "kin_qd": body_qd[kinematic_body].copy(),
            "probe_pos": body_q[probe_body, :3].copy(),
            "probe_qd": body_qd[probe_body].copy(),
            "probe_max_speed": max_probe_speed,
            "probe_displacement": float(np.linalg.norm(body_q[probe_body, :3] - initial_probe_pos)),
        }

    no_force = run_once(apply_force=False)
    with_force = run_once(apply_force=True)

    expected_final_x = x0 + vx * (steps - 1) * sim_dt

    # Prescribed motion should drive the kinematic body along +x.
    test.assertAlmostEqual(no_force["kin_pos"][0], expected_final_x, delta=4e-2)
    test.assertLess(abs(float(no_force["kin_pos"][1])), 2e-2)
    test.assertLess(abs(float(no_force["kin_pos"][2])), 2e-2)
    test.assertGreater(float(no_force["kin_qd"][0]), 2e-1)

    # Applied forces should have no/almost no effect on prescribed motion.
    np.testing.assert_allclose(with_force["kin_pos"], no_force["kin_pos"], atol=8e-3)
    _assert_quat_close(test, with_force["kin_quat"], no_force["kin_quat"], min_dot=0.9995)
    test.assertLess(np.linalg.norm(with_force["kin_qd"] - no_force["kin_qd"]), 4e-1)

    # Collision with dynamic body should happen and update velocity.
    test.assertGreater(no_force["probe_max_speed"], 3e-2)
    test.assertTrue(
        float(np.linalg.norm(no_force["probe_qd"][:3])) > 5e-3 or no_force["probe_displacement"] > 1e-3,
        "Probe should show collision-driven velocity/displacement update",
    )


def test_kinematic_revolute_root_pendulum_prescribed_motion(
    test: TestKinematicLinksCanonical,
    device,
    solver_fn,
):
    sim_dt = 1.0 / 240.0
    steps = 120
    theta0 = -1.0
    omega = 2.0

    def run_once(apply_force: bool):
        model, root_body, pendulum_body, probe_body, root_joint = _build_revolute_root_pendulum_scene(device)
        solver = solver_fn(model)
        contacts = _create_contacts(model, solver)
        state_0, state_1 = model.state(), model.state()

        root_q_start = int(model.joint_q_start.numpy()[root_joint])
        root_qd_start = int(model.joint_qd_start.numpy()[root_joint])

        max_probe_speed = 0.0
        max_pendulum_speed = 0.0
        initial_probe_pos = state_0.body_q.numpy()[probe_body, :3].copy()

        for step_idx in range(steps):
            theta = theta0 + omega * step_idx * sim_dt

            if _uses_maximal_coordinates(solver):
                body_q = state_0.body_q.numpy()
                body_qd = state_0.body_qd.numpy()
                body_q[root_body, :3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                body_q[root_body, 3:] = np.array(
                    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(theta)),
                    dtype=np.float32,
                )
                body_qd[root_body] = np.array([0.0, 0.0, 0.0, 0.0, omega, 0.0], dtype=np.float32)
                state_0.body_q.assign(body_q)
                state_0.body_qd.assign(body_qd)
            else:
                joint_q = state_0.joint_q.numpy()
                joint_qd = state_0.joint_qd.numpy()
                joint_q[root_q_start] = theta
                joint_qd[root_qd_start] = omega
                state_0.joint_q.assign(joint_q)
                state_0.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

            state_0.clear_forces()
            if apply_force:
                _set_body_wrench(state_0, root_body, KINEMATIC_TEST_WRENCH)

            if contacts is not None:
                model.collide(state_0, contacts)

            solver.step(state_0, state_1, None, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

            body_qd = state_0.body_qd.numpy()
            max_probe_speed = max(max_probe_speed, float(np.linalg.norm(body_qd[probe_body, :3])))
            max_pendulum_speed = max(max_pendulum_speed, float(np.linalg.norm(body_qd[pendulum_body])))

        body_q = state_0.body_q.numpy()
        body_qd = state_0.body_qd.numpy()
        return {
            "root_pos": body_q[root_body, :3].copy(),
            "root_quat": body_q[root_body, 3:].copy(),
            "root_qd": body_qd[root_body].copy(),
            "probe_max_speed": max_probe_speed,
            "pendulum_max_speed": max_pendulum_speed,
            "probe_displacement": np.linalg.norm(body_q[probe_body, :3] - initial_probe_pos),
        }

    no_force = run_once(apply_force=False)
    with_force = run_once(apply_force=True)

    expected_final_quat = np.array(
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(theta0 + omega * (steps - 1) * sim_dt)),
        dtype=np.float32,
    )

    # Prescribed root motion should be reflected in root orientation and angular velocity.
    _assert_quat_close(test, no_force["root_quat"], expected_final_quat, min_dot=0.99)
    test.assertGreater(np.linalg.norm(no_force["root_qd"][3:]), 5e-1)

    # Applied forces should not perturb prescribed root motion.
    np.testing.assert_allclose(with_force["root_pos"], no_force["root_pos"], atol=1.5e-2)
    _assert_quat_close(test, with_force["root_quat"], no_force["root_quat"], min_dot=0.995)
    test.assertLess(np.linalg.norm(with_force["root_qd"] - no_force["root_qd"]), 8e-1)

    # Contact response should exist and dynamic pendulum velocity should update.
    test.assertGreater(no_force["probe_max_speed"], 2e-2)
    test.assertGreater(float(no_force["probe_displacement"]), 1e-2)
    test.assertGreater(no_force["pendulum_max_speed"], 2e-2)


def test_kinematic_fixed_root_static_force_immune(
    test: TestKinematicLinksCanonical,
    device,
    solver_fn,
):
    sim_dt = 1.0 / 240.0
    steps = 140
    probe_vx = 3.0

    def run_once(apply_force: bool):
        model, static_body, probe_body, probe_joint = _build_fixed_root_scene(device)
        solver = solver_fn(model)
        contacts = _create_contacts(model, solver)
        state_0, state_1 = model.state(), model.state()

        probe_qd_start = int(model.joint_qd_start.numpy()[probe_joint])
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[probe_qd_start : probe_qd_start + 6] = np.array([probe_vx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        initial_static_q = state_0.body_q.numpy()[static_body].copy()

        for _ in range(steps):
            state_0.clear_forces()
            if apply_force:
                _set_body_wrench(state_0, static_body, KINEMATIC_TEST_WRENCH)

            if contacts is not None:
                model.collide(state_0, contacts)

            solver.step(state_0, state_1, None, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        body_qd = state_0.body_qd.numpy()
        return {
            "initial_static_q": initial_static_q,
            "static_pos": body_q[static_body, :3].copy(),
            "static_quat": body_q[static_body, 3:].copy(),
            "static_qd": body_qd[static_body].copy(),
            "probe_pos": body_q[probe_body, :3].copy(),
            "probe_qd": body_qd[probe_body].copy(),
        }

    no_force = run_once(apply_force=False)
    with_force = run_once(apply_force=True)

    # Fixed-root kinematic body remains static.
    np.testing.assert_allclose(no_force["static_pos"], no_force["initial_static_q"][:3], atol=2e-3)
    _assert_quat_close(test, no_force["static_quat"], no_force["initial_static_q"][3:], min_dot=0.9999)
    test.assertLess(np.linalg.norm(no_force["static_qd"]), 4e-1)

    # Applied forces should have no/almost no effect on static kinematic body state.
    np.testing.assert_allclose(with_force["static_pos"], no_force["static_pos"], atol=2e-3)
    _assert_quat_close(test, with_force["static_quat"], no_force["static_quat"], min_dot=0.9999)
    test.assertLess(np.linalg.norm(with_force["static_qd"] - no_force["static_qd"]), 6e-2)

    # Collision should occur and probe velocity should be updated from its initial +x launch.
    test.assertLess(no_force["probe_pos"][0], 0.25)
    test.assertGreater(abs(no_force["probe_qd"][0] - probe_vx), 2.5e-1)


def test_kinematic_runtime_toggle(
    test: TestKinematicLinksCanonical,
    device,
    solver_fn,
):
    """Toggle a body between kinematic and dynamic at runtime via notify_model_changed."""
    sim_dt = 1.0 / 240.0
    phase_steps = 60

    builder = ModelBuilder(gravity=0.0)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        is_kinematic=True,
        label="toggle_body",
    )
    builder.add_shape_sphere(body, radius=0.1)
    builder.color()
    model = builder.finalize(device=device)
    solver = solver_fn(model)
    contacts = _create_contacts(model, solver)

    state_0, state_1 = model.state(), model.state()
    applied_wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Phase 1: body is kinematic — should not move under applied force.
    for _ in range(phase_steps):
        state_0.clear_forces()
        _set_body_wrench(state_0, body, applied_wrench)
        if contacts is not None:
            model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    pos_after_kinematic = state_0.body_q.numpy()[body, :3].copy()
    test.assertLess(np.linalg.norm(pos_after_kinematic), 1e-3, "Kinematic body should not move")

    # Toggle to dynamic.
    flags = model.body_flags.numpy()
    flags[body] = int(BodyFlags.DYNAMIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(newton.solvers.SolverNotifyFlags.BODY_PROPERTIES)

    # Phase 2: body is now dynamic — should move under applied force.
    for _ in range(phase_steps):
        state_0.clear_forces()
        _set_body_wrench(state_0, body, applied_wrench)
        if contacts is not None:
            model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    pos_after_dynamic = state_0.body_q.numpy()[body, :3].copy()
    test.assertGreater(pos_after_dynamic[0], 0.05, "Dynamic body should move under applied force")

    # Toggle back to kinematic.
    flags = model.body_flags.numpy()
    flags[body] = int(BodyFlags.KINEMATIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(newton.solvers.SolverNotifyFlags.BODY_PROPERTIES)

    pos_before_rekinematic = state_0.body_q.numpy()[body, :3].copy()

    # Phase 3: body is kinematic again — should not move further.
    for _ in range(phase_steps):
        state_0.clear_forces()
        _set_body_wrench(state_0, body, applied_wrench)
        if contacts is not None:
            model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    pos_after_rekinematic = state_0.body_q.numpy()[body, :3].copy()
    displacement = np.linalg.norm(pos_after_rekinematic - pos_before_rekinematic)
    test.assertLess(displacement, 1e-3, "Re-kinematic body should not move further")


devices = get_test_devices()
solvers = {
    "featherstone": lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, iterations=5, angular_damping=0.0),
    "semi_implicit": lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
    "vbd": lambda model: newton.solvers.SolverVBD(model),
}
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        if device.is_cpu and solver_name == "mujoco_warp":
            continue

        add_function_test(
            TestKinematicLinksCanonical,
            f"test_kinematic_free_base_prescribed_motion_{solver_name}",
            test_kinematic_free_base_prescribed_motion,
            devices=[device],
            solver_fn=solver_fn,
        )
        add_function_test(
            TestKinematicLinksCanonical,
            f"test_kinematic_revolute_root_pendulum_prescribed_motion_{solver_name}",
            test_kinematic_revolute_root_pendulum_prescribed_motion,
            devices=[device],
            solver_fn=solver_fn,
        )
        add_function_test(
            TestKinematicLinksCanonical,
            f"test_kinematic_fixed_root_static_force_immune_{solver_name}",
            test_kinematic_fixed_root_static_force_immune,
            devices=[device],
            solver_fn=solver_fn,
        )
        add_function_test(
            TestKinematicLinksCanonical,
            f"test_kinematic_runtime_toggle_{solver_name}",
            test_kinematic_runtime_toggle,
            devices=[device],
            solver_fn=solver_fn,
        )


if __name__ == "__main__":
    unittest.main()
