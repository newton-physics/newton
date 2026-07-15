# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverKamino runtime model-property propagation."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _build_limited_revolute() -> newton.Model:
    """Build a tiny model: world to a single body via a limited revolute joint."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)

    builder.begin_world()
    bid = builder.add_link(
        label="link",
        mass=1.0,
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
        lock_inertia=True,
    )
    builder.add_shape_box(label="box", body=bid, hx=0.1, hy=0.1, hz=0.1)
    jid = builder.add_joint_revolute(
        label="world_to_link",
        parent=-1,
        child=bid,
        axis=newton.Axis.Y,
        limit_lower=-1.0,
        limit_upper=1.0,
    )
    builder.add_articulation([jid])
    builder.end_world()

    return builder.finalize()


def _snapshot_model_arrays(model: newton.Model) -> dict[str, np.ndarray]:
    """Copy every allocated top-level Warp array on a model."""
    return {name: value.numpy().copy() for name, value in vars(model).items() if isinstance(value, wp.array)}


def _assert_model_arrays_unchanged(
    model: newton.Model,
    snapshot: dict[str, np.ndarray],
) -> None:
    """Assert that model arrays still match a previous snapshot."""
    for name, before in snapshot.items():
        after = getattr(model, name).numpy()
        np.testing.assert_array_equal(after, before, err_msg=f"notify_model_changed mutated model.{name}")


class TestKaminoNotifyModelChanged(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_noop_flags_are_silent_and_do_not_mutate_newton_arrays(self):
        """No-op notifications are silent and leave Newton arrays untouched."""
        model = _build_limited_revolute()
        solver = SolverKamino(model)
        snapshot = _snapshot_model_arrays(model)
        noop_flags = (
            newton.ModelFlags.BODY_PROPERTIES,
            newton.ModelFlags.BODY_INERTIAL_PROPERTIES,
            newton.ModelFlags.SHAPE_PROPERTIES,
            newton.ModelFlags.JOINT_DOF_PROPERTIES,
            newton.ModelFlags.ACTUATOR_PROPERTIES,
            newton.ModelFlags.CONSTRAINT_PROPERTIES,
            newton.ModelFlags.TENDON_PROPERTIES,
        )

        with mock.patch.object(solver._kamino.msg, "warning") as warning:
            for flag in noop_flags:
                with self.subTest(flag=flag.name):
                    warning.reset_mock()
                    solver.notify_model_changed(flag)
                    warning.assert_not_called()
                    _assert_model_arrays_unchanged(model, snapshot)

    def test_unknown_flags_warn_without_raising(self):
        """Unknown flags warn while leaving Newton arrays untouched."""
        model = _build_limited_revolute()
        solver = SolverKamino(model)
        snapshot = _snapshot_model_arrays(model)
        warning_message = "SolverKamino.notify_model_changed: flags 0x%x not yet supported"
        custom_flag = 1 << 20

        with mock.patch.object(solver._kamino.msg, "warning") as warning:
            solver.notify_model_changed(custom_flag)
            solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES | custom_flag)

        warning.assert_has_calls(
            [
                mock.call(warning_message, custom_flag),
                mock.call(warning_message, custom_flag),
            ]
        )
        self.assertEqual(warning.call_count, 2)
        _assert_model_arrays_unchanged(model, snapshot)

    def test_joint_dof_limits_reference_newton(self):
        """Joint DoF limits alias Newton's arrays, so runtime changes need no notify."""
        model = _build_limited_revolute()
        solver = SolverKamino(model)
        joints = solver._model_kamino.joints

        # The Kamino limit arrays share storage with Newton's model arrays.
        self.assertEqual(joints.q_j_min.ptr, model.joint_limit_lower.ptr)
        self.assertEqual(joints.q_j_max.ptr, model.joint_limit_upper.ptr)
        self.assertEqual(joints.dq_j_max.ptr, model.joint_velocity_limit.ptr)
        self.assertEqual(joints.tau_j_max.ptr, model.joint_effort_limit.ptr)

        # In-place updates to Newton's arrays are reflected without any notify call.
        new_lower = np.full_like(model.joint_limit_lower.numpy(), -0.3)
        new_upper = np.full_like(model.joint_limit_upper.numpy(), 0.3)
        new_vel = np.full_like(model.joint_velocity_limit.numpy(), 7.0)
        new_effort = np.full_like(model.joint_effort_limit.numpy(), 70.0)
        model.joint_limit_lower.assign(new_lower)
        model.joint_limit_upper.assign(new_upper)
        model.joint_velocity_limit.assign(new_vel)
        model.joint_effort_limit.assign(new_effort)

        np.testing.assert_allclose(joints.q_j_min.numpy(), new_lower)
        np.testing.assert_allclose(joints.q_j_max.numpy(), new_upper)
        np.testing.assert_allclose(joints.dq_j_max.numpy(), new_vel)
        np.testing.assert_allclose(joints.tau_j_max.numpy(), new_effort)

    def test_body_inertial_properties_reference_newton(self):
        """Body inertial arrays alias Newton's, so runtime changes need no notify."""
        model = _build_limited_revolute()
        solver = SolverKamino(model)
        bodies = solver._model_kamino.bodies

        # The Kamino arrays share storage with Newton's model arrays.
        self.assertEqual(bodies.i_r_com_i.ptr, model.body_com.ptr)
        self.assertEqual(bodies.i_I_i.ptr, model.body_inertia.ptr)
        self.assertEqual(bodies.inv_i_I_i.ptr, model.body_inv_inertia.ptr)
        self.assertEqual(bodies.m_i.ptr, model.body_mass.ptr)

        # In-place updates to Newton's arrays are reflected without any notify call.
        new_com = model.body_com.numpy() + np.array([0.1, 0.2, 0.3], dtype=np.float32)
        new_inertia = model.body_inertia.numpy() * 2.0
        new_inv_inertia = model.body_inv_inertia.numpy() * 0.5
        model.body_com.assign(new_com)
        model.body_inertia.assign(new_inertia)
        model.body_inv_inertia.assign(new_inv_inertia)

        np.testing.assert_allclose(bodies.i_r_com_i.numpy(), new_com, atol=1e-6)
        np.testing.assert_allclose(bodies.i_I_i.numpy(), new_inertia, atol=1e-6)
        np.testing.assert_allclose(bodies.inv_i_I_i.numpy(), new_inv_inertia, atol=1e-6)

    def test_gravity_update(self):
        """Model-property notifications refresh Kamino's gravity representation."""
        model = _build_limited_revolute()
        solver = SolverKamino(model)
        gravity = np.tile(np.array([1.0, -2.0, 3.0], dtype=np.float32), (model.world_count, 1))
        acceleration = np.linalg.norm(gravity, axis=1)

        model.gravity.assign(gravity)
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

        expected_g_dir_acc = np.column_stack((gravity / acceleration[:, None], acceleration))
        expected_vector = np.column_stack((gravity, np.ones(model.world_count, dtype=np.float32)))
        np.testing.assert_allclose(solver._model_kamino.gravity.g_dir_acc.numpy(), expected_g_dir_acc, atol=1e-6)
        np.testing.assert_allclose(solver._model_kamino.gravity.vector.numpy(), expected_vector, atol=1e-6)

    def test_joint_transform_update(self):
        """Joint-property notifications recompute Kamino's parent and child frames."""
        model = _build_limited_revolute()
        solver = SolverKamino(model)
        joints = solver._model_kamino.joints
        old_X_Bj = joints.X_Bj.numpy().copy()
        old_X_Fj = joints.X_Fj.numpy().copy()

        parent_position = np.array([0.2, -0.1, 0.3], dtype=np.float32)
        child_position = np.array([-0.4, 0.5, 0.6], dtype=np.float32)
        parent_angle = 0.4
        child_angle = -0.35
        parent_rotation = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), parent_angle)
        child_rotation = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), child_angle)
        model.joint_X_p.assign([wp.transformf(wp.vec3f(*parent_position), parent_rotation)])
        model.joint_X_c.assign([wp.transformf(wp.vec3f(*child_position), child_rotation)])

        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)

        parent_cos, parent_sin = np.cos(parent_angle), np.sin(parent_angle)
        child_cos, child_sin = np.cos(child_angle), np.sin(child_angle)
        parent_rotation_matrix = np.array(
            [[parent_cos, -parent_sin, 0.0], [parent_sin, parent_cos, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        child_rotation_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, child_cos, -child_sin], [0.0, child_sin, child_cos]],
            dtype=np.float32,
        )
        body_com = model.body_com.numpy()[0]

        np.testing.assert_allclose(joints.B_r_Bj.numpy()[0], parent_position, atol=1e-6)
        np.testing.assert_allclose(joints.F_r_Fj.numpy()[0], child_position - body_com, atol=1e-6)
        np.testing.assert_allclose(joints.X_Bj.numpy()[0], parent_rotation_matrix @ old_X_Bj[0], atol=1e-6)
        np.testing.assert_allclose(joints.X_Fj.numpy()[0], child_rotation_matrix @ old_X_Fj[0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
