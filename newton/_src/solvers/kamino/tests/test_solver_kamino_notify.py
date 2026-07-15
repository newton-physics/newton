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


def _build_revolute(
    *,
    dynamic: bool = False,
    limited: bool = False,
    actuator_mode: newton.JointTargetMode = newton.JointTargetMode.NONE,
) -> newton.Model:
    """Build a tiny world-to-body revolute model for notify tests."""
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
        # None falls back to the builder default (unlimited) for the non-limited case.
        limit_lower=-1.0 if limited else None,
        limit_upper=1.0 if limited else None,
        armature=1.0 if dynamic else 0.0,
        damping=0.0,
        target_ke=0.0,
        target_kd=0.0,
        actuator_mode=actuator_mode,
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
        model = _build_revolute(limited=True)
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
        model = _build_revolute(limited=True)
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

    def test_aliased_properties_reference_newton(self):
        """Every aliased Newton array shares storage with Kamino, so in-place edits need no notify."""
        model = _build_revolute(limited=True)
        solver = SolverKamino(model)
        bodies = solver._model_kamino.bodies
        joints = solver._model_kamino.joints
        geoms = solver._model_kamino.geoms

        # (Newton model attribute, Kamino container, Kamino attribute) for each direct alias.
        aliased_properties = [
            ("body_mass", bodies, "m_i"),
            ("body_inv_mass", bodies, "inv_m_i"),
            ("body_com", bodies, "i_r_com_i"),
            ("body_inertia", bodies, "i_I_i"),
            ("body_inv_inertia", bodies, "inv_i_I_i"),
            ("joint_q", joints, "q_j_0"),
            ("joint_qd", joints, "dq_j_0"),
            ("joint_limit_lower", joints, "q_j_min"),
            ("joint_limit_upper", joints, "q_j_max"),
            ("joint_velocity_limit", joints, "dq_j_max"),
            ("joint_effort_limit", joints, "tau_j_max"),
            ("joint_armature", joints, "a_j"),
            ("joint_damping", joints, "b_j"),
            ("joint_target_ke", joints, "k_p_j"),
            ("joint_target_kd", joints, "k_d_j"),
            ("shape_scale", geoms, "params"),
            ("shape_collision_radius", geoms, "collision_radius"),
            ("shape_gap", geoms, "gap"),
            ("shape_margin", geoms, "margin"),
        ]

        for newton_name, container, kamino_name in aliased_properties:
            with self.subTest(property=newton_name):
                newton_array = getattr(model, newton_name)
                kamino_array = getattr(container, kamino_name)

                # Kamino references the exact same storage as Newton's array.
                self.assertEqual(kamino_array.ptr, newton_array.ptr)

                # In-place Newton edits are visible on the Kamino side without any notify call.
                perturbed = newton_array.numpy() + np.float32(1.0)
                newton_array.assign(perturbed)
                np.testing.assert_array_equal(kamino_array.numpy(), perturbed)

    def test_gravity_update(self):
        """Model-property notifications refresh Kamino's gravity representation."""
        model = _build_revolute(limited=True)
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
        model = _build_revolute(limited=True)
        solver = SolverKamino(model)
        joints = solver._model_kamino.joints

        parent_position = np.array([0.2, -0.1, 0.3], dtype=np.float32)
        child_position = np.array([-0.4, 0.5, 0.6], dtype=np.float32)
        parent_rotation = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), 0.4)
        child_rotation = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), -0.35)
        model.joint_X_p.assign([wp.transformf(wp.vec3f(*parent_position), parent_rotation)])
        model.joint_X_c.assign([wp.transformf(wp.vec3f(*child_position), child_rotation)])

        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)

        body_com = model.body_com.numpy()[0]
        dof_start = model.joint_qd_start.numpy()[0]
        axis = model.joint_axis.numpy()[dof_start].astype(np.float32)
        R_parent = np.array(wp.quat_to_matrix(parent_rotation)).reshape(3, 3)
        R_child = np.array(wp.quat_to_matrix(child_rotation)).reshape(3, 3)
        X_Bj = joints.X_Bj.numpy()[0]
        X_Fj = joints.X_Fj.numpy()[0]

        np.testing.assert_allclose(joints.B_r_Bj.numpy()[0], parent_position, atol=1e-6)
        np.testing.assert_allclose(joints.F_r_Fj.numpy()[0], child_position - body_com, atol=1e-6)
        np.testing.assert_allclose(
            X_Bj[:, 0],
            R_parent @ axis,
            atol=1e-6,
            err_msg="X_Bj first column must equal R(q_pj) * joint axis",
        )
        np.testing.assert_allclose(
            X_Fj[:, 0],
            R_child @ axis,
            atol=1e-6,
            err_msg="X_Fj first column must equal R(q_cj) * joint axis",
        )

    def test_dynamic_constraint_toggle_raises(self):
        """Adding or removing a joint's dynamic constraints requires solver recreation."""
        for built_dynamic in (False, True):
            with self.subTest(built_dynamic=built_dynamic):
                model = _build_revolute(dynamic=built_dynamic)
                solver = SolverKamino(model)
                built = solver._model_kamino.joints.num_dynamic_cts.numpy()[0] > 0
                self.assertEqual(built, built_dynamic)

                value = np.float32(0.0 if built_dynamic else 1.0)
                model.joint_armature.assign([value])
                model.joint_damping.assign([value])
                model.joint_target_ke.assign([value])
                model.joint_target_kd.assign([value])

                with self.assertRaisesRegex(RuntimeError, "recreate"):
                    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def test_dynamic_coefficient_edit_is_allowed(self):
        """Dynamic coefficient edits are allowed while the dynamic predicate stays true."""
        model = _build_revolute(dynamic=True)
        solver = SolverKamino(model)
        model.joint_target_ke.assign([np.float32(2.0)])

        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def test_limit_finiteness_change_raises(self):
        """Limit capacity changes require solver recreation."""
        for built_limited in (False, True):
            with self.subTest(built_limited=built_limited):
                model = _build_revolute(limited=built_limited)
                solver = SolverKamino(model)
                if built_limited:
                    model.joint_limit_lower.assign([solver._kamino.JOINT_QMIN])
                    model.joint_limit_upper.assign([solver._kamino.JOINT_QMAX])
                else:
                    model.joint_limit_lower.assign([np.float32(-1.0)])

                with self.assertRaisesRegex(RuntimeError, "recreate"):
                    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def test_limit_value_edit_is_allowed(self):
        """Finite limit value edits are allowed while limit capacity stays unchanged."""
        model = _build_revolute(limited=True)
        solver = SolverKamino(model)
        model.joint_limit_lower.assign([np.float32(-0.5)])

        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def test_actuation_mode_change_raises(self):
        """Actuation type changes between active and passive raise under either relevant model flag."""
        modes = (
            (newton.JointTargetMode.NONE, newton.JointTargetMode.POSITION),
            (newton.JointTargetMode.POSITION, newton.JointTargetMode.NONE),
        )
        flags = (
            newton.ModelFlags.ACTUATOR_PROPERTIES,
            newton.ModelFlags.JOINT_DOF_PROPERTIES,
        )
        for built_mode, changed_mode in modes:
            for flag in flags:
                with self.subTest(built_mode=built_mode, changed_mode=changed_mode, flag=flag.name):
                    model = _build_revolute(actuator_mode=built_mode)
                    solver = SolverKamino(model)
                    model.joint_target_mode.assign([int(changed_mode)])

                    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
                    with self.assertRaisesRegex(RuntimeError, "recreate"):
                        solver.notify_model_changed(flag)

    def test_active_actuation_mode_change_is_allowed(self):
        """Active mode changes propagate when the actuation partition is unchanged."""
        modes = (
            (newton.JointTargetMode.POSITION, newton.JointTargetMode.VELOCITY),
            (newton.JointTargetMode.VELOCITY, newton.JointTargetMode.POSITION),
        )
        flags = (
            newton.ModelFlags.ACTUATOR_PROPERTIES,
            newton.ModelFlags.JOINT_DOF_PROPERTIES,
        )
        for built_mode, changed_mode in modes:
            for flag in flags:
                with self.subTest(built_mode=built_mode, changed_mode=changed_mode, flag=flag.name):
                    model = _build_revolute(dynamic=True, actuator_mode=built_mode)
                    solver = SolverKamino(model)
                    model.joint_target_mode.assign([int(changed_mode)])

                    solver.notify_model_changed(flag)

                    expected = solver._kamino.JointActuationType.from_newton(changed_mode)
                    self.assertEqual(solver._model_kamino.joints.act_type.numpy()[0], expected)


if __name__ == "__main__":
    unittest.main()
