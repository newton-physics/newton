"""Regression tests for Kamino intrinsic-Euler D6 joints."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim import JointType
from newton._src.solvers.kamino._src.core.joints import JointDoFType
from newton._src.solvers.kamino._src.kinematics.joints import (
    gimbal_reciprocal_axes,
    gimbal_transported_axes,
    map_gimbal_angular_velocity_to_rates,
    select_gimbal_coords,
)
from newton._src.solvers.kamino.tests.utils.extract import extract_cts_jacobians, extract_dofs_jacobians
from newton.solvers import SolverKamino


@wp.kernel
def _evaluate_gimbal_chart(
    coords: wp.array[wp.vec3f],
    reference: wp.array[wp.vec3f],
    omega: wp.array[wp.vec3f],
    effort: wp.array[wp.vec3f],
    third_axis_sign: wp.float32,
    selected: wp.array[wp.vec3f],
    basis_product: wp.array[wp.mat33f],
    power: wp.array[wp.vec2f],
):
    """Evaluate chart selection and reciprocal-basis identities."""
    q = coords[0]
    axes = gimbal_transported_axes(q, third_axis_sign)
    rotation = (
        wp.quat_from_axis_angle(wp.vec3f(axes[:, 2]), q[2])
        * wp.quat_from_axis_angle(wp.vec3f(axes[:, 1]), q[1])
        * wp.quat_from_axis_angle(wp.vec3f(axes[:, 0]), q[0])
    )
    selected[0] = select_gimbal_coords(rotation, reference[0], third_axis_sign)
    reciprocal = gimbal_reciprocal_axes(selected[0], third_axis_sign)
    basis_product[0] = wp.transpose(reciprocal) @ gimbal_transported_axes(selected[0], third_axis_sign)
    rates = map_gimbal_angular_velocity_to_rates(selected[0], omega[0], third_axis_sign)
    power[0] = wp.vec2f(wp.dot(effort[0], rates), wp.dot(reciprocal @ effort[0], omega[0]))


def _build_rotational_d6(
    axes: tuple[newton.Axis, newton.Axis, newton.Axis], *, target_ke: float = 0.0, armature: float = 0.0
):
    """Build a minimal articulated three-axis D6 fixture."""
    builder = newton.ModelBuilder()
    parent = builder.add_link()
    child = builder.add_link()
    root = builder.add_joint_fixed(-1, parent)
    d6 = builder.add_joint_d6(
        parent,
        child,
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=axis, target_ke=target_ke, armature=armature) for axis in axes
        ],
    )
    builder.add_articulation([root, d6])
    return builder.finalize(device="cpu"), d6


class TestGimbal(unittest.TestCase):
    """Verify the rotational D6 representation."""

    def test_rotational_d6_converts_to_gimbal(self):
        """Convert a three-angular-axis D6 joint to GIMBAL."""
        model, d6 = _build_rotational_d6((newton.Axis.X, newton.Axis.Y, newton.Axis.Z))
        solver = SolverKamino(model)
        self.assertEqual(solver._model_kamino.joints.dof_type.numpy()[d6], JointDoFType.GIMBAL)

    def test_left_handed_rotational_d6_converts_to_gimbal_left_handed(self):
        """Classify an X-Z-Y D6 joint as left-handed."""
        model, d6 = _build_rotational_d6((newton.Axis.X, newton.Axis.Z, newton.Axis.Y))
        solver = SolverKamino(model)
        self.assertEqual(solver._model_kamino.joints.dof_type.numpy()[d6], JointDoFType.GIMBAL_LEFT_HANDED)

    def test_from_newton_derives_gimbal_handedness(self):
        """Derive the gimbal type from the orientation of its three axes."""
        limits = np.zeros(3, dtype=np.float32)
        right_handed = JointDoFType.from_newton(JointType.D6, 3, 3, (0, 3), limits, limits, np.eye(3, dtype=np.float32))
        left_handed = JointDoFType.from_newton(
            JointType.D6,
            3,
            3,
            (0, 3),
            limits,
            limits,
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(right_handed, JointDoFType.GIMBAL)
        self.assertEqual(left_handed, JointDoFType.GIMBAL_LEFT_HANDED)

    def test_chart_selects_nearest_equivalent_branch(self):
        """Select the equivalent Euler branch nearest the authored reference."""
        for third_axis_sign in (1.0, -1.0):
            coords = np.array([[0.4, 0.7, third_axis_sign * -0.5]], dtype=np.float32)
            alternative = np.array(
                [[coords[0, 0] + np.pi, np.pi - coords[0, 1], coords[0, 2] + third_axis_sign * np.pi]],
                dtype=np.float32,
            )
            for reference, expected in ((coords, coords), (alternative, alternative)):
                selected = wp.empty(1, dtype=wp.vec3f, device="cpu")
                product = wp.empty(1, dtype=wp.mat33f, device="cpu")
                power = wp.empty(1, dtype=wp.vec2f, device="cpu")
                wp.launch(
                    _evaluate_gimbal_chart,
                    dim=1,
                    inputs=[
                        wp.array(coords, dtype=wp.vec3f, device="cpu"),
                        wp.array(reference, dtype=wp.vec3f, device="cpu"),
                        wp.array([[0.2, -0.4, 0.3]], dtype=wp.vec3f, device="cpu"),
                        wp.array([[0.7, -0.6, 0.5]], dtype=wp.vec3f, device="cpu"),
                        third_axis_sign,
                    ],
                    outputs=[selected, product, power],
                    device="cpu",
                )
                np.testing.assert_allclose(selected.numpy(), expected, atol=1.0e-5)

    def test_reciprocal_basis_preserves_power(self):
        """Preserve dual-basis identity and generalized power away from singularity."""
        selected = wp.empty(1, dtype=wp.vec3f, device="cpu")
        product = wp.empty(1, dtype=wp.mat33f, device="cpu")
        power = wp.empty(1, dtype=wp.vec2f, device="cpu")
        coords = np.array([[0.8, np.pi / 2.0 - 0.02, 0.6]], dtype=np.float32)
        wp.launch(
            _evaluate_gimbal_chart,
            dim=1,
            inputs=[
                wp.array(coords, dtype=wp.vec3f, device="cpu"),
                wp.array(coords, dtype=wp.vec3f, device="cpu"),
                wp.array([[0.2, -0.4, 0.3]], dtype=wp.vec3f, device="cpu"),
                wp.array([[0.7, -0.6, 0.5]], dtype=wp.vec3f, device="cpu"),
                1.0,
            ],
            outputs=[selected, product, power],
            device="cpu",
        )
        np.testing.assert_allclose(product.numpy()[0], np.eye(3), atol=1.0e-5)
        np.testing.assert_allclose(power.numpy()[0, 0], power.numpy()[0, 1], atol=1.0e-5)

    def test_built_gimbal_jacobians_map_body_twists_to_rates(self):
        """Map body twists to authored rates through dense and sparse gimbal Jacobians."""
        q_expected = np.array([0.9, -0.7, 0.5], dtype=np.float32)
        qd_expected = np.array([0.4, -0.3, 0.2], dtype=np.float32)
        for axes in ((newton.Axis.X, newton.Axis.Y, newton.Axis.Z), (newton.Axis.X, newton.Axis.Z, newton.Axis.Y)):
            for sparse_jacobian in (False, True):
                with self.subTest(axes=axes, sparse_jacobian=sparse_jacobian):
                    model, d6 = _build_rotational_d6(axes, armature=0.5)
                    solver = SolverKamino(
                        model,
                        SolverKamino.Config(use_collision_detector=False, sparse_jacobian=sparse_jacobian),
                    )
                    state = model.state()
                    q_start = model.joint_q_start.numpy()[d6]
                    qd_start = model.joint_qd_start.numpy()[d6]
                    joint_q = state.joint_q.numpy()
                    joint_qd = state.joint_qd.numpy()
                    joint_q[q_start : q_start + 3] = q_expected
                    joint_qd[qd_start : qd_start + 3] = qd_expected
                    state.joint_q.assign(joint_q)
                    state.joint_qd.assign(joint_qd)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                    model_kamino = solver._model_kamino
                    solver_kamino = solver._solver_kamino
                    data = solver_kamino._data
                    data.bodies.q_i.assign(state.body_q)
                    data.bodies.u_i.assign(state.body_qd)
                    solver_kamino._update_joints_data(q_j_p=state.joint_q)
                    solver_kamino._update_jacobians()
                    jacobians = solver_kamino._jacobians
                    self.assertEqual(model_kamino.joints.num_dynamic_cts.numpy()[d6], 3)
                    j_dofs = extract_dofs_jacobians(model_kamino, jacobians)[0]
                    j_cts = extract_cts_jacobians(model_kamino, None, None, jacobians)[0]
                    body_twist = data.bodies.u_i.numpy().reshape(-1)

                    np.testing.assert_allclose(j_dofs @ body_twist, qd_expected, atol=1.0e-5)
                    np.testing.assert_allclose(j_cts[:3] @ body_twist, qd_expected, atol=1.0e-5)
                    np.testing.assert_allclose(j_cts[:3], j_dofs, atol=1.0e-6)

    def test_gimbal_metadata(self):
        """Expose three coordinates, rates, and translational constraints."""
        self.assertEqual(JointDoFType.GIMBAL.num_coords, 3)
        self.assertEqual(JointDoFType.GIMBAL.num_dofs, 3)
        self.assertEqual(JointDoFType.GIMBAL.num_cts, 3)
        self.assertTrue(JointDoFType.GIMBAL.is_pure_three_dof_rotation)
        self.assertEqual(JointDoFType.GIMBAL_LEFT_HANDED.num_coords, 3)
        self.assertEqual(JointDoFType.GIMBAL_LEFT_HANDED.num_dofs, 3)
        self.assertEqual(JointDoFType.GIMBAL_LEFT_HANDED.num_cts, 3)
        self.assertTrue(JointDoFType.GIMBAL_LEFT_HANDED.is_pure_three_dof_rotation)

    def test_left_handed_fk_ik_round_trip_uses_authored_third_axis(self):
        """Round-trip authored left-handed D6 coordinates through Newton FK and IK."""
        model, d6 = _build_rotational_d6((newton.Axis.X, newton.Axis.Z, newton.Axis.Y))
        state = model.state()
        q_start = model.joint_q_start.numpy()[d6]
        qd_start = model.joint_qd_start.numpy()[d6]
        q = np.array([0.31, -0.42, 0.53], dtype=np.float32)
        qd = np.array([-0.21, 0.34, -0.45], dtype=np.float32)
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        joint_q[q_start : q_start + 3] = q
        joint_qd[qd_start : qd_start + 3] = qd
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        recovered_q = wp.zeros_like(state.joint_q)
        recovered_qd = wp.zeros_like(state.joint_qd)
        newton.eval_ik(model, state, recovered_q, recovered_qd)
        np.testing.assert_allclose(recovered_q.numpy()[q_start : q_start + 3], q, atol=1.0e-6)
        np.testing.assert_allclose(recovered_qd.numpy()[qd_start : qd_start + 3], qd, atol=1.0e-6)

    def test_fk_reset_preserves_left_handed_coordinates_and_rates(self):
        """Reset an FK-enabled solver with authored left-handed D6 state."""
        model, d6 = _build_rotational_d6((newton.Axis.X, newton.Axis.Z, newton.Axis.Y), target_ke=1.0)
        solver = SolverKamino(model, SolverKamino.Config(use_fk_solver=True))
        state = model.state()
        q_start = model.joint_q_start.numpy()[d6]
        qd_start = model.joint_qd_start.numpy()[d6]
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        q[q_start : q_start + 3] = [0.2, -0.3, 0.4]
        qd[qd_start : qd_start + 3] = [-0.1, 0.15, -0.2]
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        solver.reset(
            state,
            config=SolverKamino.ResetConfig(
                body_poses=SolverKamino.ResetConfig.FromJointQ(state.joint_q),
                body_velocities=SolverKamino.ResetConfig.FromJointU(state.joint_qd),
            ),
        )
        np.testing.assert_allclose(state.joint_q.numpy()[q_start : q_start + 3], q[q_start : q_start + 3], atol=1.0e-5)
        np.testing.assert_allclose(
            state.joint_qd.numpy()[qd_start : qd_start + 3], qd[qd_start : qd_start + 3], atol=1.0e-5
        )


if __name__ == "__main__":
    unittest.main()
