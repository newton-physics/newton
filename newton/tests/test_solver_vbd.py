# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VBD solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_self_contact_force_norm
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    RigidContactHistory,
    RigidForceElementAdjacencyInfo,
    evaluate_angular_constraint_force_hessian,
    evaluate_linear_constraint_force_hessian,
    init_body_body_contacts_avbd,
    snapshot_body_body_contact_history,
    solve_rigid_body,
    sort_body_body_contact_indices,
    update_duals_body_body_contacts,
    update_duals_joint,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices(mode="basic")


@wp.kernel
def _eval_self_contact_norm_kernel(
    distances: wp.array[float],
    collision_radius: float,
    k: float,
    dEdD_out: wp.array[float],
    d2E_out: wp.array[float],
):
    i = wp.tid()
    dEdD, d2E = evaluate_self_contact_force_norm(distances[i], collision_radius, k)
    dEdD_out[i] = dEdD
    d2E_out[i] = d2E


@wp.kernel
def _eval_directional_joint_projection_kernel(
    linear_force_out: wp.array[wp.vec3],
    angular_torque_out: wp.array[wp.vec3],
):
    a = wp.vec3(1.0, 0.0, 0.0)
    P = wp.identity(3, float) - wp.outer(a, a)
    q_id = wp.quat_identity()
    X_wp = wp.transform(wp.vec3(0.0), q_id)
    X_wc = wp.transform(wp.vec3(4.0, 2.0, 3.0), q_id)
    force, _torque, _Hll, _Hal, _Haa = evaluate_linear_constraint_force_hessian(
        X_wp,
        X_wc,
        X_wp,
        X_wc,
        wp.transform_identity(),
        wp.transform_identity(),
        wp.vec3(0.0),
        wp.vec3(0.0),
        True,
        2.0,
        P,
        wp.vec3(5.0, 7.0, 11.0),
        wp.vec3(0.0),
        0.0,
        0.0,
        0.01,
    )
    linear_force_out[0] = force

    q_free = wp.quat_from_axis_angle(a, 0.5)
    torque, _Haa_ang, _kappa, _J = evaluate_angular_constraint_force_hessian(
        q_id,
        q_free,
        q_id,
        q_id,
        q_id,
        q_id,
        True,
        2.0,
        P,
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(5.0, 7.0, 11.0),
        wp.vec3(0.0),
        0.0,
        0.0,
        0.01,
    )
    angular_torque_out[0] = torque


def test_self_contact_barrier_c2_at_tau(test, device):
    """Barrier must be C2-continuous at d = tau (= collision_radius / 2).

    The log-barrier region (d_min < d < tau) and the outer linear-penalty
    region (tau <= d < collision_radius) share the boundary d = tau.  For
    C2 continuity both the first derivative (force) and the second
    derivative (Hessian scalar) must agree there.

    Regression for GitHub issue #2154.
    """
    collision_radius = 0.02
    k = 1.0e3
    tau = collision_radius * 0.5
    eps = tau * 1e-5

    distances = wp.array([tau - eps, tau + eps], dtype=float, device=device)
    dEdD_out = wp.zeros(2, dtype=float, device=device)
    d2E_out = wp.zeros(2, dtype=float, device=device)

    wp.launch(
        _eval_self_contact_norm_kernel,
        dim=2,
        inputs=[distances, collision_radius, k, dEdD_out, d2E_out],
        device=device,
    )

    dEdD = dEdD_out.numpy()
    d2E = d2E_out.numpy()

    np.testing.assert_allclose(
        dEdD[0],
        dEdD[1],
        rtol=1e-3,
        err_msg="Self-contact barrier force is not C1-continuous at d = tau",
    )
    np.testing.assert_allclose(
        d2E[0],
        d2E[1],
        rtol=1e-3,
        err_msg="Self-contact barrier Hessian is not C2-continuous at d = tau",
    )


def test_self_contact_barrier_c2_at_d_min(test, device):
    """Barrier must be C2-continuous at d = d_min (= 1e-5).

    The quadratic-extension region (d <= d_min) and the log-barrier region
    (d_min < d < tau) share the boundary d = d_min.  The quadratic is a
    Taylor expansion of the log-barrier at d_min, so both the first and
    second derivatives must match.
    """
    collision_radius = 0.02
    k = 1.0e3
    d_min = 1.0e-5
    eps = d_min * 1e-5

    distances = wp.array([d_min - eps, d_min + eps], dtype=float, device=device)
    dEdD_out = wp.zeros(2, dtype=float, device=device)
    d2E_out = wp.zeros(2, dtype=float, device=device)

    wp.launch(
        _eval_self_contact_norm_kernel,
        dim=2,
        inputs=[distances, collision_radius, k, dEdD_out, d2E_out],
        device=device,
    )

    dEdD = dEdD_out.numpy()
    d2E = d2E_out.numpy()

    np.testing.assert_allclose(
        dEdD[0],
        dEdD[1],
        rtol=1e-3,
        err_msg="Self-contact barrier force is not C1-continuous at d = d_min",
    )
    np.testing.assert_allclose(
        d2E[0],
        d2E[1],
        rtol=1e-3,
        err_msg="Self-contact barrier Hessian is not C2-continuous at d = d_min",
    )


def _rigid_contact_history_restore_from_match_index(test, device):
    """VBD warm-start restores from explicit match_index rows."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([4], dtype=int, device=device)
        shape0 = wp.array([0, 0, 0, 0], dtype=int, device=device)
        shape1 = wp.array([1, 1, 1, 1], dtype=int, device=device)
        point0_in = np.array(
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [13.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        point1_in = point0_in + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        point0 = wp.array(point0_in, dtype=wp.vec3, device=device)
        point1 = wp.array(point1_in, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]] * 4, dtype=wp.vec3, device=device)

        shape_ke = wp.array([100.0, 200.0], dtype=float, device=device)
        shape_kd = wp.array([1.0, 3.0], dtype=float, device=device)
        shape_mu = wp.array([0.25, 1.0], dtype=float, device=device)
        match_index = wp.array([2, -1, 0, -2], dtype=wp.int32, device=device)

        history = RigidContactHistory()
        history.lambda_ = wp.array([[0.5, 0.0, 1.0], [4.0, 5.0, 6.0], [0.0, 0.0, 7.0]], dtype=wp.vec3, device=device)
        history.stick_flag = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        history.penalty_k = wp.array([20.0, 30.0, 40.0], dtype=float, device=device)
        history.point0 = wp.array([[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        history.point1 = wp.array([[20.0, 0.0, 1.0], [21.0, 0.0, 1.0], [22.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]] * 3, dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(4, dtype=float, device=device)
        lam = wp.zeros(4, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(4, dtype=float, device=device)
        material_mu = wp.zeros(4, dtype=float, device=device)
        material_ke = wp.zeros(4, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_avbd,
            dim=4,
            inputs=[
                contact_count,
                shape0,
                shape1,
                normal,
                shape_ke,
                shape_kd,
                shape_mu,
                1,
                match_index,
                history,
                10.0,
            ],
            outputs=[
                point0,
                point1,
                penalty_k,
                lam,
                material_kd,
                material_mu,
                material_ke,
            ],
            device=device,
        )

        np.testing.assert_allclose(penalty_k.numpy(), [40.0, 10.0, 20.0, 10.0])
        np.testing.assert_allclose(lam.numpy(), [[0.0, 0.0, 7.0], [0.0, 0.0, 0.0], [0.5, 0.0, 1.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(material_ke.numpy(), [150.0] * 4)
        np.testing.assert_allclose(material_kd.numpy(), [2.0] * 4)
        np.testing.assert_allclose(material_mu.numpy(), [0.5] * 4)

        point0_out = point0.numpy()
        point1_out = point1.numpy()
        np.testing.assert_allclose(point0_out[0], [22.0, 0.0, 0.0])
        np.testing.assert_allclose(point1_out[0], [22.0, 0.0, 1.0])
        np.testing.assert_allclose(point0_out[2], point0_in[2])
        np.testing.assert_allclose(point1_out[2], point1_in[2])
        np.testing.assert_allclose(point0_out[1], point0_in[1])
        np.testing.assert_allclose(point0_out[3], point0_in[3])


def _rigid_contact_history_soft_restores_penalty_only(test, device):
    """Soft contacts restore penalty state only; saved lambda and anchors stay unused."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([1], dtype=int, device=device)
        shape0 = wp.array([0], dtype=int, device=device)
        shape1 = wp.array([1], dtype=int, device=device)
        point0_in = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        point1_in = np.array([[10.0, 0.0, 1.0]], dtype=np.float32)
        point0 = wp.array(point0_in, dtype=wp.vec3, device=device)
        point1 = wp.array(point1_in, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)

        history = RigidContactHistory()
        history.lambda_ = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device=device)
        history.stick_flag = wp.array([1], dtype=wp.int32, device=device)
        history.penalty_k = wp.array([40.0], dtype=float, device=device)
        history.point0 = wp.array([[20.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        history.point1 = wp.array([[20.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(1, dtype=float, device=device)
        lam = wp.zeros(1, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(1, dtype=float, device=device)
        material_mu = wp.zeros(1, dtype=float, device=device)
        material_ke = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_avbd,
            dim=1,
            inputs=[
                contact_count,
                shape0,
                shape1,
                normal,
                wp.array([100.0, 200.0], dtype=float, device=device),
                wp.array([1.0, 3.0], dtype=float, device=device),
                wp.array([0.25, 1.0], dtype=float, device=device),
                0,
                wp.array([0], dtype=wp.int32, device=device),
                history,
                10.0,
            ],
            outputs=[
                point0,
                point1,
                penalty_k,
                lam,
                material_kd,
                material_mu,
                material_ke,
            ],
            device=device,
        )

        np.testing.assert_allclose(penalty_k.numpy(), [40.0])
        np.testing.assert_allclose(lam.numpy(), [[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(point0.numpy(), point0_in)
        np.testing.assert_allclose(point1.numpy(), point1_in)


def _joint_angular_dual_projects_free_axis_lambda(test, device):
    """Angular dual updates should discard lambda on free angular axes."""
    with wp.ScopedDevice(device):
        joint_type = wp.array([int(newton.JointType.REVOLUTE)], dtype=wp.int32, device=device)
        joint_enabled = wp.array([True], dtype=bool, device=device)
        joint_parent = wp.array([-1], dtype=wp.int32, device=device)
        joint_child = wp.array([0], dtype=wp.int32, device=device)
        joint_x_p = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_x_c = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_axis = wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        joint_qd_start = wp.array([0], dtype=wp.int32, device=device)
        joint_constraint_start = wp.array([0], dtype=wp.int32, device=device)
        body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        body_q_rest = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_dof_dim = wp.array([[0, 0]], dtype=wp.int32, device=device)
        joint_c0_lin = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_c0_ang = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_is_hard = wp.array([1, 1, 0], dtype=wp.int32, device=device)
        joint_penalty_k_max = wp.array([10.0, 10.0, 10.0], dtype=float, device=device)
        joint_target_ke = wp.array([0.0], dtype=float, device=device)
        joint_target_pos = wp.array([0.0], dtype=float, device=device)
        joint_limit_lower = wp.array([-1.0], dtype=float, device=device)
        joint_limit_upper = wp.array([1.0], dtype=float, device=device)
        joint_limit_ke = wp.array([0.0], dtype=float, device=device)
        joint_rest_angle = wp.array([0.0], dtype=float, device=device)
        joint_penalty_k = wp.array([10.0, 10.0, 10.0], dtype=float, device=device)
        lambda_lin = wp.zeros(1, dtype=wp.vec3, device=device)
        lambda_ang = wp.array([[5.0, 2.0, 3.0]], dtype=wp.vec3, device=device)

        wp.launch(
            update_duals_joint,
            dim=1,
            inputs=[
                joint_type,
                joint_enabled,
                joint_parent,
                joint_child,
                joint_x_p,
                joint_x_c,
                joint_axis,
                joint_qd_start,
                joint_constraint_start,
                body_q,
                body_q_rest,
                joint_dof_dim,
                joint_c0_lin,
                joint_c0_ang,
                joint_is_hard,
                0.0,
                joint_penalty_k_max,
                0.0,
                0.0,
                joint_target_ke,
                joint_target_pos,
                joint_limit_lower,
                joint_limit_upper,
                joint_limit_ke,
                joint_rest_angle,
            ],
            outputs=[joint_penalty_k, lambda_lin, lambda_ang],
            device=device,
        )

        np.testing.assert_allclose(lambda_ang.numpy(), [[0.0, 2.0, 3.0]])


def _joint_force_projection_filters_free_direction(test, device):
    """Projected joint force path should not apply force along free directions."""
    with wp.ScopedDevice(device):
        linear_force = wp.zeros(1, dtype=wp.vec3, device=device)
        angular_torque = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch(
            _eval_directional_joint_projection_kernel,
            dim=1,
            outputs=[linear_force, angular_torque],
            device=device,
        )

        np.testing.assert_allclose(linear_force.numpy(), [[0.0, 11.0, 17.0]], rtol=1e-6, atol=1e-6)
        angular_torque_np = angular_torque.numpy()
        np.testing.assert_allclose(angular_torque_np[:, 0], [0.0], rtol=1e-6, atol=1e-6)
        test.assertGreater(np.linalg.norm(angular_torque_np[:, 1:]), 0.0)


def _d6_fully_free_structural_slots_are_inactive(test, device):
    """D6 structural slots should be inactive when all axes are free."""
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link()
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)

    JointDofConfig = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        -1,
        body,
        linear_axes=[
            JointDofConfig.create_unlimited(newton.Axis.X),
            JointDofConfig.create_unlimited(newton.Axis.Y),
            JointDofConfig.create_unlimited(newton.Axis.Z),
        ],
        angular_axes=[
            JointDofConfig.create_unlimited(newton.Axis.X),
            JointDofConfig.create_unlimited(newton.Axis.Y),
            JointDofConfig.create_unlimited(newton.Axis.Z),
        ],
    )
    builder.add_articulation([joint])

    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model)
    start = int(solver.joint_constraint_start.numpy()[joint])

    np.testing.assert_allclose(solver.joint_penalty_k.numpy()[start : start + 2], [0.0, 0.0])
    np.testing.assert_allclose(solver.joint_penalty_k_max.numpy()[start : start + 2], [0.0, 0.0])
    np.testing.assert_array_equal(solver.joint_is_hard.numpy()[start : start + 2], [0, 0])


def _rigid_contact_history_snapshot_copies_active_rows(test, device):
    """Snapshot writes solved state by active contact row and leaves inactive rows untouched."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([2], dtype=int, device=device)
        point0 = wp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        point1 = wp.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        lam = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=wp.vec3, device=device)
        stick = wp.array([1, 2, 3], dtype=wp.int32, device=device)
        penalty = wp.array([10.0, 20.0, 30.0], dtype=float, device=device)

        prev_lambda = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_stick = wp.zeros(3, dtype=wp.int32, device=device)
        prev_penalty = wp.zeros(3, dtype=float, device=device)
        prev_point0 = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_point1 = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_normal = wp.zeros(3, dtype=wp.vec3, device=device)

        wp.launch(
            snapshot_body_body_contact_history,
            dim=3,
            inputs=[contact_count, point0, point1, normal, lam, stick, penalty],
            outputs=[prev_lambda, prev_stick, prev_penalty, prev_point0, prev_point1, prev_normal],
            device=device,
        )

        np.testing.assert_allclose(prev_lambda.numpy()[:2], lam.numpy()[:2])
        np.testing.assert_allclose(prev_stick.numpy()[:2], [1, 2])
        np.testing.assert_allclose(prev_penalty.numpy()[:2], [10.0, 20.0])
        np.testing.assert_allclose(prev_point0.numpy()[:2], point0.numpy()[:2])
        np.testing.assert_allclose(prev_point1.numpy()[:2], point1.numpy()[:2])
        np.testing.assert_allclose(prev_normal.numpy()[:2], normal.numpy()[:2])
        np.testing.assert_allclose(prev_lambda.numpy()[2], [0.0, 0.0, 0.0])
        test.assertEqual(prev_stick.numpy()[2], 0)
        test.assertEqual(prev_penalty.numpy()[2], 0.0)


def _rigid_contact_stick_flags_require_cone_and_small_residual(test, device):
    """Contact stick flags require normal load, cone feasibility, and small tangential residual."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([4], dtype=int, device=device)
        shape0 = wp.array([0, 0, 0, 0], dtype=int, device=device)
        shape1 = wp.array([1, 2, 3, 4], dtype=int, device=device)
        point0 = wp.zeros(4, dtype=wp.vec3, device=device)
        point1 = wp.zeros(4, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]] * 4, dtype=wp.vec3, device=device)
        margin0 = wp.array([0.05, 0.05, 0.05, 0.05], dtype=float, device=device)
        margin1 = wp.array([0.05, 0.05, 0.05, 0.05], dtype=float, device=device)
        shape_body = wp.array([0, 1, 2, 3, 4], dtype=int, device=device)

        q = wp.quat_identity()
        body_q = wp.array(
            [
                wp.transform(wp.vec3(0.0, 0.0, 0.0), q),
                wp.transform(wp.vec3(1.0, 0.0, 0.0), q),
                wp.transform(wp.vec3(0.03, 0.0, 0.0), q),
                wp.transform(wp.vec3(0.01, 0.0, 0.0), q),
                wp.transform(wp.vec3(0.01, 0.0, 0.0), q),
            ],
            dtype=wp.transform,
            device=device,
        )
        body_q_prev = wp.array([wp.transform_identity()] * 5, dtype=wp.transform, device=device)
        contact_mu = wp.array([0.5, 0.5, 0.5, 0.5], dtype=float, device=device)
        contact_c0 = wp.zeros(4, dtype=wp.vec3, device=device)
        body_inv_mass = wp.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=float, device=device)
        contact_ke = wp.array([10.0, 10.0, 10.0, 10.0], dtype=float, device=device)
        penalty_k = wp.array([10.0, 10.0, 10.0, 10.0], dtype=float, device=device)
        contact_lambda = wp.zeros(4, dtype=wp.vec3, device=device)
        stick_flag = wp.zeros(4, dtype=wp.int32, device=device)

        wp.launch(
            update_duals_body_body_contacts,
            dim=4,
            inputs=[
                contact_count,
                shape0,
                shape1,
                point0,
                point1,
                normal,
                margin0,
                margin1,
                shape_body,
                body_q,
                body_q_prev,
                contact_mu,
                contact_c0,
                0.0,
                0.02,
                1,
                body_inv_mass,
                contact_ke,
                0.0,
            ],
            outputs=[penalty_k, contact_lambda, stick_flag],
            device=device,
        )

        np.testing.assert_allclose(
            contact_lambda.numpy(),
            [
                [-0.5, 0.0, 1.0],
                [-0.3, 0.0, 1.0],
                [-0.1, 0.0, 1.0],
                [-0.1, 0.0, 1.0],
            ],
        )
        np.testing.assert_array_equal(stick_flag.numpy(), [0, 0, 1, 2])

        contact_lambda.zero_()
        stick_flag.zero_()
        penalty_k = wp.array([10.0, 10.0, 10.0, 10.0], dtype=float, device=device)

        wp.launch(
            update_duals_body_body_contacts,
            dim=4,
            inputs=[
                contact_count,
                shape0,
                shape1,
                point0,
                point1,
                normal,
                margin0,
                margin1,
                shape_body,
                body_q,
                body_q_prev,
                contact_mu,
                contact_c0,
                0.0,
                0.0,
                1,
                body_inv_mass,
                contact_ke,
                0.0,
            ],
            outputs=[penalty_k, contact_lambda, stick_flag],
            device=device,
        )

        np.testing.assert_array_equal(stick_flag.numpy(), [0, 0, 0, 0])


def _rigid_body_contact_indices_are_sorted_for_deterministic_traversal(test, device):
    """Body-centric rigid contact traversal should not depend on atomic insertion order."""
    body_count = 4
    contact_capacity = 6
    with wp.ScopedDevice(device):
        contact_counts = wp.array([5, 4, 0, 8], dtype=wp.int32, device=device)
        contact_indices = wp.array(
            [
                8,
                3,
                5,
                1,
                4,
                99,
                2,
                2,
                9,
                0,
                77,
                88,
                42,
                43,
                44,
                45,
                46,
                47,
                10,
                6,
                11,
                7,
                12,
                5,
            ],
            dtype=wp.int32,
            device=device,
        )

        wp.launch(
            sort_body_body_contact_indices,
            dim=body_count,
            inputs=[contact_counts, contact_indices, contact_capacity],
            device=device,
        )

        sorted_rows = contact_indices.numpy().reshape(body_count, contact_capacity)
        np.testing.assert_array_equal(sorted_rows[0, :5], [1, 3, 4, 5, 8])
        test.assertEqual(sorted_rows[0, 5], 99)
        np.testing.assert_array_equal(sorted_rows[1, :4], [0, 2, 2, 9])
        np.testing.assert_array_equal(sorted_rows[1, 4:], [77, 88])
        np.testing.assert_array_equal(sorted_rows[2], [42, 43, 44, 45, 46, 47])
        np.testing.assert_array_equal(sorted_rows[3], [5, 6, 7, 10, 11, 12])


def _solve_single_body_contact_order(device, row_order, sort_row):
    """Solve one body with a synthetic contact row and return the body pose."""
    q_identity = wp.quat_identity()
    transform_identity = wp.transform(wp.vec3(0.0), q_identity)
    body_ids = wp.array([0], dtype=wp.int32, device=device)
    body_q = wp.array([transform_identity], dtype=wp.transform, device=device)
    body_q_prev = wp.array([transform_identity], dtype=wp.transform, device=device)
    body_q_rest = wp.array([transform_identity], dtype=wp.transform, device=device)
    body_mass = wp.array([1.0], dtype=float, device=device)
    body_inv_mass = wp.array([1.0], dtype=float, device=device)
    body_inertia = wp.array([wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)], dtype=wp.mat33, device=device)
    body_com = wp.zeros(1, dtype=wp.vec3, device=device)

    empty_i = wp.zeros(0, dtype=wp.int32, device=device)
    empty_b = wp.zeros(0, dtype=bool, device=device)
    empty_f = wp.zeros(0, dtype=float, device=device)
    empty_v = wp.zeros(0, dtype=wp.vec3, device=device)
    empty_x = wp.zeros(0, dtype=wp.transform, device=device)
    empty_dof = wp.zeros((0, 2), dtype=wp.int32, device=device)

    adjacency = RigidForceElementAdjacencyInfo()
    adjacency.body_adj_joints = empty_i
    adjacency.body_adj_joints_offsets = wp.array([0, 0], dtype=wp.int32, device=device)

    external_forces = wp.zeros(1, dtype=wp.vec3, device=device)
    external_torques = wp.zeros(1, dtype=wp.vec3, device=device)
    external_hessian_ll = wp.zeros(1, dtype=wp.mat33, device=device)
    external_hessian_al = wp.zeros(1, dtype=wp.mat33, device=device)
    external_hessian_aa = wp.zeros(1, dtype=wp.mat33, device=device)

    contact_count = wp.array([3], dtype=wp.int32, device=device)
    contact_shape0 = wp.array([0, 0, 0], dtype=wp.int32, device=device)
    contact_shape1 = wp.array([1, 1, 1], dtype=wp.int32, device=device)
    contact_points = wp.zeros(3, dtype=wp.vec3, device=device)
    contact_normals = wp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    contact_margins = wp.zeros(3, dtype=float, device=device)
    shape_body = wp.array([0, -1], dtype=wp.int32, device=device)

    # Contacts 0 and 1 are large opposing hard-contact impulses, and contact 2
    # is small. Different row orders can drop the small term before the large
    # cancellation, which is the failure mode the sorter removes.
    contact_penalty_k = wp.array([1.0, 1.0, 1.0], dtype=float, device=device)
    contact_material_kd = wp.zeros(3, dtype=float, device=device)
    contact_material_mu = wp.zeros(3, dtype=float, device=device)
    contact_lambda = wp.array([[1.0e8, 0.0, 0.0], [-1.0e8, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    contact_C0 = wp.zeros(3, dtype=wp.vec3, device=device)

    contact_capacity = len(row_order)
    body_contact_counts = wp.array([contact_capacity], dtype=wp.int32, device=device)
    body_contact_indices = wp.array(row_order, dtype=wp.int32, device=device)

    if sort_row:
        wp.launch(
            sort_body_body_contact_indices,
            dim=1,
            inputs=[body_contact_counts, body_contact_indices, contact_capacity],
            device=device,
        )

    body_q_out = wp.empty(1, dtype=wp.transform, device=device)
    wp.launch(
        solve_rigid_body,
        dim=1,
        inputs=[
            1.0,
            body_ids,
            body_q,
            body_q,
            body_q_prev,
            body_q_rest,
            body_mass,
            body_inv_mass,
            body_inertia,
            body_q,
            body_com,
            adjacency,
            empty_i,
            empty_b,
            empty_i,
            empty_i,
            empty_x,
            empty_x,
            empty_v,
            empty_i,
            empty_i,
            empty_f,
            empty_f,
            empty_v,
            empty_v,
            empty_f,
            empty_f,
            empty_f,
            empty_f,
            empty_f,
            empty_f,
            empty_f,
            empty_f,
            empty_v,
            empty_v,
            empty_v,
            empty_v,
            empty_i,
            0.95,
            empty_dof,
            empty_f,
            external_forces,
            external_torques,
            external_hessian_ll,
            external_hessian_al,
            external_hessian_aa,
            1.0e-4,
            contact_penalty_k,
            contact_material_kd,
            contact_material_mu,
            contact_lambda,
            contact_C0,
            0.95,
            1,
            contact_count,
            contact_shape0,
            contact_shape1,
            contact_points,
            contact_points,
            contact_normals,
            contact_margins,
            contact_margins,
            shape_body,
            contact_capacity,
            body_contact_counts,
            body_contact_indices,
        ],
        outputs=[body_q_out],
        device=device,
    )
    return body_q_out.numpy()[0]


def _sorted_rigid_contact_rows_make_solve_order_deterministic(test, device):
    """The same contacts should solve identically after deterministic row sorting."""
    with wp.ScopedDevice(device):
        row_a = [0, 1, 2]
        row_b = [0, 2, 1]

        unsorted_a = _solve_single_body_contact_order(device, row_a, sort_row=False)
        unsorted_b = _solve_single_body_contact_order(device, row_b, sort_row=False)
        test.assertFalse(np.array_equal(unsorted_a, unsorted_b))
        test.assertGreater(float(np.max(np.abs(unsorted_a - unsorted_b))), 0.0)

        sorted_a = _solve_single_body_contact_order(device, row_a, sort_row=True)
        sorted_b = _solve_single_body_contact_order(device, row_b, sort_row=True)
        np.testing.assert_array_equal(sorted_a, sorted_b)


def _run_vbd_rigid_contact_replay(device):
    """Run a small body-body contact scene and return final rigid state arrays."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 0.0
    builder.default_shape_cfg.mu = 0.4
    builder.add_ground_plane()

    q_identity = wp.quat_identity()
    for i, z in enumerate([0.19, 0.57, 0.95]):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.02 * (i % 2), 0.01 * i, z), q_identity),
            label=f"determinism_box_{i}",
        )
        builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)

    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        friction_epsilon=1.0e-4,
        rigid_contact_hard=True,
        rigid_contact_k_start=1.0e5,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit", deterministic=True)
    contacts = pipeline.contacts()

    for _ in range(4):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.body_qd.numpy().copy()


def _rigid_body_contacts_replay_bit_identically(test, device):
    """Repeated VBD rigid contact runs should produce bit-identical rigid states."""
    with wp.ScopedDevice(device):
        body_q_a, body_qd_a = _run_vbd_rigid_contact_replay(device)
        body_q_b, body_qd_b = _run_vbd_rigid_contact_replay(device)

    np.testing.assert_array_equal(body_q_a, body_q_b)
    np.testing.assert_array_equal(body_qd_a, body_qd_b)


class TestSolverVBD(unittest.TestCase):
    pass


add_function_test(
    TestSolverVBD, "test_self_contact_barrier_c2_at_tau", test_self_contact_barrier_c2_at_tau, devices=devices
)
add_function_test(
    TestSolverVBD, "test_self_contact_barrier_c2_at_d_min", test_self_contact_barrier_c2_at_d_min, devices=devices
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_restore_from_match_index",
    _rigid_contact_history_restore_from_match_index,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_soft_restores_penalty_only",
    _rigid_contact_history_soft_restores_penalty_only,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_joint_angular_dual_projects_free_axis_lambda",
    _joint_angular_dual_projects_free_axis_lambda,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_joint_force_projection_filters_free_direction",
    _joint_force_projection_filters_free_direction,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_d6_fully_free_structural_slots_are_inactive",
    _d6_fully_free_structural_slots_are_inactive,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_snapshot_copies_active_rows",
    _rigid_contact_history_snapshot_copies_active_rows,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_stick_flags_require_cone_and_small_residual",
    _rigid_contact_stick_flags_require_cone_and_small_residual,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_body_contact_indices_are_sorted_for_deterministic_traversal",
    _rigid_body_contact_indices_are_sorted_for_deterministic_traversal,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_sorted_rigid_contact_rows_make_solve_order_deterministic",
    _sorted_rigid_contact_rows_make_solve_order_deterministic,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_body_contacts_replay_bit_identically",
    _rigid_body_contacts_replay_bit_identically,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
