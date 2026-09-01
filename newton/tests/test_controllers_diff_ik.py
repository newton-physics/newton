# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the differential-kinematics controllers.

Kernel-level tests (:class:`TestDiffIkKernels`) exercise each Warp kernel in
``newton._src.controllers.impl.diff_ik._common`` directly against a
hand-derived numpy reference, with no :class:`Controller` involved.
Controller-class-level tests (:class:`TestControllerDiffIKModelFree`)
exercise :class:`~newton.controllers.ControllerDiffIKModelFree` — the
construction/validation/port-plumbing layer built on top of those kernels.
``model_based.py`` tests are added alongside it in a later chunk.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.controllers.impl.diff_ik._common import (
    _build_jjt_plus_damping_kernel,
    _cholesky_solve6_kernel,
    _integrate_position_kernel,
    _pose_error_kernel,
    _qd_from_y_kernel,
)
from newton._src.controllers.impl.diff_ik.model_free import ControllerDiffIKModelFree
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


# ---------------------------------------------------------------------------
# _pose_error_kernel
# ---------------------------------------------------------------------------


def test_pose_error_zero_when_poses_match(test: unittest.TestCase, device):
    pose = wp.array(
        [wp.transform(p=wp.vec3(1.0, 2.0, 3.0), q=wp.quat_rpy(0.3, -0.2, 0.5))], dtype=wp.transform, device=device
    )
    error = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_pose_error_kernel, dim=1, inputs=[pose, pose], outputs=[error], device=device)
    np.testing.assert_allclose(error.numpy(), np.zeros((1, 6)), atol=1e-6)


def test_pose_error_position_only(test: unittest.TestCase, device):
    identity_quat = wp.quat_identity()
    current = wp.array([wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=identity_quat)], dtype=wp.transform, device=device)
    desired = wp.array([wp.transform(p=wp.vec3(1.0, -2.0, 0.5), q=identity_quat)], dtype=wp.transform, device=device)
    error = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_pose_error_kernel, dim=1, inputs=[current, desired], outputs=[error], device=device)
    np.testing.assert_allclose(error.numpy()[0], [1.0, -2.0, 0.5, 0.0, 0.0, 0.0], atol=1e-6)


def test_pose_error_orientation_matches_axis_angle(test: unittest.TestCase, device):
    identity_quat = wp.quat_identity()
    axis = np.array([0.0, 0.0, 1.0])
    angle = 0.4
    desired_quat = wp.quat_from_axis_angle(wp.vec3(*axis), angle)
    current = wp.array([wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=identity_quat)], dtype=wp.transform, device=device)
    desired = wp.array([wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=desired_quat)], dtype=wp.transform, device=device)
    error = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_pose_error_kernel, dim=1, inputs=[current, desired], outputs=[error], device=device)
    np.testing.assert_allclose(error.numpy()[0][3:], axis * angle, atol=1e-5)


def test_pose_error_small_angle_is_finite(test: unittest.TestCase, device):
    """A near-identical orientation must not divide-by-zero into NaN."""
    identity_quat = wp.quat_identity()
    tiny_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1e-7)
    current = wp.array([wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=identity_quat)], dtype=wp.transform, device=device)
    desired = wp.array([wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=tiny_quat)], dtype=wp.transform, device=device)
    error = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_pose_error_kernel, dim=1, inputs=[current, desired], outputs=[error], device=device)
    result = error.numpy()[0]
    test.assertTrue(np.all(np.isfinite(result)))
    np.testing.assert_allclose(result[3:], [0.0, 0.0, 1e-7], atol=1e-8)


def test_pose_error_multiple_robots_independent(test: unittest.TestCase, device):
    identity_quat = wp.quat_identity()
    current = wp.array(
        [
            wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=identity_quat),
            wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=identity_quat),
        ],
        dtype=wp.transform,
        device=device,
    )
    desired = wp.array(
        [
            wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=identity_quat),
            wp.transform(p=wp.vec3(0.0, 2.0, 0.0), q=identity_quat),
        ],
        dtype=wp.transform,
        device=device,
    )
    error = wp.zeros(2, dtype=wp.spatial_vector, device=device)
    wp.launch(_pose_error_kernel, dim=2, inputs=[current, desired], outputs=[error], device=device)
    np.testing.assert_allclose(error.numpy()[:, :3], [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], atol=1e-6)


# ---------------------------------------------------------------------------
# DLS solve: _build_jjt_plus_damping_kernel + _cholesky_solve6_kernel + _qd_from_y_kernel
# ---------------------------------------------------------------------------


def test_build_jjt_plus_damping_matches_formula(test: unittest.TestCase, device):
    rng = np.random.default_rng(0)
    max_dofs = 4
    dof_count_val = 4
    jacobian_np = rng.normal(size=(1, 6, max_dofs)).astype(np.float32)
    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    dof_count = wp.array([dof_count_val], dtype=wp.int32, device=device)
    damping = wp.array([0.1], dtype=wp.float32, device=device)
    out = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping],
        outputs=[out],
        device=device,
    )

    j = jacobian_np[0]
    expected = j @ j.T + 0.1**2 * np.eye(6)
    np.testing.assert_allclose(out.numpy()[0], expected, atol=1e-4)


def test_cholesky_solve6_matches_numpy_solve(test: unittest.TestCase, device):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(6, 6)).astype(np.float64)
    spd = a @ a.T + 6 * np.eye(6)  # guaranteed SPD
    rhs = rng.normal(size=6)
    expected = np.linalg.solve(spd, rhs)

    spd_matrix = wp.array3d(spd[None].astype(np.float32), dtype=float, device=device)
    rhs_arr = wp.array([wp.spatial_vector(*rhs.astype(np.float32))], dtype=wp.spatial_vector, device=device)
    cholesky_factor = wp.zeros((1, 6, 6), dtype=float, device=device)
    y = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_cholesky_solve6_kernel, dim=1, inputs=[spd_matrix, rhs_arr, cholesky_factor], outputs=[y], device=device)
    np.testing.assert_allclose(np.array(y.numpy()[0]), expected, atol=1e-3)


def test_qd_from_y_matches_formula(test: unittest.TestCase, device):
    rng = np.random.default_rng(2)
    max_dofs = 3
    jacobian_np = rng.normal(size=(1, 6, max_dofs)).astype(np.float32)
    y_np = rng.normal(size=6).astype(np.float32)
    bandwidth_np = np.array([2.0, 0.5, 1.0], dtype=np.float32)

    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    y = wp.array([wp.spatial_vector(*y_np)], dtype=wp.spatial_vector, device=device)
    bandwidth = wp.array(bandwidth_np, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0, 0, 0], dtype=wp.int32, device=device)
    slot_of_dof = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=3,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof],
        outputs=[joint_qd_target],
        device=device,
    )
    expected = bandwidth_np * (jacobian_np[0].T @ y_np)
    np.testing.assert_allclose(joint_qd_target.numpy(), expected, atol=1e-4)


def test_dls_matches_ridge_regression_for_underactuated_robot(test: unittest.TestCase, device):
    """The fixed 6x6 JJᵀ+λ²I solve must equal the n x n JᵀJ+λ²I ridge solution when n < 6.

    This is the push-through identity Jᵀ(JJᵀ+λ²I)⁻¹ == (JᵀJ+λ²I)⁻¹Jᵀ, which
    holds for any shape of J as long as λ > 0 — so a robot with fewer than 6
    controlled DOFs does not need a different code path.
    """
    rng = np.random.default_rng(5)
    n_joints = 3
    max_dofs = 6  # padded width; only the first n_joints columns are used
    jacobian_np = np.zeros((1, 6, max_dofs), dtype=np.float32)
    jacobian_np[0, :, :n_joints] = rng.normal(size=(6, n_joints))
    error_np = rng.normal(size=6).astype(np.float32)
    damping_val = 0.2

    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    dof_count = wp.array([n_joints], dtype=wp.int32, device=device)
    damping = wp.array([damping_val], dtype=wp.float32, device=device)
    error = wp.array([wp.spatial_vector(*error_np)], dtype=wp.spatial_vector, device=device)

    jjt = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping],
        outputs=[jjt],
        device=device,
    )
    cholesky_factor = wp.zeros((1, 6, 6), dtype=float, device=device)
    y = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_cholesky_solve6_kernel, dim=1, inputs=[jjt, error, cholesky_factor], outputs=[y], device=device)

    bandwidth = wp.ones(n_joints, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * n_joints, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(n_joints, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(n_joints, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=n_joints,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof],
        outputs=[joint_qd_target],
        device=device,
    )

    j64 = jacobian_np[0, :, :n_joints].astype(np.float64)
    e64 = error_np.astype(np.float64)
    ridge_expected = np.linalg.solve(j64.T @ j64 + damping_val**2 * np.eye(n_joints), j64.T @ e64)
    np.testing.assert_allclose(joint_qd_target.numpy(), ridge_expected, atol=1e-3)


def test_dls_heterogeneous_dof_counts_independent(test: unittest.TestCase, device):
    """A batch mixing a 3-DOF and a 7-DOF robot solves each correctly, with no cross-talk."""
    rng = np.random.default_rng(6)
    max_dofs = 7
    dof_counts = [3, 7]
    robot_count = len(dof_counts)
    jacobian_np = np.zeros((robot_count, 6, max_dofs), dtype=np.float32)
    for robot_idx, n in enumerate(dof_counts):
        jacobian_np[robot_idx, :, :n] = rng.normal(size=(6, n))
    error_np = rng.normal(size=(robot_count, 6)).astype(np.float32)
    damping_val = 0.15

    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    dof_count = wp.array(dof_counts, dtype=wp.int32, device=device)
    damping = wp.array([damping_val] * robot_count, dtype=wp.float32, device=device)
    error = wp.array([wp.spatial_vector(*row) for row in error_np], dtype=wp.spatial_vector, device=device)

    jjt = wp.zeros((robot_count, 6, 6), dtype=float, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(robot_count, 6, 6),
        inputs=[jacobian, dof_count, damping],
        outputs=[jjt],
        device=device,
    )
    cholesky_factor = wp.zeros((robot_count, 6, 6), dtype=float, device=device)
    y = wp.zeros(robot_count, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _cholesky_solve6_kernel, dim=robot_count, inputs=[jjt, error, cholesky_factor], outputs=[y], device=device
    )

    total_dofs = sum(dof_counts)
    bandwidth = wp.ones(total_dofs, dtype=wp.float32, device=device)
    robot_of_dof = wp.array(
        np.repeat(np.arange(robot_count, dtype=np.int32), dof_counts), dtype=wp.int32, device=device
    )
    slot_of_dof = wp.array(
        np.concatenate([np.arange(n, dtype=np.int32) for n in dof_counts]), dtype=wp.int32, device=device
    )
    joint_qd_target = wp.zeros(total_dofs, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=total_dofs,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof],
        outputs=[joint_qd_target],
        device=device,
    )

    joint_qd_np = joint_qd_target.numpy()
    offset = 0
    for robot_idx, n in enumerate(dof_counts):
        j64 = jacobian_np[robot_idx, :, :n].astype(np.float64)
        e64 = error_np[robot_idx].astype(np.float64)
        expected = np.linalg.solve(j64.T @ j64 + damping_val**2 * np.eye(n), j64.T @ e64)
        np.testing.assert_allclose(joint_qd_np[offset : offset + n], expected, atol=1e-3)
        offset += n


def test_dls_zero_damping_is_pseudo_inverse(test: unittest.TestCase, device):
    """λ=0 reduces DLS to the ordinary Moore-Penrose pseudo-inverse for a full-rank Jacobian."""
    rng = np.random.default_rng(4)
    max_dofs = 6
    jacobian_np = rng.normal(size=(1, 6, max_dofs)).astype(np.float32)
    error_np = rng.normal(size=6).astype(np.float32)

    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    dof_count = wp.array([max_dofs], dtype=wp.int32, device=device)
    damping = wp.array([0.0], dtype=wp.float32, device=device)
    error = wp.array([wp.spatial_vector(*error_np)], dtype=wp.spatial_vector, device=device)

    jjt = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping],
        outputs=[jjt],
        device=device,
    )
    cholesky_factor = wp.zeros((1, 6, 6), dtype=float, device=device)
    y = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_cholesky_solve6_kernel, dim=1, inputs=[jjt, error, cholesky_factor], outputs=[y], device=device)
    bandwidth = wp.ones(max_dofs, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * max_dofs, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(max_dofs, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(max_dofs, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=max_dofs,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof],
        outputs=[joint_qd_target],
        device=device,
    )

    j64 = jacobian_np[0].astype(np.float64)
    expected = np.linalg.pinv(j64) @ error_np.astype(np.float64)
    np.testing.assert_allclose(joint_qd_target.numpy(), expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Golden-value tests, adapted from the WIP differential-kinematics
# controller's own hand-derived analytical tests. Only the cases that apply
# to this controller's current scope (fixed 6D pose task, damped-least-
# squares solver) are ported here; the rest belong in the chunks that add
# position-only tasks, the transpose method, and adaptive damping.
# ---------------------------------------------------------------------------


def test_pinv_identity_jacobian_matches_error_exactly(test: unittest.TestCase, device):
    """PINV (λ=0) on J = I_6x6: qd = pos_err padded with zero orientation rows, exactly.

    With a square identity Jacobian, J⁺ = I, so the solver output is exactly
    the raw pose error.
    """
    pos_err = np.array([0.1, 0.05, -0.03], dtype=np.float32)
    error_np = np.concatenate([pos_err, np.zeros(3, dtype=np.float32)])
    jacobian_np = np.eye(6, dtype=np.float32)[None]

    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    dof_count = wp.array([6], dtype=wp.int32, device=device)
    damping = wp.array([0.0], dtype=wp.float32, device=device)
    error = wp.array([wp.spatial_vector(*error_np)], dtype=wp.spatial_vector, device=device)

    jjt = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping],
        outputs=[jjt],
        device=device,
    )
    cholesky_factor = wp.zeros((1, 6, 6), dtype=float, device=device)
    y = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_cholesky_solve6_kernel, dim=1, inputs=[jjt, error, cholesky_factor], outputs=[y], device=device)
    bandwidth = wp.ones(6, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * 6, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(6, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=6,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof],
        outputs=[joint_qd_target],
        device=device,
    )

    np.testing.assert_allclose(joint_qd_target.numpy(), error_np, atol=1e-5)


def test_dls_pipeline_zero_when_poses_match(test: unittest.TestCase, device):
    """Full pipeline (pose error -> DLS solve) with matching poses gives zero qd."""
    pose = wp.array([wp.transform(p=wp.vec3(0.3, -0.1, 0.5), q=wp.quat_identity())], dtype=wp.transform, device=device)
    error = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_pose_error_kernel, dim=1, inputs=[pose, pose], outputs=[error], device=device)

    jacobian_np = np.eye(6, dtype=np.float32)[None]
    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    dof_count = wp.array([6], dtype=wp.int32, device=device)
    damping = wp.array([0.5], dtype=wp.float32, device=device)

    jjt = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping],
        outputs=[jjt],
        device=device,
    )
    cholesky_factor = wp.zeros((1, 6, 6), dtype=float, device=device)
    y = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_cholesky_solve6_kernel, dim=1, inputs=[jjt, error, cholesky_factor], outputs=[y], device=device)
    bandwidth = wp.ones(6, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * 6, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(6, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=6,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof],
        outputs=[joint_qd_target],
        device=device,
    )

    np.testing.assert_allclose(joint_qd_target.numpy(), np.zeros(6), atol=1e-6)


def test_dls_rotation_error_axis_angle_magnitude(test: unittest.TestCase, device):
    """30 deg rotation about x with J=I_6x6, PINV: qd[3] equals the rotation angle exactly.

    Chains :func:`_pose_error_kernel` with the DLS solve to validate the
    axis-angle error convention end to end, not just the error kernel alone.
    """
    angle = math.pi / 6
    identity_quat = wp.quat_identity()
    target_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)
    current = wp.array([wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=identity_quat)], dtype=wp.transform, device=device)
    desired = wp.array([wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=target_quat)], dtype=wp.transform, device=device)
    error = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_pose_error_kernel, dim=1, inputs=[current, desired], outputs=[error], device=device)

    jacobian_np = np.eye(6, dtype=np.float32)[None]
    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    dof_count = wp.array([6], dtype=wp.int32, device=device)
    damping = wp.array([0.0], dtype=wp.float32, device=device)

    jjt = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping],
        outputs=[jjt],
        device=device,
    )
    cholesky_factor = wp.zeros((1, 6, 6), dtype=float, device=device)
    y = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_cholesky_solve6_kernel, dim=1, inputs=[jjt, error, cholesky_factor], outputs=[y], device=device)
    bandwidth = wp.ones(6, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * 6, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(6, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=6,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof],
        outputs=[joint_qd_target],
        device=device,
    )

    result = joint_qd_target.numpy()
    np.testing.assert_allclose(result[:3], [0.0, 0.0, 0.0], atol=1e-5)
    test.assertAlmostEqual(float(result[3]), angle, places=5)
    np.testing.assert_allclose(result[4:], [0.0, 0.0], atol=1e-5)


def test_dls_one_dof_revolute_arm_matches_analytical_solution(test: unittest.TestCase, device):
    """A single revolute joint with a unit-length tool offset matches a hand-derived closed form.

    A revolute joint's Jacobian column has both a tangential-linear entry
    (index 1, from the unit-length tool offset) and a direct angular entry
    (index 5, since the joint's own velocity is the angular velocity about
    its axis). Solving ``(JJᵀ + λ²I)y = e`` via the Sherman-Morrison formula
    for this rank-1 ``J`` gives ``qd = bandwidth · error_y / (λ² + 2)``.
    """
    err_y = 0.1
    lam = 0.5
    bandwidth_val = 2.0

    jacobian_np = np.zeros((1, 6, 1), dtype=np.float32)
    jacobian_np[0, 1, 0] = 1.0  # tangential linear velocity from the unit-length tool offset
    jacobian_np[0, 5, 0] = 1.0  # the joint's own angular velocity about its axis
    error_np = np.zeros(6, dtype=np.float32)
    error_np[1] = err_y

    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    dof_count = wp.array([1], dtype=wp.int32, device=device)
    damping = wp.array([lam], dtype=wp.float32, device=device)
    error = wp.array([wp.spatial_vector(*error_np)], dtype=wp.spatial_vector, device=device)

    jjt = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping],
        outputs=[jjt],
        device=device,
    )
    cholesky_factor = wp.zeros((1, 6, 6), dtype=float, device=device)
    y = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(_cholesky_solve6_kernel, dim=1, inputs=[jjt, error, cholesky_factor], outputs=[y], device=device)
    bandwidth = wp.array([bandwidth_val], dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0], dtype=wp.int32, device=device)
    slot_of_dof = wp.array([0], dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=1,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof],
        outputs=[joint_qd_target],
        device=device,
    )

    expected = bandwidth_val * err_y / (2.0 + lam**2)
    test.assertAlmostEqual(float(joint_qd_target.numpy()[0]), expected, places=5)


# ---------------------------------------------------------------------------
# _integrate_position_kernel
# ---------------------------------------------------------------------------


def test_integrate_position_euler_step(test: unittest.TestCase, device):
    joint_q = wp.array([0.0, 1.0, -1.0], dtype=wp.float32, device=device)
    joint_qd_target = wp.array([1.0, -2.0, 0.5], dtype=wp.float32, device=device)
    dt = wp.array([0.1], dtype=wp.float32, device=device)
    joint_q_target = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(
        _integrate_position_kernel,
        dim=3,
        inputs=[joint_q, joint_qd_target, dt],
        outputs=[joint_q_target],
        device=device,
    )
    np.testing.assert_allclose(joint_q_target.numpy(), [0.1, 0.8, -0.95], atol=1e-6)


class TestDiffIkKernels(unittest.TestCase):
    pass


add_function_test(
    TestDiffIkKernels, "test_pose_error_zero_when_poses_match", test_pose_error_zero_when_poses_match, devices=devices
)
add_function_test(TestDiffIkKernels, "test_pose_error_position_only", test_pose_error_position_only, devices=devices)
add_function_test(
    TestDiffIkKernels,
    "test_pose_error_orientation_matches_axis_angle",
    test_pose_error_orientation_matches_axis_angle,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels, "test_pose_error_small_angle_is_finite", test_pose_error_small_angle_is_finite, devices=devices
)
add_function_test(
    TestDiffIkKernels,
    "test_pose_error_multiple_robots_independent",
    test_pose_error_multiple_robots_independent,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_build_jjt_plus_damping_matches_formula",
    test_build_jjt_plus_damping_matches_formula,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_cholesky_solve6_matches_numpy_solve",
    test_cholesky_solve6_matches_numpy_solve,
    devices=devices,
)
add_function_test(TestDiffIkKernels, "test_qd_from_y_matches_formula", test_qd_from_y_matches_formula, devices=devices)
add_function_test(
    TestDiffIkKernels,
    "test_dls_matches_ridge_regression_for_underactuated_robot",
    test_dls_matches_ridge_regression_for_underactuated_robot,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_dls_heterogeneous_dof_counts_independent",
    test_dls_heterogeneous_dof_counts_independent,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_dls_zero_damping_is_pseudo_inverse",
    test_dls_zero_damping_is_pseudo_inverse,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_pinv_identity_jacobian_matches_error_exactly",
    test_pinv_identity_jacobian_matches_error_exactly,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_dls_pipeline_zero_when_poses_match",
    test_dls_pipeline_zero_when_poses_match,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_dls_rotation_error_axis_angle_magnitude",
    test_dls_rotation_error_axis_angle_magnitude,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_dls_one_dof_revolute_arm_matches_analytical_solution",
    test_dls_one_dof_revolute_arm_matches_analytical_solution,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels, "test_integrate_position_euler_step", test_integrate_position_euler_step, devices=devices
)


# ---------------------------------------------------------------------------
# ControllerDiffIKModelFree
# ---------------------------------------------------------------------------


def _dofs_arr(dofs_list, device):
    """Return a wp.array[int32] from a list of per-robot DOF counts."""
    return wp.array(np.array(dofs_list, dtype=np.int32), device=device)


def _identity_jacobian(robot_count, max_dofs, device, num_rows=6):
    """Return a (robot_count, num_rows, max_dofs) identity-like Jacobian."""
    jacobian_np = np.zeros((robot_count, num_rows, max_dofs), dtype=np.float32)
    for i in range(min(num_rows, max_dofs)):
        jacobian_np[:, i, i] = 1.0
    return wp.array3d(jacobian_np, dtype=wp.float32, device=device)


def _identity_transform(robot_count, device):
    """Return a (robot_count,) array of identity transforms at the origin."""
    return wp.array(
        [wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())] * robot_count,
        dtype=wp.transform,
        device=device,
    )


class TestControllerDiffIKModelFree(unittest.TestCase):
    def test_zero_error_gives_zero_velocity(self):
        """When current tool pose equals the target pose exactly, qd_target must be zero."""
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.0, device=device
        )
        pose = _identity_transform(1, device)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = pose
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), np.zeros(6), atol=1e-6)
        np.testing.assert_allclose(outputs.joint_q_target.numpy(), np.zeros(6), atol=1e-6)

    def test_output_q_target_equals_joint_q_plus_qd_target_times_dt(self):
        device = wp.get_device()
        joint_q = [0.2, -0.3, 0.1, 0.0, 0.4, -0.1]
        dt = 0.02
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.5, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.array(joint_q, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(0.1, 0.0, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=dt)
        expected_q_target = np.array(joint_q, dtype=np.float32) + outputs.joint_qd_target.numpy() * dt
        np.testing.assert_allclose(outputs.joint_q_target.numpy(), expected_q_target, atol=1e-6)

    def test_pinv_identity_jacobian_matches_error_exactly(self):
        """J = I_6x6, λ=0: qd_target equals the raw pose error exactly."""
        device = wp.get_device()
        pos_err = np.array([0.1, 0.05, -0.03], dtype=np.float32)
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.0, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(*pos_err.tolist()), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        np.testing.assert_allclose(
            outputs.joint_qd_target.numpy(), np.concatenate([pos_err, np.zeros(3, dtype=np.float32)]), atol=1e-5
        )

    def test_rotation_error_axis_angle_magnitude(self):
        """30 deg rotation about x with J=I_6x6, λ=0: qd_target[3] equals the rotation angle exactly."""
        device = wp.get_device()
        angle = math.pi / 6
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.0, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        target_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=target_quat)], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        qd = outputs.joint_qd_target.numpy()
        np.testing.assert_allclose(qd[:3], [0.0, 0.0, 0.0], atol=1e-5)
        self.assertAlmostEqual(float(qd[3]), angle, places=5)
        np.testing.assert_allclose(qd[4:], [0.0, 0.0], atol=1e-5)

    def test_one_dof_revolute_arm_matches_analytical_solution(self):
        """A single revolute joint with a unit-length tool offset matches the hand-derived closed form.

        Same setup and formula as the kernel-level golden test, run through
        the full controller (construction, port validation, buffer wiring)
        instead of the raw kernels directly.
        """
        device = wp.get_device()
        err_y = 0.1
        lam = 0.5
        bandwidth_val = 2.0
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([1], device), bandwidth=bandwidth_val, damping=lam, device=device
        )
        jacobian_np = np.zeros((1, 6, 1), dtype=np.float32)
        jacobian_np[0, 1, 0] = 1.0
        jacobian_np[0, 5, 0] = 1.0
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(1, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(0.0, err_y, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        expected = bandwidth_val * err_y / (2.0 + lam**2)
        self.assertAlmostEqual(float(outputs.joint_qd_target.numpy()[0]), expected, places=5)

    def test_multiple_robots_independent(self):
        """Each robot's qd_target depends only on its own Jacobian and pose error."""
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6, 6], device), bandwidth=1.0, damping=0.0, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(12, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(2, device)
        inputs.desired_tool_pose_world = wp.array(
            [
                wp.transform(p=wp.vec3(0.1, 0.0, 0.0), q=wp.quat_identity()),
                wp.transform(p=wp.vec3(0.0, -0.2, 0.0), q=wp.quat_identity()),
            ],
            dtype=wp.transform,
            device=device,
        )
        inputs.jacobian_tool_world = _identity_jacobian(2, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        qd = outputs.joint_qd_target.numpy()
        np.testing.assert_allclose(qd[:6], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(qd[6:], [0.0, -0.2, 0.0, 0.0, 0.0, 0.0], atol=1e-5)

    def test_heterogeneous_dof_counts(self):
        """A 3-DOF robot and a 7-DOF robot in the same batch each match their own ridge-regression solution."""
        device = wp.get_device()
        rng = np.random.default_rng(7)
        dof_counts = [3, 7]
        max_dofs = 7
        lam = 0.2
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr(dof_counts, device), bandwidth=1.0, damping=lam, device=device
        )

        jacobian_np = np.zeros((2, 6, max_dofs), dtype=np.float32)
        for robot_idx, n in enumerate(dof_counts):
            jacobian_np[robot_idx, :, :n] = rng.normal(size=(6, n))
        error_np = rng.normal(size=(2, 6)).astype(np.float32)
        # Orientation error can't be set directly through a transform for an
        # arbitrary small-angle vector, so only the position rows are
        # exercised here; test_dls_heterogeneous_dof_counts_independent
        # (kernel-level) already covers the full 6D error.
        error_np[:, 3:] = 0.0

        total_dofs = sum(dof_counts)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(total_dofs, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(2, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(*error_np[i, :3].tolist()), q=wp.quat_identity()) for i in range(2)],
            dtype=wp.transform,
            device=device,
        )
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        qd = outputs.joint_qd_target.numpy()

        offset = 0
        for robot_idx, n in enumerate(dof_counts):
            j64 = jacobian_np[robot_idx, :, :n].astype(np.float64)
            e64 = error_np[robot_idx].astype(np.float64)
            expected = np.linalg.solve(j64.T @ j64 + lam**2 * np.eye(n), j64.T @ e64)
            np.testing.assert_allclose(qd[offset : offset + n], expected, atol=1e-3)
            offset += n

    def test_live_bandwidth_port(self):
        device = wp.get_device()
        pos_err = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=None, damping=0.0, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(*pos_err.tolist()), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        inputs.bandwidth = wp.full(6, 3.0, dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy()[:3], pos_err * 3.0, atol=1e-5)

    def test_live_damping_port(self):
        device = wp.get_device()
        err_y = 0.1
        lam = 0.5
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([1], device), bandwidth=1.0, damping=None, device=device
        )
        jacobian_np = np.zeros((1, 6, 1), dtype=np.float32)
        jacobian_np[0, 1, 0] = 1.0
        jacobian_np[0, 5, 0] = 1.0
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(1, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(0.0, err_y, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        inputs.damping = wp.array([lam], dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        expected = err_y / (2.0 + lam**2)
        self.assertAlmostEqual(float(outputs.joint_qd_target.numpy()[0]), expected, places=5)

    def test_dt_as_wp_array(self):
        device = wp.get_device()
        dt_scalar = 0.02
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.5, device=device
        )
        pose = _identity_transform(1, device)
        desired = wp.array(
            [wp.transform(p=wp.vec3(0.1, 0.0, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        jacobian = _identity_jacobian(1, 6, device)

        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = desired
        inputs.jacobian_tool_world = jacobian
        ctrl.step(inputs=inputs, outputs=outputs, dt=dt_scalar)
        qd_scalar = outputs.joint_qd_target.numpy().copy()
        q_target_scalar = outputs.joint_q_target.numpy().copy()

        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = desired
        inputs.jacobian_tool_world = jacobian
        ctrl.step(inputs=inputs, outputs=outputs, dt=wp.array([dt_scalar], dtype=wp.float32, device=device))
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), qd_scalar, atol=1e-6)
        np.testing.assert_allclose(outputs.joint_q_target.numpy(), q_target_scalar, atol=1e-6)

    def test_is_graphable(self):
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.5, device=device
        )
        self.assertTrue(ctrl.is_graphable())

    def test_inputs_bandwidth_and_damping_none_when_baked(self):
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.5, device=device
        )
        inputs = ctrl.input()
        self.assertIsNone(inputs.bandwidth)
        self.assertIsNone(inputs.damping)

    def test_inputs_bandwidth_and_damping_allocated_when_live(self):
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=None, damping=None, device=device
        )
        inputs = ctrl.input()
        self.assertIsNotNone(inputs.bandwidth)
        self.assertIsNotNone(inputs.damping)
        self.assertEqual(inputs.bandwidth.shape, (6,))
        self.assertEqual(inputs.damping.shape, (1,))

    def test_indexed_view_input_gathers(self):
        """A tool-pose input bound to an indexed view is read correctly."""
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.0, device=device
        )
        sim_poses = wp.array(
            [
                wp.transform(p=wp.vec3(9.0, 9.0, 9.0), q=wp.quat_identity()),
                wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            ],
            dtype=wp.transform,
            device=device,
        )
        view_idx = wp.array([1], dtype=wp.int32, device=device)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = sim_poses[view_idx]
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(0.2, 0.0, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy()[:3], [0.2, 0.0, 0.0], atol=1e-5)

    def test_indexed_view_float32_rank1_input_gathers(self):
        """A joint_q input bound to a float32 indexed view is read correctly.

        Distinct from test_indexed_view_input_gathers: that one exercises
        _gather_rank1_port_kernel's wp.transform instantiation, this one its
        wp.float32 instantiation — Warp compiles a separate concrete kernel
        per dtype from the same generic body, so neither test covers both.
        """
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([1], device), bandwidth=1.0, damping=0.5, device=device
        )
        sim_q = wp.array([9.0, 0.3], dtype=wp.float32, device=device)
        view_idx = wp.array([1], dtype=wp.int32, device=device)
        pose = _identity_transform(1, device)
        desired = wp.array(
            [wp.transform(p=wp.vec3(0.3, 0.0, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = sim_q[view_idx]
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = desired
        inputs.jacobian_tool_world = _identity_jacobian(1, 1, device, num_rows=6)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.1)
        # joint_q only feeds the integration step, so its gathered value
        # (0.3, not the un-viewed 9.0) shows up in q_target, not qd_target.
        np.testing.assert_allclose(
            outputs.joint_q_target.numpy()[0], 0.3 + outputs.joint_qd_target.numpy()[0] * 0.1, atol=1e-5
        )

    def test_indexed_view_jacobian_input_gathers(self):
        """A jacobian_tool_world input bound to a rank-3 indexed view is read correctly.

        Exercises _gather_rank3_port_kernel, which only runs when the
        Jacobian port is bound to a view — otherwise silently uncovered.
        """
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([1], device), bandwidth=1.0, damping=0.0, device=device
        )
        sim_jacobian = wp.array3d(
            np.stack([np.zeros((6, 1), dtype=np.float32), _identity_jacobian(1, 1, device).numpy()[0]]),
            dtype=wp.float32,
            device=device,
        )
        view_idx = wp.array([1], dtype=wp.int32, device=device)
        pose = _identity_transform(1, device)
        desired = wp.array(
            [wp.transform(p=wp.vec3(0.4, 0.0, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(1, dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = desired
        inputs.jacobian_tool_world = sim_jacobian[view_idx]
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        # With the viewed (identity) Jacobian, qd = pose error exactly; the
        # un-viewed (all-zero) block at index 0 would instead give qd = 0.
        np.testing.assert_allclose(outputs.joint_qd_target.numpy()[0], 0.4, atol=1e-5)

    def test_indexed_view_output_scatters(self):
        """An output bound to an indexed view is written correctly."""
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.0, device=device
        )
        inputs = ctrl.input()
        sim_qd = wp.zeros(12, dtype=wp.float32, device=device)
        view_idx = wp.array(np.arange(6, 12, dtype=np.int32), dtype=wp.int32, device=device)
        outputs = ctrl.output()
        outputs.joint_qd_target = sim_qd[view_idx]
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(0.1, 0.0, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        np.testing.assert_allclose(sim_qd.numpy()[:6], np.zeros(6), atol=1e-8)
        np.testing.assert_allclose(sim_qd.numpy()[6:9], [0.1, 0.0, 0.0], atol=1e-5)

    def test_disabled_bandwidth_port_written_raises(self):
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.0, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = _identity_transform(1, device)
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        inputs.bandwidth = wp.full(6, 1.0, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

    def test_wrong_shape_jacobian_raises(self):
        device = wp.get_device()
        ctrl = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.0, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = _identity_transform(1, device)
        inputs.jacobian_tool_world = wp.zeros((1, 6, 3), dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

    def test_zero_dof_robot_raises(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDiffIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6, 0], device), bandwidth=1.0, damping=0.0, device=device
            )

    def test_controlled_dofs_per_robot_is_copied(self):
        """A later mutation of the caller's own array must not affect the controller."""
        device = wp.get_device()
        dofs = _dofs_arr([6], device)
        ctrl = ControllerDiffIKModelFree(controlled_dofs_per_robot=dofs, bandwidth=1.0, damping=0.0, device=device)
        dofs.assign(np.array([1], dtype=np.int32))

        pose = _identity_transform(1, device)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = pose
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        # If the mutation had propagated, total_controlled_dofs would now be
        # 1, and running a length-6 port through the controller would raise
        # a shape mismatch instead of succeeding.
        self.assertEqual(ctrl.total_controlled_dofs, 6)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), np.zeros(6), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
