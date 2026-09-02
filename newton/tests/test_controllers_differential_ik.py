# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the differential-kinematics controllers.

Kernel-level tests (:class:`TestDiffIkKernels`) exercise each Warp kernel in
``newton._src.controllers.impl.differential_ik._common`` directly against a
hand-derived numpy reference, with no :class:`Controller` involved.
Controller-class-level tests (:class:`TestControllerDifferentialIKModelFree`)
exercise :class:`~newton.controllers.ControllerDifferentialIKModelFree` — the
construction/validation/port-plumbing layer built on top of those kernels.
``model_based.py`` tests are added alongside it in a later chunk.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.controllers.impl._common import (
    _add_term_kernel,
    _apply_spatial_matrix_kernel,
    _block_matrix_vector_multiply_kernel,
    _invert_spd_block_kernel,
    _pose_error_kernel,
)
from newton._src.controllers.impl.differential_ik._common import (
    IkMethod,
    _adaptive_damping_kernel,
    _build_jjt_plus_damping_kernel,
    _gather_jacobian_by_axis_kernel,
    _gather_task_error_kernel,
    _integrate_position_kernel,
    _joint_limit_avoidance_bias_kernel,
    _posture_bias_kernel,
    _qd_from_y_kernel,
    _scatter_pinv_transpose_by_axis_kernel,
    _smallest_eigenvalue_spd6_kernel,
    _truncated_pinv_matrix_kernel,
)
from newton._src.controllers.impl.differential_ik.model_based import ControllerDifferentialIK
from newton._src.controllers.impl.differential_ik.model_free import ControllerDifferentialIKModelFree
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


def _solve6(jjt, error, device):
    """Solve ``jjt @ y = error`` via ``_invert_spd_block_kernel`` + ``_apply_spatial_matrix_kernel``."""
    robot_count = jjt.shape[0]
    block_dim = wp.full(robot_count, 6, dtype=wp.int32, device=device)
    cholesky_factor = wp.zeros((robot_count, 6, 6), dtype=float, device=device)
    jjt_inv = wp.zeros((robot_count, 6, 6), dtype=float, device=device)
    wp.launch(
        _invert_spd_block_kernel,
        dim=robot_count,
        inputs=[jjt, block_dim, cholesky_factor],
        outputs=[jjt_inv],
        device=device,
    )
    y = wp.zeros(robot_count, dtype=wp.spatial_vector, device=device)
    wp.launch(_apply_spatial_matrix_kernel, dim=robot_count, inputs=[jjt_inv, error], outputs=[y], device=device)
    return y


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
# DLS solve: _build_jjt_plus_damping_kernel + _solve6 (_invert_spd_block_kernel + _apply_spatial_matrix_kernel) + _qd_from_y_kernel
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
    task_dim = wp.full(1, 6, dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(np.tile(np.arange(6, dtype=np.int32), (1, 1)), dtype=wp.int32, device=device)
    axis_weight = wp.full(1, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping, task_dim, active_axis_of_slot, axis_weight],
        outputs=[out],
        device=device,
    )

    j = jacobian_np[0]
    expected = j @ j.T + 0.1**2 * np.eye(6)
    np.testing.assert_allclose(out.numpy()[0], expected, atol=1e-4)


def test_build_jjt_plus_damping_task_dim_3_confines_to_top_left_block(test: unittest.TestCase, device):
    rng = np.random.default_rng(1)
    jacobian_np = rng.normal(size=(1, 6, 4)).astype(np.float32)
    jacobian = wp.array3d(jacobian_np, dtype=float, device=device)
    dof_count = wp.array([4], dtype=wp.int32, device=device)
    damping = wp.array([0.1], dtype=wp.float32, device=device)
    task_dim = wp.array([3], dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(np.array([[0, 1, 2, 0, 0, 0]], dtype=np.int32), dtype=wp.int32, device=device)
    axis_weight = wp.full(1, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    out = wp.zeros((1, 6, 6), dtype=float, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping, task_dim, active_axis_of_slot, axis_weight],
        outputs=[out],
        device=device,
    )
    out_np = out.numpy()[0]
    expected_top_left = jacobian_np[0, :3, :] @ jacobian_np[0, :3, :].T + 0.1**2 * np.eye(3)
    np.testing.assert_allclose(out_np[:3, :3], expected_top_left, atol=1e-4)
    np.testing.assert_allclose(out_np[3:, :], 0.0, atol=0)
    np.testing.assert_allclose(out_np[:, 3:], 0.0, atol=0)


def test_gather_task_error_matches_active_axis_and_weight(test: unittest.TestCase, device):
    pose_error = wp.array(
        [wp.spatial_vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), wp.spatial_vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)],
        dtype=wp.spatial_vector,
        device=device,
    )
    task_dim = wp.array([3, 6], dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(np.tile(np.arange(6, dtype=np.int32), (2, 1)), dtype=wp.int32, device=device)
    axis_weight = wp.array(
        [wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), wp.spatial_vector(1.0, 2.0, 1.0, 1.0, 1.0, 1.0)],
        dtype=wp.spatial_vector,
        device=device,
    )  # robot 1: also check a nonzero, non-unit weight
    pose_error_active = wp.zeros(2, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _gather_task_error_kernel,
        dim=2,
        inputs=[pose_error, task_dim, active_axis_of_slot, axis_weight],
        outputs=[pose_error_active],
        device=device,
    )
    out = pose_error_active.numpy()
    np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [1.0, 4.0, 3.0, 4.0, 5.0, 6.0])


def test_gather_jacobian_by_axis_matches_expected_rows(test: unittest.TestCase, device):
    """Null-space path: gathering an arbitrary (non-prefix) axis combo into compact slot order, unweighted."""
    rng = np.random.default_rng(14)
    jacobian_np = rng.normal(size=(1, 6, 4)).astype(np.float32)
    jacobian = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
    active_axes = [0, 2, 3, 4]  # drop axis 1 (Y position) and axis 5 (yaw) -- a non-prefix combo
    task_dim = wp.array([len(active_axes)], dtype=wp.int32, device=device)
    active_axis_of_slot_np = np.zeros((1, 6), dtype=np.int32)
    active_axis_of_slot_np[0, : len(active_axes)] = active_axes
    active_axis_of_slot = wp.array2d(active_axis_of_slot_np, dtype=wp.int32, device=device)
    jacobian_active = wp.zeros((1, 6, 4), dtype=wp.float32, device=device)
    wp.launch(
        _gather_jacobian_by_axis_kernel,
        dim=(1, 6, 4),
        inputs=[jacobian, active_axis_of_slot, task_dim],
        outputs=[jacobian_active],
        device=device,
    )
    out = jacobian_active.numpy()[0]
    np.testing.assert_allclose(out[: len(active_axes)], jacobian_np[0, active_axes, :], atol=1e-5)
    np.testing.assert_allclose(out[len(active_axes) :], 0.0, atol=0)


def test_scatter_pinv_transpose_by_axis_matches_expected_rows(test: unittest.TestCase, device):
    """Null-space path: scattering a compact-slot-order result back to its own canonical axis rows."""
    rng = np.random.default_rng(15)
    active_axes = [0, 2, 3, 4]
    task_dim = wp.array([len(active_axes)], dtype=wp.int32, device=device)
    active_axis_of_slot_np = np.zeros((1, 6), dtype=np.int32)
    active_axis_of_slot_np[0, : len(active_axes)] = active_axes
    active_axis_of_slot = wp.array2d(active_axis_of_slot_np, dtype=wp.int32, device=device)
    dof_count = wp.array([4], dtype=wp.int32, device=device)

    pinv_slot_np = np.zeros((1, 6, 4), dtype=np.float32)
    pinv_slot_np[0, : len(active_axes), :] = rng.normal(size=(len(active_axes), 4))
    pinv_slot = wp.array3d(pinv_slot_np, dtype=wp.float32, device=device)
    pinv_axis = wp.zeros((1, 6, 4), dtype=wp.float32, device=device)
    wp.launch(
        _scatter_pinv_transpose_by_axis_kernel,
        dim=(1, 6, 4),
        inputs=[pinv_slot, active_axis_of_slot, task_dim, dof_count],
        outputs=[pinv_axis],
        device=device,
    )
    out = pinv_axis.numpy()[0]
    for slot, axis in enumerate(active_axes):
        np.testing.assert_allclose(out[axis], pinv_slot_np[0, slot], atol=1e-6)
    for axis in [1, 5]:
        np.testing.assert_allclose(out[axis], 0.0, atol=0)


def test_smallest_eigenvalue_spd6_task_dim_3_skips_padding_zeros(test: unittest.TestCase, device):
    """A JJᵀ built with task_dim=3 has 3 guaranteed-zero padding eigenvalues; the real smallest must be found
    among the other 3, not the trivial global minimum (always 0 from the padding)."""
    rng = np.random.default_rng(12)
    j = rng.normal(size=(3, 4)).astype(np.float32)
    jjt3_np = (j @ j.T).astype(np.float32)
    padded = np.zeros((6, 6), dtype=np.float32)
    padded[:3, :3] = jjt3_np

    matrix = wp.array3d(padded[None], dtype=wp.float32, device=device)
    task_dim = wp.array([3], dtype=wp.int32, device=device)
    smallest = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(_smallest_eigenvalue_spd6_kernel, dim=1, inputs=[matrix, task_dim], outputs=[smallest], device=device)
    expected = max(np.linalg.eigvalsh(jjt3_np.astype(np.float64)).min(), 0.0)
    np.testing.assert_allclose(smallest.numpy()[0], expected, atol=1e-3)


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
    active_axis_of_slot = wp.array2d(np.tile(np.arange(6, dtype=np.int32), (1, 1)), dtype=wp.int32, device=device)
    axis_weight = wp.full(1, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    joint_qd_target = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=3,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof, active_axis_of_slot, axis_weight],
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
    task_dim = wp.full(1, 6, dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(np.tile(np.arange(6, dtype=np.int32), (1, 1)), dtype=wp.int32, device=device)
    axis_weight = wp.full(1, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping, task_dim, active_axis_of_slot, axis_weight],
        outputs=[jjt],
        device=device,
    )
    y = _solve6(jjt, error, device)

    bandwidth = wp.ones(n_joints, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * n_joints, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(n_joints, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(n_joints, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=n_joints,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof, active_axis_of_slot, axis_weight],
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
    task_dim = wp.full(robot_count, 6, dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(
        np.tile(np.arange(6, dtype=np.int32), (robot_count, 1)), dtype=wp.int32, device=device
    )
    axis_weight = wp.full(
        robot_count, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device
    )
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(robot_count, 6, 6),
        inputs=[jacobian, dof_count, damping, task_dim, active_axis_of_slot, axis_weight],
        outputs=[jjt],
        device=device,
    )
    y = _solve6(jjt, error, device)

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
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof, active_axis_of_slot, axis_weight],
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
    task_dim = wp.full(1, 6, dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(np.tile(np.arange(6, dtype=np.int32), (1, 1)), dtype=wp.int32, device=device)
    axis_weight = wp.full(1, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping, task_dim, active_axis_of_slot, axis_weight],
        outputs=[jjt],
        device=device,
    )
    y = _solve6(jjt, error, device)
    bandwidth = wp.ones(max_dofs, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * max_dofs, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(max_dofs, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(max_dofs, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=max_dofs,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof, active_axis_of_slot, axis_weight],
        outputs=[joint_qd_target],
        device=device,
    )

    j64 = jacobian_np[0].astype(np.float64)
    expected = np.linalg.pinv(j64) @ error_np.astype(np.float64)
    np.testing.assert_allclose(joint_qd_target.numpy(), expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Golden-value tests, hand-derived analytical closed forms for the
# damped-least-squares solve.
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
    task_dim = wp.full(1, 6, dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(np.tile(np.arange(6, dtype=np.int32), (1, 1)), dtype=wp.int32, device=device)
    axis_weight = wp.full(1, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping, task_dim, active_axis_of_slot, axis_weight],
        outputs=[jjt],
        device=device,
    )
    y = _solve6(jjt, error, device)
    bandwidth = wp.ones(6, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * 6, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(6, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=6,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof, active_axis_of_slot, axis_weight],
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
    task_dim = wp.full(1, 6, dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(np.tile(np.arange(6, dtype=np.int32), (1, 1)), dtype=wp.int32, device=device)
    axis_weight = wp.full(1, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping, task_dim, active_axis_of_slot, axis_weight],
        outputs=[jjt],
        device=device,
    )
    y = _solve6(jjt, error, device)
    bandwidth = wp.ones(6, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * 6, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(6, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=6,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof, active_axis_of_slot, axis_weight],
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
    task_dim = wp.full(1, 6, dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(np.tile(np.arange(6, dtype=np.int32), (1, 1)), dtype=wp.int32, device=device)
    axis_weight = wp.full(1, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping, task_dim, active_axis_of_slot, axis_weight],
        outputs=[jjt],
        device=device,
    )
    y = _solve6(jjt, error, device)
    bandwidth = wp.ones(6, dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0] * 6, dtype=wp.int32, device=device)
    slot_of_dof = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(6, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=6,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof, active_axis_of_slot, axis_weight],
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
    task_dim = wp.full(1, 6, dtype=wp.int32, device=device)
    active_axis_of_slot = wp.array2d(np.tile(np.arange(6, dtype=np.int32), (1, 1)), dtype=wp.int32, device=device)
    axis_weight = wp.full(1, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    wp.launch(
        _build_jjt_plus_damping_kernel,
        dim=(1, 6, 6),
        inputs=[jacobian, dof_count, damping, task_dim, active_axis_of_slot, axis_weight],
        outputs=[jjt],
        device=device,
    )
    y = _solve6(jjt, error, device)
    bandwidth = wp.array([bandwidth_val], dtype=wp.float32, device=device)
    robot_of_dof = wp.array([0], dtype=wp.int32, device=device)
    slot_of_dof = wp.array([0], dtype=wp.int32, device=device)
    joint_qd_target = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(
        _qd_from_y_kernel,
        dim=1,
        inputs=[jacobian, y, bandwidth, robot_of_dof, slot_of_dof, active_axis_of_slot, axis_weight],
        outputs=[joint_qd_target],
        device=device,
    )

    expected = bandwidth_val * err_y / (2.0 + lam**2)
    test.assertAlmostEqual(float(joint_qd_target.numpy()[0]), expected, places=5)


# ---------------------------------------------------------------------------
# _block_matrix_vector_multiply_kernel, _add_term_kernel
# ---------------------------------------------------------------------------


def test_block_matrix_vector_multiply_matches_formula(test: unittest.TestCase, device):
    rng = np.random.default_rng(10)
    dof_counts = [2, 3]
    max_dofs = 3
    block_matrix_np = np.zeros((2, max_dofs, max_dofs), dtype=np.float32)
    for robot_idx, n in enumerate(dof_counts):
        block_matrix_np[robot_idx, :n, :n] = rng.normal(size=(n, n))
    vec_np = rng.normal(size=sum(dof_counts)).astype(np.float32)

    block_matrix = wp.array3d(block_matrix_np, dtype=wp.float32, device=device)
    vec = wp.array(vec_np, dtype=wp.float32, device=device)
    robot_of_dof = wp.array(np.repeat(np.arange(2, dtype=np.int32), dof_counts), dtype=wp.int32, device=device)
    slot_of_dof = wp.array(
        np.concatenate([np.arange(n, dtype=np.int32) for n in dof_counts]), dtype=wp.int32, device=device
    )
    offsets_np = np.zeros(2, dtype=np.int32)
    offsets_np[1] = dof_counts[0]
    dof_offsets = wp.array(offsets_np, dtype=wp.int32, device=device)
    controlled_dofs_per_robot = wp.array(dof_counts, dtype=wp.int32, device=device)
    out = wp.zeros(sum(dof_counts), dtype=wp.float32, device=device)
    wp.launch(
        _block_matrix_vector_multiply_kernel,
        dim=sum(dof_counts),
        inputs=[block_matrix, vec, robot_of_dof, slot_of_dof, dof_offsets, controlled_dofs_per_robot],
        outputs=[out],
        device=device,
    )

    expected = np.concatenate(
        [
            block_matrix_np[0, : dof_counts[0], : dof_counts[0]] @ vec_np[: dof_counts[0]],
            block_matrix_np[1, : dof_counts[1], : dof_counts[1]] @ vec_np[dof_counts[0] :],
        ]
    )
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-4)


def test_add_term_accumulates(test: unittest.TestCase, device):
    accumulator = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    term = wp.array([0.5, -1.0, 2.0], dtype=wp.float32, device=device)
    wp.launch(_add_term_kernel, dim=3, inputs=[term], outputs=[accumulator], device=device)
    np.testing.assert_allclose(accumulator.numpy(), [1.5, 1.0, 5.0], atol=1e-6)


# ---------------------------------------------------------------------------
# _joint_limit_avoidance_bias_kernel, _posture_bias_kernel
# ---------------------------------------------------------------------------


def test_joint_limit_avoidance_zero_far_from_limits(test: unittest.TestCase, device):
    joint_q = wp.array([0.0], dtype=wp.float32, device=device)
    lower = wp.array([-1.0], dtype=wp.float32, device=device)
    upper = wp.array([1.0], dtype=wp.float32, device=device)
    dq_center = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(
        _joint_limit_avoidance_bias_kernel,
        dim=1,
        inputs=[joint_q, lower, upper, 5.0, 0.2],
        outputs=[dq_center],
        device=device,
    )
    np.testing.assert_allclose(dq_center.numpy(), [0.0], atol=1e-6)


def test_joint_limit_avoidance_full_correction_at_limit(test: unittest.TestCase, device):
    """At (or past) a limit, activation saturates to 1: bias = -gain * (q - q_mid)."""
    joint_q = wp.array([1.0], dtype=wp.float32, device=device)  # exactly at the upper limit
    lower = wp.array([-1.0], dtype=wp.float32, device=device)
    upper = wp.array([1.0], dtype=wp.float32, device=device)
    gain = 5.0
    dq_center = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(
        _joint_limit_avoidance_bias_kernel,
        dim=1,
        inputs=[joint_q, lower, upper, gain, 0.2],
        outputs=[dq_center],
        device=device,
    )
    # q_mid = 0, so bias = -gain * (1.0 - 0.0) = -gain, pulling back toward the midpoint.
    np.testing.assert_allclose(dq_center.numpy(), [-gain], atol=1e-6)


def test_joint_limit_avoidance_ramps_linearly_in_margin(test: unittest.TestCase, device):
    """Halfway into the margin, activation must be exactly 0.5."""
    margin = 0.2
    joint_q = wp.array([1.0 - margin / 2.0], dtype=wp.float32, device=device)  # margin/2 from the upper limit
    lower = wp.array([-1.0], dtype=wp.float32, device=device)
    upper = wp.array([1.0], dtype=wp.float32, device=device)
    gain = 4.0
    dq_center = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(
        _joint_limit_avoidance_bias_kernel,
        dim=1,
        inputs=[joint_q, lower, upper, gain, margin],
        outputs=[dq_center],
        device=device,
    )
    q = 1.0 - margin / 2.0
    expected = -gain * 0.5 * (q - 0.0)
    np.testing.assert_allclose(dq_center.numpy(), [expected], atol=1e-6)


def test_posture_bias_matches_formula(test: unittest.TestCase, device):
    joint_q = wp.array([0.0, 1.0], dtype=wp.float32, device=device)
    q_des_null = wp.array([0.5, 0.5], dtype=wp.float32, device=device)
    stiffness = wp.array([2.0, 3.0], dtype=wp.float32, device=device)
    dq_center = wp.zeros(2, dtype=wp.float32, device=device)
    wp.launch(_posture_bias_kernel, dim=2, inputs=[joint_q, q_des_null, stiffness], outputs=[dq_center], device=device)
    np.testing.assert_allclose(dq_center.numpy(), [1.0, -1.5], atol=1e-6)


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


def test_smallest_eigenvalue_spd6_matches_numpy(test: unittest.TestCase, device):
    rng = np.random.default_rng(11)
    j1 = rng.normal(size=(6, 6)).astype(np.float32)
    j2 = np.zeros((6, 4), dtype=np.float32)
    j2[:, :4] = rng.normal(size=(6, 4))  # rank <= 4: JJᵀ has (at least) two zero eigenvalues
    jjt_np = np.stack([j1 @ j1.T, j2 @ j2.T]).astype(np.float32)

    matrix = wp.array3d(jjt_np, dtype=wp.float32, device=device)
    task_dim = wp.full(2, 6, dtype=wp.int32, device=device)
    smallest_eigenvalue = wp.zeros(2, dtype=wp.float32, device=device)
    wp.launch(
        _smallest_eigenvalue_spd6_kernel, dim=2, inputs=[matrix, task_dim], outputs=[smallest_eigenvalue], device=device
    )
    expected = np.array([np.linalg.eigvalsh(jjt_np[i].astype(np.float64)).min() for i in range(2)])
    np.testing.assert_allclose(smallest_eigenvalue.numpy(), np.maximum(expected, 0.0), atol=1e-3)


def test_adaptive_damping_matches_formula(test: unittest.TestCase, device):
    smallest_eigenvalue = wp.array([0.0, 0.25, 100.0], dtype=wp.float32, device=device)  # sigma_min = 0, 0.5, 10
    damping_min = wp.array([0.1, 0.1, 0.1], dtype=wp.float32, device=device)
    damping_max = wp.array([2.0, 2.0, 2.0], dtype=wp.float32, device=device)
    threshold = wp.array([1.0, 1.0, 1.0], dtype=wp.float32, device=device)
    damping = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(
        _adaptive_damping_kernel,
        dim=3,
        inputs=[smallest_eigenvalue, damping_min, damping_max, threshold],
        outputs=[damping],
        device=device,
    )
    sigma_min = np.array([0.0, 0.5, 10.0])
    ratio = np.minimum(sigma_min / 1.0, 1.0)
    expected_sq = 0.1**2 + (1.0 - ratio**2) * (2.0**2 - 0.1**2)
    np.testing.assert_allclose(damping.numpy(), np.sqrt(expected_sq), atol=1e-5)
    # At the threshold and beyond, damping is exactly damping_min.
    test.assertAlmostEqual(float(damping.numpy()[2]), 0.1, places=5)
    # At full singularity, damping is exactly damping_max.
    test.assertAlmostEqual(float(damping.numpy()[0]), 2.0, places=5)


def test_truncated_pinv_matrix_matches_numpy(test: unittest.TestCase, device):
    rng = np.random.default_rng(13)
    j = rng.normal(size=(6, 4)).astype(np.float32)
    jjt_np = (j @ j.T).astype(np.float32)
    threshold = 0.05  # well below every singular value of a random 6x4 J, so nothing is truncated here

    matrix = wp.array3d(jjt_np[None], dtype=wp.float32, device=device)
    threshold_arr = wp.array([threshold], dtype=wp.float32, device=device)
    pinv_matrix = wp.zeros((1, 6, 6), dtype=wp.float32, device=device)
    wp.launch(
        _truncated_pinv_matrix_kernel, dim=1, inputs=[matrix, threshold_arr], outputs=[pinv_matrix], device=device
    )

    eigenvalues, eigenvectors = np.linalg.eigh(jjt_np.astype(np.float64))
    sigma = np.sqrt(np.maximum(eigenvalues, 0.0))
    # g(sigma) = 1/sigma^2 = 1/eigenvalue: the kernel inverts JJᵀ itself.
    g = np.where(sigma > threshold, 1.0 / np.maximum(eigenvalues, 1e-30), 0.0)
    expected = (eigenvectors * g) @ eigenvectors.T
    np.testing.assert_allclose(pinv_matrix.numpy()[0], expected, atol=1e-3)


def test_truncated_pinv_matrix_drops_singular_directions(test: unittest.TestCase, device):
    """A rank-3 JJᵀ (from a 3-DOF robot) has 3 exact-zero eigenvalues: those directions contribute nothing."""
    rng = np.random.default_rng(17)
    j = np.zeros((6, 3), dtype=np.float32)
    j[:, :3] = rng.normal(size=(6, 3))
    jjt_np = (j @ j.T).astype(np.float32)

    matrix = wp.array3d(jjt_np[None], dtype=wp.float32, device=device)
    # Above the QR eigensolver's own float32 noise floor for a structurally
    # zero eigenvalue (~1e-6, i.e. sigma ~1e-3) but well below the smallest
    # genuine singular value, so exactly the 3 structural directions drop.
    threshold_arr = wp.array([1.0e-2], dtype=wp.float32, device=device)
    pinv_matrix = wp.zeros((1, 6, 6), dtype=wp.float32, device=device)
    wp.launch(
        _truncated_pinv_matrix_kernel, dim=1, inputs=[matrix, threshold_arr], outputs=[pinv_matrix], device=device
    )
    # rank(pinv_matrix) == rank(jjt) == 3, not 6: the dropped directions
    # leave the matrix singular rather than merely small in those directions.
    rank = np.linalg.matrix_rank(pinv_matrix.numpy()[0].astype(np.float64), tol=1e-3)
    test.assertEqual(rank, 3)


class TestDiffIkKernels(unittest.TestCase):
    pass


add_function_test(
    TestDiffIkKernels, "test_pose_error_zero_when_poses_match", test_pose_error_zero_when_poses_match, devices=devices
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
    "test_build_jjt_plus_damping_task_dim_3_confines_to_top_left_block",
    test_build_jjt_plus_damping_task_dim_3_confines_to_top_left_block,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_gather_task_error_matches_active_axis_and_weight",
    test_gather_task_error_matches_active_axis_and_weight,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_gather_jacobian_by_axis_matches_expected_rows",
    test_gather_jacobian_by_axis_matches_expected_rows,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_scatter_pinv_transpose_by_axis_matches_expected_rows",
    test_scatter_pinv_transpose_by_axis_matches_expected_rows,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_smallest_eigenvalue_spd6_task_dim_3_skips_padding_zeros",
    test_smallest_eigenvalue_spd6_task_dim_3_skips_padding_zeros,
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
    TestDiffIkKernels,
    "test_block_matrix_vector_multiply_matches_formula",
    test_block_matrix_vector_multiply_matches_formula,
    devices=devices,
)
add_function_test(TestDiffIkKernels, "test_add_term_accumulates", test_add_term_accumulates, devices=devices)
add_function_test(
    TestDiffIkKernels,
    "test_joint_limit_avoidance_zero_far_from_limits",
    test_joint_limit_avoidance_zero_far_from_limits,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_joint_limit_avoidance_full_correction_at_limit",
    test_joint_limit_avoidance_full_correction_at_limit,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_joint_limit_avoidance_ramps_linearly_in_margin",
    test_joint_limit_avoidance_ramps_linearly_in_margin,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels, "test_posture_bias_matches_formula", test_posture_bias_matches_formula, devices=devices
)
add_function_test(
    TestDiffIkKernels, "test_integrate_position_euler_step", test_integrate_position_euler_step, devices=devices
)
add_function_test(
    TestDiffIkKernels,
    "test_smallest_eigenvalue_spd6_matches_numpy",
    test_smallest_eigenvalue_spd6_matches_numpy,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels, "test_adaptive_damping_matches_formula", test_adaptive_damping_matches_formula, devices=devices
)
add_function_test(
    TestDiffIkKernels,
    "test_truncated_pinv_matrix_matches_numpy",
    test_truncated_pinv_matrix_matches_numpy,
    devices=devices,
)
add_function_test(
    TestDiffIkKernels,
    "test_truncated_pinv_matrix_drops_singular_directions",
    test_truncated_pinv_matrix_drops_singular_directions,
    devices=devices,
)


# ---------------------------------------------------------------------------
# ControllerDifferentialIKModelFree
# ---------------------------------------------------------------------------


def _dofs_arr(dofs_list, device):
    """Return a wp.array[int32] from a list of per-robot DOF counts."""
    return wp.array(np.array(dofs_list, dtype=np.int32), device=device)


_POSITION_ONLY_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def _axis_weight_arr(rows, device):
    """Return a wp.array[wp.spatial_vector] axis_weight, one row (a length-6 sequence) per robot."""
    return wp.array([wp.spatial_vector(*row) for row in rows], dtype=wp.spatial_vector, device=device)


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


class TestControllerDifferentialIKModelFree(unittest.TestCase):
    def test_zero_error_gives_zero_velocity(self):
        """When current tool pose equals the target pose exactly, qd_target must be zero."""
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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

    def test_transpose_method_matches_formula(self):
        """ik_method=TRANSPOSE: qd_target = bandwidth · Jᵀe exactly, no matrix inversion."""
        device = wp.get_device()
        err_y = 0.1
        bandwidth_val = 2.0
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([1], device),
            bandwidth=bandwidth_val,
            damping=None,
            ik_method=IkMethod.TRANSPOSE,
            device=device,
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
        expected = bandwidth_val * err_y  # Jᵀe = 1.0 * err_y + 1.0 * 0 (no z-rotation error)
        self.assertAlmostEqual(float(outputs.joint_qd_target.numpy()[0]), expected, places=5)

    def test_transpose_method_rejects_explicit_damping(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                ik_method=IkMethod.TRANSPOSE,
                device=device,
            )

    def test_pseudo_inverse_method_matches_zero_damping_dls(self):
        """ik_method=PSEUDO_INVERSE matches ik_method=DAMPED_LEAST_SQUARES with damping=0 exactly."""
        device = wp.get_device()
        pos_err = np.array([0.1, 0.05, -0.03], dtype=np.float32)
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            bandwidth=1.0,
            damping=None,
            ik_method=IkMethod.PSEUDO_INVERSE,
            device=device,
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

    def test_pseudo_inverse_requires_six_dofs(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([3], device),
                bandwidth=1.0,
                damping=None,
                ik_method=IkMethod.PSEUDO_INVERSE,
                device=device,
            )

    def test_invalid_ik_method_raises(self):
        """A bogus ik_method (e.g. the string value instead of the enum member) must raise.

        Regression test: an unrecognized ik_method used to fall through
        every method-specific branch and silently produce a zero-damping
        DLS solve -- exactly IkMethod.PSEUDO_INVERSE's own solve, but
        without PSEUDO_INVERSE's dof_count >= task_dim safety check.
        """
        device = wp.get_device()
        with self.assertRaises(TypeError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([3], device),
                bandwidth=1.0,
                damping=None,
                ik_method="pseudo_inverse",
                device=device,
            )

    def test_axis_weight_zeroed_orientation_ignores_orientation_error(self):
        """DLS, J = I_6x6, orientation axes weighted 0: a large orientation error contributes nothing to qd_target."""
        device = wp.get_device()
        pos_err = np.array([0.1, 0.05, -0.03], dtype=np.float32)
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            axis_weight=_POSITION_ONLY_AXIS_WEIGHT,
            bandwidth=1.0,
            damping=0.0,
            device=device,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        target_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)  # large orientation offset
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(*pos_err.tolist()), q=target_quat)], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        expected = np.concatenate([pos_err, np.zeros(3, dtype=np.float32)])
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), expected, atol=1e-5)

    def test_axis_weight_zeroed_orientation_transpose_ignores_orientation_error(self):
        """TRANSPOSE, J = I_6x6, orientation axes weighted 0: same guarantee, with no matrix inversion involved."""
        device = wp.get_device()
        pos_err = np.array([0.1, 0.05, -0.03], dtype=np.float32)
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            axis_weight=_POSITION_ONLY_AXIS_WEIGHT,
            bandwidth=1.0,
            damping=None,
            ik_method=IkMethod.TRANSPOSE,
            device=device,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        target_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(*pos_err.tolist()), q=target_quat)], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        expected = np.concatenate([pos_err, np.zeros(3, dtype=np.float32)])
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), expected, atol=1e-5)

    def test_axis_weight_zeroed_orientation_allows_pseudo_inverse_with_three_dof_robot(self):
        """A 3-DOF robot fails IkMethod.PSEUDO_INVERSE at full 6D pose, but is allowed once orientation is dropped."""
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([3], device),
            axis_weight=_POSITION_ONLY_AXIS_WEIGHT,
            bandwidth=1.0,
            damping=None,
            ik_method=IkMethod.PSEUDO_INVERSE,
            device=device,
        )
        self.assertEqual(ctrl.controlled_robot_count, 1)

    def test_axis_weight_per_robot_shape_mismatch_raises(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6, 6], device),
                axis_weight=_axis_weight_arr([_POSITION_ONLY_AXIS_WEIGHT], device),
                bandwidth=1.0,
                damping=0.1,
                device=device,
            )

    def test_axis_weight_per_robot_list(self):
        """A heterogeneous fleet may mix a position-only robot and a full-pose robot."""
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6, 6], device),
            axis_weight=_axis_weight_arr([_POSITION_ONLY_AXIS_WEIGHT, [1.0] * 6], device),
            bandwidth=1.0,
            damping=0.0,
            device=device,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(12, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(2, device)
        target_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)
        inputs.desired_tool_pose_world = wp.array(
            [
                wp.transform(p=wp.vec3(0.1, 0.0, 0.0), q=target_quat),
                wp.transform(p=wp.vec3(0.1, 0.0, 0.0), q=target_quat),
            ],
            dtype=wp.transform,
            device=device,
        )
        inputs.jacobian_tool_world = _identity_jacobian(2, 6, device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        qd = outputs.joint_qd_target.numpy()
        # Robot 0 (POSITION): orientation rows ignored.
        np.testing.assert_allclose(qd[3:6], [0.0, 0.0, 0.0], atol=1e-5)
        # Robot 1 (POSE): orientation rows carry the real rotation-error response.
        self.assertGreater(float(np.abs(qd[9:]).max()), 1e-3)

    def test_axis_weight_zeroed_orientation_adaptive_damping_not_stuck_at_max(self):
        """A well-conditioned 3-DOF robot with orientation weighted 0 settles near λ_min, not λ_max.

        Regression guard for the padding-eigenvalue bug: without skipping the
        3 guaranteed-zero padding eigenvalues, adaptive damping would always
        see sigma_min = 0 and permanently use λ_max, regardless of how
        well-conditioned the real 3x3 task actually is.
        """
        device = wp.get_device()
        pos_err = np.array([0.1, 0.05, -0.03], dtype=np.float32)
        lam_min = 0.02
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([3], device),
            axis_weight=_POSITION_ONLY_AXIS_WEIGHT,
            bandwidth=1.0,
            damping=None,
            ik_method=IkMethod.ADAPTIVE_DAMPING,
            adaptive_damping_min=lam_min,
            adaptive_damping_max=0.5,
            adaptive_damping_threshold=0.5,
            device=device,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(*pos_err.tolist()), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        jacobian_np = np.zeros((1, 6, 3), dtype=np.float32)
        jacobian_np[0, :3, :3] = np.eye(3, dtype=np.float32)  # sigma_min = 1, well above the threshold
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        # At sigma_min = 1 >> threshold = 0.5, damping settles at exactly lam_min:
        # qd = pos_err / (1 + lam_min^2), not the near-zero response max damping would give.
        expected = pos_err / (1.0 + lam_min**2)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), expected, atol=1e-4)

    def test_axis_weight_soft_weight_matches_hand_derived_formula(self):
        """A nonzero, non-unit weight is a genuine soft trust, not just an on/off gate.

        J = I_6x6, damping > 0 so the weight doesn't algebraically cancel
        (see test_axis_weight_soft_weight_cancels_at_zero_damping for the
        case where it does): qd[i] = w_i * (w_i * e_i) / (w_i^2 + λ^2).
        """
        device = wp.get_device()
        pos_err = np.array([0.1, 0.2, -0.05], dtype=np.float32)
        lam = 1.0
        weights = [1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            axis_weight=wp.spatial_vector(*weights),
            bandwidth=1.0,
            damping=lam,
            device=device,
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
        w = np.array(weights[:3])
        expected = w * (w * pos_err) / (w**2 + lam**2)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy()[:3], expected, atol=1e-4)

    def test_axis_weight_soft_weight_cancels_at_zero_damping(self):
        """At λ=0 with J=I, a soft weight's scale cancels out exactly -- qd matches the raw error regardless.

        This is a correctness property, not a limitation: weighted DLS at
        λ=0 is still the same minimum-norm exact solve, and rescaling one
        row of an already-exactly-solvable system doesn't change its
        solution.
        """
        device = wp.get_device()
        pos_err = np.array([0.1, 0.2, -0.05], dtype=np.float32)
        weights = [1.0, 0.5, 2.0, 1.0, 1.0, 1.0]
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            axis_weight=wp.spatial_vector(*weights),
            bandwidth=1.0,
            damping=0.0,
            device=device,
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
        np.testing.assert_allclose(outputs.joint_qd_target.numpy()[:3], pos_err, atol=1e-4)

    def test_zero_weighted_axis_jacobian_row_has_exactly_zero_gradient(self):
        """A zero-weighted axis's Jacobian row must contribute exactly zero gradient, not just a tiny one.

        Regression test for the design decision documented in
        ``axis_weight``'s own docstring (gather/scatter by index, not
        multiply-by-zero): a value that is never read contributes an
        exactly-zero gradient, where "coefficient times an exactly-zero
        input" would not.
        """
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            axis_weight=_POSITION_ONLY_AXIS_WEIGHT,  # orientation axes (rows 3-5) weighted 0
            bandwidth=1.0,
            damping=None,
            ik_method=IkMethod.TRANSPOSE,  # the only IkMethod with a correct backward pass
            device=device,
            requires_grad=True,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q.assign(np.zeros(6, dtype=np.float32))
        inputs.tool_pose_world.assign([wp.transform_identity()])
        inputs.desired_tool_pose_world.assign([wp.transform(p=wp.vec3(0.1, 0.05, -0.03), q=wp.quat_identity())])
        inputs.jacobian_tool_world.assign(np.eye(6, dtype=np.float32)[None, :, :])

        tape = wp.Tape()
        with tape:
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        tape.backward(grads={outputs.joint_qd_target: wp.ones(6, dtype=wp.float32, device=device)})

        jacobian_grad = inputs.jacobian_tool_world.grad.numpy()
        np.testing.assert_array_equal(jacobian_grad[0, 3:, :], np.zeros((3, 6), dtype=np.float32))
        self.assertGreater(np.abs(jacobian_grad[0, :3, :]).max(), 1e-6)

    def test_requires_grad_rejects_every_ik_method_except_transpose(self):
        """Only IkMethod.TRANSPOSE may be combined with requires_grad=True.

        Every other method routes through ``_invert_spd_block_kernel`` or
        ``_truncated_pinv_matrix_kernel``, both marked
        ``enable_backward=False`` because their backward pass does not match
        a finite difference (see the kernels' docstrings in ``_common.py``
        and ``differential_ik/_common.py``). Constructing a requires_grad
        controller with one of them would silently produce an incomplete
        gradient, so the constructor rejects it instead.
        """
        device = wp.get_device()
        method_kwargs = {
            IkMethod.DAMPED_LEAST_SQUARES: {"damping": 0.1},
            IkMethod.PSEUDO_INVERSE: {"damping": None},
            IkMethod.ADAPTIVE_DAMPING: {
                "damping": None,
                "adaptive_damping_min": 0.02,
                "adaptive_damping_max": 0.5,
                "adaptive_damping_threshold": 0.3,
            },
            IkMethod.TRUNCATED_SVD: {"damping": None, "truncated_svd_threshold": 0.05},
        }
        for ik_method, extra_kwargs in method_kwargs.items():
            with self.subTest(ik_method=ik_method), self.assertRaises(ValueError):
                ControllerDifferentialIKModelFree(
                    controlled_dofs_per_robot=_dofs_arr([6], device),
                    bandwidth=1.0,
                    ik_method=ik_method,
                    device=device,
                    requires_grad=True,
                    **extra_kwargs,
                )

    def test_gradient_matches_finite_difference_for_transpose(self):
        """TRANSPOSE's analytic gradient matches a central finite difference.

        The only IkMethod with a correct backward pass, since it never
        inverts a matrix or takes an eigendecomposition -- q̇ = bandwidth·Jᵀe
        is differentiated by the generic mechanism without issue.
        """
        device = wp.get_device()
        rng = np.random.default_rng(31)
        jacobian_np = rng.normal(size=(1, 6, 6)).astype(np.float32)
        pos_err = np.array([0.1, 0.05, -0.03], dtype=np.float32)

        def qd_target_sum(ctrl, jacobian_value):
            inputs = ctrl.input()
            outputs = ctrl.output()
            inputs.joint_q.assign(np.zeros(6, dtype=np.float32))
            inputs.tool_pose_world.assign([wp.transform_identity()])
            inputs.desired_tool_pose_world.assign([wp.transform(p=wp.vec3(*pos_err.tolist()), q=wp.quat_identity())])
            inputs.jacobian_tool_world.assign(jacobian_value)
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
            return float(outputs.joint_qd_target.numpy().sum())

        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            bandwidth=1.0,
            damping=None,
            ik_method=IkMethod.TRANSPOSE,
            device=device,
            requires_grad=True,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q.assign(np.zeros(6, dtype=np.float32))
        inputs.tool_pose_world.assign([wp.transform_identity()])
        inputs.desired_tool_pose_world.assign([wp.transform(p=wp.vec3(*pos_err.tolist()), q=wp.quat_identity())])
        inputs.jacobian_tool_world.assign(jacobian_np)

        tape = wp.Tape()
        with tape:
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        tape.backward(grads={outputs.joint_qd_target: wp.ones(6, dtype=wp.float32, device=device)})
        analytic_grad = inputs.jacobian_tool_world.grad.numpy().copy()

        # Central finite difference on a few representative entries, not all
        # 36 -- enough to catch a broken backward pass without an expensive,
        # over-precise sweep.
        eps = 1.0e-3
        for row, col in [(0, 0), (2, 3), (5, 5)]:
            plus = jacobian_np.copy()
            plus[0, row, col] += eps
            minus = jacobian_np.copy()
            minus[0, row, col] -= eps
            numeric_grad = (qd_target_sum(ctrl, plus) - qd_target_sum(ctrl, minus)) / (2.0 * eps)
            np.testing.assert_allclose(analytic_grad[0, row, col], numeric_grad, atol=1.0e-2, rtol=1.0e-2)

    def test_pseudo_inverse_rejects_explicit_damping(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                ik_method=IkMethod.PSEUDO_INVERSE,
                device=device,
            )

    def test_adaptive_damping_uses_lambda_min_far_from_singularity(self):
        """J = I_6x6 (sigma_min = 1) with threshold below 1: adaptive damping settles at λ_min exactly."""
        device = wp.get_device()
        pos_err = np.array([0.1, 0.05, -0.03], dtype=np.float32)
        lam_min = 0.3
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            bandwidth=1.0,
            damping=None,
            ik_method=IkMethod.ADAPTIVE_DAMPING,
            adaptive_damping_min=lam_min,
            adaptive_damping_max=1.0,
            adaptive_damping_threshold=0.5,
            device=device,
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
        expected = np.concatenate([pos_err, np.zeros(3, dtype=np.float32)]) / (1.0 + lam_min**2)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), expected, atol=1e-5)

    def test_adaptive_damping_uses_lambda_max_for_structurally_rank_deficient_robot(self):
        """A 1-DOF robot's JJᵀ always has sigma_min = 0 (rank <= 1 < 6), so adaptive damping settles at λ_max."""
        device = wp.get_device()
        err_y = 0.1
        bandwidth_val = 2.0
        lam_max = 0.5
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([1], device),
            bandwidth=bandwidth_val,
            damping=None,
            ik_method=IkMethod.ADAPTIVE_DAMPING,
            adaptive_damping_min=0.0,
            adaptive_damping_max=lam_max,
            adaptive_damping_threshold=1.0,
            device=device,
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
        expected = bandwidth_val * err_y / (2.0 + lam_max**2)
        self.assertAlmostEqual(float(outputs.joint_qd_target.numpy()[0]), expected, places=5)

    def test_adaptive_damping_requires_all_three_params(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=None,
                ik_method=IkMethod.ADAPTIVE_DAMPING,
                adaptive_damping_min=0.1,
                adaptive_damping_max=1.0,
                # adaptive_damping_threshold omitted
                device=device,
            )

    def test_adaptive_damping_rejects_explicit_damping(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                ik_method=IkMethod.ADAPTIVE_DAMPING,
                adaptive_damping_min=0.1,
                adaptive_damping_max=1.0,
                adaptive_damping_threshold=0.5,
                device=device,
            )

    def test_adaptive_damping_params_rejected_for_other_methods(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                adaptive_damping_min=0.1,
                adaptive_damping_max=1.0,
                adaptive_damping_threshold=0.5,
                device=device,
            )

    def test_truncated_svd_matches_pinv_when_well_conditioned(self):
        """A generic, well-conditioned 6x6 J: every direction clears the threshold, so qd = J^+ @ e exactly."""
        device = wp.get_device()
        rng = np.random.default_rng(29)
        jacobian_np = rng.normal(size=(1, 6, 6)).astype(np.float32)
        pos_err = np.array([0.1, 0.05, -0.03], dtype=np.float32)
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            bandwidth=1.0,
            damping=None,
            ik_method=IkMethod.TRUNCATED_SVD,
            truncated_svd_threshold=1.0e-2,  # well below every singular value of a random 6x6 J
            device=device,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(*pos_err.tolist()), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        j64 = jacobian_np[0].astype(np.float64)
        error_np = np.concatenate([pos_err, np.zeros(3, dtype=np.float32)]).astype(np.float64)
        expected = np.linalg.pinv(j64) @ error_np
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), expected, atol=1e-3)

    def test_truncated_svd_matches_spectral_filter_for_rank_deficient_robot(self):
        """A 5-DOF robot's JJᵀ has one exact-zero eigenvalue: dropped, unlike PSEUDO_INVERSE which forbids dof<6."""
        device = wp.get_device()
        rng = np.random.default_rng(23)
        n = 5
        # Above the QR eigensolver's own float32 noise floor for the one
        # structurally zero eigenvalue (~1e-6, i.e. sigma ~1e-3), but well
        # below every genuine singular value of a random 6x5 Jacobian.
        threshold = 1.0e-2
        jacobian_np = rng.normal(size=(1, 6, n)).astype(np.float32)
        pos_err = rng.normal(size=3).astype(np.float32)

        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([n], device),
            bandwidth=1.0,
            damping=None,
            ik_method=IkMethod.TRUNCATED_SVD,
            truncated_svd_threshold=threshold,
            device=device,
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(n, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(*pos_err.tolist()), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        j64 = jacobian_np[0].astype(np.float64)
        error_np = np.concatenate([pos_err, np.zeros(3, dtype=np.float32)]).astype(np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(j64 @ j64.T)
        sigma = np.sqrt(np.maximum(eigenvalues, 0.0))
        # g(sigma) = 1/sigma^2 = 1/eigenvalue: JJᵀ itself is being inverted.
        g = np.where(sigma > threshold, 1.0 / np.maximum(eigenvalues, 1e-30), 0.0)
        w = (eigenvectors * g) @ eigenvectors.T @ error_np
        expected = j64.T @ w
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), expected, atol=1e-3)
        # Independent check: J is rank 5 (padded to 6x5), so its own
        # np.linalg.pinv already drops exactly the null direction, matching
        # the kernel's truncation without re-deriving its formula.
        expected_via_pinv = np.linalg.pinv(j64) @ error_np
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), expected_via_pinv, atol=1e-3)

    def test_truncated_svd_requires_threshold(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=None,
                ik_method=IkMethod.TRUNCATED_SVD,
                device=device,
            )

    def test_truncated_svd_rejects_explicit_damping(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                ik_method=IkMethod.TRUNCATED_SVD,
                truncated_svd_threshold=0.01,
                device=device,
            )

    def test_truncated_svd_threshold_rejected_for_other_methods(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                truncated_svd_threshold=0.01,
                device=device,
            )

    def test_multiple_robots_independent(self):
        """Each robot's qd_target depends only on its own Jacobian and pose error."""
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.5, device=device
        )
        self.assertTrue(ctrl.is_graphable())

    def test_inputs_bandwidth_and_damping_none_when_baked(self):
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.5, device=device
        )
        inputs = ctrl.input()
        self.assertIsNone(inputs.bandwidth)
        self.assertIsNone(inputs.damping)

    def test_inputs_bandwidth_and_damping_allocated_when_live(self):
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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

    def test_oversized_output_raises(self):
        """An output bound to a larger-than-expected array raises.

        wp.copy accepts a destination larger than the source and silently
        writes only a prefix, so this specific direction (too large, not
        too small) has to be caught explicitly.
        """
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.0, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = _identity_transform(1, device)
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        outputs.joint_qd_target = wp.zeros(8, dtype=wp.float32, device=device)  # controller has 6 controlled DOFs
        with self.assertRaises(ValueError):
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

    def test_disabled_bandwidth_port_written_raises(self):
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
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
        ctrl = ControllerDifferentialIKModelFree(
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
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6, 0], device), bandwidth=1.0, damping=0.0, device=device
            )

    def test_controlled_dofs_per_robot_is_copied(self):
        """A later mutation of the caller's own array must not affect the controller."""
        device = wp.get_device()
        dofs = _dofs_arr([6], device)
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=dofs, bandwidth=1.0, damping=0.0, device=device
        )
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

    def test_null_space_control_allowed_with_fewer_than_six_dofs_when_damped(self):
        """A redundant low-DOF arm (e.g. a planar 4R arm) may enable null-space control as long
        as null_space_damping > 0 regularizes the otherwise rank-deficient JJᵀ."""
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([4], device),
            bandwidth=1.0,
            damping=0.1,
            use_joint_limit_avoidance=True,
            joint_limit_avoidance_gain=1.0,
            joint_limit_avoidance_margin=0.1,
            joint_pos_lower=wp.full(4, -1.0, dtype=wp.float32, device=device),
            joint_pos_upper=wp.full(4, 1.0, dtype=wp.float32, device=device),
            null_space_damping=0.5,
            device=device,
        )
        # A planar task: only the x, y, and yaw rows of the Jacobian are
        # nonzero, so JJᵀ is structurally rank-deficient (rank <= 3) without
        # damping — undamped, this would produce a physically meaningless
        # (or NaN) projector; the whole point of null_space_damping.
        rng = np.random.default_rng(15)
        jacobian_np = np.zeros((1, 6, 4), dtype=np.float32)
        jacobian_np[0, [0, 1, 5], :] = rng.normal(size=(3, 4))
        pose = _identity_transform(1, device)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.array([0.99, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = pose
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        qd = outputs.joint_qd_target.numpy()
        self.assertTrue(np.all(np.isfinite(qd)))
        self.assertLess(float(qd[0]), 0.0)  # still pulls DOF 0 away from its limit

    def test_null_space_damping_rejected_without_null_space_enabled(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                null_space_damping=0.1,
                device=device,
            )

    def test_baked_zero_null_space_damping_rejected_for_underactuated_robot(self):
        """A baked null_space_damping <= 0 for a robot with fewer than 6 DOFs raises at construction."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([4], device),
                bandwidth=1.0,
                damping=0.1,
                use_joint_limit_avoidance=True,
                joint_limit_avoidance_gain=1.0,
                joint_limit_avoidance_margin=0.1,
                joint_pos_lower=wp.full(4, -1.0, dtype=wp.float32, device=device),
                joint_pos_upper=wp.full(4, 1.0, dtype=wp.float32, device=device),
                null_space_damping=0.0,
                device=device,
            )

    def test_baked_zero_null_space_damping_allowed_for_six_dof_robot(self):
        """A baked null_space_damping of exactly 0 is fine when every robot has >= 6 controlled DOFs."""
        device = wp.get_device()
        ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device),
            bandwidth=1.0,
            damping=0.1,
            use_joint_limit_avoidance=True,
            joint_limit_avoidance_gain=1.0,
            joint_limit_avoidance_margin=0.1,
            joint_pos_lower=wp.full(6, -1.0, dtype=wp.float32, device=device),
            joint_pos_upper=wp.full(6, 1.0, dtype=wp.float32, device=device),
            null_space_damping=0.0,
            device=device,
        )

    def test_live_null_space_damping_port_matches_baked(self):
        """A live null_space_damping input must produce the same result as the equivalent baked value."""
        device = wp.get_device()
        rng = np.random.default_rng(16)
        jacobian_np = rng.normal(size=(1, 6, 7)).astype(np.float32)
        pose = _identity_transform(1, device)

        baked_ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([7], device),
            bandwidth=1.0,
            damping=0.1,
            use_joint_limit_avoidance=True,
            joint_limit_avoidance_gain=1.0,
            joint_limit_avoidance_margin=0.1,
            joint_pos_lower=wp.full(7, -1.0, dtype=wp.float32, device=device),
            joint_pos_upper=wp.full(7, 1.0, dtype=wp.float32, device=device),
            null_space_damping=0.3,
            device=device,
        )
        baked_inputs = baked_ctrl.input()
        baked_outputs = baked_ctrl.output()
        baked_inputs.joint_q = wp.array([0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
        baked_inputs.tool_pose_world = pose
        baked_inputs.desired_tool_pose_world = pose
        baked_inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        baked_ctrl.step(inputs=baked_inputs, outputs=baked_outputs, dt=0.01)

        live_ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([7], device),
            bandwidth=1.0,
            damping=0.1,
            use_joint_limit_avoidance=True,
            joint_limit_avoidance_gain=1.0,
            joint_limit_avoidance_margin=0.1,
            joint_pos_lower=wp.full(7, -1.0, dtype=wp.float32, device=device),
            joint_pos_upper=wp.full(7, 1.0, dtype=wp.float32, device=device),
            null_space_damping=None,
            device=device,
        )
        live_inputs = live_ctrl.input()
        live_outputs = live_ctrl.output()
        live_inputs.joint_q = wp.array([0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
        live_inputs.tool_pose_world = pose
        live_inputs.desired_tool_pose_world = pose
        live_inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        live_inputs.null_space_damping = wp.full(1, 0.3, dtype=wp.float32, device=device)
        live_ctrl.step(inputs=live_inputs, outputs=live_outputs, dt=0.01)

        np.testing.assert_allclose(
            live_outputs.joint_qd_target.numpy(), baked_outputs.joint_qd_target.numpy(), atol=1e-5
        )

    def test_joint_limit_avoidance_requires_positive_gain(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                use_joint_limit_avoidance=True,
                joint_limit_avoidance_gain=0.0,
                joint_limit_avoidance_margin=0.1,
                joint_pos_lower=wp.full(6, -1.0, dtype=wp.float32, device=device),
                joint_pos_upper=wp.full(6, 1.0, dtype=wp.float32, device=device),
                device=device,
            )

    def test_joint_limit_avoidance_requires_limits(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                use_joint_limit_avoidance=True,
                joint_limit_avoidance_gain=1.0,
                joint_limit_avoidance_margin=0.1,
                device=device,
            )

    def test_joint_pos_limits_rejected_without_avoidance_enabled(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                joint_pos_lower=wp.full(6, -1.0, dtype=wp.float32, device=device),
                joint_pos_upper=wp.full(6, 1.0, dtype=wp.float32, device=device),
                device=device,
            )

    def test_null_space_stiffness_rejected_without_posture_enabled(self):
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerDifferentialIKModelFree(
                controlled_dofs_per_robot=_dofs_arr([6], device),
                bandwidth=1.0,
                damping=0.1,
                null_space_stiffness=1.0,
                device=device,
            )

    def test_null_space_velocity_does_not_disturb_primary_task(self):
        """With zero primary-task error, the entire qd output must satisfy J @ qd == 0."""
        device = wp.get_device()
        rng = np.random.default_rng(11)
        jacobian_np = rng.normal(size=(1, 6, 7)).astype(np.float32)
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([7], device),
            bandwidth=1.0,
            damping=0.1,
            use_joint_limit_avoidance=True,
            joint_limit_avoidance_gain=2.0,
            joint_limit_avoidance_margin=0.3,
            joint_pos_lower=wp.full(7, -1.0, dtype=wp.float32, device=device),
            joint_pos_upper=wp.full(7, 1.0, dtype=wp.float32, device=device),
            use_null_space_posture_control=True,
            null_space_stiffness=1.0,
            device=device,
        )
        pose = _identity_transform(1, device)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.array([0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = pose
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        inputs.q_des_null = wp.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        qd = outputs.joint_qd_target.numpy()
        np.testing.assert_allclose(jacobian_np[0].astype(np.float64) @ qd.astype(np.float64), np.zeros(6), atol=1e-3)

    def test_null_space_velocity_does_not_disturb_primary_task_with_zeroed_axis_weight(self):
        """With zero primary-task error, qd must satisfy J_active @ qd == 0 even when axis_weight excludes 3 of 6 axes.

        Combines two mechanisms that had never been tested together: a
        redundant robot resolved via a reduced task_dim (``axis_weight``),
        and null-space posture control's own gather/scatter path, which
        exists specifically to keep the null-space projector consistent
        with a non-full ``task_dim``.
        """
        device = wp.get_device()
        rng = np.random.default_rng(21)
        # A planar-style task: only X, Y, and yaw (rows 0, 1, 5) are active;
        # Z, roll, pitch (rows 2, 3, 4) are structurally excluded.
        jacobian_np = np.zeros((1, 6, 4), dtype=np.float32)
        jacobian_np[0, [0, 1, 5], :] = rng.normal(size=(3, 4))
        axis_weight = wp.spatial_vector(1.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([4], device),
            axis_weight=axis_weight,
            bandwidth=1.0,
            damping=0.1,
            use_null_space_posture_control=True,
            null_space_stiffness=1.0,
            # 0 is exact here (not just "safe"): dof_count=4 >= task_dim=3,
            # so the active-axis JJᵀ is generically full rank without
            # regularization, unlike the under-actuated case that requires
            # null_space_damping > 0.
            null_space_damping=0.0,
            device=device,
        )
        pose = _identity_transform(1, device)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.array([0.1, -0.2, 0.3, -0.1], dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = pose  # zero primary-task error
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        # Pulls hard toward a posture far from the current one.
        inputs.q_des_null = wp.array([0.9, 0.9, 0.9, 0.9], dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        qd = outputs.joint_qd_target.numpy()

        j_active = jacobian_np[0, [0, 1, 5], :]
        np.testing.assert_allclose(j_active.astype(np.float64) @ qd.astype(np.float64), np.zeros(3), atol=1e-3)
        # The null-space pull actually did something (the redundant DOFs were used).
        self.assertGreater(np.abs(qd).max(), 1e-3)

    def test_joint_limit_avoidance_pulls_away_from_limit(self):
        device = wp.get_device()
        rng = np.random.default_rng(12)
        jacobian_np = rng.normal(size=(1, 6, 7)).astype(np.float32)
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([7], device),
            bandwidth=1.0,
            damping=0.1,
            use_joint_limit_avoidance=True,
            joint_limit_avoidance_gain=2.0,
            joint_limit_avoidance_margin=0.3,
            joint_pos_lower=wp.full(7, -1.0, dtype=wp.float32, device=device),
            joint_pos_upper=wp.full(7, 1.0, dtype=wp.float32, device=device),
            device=device,
        )
        pose = _identity_transform(1, device)
        inputs = ctrl.input()
        outputs = ctrl.output()
        # DOF 0 is nearly at its upper limit; every other DOF is centered.
        inputs.joint_q = wp.array([0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = pose
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        self.assertLess(float(outputs.joint_qd_target.numpy()[0]), 0.0)

    def test_null_space_posture_pulls_toward_target(self):
        device = wp.get_device()
        rng = np.random.default_rng(13)
        jacobian_np = rng.normal(size=(1, 6, 7)).astype(np.float32)
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([7], device),
            bandwidth=1.0,
            damping=0.1,
            use_null_space_posture_control=True,
            null_space_stiffness=1.0,
            device=device,
        )
        pose = _identity_transform(1, device)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(7, dtype=wp.float32, device=device)
        inputs.tool_pose_world = pose
        inputs.desired_tool_pose_world = pose
        inputs.jacobian_tool_world = wp.array3d(jacobian_np, dtype=wp.float32, device=device)
        inputs.q_des_null = wp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        self.assertGreater(float(outputs.joint_qd_target.numpy()[0]), 0.0)

    def test_disabled_q_des_null_written_raises(self):
        device = wp.get_device()
        ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([6], device), bandwidth=1.0, damping=0.0, device=device
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(6, dtype=wp.float32, device=device)
        inputs.tool_pose_world = _identity_transform(1, device)
        inputs.desired_tool_pose_world = _identity_transform(1, device)
        inputs.jacobian_tool_world = _identity_jacobian(1, 6, device)
        inputs.q_des_null = wp.zeros(6, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

    def test_live_null_space_stiffness_port_matches_baked(self):
        """A live null_space_stiffness input must produce the same result as the equivalent baked value."""
        device = wp.get_device()
        rng = np.random.default_rng(14)
        jacobian_np = rng.normal(size=(1, 6, 7)).astype(np.float32)
        pose = _identity_transform(1, device)
        q_des_null = wp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
        jacobian = wp.array3d(jacobian_np, dtype=wp.float32, device=device)

        baked_ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([7], device),
            bandwidth=1.0,
            damping=0.1,
            use_null_space_posture_control=True,
            null_space_stiffness=5.0,
            device=device,
        )
        baked_inputs = baked_ctrl.input()
        baked_outputs = baked_ctrl.output()
        baked_inputs.joint_q = wp.zeros(7, dtype=wp.float32, device=device)
        baked_inputs.tool_pose_world = pose
        baked_inputs.desired_tool_pose_world = pose
        baked_inputs.jacobian_tool_world = jacobian
        baked_inputs.q_des_null = q_des_null
        baked_ctrl.step(inputs=baked_inputs, outputs=baked_outputs, dt=0.01)

        live_ctrl = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=_dofs_arr([7], device),
            bandwidth=1.0,
            damping=0.1,
            use_null_space_posture_control=True,
            null_space_stiffness=None,
            device=device,
        )
        live_inputs = live_ctrl.input()
        live_outputs = live_ctrl.output()
        live_inputs.joint_q = wp.zeros(7, dtype=wp.float32, device=device)
        live_inputs.tool_pose_world = pose
        live_inputs.desired_tool_pose_world = pose
        live_inputs.jacobian_tool_world = jacobian
        live_inputs.q_des_null = q_des_null
        live_inputs.null_space_stiffness = wp.full(7, 5.0, dtype=wp.float32, device=device)
        live_ctrl.step(inputs=live_inputs, outputs=live_outputs, dt=0.01)

        np.testing.assert_allclose(
            live_outputs.joint_qd_target.numpy(), baked_outputs.joint_qd_target.numpy(), atol=1e-5
        )


# ---------------------------------------------------------------------------
# ControllerDifferentialIK
# ---------------------------------------------------------------------------


def _build_two_link_arm_with_tool_site(device):
    builder = newton.ModelBuilder()
    link0 = builder.add_link()
    link1 = builder.add_link()
    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link0,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    j1 = builder.add_joint_revolute(
        parent=link0,
        child=link1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j0, j1], label="arm")
    builder.add_site(link1, label="tip", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))
    return builder.finalize(device=device)


def _build_two_robot_arms_with_tool_sites(device):
    """Robot 0: 1-DOF arm ("tool0"). Robot 1: 2-DOF arm ("tool1")."""
    builder = newton.ModelBuilder()
    l0 = builder.add_link()
    j0 = builder.add_joint_revolute(
        parent=-1,
        child=l0,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j0], label="robot0")
    builder.add_site(l0, label="tool0", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

    l1a = builder.add_link()
    l1b = builder.add_link()
    j1a = builder.add_joint_revolute(
        parent=-1,
        child=l1a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(3.0, 0.0, 0.0)),
        child_xform=wp.transform_identity(),
    )
    j1b = builder.add_joint_revolute(
        parent=l1a,
        child=l1b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j1a, j1b], label="robot1")
    builder.add_site(l1b, label="tool1", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))
    return builder.finalize(device=device)


def _build_seven_dof_chain_with_tool_site(device):
    """A redundant 7-revolute-joint chain (task is 6D, so 1 DOF of null-space freedom)."""
    builder = newton.ModelBuilder()
    parent = -1
    parent_xform = wp.transform_identity()
    links = []
    axes = [
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
    ]
    joints = []
    for axis in axes:
        link = builder.add_link()
        j = builder.add_joint_revolute(
            parent=parent,
            child=link,
            axis=axis,
            parent_xform=parent_xform,
            child_xform=wp.transform_identity(),
        )
        joints.append(j)
        links.append(link)
        parent = link
        parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.2))
    builder.add_articulation(joints, label="arm")
    builder.add_site(links[-1], label="tip", xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.2), q=wp.quat_identity()))
    return builder.finalize(device=device)


class TestControllerDifferentialIK(unittest.TestCase):
    def test_zero_error_gives_zero_velocity(self):
        device = wp.get_device()
        model = _build_two_link_arm_with_tool_site(device)
        ctrl = ControllerDifferentialIK(model, tool_sites="tip", bandwidth=1.0, damping=0.1)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=device)
        inputs.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        # Home pose: two unit links along +x -> tip at (2, 0, 0).
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(2.0, 0.0, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), np.zeros(2), atol=1e-5)

    def test_ik_method_forwarded_to_inner_controller(self):
        """ik_method=TRANSPOSE is forwarded to the inner ControllerDifferentialIKModelFree, not silently dropped."""
        device = wp.get_device()
        model = _build_two_link_arm_with_tool_site(device)
        ctrl = ControllerDifferentialIK(
            model, tool_sites="tip", bandwidth=1.0, damping=None, ik_method=IkMethod.TRANSPOSE
        )
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=device)
        inputs.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(2.0, 0.0, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), np.zeros(2), atol=1e-5)

    def test_converges_to_target_for_every_ik_method(self):
        """Every IkMethod drives a real, redundant arm to a reachable target through the model-based wrapper.

        DAMPED_LEAST_SQUARES and TRANSPOSE already have dedicated
        convergence/formula tests elsewhere; PSEUDO_INVERSE, ADAPTIVE_DAMPING,
        and TRUNCATED_SVD had never been exercised end-to-end through
        ControllerDifferentialIK before this test.
        """
        device = wp.get_device()
        # TRANSPOSE gets its own higher bandwidth and looser tolerance:
        # unlike the JJᵀ-inverting methods, it has no built-in scaling
        # against the task's own conditioning, and — a known property of
        # the method, not a bug — settles to a small nonzero steady-state
        # residual rather than converging arbitrarily close to zero error.
        method_kwargs = {
            IkMethod.DAMPED_LEAST_SQUARES: ({"bandwidth": 1.0, "damping": 0.05}, 0.1),
            IkMethod.PSEUDO_INVERSE: ({"bandwidth": 1.0, "damping": None}, 0.1),
            IkMethod.TRANSPOSE: ({"bandwidth": 6.0, "damping": None}, 0.15),
            IkMethod.ADAPTIVE_DAMPING: (
                {
                    "bandwidth": 1.0,
                    "damping": None,
                    "adaptive_damping_min": 0.01,
                    "adaptive_damping_max": 0.5,
                    "adaptive_damping_threshold": 0.1,
                },
                0.1,
            ),
            IkMethod.TRUNCATED_SVD: ({"bandwidth": 1.0, "damping": None, "truncated_svd_threshold": 0.01}, 0.1),
        }
        # A well-within-reach target (chain reach is 7 * 0.2 = 1.4 m) for a
        # 7-DOF chain, so PSEUDO_INVERSE's dof_count >= task_dim requirement
        # is satisfied too.
        target_pos = np.array([0.3, 0.4, 0.9], dtype=np.float32)
        for ik_method, (extra_kwargs, atol) in method_kwargs.items():
            with self.subTest(ik_method=ik_method):
                model = _build_seven_dof_chain_with_tool_site(device)
                ctrl = ControllerDifferentialIK(model, tool_sites="tip", ik_method=ik_method, **extra_kwargs)
                inputs = ctrl.input()
                outputs = ctrl.output()
                target = wp.array(
                    [wp.transform(p=wp.vec3(*target_pos.tolist()), q=wp.quat_identity())],
                    dtype=wp.transform,
                    device=device,
                )
                q = np.full(7, 0.2, dtype=np.float32)
                dt = 0.05
                for _ in range(300):
                    inputs.joint_q = wp.array(q, dtype=wp.float32, device=device)
                    inputs.joint_qd = wp.zeros(7, dtype=wp.float32, device=device)
                    inputs.desired_tool_pose_world = target
                    ctrl.step(inputs=inputs, outputs=outputs, dt=dt)
                    q = outputs.joint_q_target.numpy().copy()

                state = model.state()
                newton.eval_fk(
                    model,
                    wp.array(q, dtype=wp.float32, device=device),
                    wp.zeros(7, dtype=wp.float32, device=device),
                    state,
                )
                tip_pos = wp.transform_point(wp.transform(*state.body_q.numpy()[6]), wp.vec3(0.0, 0.0, 0.2))
                np.testing.assert_allclose(np.array(tip_pos), target_pos, atol=atol)

    def test_step_resolves_tool_pose_matching_forward_kinematics(self):
        device = wp.get_device()
        model = _build_two_link_arm_with_tool_site(device)
        ctrl = ControllerDifferentialIK(model, tool_sites="tip", bandwidth=1.0, damping=0.1)
        inputs = ctrl.input()
        outputs = ctrl.output()
        joint_q = np.array([0.3, -0.4], dtype=np.float32)
        inputs.joint_q = wp.array(joint_q, dtype=wp.float32, device=device)
        inputs.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        # Feed the current tool pose back as the target: zero pose error means
        # the controller's own FK-resolved tool pose must equal the
        # independently-computed one, else qd would be nonzero.
        state = model.state()
        newton.eval_fk(
            model,
            wp.array(joint_q, dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            state,
        )
        tip_pose = state.body_q.numpy()[1]
        tip_world = wp.transform(*tip_pose) * wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity())
        inputs.desired_tool_pose_world = wp.array([tip_world], dtype=wp.transform, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        np.testing.assert_allclose(outputs.joint_qd_target.numpy(), np.zeros(2), atol=1e-4)
        # The public tool_pose_world property exposes the same pose the
        # controller resolved and used internally, not just a side effect
        # inferred from zero qd_target above.
        np.testing.assert_allclose(ctrl.tool_pose_world.numpy()[0], np.array(tip_world), atol=1e-5)

    def test_two_link_arm_converges_to_target(self):
        device = wp.get_device()
        model = _build_two_link_arm_with_tool_site(device)
        ctrl = ControllerDifferentialIK(model, tool_sites="tip", bandwidth=1.0, damping=0.05)
        inputs = ctrl.input()
        outputs = ctrl.output()
        target = wp.array(
            [wp.transform(p=wp.vec3(1.2, 0.8, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        q = np.zeros(2, dtype=np.float32)
        dt = 0.05
        for _ in range(200):
            inputs.joint_q = wp.array(q, dtype=wp.float32, device=device)
            inputs.joint_qd = wp.zeros(2, dtype=wp.float32, device=device)
            inputs.desired_tool_pose_world = target
            ctrl.step(inputs=inputs, outputs=outputs, dt=dt)
            q = outputs.joint_q_target.numpy().copy()

        state = model.state()
        newton.eval_fk(
            model, wp.array(q, dtype=wp.float32, device=device), wp.zeros(2, dtype=wp.float32, device=device), state
        )
        tip_pos = wp.transform_point(wp.transform(*state.body_q.numpy()[1]), wp.vec3(1.0, 0.0, 0.0))
        # DLS damping (λ=0.05) leaves a small steady-state tracking bias by
        # design, not just a numerical-convergence tolerance.
        np.testing.assert_allclose(np.array(tip_pos), [1.2, 0.8, 0.0], atol=0.1)

    def test_heterogeneous_fleet_selection(self):
        device = wp.get_device()
        model = _build_two_robot_arms_with_tool_sites(device)
        ctrl = ControllerDifferentialIK(model, tool_sites=["tool0", "tool1"], bandwidth=1.0, damping=0.1)
        self.assertEqual(ctrl.controlled_robot_count, 2)
        self.assertEqual(ctrl.total_controlled_dofs, 3)
        self.assertEqual(ctrl.max_controlled_dofs, 2)

    def test_heterogeneous_fleet_step_has_no_cross_talk_between_robots(self):
        """Moving one robot's target must not perturb another independently-selected robot's output.

        Robot 0's target is set to its own exact home pose (zero error);
        robot 1's target is displaced, forcing it to move. If Jacobian or
        error data ever crossed between robots' padded per-robot buffers,
        robot 0's supposedly-zero output would pick up robot 1's motion.
        """
        device = wp.get_device()
        model = _build_two_robot_arms_with_tool_sites(device)
        ctrl = ControllerDifferentialIK(model, tool_sites=["tool0", "tool1"], bandwidth=1.0, damping=0.1)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        inputs.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        # Robot 0 (1-DOF arm at origin): tool0 is at (1, 0, 0) at q=0 --
        # target set to that exact home pose, zero error. Robot 1 (2-DOF arm
        # based at (3, 0, 0)): tool1 is at (5, 0, 0) at q=0 -- target
        # displaced, forcing it to move.
        inputs.desired_tool_pose_world = wp.array(
            [
                wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()),
                wp.transform(p=wp.vec3(5.3, 0.2, 0.0), q=wp.quat_identity()),
            ],
            dtype=wp.transform,
            device=device,
        )
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        qd = outputs.joint_qd_target.numpy()
        # Robot 0's single DOF comes first, matching the grouped-by-robot
        # compact layout (robot 0's DOFs first, then robot 1's).
        np.testing.assert_allclose(qd[0], 0.0, atol=1e-5)
        self.assertGreater(np.abs(qd[1:]).max(), 1e-3)

    def test_subset_of_articulations(self):
        device = wp.get_device()
        model = _build_two_robot_arms_with_tool_sites(device)
        ctrl = ControllerDifferentialIK(model, articulations="robot0", tool_sites="tool0", bandwidth=1.0, damping=0.1)
        self.assertEqual(ctrl.controlled_robot_count, 1)
        self.assertEqual(ctrl.total_controlled_dofs, 1)

    def test_tool_pattern_matching_multiple_sites_on_one_robot_raises(self):
        device = wp.get_device()
        builder = newton.ModelBuilder()
        link0 = builder.add_link()
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j0], label="arm")
        builder.add_site(link0, label="tool_a", xform=wp.transform_identity())
        builder.add_site(link0, label="tool_b", xform=wp.transform_identity())
        model = builder.finalize(device=device)
        with self.assertRaises(ValueError):
            ControllerDifferentialIK(model, tool_sites=["tool_a", "tool_b"], bandwidth=1.0, damping=0.1)

    def test_tool_site_missing_raises(self):
        device = wp.get_device()
        model = _build_two_link_arm_with_tool_site(device)
        with self.assertRaises(ValueError):
            ControllerDifferentialIK(model, tool_sites="nonexistent", bandwidth=1.0, damping=0.1)

    def test_world_attached_site_is_ignored_and_cannot_be_selected_as_tool(self):
        """A site attached to no body (ModelBuilder.add_site(-1, ...)) must never be mistaken for a tool site.

        Regression test: such a site's body index (-1) must not alias, via
        plain NumPy fancy indexing, onto whatever articulation the model's
        last body happens to belong to.
        """
        device = wp.get_device()
        builder = newton.ModelBuilder()
        link0 = builder.add_link()
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j0], label="arm")
        builder.add_site(link0, label="tip", xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))
        builder.add_site(-1, label="world_ref", xform=wp.transform_identity())
        model = builder.finalize(device=device)

        # An unrelated world-attached site must not disturb normal resolution.
        ControllerDifferentialIK(model, tool_sites="tip", bandwidth=1.0, damping=0.1)

        # Explicitly requesting the world-attached site must raise, not
        # silently alias onto another articulation or reach a Jacobian
        # index computation with a bogus body/joint index.
        with self.assertRaises(ValueError):
            ControllerDifferentialIK(model, tool_sites="world_ref", bandwidth=1.0, damping=0.1)

    def test_is_graphable(self):
        device = wp.get_device()
        model = _build_two_link_arm_with_tool_site(device)
        ctrl = ControllerDifferentialIK(model, tool_sites="tip", bandwidth=1.0, damping=0.1)
        self.assertTrue(ctrl.is_graphable())

    def test_live_bandwidth_and_damping_forwarded(self):
        device = wp.get_device()
        model = _build_two_link_arm_with_tool_site(device)
        ctrl = ControllerDifferentialIK(model, tool_sites="tip", bandwidth=None, damping=None)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=device)
        inputs.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(1.9, 0.2, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.bandwidth = wp.full(2, 3.0, dtype=wp.float32, device=device)
        inputs.damping = wp.full(1, 0.2, dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)
        self.assertTrue(np.all(np.isfinite(outputs.joint_qd_target.numpy())))

    def test_disabled_bandwidth_port_written_raises(self):
        device = wp.get_device()
        model = _build_two_link_arm_with_tool_site(device)
        ctrl = ControllerDifferentialIK(model, tool_sites="tip", bandwidth=1.0, damping=0.1)
        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=device)
        inputs.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        inputs.desired_tool_pose_world = wp.array(
            [wp.transform(p=wp.vec3(2.0, 0.0, 0.0), q=wp.quat_identity())], dtype=wp.transform, device=device
        )
        inputs.bandwidth = wp.full(2, 1.0, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

    def test_null_space_velocity_does_not_disturb_primary_task_for_redundant_chain(self):
        """With zero primary-task error, the entire qd output must satisfy J @ qd == 0."""
        device = wp.get_device()
        model = _build_seven_dof_chain_with_tool_site(device)
        ctrl = ControllerDifferentialIK(
            model,
            tool_sites="tip",
            bandwidth=1.0,
            damping=0.1,
            use_null_space_posture_control=True,
            null_space_stiffness=1.0,
            null_space_damping=0.1,
        )
        joint_q = np.full(7, 0.3, dtype=np.float32)
        state = model.state()
        newton.eval_fk(
            model,
            wp.array(joint_q, dtype=wp.float32, device=device),
            wp.zeros(7, dtype=wp.float32, device=device),
            state,
        )
        tip_pose = state.body_q.numpy()[6]
        tip_world = wp.transform(*tip_pose) * wp.transform(p=wp.vec3(0.0, 0.0, 0.2), q=wp.quat_identity())

        inputs = ctrl.input()
        outputs = ctrl.output()
        inputs.joint_q = wp.array(joint_q, dtype=wp.float32, device=device)
        inputs.joint_qd = wp.zeros(7, dtype=wp.float32, device=device)
        inputs.desired_tool_pose_world = wp.array([tip_world], dtype=wp.transform, device=device)
        inputs.q_des_null = wp.array(np.full(7, 0.5, dtype=np.float32), dtype=wp.float32, device=device)
        ctrl.step(inputs=inputs, outputs=outputs, dt=0.01)

        qd = outputs.joint_qd_target.numpy()
        self.assertTrue(np.any(np.abs(qd) > 1e-4))  # the null-space term did something

        # Verify J @ qd == 0 via a finite-difference tip-position Jacobian,
        # computed independently of the controller (no internal state read).
        def _tip_position(q_np):
            s = model.state()
            newton.eval_fk(
                model, wp.array(q_np, dtype=wp.float32, device=device), wp.zeros(7, dtype=wp.float32, device=device), s
            )
            pose = wp.transform(*s.body_q.numpy()[6])
            return np.array(wp.transform_point(pose, wp.vec3(0.0, 0.0, 0.2)))

        eps = 1e-4
        jacobian_pos = np.zeros((3, 7), dtype=np.float64)
        for i in range(7):
            q_plus = joint_q.copy()
            q_plus[i] += eps
            q_minus = joint_q.copy()
            q_minus[i] -= eps
            jacobian_pos[:, i] = (_tip_position(q_plus) - _tip_position(q_minus)) / (2 * eps)

        position_velocity = jacobian_pos @ qd.astype(np.float64)
        np.testing.assert_allclose(position_velocity, np.zeros(3), atol=1e-2)


if __name__ == "__main__":
    unittest.main()
