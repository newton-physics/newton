# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for the differential-kinematics controllers.

A controlled robot's task is always a full 6D pose (3 position + 3
orientation), so every task-space quantity here is a fixed-size
``wp.spatial_vector`` or a fixed ``6x6`` block; there is no per-axis
selection or operational-frame rotation (yet) — pose error is computed
directly in world frame. The Jacobian is laid out as ``(robot_count, 6,
max_dofs)``, each column a per-DOF twist about the tool point, expressed in
world-frame coordinates, with columns beyond a robot's own controlled-DOF
count left unused (see ``dof_count``).

The inverse-Jacobian solve is isolated to its own group of kernels
(``_build_jjt_plus_damping_kernel``, ``_cholesky_solve6_kernel``,
``_qd_from_y_kernel``) so that a future solver — plain transpose,
zero-damping pseudo-inverse, or singularity-adaptive damping — can be added
as its own kernel group without touching pose-error, null-space, or
integration code. Only damped least squares (Levenberg-Marquardt-style
Tikhonov regularization) is implemented so far,
``q̇ = bandwidth · Jᵀ(JJᵀ + λ²I)⁻¹e``.

This single fixed ``6x6`` form is exact for a robot with any number of
controlled DOFs, not just ``n_joints ≥ 6``: the push-through identity
``Jᵀ(JJᵀ + λ²I)⁻¹ == (JᵀJ + λ²I)⁻¹Jᵀ`` holds for any shape of ``J`` whenever
``λ > 0``, so there is no separate "overdetermined" ``n_joints x n_joints``
code path to get wrong for a heterogeneous fleet mixing DOF counts — every
robot solves the same fixed ``6x6`` system regardless of its own DOF count.
This only breaks down at exactly ``λ = 0`` (the zero-damping pseudo-inverse,
not implemented yet): ``JJᵀ`` is then rank-deficient whenever
``dof_count < 6``, and while the Cholesky pivot floor below keeps that from
producing NaN, it does not produce a meaningful pseudo-inverse in that
regime — a caller wanting an undamped solve will need a per-robot DOF-count
check when that solver is added.
"""

from __future__ import annotations

import numpy as np
import warp as wp

# Cholesky pivots are clamped above this, scaled by the pivot's own
# magnitude, so float32 cancellation noise on a near-singular matrix can't
# drive a pivot negative (which would make the square root below NaN).
_FLOAT32_EPS = wp.constant(wp.float32(np.finfo(np.float32).eps))


# ---------------------------------------------------------------------------
# Task-space pose error: how far the tool is from where it should be.
# ---------------------------------------------------------------------------


@wp.kernel
def _pose_error_kernel(
    current_pose: wp.array[wp.transform],  # (robot_count,) current tool pose, world frame
    desired_pose: wp.array[wp.transform],  # (robot_count,) desired tool pose, world frame
    # outputs
    pose_error: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) (position error, orientation error), world frame, desired minus current
):
    """Task-space pose error, ``(desired_position - current_position, orientation_error)``.

    The position error is a plain vector difference.

    The orientation error is the axis-angle rotation that would carry the
    current orientation to the desired one: rotate the current orientation by
    ``angle`` about ``axis`` and it lands on the desired orientation. It
    shrinks to zero exactly when the two orientations agree, matching the
    position error's "desired minus current" sign so both halves of the 6D
    error can be driven to zero by the same kind of proportional term.

    Derivation: with quaternions written so ``q * p`` composes like Warp's
    ``transform *`` (apply ``p`` first, then ``q``), the rotation that "undoes
    current, then applies desired" is ``quat_error = q_desired * q_current^-1``.
    Its axis-angle form is exactly that carrying rotation. Extracting it
    inlines Warp's own ``quat_to_axis_angle`` formula rather than calling it
    directly, because that builtin divides by the quaternion's vector-part
    norm with no guard — it returns NaN once the two orientations are close
    enough that the norm underflows, which is exactly the common steady-state
    case for a pose tracker. The small-angle branch below is quat_error's
    first-order Taylor expansion instead: for a unit quaternion near
    identity, ``quat_error ~= (1, half_angle * axis)``, so ``2 * vector_part
    ~= angle * axis`` directly, with no division at all.
    """
    robot_idx = wp.tid()

    current = current_pose[robot_idx]
    desired = desired_pose[robot_idx]
    position_error = wp.transform_get_translation(desired) - wp.transform_get_translation(current)

    quat_current = wp.transform_get_rotation(current)
    quat_desired = wp.transform_get_rotation(desired)
    quat_error = quat_desired * wp.quat_inverse(quat_current)
    # Every unit quaternion has two equally valid representations, q and -q;
    # picking the one with a non-negative scalar part is what keeps the
    # extracted angle in [0, pi] (the shorter of the two possible rotations)
    # instead of occasionally reporting the longer way around.
    if quat_error[3] < 0.0:
        quat_error = -quat_error

    quat_error_vector = wp.vec3(quat_error[0], quat_error[1], quat_error[2])
    quat_error_vector_norm = wp.length(quat_error_vector)
    if quat_error_vector_norm > 1.0e-8:
        angle = 2.0 * wp.atan2(quat_error_vector_norm, quat_error[3])
        orientation_error = (quat_error_vector / quat_error_vector_norm) * angle
    else:
        orientation_error = 2.0 * quat_error_vector

    pose_error[robot_idx] = wp.spatial_vector(position_error, orientation_error)


# ---------------------------------------------------------------------------
# Damped least squares: q̇ = bandwidth · Jᵀ(JJᵀ + λ²I)⁻¹e, the minimum-norm
# solution. Built in three kernels — normal-equations matrix, Cholesky solve,
# finish — so a future solver only has to replace this group, not the error
# or integration kernels around it.
# ---------------------------------------------------------------------------


@wp.kernel
def _build_jjt_plus_damping_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, world coords
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    damping: wp.array[wp.float32],  # (robot_count,) DLS damping λ, added as λ² on the diagonal
    # outputs
    jjt_plus_damping: wp.array3d[float],  # (robot_count, 6, 6) = J @ Jᵀ + λ² * I
):
    """Build the damped-least-squares normal-equations matrix, ``JJᵀ + λ²I``.

    Always the full 6x6 (not padded/guarded by ``dof_count`` on the output
    side): the task space is always exactly 6D, so this matrix is exactly
    6x6 regardless of how many DOFs a robot has. ``λ² > 0`` keeps it positive
    definite (and so invertible) even when ``JJᵀ`` alone is singular or
    rank-deficient, e.g. at a kinematic singularity or when a robot controls
    fewer than 6 DOFs.
    """
    robot_idx, row, col = wp.tid()
    robot_dof_count = dof_count[robot_idx]
    total = float(0.0)
    for dof in range(robot_dof_count):
        total += jacobian_tool_world[robot_idx, row, dof] * jacobian_tool_world[robot_idx, col, dof]
    if row == col:
        lam = damping[robot_idx]
        total += lam * lam
    jjt_plus_damping[robot_idx, row, col] = total


@wp.kernel
def _cholesky_solve6_kernel(
    spd_matrix: wp.array3d[float],  # (robot_count, 6, 6) symmetric positive-definite, e.g. JJᵀ + λ²I
    rhs: wp.array[wp.spatial_vector],  # (robot_count,) right-hand side, e.g. pose_error
    # scratch, preallocated by the caller (not valid on entry; written and then read within this kernel)
    cholesky_factor: wp.array3d[float],  # (robot_count, 6, 6) lower-triangular L such that spd_matrix = L Lᵀ
    # outputs
    y: wp.array[wp.spatial_vector],  # (robot_count,) solves spd_matrix @ y = rhs
):
    """Solve a batch of fixed 6x6 SPD systems for a single right-hand side, via Cholesky factorization.

    Forward-substitutes ``L y' = rhs``, then back-substitutes ``Lᵀ y = y'``.
    Solving directly for one right-hand side, rather than forming the full
    inverse, is cheaper here since DLS only ever needs ``(JJᵀ + λ²I)⁻¹e``
    applied to the single vector ``e``, never the matrix itself.
    """
    robot_idx = wp.tid()

    for col in range(6):
        diagonal_term = spd_matrix[robot_idx, col, col]
        for prior_col in range(col):
            diagonal_term -= cholesky_factor[robot_idx, col, prior_col] * cholesky_factor[robot_idx, col, prior_col]
        diagonal_term = wp.max(diagonal_term, _FLOAT32_EPS * wp.max(wp.abs(spd_matrix[robot_idx, col, col]), 1.0))
        diagonal_value = wp.sqrt(diagonal_term)
        cholesky_factor[robot_idx, col, col] = diagonal_value
        for row in range(col + 1, 6):
            off_diagonal_term = spd_matrix[robot_idx, row, col]
            for prior_col in range(col):
                off_diagonal_term -= (
                    cholesky_factor[robot_idx, row, prior_col] * cholesky_factor[robot_idx, col, prior_col]
                )
            cholesky_factor[robot_idx, row, col] = off_diagonal_term / diagonal_value

    forward_solution = wp.spatial_vector()
    for row in range(6):
        right_hand_side = rhs[robot_idx][row]
        for prior_row in range(row):
            right_hand_side -= cholesky_factor[robot_idx, row, prior_row] * forward_solution[prior_row]
        forward_solution[row] = right_hand_side / cholesky_factor[robot_idx, row, row]

    solution = wp.spatial_vector()
    for reverse_row in range(6):
        row = 5 - reverse_row
        right_hand_side = forward_solution[row]
        for later_row in range(row + 1, 6):
            right_hand_side -= cholesky_factor[robot_idx, later_row, row] * solution[later_row]
        solution[row] = right_hand_side / cholesky_factor[robot_idx, row, row]

    y[robot_idx] = solution


@wp.kernel
def _qd_from_y_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, world coords
    y: wp.array[wp.spatial_vector],  # (robot_count,) solves (JJᵀ + λ²I) y = pose_error
    bandwidth: wp.array[wp.float32],  # (total_controlled_dofs,) output scale gain
    robot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> column within that robot's Jacobian
    # outputs
    joint_qd: wp.array[wp.float32],  # (total_controlled_dofs,) compact = bandwidth * Jᵀ @ y
):
    """Finish the damped-least-squares solve, ``q̇ = bandwidth · Jᵀy``, straight into the compact per-DOF layout.

    Row ``dof`` of ``Jᵀ`` is column ``slot_of_dof[dof]`` of robot
    ``robot_of_dof[dof]``'s Jacobian — loading it back into a
    ``wp.spatial_vector`` here lets this kernel use Warp's built-in dot
    product instead of a hand-rolled sum.
    """
    dof = wp.tid()
    robot = robot_of_dof[dof]
    slot = slot_of_dof[dof]

    jacobian_column = wp.spatial_vector()
    for row in range(6):
        jacobian_column[row] = jacobian_tool_world[robot, row, slot]

    joint_qd[dof] = bandwidth[dof] * wp.dot(jacobian_column, y[robot])


# ---------------------------------------------------------------------------
# Integration: one-step-ahead joint position target from the solved velocity.
# ---------------------------------------------------------------------------


@wp.kernel
def _integrate_position_kernel(
    joint_q: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_qd: wp.array[wp.float32],  # (total_controlled_dofs,)
    dt: wp.array[wp.float32],  # (1,) step duration [s]
    # outputs
    joint_q_target: wp.array[wp.float32],  # (total_controlled_dofs,) = joint_q + joint_qd * dt
):
    """Explicit-Euler joint position target, ``q_target = q + q̇·dt``."""
    dof = wp.tid()
    joint_q_target[dof] = joint_q[dof] + joint_qd[dof] * dt[0]
