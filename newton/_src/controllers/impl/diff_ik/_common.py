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

from typing import Any

import numpy as np
import warp as wp

from ....core.types import Devicelike
from ....math import velocity_at_point

# Cholesky pivots are clamped above this, scaled by the pivot's own
# magnitude, so float32 cancellation noise on a near-singular matrix can't
# drive a pivot negative (which would make the square root below NaN).
_FLOAT32_EPS = wp.constant(wp.float32(np.finfo(np.float32).eps))


# ---------------------------------------------------------------------------
# Tool resolution: the model-based controller resolves each robot's tool
# point from a Newton *site* (a body-fixed offset, ``tool_body`` +
# ``coordinate_change_body_from_tool``), one per robot. These kernels shift
# :func:`~newton.eval_jacobian`'s COM-referenced, world-frame output to the
# tool point, still in world-frame coordinates. Unlike a task that also needs
# the tool's current *twist* (e.g. a velocity-damping term), differential
# kinematics only needs its pose, so there is no twist-shifting kernel here.
#
# A transform that is actively being *composed* with another is named
# ``coordinate_change_TARGET_from_SOURCE``: given a point's coordinates in
# the SOURCE frame, it produces that same point's coordinates in the TARGET
# frame. Warp's ``*`` composes transforms as ``(A * B)(p) = A(B(p))`` (right
# operand applied first), so this naming makes a chain cancel visibly, left
# to right: ``coordinate_change_world_from_body *
# coordinate_change_body_from_tool == tool_pose_world``.
# ---------------------------------------------------------------------------


@wp.kernel
def _tool_pose_kernel(
    body_q: wp.array[wp.transform],  # (body_count,) coordinate_change_world_from_body per body
    tool_body: wp.array[wp.int32],  # (robot_count,) -> body index of each robot's tool site
    coordinate_change_body_from_tool: wp.array[wp.transform],  # (robot_count,) tool site's body-local transform
    # outputs
    tool_pose_world: wp.array[wp.transform],  # (robot_count,) world pose of the tool frame
):
    robot_idx = wp.tid()
    tool_body_idx = tool_body[robot_idx]
    tool_pose_world[robot_idx] = body_q[tool_body_idx] * coordinate_change_body_from_tool[robot_idx]


@wp.kernel
def _shift_jacobian_to_tool_kernel(
    jacobian_com_world: wp.array3d[
        float
    ],  # (articulation_count, max_links*6, max_dofs) columns are twists about each link's COM point, in world coords
    body_q: wp.array[wp.transform],  # (body_count,) coordinate_change_world_from_body per body
    body_com_body: wp.array[wp.vec3],  # (body_count,) COM position, in the body's own local frame
    tool_body: wp.array[wp.int32],  # (robot_count,) -> body index of each robot's tool site
    coordinate_change_body_from_tool: wp.array[wp.transform],  # (robot_count,) tool site's body-local transform
    robot_articulation: wp.array[wp.int32],  # (robot_count,) -> articulation index into jacobian_com_world
    robot_link_idx: wp.array[wp.int32],  # (robot_count,) -> row-block index of the tool's link, within its articulation
    articulation_dof_idx_of_padded_dof_idx: wp.array2d[
        wp.int32
    ],  # (robot_count, max_dofs) padded_dof_idx -> articulation_dof_idx, jacobian_com_world's own column numbering
    controlled_dofs_per_robot: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
):
    """Shift a COM-referenced Jacobian to the tool point, one output column at a time.

    A controlled robot's DOFs are not necessarily the first columns of its
    own articulation's Jacobian -- ``joints`` may select a non-prefix subset,
    or skip an uncontrolled joint interspersed among controlled ones -- so
    ``articulation_dof_idx_of_padded_dof_idx`` remaps each padded output
    column (``padded_dof_idx``) to the actual column ``jacobian_com_world``
    stores it at (``articulation_dof_idx``).
    """
    robot_idx, padded_dof_idx = wp.tid()
    if padded_dof_idx >= controlled_dofs_per_robot[robot_idx]:
        return
    articulation_idx = robot_articulation[robot_idx]
    link_row_start = robot_link_idx[robot_idx] * 6
    articulation_dof_idx = articulation_dof_idx_of_padded_dof_idx[robot_idx, padded_dof_idx]

    tool_body_idx = tool_body[robot_idx]
    coordinate_change_world_from_body = body_q[tool_body_idx]
    tool_pose_world = coordinate_change_world_from_body * coordinate_change_body_from_tool[robot_idx]
    tool_point_world = wp.transform_get_translation(tool_pose_world)
    body_com_world = wp.transform_point(coordinate_change_world_from_body, body_com_body[tool_body_idx])
    com_to_tool_offset_world = tool_point_world - body_com_world

    jacobian_column_com_world = wp.spatial_vector(
        jacobian_com_world[articulation_idx, link_row_start + 0, articulation_dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 1, articulation_dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 2, articulation_dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 3, articulation_dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 4, articulation_dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 5, articulation_dof_idx],
    )
    jacobian_column_tool_world = wp.spatial_vector(
        velocity_at_point(jacobian_column_com_world, com_to_tool_offset_world),
        wp.spatial_bottom(jacobian_column_com_world),
    )
    for row in range(6):
        jacobian_tool_world[robot_idx, row, padded_dof_idx] = jacobian_column_tool_world[row]


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


@wp.func
def _cholesky_factor6(matrix: wp.array3d[float], robot_idx: int, cholesky_factor: wp.array3d[float]):
    """In-place lower-triangular Cholesky factorization of one batch entry of a fixed 6x6 SPD matrix.

    Writes into ``cholesky_factor[robot_idx]`` such that ``matrix[robot_idx]
    == cholesky_factor[robot_idx] @ cholesky_factor[robot_idx]ᵀ``. Shared by
    every kernel that needs to solve or pseudo-invert a 6x6 SPD system, so
    the factorization itself is written once.
    """
    for col in range(6):
        diagonal_term = matrix[robot_idx, col, col]
        for prior_col in range(col):
            diagonal_term -= cholesky_factor[robot_idx, col, prior_col] * cholesky_factor[robot_idx, col, prior_col]
        diagonal_term = wp.max(diagonal_term, _FLOAT32_EPS * wp.max(wp.abs(matrix[robot_idx, col, col]), 1.0))
        diagonal_value = wp.sqrt(diagonal_term)
        cholesky_factor[robot_idx, col, col] = diagonal_value
        for row in range(col + 1, 6):
            off_diagonal_term = matrix[robot_idx, row, col]
            for prior_col in range(col):
                off_diagonal_term -= (
                    cholesky_factor[robot_idx, row, prior_col] * cholesky_factor[robot_idx, col, prior_col]
                )
            cholesky_factor[robot_idx, row, col] = off_diagonal_term / diagonal_value


@wp.func
def _cholesky_solve6(cholesky_factor: wp.array3d[float], robot_idx: int, rhs: wp.spatial_vector) -> wp.spatial_vector:
    """Forward/back-substitute a single 6-vector right-hand side against an already-factorized Cholesky factor.

    Forward-substitutes ``L y' = rhs``, then back-substitutes ``Lᵀ y = y'``.
    Requires :func:`_cholesky_factor6` to have already written
    ``cholesky_factor[robot_idx]``.
    """
    forward_solution = wp.spatial_vector()
    for row in range(6):
        right_hand_side = rhs[row]
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
    return solution


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

    Solving directly for one right-hand side, rather than forming the full
    inverse, is cheaper here since DLS only ever needs ``(JJᵀ + λ²I)⁻¹e``
    applied to the single vector ``e``, never the matrix itself.
    """
    robot_idx = wp.tid()
    _cholesky_factor6(spd_matrix, robot_idx, cholesky_factor)
    y[robot_idx] = _cholesky_solve6(cholesky_factor, robot_idx, rhs[robot_idx])


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
    joint_qd_target: wp.array[wp.float32],  # (total_controlled_dofs,) compact = bandwidth * Jᵀ @ y
):
    """Finish the damped-least-squares solve, ``q̇_target = bandwidth · Jᵀy``, straight into the compact per-DOF layout.

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

    joint_qd_target[dof] = bandwidth[dof] * wp.dot(jacobian_column, y[robot])


# ---------------------------------------------------------------------------
# Null-space projector: N = I - Jᵀ @ (JJᵀ + λ_null²I)⁻¹J, a *damped* kinematic
# (Moore-Penrose) projector. ``λ_null`` is its own damping, independent of the primary task's DLS damping.
#
# Damping (λ_null > 0) makes JJᵀ + λ_null²I SPD for any Jacobian, including
# one that is not full row rank -- e.g. a redundant low-DOF arm whose task is
# itself lower-dimensional than 6 (a 4R planar arm controlling only a 2D or
# 3D in-plane task has exactly this shape: 4 controlled DOFs, a structurally
# rank-deficient 6x6 JJᵀ, yet still genuinely redundant). This is the same
# Tikhonov-regularization argument as the primary DLS solve, applied to the
# projector instead: J @ N is no longer exactly zero, but
# ``J @ N = λ_null²(JJᵀ + λ_null²I)⁻¹J``, a residual of order ``λ_null²`` --
# the same order of primary-task tracking error the DLS solve itself already
# accepts near a singularity, not a new class of imprecision.
# ---------------------------------------------------------------------------


@wp.kernel
def _jacobian_pinv_transpose_kernel(
    jjt_plus_damping: wp.array3d[float],  # (robot_count, 6, 6) symmetric positive-definite, J @ Jᵀ + λ_null²I
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, world coords
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # scratch, preallocated by the caller (not valid on entry; written and then read within this kernel)
    cholesky_factor: wp.array3d[float],  # (robot_count, 6, 6) lower-triangular L such that jjt_plus_damping = L Lᵀ
    # outputs
    jacobian_pinv_transpose: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) = (JJᵀ + λ_null²I)⁻¹ @ J; zero beyond dof_count
):
    """The damped pseudo-inverse-transpose, ``(JJᵀ + λ_null²I)⁻¹ @ J``, solved column by column.

    Reduces to the exact Moore-Penrose ``(J⁺)ᵀ`` as ``λ_null → 0`` for a
    full-row-rank ``J``. Factorizes ``jjt_plus_damping`` once per robot, then
    solves ``jjt_plus_damping @ x = J[:, dof]`` for each controlled DOF's
    column — the batch of solutions is exactly ``(JJᵀ + λ_null²I)⁻¹ @ J``,
    without ever forming the inverse matrix on its own.
    """
    robot_idx = wp.tid()
    _cholesky_factor6(jjt_plus_damping, robot_idx, cholesky_factor)
    for dof in range(dof_count[robot_idx]):
        column = wp.spatial_vector()
        for row in range(6):
            column[row] = jacobian_tool_world[robot_idx, row, dof]
        solution = _cholesky_solve6(cholesky_factor, robot_idx, column)
        for row in range(6):
            jacobian_pinv_transpose[robot_idx, row, dof] = solution[row]


@wp.kernel
def _null_space_projector_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, world coords
    jacobian_pinv_transpose: wp.array3d[float],  # (robot_count, 6, max_dofs) = (JJᵀ)⁻¹ @ J; zero beyond dof_count
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    null_space_projector: wp.array3d[
        float
    ],  # (robot_count, max_dofs, max_dofs) = I - Jᵀ @ jacobian_pinv_transpose; untouched beyond dof_count
):
    """The null-space projector, ``N = I - Jᵀ @ (JJᵀ)⁻¹J``."""
    robot_idx, row, col = wp.tid()
    robot_dof_count = dof_count[robot_idx]
    if row >= robot_dof_count or col >= robot_dof_count:
        return

    identity_entry = float(0.0)
    if row == col:
        identity_entry = 1.0

    total = float(0.0)
    for k in range(6):
        total += jacobian_tool_world[robot_idx, k, row] * jacobian_pinv_transpose[robot_idx, k, col]
    null_space_projector[robot_idx, row, col] = identity_entry - total


@wp.kernel
def _block_matrix_vector_multiply_kernel(
    block_matrix: wp.array3d[float],  # (controlled_robot_count, max_controlled_dofs, max_controlled_dofs)
    vec: wp.array[wp.float32],  # (total_controlled_dofs,)
    robot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> row within that robot's block
    dof_offsets: wp.array[wp.int32],  # (controlled_robot_count,) -> first flat DOF of each robot
    controlled_dofs_per_robot: wp.array[wp.int32],  # (controlled_robot_count,)
    # outputs
    out: wp.array[wp.float32],  # (total_controlled_dofs,) = block_matrix @ vec
):
    """Multiply a compact per-DOF vector by a padded per-robot square matrix, ``out = block_matrix @ vec``."""
    dof = wp.tid()
    robot = robot_of_dof[dof]
    row = slot_of_dof[dof]
    row_base = dof_offsets[robot]
    acc = float(0.0)
    for col in range(controlled_dofs_per_robot[robot]):
        acc = acc + block_matrix[robot, row, col] * vec[row_base + col]
    out[dof] = acc


@wp.kernel
def _add_term_kernel(
    term: wp.array[wp.float32],  # (total_controlled_dofs,)
    # outputs
    accumulator: wp.array[wp.float32],  # (total_controlled_dofs,) += term
):
    dof = wp.tid()
    accumulator[dof] = accumulator[dof] + term[dof]


# ---------------------------------------------------------------------------
# Null-space secondary objectives: a joint-space bias vector, projected
# through the null-space projector above so it never disturbs the primary
# task. Joint-limit avoidance and posture control both produce this kind of
# bias and may be combined (added together) before projecting.
# ---------------------------------------------------------------------------


@wp.kernel
def _joint_limit_avoidance_bias_kernel(
    joint_q: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_pos_lower: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_pos_upper: wp.array[wp.float32],  # (total_controlled_dofs,)
    gain: wp.float32,  # joint-centering gain
    margin: wp.float32,  # activation ramps 0 -> 1 as the distance to the nearer limit shrinks from margin to 0
    # outputs
    dq_center: wp.array[wp.float32],  # (total_controlled_dofs,) = -gain * activation * (q - q_mid)
):
    """Joint-limit-avoidance bias: pulls a DOF toward its range midpoint as it nears either limit.

    ``activation`` is 0 while more than ``margin`` away from both limits,
    ramps linearly to 1 at either limit, and stays 1 beyond it — a DOF
    already past its limit gets the full correction, not none.
    """
    dof = wp.tid()
    q = joint_q[dof]
    lower = joint_pos_lower[dof]
    upper = joint_pos_upper[dof]
    q_mid = 0.5 * (lower + upper)

    dist_to_limit = wp.min(q - lower, upper - q)
    activation = float(0.0)
    if dist_to_limit <= 0.0:
        activation = 1.0
    elif dist_to_limit < margin:
        activation = 1.0 - dist_to_limit / margin

    dq_center[dof] = -gain * activation * (q - q_mid)


@wp.kernel
def _posture_bias_kernel(
    joint_q: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_q_des_null: wp.array[wp.float32],  # (total_controlled_dofs,)
    stiffness: wp.array[wp.float32],  # (total_controlled_dofs,)
    # outputs
    dq_center: wp.array[wp.float32],  # (total_controlled_dofs,) = stiffness * (joint_q_des_null - joint_q)
):
    """Null-space posture bias, a proportional-only joint-space pull toward ``joint_q_des_null``."""
    dof = wp.tid()
    dq_center[dof] = stiffness[dof] * (joint_q_des_null[dof] - joint_q[dof])


# ---------------------------------------------------------------------------
# Integration: one-step-ahead joint position target from the solved velocity.
# ---------------------------------------------------------------------------


@wp.kernel
def _integrate_position_kernel(
    joint_q: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_qd_target: wp.array[wp.float32],  # (total_controlled_dofs,)
    dt: wp.array[wp.float32],  # (1,) step duration [s]
    # outputs
    joint_q_target: wp.array[wp.float32],  # (total_controlled_dofs,) = joint_q + joint_qd_target * dt
):
    """Explicit-Euler joint position target, ``q_target = q + q̇_target·dt``."""
    dof = wp.tid()
    joint_q_target[dof] = joint_q[dof] + joint_qd_target[dof] * dt[0]


# ---------------------------------------------------------------------------
# Port plumbing: wp.copy is not recordable under APIC graph capture when
# either side is non-contiguous, which every indexed-view port is. These
# kernels do the same work in a form that captures and serialises. A
# controller launches them at its own port length: one entry per controlled
# DOF for a compact port, one per robot for a per-robot port.
# ---------------------------------------------------------------------------


@wp.kernel
def _gather_rank1_port_kernel(
    port: wp.indexedarray(dtype=Any),  # view of a simulation-sized array
    # outputs
    out: wp.array[Any],  # one entry per element the view addresses
):
    dof = wp.tid()
    out[dof] = port[dof]


@wp.kernel
def _gather_rank3_port_kernel(
    port: wp.indexedarray(dtype=wp.float32, ndim=3),  # view of a simulation-sized 3-D array
    # outputs
    out: wp.array3d[wp.float32],  # one entry per element the view addresses
):
    i, j, k = wp.tid()
    out[i, j, k] = port[i, j, k]


@wp.kernel
def _scatter_port_kernel(
    values: wp.array[wp.float32],  # one entry per element the view addresses
    # outputs
    port: wp.indexedarray[wp.float32],  # view of a simulation-sized array
):
    dof = wp.tid()
    port[dof] = values[dof]


# dtype -> (rank -> gather kernel), the set of dtype/rank combinations this
# controller family's ports use. Extend this table, not _read_port itself,
# to support a new port dtype or rank. Every rank-1 dtype shares
# _gather_rank1_port_kernel: it's generic over dtype (Any), so Warp compiles
# one concrete kernel per dtype the table actually uses, from a single body.
_GATHER_KERNELS_BY_DTYPE_AND_RANK = {
    wp.float32: {1: _gather_rank1_port_kernel, 3: _gather_rank3_port_kernel},
    wp.transform: {1: _gather_rank1_port_kernel},
}


def _read_port(
    port: wp.array | wp.indexedarray,
    buffer: wp.array,
    shape: int | tuple[int, ...],
    device: Devicelike,
) -> None:
    """Copy a bound port into an internal buffer, whatever it is bound to.

    A view has to go through a kernel: :func:`warp.copy` is not recordable
    under APIC graph capture when either side is non-contiguous, so using it
    here would make a controller that reports ``is_graphable()`` fail to
    export.

    Args:
        port: The caller-bound port, a :class:`warp.array` or a view of one.
            Any dtype/rank combination in :data:`_GATHER_KERNELS_BY_DTYPE_AND_RANK`
            is supported when ``port`` is a view; a plain array supports any
            dtype/rank, since :func:`warp.copy` doesn't care.
        buffer: Destination, matching ``port`` in shape and dtype.
        shape: Launch shape — the length for a 1-D port, ``(robots, rows, cols)``
            for the Jacobian.
        device: Device to launch on.
    """
    if not isinstance(port, wp.indexedarray):
        wp.copy(buffer, port)
        return

    # A kernel parameter's dtype and dimensionality are part of its type, so
    # a view needs the kernel that matches both.
    kernels_by_rank = _GATHER_KERNELS_BY_DTYPE_AND_RANK.get(port.dtype)
    kernel = kernels_by_rank.get(port.ndim) if kernels_by_rank is not None else None
    if kernel is None:
        raise TypeError(
            f"_read_port has no gather kernel for a {port.ndim}-D indexed array of dtype {port.dtype}; "
            f"add one to _GATHER_KERNELS_BY_DTYPE_AND_RANK in controllers/impl/diff_ik/_common.py."
        )
    wp.launch(kernel, dim=shape, inputs=[port], outputs=[buffer], device=device)
