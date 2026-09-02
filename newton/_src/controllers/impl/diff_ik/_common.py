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
(``_build_jjt_plus_damping_kernel``, plus the shared
``_invert_spd_block_kernel``/``_apply_spatial_matrix_kernel`` in
``controllers/impl/_common.py``, and ``_qd_from_y_kernel``) so that a future
solver — plain transpose, zero-damping pseudo-inverse, or
singularity-adaptive damping — can be added as its own kernel group without
touching pose-error, null-space, or integration code. Only damped least
squares (Levenberg-Marquardt-style Tikhonov regularization) is implemented
so far, ``q̇ = bandwidth · Jᵀ(JJᵀ + λ²I)⁻¹e``.

This single fixed ``6x6`` form is exact for a robot with any number of
controlled DOFs, not just ``n_joints ≥ 6``: the push-through identity
``Jᵀ(JJᵀ + λ²I)⁻¹ == (JᵀJ + λ²I)⁻¹Jᵀ`` holds for any shape of ``J`` whenever
``λ > 0``, so there is no separate "overdetermined" ``n_joints x n_joints``
code path to get wrong for a heterogeneous fleet mixing DOF counts — every
robot solves the same fixed ``6x6`` system regardless of its own DOF count.
This only breaks down at exactly ``λ = 0`` (the zero-damping pseudo-inverse,
not implemented yet): ``JJᵀ`` is then rank-deficient whenever
``dof_count < 6``, and while the Cholesky pivot floor in
``_invert_spd_block_kernel`` keeps that from producing NaN, it does not
produce a meaningful pseudo-inverse in that regime — a caller wanting an
undamped solve will need a per-robot DOF-count check when that solver is
added.
"""

from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------------------
# Tool pose resolution: the model-based controller resolves each robot's tool
# point from a Newton *site* (a body-fixed offset, ``tool_body`` +
# ``coordinate_change_body_from_tool``), one per robot. Differential
# kinematics only needs the tool's pose, not its twist, so this is a
# pose-only kernel.
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


# ---------------------------------------------------------------------------
# Damped least squares: q̇ = bandwidth · Jᵀ(JJᵀ + λ²I)⁻¹e, the minimum-norm
# solution. Built in three steps — normal-equations matrix, invert-and-apply,
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
# Null-space secondary objectives.
#
# The null-space projector itself, and the damped pseudo-inverse-transpose
# ``(JJᵀ + λ_null²I)⁻¹ @ J`` it is built from, use kernels shared with other
# controller families (``_invert_spd_block_kernel``,
# ``_task_matrix_times_jacobian_kernel``, ``_null_space_projector_kernel`` in
# ``controllers/impl/_common.py``), not dedicated kernels here — see
# :class:`ControllerDiffIKModelFree`. ``λ_null`` is its own damping,
# independent of the primary task's DLS damping: it makes ``JJᵀ + λ_null²I``
# SPD for any Jacobian, including one that is not full row rank (e.g. a
# redundant low-DOF arm whose task is itself lower-dimensional than 6, such
# as a 4R planar arm controlling a 2D or 3D in-plane task). The tradeoff is
# the same kind the primary DLS solve already accepts: the resulting
# projector satisfies ``J @ N = λ_null²(JJᵀ + λ_null²I)⁻¹J`` — a residual of
# order ``λ_null²``, not a new class of imprecision.
#
# The kernels below produce a joint-space bias vector, projected through
# that projector so it never disturbs the primary task. Joint-limit
# avoidance and posture control both produce this kind of bias and may be
# combined (added together) before projecting.
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
