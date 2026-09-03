# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for the differential-kinematics controllers.

A controlled robot's task-space quantity is a fixed-size ``wp.spatial_vector``
or a fixed ``6x6`` block, always, regardless of how each of the 6 canonical
axes is weighted — see the section comment above ``_gather_task_error_kernel``
for how per-axis weighting (``axis_weight``) is applied and why a zero-weighted
axis is excluded structurally rather than multiplied by zero.

The inverse-Jacobian solve is isolated to its own group of kernels (see the
section comment above ``_build_jjt_plus_damping_kernel``) so that a solver
can be selected per instance via :class:`DifferentialIKMethod` without touching
pose-error, null-space, or integration code.
"""

from __future__ import annotations

import enum

import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

# Tolerance passed to symmetric_eigenvalues_qr: the QR algorithm iterates
# until the off-diagonal terms of the tridiagonalized matrix fall below this,
# relative to the diagonal terms.
_EIGENVALUE_QR_TOL = wp.constant(wp.float32(1.0e-6))


class DifferentialIKMethod(enum.Enum):
    """Inverse-Jacobian solve method for :class:`ControllerDifferentialIKModelFree`/:class:`ControllerDifferentialIK`.

    Import directly from ``newton.controllers``, the same way as any other
    top-level enum (e.g. ``JointTargetMode``): ``from newton.controllers
    import DifferentialIKMethod``.
    """

    DAMPED_LEAST_SQUARES = "damped_least_squares"
    """``q̇ = bandwidth · Jᵀ(JJᵀ + λ²I)⁻¹e``. The default; uses ``damping``."""

    PSEUDO_INVERSE = "pseudo_inverse"
    """Zero-damping Moore-Penrose pseudo-inverse (``λ = 0`` in the same solve). Requires every robot to have at
    least as many controlled DOFs as its own task dimension (the number of nonzero ``axis_weight`` entries), and
    ``damping=None`` (there is no λ to set)."""

    TRANSPOSE = "transpose"
    """``q̇ = bandwidth · Jᵀe``, no matrix inversion. Requires ``damping=None`` (there is no λ to set)."""

    ADAPTIVE_DAMPING = "adaptive_damping"
    """Damped least squares with λ computed each step from ``JJᵀ``'s smallest eigenvalue (Maciejewski-Klein
    singularity-robust damping), instead of a fixed ``damping``. Requires ``damping=None`` and
    ``adaptive_damping_min``/``adaptive_damping_max``/``adaptive_damping_threshold``."""

    TRUNCATED_SVD = "truncated_svd"
    """Per-direction pseudo-inverse from ``JJᵀ``'s full eigendecomposition: a task-space direction with singular
    value above ``truncated_svd_threshold`` is inverted exactly (``1/sigma²``), one below it is dropped entirely
    (``0``) rather than damped. Requires ``damping=None`` and ``truncated_svd_threshold``."""


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
# axis_weight: the shared _pose_error_kernel (controllers/impl/_common.py)
# always computes a full 6D pose error, since other controller families have
# no notion of per-axis weighting — this kernel gathers only the axes with a
# nonzero axis_weight into a compact, contiguous ``task_dim``-wide
# representation (weighted by that same axis_weight), via a per-robot table
# (``active_axis_of_slot``) built once at construction from whichever axes
# are active. The gather is what keeps a zero-weighted axis's row/column
# structurally excluded from every downstream kernel's arithmetic — not
# merely multiplied by zero — which matters for both numerical robustness at
# ``λ = 0`` and gradient correctness under ``requires_grad``: a value that is
# never read contributes an exactly-zero gradient, where "coefficient times
# an exactly-zero input" would not.
# ---------------------------------------------------------------------------


@wp.kernel
def _gather_task_error_kernel(
    pose_error: wp.array[wp.spatial_vector],  # (robot_count,) full 6D error, canonical axis order
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    active_axis_of_slot: wp.array2d[wp.int32],  # (robot_count, 6) compact slot -> canonical axis, slot < task_dim
    axis_weight: wp.array[wp.spatial_vector],  # (robot_count,) per-canonical-axis weight, > 0 where active
    # outputs
    pose_error_active: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) compact, weighted: slot < task_dim real, rest exactly zero
):
    """Gather a pose error's active axes into a compact, contiguous, weighted representation, ``e_weighted = diag(w) @ e``.

    Load-bearing for ``DifferentialIKMethod.TRANSPOSE`` (which uses this directly as
    ``y``, with nothing else to filter it); every solve that inverts ``JJᵀ``
    also consumes this rather than the raw 6D error, so ``pose_error``
    reads the same way everywhere — the error actually being driven to
    zero — regardless of which solver is selected.
    """
    robot_idx = wp.tid()
    dim = task_dim[robot_idx]
    error = pose_error[robot_idx]
    result = wp.spatial_vector()
    for slot in range(6):
        if slot < dim:
            axis = active_axis_of_slot[robot_idx, slot]
            result[slot] = axis_weight[robot_idx][axis] * error[axis]
        else:
            result[slot] = 0.0
    pose_error_active[robot_idx] = result


# ---------------------------------------------------------------------------
# Damped least squares: q̇ = bandwidth · Jᵀ(JJᵀ + λ²I)⁻¹e, the minimum-norm
# solution. Built in three steps — normal-equations matrix, invert-and-apply,
# finish — so a future solver only has to replace this group, not the error
# or integration kernels around it.
#
# This single fixed 6x6 form is exact for a robot with any number of
# controlled DOFs, not just n_joints >= 6: the push-through identity
# Jᵀ(JJᵀ + λ²I)⁻¹ == (JᵀJ + λ²I)⁻¹Jᵀ holds for any shape of J whenever
# λ > 0, so there is no separate "overdetermined" n_joints x n_joints code
# path to get wrong for a heterogeneous fleet mixing DOF counts. This only
# breaks down at exactly λ = 0 (DifferentialIKMethod.PSEUDO_INVERSE): JJᵀ is then
# rank-deficient whenever dof_count is below the robot's own task dimension,
# and while the Cholesky pivot floor in _invert_spd_block_kernel keeps that
# from producing NaN, it does not produce a meaningful pseudo-inverse in
# that regime, so DifferentialIKMethod.PSEUDO_INVERSE requires every robot to have at
# least as many controlled DOFs as its own task dimension.
# ---------------------------------------------------------------------------


@wp.kernel
def _build_jjt_plus_damping_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, world coords, canonical axis order
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    damping: wp.array[wp.float32],  # (robot_count,) DLS damping λ, added as λ² on the diagonal
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    active_axis_of_slot: wp.array2d[wp.int32],  # (robot_count, 6) compact slot -> canonical axis, slot < task_dim
    axis_weight: wp.array[wp.spatial_vector],  # (robot_count,) per-canonical-axis weight, > 0 where active
    # outputs
    jjt_plus_damping: wp.array3d[
        float
    ],  # (robot_count, 6, 6) top-left task_dim x task_dim = J_w @ J_wᵀ + λ² * I; untouched elsewhere
):
    """Build the damped-least-squares normal-equations matrix, ``J_w J_wᵀ + λ²I`` (``J_w = diag(w) @ J``), in compact slot space.

    Only writes the top-left ``task_dim x task_dim`` corner. Row/col
    ``slot`` comes from Jacobian axis ``active_axis_of_slot[slot]``,
    weighted by that axis's own ``axis_weight`` — so active axes can be any
    combination of the 6 canonical ones, not just a leading prefix. ``λ²``
    (unweighted) is added on the diagonal to keep the corner invertible even
    where ``J_w J_wᵀ`` alone is singular or rank-deficient.
    """
    robot_idx, row, col = wp.tid()
    if row >= task_dim[robot_idx] or col >= task_dim[robot_idx]:
        return
    axis_row = active_axis_of_slot[robot_idx, row]
    axis_col = active_axis_of_slot[robot_idx, col]
    weight_row = axis_weight[robot_idx][axis_row]
    weight_col = axis_weight[robot_idx][axis_col]
    robot_dof_count = dof_count[robot_idx]
    total = float(0.0)
    for dof in range(robot_dof_count):
        total += jacobian_tool_world[robot_idx, axis_row, dof] * jacobian_tool_world[robot_idx, axis_col, dof]
    total *= weight_row * weight_col
    if row == col:
        lam = damping[robot_idx]
        total += lam * lam
    jjt_plus_damping[robot_idx, row, col] = total


@wp.kernel
def _qd_from_y_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, world coords, canonical axis order
    y: wp.array[wp.spatial_vector],  # (robot_count,) compact slot space, solves (J_w J_wᵀ + λ²I) y = pose_error_active
    bandwidth: wp.array[wp.float32],  # (total_controlled_dofs,) output scale gain
    robot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> column within that robot's Jacobian
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    active_axis_of_slot: wp.array2d[wp.int32],  # (robot_count, 6) compact slot -> canonical axis, slot < task_dim
    axis_weight: wp.array[wp.spatial_vector],  # (robot_count,) per-canonical-axis weight, > 0 where active
    # outputs
    joint_qd_target: wp.array[wp.float32],  # (total_controlled_dofs,) compact = bandwidth * J_wᵀ @ y
):
    """Finish the damped-least-squares solve, ``q̇_target = bandwidth · J_wᵀy`` (``J_w = diag(w) @ J``), into the compact per-DOF layout.

    Row ``slot`` of ``J_wᵀ`` is gathered from Jacobian axis
    ``active_axis_of_slot[slot]``, weighted by that axis's ``axis_weight``.
    The dot product with ``y`` is summed only over ``slot < task_dim``:
    every producer of ``y`` (``_gather_task_error_kernel`` for
    ``DifferentialIKMethod.TRANSPOSE``, ``_apply_spatial_matrix_kernel`` for every
    matrix-inverting method) does leave ``y``'s slots beyond ``task_dim``
    exactly zero, but this kernel does not rely on that -- it stops at
    ``task_dim`` itself, so a future solver path that forgot to zero-pad
    would still be summed correctly here, not silently corrupted.
    """
    dof = wp.tid()
    robot = robot_of_dof[dof]
    slot = slot_of_dof[dof]
    dim = task_dim[robot]

    jacobian_column = wp.spatial_vector()
    for task_slot in range(6):
        if task_slot < dim:
            axis = active_axis_of_slot[robot, task_slot]
            jacobian_column[task_slot] = axis_weight[robot][axis] * jacobian_tool_world[robot, axis, slot]
        else:
            jacobian_column[task_slot] = 0.0

    joint_qd_target[dof] = bandwidth[dof] * wp.dot(jacobian_column, y[robot])


# ---------------------------------------------------------------------------
# Adaptive damping (DifferentialIKMethod.ADAPTIVE_DAMPING): λ is computed each step from
# the smallest eigenvalue of the (undamped) JJᵀ, instead of being a fixed
# input, so damping stays near ``adaptive_damping_min`` away from a
# singularity and ramps up toward ``adaptive_damping_max`` only as the robot
# approaches one. Feeds into the same _build_jjt_plus_damping_kernel used by
# every other method.
# ---------------------------------------------------------------------------


@wp.kernel
def _smallest_eigenvalue_spd6_kernel(
    matrix: wp.array3d[float],  # (robot_count, 6, 6) symmetric matrix, e.g. undamped JJᵀ
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes, 1-6
    # outputs
    smallest_eigenvalue: wp.array[wp.float32],  # (robot_count,) smallest real eigenvalue, clamped to >= 0
):
    """Smallest eigenvalue among a symmetric 6x6 matrix's ``task_dim`` real ones, skipping padding.

    For ``matrix = JJᵀ``, this is ``sigma_min²`` — zero exactly at a
    kinematic singularity. The ``6 - task_dim`` padding entries outside the
    real top-left corner are exactly zero, and a PSD matrix's eigenvalues
    are never negative, so sorting those out from the smallest end recovers
    the true smallest eigenvalue. Clamped to non-negative since float32
    error can land a fully degenerate eigenvalue just below zero.
    """
    robot_idx = wp.tid()

    local_matrix = wp.spatial_matrix()
    for row in range(6):
        for col in range(6):
            local_matrix[row, col] = matrix[robot_idx, row, col]

    eigenvalues, _ = symmetric_eigenvalues_qr(local_matrix, _EIGENVALUE_QR_TOL)

    padding_count = 6 - task_dim[robot_idx]
    for _ in range(padding_count):
        smallest_idx = int(0)
        smallest_val = eigenvalues[0]
        for i in range(1, 6):
            if eigenvalues[i] < smallest_val:
                smallest_val = eigenvalues[i]
                smallest_idx = i
        eigenvalues[smallest_idx] = 1.0e30  # excluded from every later pass, including the final reduction below

    smallest = eigenvalues[0]
    for i in range(1, 6):
        smallest = wp.min(smallest, eigenvalues[i])
    smallest_eigenvalue[robot_idx] = wp.max(smallest, 0.0)


@wp.kernel
def _adaptive_damping_kernel(
    smallest_eigenvalue: wp.array[wp.float32],  # (robot_count,) sigma_min² of the undamped JJᵀ
    damping_min: wp.array[wp.float32],  # (robot_count,) λ far from any singularity
    damping_max: wp.array[wp.float32],  # (robot_count,) λ at a full singularity (sigma_min = 0)
    singular_value_threshold: wp.array[wp.float32],  # (robot_count,) sigma_min below which damping starts ramping up
    # outputs
    damping: wp.array[wp.float32],  # (robot_count,) λ to pass into _build_jjt_plus_damping_kernel
):
    """Maciejewski-Klein singularity-robust damping, ``λ²(sigma_min)``.

    ``λ² = λ_min² + (1 - (sigma_min/ε)²) · (λ_max² - λ_min²)``, clamped so
    ``sigma_min ≥ ε`` (comfortably non-singular) gives exactly ``λ_min`` and
    ``sigma_min = 0`` (fully singular) gives exactly ``λ_max``.
    """
    robot_idx = wp.tid()
    sigma_min = wp.sqrt(smallest_eigenvalue[robot_idx])
    ratio = wp.min(sigma_min / singular_value_threshold[robot_idx], 1.0)
    lam_min = damping_min[robot_idx]
    lam_max = damping_max[robot_idx]
    lam_sq = lam_min * lam_min + (1.0 - ratio * ratio) * (lam_max * lam_max - lam_min * lam_min)
    damping[robot_idx] = wp.sqrt(lam_sq)


@wp.kernel(enable_backward=False)
def _truncated_pinv_matrix_kernel(
    matrix: wp.array3d[float],  # (robot_count, 6, 6) undamped JJᵀ
    singular_value_threshold: wp.array[wp.float32],  # (robot_count,) sigma below which a direction is dropped
    # outputs
    pinv_matrix: wp.array3d[float],  # (robot_count, 6, 6) = U diag(g(sigma_i)) Uᵀ, g(s) = 1/s^2 if s > threshold else 0
):
    """Truncated-SVD pseudo-inverse of ``JJᵀ``, filtered per singular value rather than damped as a whole.

    ``matrix`` is ``JJᵀ`` itself; its eigenvalues are ``sigma_i²``, the
    squared singular values of ``J``, so inverting it exactly takes
    ``1/sigma_i²`` per direction. Each of the (at most 6) task-space
    directions is either inverted exactly or dropped entirely (``0``),
    depending on whether its own singular value clears
    ``singular_value_threshold`` — unlike ``_build_jjt_plus_damping_kernel``'s
    Tikhonov damping, which shifts every direction by the same ``λ²`` and
    never truncates any of them, this has no smooth transition between the
    two regimes.

    Backward disabled: its analytic gradient (via ``symmetric_eigenvalues_qr``
    plus an in-kernel selection/threshold loop over the eigenpairs) disagrees
    with a finite difference by orders of magnitude, not just numerical
    noise. Any caller under an active tape gets an exact-zero gradient
    contribution from this kernel instead. Root cause not yet isolated;
    deferred to a follow-up.
    """
    robot_idx = wp.tid()

    local_matrix = wp.spatial_matrix()
    for row in range(6):
        for col in range(6):
            local_matrix[row, col] = matrix[robot_idx, row, col]

    eigenvalues, eigenvectors_by_row = symmetric_eigenvalues_qr(local_matrix, _EIGENVALUE_QR_TOL)
    threshold = singular_value_threshold[robot_idx]

    for row in range(6):
        for col in range(6):
            total = float(0.0)
            for i in range(6):
                eigenvalue = wp.max(eigenvalues[i], 0.0)
                sigma = wp.sqrt(eigenvalue)
                if sigma > threshold:
                    total += eigenvectors_by_row[i, row] * eigenvectors_by_row[i, col] / eigenvalue
            pinv_matrix[robot_idx, row, col] = total


# ---------------------------------------------------------------------------
# Null-space secondary objectives.
#
# The null-space projector and its damped pseudo-inverse-transpose
# ``(JJᵀ + λ_null²I)⁻¹ @ J`` reuse kernels shared with other controller
# families (``_invert_spd_block_kernel``, ``_task_matrix_times_jacobian_kernel``,
# ``_null_space_projector_kernel`` in ``controllers/impl/_common.py``), not
# dedicated kernels here. ``λ_null`` is independent of the primary task's DLS
# damping; it keeps ``JJᵀ + λ_null²I`` SPD even for a rank-deficient Jacobian
# (e.g. a redundant low-DOF arm with a lower-than-6D task), at the cost of a
# ``J @ N`` residual of order ``λ_null²`` instead of exactly zero.
#
# Those shared kernels expect the Jacobian in canonical axis order and know
# nothing about ``axis_weight``, so ``_gather_jacobian_by_axis_kernel``/
# ``_scatter_pinv_transpose_by_axis_kernel`` below convert to and from
# compact slot order around them.
#
# The kernels below produce a joint-space bias, projected through that
# projector so it never disturbs the primary task; joint-limit avoidance and
# posture control may be combined (added) before projecting.
# ---------------------------------------------------------------------------


@wp.kernel
def _gather_jacobian_by_axis_kernel(
    jacobian_tool_world: wp.array3d[float],  # (robot_count, 6, max_dofs) canonical axis order
    active_axis_of_slot: wp.array2d[wp.int32],  # (robot_count, 6) compact slot -> canonical axis, slot < task_dim
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    # outputs
    jacobian_active: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) compact slot order; rows >= task_dim untouched (zero)
):
    """Gather a Jacobian's active-axis rows into compact slot order, for ``_task_matrix_times_jacobian_kernel``."""
    robot_idx, slot, col = wp.tid()
    if slot >= task_dim[robot_idx]:
        return
    axis = active_axis_of_slot[robot_idx, slot]
    jacobian_active[robot_idx, slot, col] = jacobian_tool_world[robot_idx, axis, col]


@wp.kernel
def _scatter_pinv_transpose_by_axis_kernel(
    pinv_transpose_slot: wp.array3d[float],  # (robot_count, 6, max_dofs) compact slot order
    active_axis_of_slot: wp.array2d[wp.int32],  # (robot_count, 6) compact slot -> canonical axis, slot < task_dim
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    pinv_transpose_axis: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) canonical axis order; rows for inactive axes untouched (zero)
):
    """Scatter a compact-slot-order pinv-transpose back to canonical axis order, for ``_null_space_projector_kernel``."""
    robot_idx, slot, col = wp.tid()
    if slot >= task_dim[robot_idx] or col >= dof_count[robot_idx]:
        return
    axis = active_axis_of_slot[robot_idx, slot]
    pinv_transpose_axis[robot_idx, axis, col] = pinv_transpose_slot[robot_idx, slot, col]


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
