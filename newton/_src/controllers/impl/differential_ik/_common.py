# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for the differential-kinematics controllers.

A controlled robot's task-space quantity is a fixed-size ``wp.spatial_vector``
or a fixed ``6x6`` block, always, regardless of how each of the 6 canonical
axes is weighted — see the section comment above ``_gather_task_error_kernel``
for how per-axis weighting (``axis_weight``) is applied and why a zero-weighted
axis is excluded structurally rather than multiplied by zero.

The inverse-Jacobian solve is isolated to its own group of kernels (see the
section comment above ``_svd_one_sided_jacobi``) so that a solver can be
selected per instance via :class:`DifferentialIKMethod` without touching
pose-error, null-space, or integration code. Every method but ``TRANSPOSE``
(which needs no matrix inversion at all) shares one SVD of the task
Jacobian -- see that section comment for why, and how each method's own
per-singular-direction gain differs.
"""

from __future__ import annotations

import enum
from typing import Any

import warp as wp

# Tolerance/sweep cap passed to _svd_one_sided_jacobi: a sweep is converged
# once every column pair's off-diagonal-to-diagonal ratio drops below this;
# max_sweeps is a safety cap, not a tuning knob -- see _svd_one_sided_jacobi's
# own docstring for why convergence is expected well before it for a matrix
# this small.
_JACOBI_SVD_TOL = wp.constant(wp.float32(1.0e-6))
_JACOBI_SVD_MAX_SWEEPS = 30


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
def _gather_and_weight_jacobian_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, world coords, canonical axis order
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    active_axis_of_slot: wp.array2d[wp.int32],  # (robot_count, 6) compact slot -> canonical axis, slot < task_dim
    axis_weight: wp.array[wp.spatial_vector],  # (robot_count,) per-canonical-axis weight, > 0 where active
    # outputs
    jacobian_weighted: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) = diag(w) @ J, gathered to compact slot rows; rows >= task_dim exactly zero
):
    """Gather+weight a Jacobian's active-axis rows into compact slot order, ``J_w = diag(w) @ J``.

    Feeds :func:`_svd_one_sided_jacobi`, which requires an all-zero row
    beyond a robot's own ``task_dim`` (unlike its column padding, which is
    bounded by an explicit ``n_columns`` instead) — this kernel writes that
    zero explicitly rather than relying on ``jacobian_weighted`` having been
    pre-zeroed, since it is reused, unzeroed, across every step.
    """
    robot_idx, row, col = wp.tid()
    if row >= task_dim[robot_idx]:
        jacobian_weighted[robot_idx, row, col] = 0.0
        return
    axis = active_axis_of_slot[robot_idx, row]
    jacobian_weighted[robot_idx, row, col] = axis_weight[robot_idx][axis] * jacobian_tool_world[robot_idx, axis, col]


@wp.func
def _damped_pinv_singular_value(sigma: float, lam: float):
    """The i-th singular value of a damped pseudo-inverse, ``sigma / (sigma² + λ²)``.

    ``S`` holds a matrix's own singular values; this holds the
    corresponding singular values of its (damped) pseudo-inverse -- with
    ``λ = 0`` this would be exactly ``1/sigma`` (a zero-guard keeps it
    finite at ``sigma = 0``, since the eigenvalue analog of that pivot floor
    is ``_invert_spd_block_kernel``'s Cholesky pivot floor elsewhere in this
    codebase); damping shifts it smoothly toward ``0`` as ``sigma`` shrinks
    instead of blowing up near a singularity. The same formula, with a
    different ``λ`` each time, is every matrix-inverting
    :class:`DifferentialIKMethod` but ``TRUNCATED_SVD``:

    - ``DAMPED_LEAST_SQUARES``: ``λ`` is the caller's fixed ``damping``.
    - ``PSEUDO_INVERSE``: ``λ = 0`` -- the zero-guard is what keeps a
      near-singular direction finite instead of raising it to infinity.
    - ``ADAPTIVE_DAMPING``: ``λ`` is recomputed every step from the
      smallest singular value itself (see ``_adaptive_damping_kernel``).
    """
    denominator = sigma * sigma + lam * lam
    return wp.where(denominator > 1.0e-12, sigma / denominator, 0.0)


@wp.func
def _truncated_pinv_singular_value(sigma: float, threshold: float):
    """The i-th singular value of a truncated pseudo-inverse, for :class:`DifferentialIKMethod.TRUNCATED_SVD` only.

    Exactly ``1/sigma`` if ``sigma`` clears ``threshold``, otherwise exactly
    ``0`` -- a hard cutoff, unlike :func:`_damped_pinv_singular_value`'s
    smooth transition toward ``0``.
    """
    return wp.where(sigma > threshold, 1.0 / sigma, 0.0)


@wp.func
def _projected_error(u: wp.array3d[float], robot_idx: int, i: int, error: wp.spatial_vector):
    """``(Uᵀe)_i``, the task-space error projected onto the i-th left singular direction of ``J_w``."""
    total = float(0.0)
    for row in range(6):
        total += u[robot_idx, row, i] * error[row]
    return total


@wp.kernel
def _qd_in_singular_basis_damped_kernel(
    u: wp.array3d[float],  # (robot_count, 6, 6) from _svd_one_sided_jacobi_kernel on J_w
    s: wp.array2d[float],  # (robot_count, max_dofs) singular values of J_w, sorted descending
    pose_error_active: wp.array[wp.spatial_vector],  # (robot_count,) e_w = diag(w) @ e, compact slot order
    damping: wp.array[wp.float32],  # (robot_count,) DLS damping λ; 0 for DifferentialIKMethod.PSEUDO_INVERSE
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    qd_in_singular_basis: wp.array2d[
        float
    ],  # (robot_count, max_dofs) = pinv_singular_value_i * (Uᵀe_w)_i; 0 beyond min(task_dim, dof_count)
):
    """Joint velocity expressed in ``J_w``'s own singular basis, one damped pseudo-inverse direction at a time.

    Feeds :func:`_qd_from_singular_basis_kernel` to finish ``q̇ = bandwidth
    · V @ qd_in_singular_basis`` (the rotation from that basis back into
    actual joint coordinates), which is algebraically ``bandwidth ·
    Jᵀ(JJᵀ + λ²I)⁻¹e_w`` (the same damped-least-squares law as
    ``DifferentialIKMethod.DAMPED_LEAST_SQUARES``'s docstring), derived from
    ``J_w``'s own SVD rather than an explicit ``J_w J_wᵀ`` inverse -- see
    the section comment above ``_svd_one_sided_jacobi`` for why. At most
    ``min(task_dim, dof_count)`` of the ``max_dofs`` singular directions are
    genuine (the rest are padding, either from ``J_w``'s own zero-column
    padding or from a structurally rank-deficient ``J_w``); every other
    entry is left exactly zero, so a caller summing over all ``max_dofs``
    columns adds nothing from them.
    """
    robot_idx, i = wp.tid()
    genuine_count = wp.min(task_dim[robot_idx], dof_count[robot_idx])
    if i >= genuine_count:
        qd_in_singular_basis[robot_idx, i] = 0.0
        return
    pinv_singular_value = _damped_pinv_singular_value(s[robot_idx, i], damping[robot_idx])
    qd_in_singular_basis[robot_idx, i] = pinv_singular_value * _projected_error(
        u, robot_idx, i, pose_error_active[robot_idx]
    )


@wp.kernel
def _qd_in_singular_basis_truncated_kernel(
    u: wp.array3d[float],  # (robot_count, 6, 6) from _svd_one_sided_jacobi_kernel on J_w
    s: wp.array2d[float],  # (robot_count, max_dofs) singular values of J_w, sorted descending
    pose_error_active: wp.array[wp.spatial_vector],  # (robot_count,) e_w = diag(w) @ e, compact slot order
    singular_value_threshold: wp.array[wp.float32],  # (robot_count,) sigma below which a direction is dropped
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    qd_in_singular_basis: wp.array2d[
        float
    ],  # (robot_count, max_dofs) = pinv_singular_value_i * (Uᵀe_w)_i; 0 beyond min(task_dim, dof_count)
):
    """Joint velocity expressed in ``J_w``'s own singular basis, for :class:`DifferentialIKMethod.TRUNCATED_SVD`.

    Same role as :func:`_qd_in_singular_basis_damped_kernel`, but via
    :func:`_truncated_pinv_singular_value`'s hard cutoff instead of a smooth
    damped one. See that kernel for the genuine-vs-padding singular
    direction count.
    """
    robot_idx, i = wp.tid()
    genuine_count = wp.min(task_dim[robot_idx], dof_count[robot_idx])
    if i >= genuine_count:
        qd_in_singular_basis[robot_idx, i] = 0.0
        return
    pinv_singular_value = _truncated_pinv_singular_value(s[robot_idx, i], singular_value_threshold[robot_idx])
    qd_in_singular_basis[robot_idx, i] = pinv_singular_value * _projected_error(
        u, robot_idx, i, pose_error_active[robot_idx]
    )


@wp.kernel
def _pinv_singular_value_damped_kernel(
    s: wp.array2d[float],  # (robot_count, max_dofs) singular values of J, sorted descending
    damping: wp.array[wp.float32],  # (robot_count,) damping λ
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    pinv_singular_value: wp.array2d[
        float
    ],  # (robot_count, max_dofs) per-direction pseudo-inverse singular value; 0 beyond min(task_dim, dof_count)
):
    """Bare per-direction pseudo-inverse singular value (:func:`_damped_pinv_singular_value`), for the null-space projector.

    Unlike :func:`_qd_in_singular_basis_damped_kernel`, this does not also
    fold in ``Uᵀe`` -- :func:`_svd_reconstruct_scaled_kernel` combines it
    with ``U``/``V`` directly into a matrix, rather than applying it to a
    single task-space vector.
    """
    robot_idx, i = wp.tid()
    genuine_count = wp.min(task_dim[robot_idx], dof_count[robot_idx])
    if i >= genuine_count:
        pinv_singular_value[robot_idx, i] = 0.0
        return
    pinv_singular_value[robot_idx, i] = _damped_pinv_singular_value(s[robot_idx, i], damping[robot_idx])


@wp.kernel
def _svd_reconstruct_scaled_kernel(
    u: wp.array3d[float],  # (robot_count, 6, 6)
    pinv_singular_value: wp.array2d[
        float
    ],  # (robot_count, max_dofs) per-direction pseudo-inverse singular value, e.g. from _pinv_singular_value_damped_kernel
    v: wp.array3d[float],  # (robot_count, max_dofs, max_dofs)
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    reconstructed: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) = U @ diag(pinv_singular_value) @ Vᵀ, i.e. Jᵀ(JJᵀ + λ²I)⁻¹ transposed; 0 beyond dof_count
):
    """Reassemble a pseudo-inverse-transpose, ``U @ diag(pinv_singular_value) @ Vᵀ``, from an SVD and its per-direction pseudo-inverse singular values.

    With ``pinv_singular_value`` from :func:`_pinv_singular_value_damped_kernel`,
    this is ``(JJᵀ + λ²I)⁻¹ @ J`` -- the damped pseudo-inverse-transpose
    :func:`_null_space_projector_kernel` needs, computed directly from
    ``J``'s own SVD rather than by inverting ``JJᵀ + λ²I``.
    """
    robot_idx, row, col = wp.tid()
    if col >= dof_count[robot_idx]:
        reconstructed[robot_idx, row, col] = 0.0
        return
    total = float(0.0)
    for i in range(wp.min(dof_count[robot_idx], 6)):
        total += u[robot_idx, row, i] * pinv_singular_value[robot_idx, i] * v[robot_idx, col, i]
    reconstructed[robot_idx, row, col] = total


@wp.kernel
def _qd_from_singular_basis_kernel(
    v: wp.array3d[float],  # (robot_count, max_dofs, max_dofs) from _svd_one_sided_jacobi_kernel on J_w
    qd_in_singular_basis: wp.array2d[
        float
    ],  # (robot_count, max_dofs) from _qd_in_singular_basis_damped_kernel/_qd_in_singular_basis_truncated_kernel
    bandwidth: wp.array[wp.float32],  # (total_controlled_dofs,) output scale gain
    robot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> column within that robot's Jacobian
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    joint_qd_target: wp.array[wp.float32],  # (total_controlled_dofs,) compact = bandwidth * V @ qd_in_singular_basis
):
    """Rotate ``qd_in_singular_basis`` out of ``J_w``'s singular basis into actual joint coordinates.

    ``q̇_target = bandwidth · V @ qd_in_singular_basis``: ``V``'s columns
    are ``J_w``'s right singular vectors, so this is the change of basis
    from "one coefficient per singular direction" back to "one value per
    controlled DOF", the SVD-based solve's counterpart to
    :func:`_qd_from_y_kernel`'s ``Jᵀ @ y``.
    """
    dof = wp.tid()
    robot = robot_of_dof[dof]
    slot = slot_of_dof[dof]
    total = float(0.0)
    for i in range(dof_count[robot]):
        total += v[robot, slot, i] * qd_in_singular_basis[robot, i]
    joint_qd_target[dof] = bandwidth[dof] * total


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
    ``_gather_task_error_kernel`` (this kernel's only caller, via
    ``DifferentialIKMethod.TRANSPOSE``) does leave ``y``'s slots beyond
    ``task_dim`` exactly zero, but this kernel does not rely on that -- it
    stops at ``task_dim`` itself, so a future caller that forgot to
    zero-pad would still be summed correctly here, not silently corrupted.
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
# J_w's own smallest singular value, read directly from the same SVD every
# other matrix-inverting method also uses -- see the section comment above
# _svd_one_sided_jacobi -- instead of being a fixed input, so damping stays
# near ``adaptive_damping_min`` away from a singularity and ramps up toward
# ``adaptive_damping_max`` only as the robot approaches one.
# ---------------------------------------------------------------------------


@wp.kernel
def _adaptive_damping_kernel(
    s: wp.array2d[float],  # (robot_count, max_dofs) singular values of J_w, sorted descending
    task_dim: wp.array[wp.int32],  # (robot_count,) number of active axes
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    damping_min: wp.array[wp.float32],  # (robot_count,) λ far from any singularity
    damping_max: wp.array[wp.float32],  # (robot_count,) λ at a full singularity (sigma_min = 0)
    singular_value_threshold: wp.array[wp.float32],  # (robot_count,) sigma_min below which damping starts ramping up
    # outputs
    damping: wp.array[wp.float32],  # (robot_count,) λ to pass into the damped-pseudo-inverse kernels above
):
    """Maciejewski-Klein singularity-robust damping, ``λ²(sigma_min)``.

    ``λ² = λ_min² + (1 - (sigma_min/ε)²) · (λ_max² - λ_min²)``, clamped so
    ``sigma_min ≥ ε`` (comfortably non-singular) gives exactly ``λ_min`` and
    ``sigma_min = 0`` (fully singular) gives exactly ``λ_max``.

    ``sigma_min`` is ``J_w``'s own smallest singular value over its
    ``task_dim`` task directions -- deliberately indexed by ``task_dim``,
    not ``min(task_dim, dof_count)``: a robot with fewer controlled DOFs
    than its own task dimension (``dof_count < task_dim``) cannot reach
    every task direction at all, which is itself a (structural, permanent)
    singularity that must read as ``sigma_min = 0``. That branch returns
    before ever touching ``s``: its own column count is ``max_dofs`` (the
    largest ``dof_count`` across the whole batch, not padded up to 6), so
    ``task_dim - 1`` is only a valid index into it once ``dof_count >=
    task_dim`` is confirmed -- reading it unconditionally would be an
    out-of-bounds access whenever some robot's ``dof_count`` (and so the
    whole batch's ``max_dofs``) is smaller than its own ``task_dim``.
    """
    robot_idx = wp.tid()
    if dof_count[robot_idx] < task_dim[robot_idx]:
        sigma_min = 0.0
    else:
        sigma_min = s[robot_idx, task_dim[robot_idx] - 1]
    ratio = wp.min(sigma_min / singular_value_threshold[robot_idx], 1.0)
    lam_min = damping_min[robot_idx]
    lam_max = damping_max[robot_idx]
    lam_sq = lam_min * lam_min + (1.0 - ratio * ratio) * (lam_max * lam_max - lam_min * lam_min)
    damping[robot_idx] = wp.sqrt(lam_sq)


# ---------------------------------------------------------------------------
# Null-space secondary objectives.
#
# The null-space projector, ``_null_space_projector_kernel`` in
# ``controllers/impl/_common.py`` (shared with other controller families),
# needs the damped pseudo-inverse-transpose ``(JJᵀ + λ_null²I)⁻¹ @ J``. Built
# the same SVD-of-``J``-only way as the primary task solve (see the section
# comment above ``_svd_one_sided_jacobi``): SVD the (unweighted) Jacobian,
# then ``_pinv_singular_value_damped_kernel`` + ``_svd_reconstruct_scaled_kernel``
# reassemble it, instead of inverting ``JJᵀ + λ_null²I`` directly. ``λ_null`` is
# independent of the primary task's DLS damping; it keeps ``JJᵀ + λ_null²I``
# SPD even for a rank-deficient Jacobian (e.g. a redundant low-DOF arm with a
# lower-than-6D task), at the cost of a ``J @ N`` residual of order
# ``λ_null²`` instead of exactly zero.
#
# ``_null_space_projector_kernel`` expects the Jacobian in canonical axis
# order and knows nothing about ``axis_weight``, so ``_gather_jacobian_by_axis_kernel``/
# ``_scatter_pinv_transpose_by_axis_kernel`` below convert to and from
# compact slot order around it -- the null-space projector's own
# regularization stays deliberately unweighted (every axis weight 1), unlike
# the primary task solve's ``J_w``.
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
    """Gather a Jacobian's active-axis rows into compact slot order, for the null-space projector's own SVD."""
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


# ---------------------------------------------------------------------------
# One-sided Jacobi SVD -- the shared basis for every matrix-inverting
# DifferentialIKMethod (everything but TRANSPOSE, which needs no inversion).
#
# Deriving a pseudo-inverse from an eigendecomposition of JJᵀ (e.g. via
# Warp's ``symmetric_eigenvalues_qr``) squares J's condition number before
# any float32 arithmetic happens, so a small-but-retained
# singular value of J can come back badly corrupted (confirmed: relative
# error in the thousands on a Jacobian with singular values spanning
# [1, 1e-4]). ``_svd_one_sided_jacobi`` below operates on J's own columns
# directly and never forms JJᵀ, avoiding that squaring -- validated in
# isolation by ``test_svd_one_sided_jacobi_*`` in the test suite.
#
# The primary task solve (``_gather_and_weight_jacobian_kernel`` +
# ``_svd_one_sided_jacobi_kernel`` on ``J_w = diag(w) @ J``) and the
# null-space projector's own solve (``_gather_jacobian_by_axis_kernel`` +
# ``_svd_one_sided_jacobi_kernel`` on the unweighted ``J``) each run one SVD
# per step; every method-specific pseudo-inverse singular value (damped
# least squares, the zero-damping Moore-Penrose pseudo-inverse, adaptive
# damping, truncated SVD) is then just a different per-singular-direction
# transform of that same ``U``/``S``/``V`` -- see
# ``_damped_pinv_singular_value``, ``_truncated_pinv_singular_value``, and
# ``_adaptive_damping_kernel`` (which reads ``S`` directly).
# ---------------------------------------------------------------------------


@wp.func
def _svd_one_sided_jacobi(A: Any, n_columns: int, tol: Any, max_sweeps: int):
    """One-sided (Hestenes) Jacobi SVD of a small (possibly non-square) matrix, ``A = U @ diag(S) @ Vᵀ``.

    Correct and simple, not the fastest: intended for small matrices (at
    most a handful of rows/columns), where its guaranteed convergence
    (Forsythe & Henrici 1960 for cyclic Jacobi; Hestenes 1958 for this
    one-sided SVD form) and its column-of-``A``-only formulation matter more
    than raw throughput. It never explicitly forms ``AᵀA``, unlike deriving
    an SVD from a symmetric eigensolver on ``AᵀA``, which squares ``A``'s
    condition number before any arithmetic happens (Demmel & Veselić 1992:
    Jacobi achieves higher relative accuracy on small singular values than
    that approach for exactly this reason). Not validated for large
    matrices, where the O(sweeps · dim²) sweep cost and this convergence
    behavior may not be worth it relative to a bidiagonalization-based SVD.

    ``A``'s columns may be padded beyond a true problem smaller than ``A``
    itself, with every "true" entry in the leftmost ``n_columns`` columns and
    every other entry exactly zero, matching how a Jacobian is already
    padded elsewhere in this module -- ``n_columns`` bounds every loop to those
    columns, both so a batch of differently-sized problems can share one
    padded buffer/kernel, and to skip sweeping over columns already known
    to contribute nothing. Padding is not a safety requirement here the way
    it is for ``symmetric_eigenvalues_qr``: an all-zero padding column has
    zero norm, which this function's own convergence check already treats
    as trivially converged (see ``denominator > zero`` below), so it would
    still terminate cleanly with ``n_columns`` left at ``A``'s full column count.

    Args:
        A: Matrix to decompose, ``m`` rows by ``n`` columns, columns
            optionally padded (see above). Rows are never padded/masked --
            an all-zero row is harmless to this algorithm, so callers with
            row padding of their own (e.g. a padded task dimension) don't
            need a separate parameter to mask it.
        n_columns: Number of true, unpadded leftmost columns of ``A``. At most
            ``m`` of them can have a nonzero singular value (rank ≤ ``m``
            whenever ``n_columns > m``); ``U``/``S`` beyond index ``m`` are left
            at their identity/zero initialization rather than written.
        tol: A sweep is converged once every column pair's
            off-diagonal-to-diagonal ratio drops below this.
        max_sweeps: Upper bound on the number of full sweeps over all
            column pairs -- a safety cap, not a tuning knob; convergence is
            expected well before this for any well-scaled small matrix.

    Returns:
        ``(U, S, V)`` such that ``A = U @ diag(S) @ Vᵀ`` over the leftmost
        ``n_columns`` columns, ``U`` (``m x m``)/``V`` (``n x n``) orthonormal
        there, ``S`` (length ``n``) sorted descending over its first
        ``min(n_columns, m)`` entries. A singular value at or below numerical
        zero leaves its own column of ``U`` as the zero vector, since no
        direction is well-defined there. Every row/column at or beyond
        ``n_columns`` (in ``V``) or ``min(n_columns, m)`` (in ``U``/``S``) is left
        untouched, at whatever ``A`` itself (for ``U``) or the identity
        (for ``V``) already had there.
    """
    zero = A.dtype(0.0)
    one = A.dtype(1.0)
    two = A.dtype(2.0)

    # At's rows are A's columns, so a Jacobi rotation of two columns of A is
    # a plain row update here -- no separate column-extraction step needed.
    at = wp.transpose(A)
    row0 = type(A[0])()  # length n (column count of A, row count of At)
    col0 = type(at[0])()  # length m (row count of A, column count of At)
    vt = wp.identity(n=type(row0).length, dtype=A.dtype)

    for _sweep in range(max_sweeps):
        max_off_diagonal_ratio = zero
        for i in range(n_columns - 1):
            for j in range(i + 1, n_columns):
                col_i = at[i]
                col_j = at[j]
                alpha = wp.dot(col_i, col_i)
                beta = wp.dot(col_j, col_j)
                gamma = wp.dot(col_i, col_j)
                denominator = wp.sqrt(alpha * beta)
                ratio = wp.where(denominator > zero, wp.abs(gamma) / denominator, zero)
                max_off_diagonal_ratio = wp.max(max_off_diagonal_ratio, ratio)
                if ratio > tol:
                    # Golub & Van Loan's robust symmetric Jacobi rotation,
                    # applied here to A's columns (via at's rows) and
                    # accumulated into V (via vt's rows) the same way.
                    zeta = (beta - alpha) / (two * gamma)
                    sign_zeta = wp.where(zeta >= zero, one, -one)
                    t = sign_zeta / (wp.abs(zeta) + wp.sqrt(one + zeta * zeta))
                    c = one / wp.sqrt(one + t * t)
                    s = c * t

                    at[i] = c * col_i - s * col_j
                    at[j] = s * col_i + c * col_j

                    v_i = vt[i]
                    v_j = vt[j]
                    vt[i] = c * v_i - s * v_j
                    vt[j] = s * v_i + c * v_j
        if max_off_diagonal_ratio < tol:
            break

    # Candidate norm of every one of the n_columns converged columns of at
    # -- NOT just the first u_dim of them. Convergence only drives
    # off-diagonal correlations to zero; it does not guarantee the largest
    # singular directions land in any particular prefix of the columns, so
    # picking the top u_dim by value requires looking at all of them,
    # whenever n_columns > m (more columns than rows: at most m of them can
    # be genuinely nonzero, but which ones is not known in advance).
    candidate_sigma = type(row0)()
    for i in range(n_columns):
        candidate_sigma[i] = wp.length(at[i])

    s_vec = type(row0)()
    ut = wp.identity(n=type(col0).length, dtype=A.dtype)
    u_dim = wp.min(n_columns, type(col0).length)
    for i in range(u_dim):
        # Repeated argmax selection over the remaining candidates, largest
        # first: also produces s_vec sorted descending, with no separate
        # sort pass needed. at's and vt's rows are swapped into place the
        # same way candidate_sigma is (not just looked up by the winner's
        # original index), so a later round's search always reads
        # candidate_sigma[i]/at[i]/vt[i] as the same still-associated
        # triple -- looking up ``at[largest_idx]`` after only
        # candidate_sigma had been swapped in a previous round would read
        # the wrong row, since position i no longer corresponds to
        # A's original column i once candidate_sigma has been reordered.
        largest_idx = i
        largest_val = candidate_sigma[i]
        for k in range(i + 1, n_columns):
            if candidate_sigma[k] > largest_val:
                largest_val = candidate_sigma[k]
                largest_idx = k
        if largest_idx != i:
            sigma_tmp = candidate_sigma[i]
            candidate_sigma[i] = candidate_sigma[largest_idx]
            candidate_sigma[largest_idx] = sigma_tmp
            at_tmp = at[i]
            at[i] = at[largest_idx]
            at[largest_idx] = at_tmp
            v_tmp = vt[i]
            vt[i] = vt[largest_idx]
            vt[largest_idx] = v_tmp
        sigma = candidate_sigma[i]
        s_vec[i] = sigma
        ut[i] = wp.where(sigma > A.dtype(1.0e-12), at[i] / sigma, ut[i] * zero)

    return wp.transpose(ut), s_vec, wp.transpose(vt)


@wp.kernel
def _svd_one_sided_jacobi_kernel(
    matrix: wp.array[Any],  # (batch_count,) matrix-typed elements, any fixed (m, n) shape
    n_columns: wp.array[wp.int32],  # (batch_count,) number of true, unpadded leftmost columns of each matrix
    tol: wp.float32,
    max_sweeps: wp.int32,
    # outputs
    u: wp.array[Any],  # (batch_count,) m x m
    s: wp.array[Any],  # (batch_count,) length n
    v: wp.array[Any],  # (batch_count,) n x n
):
    """Batched, size-generic instantiation of :func:`_svd_one_sided_jacobi`.

    Generic over ``matrix``'s (and so ``u``/``s``/``v``'s) concrete element
    type -- Warp compiles one specialization per distinct matrix shape
    actually launched with, so this single kernel definition covers every
    size, rather than needing a hand-written kernel per shape.

    ``Any`` here is broader than the real requirement: every array argument
    must hold a matrix-shaped element type (e.g. ``wp.mat33``), not an
    arbitrary Warp dtype -- a vector or scalar would fail to compile inside
    :func:`_svd_one_sided_jacobi` (e.g. at its ``wp.transpose(A)`` call),
    not silently misbehave, but that constraint isn't expressible in the
    type hint itself. Warp's kernel-genericity mechanism only dispatches on
    ``Any``; the more precise ``warp._src.types.Matrix[Scalar, Rows, Cols]``
    both isn't public API and (confirmed by testing) isn't actually wired
    into that dispatch mechanism, so it isn't a usable substitute here.
    """
    idx = wp.tid()
    u_local, s_local, v_local = _svd_one_sided_jacobi(matrix[idx], n_columns[idx], tol, max_sweeps)
    u[idx] = u_local
    s[idx] = s_local
    v[idx] = v_local
