# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels used by more than one controller family.

Every 1-D buffer here is compact — one entry per controlled DOF, robot 0's DOFs
first, then robot 1's — so every kernel is a flat 1-D launch with no padding to
skip. The exception is a padded per-robot matrix (e.g. a mass matrix), which
:func:`~newton.eval_mass_matrix` produces as one square block per articulation:
:func:`_block_matrix_vector_multiply_kernel` stays a flat 1-D launch and indexes
into those blocks, while the gather kernels launch over them directly.

A kernel belongs here, rather than in one controller family's own ``_common.py``,
once a second family needs the identical operation: joint-space PD and
compact-vector accumulation, the block-matrix-vector multiply, the
view-safe port-plumbing helpers, pose error, a tool-point Jacobian shift,
the null-space projector, and a batched small-SPD-matrix inverse (plus two
task-space matrix-vector/matrix-Jacobian products built on it) are all used
by more than one of the joint-space, operational-space, and
differential-kinematics controller families.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ...core.types import Devicelike
from ...math import velocity_at_point

# Cholesky pivots are clamped above this, scaled by the pivot's own
# magnitude, so float32 cancellation noise on a near-singular matrix can't
# drive a pivot negative (which would make the square root below NaN).
_FLOAT32_EPS = wp.constant(wp.float32(np.finfo(np.float32).eps))


@wp.kernel
def _pose_error_kernel(
    current_pose: wp.array[wp.transform],  # (robot_count,) current tool pose
    desired_pose: wp.array[wp.transform],  # (robot_count,) desired tool pose, same frame as current_pose
    # outputs
    pose_error: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) (position error, orientation error), same frame as the inputs: desired minus current
):
    """Task-space pose error, ``(desired_position - current_position, orientation_error)``.

    The position error is a plain vector difference.

    The orientation error is the axis-angle rotation that would carry the
    current orientation to the desired one: rotate the current orientation
    by ``angle`` about ``axis`` and it lands on the desired orientation. It
    shrinks to zero exactly when the two orientations agree, matching the
    position error's "desired minus current" sign so both halves of the 6D
    error can be driven to zero by the same kind of proportional term.

    Derivation: with quaternions written so ``q * p`` composes like Warp's
    ``transform *`` (apply ``p`` first, then ``q``), the rotation that "undoes
    current, then applies desired" is ``quat_error = q_desired * q_current^-1``.
    Its axis-angle form is exactly that carrying rotation. Extracting it
    inlines Warp's own ``quat_to_axis_angle`` formula
    (``newton/native/quat.h``) rather than calling it directly, because that
    builtin divides by the quaternion's vector-part norm with no guard — it
    returns NaN once the two orientations are close enough that the norm
    underflows, which is exactly the common steady-state case for a pose
    tracker. The small-angle branch below is quat_error's first-order Taylor
    expansion instead: for a unit quaternion near identity,
    ``quat_error ~= (1, half_angle * axis)``, so ``2 * vector_part ~= angle *
    axis`` directly, with no division at all.
    """
    robot_idx = wp.tid()

    current = current_pose[robot_idx]
    position_error = wp.transform_get_translation(desired_pose[robot_idx]) - wp.transform_get_translation(current)

    quat_current = wp.transform_get_rotation(current)
    quat_desired = wp.transform_get_rotation(desired_pose[robot_idx])
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


@wp.kernel
def _null_space_projector_kernel(
    jacobian_tool: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are per-DOF twists about the tool point, in the caller's task frame
    jacobian_pinv_transpose: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) either pseudo-inverse-transpose variant; zero beyond dof_count
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    null_space_projector: wp.array3d[
        float
    ],  # (robot_count, max_dofs, max_dofs) = I - J^T @ jacobian_pinv_transpose; untouched beyond dof_count
):
    """The null-space projector, ``N = I - J^T @ jacobian_pinv_transpose``.

    Frame-agnostic: ``jacobian_tool`` and ``jacobian_pinv_transpose`` just
    need to be expressed in the same frame as each other, whatever that is
    (e.g. world for differential IK, the operational frame for hybrid
    force/motion control).

    A joint torque built as ``N @ M @ a``, for any joint acceleration ``a``
    and the joint-space mass matrix ``M``, produces zero task-space
    acceleration — but only when ``jacobian_pinv_transpose`` is the
    dynamically-consistent variant: ``J @ M^-1 @ N == 0`` in that case, and
    generally nonzero for the Moore-Penrose variant, since that one ignores
    the robot's inertia. A purely kinematic caller with no mass matrix at
    all (e.g. differential kinematics) always uses the Moore-Penrose
    variant, for which ``J @ N == 0`` holds exactly only when
    ``jacobian_pinv_transpose`` is undamped; a damped variant leaves a
    residual of the same order as its own damping.
    """
    robot_idx, row, col = wp.tid()
    robot_dof_count = dof_count[robot_idx]
    if row >= robot_dof_count or col >= robot_dof_count:
        return

    identity_entry = float(0.0)
    if row == col:
        identity_entry = 1.0

    total = float(0.0)
    for k in range(6):
        total += jacobian_tool[robot_idx, k, row] * jacobian_pinv_transpose[robot_idx, k, col]
    null_space_projector[robot_idx, row, col] = identity_entry - total


@wp.kernel(enable_backward=False)
def _invert_spd_block_kernel(
    spd_matrix: wp.array3d[float],  # (block_count, max_dim, max_dim) symmetric positive-definite matrix per block
    block_dim: wp.array[wp.int32],  # (block_count,) size of the used top-left submatrix of each block
    # scratch, preallocated by the caller (not valid on entry; written and then read within this kernel)
    cholesky_factor: wp.array3d[
        float
    ],  # (block_count, max_dim, max_dim) lower-triangular L such that spd_matrix = L L^T
    # outputs
    spd_matrix_inv: wp.array3d[
        float
    ],  # (block_count, max_dim, max_dim) inverse of the top-left block_dim x block_dim submatrix; untouched elsewhere
):
    """Explicit inverse of a batch of small SPD matrices, via Cholesky factorization.

    Column c of the inverse solves ``spd_matrix @ x = e_c`` (e_c the c'th
    standard basis vector), found by forward-substituting ``L y = e_c`` and
    then back-substituting ``L^T x = y``. No dense-inverse routine (cofactor
    expansion, Gauss-Jordan) is used — this is the numerically standard way to
    invert a small SPD matrix, and the same recipe
    ``newton/_src/actuators/response_oracle.py`` uses for the same reason.

    Backward disabled: this kernel's forward/back-substitution loops read
    values written earlier in the same launch (an intra-kernel recurrence),
    a pattern Warp's generic adjoint generation does not differentiate
    correctly -- gradients through it are silently wrong, not merely
    unsupported. Any caller under an active tape gets an exact-zero gradient
    contribution from this kernel instead. Fixing this needs a hand-written
    adjoint or an algorithm restructured to avoid same-launch recurrence;
    deferred to a follow-up.
    """
    block_idx = wp.tid()
    block_size = block_dim[block_idx]

    # Cholesky factorization: spd_matrix == cholesky_factor @ cholesky_factor^T.
    for col in range(block_size):
        diagonal_term = spd_matrix[block_idx, col, col]
        for prior_col in range(col):
            diagonal_term -= cholesky_factor[block_idx, col, prior_col] * cholesky_factor[block_idx, col, prior_col]
        diagonal_term = wp.max(diagonal_term, _FLOAT32_EPS * wp.max(wp.abs(spd_matrix[block_idx, col, col]), 1.0))
        diagonal_value = wp.sqrt(diagonal_term)
        cholesky_factor[block_idx, col, col] = diagonal_value
        for row in range(col + 1, block_size):
            off_diagonal_term = spd_matrix[block_idx, row, col]
            for prior_col in range(col):
                off_diagonal_term -= (
                    cholesky_factor[block_idx, row, prior_col] * cholesky_factor[block_idx, col, prior_col]
                )
            cholesky_factor[block_idx, row, col] = off_diagonal_term / diagonal_value

    # Solve spd_matrix @ x = e_column for every column, writing x into that column of the inverse.
    for column in range(block_size):
        # Forward substitution: cholesky_factor @ y = e_column.
        for row in range(block_size):
            right_hand_side = float(0.0)
            if row == column:
                right_hand_side = 1.0
            for prior_row in range(row):
                right_hand_side -= (
                    cholesky_factor[block_idx, row, prior_row] * spd_matrix_inv[block_idx, prior_row, column]
                )
            spd_matrix_inv[block_idx, row, column] = right_hand_side / cholesky_factor[block_idx, row, row]
        # Back substitution: cholesky_factor^T @ x = y, overwriting y with x in place.
        for reverse_row in range(block_size):
            row = block_size - 1 - reverse_row
            right_hand_side = spd_matrix_inv[block_idx, row, column]
            for later_row in range(row + 1, block_size):
                right_hand_side -= (
                    cholesky_factor[block_idx, later_row, row] * spd_matrix_inv[block_idx, later_row, column]
                )
            spd_matrix_inv[block_idx, row, column] = right_hand_side / cholesky_factor[block_idx, row, row]


@wp.kernel
def _apply_spatial_matrix_kernel(
    matrix: wp.array3d[float],  # (robot_count, 6, 6) a 6x6 task-space matrix, from _invert_spd_block_kernel
    vector: wp.array[wp.spatial_vector],  # (robot_count,) a task-space vector
    # outputs
    result: wp.array[wp.spatial_vector],  # (robot_count,) = matrix @ vector
):
    """Multiply a 6x6 task-space matrix by a task-space vector, ``result = matrix @ vector``.

    ``matrix`` is stored as a plain ``(robot_count, 6, 6)`` float array — not
    a ``wp.spatial_matrix`` array — because it typically comes from
    :func:`_invert_spd_block_kernel`, which also produces per-robot square
    matrices of other, larger sizes; a fixed-size ``spatial_matrix`` only
    fits the always-exactly-6x6 case. This kernel loads ``matrix`` into a
    local ``wp.spatial_matrix`` so it can use Warp's built-in matrix-vector
    product rather than a hand-rolled accumulation loop.
    """
    robot_idx = wp.tid()

    local_matrix = wp.spatial_matrix()
    for row in range(6):
        for col in range(6):
            local_matrix[row, col] = matrix[robot_idx, row, col]

    result[robot_idx] = local_matrix * vector[robot_idx]


@wp.kernel
def _task_matrix_times_jacobian_kernel(
    task_matrix: wp.array3d[float],  # (robot_count, 6, 6) symmetric task-space matrix, e.g. a 6x6 inverse
    jacobian_tool: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are per-DOF twists about the tool point, in the caller's task frame
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    result: wp.array3d[float],  # (robot_count, 6, max_dofs) = task_matrix @ jacobian_tool; zero beyond dof_count
):
    """Multiply a 6x6 task-space matrix by a tool-point Jacobian, ``result = task_matrix @ jacobian_tool``.

    Frame-agnostic: ``task_matrix`` and ``jacobian_tool`` just need to be
    expressed in the same frame as each other, whatever that is (e.g. world
    for differential IK, the operational frame for hybrid force/motion
    control).
    """
    robot_idx, row, col = wp.tid()
    if col >= dof_count[robot_idx]:
        return
    total = float(0.0)
    for task_axis in range(6):
        total += task_matrix[robot_idx, row, task_axis] * jacobian_tool[robot_idx, task_axis, col]
    result[robot_idx, row, col] = total


@wp.kernel
def _pd_term_kernel(
    joint_q: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_qd: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_q_des: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_qd_des: wp.array[wp.float32],  # (total_controlled_dofs,)
    stiffness: wp.array[wp.float32],  # (total_controlled_dofs,)
    damping: wp.array[wp.float32],  # (total_controlled_dofs,)
    out: wp.array[wp.float32],  # (total_controlled_dofs,)
):
    dof = wp.tid()
    out[dof] = stiffness[dof] * (joint_q_des[dof] - joint_q[dof]) + damping[dof] * (joint_qd_des[dof] - joint_qd[dof])


@wp.kernel
def _add_term_kernel(
    term: wp.array[wp.float32],  # (total_controlled_dofs,)
    tau: wp.array[wp.float32],  # (total_controlled_dofs,)
):
    dof = wp.tid()
    tau[dof] = tau[dof] + term[dof]


@wp.kernel
def _block_matrix_vector_multiply_kernel(
    block_matrix: wp.array3d[wp.float32],  # (controlled_robot_count, max_controlled_dofs, max_controlled_dofs)
    vec: wp.array[wp.float32],  # (total_controlled_dofs,)
    robot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> row within that robot's block
    dof_offsets: wp.array[wp.int32],  # (controlled_robot_count,) -> first flat DOF of each robot
    controlled_dofs_per_robot: wp.array[wp.int32],  # (controlled_robot_count,)
    out: wp.array[wp.float32],  # (total_controlled_dofs,)
):
    """Multiply a compact per-DOF vector by a padded per-robot square matrix, ``out = block_matrix @ vec``.

    ``block_matrix`` need not be a mass matrix — any per-robot square matrix in
    the same padded ``(controlled_robot_count, max_controlled_dofs,
    max_controlled_dofs)`` layout works, e.g. a null-space projector.
    """
    dof = wp.tid()
    robot = robot_of_dof[dof]
    row = slot_of_dof[dof]
    row_base = dof_offsets[robot]
    acc = float(0.0)
    for col in range(controlled_dofs_per_robot[robot]):
        acc = acc + block_matrix[robot, row, col] * vec[row_base + col]
    out[dof] = acc


@wp.kernel
def _gather_mass_matrix_blocks_kernel(
    model_mass_matrix: wp.array3d[wp.float32],  # (model_robot_count, model_max_dofs, model_max_dofs)
    model_robot_index: wp.array[wp.int32],  # (controlled_robot_count,) -> that robot's index in the model
    articulation_dof_idx_of_padded_dof_idx: wp.array2d[
        wp.int32
    ],  # (controlled_robot_count, max_controlled_dofs) padded_dof_idx -> DOF index within its robot
    controlled_dofs_per_robot: wp.array[wp.int32],  # (controlled_robot_count,)
    out: wp.array3d[wp.float32],  # (controlled_robot_count, max_controlled_dofs, max_controlled_dofs)
):
    robot, padded_row_dof_idx, padded_col_dof_idx = wp.tid()
    if padded_row_dof_idx >= controlled_dofs_per_robot[robot] or padded_col_dof_idx >= controlled_dofs_per_robot[robot]:
        return
    model_robot = model_robot_index[robot]
    articulation_row_dof_idx = articulation_dof_idx_of_padded_dof_idx[robot, padded_row_dof_idx]
    articulation_col_dof_idx = articulation_dof_idx_of_padded_dof_idx[robot, padded_col_dof_idx]
    out[robot, padded_row_dof_idx, padded_col_dof_idx] = model_mass_matrix[
        model_robot, articulation_row_dof_idx, articulation_col_dof_idx
    ]


# wp.copy is not recordable under APIC graph capture when either side is
# non-contiguous, which every indexed-view port is. These two kernels do the
# same work in a form that captures and serialises. Controllers launch them
# at their own port length: one entry per controlled DOF for a compact port, one
# per model coordinate or DOF for a model-based controller's whole-model ports.


@wp.kernel
def _gather_rank1_port_kernel(
    port: wp.indexedarray(dtype=Any),  # view of a simulation-sized array
    out: wp.array[Any],  # one entry per element the view addresses
):
    dof = wp.tid()
    out[dof] = port[dof]


@wp.kernel
def _gather_mass_matrix_port_kernel(
    port: wp.indexedarray(dtype=wp.float32, ndim=3),  # view selecting robots from a larger set of blocks
    out: wp.array3d[wp.float32],  # (controlled_robot_count, max_controlled_dofs, max_controlled_dofs)
):
    robot, row, col = wp.tid()
    out[robot, row, col] = port[robot, row, col]


@wp.kernel
def _scatter_port_kernel(
    values: wp.array[wp.float32],  # one entry per element the view addresses
    port: wp.indexedarray[wp.float32],  # view of a simulation-sized array
):
    dof = wp.tid()
    port[dof] = values[dof]


# dtype -> (rank -> gather kernel), the set of dtype/rank combinations any
# controller's ports currently use. Extend this table, not _read_port itself,
# when a controller needs a new port dtype or rank. Every rank-1 dtype shares
# _gather_rank1_port_kernel: it's generic over dtype (Any), so Warp compiles
# one concrete kernel per dtype the table actually uses, from a single body.
_GATHER_KERNELS_BY_DTYPE_AND_RANK = {
    wp.float32: {1: _gather_rank1_port_kernel, 3: _gather_mass_matrix_port_kernel},
    wp.transform: {1: _gather_rank1_port_kernel},
    wp.spatial_vector: {1: _gather_rank1_port_kernel},
    wp.quat: {1: _gather_rank1_port_kernel},
}


def _read_port(
    port: wp.array | wp.indexedarray,
    buffer: wp.array,
    shape: int | tuple[int, ...],
    device: Devicelike,
) -> None:
    """Copy a bound port into an internal buffer, whatever it is bound to.

    A view has to go through a kernel: :func:`warp.copy` is not recordable under
    APIC graph capture when either side is non-contiguous, so using it here would
    make a controller that reports ``is_graphable()`` fail to export.

    Args:
        port: The caller-bound port, a :class:`warp.array` or a view of one.
            Any dtype/rank combination in :data:`_GATHER_KERNELS_BY_DTYPE_AND_RANK`
            is supported when ``port`` is a view; a plain array supports any
            dtype/rank, since :func:`warp.copy` doesn't care.
        buffer: Destination, matching ``port`` in shape and dtype.
        shape: Launch shape — the length for a 1-D port, ``(robots, rows, cols)``
            for a padded per-robot matrix.
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
            f"add one to _GATHER_KERNELS_BY_DTYPE_AND_RANK in controllers/impl/_common.py."
        )
    wp.launch(kernel, dim=shape, inputs=[port], outputs=[buffer], device=device)
