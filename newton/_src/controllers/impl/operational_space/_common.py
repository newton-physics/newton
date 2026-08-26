# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for the operational-space controller family.

The operational-space frame is called the *tool* frame throughout. It is
implemented as a Newton *site* (a body-fixed offset, ``tool_body`` +
``coordinate_change_body_from_tool``), resolved once per robot, one entry per
robot (one task per environment). "Site" only appears where these kernels are
literally reading that underlying Newton primitive; every resolved,
per-step quantity is named "tool". These kernels shift the COM twists
:func:`~newton.eval_jacobian` and :class:`~newton.State` produce — each one
about the body's COM point, expressed in world-frame coordinates,
per-articulation-local link/DOF indexing — to the tool point, still
expressed in world-frame coordinates.

Every transform variable is named ``coordinate_change_TARGET_from_SOURCE``:
given a point's coordinates in the SOURCE frame, it produces that same
point's coordinates in the TARGET frame. Equivalently, its translation is the
SOURCE frame's origin, expressed in the TARGET frame — so
``coordinate_change_world_from_body``'s translation is directly the body's
position in world coordinates, with no inversion needed. Warp's ``*``
composes transforms as ``(A * B)(p) = A(B(p))`` (right operand applied
first), so this naming makes a chain of transforms cancel visibly, left to
right: ``coordinate_change_world_from_body * coordinate_change_body_from_tool
== coordinate_change_world_from_tool`` — the adjacent ``body``s are the frame
the right transform's output and the left transform's input agree on.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ....math import velocity_at_point

# Cholesky pivots are clamped above this, scaled by the pivot's own magnitude,
# so float32 cancellation noise on a near-singular matrix can't drive a
# pivot negative (which would make the square root below NaN).
_FLOAT32_EPS = wp.constant(wp.float32(np.finfo(np.float32).eps))


@wp.kernel
def _tool_pose_and_twist_kernel(
    body_q: wp.array[wp.transform],  # (body_count,) coordinate_change_world_from_body per body
    body_qd_world: wp.array[wp.spatial_vector],  # (body_count,) twist about the COM point, in world coords (v_com, w)
    body_com_body: wp.array[wp.vec3],  # (body_count,) COM position, in the body's own local frame
    tool_body: wp.array[wp.int32],  # (robot_count,) -> body index of each robot's tool site
    coordinate_change_body_from_tool: wp.array[wp.transform],  # (robot_count,) tool site's body-local transform
    # outputs
    coordinate_change_world_from_tool: wp.array[wp.transform],  # (robot_count,) world pose of the tool frame
    tool_twist_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) twist about the tool point, in world coords (v_tool, w)
):
    robot_idx = wp.tid()
    tool_body_idx = tool_body[robot_idx]
    coordinate_change_world_from_body = body_q[tool_body_idx]
    coordinate_change_world_from_tool[robot_idx] = (
        coordinate_change_world_from_body * coordinate_change_body_from_tool[robot_idx]
    )

    tool_point_world = wp.transform_get_translation(coordinate_change_world_from_tool[robot_idx])
    body_com_world = wp.transform_point(coordinate_change_world_from_body, body_com_body[tool_body_idx])
    com_to_tool_offset_world = tool_point_world - body_com_world
    # Angular velocity is the same everywhere on a rigid body, so only the
    # linear part changes when shifting the twist's reference point.
    body_twist_com_world = body_qd_world[tool_body_idx]
    tool_twist_world[robot_idx] = wp.spatial_vector(
        velocity_at_point(body_twist_com_world, com_to_tool_offset_world), wp.spatial_bottom(body_twist_com_world)
    )


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
    # outputs
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
):
    robot_idx, dof_idx = wp.tid()
    articulation_idx = robot_articulation[robot_idx]
    link_row_start = robot_link_idx[robot_idx] * 6

    tool_body_idx = tool_body[robot_idx]
    coordinate_change_world_from_body = body_q[tool_body_idx]
    coordinate_change_world_from_tool = coordinate_change_world_from_body * coordinate_change_body_from_tool[robot_idx]
    tool_point_world = wp.transform_get_translation(coordinate_change_world_from_tool)
    body_com_world = wp.transform_point(coordinate_change_world_from_body, body_com_body[tool_body_idx])
    com_to_tool_offset_world = tool_point_world - body_com_world

    jacobian_column_com_world = wp.spatial_vector(
        jacobian_com_world[articulation_idx, link_row_start + 0, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 1, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 2, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 3, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 4, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 5, dof_idx],
    )
    jacobian_column_tool_world = wp.spatial_vector(
        velocity_at_point(jacobian_column_com_world, com_to_tool_offset_world),
        wp.spatial_bottom(jacobian_column_com_world),
    )
    for row in range(6):
        jacobian_tool_world[robot_idx, row, dof_idx] = jacobian_column_tool_world[row]


# ---------------------------------------------------------------------------
# Operational-space mass matrix: Lambda = (J M^-1 J^T)^-1.
#
# Both M (the joint-space mass matrix) and Lambda^-1 = J M^-1 J^T are
# symmetric positive-definite, so _invert_spd_block_kernel below is used
# twice: once to invert M (block_dim = each robot's controlled-DOF count),
# once to invert Lambda^-1 (block_dim = 6, the fixed task dimension).
#
# TODO(operational-space controller): Lambda^-1 = J M^-1 J^T only has rank
# min(6, controlled_dof_count). For a robot with fewer than 6 controlled
# DOFs, it is genuinely singular, not just ill-conditioned — the Cholesky
# pivot floor in _invert_spd_block_kernel keeps that from producing NaN, but
# it produces a huge, physically meaningless Lambda entry along the
# uncontrollable directions instead (verified empirically: eigenvalues up to
# ~1e8 for a 2-DOF arm, ~1e6 for a 5-DOF arm, vs. O(1-100) for 6+ DOF).
# ControllerOperationalSpace(ModelFree) should raise at construction when
# use_inertia_decoupling=True (full, non-partial) and a robot's
# controlled-DOF count < 6, matching ControllerJointImpedance's pattern of
# validating configuration up front rather than letting it misbehave silently
# at runtime.
# ---------------------------------------------------------------------------


@wp.kernel
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
    """
    block_idx = wp.tid()
    n = block_dim[block_idx]

    # Cholesky factorization: spd_matrix == cholesky_factor @ cholesky_factor^T.
    for col in range(n):
        diagonal_term = spd_matrix[block_idx, col, col]
        for k in range(col):
            diagonal_term -= cholesky_factor[block_idx, col, k] * cholesky_factor[block_idx, col, k]
        diagonal_term = wp.max(diagonal_term, _FLOAT32_EPS * wp.max(wp.abs(spd_matrix[block_idx, col, col]), 1.0))
        diagonal_value = wp.sqrt(diagonal_term)
        cholesky_factor[block_idx, col, col] = diagonal_value
        for row in range(col + 1, n):
            off_diagonal_term = spd_matrix[block_idx, row, col]
            for k in range(col):
                off_diagonal_term -= cholesky_factor[block_idx, row, k] * cholesky_factor[block_idx, col, k]
            cholesky_factor[block_idx, row, col] = off_diagonal_term / diagonal_value

    # Solve spd_matrix @ x = e_c for every column c, writing x into column c of the inverse.
    for c in range(n):
        # Forward substitution: cholesky_factor @ y = e_c.
        for row in range(n):
            right_hand_side = float(0.0)
            if row == c:
                right_hand_side = 1.0
            for k in range(row):
                right_hand_side -= cholesky_factor[block_idx, row, k] * spd_matrix_inv[block_idx, k, c]
            spd_matrix_inv[block_idx, row, c] = right_hand_side / cholesky_factor[block_idx, row, row]
        # Back substitution: cholesky_factor^T @ x = y, overwriting y with x in place.
        for reverse_row in range(n):
            row = n - 1 - reverse_row
            right_hand_side = spd_matrix_inv[block_idx, row, c]
            for k in range(row + 1, n):
                right_hand_side -= cholesky_factor[block_idx, k, row] * spd_matrix_inv[block_idx, k, c]
            spd_matrix_inv[block_idx, row, c] = right_hand_side / cholesky_factor[block_idx, row, row]


@wp.kernel
def _operational_space_mass_matrix_inverse_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
    mass_matrix_inv: wp.array3d[
        float
    ],  # (robot_count, max_dofs, max_dofs) inverse of the controlled-DOF mass matrix; zero beyond dof_count
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    operational_space_mass_matrix_inv: wp.array3d[
        float
    ],  # (robot_count, 6, 6) = jacobian_tool_world @ mass_matrix_inv @ jacobian_tool_world^T
):
    """The inverse operational-space mass matrix, ``Lambda^-1 = J M^-1 J^T``.

    Still needs a 6x6 inverse (via :func:`_invert_spd_block_kernel`) to become
    the operational-space mass matrix Lambda that maps a desired task-space
    acceleration to the task-space force that would produce it.
    """
    robot_idx, row, col = wp.tid()
    n = dof_count[robot_idx]

    total = float(0.0)
    for a in range(n):
        for b in range(n):
            total += (
                jacobian_tool_world[robot_idx, row, a]
                * mass_matrix_inv[robot_idx, a, b]
                * jacobian_tool_world[robot_idx, col, b]
            )
    operational_space_mass_matrix_inv[robot_idx, row, col] = total


# ---------------------------------------------------------------------------
# Task-space pose error: how far the tool is from where it should be.
# ---------------------------------------------------------------------------


@wp.kernel
def _pose_error_kernel(
    coordinate_change_world_from_tool: wp.array[wp.transform],  # (robot_count,) current world pose of the tool frame
    coordinate_change_world_from_desired_tool: wp.array[
        wp.transform
    ],  # (robot_count,) desired world pose of the tool frame
    # outputs
    pose_error_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) (position error, orientation error) in world coords: desired minus current
):
    """Task-space pose error, ``(desired_position - current_position, orientation_error)``.

    The position error is a plain vector difference, in world coordinates.

    The orientation error is the axis-angle rotation that would carry the
    current tool orientation to the desired one, in world coordinates: rotate
    the current orientation by ``angle`` about ``axis`` and it lands on the
    desired orientation. It shrinks to zero exactly when the two orientations
    agree, matching the position error's "desired minus current" sign so both
    halves of the 6D error can be driven to zero by the same kind of
    proportional term.

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

    coordinate_change_world_from_current_tool = coordinate_change_world_from_tool[robot_idx]
    position_error_world = wp.transform_get_translation(
        coordinate_change_world_from_desired_tool[robot_idx]
    ) - wp.transform_get_translation(coordinate_change_world_from_current_tool)

    quat_current = wp.transform_get_rotation(coordinate_change_world_from_current_tool)
    quat_desired = wp.transform_get_rotation(coordinate_change_world_from_desired_tool[robot_idx])
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
        orientation_error_world = (quat_error_vector / quat_error_vector_norm) * angle
    else:
        orientation_error_world = 2.0 * quat_error_vector

    pose_error_world[robot_idx] = wp.spatial_vector(position_error_world, orientation_error_world)


# ---------------------------------------------------------------------------
# Task-space impedance law: pose/velocity error -> desired task-space
# acceleration -> task-space force -> joint torque.
# ---------------------------------------------------------------------------


@wp.kernel
def _task_space_pd_kernel(
    pose_error_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) (position error, orientation error) from _pose_error_kernel
    tool_twist_world: wp.array[wp.spatial_vector],  # (robot_count,) current tool twist, world coords
    desired_twist_world: wp.array[wp.spatial_vector],  # (robot_count,) desired tool twist, world coords
    stiffness: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) per-axis proportional gain Kp; [1/s^2] if inertial decoupling follows, else [N/m or N*m/rad]
    damping: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) per-axis derivative gain Kd; [1/s] if inertial decoupling follows, else [N*s/m or N*m*s/rad]
    # outputs
    desired_task_acceleration_world: wp.array[wp.spatial_vector],  # (robot_count,) Kp .* pose_error + Kd .* twist_error
):
    """Task-space spring-damper term, ``Kp .* pose_error + Kd .* (desired_twist - current_twist)``.

    The same law as :func:`_pd_term_kernel` in the joint-impedance controller
    family, just operating on a 6D task-space error instead of a per-DOF one:
    a proportional term pulling the tool toward the desired pose, plus a
    derivative term pulling its twist toward the desired twist. Gains are
    per-axis (diagonal), not a full 6x6 matrix, since there is no task-frame
    rotation layer here for a matrix-valued gain to matter — see the module
    docstring.
    """
    robot_idx = wp.tid()
    pose_error = pose_error_world[robot_idx]
    twist_error = desired_twist_world[robot_idx] - tool_twist_world[robot_idx]
    proportional_gain = stiffness[robot_idx]
    derivative_gain = damping[robot_idx]

    result = wp.spatial_vector()
    for axis in range(6):
        result[axis] = proportional_gain[axis] * pose_error[axis] + derivative_gain[axis] * twist_error[axis]
    desired_task_acceleration_world[robot_idx] = result


@wp.kernel
def _apply_spatial_matrix_kernel(
    matrix: wp.array3d[float],  # (robot_count, 6, 6) e.g. Lambda (from _invert_spd_block_kernel) or a selection matrix
    vector: wp.array[wp.spatial_vector],  # (robot_count,) e.g. a desired task-space acceleration, or a wrench
    # outputs
    result: wp.array[wp.spatial_vector],  # (robot_count,) = matrix @ vector
):
    """Multiply a 6x6 task-space matrix by a task-space vector, ``result = matrix @ vector``.

    Two unrelated uses share this kernel because the operation is identical:

    - **Inertial decoupling**: ``matrix = Lambda`` (the operational-space mass
      matrix), ``vector`` a desired task-space acceleration, ``result`` the
      task-space force that produces it — the operational-space analogue of
      ``F = m*a``. Skipping this step entirely (using the acceleration
      directly as the force) is the task-space-impedance alternative, which
      ignores the tool's effective inertia — the same ``use_inertia_decoupling``
      choice :class:`ControllerJointImpedanceModelFree` offers at the joint level.
    - **Selection masking**: ``matrix`` a world-frame selection matrix (from
      :func:`_rotate_selection_matrix_kernel`), ``vector`` a candidate
      task-space force or wrench, ``result`` its component along the
      selected axes only.

    ``matrix`` is stored as a plain ``(robot_count, 6, 6)`` float array — not
    a ``wp.spatial_matrix`` array — because Lambda comes from
    :func:`_invert_spd_block_kernel`, which also produces the joint-space
    mass-matrix inverse, whose block size varies per robot and can exceed 6;
    a fixed-size ``spatial_matrix`` only fits Lambda's always-exactly-6x6
    case. This kernel loads ``matrix`` into a local ``wp.spatial_matrix`` so
    it can use Warp's built-in matrix-vector product rather than a
    hand-rolled accumulation loop.
    """
    robot_idx = wp.tid()

    local_matrix = wp.spatial_matrix()
    for row in range(6):
        for col in range(6):
            local_matrix[row, col] = matrix[robot_idx, row, col]

    result[robot_idx] = local_matrix * vector[robot_idx]


@wp.kernel
def _jacobian_transpose_force_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
    task_space_force_world: wp.array[wp.spatial_vector],  # (robot_count,) task-space force/wrench to map to joints
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    joint_torque: wp.array2d[float],  # (robot_count, max_dofs) = jacobian_tool_world^T @ task_space_force_world
):
    """Map a task-space force to joint torques, ``tau = J^T @ F``.

    Columns at or past ``dof_count[robot_idx]`` are left unwritten — they are
    padding, not part of any robot's controlled-DOF set. Row ``dof_idx`` of
    ``J^T`` is column ``dof_idx`` of ``J``, which is exactly what
    :func:`_shift_jacobian_to_tool_kernel` produced one spatial-vector column
    at a time — loading it back into a ``wp.spatial_vector`` here lets this
    kernel use Warp's built-in dot product instead of a hand-rolled sum.
    """
    robot_idx, dof_idx = wp.tid()
    if dof_idx >= dof_count[robot_idx]:
        return

    jacobian_column = wp.spatial_vector()
    for row in range(6):
        jacobian_column[row] = jacobian_tool_world[robot_idx, row, dof_idx]

    joint_torque[robot_idx, dof_idx] = wp.dot(jacobian_column, task_space_force_world[robot_idx])


# ---------------------------------------------------------------------------
# Null-space projector: N = I - J^T @ jacobian_pinv_transpose.
#
# jacobian_pinv_transpose has two variants:
#   - dynamically consistent: Lambda @ J @ M^-1 (needs the mass matrix)
#   - Moore-Penrose: (J @ J^T)^-1 @ J (kinematic only, ignores inertia)
# Only the dynamically-consistent variant guarantees that a joint torque
# entirely in the null space, tau_null = N^T @ M @ a for any joint
# acceleration a, produces zero task-space acceleration. That guarantee is
# the identity J @ M^-1 @ N^T == 0, which the module tests check directly
# (and check does *not* hold for the Moore-Penrose variant, in general).
# ---------------------------------------------------------------------------


@wp.kernel
def _jacobian_times_jacobian_transpose_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    jacobian_times_jacobian_transpose: wp.array3d[float],  # (robot_count, 6, 6) = jacobian_tool_world @ its transpose
):
    """``J @ J^T``, the purely kinematic (inertia-blind) analogue of ``Lambda^-1 = J M^-1 J^T``.

    Its inverse (via :func:`_invert_spd_block_kernel`) gives the 6x6 factor
    the Moore-Penrose pseudo-inverse transpose needs, ``(J @ J^T)^-1 @ J``.
    """
    robot_idx, row, col = wp.tid()
    n = dof_count[robot_idx]
    total = float(0.0)
    for k in range(n):
        total += jacobian_tool_world[robot_idx, row, k] * jacobian_tool_world[robot_idx, col, k]
    jacobian_times_jacobian_transpose[robot_idx, row, col] = total


@wp.kernel
def _task_matrix_times_jacobian_kernel(
    task_matrix: wp.array3d[float],  # (robot_count, 6, 6) symmetric: Lambda, or (J @ J^T)^-1
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    result: wp.array3d[float],  # (robot_count, 6, max_dofs) = task_matrix @ jacobian_tool_world; zero beyond dof_count
):
    """The 6x6-matrix-times-Jacobian step shared by both pseudo-inverse-transpose variants.

    With ``task_matrix = Lambda`` this is the first half of the
    dynamically-consistent ``jacobian_pinv_transpose = Lambda @ J @ M^-1``
    (still needs :func:`_apply_mass_matrix_inv_on_right_kernel`). With
    ``task_matrix = (J @ J^T)^-1`` this *is* the Moore-Penrose
    ``jacobian_pinv_transpose`` already, with no further step needed.
    """
    robot_idx, row, col = wp.tid()
    if col >= dof_count[robot_idx]:
        return
    total = float(0.0)
    for k in range(6):
        total += task_matrix[robot_idx, row, k] * jacobian_tool_world[robot_idx, k, col]
    result[robot_idx, row, col] = total


@wp.kernel
def _apply_mass_matrix_inv_on_right_kernel(
    matrix: wp.array3d[float],  # (robot_count, 6, max_dofs), e.g. Lambda @ jacobian_tool_world
    mass_matrix_inv: wp.array3d[
        float
    ],  # (robot_count, max_dofs, max_dofs) inverse of the controlled-DOF mass matrix; zero beyond dof_count
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    result: wp.array3d[float],  # (robot_count, 6, max_dofs) = matrix @ mass_matrix_inv; zero beyond dof_count
):
    """Right-multiply by ``M^-1``, the remaining step of the dynamically-consistent pseudo-inverse transpose.

    Given ``matrix = Lambda @ jacobian_tool_world`` (from
    :func:`_task_matrix_times_jacobian_kernel`), this completes
    ``jacobian_pinv_transpose = Lambda @ J @ M^-1``.
    """
    robot_idx, row, col = wp.tid()
    n = dof_count[robot_idx]
    if col >= n:
        return
    total = float(0.0)
    for k in range(n):
        total += matrix[robot_idx, row, k] * mass_matrix_inv[robot_idx, k, col]
    result[robot_idx, row, col] = total


@wp.kernel
def _null_space_projector_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
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

    A joint torque ``N^T @ (anything)`` stays entirely in the null space of
    the task, in the sense that it doesn't change the joint-space-to-task-space
    kinematic relationship — see the module docstring and tests for what
    additional guarantee holds depending on which ``jacobian_pinv_transpose``
    variant built ``N``.
    """
    robot_idx, row, col = wp.tid()
    n = dof_count[robot_idx]
    if row >= n or col >= n:
        return

    identity_entry = float(0.0)
    if row == col:
        identity_entry = 1.0

    total = float(0.0)
    for k in range(6):
        total += jacobian_tool_world[robot_idx, k, row] * jacobian_pinv_transpose[robot_idx, k, col]
    null_space_projector[robot_idx, row, col] = identity_entry - total


# ---------------------------------------------------------------------------
# Motion/force selection and contact-wrench control.
#
# Which of the 6 task axes are motion-controlled vs. force-controlled is
# naturally a tool-local choice (e.g. "force-control along the insertion
# axis" for a peg-in-hole task should track the tool's current orientation,
# not stay fixed in world), so the selection weights are rotated into world
# coordinates every step from a fixed tool-local specification.
#
# The motion, force, and (if used) null-space joint torques are each mapped
# to joint space by their own :func:`_jacobian_transpose_force_kernel` call
# and then summed there -- not summed as task-space forces first. This
# matches how the reference controller (Isaac Lab's OperationalSpaceController)
# accumulates every contribution as ``joint_efforts +=``, and lets the
# summation reuse ``_add_term_kernel`` from ``joint_impedance/_common.py``
# rather than needing a task-space-vector accumulator kernel here.
# ---------------------------------------------------------------------------


@wp.kernel
def _rotate_selection_matrix_kernel(
    coordinate_change_world_from_tool: wp.array[wp.transform],  # (robot_count,) world pose of the tool frame
    selection_axes_tool: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) diagonal selection weight per axis (0/1, or any scalar weight), tool-local: (linear x,y,z, angular x,y,z)
    # outputs
    selection_matrix_world: wp.array3d[
        float
    ],  # (robot_count, 6, 6) block-diagonal; the linear-angular cross blocks are always zero
):
    """Rotate a tool-local diagonal selection into a world-frame selection matrix.

    A rotation only mixes the 3 linear axes among themselves, and separately
    the 3 angular axes among themselves, so the world-frame result stays
    block-diagonal — it is generally no longer itself diagonal (unless the
    tool-local weights within a block are all equal), which is why it has to
    be represented as a full 6x6 matrix rather than another diagonal/vector.
    """
    robot_idx = wp.tid()
    axes = selection_axes_tool[robot_idx]
    quat_world_from_tool = wp.transform_get_rotation(coordinate_change_world_from_tool[robot_idx])

    rotation_col_x = wp.quat_rotate(quat_world_from_tool, wp.vec3(1.0, 0.0, 0.0))
    rotation_col_y = wp.quat_rotate(quat_world_from_tool, wp.vec3(0.0, 1.0, 0.0))
    rotation_col_z = wp.quat_rotate(quat_world_from_tool, wp.vec3(0.0, 0.0, 1.0))
    rotation = wp.matrix_from_cols(rotation_col_x, rotation_col_y, rotation_col_z)
    rotation_transpose = wp.transpose(rotation)

    linear_diag = wp.mat33(axes[0], 0.0, 0.0, 0.0, axes[1], 0.0, 0.0, 0.0, axes[2])
    angular_diag = wp.mat33(axes[3], 0.0, 0.0, 0.0, axes[4], 0.0, 0.0, 0.0, axes[5])
    linear_block_world = rotation * linear_diag * rotation_transpose
    angular_block_world = rotation * angular_diag * rotation_transpose

    for row in range(3):
        for col in range(3):
            selection_matrix_world[robot_idx, row, col] = linear_block_world[row, col]
            selection_matrix_world[robot_idx, row + 3, col + 3] = angular_block_world[row, col]
            selection_matrix_world[robot_idx, row, col + 3] = 0.0
            selection_matrix_world[robot_idx, row + 3, col] = 0.0


@wp.kernel
def _closed_loop_wrench_command_kernel(
    desired_wrench_world: wp.array[wp.spatial_vector],  # (robot_count,) desired contact wrench (force, moment), world coords
    measured_wrench_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) measured contact wrench (force, moment), world coords, e.g. from a 6-axis force/torque sensor
    stiffness: wp.array[wp.spatial_vector],  # (robot_count,) per-axis proportional gain Kp
    # outputs
    wrench_command_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) closed-loop wrench command, desired + Kp .* (desired - measured)
):
    """Closed-loop contact-wrench command, ``desired + Kp .* (desired - measured)``.

    Same law as :func:`_task_space_pd_kernel`'s proportional term, applied
    uniformly across all 6 wrench axes — this assumes the full wrench
    (force and moment) is measurable, e.g. from a 6-axis force/torque sensor.
    """
    robot_idx = wp.tid()
    desired = desired_wrench_world[robot_idx]
    measured = measured_wrench_world[robot_idx]
    proportional_gain = stiffness[robot_idx]

    result = wp.spatial_vector()
    for axis in range(6):
        wrench_error = desired[axis] - measured[axis]
        result[axis] = desired[axis] + proportional_gain[axis] * wrench_error
    wrench_command_world[robot_idx] = result
