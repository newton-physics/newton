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

A transform that is actively being *composed* with another is named
``coordinate_change_TARGET_from_SOURCE``: given a point's coordinates in the
SOURCE frame, it produces that same point's coordinates in the TARGET frame.
Equivalently, its translation is the SOURCE frame's origin, expressed in the
TARGET frame — so ``coordinate_change_world_from_body``'s translation is
directly the body's position in world coordinates, with no inversion needed.
Warp's ``*`` composes transforms as ``(A * B)(p) = A(B(p))`` (right operand
applied first), so this naming makes a chain of transforms cancel visibly,
left to right: ``coordinate_change_world_from_body *
coordinate_change_body_from_tool == tool_pose_world`` — the adjacent
``body``s are the frame the right transform's output and the left
transform's input agree on.

Once a transform is just data — the result of a composition, not itself
being multiplied against anything else — it's named like any other
frame-tagged quantity instead: ``tool_pose_world`` (the composition's
result above), matching ``tool_twist_world`` right next to it. The
``coordinate_change_*`` naming only earns its keep where a chain is actually
being built.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ....math import velocity_at_point

# Cholesky pivots are clamped above this, scaled by the pivot's own magnitude,
# so float32 cancellation noise on a near-singular matrix can't drive a
# pivot negative (which would make the square root below NaN).
_FLOAT32_EPS = wp.constant(wp.float32(np.finfo(np.float32).eps))


@wp.func
def _rotate_spatial_vector(quat_target_from_source: wp.quat, vector_source: wp.spatial_vector) -> wp.spatial_vector:
    """Rotate a spatial vector's linear/angular halves independently by the same rotation.

    A twist, force, or wrench's components change under a change of basis
    the same way any other vector's do -- a pure rotation, no translation
    and no dependence on the target/source frames' relative motion (this is
    a coordinate re-expression of the same physical quantity, not a moving-
    reference-frame velocity correction). Shared by every kernel that needs
    to re-express one of these in a different frame: :func:`_pose_twist_to_frame_kernel`
    (the tool's twist), :func:`_rotate_jacobian_to_frame_kernel` (each
    Jacobian column), and the wrench command kernels below (the desired/
    measured wrench).
    """
    linear_target = wp.quat_rotate(quat_target_from_source, wp.spatial_top(vector_source))
    angular_target = wp.quat_rotate(quat_target_from_source, wp.spatial_bottom(vector_source))
    return wp.spatial_vector(linear_target, angular_target)


@wp.kernel
def _tool_pose_and_twist_kernel(
    body_q: wp.array[wp.transform],  # (body_count,) coordinate_change_world_from_body per body
    body_qd_world: wp.array[wp.spatial_vector],  # (body_count,) twist about the COM point, in world coords (v_com, w)
    body_com_body: wp.array[wp.vec3],  # (body_count,) COM position, in the body's own local frame
    tool_body: wp.array[wp.int32],  # (robot_count,) -> body index of each robot's tool site
    coordinate_change_body_from_tool: wp.array[wp.transform],  # (robot_count,) tool site's body-local transform
    # outputs
    tool_pose_world: wp.array[wp.transform],  # (robot_count,) world pose of the tool frame
    tool_twist_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) twist about the tool point, in world coords (v_tool, w)
):
    robot_idx = wp.tid()
    tool_body_idx = tool_body[robot_idx]
    coordinate_change_world_from_body = body_q[tool_body_idx]
    tool_pose_world[robot_idx] = coordinate_change_world_from_body * coordinate_change_body_from_tool[robot_idx]

    tool_point_world = wp.transform_get_translation(tool_pose_world[robot_idx])
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
def _rotate_jacobian_to_frame_kernel(
    frame_pose_world: wp.array[wp.transform],  # (robot_count,) world pose of the target frame
    jacobian_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    jacobian_in_frame: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns rotated into the target frame; zero beyond dof_count
):
    """Rotate every column of a tool-point Jacobian from world into an arbitrary frame.

    Each column is itself a spatial vector (a per-DOF unit twist), so this
    is :func:`_rotate_spatial_vector` applied column by column. Downstream,
    Lambda, the null-space projector, and both J^T force mappings are
    computed entirely from the result -- the world-frame Jacobian is never
    read again after this, since ``J_frame^T @ (...)`` sandwiches built
    consistently from the same rotation always equal their world-frame
    counterparts exactly (the rotation cancels: ``(R @ J)^T @ (R @ ...) ==
    J^T @ (...)`` for any orthogonal ``R``).
    """
    robot_idx, padded_dof_idx = wp.tid()
    if padded_dof_idx >= dof_count[robot_idx]:
        return
    quat_frame_from_world = wp.quat_inverse(wp.transform_get_rotation(frame_pose_world[robot_idx]))

    column_world = wp.spatial_vector()
    for row in range(6):
        column_world[row] = jacobian_world[robot_idx, row, padded_dof_idx]
    column_in_frame = _rotate_spatial_vector(quat_frame_from_world, column_world)
    for row in range(6):
        jacobian_in_frame[robot_idx, row, padded_dof_idx] = column_in_frame[row]


# ---------------------------------------------------------------------------
# Operational-space mass matrix: Lambda = (J M^-1 J^T)^-1.
#
# Both M (the joint-space mass matrix) and Lambda^-1 = J M^-1 J^T are
# symmetric positive-definite, so _invert_spd_block_kernel below is used
# twice: once to invert M (block_dim = each robot's controlled-DOF count),
# once to invert Lambda^-1 (block_dim = 6, the fixed task dimension).
#
# Lambda^-1 = J M^-1 J^T only has rank min(6, controlled_dof_count). For a
# robot with fewer than 6 controlled DOFs, it is genuinely singular, not
# just ill-conditioned — the Cholesky pivot floor in _invert_spd_block_kernel
# keeps that from producing NaN, but it produces a huge, physically
# meaningless Lambda entry along the uncontrollable directions instead
# (verified empirically: eigenvalues up to ~1e8 for a 2-DOF arm, ~1e6 for a
# 5-DOF arm, vs. O(1-100) for 6+ DOF). ControllerOperationalSpaceModelFree
# raises at construction instead, when use_inertia_decoupling=True and a
# robot has fewer than 6 controlled DOFs, rather than letting this misbehave
# silently at runtime.
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
    robot_dof_count = dof_count[robot_idx]

    total = float(0.0)
    for dof_a in range(robot_dof_count):
        for dof_b in range(robot_dof_count):
            total += (
                jacobian_tool_world[robot_idx, row, dof_a]
                * mass_matrix_inv[robot_idx, dof_a, dof_b]
                * jacobian_tool_world[robot_idx, col, dof_b]
            )
    operational_space_mass_matrix_inv[robot_idx, row, col] = total


# ---------------------------------------------------------------------------
# Operational frame: commands (desired pose/twist) and gains are specified
# relative to this frame, not directly in world coordinates -- it may be
# fixed or time-varying, and need not coincide with the tool's own current
# orientation (e.g. a frame aligned to a work surface, tracked independently
# of how the tool itself is oriented).
# ---------------------------------------------------------------------------


@wp.kernel
def _pose_twist_to_frame_kernel(
    frame_pose_world: wp.array[wp.transform],  # (robot_count,) world pose of the target frame
    pose_world: wp.array[wp.transform],  # (robot_count,) a world pose, e.g. the tool's current pose
    twist_world: wp.array[wp.spatial_vector],  # (robot_count,) the same body's twist, world coords
    # outputs
    pose_in_frame: wp.array[wp.transform],  # (robot_count,) pose_world, relative to the target frame
    twist_in_frame: wp.array[wp.spatial_vector],  # (robot_count,) twist_world, components expressed in the target frame
):
    """Express a world pose/twist relative to an arbitrary frame instead of world."""
    robot_idx = wp.tid()
    coordinate_change_frame_from_world = wp.transform_inverse(frame_pose_world[robot_idx])
    pose_in_frame[robot_idx] = coordinate_change_frame_from_world * pose_world[robot_idx]

    quat_frame_from_world = wp.transform_get_rotation(coordinate_change_frame_from_world)
    twist_in_frame[robot_idx] = _rotate_spatial_vector(quat_frame_from_world, twist_world[robot_idx])


# ---------------------------------------------------------------------------
# Task-space pose error: how far the tool is from where it should be. Frame-
# agnostic -- both poses just need to already be expressed in the same
# frame, which is always the operational frame by the time this runs.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task-space impedance law: pose/velocity error -> desired task-space
# acceleration -> task-space force -> joint torque.
# ---------------------------------------------------------------------------


@wp.kernel
def _task_space_pd_kernel(
    pose_error_operational: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) (position error, orientation error) from _pose_error_kernel, operational frame
    tool_twist_operational: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) current tool twist, components expressed in the operational frame
    desired_twist_operational: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) desired tool twist, components expressed in the operational frame
    stiffness: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) per-axis proportional gain Kp, operational-frame-local; [1/s^2] if inertial decoupling follows, else [N/m or N*m/rad]
    damping: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) per-axis derivative gain Kd, operational-frame-local; [1/s] if inertial decoupling follows, else [N*s/m or N*m*s/rad]
    # outputs
    desired_task_acceleration_operational: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) Kp .* pose_error + Kd .* twist_error, in the operational frame
):
    """Task-space spring-damper term, ``Kp .* pose_error + Kd .* (desired_twist - current_twist)``.

    The same law as :func:`_pd_term_kernel` in the joint-impedance controller
    family, just operating on a 6D task-space error instead of a per-DOF one:
    a proportional term pulling the tool toward the desired pose, plus a
    derivative term pulling its twist toward the desired twist.

    Kp/Kd are specified per-axis in the operational frame -- e.g. "stiff
    along the insertion axis" should stay true as that frame reorients, not
    silently become "stiff along whatever world axis the insertion axis
    started aligned with". Every input and the output are already expressed
    in the operational frame (:func:`_pose_twist_to_frame_kernel` puts the
    tool's own state there, callers pass the desired twist/pose error the
    same way, and Lambda -- the next consumer of this output -- is itself
    computed from a Jacobian rotated into the operational frame), so this
    reduces to a plain per-axis multiply with no rotation anywhere at all.
    """
    robot_idx = wp.tid()
    pose_error = pose_error_operational[robot_idx]
    twist_error = desired_twist_operational[robot_idx] - tool_twist_operational[robot_idx]
    kp = stiffness[robot_idx]
    kd = damping[robot_idx]

    desired_task_acceleration_operational[robot_idx] = wp.cw_mul(kp, pose_error) + wp.cw_mul(kd, twist_error)


@wp.kernel
def _apply_spatial_matrix_kernel(
    matrix: wp.array3d[float],  # (robot_count, 6, 6) Lambda, from _invert_spd_block_kernel
    vector: wp.array[wp.spatial_vector],  # (robot_count,) a desired task-space acceleration
    # outputs
    result: wp.array[wp.spatial_vector],  # (robot_count,) = matrix @ vector
):
    """Multiply a 6x6 task-space matrix by a task-space vector, ``result = matrix @ vector``.

    Used for inertial decoupling: ``matrix = Lambda`` (the operational-space
    mass matrix), ``vector`` a desired task-space acceleration, ``result``
    the task-space force that produces it — the operational-space analogue
    of ``F = m*a``. Skipping this step entirely (using the acceleration
    directly as the force) is the task-space-impedance alternative, which
    ignores the tool's effective inertia. (Frame-local axis selection uses
    :func:`_apply_generalized_task_specification_matrix_kernel` instead,
    which never builds a 6x6 matrix at all.)

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
    robot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> column within that robot's Jacobian
    # outputs
    joint_torque: wp.array[float],  # (total_controlled_dofs,) compact = jacobian_tool_world^T @ task_space_force_world
):
    """Map a task-space force to joint torques, ``tau = J^T @ F``, straight into the compact per-DOF layout.

    Row ``dof`` of ``J^T`` is column ``slot_of_dof[dof]`` of robot
    ``robot_of_dof[dof]``'s Jacobian, which is exactly what
    :func:`_shift_jacobian_to_tool_kernel` produced one spatial-vector column
    at a time — loading it back into a ``wp.spatial_vector`` here lets this
    kernel use Warp's built-in dot product instead of a hand-rolled sum.
    ``robot_of_dof``/``slot_of_dof`` are the same compact-DOF lookup tables
    :func:`_block_matrix_vector_multiply_kernel` (``controllers/impl/_common.py``)
    uses, so no padding columns are ever read: every compact index is a real,
    controlled DOF.
    """
    dof = wp.tid()
    robot = robot_of_dof[dof]
    slot = slot_of_dof[dof]

    jacobian_column = wp.spatial_vector()
    for row in range(6):
        jacobian_column[row] = jacobian_tool_world[robot, row, slot]

    joint_torque[dof] = wp.dot(jacobian_column, task_space_force_world[robot])


# ---------------------------------------------------------------------------
# Null-space projector: N = I - J^T @ jacobian_pinv_transpose.
#
# jacobian_pinv_transpose has two variants:
#   - dynamically consistent: Lambda @ J @ M^-1 (needs the mass matrix)
#   - Moore-Penrose: (J @ J^T)^-1 @ J (kinematic only, ignores inertia)
# Only the dynamically-consistent variant guarantees that a joint torque
# entirely in the null space, tau_null = N @ M @ a for any joint
# acceleration a, produces zero task-space acceleration. That guarantee is
# the identity J @ M^-1 @ N == 0, which the module tests check directly
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
    robot_dof_count = dof_count[robot_idx]
    total = float(0.0)
    for dof in range(robot_dof_count):
        total += jacobian_tool_world[robot_idx, row, dof] * jacobian_tool_world[robot_idx, col, dof]
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
    for task_axis in range(6):
        total += task_matrix[robot_idx, row, task_axis] * jacobian_tool_world[robot_idx, task_axis, col]
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
    robot_dof_count = dof_count[robot_idx]
    if col >= robot_dof_count:
        return
    total = float(0.0)
    for dof in range(robot_dof_count):
        total += matrix[robot_idx, row, dof] * mass_matrix_inv[robot_idx, dof, col]
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

    A joint torque built as ``N @ M @ a``, for any joint acceleration ``a``
    and the joint-space mass matrix ``M``, produces zero task-space
    acceleration — but only when ``jacobian_pinv_transpose`` is the
    dynamically-consistent variant: ``J @ M^-1 @ N == 0`` in that case, and
    generally nonzero for the Moore-Penrose variant, since that one ignores
    the robot's inertia.
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
        total += jacobian_tool_world[robot_idx, k, row] * jacobian_pinv_transpose[robot_idx, k, col]
    null_space_projector[robot_idx, row, col] = identity_entry - total


# ---------------------------------------------------------------------------
# Motion/force selection and contact-wrench control.
#
# Motion- vs. force-controlled task axes need not share one frame: the
# linear/force selection is most naturally expressed relative to a frame S_f
# (e.g. a contact surface's normal), the angular/moment selection relative to
# a possibly *different* frame S_tau (e.g. a compliant-rotation axis) -- the
# generalized task specification matrix from Khatib, O. (1987), "A unified
# approach for motion and force control of robot manipulators: The
# operational space formulation," IEEE Journal of Robotics and Automation,
# 3(1), 43-53. In this file's own S_f/S_tau convention (quat_operational_from_sf/
# quat_operational_from_stau, rotating INTO the operational frame -- the
# paper's S_f/S_tau transposed): Omega = diag(S_f . Sigma_f . S_f^T,
# S_tau . Sigma_tau . S_tau^T). S_f/S_tau are themselves relative to the
# operational frame, applied to vectors already expressed there
# (:func:`_apply_generalized_task_specification_matrix_kernel`), so selection
# never touches world frame.
#
# Applied once, before Lambda for the motion branch and once on the combined
# wrench command for the force branch -- not a second time afterward, since
# Lambda's own coupling is exactly what should propagate through a selected
# acceleration; see the module-level docstring in ``model_free.py`` for the
# full derivation.
#
# Motion, force, and null-space joint torques are each mapped to joint space
# by their own :func:`_jacobian_transpose_force_kernel` call and summed there
# via ``_add_term_kernel`` (``controllers/impl/_common.py``) -- not summed as
# task-space forces first.
# ---------------------------------------------------------------------------


@wp.kernel
def _apply_generalized_task_specification_matrix_kernel(
    quat_operational_from_sf: wp.array[wp.quat],  # (robot_count,) S_f, relative to the operational frame
    quat_operational_from_stau: wp.array[wp.quat],  # (robot_count,) S_tau, relative to the operational frame
    selection_axes: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) diagonal selection weight per axis: linear half in S_f, angular half in S_tau
    vector_operational: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) a task-space acceleration/force/wrench, already in the operational frame
    # outputs
    masked_vector_operational: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) vector_operational's component along selected axes, still in the operational frame
):
    """Apply the generalized task specification matrix, zeroing excluded task axes.

    ``Omega = diag(S_f . Sigma_f . S_f^T, S_tau . Sigma_tau . S_tau^T)``, with
    ``S_f``/``S_tau`` meaning this kernel's own ``quat_operational_from_sf``/
    ``quat_operational_from_stau`` (rotating INTO the operational frame) --
    the transpose of Khatib, O. (1987)'s own S_f/S_tau convention (which
    rotate the opposite way, operational frame into the selection frame),
    "A unified approach for motion and force control of robot manipulators:
    The operational space formulation," IEEE Journal of Robotics and
    Automation, 3(1), 43-53. The linear half is rotated into
    ``S_f``, masked by ``Sigma_f``, and rotated back; the angular half does
    the same independently through ``S_tau``. ``S_f`` and ``S_tau`` need not
    agree -- e.g. the force-control direction (surface normal) and the
    compliant-rotation axis of a task generally differ. No world pose is
    needed here -- S_f/S_tau are defined relative to the operational frame
    directly, so this never touches world frame at all.
    """
    robot_idx = wp.tid()
    axes = selection_axes[robot_idx]
    vector = vector_operational[robot_idx]
    quat_sf = quat_operational_from_sf[robot_idx]
    quat_stau = quat_operational_from_stau[robot_idx]

    linear_sf = wp.quat_rotate_inv(quat_sf, wp.spatial_top(vector))
    masked_linear_sf = wp.cw_mul(wp.spatial_top(axes), linear_sf)
    masked_linear = wp.quat_rotate(quat_sf, masked_linear_sf)

    angular_stau = wp.quat_rotate_inv(quat_stau, wp.spatial_bottom(vector))
    masked_angular_stau = wp.cw_mul(wp.spatial_bottom(axes), angular_stau)
    masked_angular = wp.quat_rotate(quat_stau, masked_angular_stau)

    masked_vector_operational[robot_idx] = wp.spatial_vector(masked_linear, masked_angular)


@wp.kernel
def _wrench_feedforward_kernel(
    operational_frame_pose_world: wp.array[wp.transform],  # (robot_count,) world pose of the operational frame
    desired_wrench_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) desired contact wrench (force, moment), world coords
    # outputs
    wrench_command_operational: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) accumulator; desired_wrench_world, rotated into the operational frame, is added to it
):
    """Add the feedforward wrench term, ``desired_wrench_world`` rotated into the operational frame.

    ``wrench_command_operational`` is a running accumulator: the caller
    zeros it once per step, then launches this kernel and/or
    :func:`_wrench_feedback_kernel`, whichever are enabled, each adding
    its own term. The rotation is needed because the desired wrench is
    given in world coordinates but everything downstream (selection
    masking, the J^T force mapping) runs in the operational frame.
    """
    robot_idx = wp.tid()
    quat_operational_from_world = wp.quat_inverse(wp.transform_get_rotation(operational_frame_pose_world[robot_idx]))
    wrench_command_operational[robot_idx] = wrench_command_operational[robot_idx] + _rotate_spatial_vector(
        quat_operational_from_world, desired_wrench_world[robot_idx]
    )


@wp.kernel
def _wrench_feedback_kernel(
    operational_frame_pose_world: wp.array[wp.transform],  # (robot_count,) world pose of the operational frame
    desired_wrench_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) desired contact wrench (force, moment), world coords, used as the feedback setpoint
    measured_wrench_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) measured contact wrench (force, moment), world coords, e.g. from a 6-axis force/torque sensor
    stiffness: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) per-axis proportional feedback gain Kp, operational-frame-local
    # outputs
    wrench_command_operational: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) accumulator; Kp .* (desired - measured) is added to it
):
    """Add the feedback wrench term, ``Kp .* (desired - measured)``, both rotated into the operational frame.

    ``wrench_command_operational`` is a running accumulator: the caller
    zeros it once per step, then launches this kernel and/or
    :func:`_wrench_feedforward_kernel`, whichever are enabled, each adding
    its own term. This assumes the full wrench (force and moment) is
    measurable, e.g. from a 6-axis force/torque sensor.
    """
    robot_idx = wp.tid()
    quat_operational_from_world = wp.quat_inverse(wp.transform_get_rotation(operational_frame_pose_world[robot_idx]))
    desired_operational = _rotate_spatial_vector(quat_operational_from_world, desired_wrench_world[robot_idx])
    measured_operational = _rotate_spatial_vector(quat_operational_from_world, measured_wrench_world[robot_idx])
    wrench_error_operational = desired_operational - measured_operational
    kp = stiffness[robot_idx]

    feedback_operational = wp.cw_mul(kp, wrench_error_operational)
    wrench_command_operational[robot_idx] = wrench_command_operational[robot_idx] + feedback_operational
