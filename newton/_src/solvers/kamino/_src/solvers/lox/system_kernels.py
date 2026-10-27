# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assembly kernels of the LOX smooth body systems.

Each factor block is one connected dynamic-body component and each dynamic body
contributes six linear-first velocity unknowns; prescribed bodies map to ``-1`` in the
matrix layout. The matrix entries of a body are read and written through its matrix rows, which follow
the row order of the factorization once it is known. The consensus body weights use the body mass matrix
``M = diag(m I_3, I_world)``, referenced at the body's center of mass.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from ...core.joints import JointDoFType
from ...core.math import compute_gyroscopic_torque
from ...core.types import mat36f, mat66f, vec6f, vec6i

###
# Module interface
###

__all__ = [
    "ProjectionWeights",
    "_assemble_body_inertial_systems",
    "_assemble_joint_rows",
    "_assemble_rod_geometric_stiffness",
    "_build_candidate_right_hand_side",
    "_compute_free_body_blocks",
    "_compute_patch_weights",
    "_invert_single_body_blocks",
    "_solve_single_body_blocks",
    "apply_body_weight",
    "compute_body_explicit_wrench",
    "compute_structural_correction",
    "is_rotational_structural_row",
    "make_patch_schur_kernel",
    "normalize_symmetric_matrix",
    "unpack_body_solution",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_EIGENVALUE_TOLERANCE = 1.0e-7
"""Relative off-diagonal tolerance of the QR eigenvalues of the normalized body blocks."""

_EIGENVALUE_FLUSH_FRACTION = 1.0e-10
"""Magnitude, relative to the largest entry, at or below which symmetric matrices flush their entries to zero
before a QR eigenvalue solve.

The Householder reflections of :func:`warp.fem.linalg.symmetric_eigenvalues_qr` square the matrix
entries. Flushing the entries far below the matrix scale keeps these squares in the normal float32
range, where the reflections stay finite.
"""


###
# Types
###


@wp.struct
class _PrimalRowContribution:
    """Matrix and right-hand-side contribution of one body-space term."""

    matrix: mat66f
    """Symmetric contribution to the body-space operator."""
    right_hand_side: vec6f
    """Contribution to the body-space right-hand side."""


###
# Functions
###


@wp.struct
class ProjectionWeights:
    """Body weights of the unilateral projection and the twist responses of the rows prepared from them."""

    inverse_weight: wp.array[mat66f]
    """Inverse body weight ``W^-1`` of each body."""
    box_response: wp.array2d[vec6f]
    """Twist response ``W^-1 J^T`` of each box row on its bodies A and B."""
    contact_response: wp.array2d[mat36f]
    """Transposed twist response ``J W^-1`` of each contact on its bodies A and B."""


@wp.func
def apply_body_weight(weight: mat66f, value: vec6f) -> vec6f:
    """Return ``weight @ value`` for a symmetric body weight or its inverse, reading its upper triangle."""
    result = vec6f(0.0)
    for row in range(6):
        diagonal = weight[row, row]
        result[row] += diagonal * value[row]
        for col in range(row + 1, 6):
            entry = weight[row, col]
            result[row] += entry * value[col]
            result[col] += entry * value[row]
    return result


@wp.func
def is_rotational_structural_row(jacobian_a: vec6f, jacobian_b: vec6f) -> bool:
    """Return whether a structural row constrains a rotation [rad] rather than a translation [m].

    Kamino's rotational constraint rows act on the angular twists alone, so their Jacobians have
    no linear part on either body.
    """
    linear_a = wp.vec3f(jacobian_a[0], jacobian_a[1], jacobian_a[2])
    linear_b = wp.vec3f(jacobian_b[0], jacobian_b[1], jacobian_b[2])
    return wp.length_sq(linear_a) + wp.length_sq(linear_b) == 0.0


@wp.func
def compute_structural_correction(
    residual: wp.float32, dead_zone: wp.float32, max_correction: wp.float32
) -> wp.float32:
    """Return the part of a joint residual corrected in one step.

    The part of the residual beyond ``dead_zone`` is corrected, by at most ``max_correction``.
    """
    return wp.sign(residual) * wp.clamp(wp.abs(residual) - dead_zone, 0.0, max_correction)


@wp.func
def normalize_symmetric_matrix(matrix: Any, size: wp.int32):
    """Scale a symmetric float32 matrix to a unit max norm and flush its entries far below that norm to zero.

    The normalized matrix suits :func:`warp.fem.linalg.symmetric_eigenvalues_qr`, whose eigenvalues
    multiplied by the scale are those of ``matrix``; see :data:`_EIGENVALUE_FLUSH_FRACTION`.

    Args:
        matrix: Symmetric ``size x size`` float32 matrix.
        size: Dimension of ``matrix``.

    Returns:
        The normalized matrix and its max-norm scale. A zero or non-finite scale returns ``matrix`` itself.
    """
    scale = wp.float32(0.0)
    for row in range(size):
        for col in range(size):
            scale = wp.max(scale, wp.abs(matrix[row, col]))
    if not wp.isfinite(scale) or scale <= 0.0:
        return matrix, scale
    normalized = matrix / scale
    for row in range(size):
        for col in range(size):
            if wp.abs(normalized[row, col]) <= _EIGENVALUE_FLUSH_FRACTION:
                normalized[row, col] = 0.0
    return normalized, scale


@wp.func
def _make_spatial_mass_matrix(mass: wp.float32, inertia_world: wp.mat33f) -> mat66f:
    """Construct a linear-first spatial mass matrix about the center of mass.

    Args:
        mass: Body mass [kg].
        inertia_world: World-space inertia about the center of mass [kg m^2].

    Returns:
        ``diag(mass I_3, inertia_world)``.
    """
    matrix = mat66f(0.0)
    for index in range(3):
        matrix[index, index] = mass
    for row in range(3):
        for col in range(3):
            matrix[row + 3, col + 3] = inertia_world[row, col]
    return matrix


@wp.func
def compute_body_explicit_wrench(
    mass: wp.float32,
    inertia_world: wp.mat33f,
    velocity_previous: vec6f,
    external_wrench: vec6f,
    actuation_wrench: vec6f,
    gravity: wp.vec3f,
    time_step: wp.float32,
) -> vec6f:
    """Combine explicit body forces in world coordinates.

    The gyroscopic torque is Kamino's :func:`compute_gyroscopic_torque`, which preserves the magnitude of
    the angular momentum over the time step, as the Kamino integrators do.

    Args:
        mass: Body mass [kg].
        inertia_world: World-space inertia about the center of mass [kg m^2].
        velocity_previous: Begin-of-step linear-first body twist [m/s, rad/s].
        external_wrench: External body wrench excluding gravity [N, N m].
        actuation_wrench: Joint-actuation body wrench [N, N m].
        gravity: World-space gravitational acceleration [m/s^2].
        time_step: Time step [s].

    Returns:
        Explicit force and torque, including gravity and the gyroscopic torque [N, N m].
    """
    result = external_wrench + actuation_wrench
    for index in range(3):
        result[index] += mass * gravity[index]

    angular_velocity = wp.vec3f(velocity_previous[3], velocity_previous[4], velocity_previous[5])
    gyroscopic_torque = compute_gyroscopic_torque(time_step, inertia_world, angular_velocity)
    for index in range(3):
        result[index + 3] += gyroscopic_torque[index]
    return result


@wp.func
def _compute_body_inertial_system(
    mass: wp.float32,
    inertia_world: wp.mat33f,
    velocity_previous: vec6f,
    force_explicit: vec6f,
    time_step: wp.float32,
) -> _PrimalRowContribution:
    """Compute the inertial operator and explicit-force right-hand side.

    Args:
        mass: Body mass [kg].
        inertia_world: World-space inertia about the center of mass [kg m^2].
        velocity_previous: Begin-of-step linear-first body twist [m/s, rad/s].
        force_explicit: Explicit body wrench [N, N m].
        time_step: Time step [s].

    Returns:
        The mass matrix and ``M v_previous + h force_explicit``.
    """
    result = _PrimalRowContribution()
    result.matrix = _make_spatial_mass_matrix(mass, inertia_world)
    result.right_hand_side = result.matrix @ velocity_previous + time_step * force_explicit
    return result


@wp.func
def _compute_dynamic_joint_row(
    effective_inertia: wp.float32,
    free_velocity: wp.float32,
) -> tuple[wp.float32, wp.float32]:
    """Eliminate one implicit joint-dynamics coordinate into body space.

    The joint equation is ``effective_inertia * dq = h_joint`` with
    ``free_velocity = h_joint / effective_inertia`` and ``dq = J v``.

    Args:
        effective_inertia: Positive implicit joint inertia.
        free_velocity: Implicit joint free velocity.

    Returns:
        The coefficients of the matrix term ``effective_inertia J^T J`` and of the
        right-hand-side term ``effective_inertia free_velocity J^T``.
    """
    return effective_inertia, effective_inertia * free_velocity


@wp.func
def _compute_augmented_joint_row(
    residual: wp.float32,
    penalty: wp.float32,
    time_step: wp.float32,
    linearization_velocity: wp.float32,
) -> tuple[wp.float32, wp.float32]:
    """Linearize the augmented Lagrangian penalty of one structural joint row.

    For ``C(q(v)) ~= residual + h J (v - v_k)``, this returns the coefficients of the frozen terms in

    ``(M + h^2 penalty J^T J) v``
    ``= f - h penalty residual J^T + h^2 penalty J^T J v_k + h J^T lambda``.

    The multiplier term ``h J^T lambda`` changes every iteration and is added to the
    candidate right-hand side.

    Args:
        residual: Current joint position residual [m or rad].
        penalty: Positive augmented penalty [N/m or N m/rad].
        time_step: Time step [s].
        linearization_velocity: Row velocity ``J v_k`` at the linearization
            twist [m/s or rad/s]. Pass zero for the first linearly implicit
            assembly about the begin-step pose.

    Returns:
        The coefficients of the matrix term ``h^2 penalty J^T J`` and of the right-hand-side term
        ``(-h penalty residual + h^2 penalty J v_k) J^T``.
    """
    return time_step * time_step * penalty, (
        -time_step * penalty * residual + time_step * time_step * penalty * linearization_velocity
    )


@wp.func
def _matrix_index(
    matrix_offset: wp.int32,
    dimension: wp.int32,
    row: wp.int32,
    col: wp.int32,
) -> wp.int32:
    return matrix_offset + dimension * row + col


@wp.func
def _body_matrix_rows(
    vector_offset: wp.int32,
    inverse_permutation: wp.array[wp.int32],
    body: wp.int32,
    dimension: wp.int32,
) -> vec6i:
    """Return the matrix rows of the six unknowns of a local body, in the row order of the factorization
    when an inverse permutation is given. Single-body blocks keep their packed rows."""
    rows = vec6i()
    for axis in range(6):
        row = 6 * body + axis
        if inverse_permutation and dimension > 6:
            row = inverse_permutation[vector_offset + row]
        rows[axis] = row
    return rows


@wp.func
def _body_matrix_row(
    vector_offset: wp.int32,
    inverse_permutation: wp.array[wp.int32],
    body: wp.int32,
    axis: wp.int32,
    dimension: wp.int32,
) -> wp.int32:
    """Return the matrix row of one unknown of a local body, see :func:`_body_matrix_rows`."""
    row = 6 * body + axis
    if inverse_permutation and dimension > 6:
        row = inverse_permutation[vector_offset + row]
    return row


@wp.func
def _select_axis(value: vec6f, axis: wp.int32) -> wp.float32:
    """Return entry ``axis`` of ``value`` by selection, which keeps ``value`` in registers for a per-thread ``axis``."""
    result = value[0]
    for index in range(1, 6):
        result = wp.where(axis == index, value[index], result)
    return result


@wp.func
def unpack_body_solution(
    bid: wp.int32,
    body_vector_index: wp.array[wp.int32],
    packed_solution: wp.array[wp.float32],
    prescribed_twist: wp.array[vec6f],
) -> vec6f:
    """Return the twist of a body from the packed solution, or its prescribed twist outside the system."""
    source_offset = body_vector_index[bid]
    value = vec6f(0.0)
    if prescribed_twist:
        value = prescribed_twist[bid]
    if source_offset >= 0:
        for index in range(6):
            value[index] = packed_solution[source_offset + index]
    return value


###
# Kernels
###


@wp.kernel
def _assemble_body_inertial_systems(
    # Inputs:
    body_world: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    body_edge_offset: wp.array[wp.int32],
    body_edge_neighbor: wp.array[wp.int32],
    model_bodies_m_i: wp.array[wp.float32],
    data_bodies_I_i: wp.array[wp.mat33f],
    velocity_previous: wp.array[vec6f],
    data_bodies_w_e_i: wp.array[vec6f],
    data_bodies_w_a_i: wp.array[vec6f],
    model_gravity_vector: wp.array[wp.vec3f],
    model_time_dt: wp.array[wp.float32],
    inverse_permutation: wp.array[wp.int32],
    # Outputs:
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    """Write the inertial terms of each body and clear the off-diagonal blocks of its body edges.

    Six threads assemble each body, one per unknown. Thread ``axis`` writes column ``axis`` of the diagonal
    block of its body and entry ``axis`` of the right-hand side, and clears column ``axis`` of both
    off-diagonal blocks of each body edge listed for its body. These blocks hold every matrix entry that
    the joint assembly adds to, and the other matrix entries hold zeros.
    """
    bid, axis = wp.tid()
    block = body_block[bid]
    if block < 0:
        return
    wid = body_world[bid]
    dt = model_time_dt[wid]
    local_body = body_local[bid]
    dimension = dimensions[block]
    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    force_explicit = compute_body_explicit_wrench(
        model_bodies_m_i[bid],
        data_bodies_I_i[bid],
        velocity_previous[bid],
        data_bodies_w_e_i[bid],
        data_bodies_w_a_i[bid],
        model_gravity_vector[wid],
        dt,
    )
    contribution = _compute_body_inertial_system(
        model_bodies_m_i[bid], data_bodies_I_i[bid], velocity_previous[bid], force_explicit, dt
    )
    right_hand_side[vector_offset + 6 * local_body + axis] = _select_axis(contribution.right_hand_side, axis)

    # The six threads of a body write contiguous columns of each matrix row
    rows = _body_matrix_rows(vector_offset, inverse_permutation, local_body, dimension)
    col = _body_matrix_row(vector_offset, inverse_permutation, local_body, axis, dimension)
    for row in range(6):
        value = _select_axis(contribution.matrix[row], axis)
        matrix[_matrix_index(matrix_offset, dimension, rows[row], col)] = value
    for edge in range(body_edge_offset[bid], body_edge_offset[bid + 1]):
        neighbor = body_local[body_edge_neighbor[edge]]
        neighbor_rows = _body_matrix_rows(vector_offset, inverse_permutation, neighbor, dimension)
        neighbor_col = _body_matrix_row(vector_offset, inverse_permutation, neighbor, axis, dimension)
        for row in range(6):
            matrix[_matrix_index(matrix_offset, dimension, rows[row], neighbor_col)] = 0.0
            matrix[_matrix_index(matrix_offset, dimension, neighbor_rows[row], col)] = 0.0


@wp.kernel
def _assemble_joint_rows(
    # Inputs:
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    joint_world: wp.array[wp.int32],
    joint_body_a: wp.array[wp.int32],
    joint_body_b: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    prescribed_twist: wp.array[vec6f],
    model_time_dt: wp.array[wp.float32],
    dynamic_offset: wp.array[wp.int32],
    dynamic_count: wp.array[wp.int32],
    dynamic_jacobian_a: wp.array[vec6f],
    dynamic_jacobian_b: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    dynamic_free_velocity: wp.array[wp.float32],
    structural_offset: wp.array[wp.int32],
    structural_count: wp.array[wp.int32],
    structural_jacobian_a: wp.array[vec6f],
    structural_jacobian_b: wp.array[vec6f],
    structural_residual: wp.array[wp.float32],
    structural_effective_mass: wp.array[wp.float32],
    structural_reaction: wp.array[wp.float32],
    joint_penalty_scale: wp.array[wp.float32],
    joint_max_correction: wp.float32,
    joint_translation_dead_zone: wp.float32,
    joint_rotation_dead_zone: wp.float32,
    inverse_permutation: wp.array[wp.int32],
    # Outputs:
    structural_penalty: wp.array[wp.float32],
    structural_body_impulse: wp.array[vec6f],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    """Assemble the dynamic rows and the frozen penalty terms of the structural rows of each joint.

    Twelve threads assemble each joint, one per unknown of its two bodies. The thread of unknown ``axis``
    of a body sums, over the rows of the joint, row ``axis`` of the diagonal block of its body and of the
    block that couples its body to the other body. By symmetry, it adds them once to column ``axis`` of
    the diagonal block and of the block that couples the other body to its body, so that the six threads
    of a body add contiguous entries of each matrix row. Both sides evaluate each product of two Jacobian
    entries in the same order, so the two coupling blocks are exact transposes. With a
    ``joint_penalty_scale``, the first thread of each joint writes the penalties of its structural rows,
    zero for the rows left out, and the penalized rows add their multiplier impulse ``h J^T lambda`` to
    ``structural_body_impulse``.
    """
    jid, unknown = wp.tid()
    side = unknown // 6
    axis = unknown - 6 * side
    bid_a = joint_body_a[jid]
    bid_b = joint_body_b[jid]
    body_count = body_block.shape[0]
    bid_a_local = wp.int32(-1)
    bid_b_local = wp.int32(-1)
    block = wp.int32(-1)
    if (bid_a >= 0 or bid_b >= 0) and bid_a < body_count and bid_b < body_count:
        if bid_a >= 0:
            bid_a_local = body_local[bid_a]
        if bid_b >= 0:
            bid_b_local = body_local[bid_b]
        if bid_a_local >= 0 or bid_b_local >= 0:
            block = body_block[bid_a] if bid_a_local >= 0 else body_block[bid_b]
            if block >= dimensions.shape[0] or (bid_b_local >= 0 and body_block[bid_b] != block):
                block = -1

    structural_start = structural_offset[jid]
    structural_end = structural_start + structural_count[jid]
    # The rows take the scale of the component of their dynamic bodies
    wid = joint_world[jid]
    dt = wp.float32(0.0)
    world_penalty_scale = wp.float32(0.0)
    if block >= 0 and wid >= 0:
        dt = model_time_dt[wid]
        if joint_penalty_scale and dt > 0.0:
            world_penalty_scale = joint_penalty_scale[wp.where(bid_a_local >= 0, bid_a, bid_b)]
    if joint_penalty_scale and world_penalty_scale <= 0.0 and unknown == 0:
        for row in range(structural_start, structural_end):
            structural_penalty[row] = 0.0
    if block < 0:
        return

    # The rows are linearized about zero dynamic twists and the prescribed twists of the other bodies.
    a_twist = vec6f(0.0)
    b_twist = vec6f(0.0)
    if prescribed_twist and bid_a >= 0 and bid_a_local < 0:
        a_twist = prescribed_twist[bid_a]
    if prescribed_twist and bid_b >= 0 and bid_b_local < 0:
        b_twist = prescribed_twist[bid_b]

    # The thread assembles one unknown of the body of its side, A or B
    on_a = side == 0
    bid = wp.where(on_a, bid_a, bid_b)
    bid_local = wp.where(on_a, bid_a_local, bid_b_local)
    other_local = wp.where(on_a, bid_b_local, bid_a_local)
    body_row = vec6f(0.0)
    cross_row = vec6f(0.0)
    right_hand_side_entry = wp.float32(0.0)
    row_count = wp.int32(0)

    dynamic_start = dynamic_offset[jid]
    for row in range(dynamic_start, dynamic_start + dynamic_count[jid]):
        inertia = dynamic_effective_inertia[row]
        if inertia <= 0.0:
            continue
        a_jacobian = dynamic_jacobian_a[row]
        b_jacobian = dynamic_jacobian_b[row]
        velocity = dynamic_free_velocity[row] - wp.dot(a_jacobian, a_twist) - wp.dot(b_jacobian, b_twist)
        coefficient, right_hand_side_coefficient = _compute_dynamic_joint_row(inertia, velocity)
        jacobian = wp.where(on_a, a_jacobian, b_jacobian)
        jacobian_axis = _select_axis(jacobian, axis)
        body_row += coefficient * (jacobian_axis * jacobian)
        cross_row += coefficient * (jacobian_axis * wp.where(on_a, b_jacobian, a_jacobian))
        right_hand_side_entry += right_hand_side_coefficient * jacobian_axis
        row_count += 1

    if world_penalty_scale > 0.0:
        impulse = wp.float32(0.0)
        for row in range(structural_start, structural_end):
            if structural_effective_mass[row] <= 0.0:
                if unknown == 0:
                    structural_penalty[row] = 0.0
                continue
            row_penalty = world_penalty_scale * structural_effective_mass[row] / (dt * dt)
            if unknown == 0:
                structural_penalty[row] = row_penalty
            a_jacobian = structural_jacobian_a[row]
            b_jacobian = structural_jacobian_b[row]
            jacobian = wp.where(on_a, a_jacobian, b_jacobian)
            jacobian_axis = _select_axis(jacobian, axis)
            if row_penalty > 0.0:
                impulse += structural_reaction[row] * jacobian_axis
            # A small violation is left to average out, and a large one is corrected by at most
            # joint_max_correction per step
            dead_zone = joint_translation_dead_zone
            if is_rotational_structural_row(a_jacobian, b_jacobian):
                dead_zone = joint_rotation_dead_zone
            row_residual = compute_structural_correction(structural_residual[row], dead_zone, joint_max_correction)
            linearization_velocity = -wp.dot(a_jacobian, a_twist) - wp.dot(b_jacobian, b_twist)
            coefficient, right_hand_side_coefficient = _compute_augmented_joint_row(
                row_residual, row_penalty, dt, linearization_velocity
            )
            body_row += coefficient * (jacobian_axis * jacobian)
            cross_row += coefficient * (jacobian_axis * wp.where(on_a, b_jacobian, a_jacobian))
            right_hand_side_entry += right_hand_side_coefficient * jacobian_axis
            row_count += 1
        if bid >= 0:
            # Joints that share a body add to its impulse concurrently, so the sum is atomic on every Warp version
            body_impulse = vec6f(0.0)
            body_impulse[axis] = dt * impulse
            wp.atomic_add(structural_body_impulse, bid, body_impulse)

    if row_count == 0 or bid_local < 0:
        return
    dimension = dimensions[block]
    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    wp.atomic_add(right_hand_side, vector_offset + 6 * bid_local + axis, right_hand_side_entry)
    col = _body_matrix_row(vector_offset, inverse_permutation, bid_local, axis, dimension)
    rows = _body_matrix_rows(vector_offset, inverse_permutation, bid_local, dimension)
    for row in range(6):
        wp.atomic_add(matrix, _matrix_index(matrix_offset, dimension, rows[row], col), body_row[row])
    if other_local >= 0:
        other_rows = _body_matrix_rows(vector_offset, inverse_permutation, other_local, dimension)
        for row in range(6):
            wp.atomic_add(matrix, _matrix_index(matrix_offset, dimension, other_rows[row], col), cross_row[row])


@wp.func
def _compute_point_geometric_stiffness(force: wp.vec3f, arm: wp.vec3f) -> wp.mat33f:
    """Return the rotational geometric stiffness of the tension of a point force.

    A force ``force`` along the arm ``arm`` from the center of mass has the rotational stiffness
    ``(force . arm) (I - arm arm^T / |arm|^2)``, the stiffness of a pendulum hanging from the point. The
    component of the force along the arm sets this stiffness, which is positive semidefinite when the
    force pulls the point away from the center of mass.
    """
    tension = wp.max(wp.dot(force, arm), 0.0)
    length_sq = wp.dot(arm, arm)
    if tension <= 0.0 or length_sq <= 0.0:
        return wp.mat33f(0.0)
    return tension * (wp.identity(n=3, dtype=wp.float32) - wp.outer(arm, arm) / length_sq)


@wp.kernel
def _assemble_rod_geometric_stiffness(
    # Inputs:
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    joint_world: wp.array[wp.int32],
    joint_dof_type: wp.array[wp.int32],
    joint_dofs_offset: wp.array[wp.int32],
    joint_coords_offset: wp.array[wp.int32],
    joint_body_a: wp.array[wp.int32],
    joint_body_b: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_q_ref: wp.array[wp.float32],
    joint_pose: wp.array[wp.transformf],
    body_pose: wp.array[wp.transformf],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    prescribed_twist: wp.array[vec6f],
    model_time_dt: wp.array[wp.float32],
    dynamic_offset: wp.array[wp.int32],
    dynamic_count: wp.array[wp.int32],
    dynamic_dof: wp.array[wp.int32],
    dynamic_jacobian_a: wp.array[vec6f],
    dynamic_jacobian_b: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    dynamic_free_velocity: wp.array[wp.float32],
    inverse_permutation: wp.array[wp.int32],
    # Outputs:
    matrix: wp.array[wp.float32],
):
    """Add the rotational geometric stiffness of the stretch and shear rows of each stretched rod joint.

    The stretch and shear rows of a rod act at the joint point, and their elastic force at the beginning of
    the step ``effective_inertia * free_velocity / h``, taken at the prescribed twists of the bodies outside
    the system as in :func:`_assemble_joint_rows`, turns with the arm of each body. When the rod is
    longer than its rest length, each dynamic body adds ``h^2`` times the stiffness of the tension of this
    force to its angular block, see :func:`_compute_point_geometric_stiffness`. The right-hand side holds
    the elastic force already, so the matrix term completes the linearization of the rows about the
    beginning of the step.
    """
    jid, side = wp.tid()
    if joint_dof_type[jid] != JointDoFType.ROD:
        return
    stretch_coord = joint_coords_offset[jid] + 2
    if joint_q[stretch_coord] <= joint_q_ref[stretch_coord]:
        return
    bid_a = joint_body_a[jid]
    bid_b = joint_body_b[jid]
    bid = wp.where(side == 0, bid_a, bid_b)
    if bid < 0 or bid >= body_block.shape[0]:
        return
    block = body_block[bid]
    bid_local = body_local[bid]
    wid = joint_world[jid]
    if block < 0 or bid_local < 0 or wid < 0 or block >= dimensions.shape[0]:
        return
    dt = model_time_dt[wid]
    if dt <= 0.0:
        return

    # The rows are linearized about zero dynamic twists and the prescribed twists of the other bodies
    a_twist = vec6f(0.0)
    b_twist = vec6f(0.0)
    if prescribed_twist and bid_a >= 0 and body_local[bid_a] < 0:
        a_twist = prescribed_twist[bid_a]
    if prescribed_twist and bid_b >= 0 and body_local[bid_b] < 0:
        b_twist = prescribed_twist[bid_b]

    # Sum the elastic force of the stretch and shear rows on the body
    force = wp.vec3f(0.0)
    dofs_offset = joint_dofs_offset[jid]
    dynamic_start = dynamic_offset[jid]
    for row in range(dynamic_start, dynamic_start + dynamic_count[jid]):
        if dynamic_dof[row] - dofs_offset >= 3:
            continue
        a_jacobian = dynamic_jacobian_a[row]
        b_jacobian = dynamic_jacobian_b[row]
        velocity = dynamic_free_velocity[row] - wp.dot(a_jacobian, a_twist) - wp.dot(b_jacobian, b_twist)
        jacobian = wp.where(side == 0, a_jacobian, b_jacobian)
        row_force = dynamic_effective_inertia[row] * velocity / dt
        force += row_force * wp.vec3f(jacobian[0], jacobian[1], jacobian[2])

    arm = wp.transform_get_translation(joint_pose[jid]) - wp.transform_get_translation(body_pose[bid])
    stiffness = dt * dt * _compute_point_geometric_stiffness(force, arm)

    dimension = dimensions[block]
    matrix_offset = matrix_offsets[block]
    rows = _body_matrix_rows(vector_offsets[block], inverse_permutation, bid_local, dimension)
    for row in range(3):
        for col in range(3):
            wp.atomic_add(
                matrix, _matrix_index(matrix_offset, dimension, rows[3 + row], rows[3 + col]), stiffness[row, col]
            )


@wp.func
def _read_body_block(
    matrix: wp.array[wp.float32], offset: wp.int32, dimension: wp.int32, rows: vec6i, cols: vec6i
) -> mat66f:
    """Read the ``6 x 6`` block of two bodies from a dense block matrix, given their matrix rows."""
    block = mat66f(0.0)
    for row in range(6):
        for col in range(6):
            block[row, col] = matrix[_matrix_index(offset, dimension, rows[row], cols[col])]
    return block


@wp.func
def _store_lower_mat66(array: wp.array2d[wp.float32], index: wp.int32, matrix: mat66f):
    """Store the lower triangle of a ``6 x 6`` matrix in column ``index`` of an array of ``21`` rows.

    Consecutive bodies hold consecutive entries of each row, so the threads of consecutive bodies access
    contiguous memory.
    """
    for row in range(6):
        for col in range(6):
            if col <= row:
                array[row * (row + 1) // 2 + col, index] = matrix[row, col]


@wp.func
def _load_lower_mat66(array: wp.array2d[wp.float32], index: wp.int32) -> mat66f:
    """Load a lower triangular ``6 x 6`` matrix stored by :func:`_store_lower_mat66`."""
    matrix = mat66f(0.0)
    for row in range(6):
        for col in range(6):
            if col <= row:
                matrix[row, col] = array[row * (row + 1) // 2 + col, index]
    return matrix


@wp.func
def _load_symmetric_mat66(array: wp.array2d[wp.float32], index: wp.int32) -> mat66f:
    """Load a symmetric ``6 x 6`` matrix whose lower triangle is stored by :func:`_store_lower_mat66`."""
    matrix = mat66f(0.0)
    for row in range(6):
        for col in range(6):
            if col <= row:
                value = array[row * (row + 1) // 2 + col, index]
                matrix[row, col] = value
                matrix[col, row] = value
    return matrix


@wp.func
def _joint_row_side_block(
    kind: wp.int32,
    row: wp.int32,
    side: wp.int32,
    structural_jacobian_a: wp.array[vec6f],
    structural_jacobian_b: wp.array[vec6f],
    structural_penalty: wp.array[wp.float32],
    structural_world: wp.array[wp.int32],
    dynamic_jacobian_a: wp.array[vec6f],
    dynamic_jacobian_b: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    model_time_dt: wp.array[wp.float32],
) -> mat66f:
    """Return the diagonal-block term ``c J^T J`` of one joint row on one of its bodies."""
    if kind == 0:
        dt = model_time_dt[structural_world[row]]
        jacobian = wp.where(side == 0, structural_jacobian_a[row], structural_jacobian_b[row])
        return (dt * dt * structural_penalty[row]) * wp.outer(jacobian, jacobian)
    jacobian = wp.where(side == 0, dynamic_jacobian_a[row], dynamic_jacobian_b[row])
    return dynamic_effective_inertia[row] * wp.outer(jacobian, jacobian)


@wp.func
def _cholesky_mat66(matrix: mat66f) -> tuple[mat66f, wp.bool]:
    """Return the lower Cholesky factor of a symmetric ``6 x 6`` matrix and whether it is positive definite."""
    lower = mat66f(0.0)
    for col in range(6):
        diagonal = matrix[col, col]
        for inner in range(col):
            diagonal -= lower[col, inner] * lower[col, inner]
        if not (diagonal > 0.0) or not wp.isfinite(diagonal):
            return lower, False
        root = wp.sqrt(diagonal)
        lower[col, col] = root
        inverse_root = 1.0 / root
        for row in range(col + 1, 6):
            value = matrix[row, col]
            for inner in range(col):
                value -= lower[row, inner] * lower[col, inner]
            lower[row, col] = value * inverse_root
    return lower, True


@wp.func
def _solve_lower_vec6(lower: mat66f, rhs: vec6f) -> vec6f:
    """Solve the lower triangular ``6 x 6`` system ``L x = b`` by forward substitution."""
    solved = vec6f(0.0)
    for axis in range(6):
        value = rhs[axis]
        for inner in range(axis):
            value -= lower[axis, inner] * solved[inner]
        solved[axis] = value / lower[axis, axis]
    return solved


@wp.func
def _solve_lower_mat66(lower: mat66f, rhs: mat66f) -> mat66f:
    """Solve the lower triangular ``6 x 6`` system ``L X = B`` by forward substitution."""
    solved = mat66f(0.0)
    for axis in range(6):
        for col in range(6):
            value = rhs[axis, col]
            for inner in range(axis):
                value -= lower[axis, inner] * solved[inner, col]
            solved[axis, col] = value / lower[axis, axis]
    return solved


@wp.kernel
def _compute_free_body_blocks(
    # Inputs:
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    inverse_permutation: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    incidence_offset: wp.array[wp.int32],
    incidence_kind: wp.array[wp.int32],
    incidence_row: wp.array[wp.int32],
    incidence_side: wp.array[wp.int32],
    structural_jacobian_a: wp.array[vec6f],
    structural_jacobian_b: wp.array[vec6f],
    structural_penalty: wp.array[wp.float32],
    structural_world: wp.array[wp.int32],
    dynamic_jacobian_a: wp.array[vec6f],
    dynamic_jacobian_b: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    model_time_dt: wp.array[wp.float32],
    # Outputs:
    diagonal_block: wp.array2d[wp.float32],
    diagonal_lower: wp.array2d[wp.float32],
    free_block: wp.array2d[wp.float32],
    free_lower: wp.array2d[wp.float32],
):
    """Return the diagonal block of each body without its joint rows to other dynamic bodies, and its Cholesky factor.

    The incidence lists only hold these rows; the rows to static bodies stay in the block. The factor is zero
    when the block is not positive definite, e.g. for a massless body. ``diagonal_block`` and
    ``diagonal_lower`` receive the full diagonal block and its Cholesky factor, or zero, which the pinned
    patches of the joint neighbors share. The outputs hold the lower triangles of symmetric and lower
    triangular blocks, see :func:`_store_lower_mat66`.
    """
    bid = wp.tid()
    block = body_block[bid]
    if block < 0:
        return
    rows = _body_matrix_rows(vector_offsets[block], inverse_permutation, body_local[bid], dimensions[block])
    result = _read_body_block(matrix, matrix_offsets[block], dimensions[block], rows, rows)
    diagonal = 0.5 * (result + wp.transpose(result))
    _store_lower_mat66(diagonal_block, bid, diagonal)
    lower, valid = _cholesky_mat66(diagonal)
    if not valid:
        lower = mat66f(0.0)
    _store_lower_mat66(diagonal_lower, bid, lower)
    for incident in range(incidence_offset[bid], incidence_offset[bid + 1]):
        result -= _joint_row_side_block(
            incidence_kind[incident],
            incidence_row[incident],
            incidence_side[incident],
            structural_jacobian_a,
            structural_jacobian_b,
            structural_penalty,
            structural_world,
            dynamic_jacobian_a,
            dynamic_jacobian_b,
            dynamic_effective_inertia,
            model_time_dt,
        )
    symmetric = 0.5 * (result + wp.transpose(result))
    _store_lower_mat66(free_block, bid, symmetric)
    lower, valid = _cholesky_mat66(symmetric)
    if not valid:
        lower = mat66f(0.0)
    _store_lower_mat66(free_lower, bid, lower)


@wp.func
def _joint_row(
    kind: wp.int32,
    row: wp.int32,
    structural_jacobian_a: wp.array[vec6f],
    structural_jacobian_b: wp.array[vec6f],
    structural_penalty: wp.array[wp.float32],
    structural_world: wp.array[wp.int32],
    dynamic_jacobian_a: wp.array[vec6f],
    dynamic_jacobian_b: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    model_time_dt: wp.array[wp.float32],
) -> tuple[wp.float32, vec6f, vec6f]:
    """Return the matrix coefficient and the Jacobians on both bodies of one structural or dynamic joint row."""
    if kind == 0:
        dt = model_time_dt[structural_world[row]]
        return dt * dt * structural_penalty[row], structural_jacobian_a[row], structural_jacobian_b[row]
    return dynamic_effective_inertia[row], dynamic_jacobian_a[row], dynamic_jacobian_b[row]


@wp.func
def _finish_body_weight(
    bid: wp.int32, schur: mat66f, weight: wp.array[mat66f], inverse_weight: wp.array[mat66f]
) -> mat66f:
    """Store and return the symmetrized weight, with its inverse, or zero when it is not positive definite."""
    symmetric = 0.5 * (schur + wp.transpose(schur))
    inverse, ok = _invert_spd_mat66(symmetric)
    if not ok:
        weight[bid] = mat66f(0.0)
        inverse_weight[bid] = mat66f(0.0)
        return mat66f(0.0)
    weight[bid] = symmetric
    inverse_weight[bid] = inverse
    return symmetric


@wp.func
def _free_patch_schur(
    bid: wp.int32,
    free_block: wp.array2d[wp.float32],
    free_lower: wp.array2d[wp.float32],
    neighbor_offset: wp.array[wp.int32],
    neighbor_body: wp.array[wp.int32],
    neighbor_row_offset: wp.array[wp.int32],
    neighbor_row_kind: wp.array[wp.int32],
    neighbor_row: wp.array[wp.int32],
    neighbor_row_side: wp.array[wp.int32],
    structural_jacobian_a: wp.array[vec6f],
    structural_jacobian_b: wp.array[vec6f],
    structural_penalty: wp.array[wp.float32],
    structural_world: wp.array[wp.int32],
    dynamic_jacobian_a: wp.array[vec6f],
    dynamic_jacobian_b: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    model_time_dt: wp.array[wp.float32],
) -> mat66f:
    """Return the Schur complement ``S_free`` of body ``i`` in its free patch, in Woodbury form.

    In the free patch, each joint neighbor ``j`` keeps only its joint to ``i``: with ``F_j`` the block of ``j``
    without its joint rows to dynamic bodies, and the at most six rows of the joint ``J_e = [J_ei, J_ej]``
    with coefficients ``C``,

        ``S_free = F_i + sum_e J_ei^T (C^-1 + J_ej F_j^-1 J_ej^T)^-1 J_ei``,

    a sum of positive semi-definite terms that remains accurate for stiff joints and tends to the composite
    inertia of the patch. A neighbor whose ``F_j`` is not positive definite, e.g. a massless body, adds
    nothing. The result bounds ``[(A^-1)_ii]^-1`` from below.
    """
    schur = _load_symmetric_mat66(free_block, bid)
    for entry in range(neighbor_offset[bid], neighbor_offset[bid + 1]):
        neighbor = neighbor_body[entry]
        if not (free_lower[0, neighbor] > 0.0):
            continue
        lower = _load_lower_mat66(free_lower, neighbor)
        # Joint rows on the body and rows of L_j^-1 J_ej^T in fixed slots; an unused slot is a zero row with a
        # unit diagonal in K = C^-1 + J_ej F_j^-1 J_ej^T
        row_begin = neighbor_row_offset[entry]
        row_count = neighbor_row_offset[entry + 1] - row_begin
        body_rows = mat66f(0.0)
        forward_rows = mat66f(0.0)
        joint_matrix = mat66f(0.0)
        for slot in range(6):
            joint_matrix[slot, slot] = 1.0
            if slot < row_count:
                row_entry = row_begin + slot
                coefficient, jacobian_a, jacobian_b = _joint_row(
                    neighbor_row_kind[row_entry],
                    neighbor_row[row_entry],
                    structural_jacobian_a,
                    structural_jacobian_b,
                    structural_penalty,
                    structural_world,
                    dynamic_jacobian_a,
                    dynamic_jacobian_b,
                    dynamic_effective_inertia,
                    model_time_dt,
                )
                if coefficient > 0.0:
                    neighbor_side = neighbor_row_side[row_entry]
                    neighbor_jacobian = wp.where(neighbor_side == 0, jacobian_a, jacobian_b)
                    forward_rows[slot] = _solve_lower_vec6(lower, neighbor_jacobian)
                    body_rows[slot] = wp.where(neighbor_side == 0, jacobian_b, jacobian_a)
                    joint_matrix[slot, slot] = 1.0 / coefficient
        # The factorization reads the lower triangle of K only
        for row in range(6):
            for col in range(6):
                if col <= row:
                    value = joint_matrix[row, col]
                    for axis in range(6):
                        value += forward_rows[row, axis] * forward_rows[col, axis]
                    joint_matrix[row, col] = value
        joint_lower, joint_valid = _cholesky_mat66(joint_matrix)
        if not joint_valid:
            continue
        # Z = P^-1 J_ei with K = P P^T, then J_ei^T K^-1 J_ei = Z^T Z
        solved = _solve_lower_mat66(joint_lower, body_rows)
        # Z^T Z is symmetric, so each pair of columns is evaluated once
        for row in range(6):
            for col in range(6):
                if col <= row:
                    value = wp.float32(0.0)
                    for axis in range(6):
                        value += solved[axis, row] * solved[axis, col]
                    schur[row, col] += value
                    if col < row:
                        schur[col, row] += value
    return schur


@wp.func
def _free_patch_schur_subtraction(
    bid: wp.int32,
    free_block: wp.array2d[wp.float32],
    neighbor_offset: wp.array[wp.int32],
    neighbor_body: wp.array[wp.int32],
    neighbor_row_offset: wp.array[wp.int32],
    neighbor_row_kind: wp.array[wp.int32],
    neighbor_row: wp.array[wp.int32],
    neighbor_row_side: wp.array[wp.int32],
    structural_jacobian_a: wp.array[vec6f],
    structural_jacobian_b: wp.array[vec6f],
    structural_penalty: wp.array[wp.float32],
    structural_world: wp.array[wp.int32],
    dynamic_jacobian_a: wp.array[vec6f],
    dynamic_jacobian_b: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    model_time_dt: wp.array[wp.float32],
) -> mat66f:
    """Return ``S_free = F_i + sum_e (R_ei - A_ij (F_j + R_ej)^-1 A_ji)``, the free-patch Schur complement of a body.

    Equivalent to :func:`_free_patch_schur` for joints with more than six rows between two bodies.
    """
    schur = _load_symmetric_mat66(free_block, bid)
    for entry in range(neighbor_offset[bid], neighbor_offset[bid + 1]):
        neighbor_matrix = _load_symmetric_mat66(free_block, neighbor_body[entry])
        coupling = mat66f(0.0)
        for row_entry in range(neighbor_row_offset[entry], neighbor_row_offset[entry + 1]):
            coefficient, jacobian_a, jacobian_b = _joint_row(
                neighbor_row_kind[row_entry],
                neighbor_row[row_entry],
                structural_jacobian_a,
                structural_jacobian_b,
                structural_penalty,
                structural_world,
                dynamic_jacobian_a,
                dynamic_jacobian_b,
                dynamic_effective_inertia,
                model_time_dt,
            )
            neighbor_side = neighbor_row_side[row_entry]
            neighbor_jacobian = wp.where(neighbor_side == 0, jacobian_a, jacobian_b)
            body_jacobian = wp.where(neighbor_side == 0, jacobian_b, jacobian_a)
            scaled = coefficient * neighbor_jacobian
            neighbor_matrix += wp.outer(scaled, neighbor_jacobian)
            coupling += wp.outer(scaled, body_jacobian)
            schur += wp.outer(coefficient * body_jacobian, body_jacobian)
        lower, valid = _cholesky_mat66(neighbor_matrix)
        if not valid:
            continue
        solved = _solve_lower_mat66(lower, coupling)
        schur -= wp.transpose(solved) @ solved
    return schur


@wp.func
def _geometric_mean_mat66(
    lower_bound: mat66f, upper_bound: mat66f, beta: wp.float32, sigma: wp.float32
) -> tuple[mat66f, mat66f, wp.bool]:
    """Return the matrix geometric mean ``A # B = L (L^-1 B L^-T)^1/2 L^T`` of two symmetric positive-definite
    ``6 x 6`` matrices, with ``A = L L^T``, its inverse, and whether it is defined.

    ``A`` is the lower bound, so the eigenvalues of ``L^-1 B L^-T`` are clamped at one, which keeps the
    mean at least ``A`` against round-off in ``B``. A positive ``beta`` clamps each eigenvalue ``r`` of the
    square root to ``max(sigma r, min(r, beta))``, which keeps the result at most ``beta A`` and at least
    ``sigma`` times the mean.
    """
    lower, valid = _cholesky_mat66(lower_bound)
    if not valid:
        return lower_bound, lower_bound, False
    # X = L^-1 B, then M = L^-1 X^T = L^-1 B L^-T
    solved = _solve_lower_mat66(lower, upper_bound)
    congruent = _solve_lower_mat66(lower, wp.transpose(solved))
    normalized, scale = normalize_symmetric_matrix(0.5 * (congruent + wp.transpose(congruent)), 6)
    if not wp.isfinite(scale) or scale <= 0.0:
        return lower_bound, lower_bound, False
    # M = P^T D P, with one eigenvector per row of P. With U = P L^T and its inverse transpose V = P L^-1,
    # the mean is U^T D^1/2 U and its inverse V^T D^-1/2 V
    eigenvalues, eigenvectors = symmetric_eigenvalues_qr(normalized, _EIGENVALUE_TOLERANCE)
    factor = eigenvectors @ wp.transpose(lower)
    inverse_factor = eigenvectors @ _invert_lower_mat66(lower)
    mean = mat66f(0.0)
    inverse = mat66f(0.0)
    for index in range(6):
        ratio = wp.sqrt(wp.max(scale * eigenvalues[index], 1.0))
        if beta > 0.0:
            ratio = wp.max(sigma * ratio, wp.min(ratio, beta))
        inverse_ratio = 1.0 / ratio
        # Each pair of entries is evaluated once, which keeps both results exactly symmetric
        for row in range(6):
            for col in range(6):
                if col <= row:
                    value = ratio * (factor[index, row] * factor[index, col])
                    inverse_value = inverse_ratio * (inverse_factor[index, row] * inverse_factor[index, col])
                    mean[row, col] += value
                    inverse[row, col] += inverse_value
                    if col < row:
                        mean[col, row] += value
                        inverse[col, row] += inverse_value
    return mean, inverse, True


@cache
def make_patch_schur_kernel(woodbury: bool):
    """Build the kernel of the free and pinned patch Schur complements of each body.

    The free complement :func:`_free_patch_schur` bounds ``[(A^-1)_ii]^-1`` from below. In the pinned patch,
    the joint neighbors ``j`` keep their full rows of ``A`` with the other bodies held fixed, so
    ``S_pinned = A_ii - sum_j A_ij A_jj^-1 A_ji`` bounds it from above. Both complements are stored as the
    lower triangles of symmetric blocks, see :func:`_store_lower_mat66`.

    Args:
        woodbury: Whether the free complement takes the Woodbury form, for at most six joint rows between two
            bodies, or :func:`_free_patch_schur_subtraction`.
    """

    @wp.kernel
    def _compute_patch_schur(
        # Inputs:
        body_block: wp.array[wp.int32],
        body_local: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        matrix_offsets: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
        inverse_permutation: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        body_weight_enabled: wp.array[wp.int32],
        diagonal_block: wp.array2d[wp.float32],
        diagonal_lower: wp.array2d[wp.float32],
        free_block: wp.array2d[wp.float32],
        free_lower: wp.array2d[wp.float32],
        neighbor_offset: wp.array[wp.int32],
        neighbor_body: wp.array[wp.int32],
        neighbor_row_offset: wp.array[wp.int32],
        neighbor_row_kind: wp.array[wp.int32],
        neighbor_row: wp.array[wp.int32],
        neighbor_row_side: wp.array[wp.int32],
        structural_jacobian_a: wp.array[vec6f],
        structural_jacobian_b: wp.array[vec6f],
        structural_penalty: wp.array[wp.float32],
        structural_world: wp.array[wp.int32],
        dynamic_jacobian_a: wp.array[vec6f],
        dynamic_jacobian_b: wp.array[vec6f],
        dynamic_effective_inertia: wp.array[wp.float32],
        model_time_dt: wp.array[wp.float32],
        # Outputs:
        free_schur: wp.array2d[wp.float32],
        pinned_schur: wp.array2d[wp.float32],
    ):
        bid = wp.tid()
        block = body_block[bid]
        if block < 0 or body_weight_enabled[bid] == 0:
            return
        if wp.static(woodbury):
            free = _free_patch_schur(
                bid,
                free_block,
                free_lower,
                neighbor_offset,
                neighbor_body,
                neighbor_row_offset,
                neighbor_row_kind,
                neighbor_row,
                neighbor_row_side,
                structural_jacobian_a,
                structural_jacobian_b,
                structural_penalty,
                structural_world,
                dynamic_jacobian_a,
                dynamic_jacobian_b,
                dynamic_effective_inertia,
                model_time_dt,
            )
        else:
            free = _free_patch_schur_subtraction(
                bid,
                free_block,
                neighbor_offset,
                neighbor_body,
                neighbor_row_offset,
                neighbor_row_kind,
                neighbor_row,
                neighbor_row_side,
                structural_jacobian_a,
                structural_jacobian_b,
                structural_penalty,
                structural_world,
                dynamic_jacobian_a,
                dynamic_jacobian_b,
                dynamic_effective_inertia,
                model_time_dt,
            )
        _store_lower_mat66(free_schur, bid, 0.5 * (free + wp.transpose(free)))
        offset = matrix_offsets[block]
        dimension = dimensions[block]
        vector_offset = vector_offsets[block]
        rows = _body_matrix_rows(vector_offset, inverse_permutation, body_local[bid], dimension)
        pinned = _load_symmetric_mat66(diagonal_block, bid)
        for entry in range(neighbor_offset[bid], neighbor_offset[bid + 1]):
            neighbor = neighbor_body[entry]
            if not (diagonal_lower[0, neighbor] > 0.0):
                continue
            neighbor_lower = _load_lower_mat66(diagonal_lower, neighbor)
            neighbor_rows = _body_matrix_rows(vector_offset, inverse_permutation, body_local[neighbor], dimension)
            # Y = L_j^-1 A_ji, then A_ij A_jj^-1 A_ji = Y^T Y
            coupling = _read_body_block(matrix, offset, dimension, neighbor_rows, rows)
            solved = _solve_lower_mat66(neighbor_lower, coupling)
            pinned -= wp.transpose(solved) @ solved
        _store_lower_mat66(pinned_schur, bid, 0.5 * (pinned + wp.transpose(pinned)))

    return _compute_patch_schur


@wp.kernel
def _compute_patch_weights(
    # Inputs:
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    inverse_permutation: wp.array[wp.int32],
    body_weight_enabled: wp.array[wp.int32],
    diagonal_block: wp.array2d[wp.float32],
    free_schur: wp.array2d[wp.float32],
    pinned_schur: wp.array2d[wp.float32],
    beta: wp.float32,
    sigma: wp.float32,
    matrix_weight_scale: wp.float32,
    # Outputs:
    matrix: wp.array[wp.float32],
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
    single_body_inverse: wp.array[mat66f],
):
    """Set the body weights ``W`` to the geometric means of the free and pinned patch Schur complements.

    ``S_free # S_pinned`` lies between both complements, see :func:`make_patch_schur_kernel`. A positive
    ``beta`` caps it at ``beta S_free``, and ``sigma`` keeps at least that fraction of the mean, see
    :func:`_geometric_mean_mat66`. A positive ``matrix_weight_scale`` sets the diagonal block of each body in
    the matrix to the symmetrized ``diagonal_block`` plus that multiple of the weight. A body alone in its
    block also stores the inverse of its weighted block, which is its whole factorization.
    """
    bid = wp.tid()
    block = body_block[bid]
    if block < 0:
        weight[bid] = mat66f(0.0)
        inverse_weight[bid] = mat66f(0.0)
        return
    mean = mat66f(0.0)
    enabled = body_weight_enabled[bid] != 0
    if enabled:
        free = _load_symmetric_mat66(free_schur, bid)
        mean, inverse, valid = _geometric_mean_mat66(free, _load_symmetric_mat66(pinned_schur, bid), beta, sigma)
        if valid:
            weight[bid] = mean
            inverse_weight[bid] = inverse
        else:
            mean = _finish_body_weight(bid, free, weight, inverse_weight)
    else:
        weight[bid] = mat66f(0.0)
        inverse_weight[bid] = mat66f(0.0)
    dimension = dimensions[block]
    write_matrix = enabled and matrix_weight_scale > 0.0
    if write_matrix or dimension == 6:
        weighted = _load_symmetric_mat66(diagonal_block, bid) + matrix_weight_scale * mean
        if write_matrix:
            offset = matrix_offsets[block]
            rows = _body_matrix_rows(vector_offsets[block], inverse_permutation, body_local[bid], dimension)
            for row in range(6):
                for col in range(6):
                    matrix[_matrix_index(offset, dimension, rows[row], rows[col])] = weighted[row, col]
        if dimension == 6:
            inverse_block, _valid = _invert_spd_mat66(weighted)
            single_body_inverse[bid] = inverse_block


@wp.func
def _invert_lower_mat66(lower: mat66f) -> mat66f:
    """Invert a lower triangular ``6 x 6`` matrix with a positive diagonal by forward substitution of each column."""
    inverse_lower = mat66f(0.0)
    for col in range(6):
        for row in range(6):
            if row >= col:
                value = wp.where(row == col, wp.float32(1.0), wp.float32(0.0))
                for inner in range(6):
                    if inner >= col and inner < row:
                        value -= lower[row, inner] * inverse_lower[inner, col]
                inverse_lower[row, col] = value / lower[row, row]
    return inverse_lower


@wp.func
def _invert_spd_mat66(matrix: mat66f) -> tuple[mat66f, wp.bool]:
    """Invert a symmetric positive-definite ``6 x 6`` matrix as ``L^-T L^-1`` from its Cholesky factor ``L``.

    The inverse is exactly symmetric, since each pair of its entries is evaluated once.
    """
    lower, valid = _cholesky_mat66(matrix)
    inverse = mat66f(0.0)
    if not valid:
        return inverse, False
    inverse_lower = _invert_lower_mat66(lower)
    for row in range(6):
        for col in range(6):
            if col <= row:
                value = wp.float32(0.0)
                for inner in range(6):
                    if inner >= row:
                        value += inverse_lower[inner, row] * inverse_lower[inner, col]
                inverse[row, col] = value
                inverse[col, row] = value
    return inverse, True


@wp.kernel
def _build_candidate_right_hand_side(
    # Inputs:
    body_vector_index: wp.array[wp.int32],
    smooth_right_hand_side: wp.array[wp.float32],
    weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    anchor_scale: wp.float32,
    splitting_dual: wp.array[vec6f],
    body_effort_offset: wp.array[wp.int32],
    body_effort_index: wp.array[wp.int32],
    body_effort_side: wp.array[wp.int32],
    effort_dynamic_row: wp.array[wp.int32],
    dynamic_jacobian_a: wp.array[vec6f],
    dynamic_jacobian_b: wp.array[vec6f],
    effort_counter: wp.array[wp.float32],
    structural_impulse: wp.array[vec6f],
    body_block: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    single_body_inverse: wp.array[mat66f],
    # Outputs:
    candidate_right_hand_side: wp.array[wp.float32],
    packed_solution: wp.array[wp.float32],
):
    """Add the splitting target and the current row impulses of each body to the frozen right-hand side.

    A body alone in its block writes its solution through the inverse of its block.
    """
    bid = wp.tid()
    body_offset = body_vector_index[bid]
    if body_offset < 0:
        return
    anchor = vec6f(0.0)
    if projected_twist:
        anchor = anchor_scale * projected_twist[bid]
    target = apply_body_weight(weight[bid], anchor + splitting_dual[bid])
    # Structural multiplier impulses h J^T lambda of the rows in the smooth system
    target += structural_impulse[bid]
    # Effort-limit counter-impulses J^T c
    effort_start = body_effort_offset[bid]
    effort_end = body_effort_offset[bid + 1]
    for incidence in range(effort_start, effort_end):
        effort = body_effort_index[incidence]
        dynamic_row = effort_dynamic_row[effort]
        jacobian = dynamic_jacobian_a[dynamic_row]
        if body_effort_side[incidence] != 0:
            jacobian = dynamic_jacobian_b[dynamic_row]
        target += effort_counter[effort] * jacobian
    candidate = vec6f(0.0)
    for row in range(6):
        candidate[row] = smooth_right_hand_side[body_offset + row] + target[row]
        candidate_right_hand_side[body_offset + row] = candidate[row]
    if dimensions[body_block[bid]] == 6:
        solution = single_body_inverse[bid] * candidate
        for row in range(6):
            packed_solution[body_offset + row] = solution[row]


@wp.kernel
def _invert_single_body_blocks(
    # Inputs:
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    # Outputs:
    single_body_inverse: wp.array[mat66f],
):
    """Invert the matrix of each body alone in its factor block."""
    bid = wp.tid()
    block = body_block[bid]
    if block < 0 or dimensions[block] != 6:
        return
    rows = vec6i()
    for axis in range(6):
        rows[axis] = 6 * body_local[bid] + axis
    block_matrix = _read_body_block(matrix, matrix_offsets[block], 6, rows, rows)
    inverse_block, _valid = _invert_spd_mat66(0.5 * (block_matrix + wp.transpose(block_matrix)))
    single_body_inverse[bid] = inverse_block


@wp.kernel
def _solve_single_body_blocks(
    # Inputs:
    body_block: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    single_body_inverse: wp.array[mat66f],
    right_hand_side: wp.array[wp.float32],
    # Outputs:
    solution: wp.array[wp.float32],
):
    """Solve the blocks that hold a single body with their inverses."""
    bid = wp.tid()
    block = body_block[bid]
    if block < 0 or dimensions[block] != 6:
        return
    body_offset = body_vector_index[bid]
    value = vec6f(0.0)
    for row in range(6):
        value[row] = right_hand_side[body_offset + row]
    value = single_body_inverse[bid] * value
    for row in range(6):
        solution[body_offset + row] = value[row]
