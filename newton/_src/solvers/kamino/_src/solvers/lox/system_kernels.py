# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assembly kernels of the LOX smooth body systems.

Each factor block is one connected dynamic-body component and each dynamic body
contributes six linear-first velocity unknowns; prescribed bodies map to ``-1`` in the
matrix layout. The consensus body weights use the body mass matrix
``M = diag(m I_3, I_world)``, referenced at the body's center of mass.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from ...core.math import compute_gyroscopic_torque
from ...core.types import mat36f, mat66f, vec6f

###
# Module interface
###

__all__ = [
    "ProjectionWeights",
    "_add_body_weights_to_matrix",
    "_assemble_body_inertial_systems",
    "_assemble_dynamic_joint_rows",
    "_assemble_structural_joint_rows",
    "_build_candidate_right_hand_side",
    "_compute_free_body_blocks",
    "apply_body_weight",
    "compute_body_explicit_wrench",
    "compute_structural_correction",
    "is_rotational_structural_row",
    "make_patch_weights_kernel",
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
    jacobian: vec6f,
    effective_inertia: wp.float32,
    free_velocity: wp.float32,
) -> _PrimalRowContribution:
    """Eliminate one implicit joint-dynamics coordinate into body space.

    The joint equation is ``effective_inertia * dq = h_joint`` with
    ``free_velocity = h_joint / effective_inertia`` and ``dq = J v``.

    Args:
        jacobian: Linear-first joint velocity Jacobian row.
        effective_inertia: Positive implicit joint inertia.
        free_velocity: Implicit joint free velocity.

    Returns:
        ``effective_inertia J^T J`` and
        ``effective_inertia free_velocity J^T``.
    """
    result = _PrimalRowContribution()
    result.matrix = effective_inertia * wp.outer(jacobian, jacobian)
    result.right_hand_side = effective_inertia * free_velocity * jacobian
    return result


@wp.func
def _compute_augmented_joint_row(
    jacobian: vec6f,
    residual: wp.float32,
    penalty: wp.float32,
    time_step: wp.float32,
    linearization_velocity: wp.float32,
) -> _PrimalRowContribution:
    """Linearize the augmented Lagrangian penalty of one structural joint row.

    For ``C(q(v)) ~= residual + h J (v - v_k)``, this returns the frozen terms in

    ``(M + h^2 penalty J^T J) v``
    ``= f - h penalty residual J^T + h^2 penalty J^T J v_k + h J^T lambda``.

    The multiplier term ``h J^T lambda`` changes every iteration and is added to the
    candidate right-hand side.

    Args:
        jacobian: Linear-first structural joint Jacobian row.
        residual: Current joint position residual [m or rad].
        penalty: Positive augmented penalty [N/m or N m/rad].
        time_step: Time step [s].
        linearization_velocity: Row velocity ``J v_k`` at the linearization
            twist [m/s or rad/s]. Pass zero for the first linearly implicit
            assembly about the begin-step pose.

    Returns:
        The structural matrix and right-hand-side contributions.
    """
    result = _PrimalRowContribution()
    result.matrix = time_step * time_step * penalty * wp.outer(jacobian, jacobian)
    result.right_hand_side = (
        -time_step * penalty * residual + time_step * time_step * penalty * linearization_velocity
    ) * jacobian
    return result


@wp.func
def _matrix_index(
    matrix_offset: wp.int32,
    dimension: wp.int32,
    row: wp.int32,
    col: wp.int32,
) -> wp.int32:
    return matrix_offset + dimension * row + col


@wp.func
def _atomic_add_body_vector(
    vector: wp.array[wp.float32],
    vector_offset: wp.int32,
    body: wp.int32,
    value: vec6f,
):
    body_offset = vector_offset + 6 * body
    for row in range(6):
        wp.atomic_add(vector, body_offset + row, value[row])


@wp.func
def _atomic_add_body_block(
    matrix: wp.array[wp.float32],
    matrix_offset: wp.int32,
    dimension: wp.int32,
    row_body: wp.int32,
    col_body: wp.int32,
    value: mat66f,
):
    row_offset = 6 * row_body
    col_offset = 6 * col_body
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, row_offset + row, col_offset + col)
            wp.atomic_add(matrix, index, value[row, col])


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
    model_bodies_m_i: wp.array[wp.float32],
    data_bodies_I_i: wp.array[wp.mat33f],
    velocity_previous: wp.array[vec6f],
    data_bodies_w_e_i: wp.array[vec6f],
    data_bodies_w_a_i: wp.array[vec6f],
    model_gravity_vector: wp.array[wp.vec3f],
    model_time_dt: wp.array[wp.float32],
    # Outputs:
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    bid = wp.tid()
    wid = body_world[bid]
    dt = model_time_dt[wid]
    block = body_block[bid]
    if block < 0:
        return
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

    body_offset = 6 * local_body
    for row in range(6):
        right_hand_side[vector_offset + body_offset + row] = contribution.right_hand_side[row]
        for col in range(6):
            matrix[_matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)] = contribution.matrix[
                row, col
            ]


@wp.kernel
def _assemble_dynamic_joint_rows(
    # Inputs:
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    prescribed_twist: wp.array[vec6f],
    # Outputs:
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    joint_row = wp.tid()
    bid_a = body_a[joint_row]
    bid_b = body_b[joint_row]
    body_count = body_block.shape[0]
    if effective_inertia[joint_row] <= 0.0 or (bid_a < 0 and bid_b < 0) or bid_a >= body_count or bid_b >= body_count:
        return

    bid_a_local = body_local[bid_a] if bid_a >= 0 else -1
    bid_b_local = body_local[bid_b] if bid_b >= 0 else -1
    if bid_a_local < 0 and bid_b_local < 0:
        return
    block = body_block[bid_a] if bid_a_local >= 0 else body_block[bid_b]
    if block < 0 or block >= dimensions.shape[0] or (bid_b_local >= 0 and body_block[bid_b] != block):
        return
    dimension = dimensions[block]

    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    inertia = effective_inertia[joint_row]
    velocity = free_velocity[joint_row]
    a_jacobian = jacobian_a[joint_row]
    b_jacobian = jacobian_b[joint_row]
    if prescribed_twist and bid_a >= 0 and bid_a_local < 0:
        velocity -= wp.dot(a_jacobian, prescribed_twist[bid_a])
    if prescribed_twist and bid_b >= 0 and bid_b_local < 0:
        velocity -= wp.dot(b_jacobian, prescribed_twist[bid_b])

    if bid_a_local >= 0:
        a_contribution = _compute_dynamic_joint_row(a_jacobian, inertia, velocity)
        _atomic_add_body_block(matrix, matrix_offset, dimension, bid_a_local, bid_a_local, a_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, bid_a_local, a_contribution.right_hand_side)
    if bid_b_local >= 0:
        b_contribution = _compute_dynamic_joint_row(b_jacobian, inertia, velocity)
        _atomic_add_body_block(matrix, matrix_offset, dimension, bid_b_local, bid_b_local, b_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, bid_b_local, b_contribution.right_hand_side)
    if bid_a_local >= 0 and bid_b_local >= 0:
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            bid_a_local,
            bid_b_local,
            wp.outer(inertia * a_jacobian, b_jacobian),
        )
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            bid_b_local,
            bid_a_local,
            wp.outer(inertia * b_jacobian, a_jacobian),
        )


@wp.kernel
def _assemble_structural_joint_rows(
    # Inputs:
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    row_world: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    residual: wp.array[wp.float32],
    effective_mass: wp.array[wp.float32],
    prescribed_twist: wp.array[vec6f],
    model_time_dt: wp.array[wp.float32],
    joint_penalty_scale: wp.array[wp.float32],
    joint_max_correction: wp.float32,
    joint_translation_dead_zone: wp.float32,
    joint_rotation_dead_zone: wp.float32,
    # Outputs:
    penalty: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    """Assemble the frozen penalty terms of each structural row, or a zero penalty for rows left out."""
    joint_row = wp.tid()
    wid = row_world[joint_row]
    if wid < 0:
        penalty[joint_row] = 0.0
        return
    dt = model_time_dt[wid]
    bid_a = body_a[joint_row]
    bid_b = body_b[joint_row]
    body_count = body_block.shape[0]
    if (
        dt <= 0.0
        or effective_mass[joint_row] <= 0.0
        or (bid_a < 0 and bid_b < 0)
        or bid_a >= body_count
        or bid_b >= body_count
    ):
        penalty[joint_row] = 0.0
        return
    # Both bodies of a joint share the scale of their component
    world_penalty_scale = joint_penalty_scale[wp.where(bid_a >= 0, bid_a, bid_b)]
    if world_penalty_scale <= 0.0:
        penalty[joint_row] = 0.0
        return

    bid_a_local = body_local[bid_a] if bid_a >= 0 else -1
    bid_b_local = body_local[bid_b] if bid_b >= 0 else -1
    if bid_a_local < 0 and bid_b_local < 0:
        penalty[joint_row] = 0.0
        return
    block = body_block[bid_a] if bid_a_local >= 0 else body_block[bid_b]
    if block < 0 or block >= dimensions.shape[0] or (bid_b_local >= 0 and body_block[bid_b] != block):
        penalty[joint_row] = 0.0
        return
    dimension = dimensions[block]

    row_penalty = world_penalty_scale * effective_mass[joint_row] / (dt * dt)
    penalty[joint_row] = row_penalty
    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    a_jacobian = jacobian_a[joint_row]
    b_jacobian = jacobian_b[joint_row]
    # A small violation is left to average out, and a large one is corrected by at most joint_max_correction per step
    dead_zone = joint_translation_dead_zone
    if is_rotational_structural_row(a_jacobian, b_jacobian):
        dead_zone = joint_rotation_dead_zone
    row_residual = compute_structural_correction(residual[joint_row], dead_zone, joint_max_correction)
    # The rows are linearized about zero dynamic twists and the prescribed twists of the other bodies.
    linearization_velocity = wp.float32(0.0)
    if prescribed_twist and bid_a >= 0 and bid_a_local < 0:
        linearization_velocity -= wp.dot(a_jacobian, prescribed_twist[bid_a])
    if prescribed_twist and bid_b >= 0 and bid_b_local < 0:
        linearization_velocity -= wp.dot(b_jacobian, prescribed_twist[bid_b])

    if bid_a_local >= 0:
        a_contribution = _compute_augmented_joint_row(a_jacobian, row_residual, row_penalty, dt, linearization_velocity)
        _atomic_add_body_block(matrix, matrix_offset, dimension, bid_a_local, bid_a_local, a_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, bid_a_local, a_contribution.right_hand_side)
    if bid_b_local >= 0:
        b_contribution = _compute_augmented_joint_row(b_jacobian, row_residual, row_penalty, dt, linearization_velocity)
        _atomic_add_body_block(matrix, matrix_offset, dimension, bid_b_local, bid_b_local, b_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, bid_b_local, b_contribution.right_hand_side)
    if bid_a_local >= 0 and bid_b_local >= 0:
        cross_scale = dt * dt * row_penalty
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            bid_a_local,
            bid_b_local,
            wp.outer(cross_scale * a_jacobian, b_jacobian),
        )
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            bid_b_local,
            bid_a_local,
            wp.outer(cross_scale * b_jacobian, a_jacobian),
        )


@wp.func
def _read_body_block(
    matrix: wp.array[wp.float32], offset: wp.int32, dimension: wp.int32, row_body: wp.int32, col_body: wp.int32
) -> mat66f:
    """Read the ``6 x 6`` block of two local bodies from a dense block matrix."""
    block = mat66f(0.0)
    for row in range(6):
        for col in range(6):
            block[row, col] = matrix[_matrix_index(offset, dimension, 6 * row_body + row, 6 * col_body + col)]
    return block


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


@wp.kernel
def _compute_free_body_blocks(
    # Inputs:
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
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
    free_block: wp.array[mat66f],
    free_lower: wp.array[mat66f],
):
    """Return the diagonal block of each body without its joint rows to other dynamic bodies, and its Cholesky factor.

    The incidence lists only hold these rows; the rows to static bodies stay in the block. The factor is zero
    when the block is not positive definite, e.g. for a massless body.
    """
    bid = wp.tid()
    block = body_block[bid]
    if block < 0:
        return
    result = _read_body_block(matrix, matrix_offsets[block], dimensions[block], body_local[bid], body_local[bid])
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
    free_block[bid] = result
    lower, valid = _cholesky_mat66(0.5 * (result + wp.transpose(result)))
    if valid:
        free_lower[bid] = lower
    else:
        free_lower[bid] = mat66f(0.0)


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
def _finish_body_weight(bid: wp.int32, schur: mat66f, weight: wp.array[mat66f], inverse_weight: wp.array[mat66f]):
    """Store the symmetrized weight and its inverse, or zero when it is not positive definite."""
    symmetric = 0.5 * (schur + wp.transpose(schur))
    inverse, ok = _invert_spd_mat66(symmetric)
    if not ok:
        weight[bid] = mat66f(0.0)
        inverse_weight[bid] = mat66f(0.0)
        return
    weight[bid] = symmetric
    inverse_weight[bid] = inverse


@wp.func
def _free_patch_schur(
    bid: wp.int32,
    free_block: wp.array[mat66f],
    free_lower: wp.array[mat66f],
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
    schur = free_block[bid]
    for entry in range(neighbor_offset[bid], neighbor_offset[bid + 1]):
        lower = free_lower[neighbor_body[entry]]
        if not (lower[0, 0] > 0.0):
            continue
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
                    body_jacobian = wp.where(neighbor_side == 0, jacobian_b, jacobian_a)
                    forward = vec6f(0.0)
                    for axis in range(6):
                        value = neighbor_jacobian[axis]
                        for inner in range(axis):
                            value -= lower[axis, inner] * forward[inner]
                        forward[axis] = value / lower[axis, axis]
                    for axis in range(6):
                        forward_rows[slot, axis] = forward[axis]
                        body_rows[slot, axis] = body_jacobian[axis]
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
        solved = mat66f(0.0)
        for axis in range(6):
            for col in range(6):
                value = body_rows[axis, col]
                for inner in range(axis):
                    value -= joint_lower[axis, inner] * solved[inner, col]
                solved[axis, col] = value / joint_lower[axis, axis]
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
    free_block: wp.array[mat66f],
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
    schur = free_block[bid]
    for entry in range(neighbor_offset[bid], neighbor_offset[bid + 1]):
        neighbor_matrix = free_block[neighbor_body[entry]]
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
        solved = mat66f(0.0)
        for axis in range(6):
            for col in range(6):
                value = coupling[axis, col]
                for inner in range(axis):
                    value -= lower[axis, inner] * solved[inner, col]
                solved[axis, col] = value / lower[axis, axis]
        schur -= wp.transpose(solved) @ solved
    return schur


@wp.func
def _geometric_mean_mat66(
    lower_bound: mat66f, upper_bound: mat66f, beta: wp.float32, sigma: wp.float32
) -> tuple[mat66f, wp.bool]:
    """Return the matrix geometric mean ``A # B = L (L^-1 B L^-T)^1/2 L^T`` of two symmetric positive-definite
    ``6 x 6`` matrices, with ``A = L L^T``, and whether it is defined.

    ``A`` is the lower bound, so the eigenvalues of ``L^-1 B L^-T`` are clamped at one, which keeps the
    mean at least ``A`` against round-off in ``B``. A positive ``beta`` clamps each eigenvalue ``r`` of the
    square root to ``max(sigma r, min(r, beta))``, which keeps the result at most ``beta A`` and at least
    ``sigma`` times the mean.
    """
    lower, valid = _cholesky_mat66(lower_bound)
    if not valid:
        return lower_bound, False
    # X = L^-1 B, then M = L^-1 X^T = L^-1 B L^-T
    solved = mat66f(0.0)
    for axis in range(6):
        for col in range(6):
            value = upper_bound[axis, col]
            for inner in range(axis):
                value -= lower[axis, inner] * solved[inner, col]
            solved[axis, col] = value / lower[axis, axis]
    transposed = wp.transpose(solved)
    congruent = mat66f(0.0)
    for axis in range(6):
        for col in range(6):
            value = transposed[axis, col]
            for inner in range(axis):
                value -= lower[axis, inner] * congruent[inner, col]
            congruent[axis, col] = value / lower[axis, axis]
    normalized, scale = normalize_symmetric_matrix(0.5 * (congruent + wp.transpose(congruent)), 6)
    if not wp.isfinite(scale) or scale <= 0.0:
        return lower_bound, False
    # M = P^T D P, with one eigenvector per row of P
    eigenvalues, eigenvectors = symmetric_eigenvalues_qr(normalized, _EIGENVALUE_TOLERANCE)
    root = mat66f(0.0)
    for index in range(6):
        vector = eigenvectors[index]
        ratio = wp.sqrt(wp.max(scale * eigenvalues[index], 1.0))
        if beta > 0.0:
            ratio = wp.max(sigma * ratio, wp.min(ratio, beta))
        root += ratio * wp.outer(vector, vector)
    return lower @ root @ wp.transpose(lower), True


@cache
def make_patch_weights_kernel(woodbury: bool):
    """Build the kernel of the body weights ``W``.

    ``W_i`` is the geometric mean of the Schur complements of body ``i`` in its free and pinned patches. The
    free complement :func:`_free_patch_schur` bounds ``[(A^-1)_ii]^-1`` from below. In the pinned patch, the
    joint neighbors ``j`` keep their full rows of ``A`` with the other bodies held fixed, so
    ``S_pinned = A_ii - sum_j A_ij A_jj^-1 A_ji`` bounds it from above. The weight
    ``S_free # S_pinned`` lies between both bounds. A positive ``beta`` caps it at ``beta S_free``, and
    ``sigma`` keeps at least that fraction of the mean, see :func:`_geometric_mean_mat66`.

    Args:
        woodbury: Whether the free complement takes the Woodbury form, for at most six joint rows between two
            bodies, or :func:`_free_patch_schur_subtraction`.
    """

    @wp.kernel
    def _compute_patch_weights(
        # Inputs:
        body_block: wp.array[wp.int32],
        body_local: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        matrix_offsets: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        body_weight_enabled: wp.array[wp.int32],
        free_block: wp.array[mat66f],
        free_lower: wp.array[mat66f],
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
        beta: wp.float32,
        sigma: wp.float32,
        # Outputs:
        weight: wp.array[mat66f],
        inverse_weight: wp.array[mat66f],
    ):
        bid = wp.tid()
        block = body_block[bid]
        if block < 0 or body_weight_enabled[bid] == 0:
            weight[bid] = mat66f(0.0)
            inverse_weight[bid] = mat66f(0.0)
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
        offset = matrix_offsets[block]
        dimension = dimensions[block]
        local = body_local[bid]
        pinned = _read_body_block(matrix, offset, dimension, local, local)
        for entry in range(neighbor_offset[bid], neighbor_offset[bid + 1]):
            neighbor_local = body_local[neighbor_body[entry]]
            neighbor_matrix = _read_body_block(matrix, offset, dimension, neighbor_local, neighbor_local)
            neighbor_lower, neighbor_valid = _cholesky_mat66(0.5 * (neighbor_matrix + wp.transpose(neighbor_matrix)))
            if not neighbor_valid:
                continue
            # Y = L_j^-1 A_ji, then A_ij A_jj^-1 A_ji = Y^T Y
            coupling = _read_body_block(matrix, offset, dimension, neighbor_local, local)
            solved = mat66f(0.0)
            for axis in range(6):
                for col in range(6):
                    value = coupling[axis, col]
                    for inner in range(axis):
                        value -= neighbor_lower[axis, inner] * solved[inner, col]
                    solved[axis, col] = value / neighbor_lower[axis, axis]
            pinned -= wp.transpose(solved) @ solved
        mean, valid = _geometric_mean_mat66(
            0.5 * (free + wp.transpose(free)), 0.5 * (pinned + wp.transpose(pinned)), beta, sigma
        )
        if not valid:
            mean = free
        _finish_body_weight(bid, mean, weight, inverse_weight)

    return _compute_patch_weights


@wp.func
def _invert_spd_mat66(matrix: mat66f) -> tuple[mat66f, wp.bool]:
    """Invert a symmetric positive-definite ``6 x 6`` matrix as ``L^-T L^-1`` from its Cholesky factor ``L``.

    The inverse is exactly symmetric, since each pair of its entries is evaluated once.
    """
    lower, valid = _cholesky_mat66(matrix)
    inverse = mat66f(0.0)
    if not valid:
        return inverse, False
    # Inverse factor L^-1, lower triangular, by forward substitution of each column
    inverse_lower = mat66f(0.0)
    for col in range(6):
        for row in range(6):
            if row >= col:
                value = wp.where(row == col, wp.float32(1.0), wp.float32(0.0))
                for inner in range(6):
                    if inner >= col and inner < row:
                        value -= lower[row, inner] * inverse_lower[inner, col]
                inverse_lower[row, col] = value / lower[row, row]
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
def _add_body_weights_to_matrix(
    # Inputs:
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    weight: wp.array[mat66f],
    scale: wp.float32,
    # Outputs:
    matrix: wp.array[wp.float32],
):
    """Add the scaled splitting weight of each body to its diagonal block of the matrix."""
    bid = wp.tid()
    block = body_block[bid]
    if block < 0:
        return
    dimension = dimensions[block]
    matrix_offset = matrix_offsets[block]
    body_offset = 6 * body_local[bid]
    body_weight = scale * weight[bid]
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)
            matrix[index] += body_weight[row, col]


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
    # Outputs:
    candidate_right_hand_side: wp.array[wp.float32],
):
    """Add the splitting target and the current row impulses of each body to the frozen right-hand side."""
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
    for row in range(6):
        candidate_right_hand_side[body_offset + row] = smooth_right_hand_side[body_offset + row] + target[row]
