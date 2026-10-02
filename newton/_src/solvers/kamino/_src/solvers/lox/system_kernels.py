# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assembly kernels of the LOX smooth body systems.

Each factor block is one connected dynamic-body component and each dynamic body
contributes six linear-first velocity unknowns; prescribed bodies map to ``-1`` in the
matrix layout. The consensus body weights use the body mass matrix
``M = diag(m I_3, I_world)``, referenced at the body's center of mass.
"""

from __future__ import annotations

import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from ...core.math import compute_gyroscopic_torque
from ...core.types import mat66f, vec6f

###
# Module interface
###

__all__ = [
    "_assemble_body_inertial_systems",
    "_assemble_dynamic_joint_rows",
    "_assemble_structural_joint_rows",
    "_build_candidate_right_hand_side",
    "_compute_body_weights_and_add",
    "apply_body_weight",
    "compute_body_explicit_wrench",
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


@wp.func
def apply_body_weight(weight: mat66f, value: vec6f) -> vec6f:
    """Return ``weight @ value`` for a body weight or its inverse, reading only its nonzero blocks.

    Body weights are block diagonal, with a diagonal linear block and a full angular block
    (see :func:`_compute_body_weight_mass_proportional`).
    """
    angular_block = wp.mat33f(
        weight[3, 3],
        weight[3, 4],
        weight[3, 5],
        weight[4, 3],
        weight[4, 4],
        weight[4, 5],
        weight[5, 3],
        weight[5, 4],
        weight[5, 5],
    )
    angular = angular_block @ wp.vec3f(value[3], value[4], value[5])
    return vec6f(
        weight[0, 0] * value[0],
        weight[1, 1] * value[1],
        weight[2, 2] * value[2],
        angular[0],
        angular[1],
        angular[2],
    )


@wp.func
def _symmetrize_mat33(value: wp.mat33f) -> wp.mat33f:
    return 0.5 * (value + wp.transpose(value))


@wp.func
def _symmetrize_mat66(value: mat66f) -> mat66f:
    return 0.5 * (value + wp.transpose(value))


@wp.func
def _compute_body_weight_mass_proportional(
    smooth_diagonal: mat66f,
    mass: wp.float32,
    inertia_world: wp.mat33f,
    sigma: wp.float32,
    beta: wp.float32,
    mass_floor: wp.float32 = 1.0e-8,
    inertia_floor: wp.float32 = 1.0e-10,
    eta_floor: wp.float32 = 1.0e-6,
) -> tuple[mat66f, mat66f]:
    """Compute a mass-proportional rigid-body weight and its inverse.

    The implementation forms the symmetric normalization
    ``M^-1/2 smooth_diagonal M^-1/2``. The world-space inertia square root
    comes from a symmetric ``3 x 3`` eigendecomposition, and the normalized
    minimum eigenvalue from the bounded QR algorithm of :mod:`warp.fem.linalg`.
    This device work is suitable for CUDA graph capture.

    Finite asymmetric matrices are symmetrized. Positive mass below
    ``mass_floor``, inertia eigenvalues below ``inertia_floor``, and normalized
    smooth eigenvalues below ``eta_floor`` are clamped. Non-finite input,
    or nonpositive mass return zero matrices.

    Args:
        smooth_diagonal: Symmetric ``6 x 6`` body block ``A_ii`` in
            linear-first twist order.
        mass: Positive body mass [kg].
        inertia_world: Symmetric body inertia about its center of mass in
            world axes [kg m^2].
        sigma: Lower spectral fraction in the weight clamp.
        beta: Nominal weight transition threshold. The ``sigma`` floor may
            exceed it for sufficiently stiff modes.
        mass_floor: Minimum accepted positive mass [kg].
        inertia_floor: Minimum principal moment [kg m^2].
        eta_floor: Minimum dimensionless normalized smooth eigenvalue.

    Returns:
        The weight ``W = alpha M`` and its inverse.
    """
    if not wp.isfinite(mass) or not wp.isfinite(inertia_world) or not wp.isfinite(smooth_diagonal) or mass <= 0.0:
        return mat66f(0.0), mat66f(0.0)

    regularized_mass = mass
    if regularized_mass < mass_floor:
        regularized_mass = mass_floor

    symmetric_inertia = _symmetrize_mat33(inertia_world)
    symmetric_smooth = _symmetrize_mat66(smooth_diagonal)

    inertia_axes, inertia_eigenvalues = wp.eig3(symmetric_inertia)
    if not wp.isfinite(inertia_eigenvalues) or not wp.isfinite(inertia_axes):
        return mat66f(0.0), mat66f(0.0)
    regularized_inertia_eigenvalues = wp.max(inertia_eigenvalues, wp.vec3f(inertia_floor))
    inverse_inertia_eigenvalues = wp.vec3f(0.0)
    inverse_sqrt_inertia_eigenvalues = wp.vec3f(0.0)
    for index in range(3):
        inverse_inertia_eigenvalues[index] = 1.0 / regularized_inertia_eigenvalues[index]
        inverse_sqrt_inertia_eigenvalues[index] = 1.0 / wp.sqrt(regularized_inertia_eigenvalues[index])

    regularized_inertia = wp.mat33f(0.0)
    inverse_inertia = wp.mat33f(0.0)
    inverse_sqrt_inertia = wp.mat33f(0.0)
    for row in range(3):
        for col in range(3):
            for index in range(3):
                axis_product = inertia_axes[row, index] * inertia_axes[col, index]
                regularized_inertia[row, col] += axis_product * regularized_inertia_eigenvalues[index]
                inverse_inertia[row, col] += axis_product * inverse_inertia_eigenvalues[index]
                inverse_sqrt_inertia[row, col] += axis_product * inverse_sqrt_inertia_eigenvalues[index]

    inverse_sqrt_mass = 1.0 / wp.sqrt(regularized_mass)
    inverse_sqrt_spatial_mass = mat66f(0.0)
    for index in range(3):
        inverse_sqrt_spatial_mass[index, index] = inverse_sqrt_mass
    for row in range(3):
        for col in range(3):
            inverse_sqrt_spatial_mass[row + 3, col + 3] = inverse_sqrt_inertia[row, col]

    normalized_smooth = inverse_sqrt_spatial_mass @ symmetric_smooth @ inverse_sqrt_spatial_mass
    normalized_smooth = _symmetrize_mat66(normalized_smooth)
    eigenvalues, _eigenvectors = symmetric_eigenvalues_qr(normalized_smooth, _EIGENVALUE_TOLERANCE)
    eta = wp.min(eigenvalues)
    if not wp.isfinite(eta):
        return mat66f(0.0), mat66f(0.0)
    if eta < eta_floor:
        eta = eta_floor

    alpha = wp.max(sigma * eta, wp.min(beta, eta))
    if not wp.isfinite(alpha) or alpha <= 0.0:
        return mat66f(0.0), mat66f(0.0)

    weight = mat66f(0.0)
    inverse_weight = mat66f(0.0)
    for index in range(3):
        weight[index, index] = alpha * regularized_mass
        inverse_weight[index, index] = 1.0 / (alpha * regularized_mass)
    for row in range(3):
        for col in range(3):
            weight[row + 3, col + 3] = alpha * regularized_inertia[row, col]
            inverse_weight[row + 3, col + 3] = inverse_inertia[row, col] / alpha

    return weight, inverse_weight


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
    # Outputs:
    penalty: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    """Assemble the frozen penalty terms of each structural row, or a zero penalty for rows left out."""
    joint_row = wp.tid()
    wid = row_world[joint_row]
    if wid < 0 or wid >= joint_penalty_scale.shape[0]:
        penalty[joint_row] = 0.0
        return
    dt = model_time_dt[wid]
    bid_a = body_a[joint_row]
    bid_b = body_b[joint_row]
    body_count = body_block.shape[0]
    world_penalty_scale = joint_penalty_scale[wid]
    if (
        dt <= 0.0
        or world_penalty_scale <= 0.0
        or effective_mass[joint_row] <= 0.0
        or (bid_a < 0 and bid_b < 0)
        or bid_a >= body_count
        or bid_b >= body_count
    ):
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
    # A large violation is corrected by at most joint_max_correction per step
    row_residual = wp.clamp(residual[joint_row], -joint_max_correction, joint_max_correction)
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


@wp.kernel
def _compute_body_weights_and_add(
    # Inputs:
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    body_weight_enabled: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    model_bodies_m_i: wp.array[wp.float32],
    data_bodies_I_i: wp.array[wp.mat33f],
    sigma: wp.float32,
    beta: wp.float32,
    # Outputs:
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
    matrix: wp.array[wp.float32],
):
    """Compute the splitting weight of each enabled body from its diagonal block of ``A`` and add it in place.

    Each body owns its diagonal block, which one thread updates in place.
    """
    bid = wp.tid()
    block = body_block[bid]
    if block < 0 or body_weight_enabled[bid] == 0:
        weight[bid] = mat66f(0.0)
        inverse_weight[bid] = mat66f(0.0)
        return
    local_body = body_local[bid]
    dimension = dimensions[block]
    matrix_offset = matrix_offsets[block]
    body_offset = 6 * local_body
    smooth_diagonal = mat66f(0.0)
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)
            smooth_diagonal[row, col] = matrix[index]

    body_weight, body_inverse_weight = _compute_body_weight_mass_proportional(
        smooth_diagonal,
        model_bodies_m_i[bid],
        data_bodies_I_i[bid],
        sigma,
        beta,
    )
    weight[bid] = body_weight
    inverse_weight[bid] = body_inverse_weight
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
    target = apply_body_weight(weight[bid], projected_twist[bid] + splitting_dual[bid])
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
