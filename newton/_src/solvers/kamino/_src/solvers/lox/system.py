# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-diagonal primal body systems for the LOX backend.

This module owns only solver-internal, contact-free body-space assembly. Each
factor block is one connected dynamic-body component and each dynamic body
contributes six linear-first velocity unknowns. Prescribed bodies retain their
global packed body indices but map to ``-1`` in the matrix layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import warp as wp

from ...core.types import mat66f, vec6f
from ...linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, HybridLLTBlockedSolver
from .weight import (
    BODY_WEIGHT_STATUS_VALID,
    compute_body_weight_mass_proportional,
)

__all__ = ["BatchedPrimalBodySystem"]

wp.set_module_options({"enable_backward": False})

_RCM_MIN_DIMENSION = 256


@wp.struct
class PrimalRowContribution:
    """Matrix and right-hand-side contribution of one body-space term."""

    matrix: mat66f
    """Symmetric contribution to the body-space operator."""
    right_hand_side: vec6f
    """Contribution to the body-space right-hand side."""


@wp.func
def make_spatial_mass_matrix(mass: wp.float32, inertia_world: wp.mat33f) -> mat66f:
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
) -> vec6f:
    """Combine explicit body forces in world coordinates.

    Args:
        mass: Body mass [kg].
        inertia_world: World-space inertia about the center of mass [kg m^2].
        velocity_previous: Begin-of-step linear-first body twist [m/s, rad/s].
        external_wrench: External body wrench excluding gravity [N, N m].
        actuation_wrench: Joint-actuation body wrench [N, N m].
        gravity: World-space gravitational acceleration [m/s^2].

    Returns:
        Explicit force and torque, including gravity and the gyroscopic torque [N, N m].
    """
    result = external_wrench + actuation_wrench
    for index in range(3):
        result[index] += mass * gravity[index]

    angular_velocity = wp.vec3f(velocity_previous[3], velocity_previous[4], velocity_previous[5])
    gyroscopic_torque = -wp.cross(angular_velocity, inertia_world @ angular_velocity)
    for index in range(3):
        result[index + 3] += gyroscopic_torque[index]
    return result


@wp.func
def compute_body_inertial_system(
    mass: wp.float32,
    inertia_world: wp.mat33f,
    velocity_previous: vec6f,
    force_explicit: vec6f,
    time_step: wp.float32,
) -> PrimalRowContribution:
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
    result = PrimalRowContribution()
    result.matrix = make_spatial_mass_matrix(mass, inertia_world)
    result.right_hand_side = result.matrix @ velocity_previous + time_step * force_explicit
    return result


@wp.func
def compute_dynamic_joint_row(
    jacobian: vec6f,
    effective_inertia: wp.float32,
    free_velocity: wp.float32,
) -> PrimalRowContribution:
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
    result = PrimalRowContribution()
    result.matrix = effective_inertia * wp.outer(jacobian, jacobian)
    result.right_hand_side = effective_inertia * free_velocity * jacobian
    return result


@wp.func
def compute_augmented_joint_row(
    jacobian: vec6f,
    residual: wp.float32,
    multiplier: wp.float32,
    penalty: wp.float32,
    time_step: wp.float32,
    linearization_velocity: wp.float32,
) -> PrimalRowContribution:
    """Linearize one structural joint row with augmented Lagrangian forces.

    For ``C(q(v)) ~= residual + h J (v - v_k)``, this returns the terms in

    ``(M + h^2 penalty J^T J) v``
    ``= f - h J^T (multiplier + penalty residual)``
    ``+ h^2 penalty J^T J v_k``.

    Args:
        jacobian: Linear-first structural joint Jacobian row.
        residual: Current joint position residual [m or rad].
        multiplier: Persistent structural multiplier [N or N m].
        penalty: Positive augmented penalty [N/m or N m/rad].
        time_step: Time step [s].
        linearization_velocity: Row velocity ``J v_k`` at the linearization
            twist [m/s or rad/s]. Pass zero for the first linearly implicit
            assembly about the begin-step pose.

    Returns:
        The structural matrix and right-hand-side contributions.
    """
    result = PrimalRowContribution()
    result.matrix = time_step * time_step * penalty * wp.outer(jacobian, jacobian)
    result.right_hand_side = (
        -time_step * (multiplier + penalty * residual) + time_step * time_step * penalty * linearization_velocity
    ) * jacobian
    return result


@lru_cache(maxsize=32)
def _expand_body_symbolic_adjacency(
    body_count: int,
    body_edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    """Expand a reusable body graph into its six-DoF scalar adjacency."""
    body_adjacency = np.eye(body_count, dtype=bool)
    if body_edges:
        edges = np.asarray(body_edges, dtype=np.int32)
        body_adjacency[edges[:, 0], edges[:, 1]] = True
        body_adjacency[edges[:, 1], edges[:, 0]] = True
    scalar_adjacency = np.kron(body_adjacency, np.ones((6, 6), dtype=bool))
    np.fill_diagonal(scalar_adjacency, False)
    return tuple(tuple(np.flatnonzero(row).tolist()) for row in scalar_adjacency)


@wp.kernel
def _mark_blocks_with_unilaterals(
    body_component: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    block_has_unilateral: wp.array[wp.int32],
):
    body = wp.tid()
    component = body_component[body]
    if component >= 0 and body_has_unilateral[body] != 0:
        wp.atomic_max(block_has_unilateral, component, 1)


@wp.kernel
def _enable_bodies_in_weighted_blocks(
    body_component: wp.array[wp.int32],
    block_has_unilateral: wp.array[wp.int32],
    body_weight_enabled: wp.array[wp.int32],
):
    body = wp.tid()
    component = body_component[body]
    body_weight_enabled[body] = wp.where(component >= 0 and block_has_unilateral[component] != 0, 1, 0)


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


@wp.kernel
def _unpack_body_solution(
    body_vector_index: wp.array[wp.int32],
    packed_solution: wp.array[wp.float32],
    prescribed_twist: wp.array[vec6f],
    body_solution: wp.array[vec6f],
):
    body = wp.tid()
    source_offset = body_vector_index[body]
    value = vec6f(0.0)
    if prescribed_twist:
        value = prescribed_twist[body]
    if source_offset >= 0:
        for index in range(6):
            value[index] = packed_solution[source_offset + index]
    body_solution[body] = value


@wp.kernel
def _assemble_body_inertial_systems(
    body_world: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    mass: wp.array[wp.float32],
    inertia_world: wp.array[wp.mat33f],
    velocity_previous: wp.array[vec6f],
    evaluation_velocity: wp.array[vec6f],
    external_wrench: wp.array[vec6f],
    actuation_wrench: wp.array[vec6f],
    gravity: wp.array[wp.vec3f],
    time_step: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    body = wp.tid()
    dt = time_step[body_world[body]]
    block = body_block[body]
    if block < 0:
        return
    local_body = body_local[body]
    dimension = dimensions[block]
    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    force_explicit = compute_body_explicit_wrench(
        mass[body],
        inertia_world[body],
        evaluation_velocity[body],
        external_wrench[body],
        actuation_wrench[body],
        gravity[body_world[body]],
    )
    contribution = compute_body_inertial_system(
        mass[body], inertia_world[body], velocity_previous[body], force_explicit, dt
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
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    prescribed_twist: wp.array[vec6f],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    joint_row = wp.tid()
    first = body_first[joint_row]
    second = body_second[joint_row]
    body_count = body_block.shape[0]
    if effective_inertia[joint_row] <= 0.0 or (first < 0 and second < 0) or first >= body_count or second >= body_count:
        return

    first_local = body_local[first] if first >= 0 else -1
    second_local = body_local[second] if second >= 0 else -1
    if first_local < 0 and second_local < 0:
        return
    block = body_block[first] if first_local >= 0 else body_block[second]
    if block < 0 or block >= dimensions.shape[0] or (second_local >= 0 and body_block[second] != block):
        return
    dimension = dimensions[block]

    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    inertia = effective_inertia[joint_row]
    velocity = free_velocity[joint_row]
    first_jacobian = jacobian_first[joint_row]
    second_jacobian = jacobian_second[joint_row]
    if prescribed_twist and first >= 0 and first_local < 0:
        velocity -= wp.dot(first_jacobian, prescribed_twist[first])
    if prescribed_twist and second >= 0 and second_local < 0:
        velocity -= wp.dot(second_jacobian, prescribed_twist[second])

    if first_local >= 0:
        first_contribution = compute_dynamic_joint_row(first_jacobian, inertia, velocity)
        _atomic_add_body_block(matrix, matrix_offset, dimension, first_local, first_local, first_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, first_local, first_contribution.right_hand_side)
    if second_local >= 0:
        second_contribution = compute_dynamic_joint_row(second_jacobian, inertia, velocity)
        _atomic_add_body_block(matrix, matrix_offset, dimension, second_local, second_local, second_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, second_local, second_contribution.right_hand_side)
    if first_local >= 0 and second_local >= 0:
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            first_local,
            second_local,
            wp.outer(inertia * first_jacobian, second_jacobian),
        )
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            second_local,
            first_local,
            wp.outer(inertia * second_jacobian, first_jacobian),
        )


@wp.kernel
def _assemble_structural_joint_rows(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    row_world: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    residual: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    effective_mass: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    prescribed_twist: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    joint_penalty_scale: wp.array[wp.float32],
    penalty: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    joint_row = wp.tid()
    world = row_world[joint_row]
    if world < 0 or world >= joint_penalty_scale.shape[0]:
        penalty[joint_row] = 0.0
        return
    dt = time_step[world]
    first = body_first[joint_row]
    second = body_second[joint_row]
    body_count = body_block.shape[0]
    world_penalty_scale = joint_penalty_scale[world]
    if (
        dt <= 0.0
        or world_penalty_scale <= 0.0
        or effective_mass[joint_row] <= 0.0
        or (first < 0 and second < 0)
        or first >= body_count
        or second >= body_count
    ):
        penalty[joint_row] = 0.0
        return

    first_local = body_local[first] if first >= 0 else -1
    second_local = body_local[second] if second >= 0 else -1
    if first_local < 0 and second_local < 0:
        penalty[joint_row] = 0.0
        return
    block = body_block[first] if first_local >= 0 else body_block[second]
    if block < 0 or block >= dimensions.shape[0] or (second_local >= 0 and body_block[second] != block):
        penalty[joint_row] = 0.0
        return
    dimension = dimensions[block]

    row_penalty = world_penalty_scale * effective_mass[joint_row] / (dt * dt)
    penalty[joint_row] = row_penalty
    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    first_jacobian = jacobian_first[joint_row]
    second_jacobian = jacobian_second[joint_row]
    row_residual = residual[joint_row]
    # Kamino stores the reaction applied as +J^T lambda. The augmented
    # Lagrangian dual used by the primal row has the opposite sign.
    row_multiplier = -reaction[joint_row]
    linearization_velocity = wp.float32(0.0)
    if first >= 0:
        linearization_velocity += wp.dot(first_jacobian, linearization_twist[first])
    if second >= 0:
        linearization_velocity += wp.dot(second_jacobian, linearization_twist[second])
    if prescribed_twist and first >= 0 and first_local < 0:
        linearization_velocity -= wp.dot(first_jacobian, prescribed_twist[first])
    if prescribed_twist and second >= 0 and second_local < 0:
        linearization_velocity -= wp.dot(second_jacobian, prescribed_twist[second])

    if first_local >= 0:
        first_contribution = compute_augmented_joint_row(
            first_jacobian, row_residual, row_multiplier, row_penalty, dt, linearization_velocity
        )
        _atomic_add_body_block(matrix, matrix_offset, dimension, first_local, first_local, first_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, first_local, first_contribution.right_hand_side)
    if second_local >= 0:
        second_contribution = compute_augmented_joint_row(
            second_jacobian, row_residual, row_multiplier, row_penalty, dt, linearization_velocity
        )
        _atomic_add_body_block(matrix, matrix_offset, dimension, second_local, second_local, second_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, second_local, second_contribution.right_hand_side)
    if first_local >= 0 and second_local >= 0:
        cross_scale = dt * dt * row_penalty
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            first_local,
            second_local,
            wp.outer(cross_scale * first_jacobian, second_jacobian),
        )
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            second_local,
            first_local,
            wp.outer(cross_scale * second_jacobian, first_jacobian),
        )


@wp.kernel
def _compute_body_weights_and_add(
    body_component: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    smooth_matrix: wp.array[wp.float32],
    mass: wp.array[wp.float32],
    inertia_world: wp.array[wp.mat33f],
    sigma: wp.float32,
    beta: wp.float32,
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
    eta: wp.array[wp.float32],
    alpha: wp.array[wp.float32],
    status: wp.array[wp.int32],
    weighted_matrix: wp.array[wp.float32],
):
    body = wp.tid()
    component = body_component[body]
    if component < 0 or body_has_unilateral[body] == 0:
        weight[body] = mat66f(0.0)
        inverse_weight[body] = mat66f(0.0)
        eta[body] = 0.0
        alpha[body] = 0.0
        status[body] = BODY_WEIGHT_STATUS_VALID
        return
    local_body = body_local[body]
    dimension = dimensions[component]
    matrix_offset = matrix_offsets[component]
    body_offset = 6 * local_body
    smooth_diagonal = mat66f(0.0)
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)
            smooth_diagonal[row, col] = smooth_matrix[index]

    result = compute_body_weight_mass_proportional(
        smooth_diagonal,
        mass[body],
        inertia_world[body],
        sigma,
        beta,
    )
    weight[body] = result.weight
    inverse_weight[body] = result.inverse_weight
    eta[body] = result.eta
    alpha[body] = result.alpha
    status[body] = result.status
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)
            weighted_matrix[index] += result.weight[row, col]


@wp.kernel
def _build_candidate_right_hand_side(
    body_vector_index: wp.array[wp.int32],
    smooth_right_hand_side: wp.array[wp.float32],
    weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    candidate_right_hand_side: wp.array[wp.float32],
):
    body = wp.tid()
    body_offset = body_vector_index[body]
    if body_offset < 0:
        return
    weighted_target = weight[body] @ (projected_twist[body] + splitting_dual[body])
    for row in range(6):
        candidate_right_hand_side[body_offset + row] = smooth_right_hand_side[body_offset + row] + weighted_target[row]


@wp.kernel
def _build_candidate_right_hand_side_with_effort(
    body_vector_index: wp.array[wp.int32],
    smooth_right_hand_side: wp.array[wp.float32],
    weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    body_effort_offset: wp.array[wp.int32],
    body_effort_index: wp.array[wp.int32],
    body_effort_side: wp.array[wp.int32],
    effort_dynamic_row: wp.array[wp.int32],
    dynamic_jacobian_first: wp.array[vec6f],
    dynamic_jacobian_second: wp.array[vec6f],
    effort_counter_applied: wp.array[wp.float32],
    candidate_right_hand_side: wp.array[wp.float32],
):
    body = wp.tid()
    body_offset = body_vector_index[body]
    if body_offset < 0:
        return
    target = weight[body] @ (projected_twist[body] + splitting_dual[body])
    effort_start = body_effort_offset[body]
    effort_end = body_effort_offset[body + 1]
    for incidence in range(effort_start, effort_end):
        effort = body_effort_index[incidence]
        dynamic_row = effort_dynamic_row[effort]
        jacobian = dynamic_jacobian_first[dynamic_row]
        if body_effort_side[incidence] != 0:
            jacobian = dynamic_jacobian_second[dynamic_row]
        target += effort_counter_applied[effort] * jacobian
    for row in range(6):
        candidate_right_hand_side[body_offset + row] = smooth_right_hand_side[body_offset + row] + target[row]


class BatchedPrimalBodySystem:
    """Allocated dense contact-free systems over independent body components.

    World-level bookkeeping remains in model body order. The dense matrices and
    vectors are instead packed by ``body_components`` so disconnected
    articulations can be factorized independently. If no components are
    supplied, each world is retained as one factor block.
    """

    def __init__(
        self,
        body_counts: Sequence[int],
        device: wp.DeviceLike = None,
        *,
        body_components: Sequence[Sequence[int]] | None = None,
        dynamic_bodies: Sequence[int] | None = None,
        body_edges: Sequence[tuple[int, int]] | None = None,
    ):
        if len(body_counts) == 0:
            raise ValueError("At least one world is required.")
        if any(not isinstance(count, int) or count < 0 for count in body_counts) or sum(body_counts) == 0:
            raise ValueError("Body counts must be non-negative and include at least one active body.")

        self.device = wp.get_device(device)
        self.body_counts = tuple(body_counts)
        self.num_worlds = len(body_counts)
        self.num_bodies = sum(body_counts)
        body_counts_np = np.asarray(body_counts, dtype=np.int32)
        world_body_offsets = np.empty(self.num_worlds + 1, dtype=np.int32)
        world_body_offsets[0] = 0
        np.cumsum(body_counts_np, out=world_body_offsets[1:])
        body_world_np = np.repeat(np.arange(self.num_worlds, dtype=np.int32), body_counts_np)

        if dynamic_bodies is None:
            dynamic_body_np = np.arange(self.num_bodies, dtype=np.int32)
        else:
            dynamic_body_np = np.asarray(dynamic_bodies, dtype=np.int32)
            if np.unique(dynamic_body_np).size != dynamic_body_np.size:
                raise ValueError("dynamic_bodies must not contain duplicates.")
            if np.any((dynamic_body_np < 0) | (dynamic_body_np >= self.num_bodies)):
                raise ValueError("dynamic_bodies must reference packed bodies.")
        dynamic_body_indices = tuple(dynamic_body_np.tolist())

        if body_components is None:
            dynamic_mask = np.zeros(self.num_bodies, dtype=bool)
            dynamic_mask[dynamic_body_np] = True
            component_bodies = np.flatnonzero(dynamic_mask).astype(np.int32)
            dynamic_world = body_world_np[component_bodies]
            boundaries = np.flatnonzero(dynamic_world[1:] != dynamic_world[:-1]) + 1
            component_arrays = np.split(component_bodies, boundaries) if component_bodies.size else []
        else:
            component_arrays = [np.asarray(component, dtype=np.int32) for component in body_components]
            if (not component_arrays and dynamic_body_np.size) or any(
                component.size == 0 for component in component_arrays
            ):
                raise ValueError("Each body component must contain at least one body.")
            flattened = np.concatenate(component_arrays) if component_arrays else np.empty(0, dtype=np.int32)
            if not np.array_equal(np.sort(flattened), np.sort(dynamic_body_np)):
                raise ValueError("body_components must contain every dynamic body exactly once.")
            component_counts = np.asarray([component.size for component in component_arrays], dtype=np.int32)
            component_world = body_world_np[flattened[np.cumsum(component_counts) - component_counts]]
            if np.any(body_world_np[flattened] != np.repeat(component_world, component_counts)):
                raise ValueError("A body component cannot span multiple worlds.")
        self.dynamic_bodies = dynamic_body_indices
        if component_arrays:
            block_body_counts_np = np.asarray([component.size for component in component_arrays], dtype=np.int32)
            component_offsets = np.empty(block_body_counts_np.size + 1, dtype=np.int32)
            component_offsets[0] = 0
            np.cumsum(block_body_counts_np, out=component_offsets[1:])
            flattened_components = np.concatenate(component_arrays)
            block_world_np = body_world_np[flattened_components[component_offsets[:-1]]]
            self.block_body_counts = tuple(block_body_counts_np.tolist())
            self.block_world_host = tuple(block_world_np.tolist())
            storage_dimensions_np = 6 * block_body_counts_np
            storage_dimensions = storage_dimensions_np.tolist()
            active_dimensions = storage_dimensions
        else:
            # Dense multi-linear storage requires one positive allocation size,
            # while the active dimension remains zero.
            self.block_body_counts = (0,)
            self.block_world_host = (0,)
            storage_dimensions = [1]
            active_dimensions = [0]
        self.num_blocks = len(self.block_body_counts)

        self.info = DenseSquareMultiLinearInfo()
        self.info.finalize(dimensions=storage_dimensions, dtype=wp.float32, itype=wp.int32, device=self.device)
        if active_dimensions != storage_dimensions:
            self.info.dim = wp.array(active_dimensions, dtype=wp.int32, device=self.device)
        self.smooth_matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        self.weighted_matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        self.right_hand_side = wp.zeros(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.candidate_right_hand_side = wp.zeros(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self._packed_solution = wp.zeros(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.body_solution = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.weight = wp.zeros(self.num_bodies, dtype=mat66f, device=self.device)
        self.inverse_weight = wp.zeros(self.num_bodies, dtype=mat66f, device=self.device)
        self.weight_eta = wp.zeros(self.num_bodies, dtype=wp.float32, device=self.device)
        self.weight_alpha = wp.zeros(self.num_bodies, dtype=wp.float32, device=self.device)
        self.weight_status = wp.zeros(self.num_bodies, dtype=wp.int32, device=self.device)
        body_block_np = np.full(self.num_bodies, -1, dtype=np.int32)
        body_local_np = np.full(self.num_bodies, -1, dtype=np.int32)
        body_vector_index_np = np.full(self.num_bodies, -1, dtype=np.int32)
        if component_arrays:
            block_indices = np.repeat(np.arange(self.num_blocks, dtype=np.int32), block_body_counts_np)
            local_indices = np.arange(flattened_components.size, dtype=np.int32) - np.repeat(
                component_offsets[:-1], block_body_counts_np
            )
            block_vector_offsets = 6 * component_offsets[:-1]
            body_block_np[flattened_components] = block_indices
            body_local_np[flattened_components] = local_indices
            body_vector_index_np[flattened_components] = (
                np.repeat(block_vector_offsets, block_body_counts_np) + 6 * local_indices
            )

        self.body_world_host = tuple(body_world_np.tolist())
        self.body_block_host = tuple(body_block_np.tolist())
        self.body_local_host = tuple(body_local_np.tolist())
        self.body_vector_index_host = tuple(body_vector_index_np.tolist())
        body_edges_np = np.asarray(body_edges if body_edges is not None else (), dtype=np.int32).reshape((-1, 2))
        body_edges_np = body_edges_np[np.all(body_edges_np >= 0, axis=1)]
        body_edges_np.sort(axis=1)
        body_edges_np = np.unique(body_edges_np, axis=0)
        self.body_edges_host = frozenset(map(tuple, body_edges_np.tolist()))
        self._body_edge_keys_host = body_edges_np[:, 0].astype(np.int64) * self.num_bodies + body_edges_np[:, 1]
        self.body_world = wp.array(body_world_np, dtype=wp.int32, device=self.device)
        self.body_block = wp.array(body_block_np, dtype=wp.int32, device=self.device)
        self.body_local = wp.array(body_local_np, dtype=wp.int32, device=self.device)
        self.body_vector_index = wp.array(body_vector_index_np, dtype=wp.int32, device=self.device)
        self.body_weight_enabled = wp.ones(self.num_bodies, dtype=wp.int32, device=self.device)
        self.block_has_unilateral = wp.ones(self.num_blocks, dtype=wp.int32, device=self.device)
        self.selective_body_weights = False
        self.operator = DenseLinearOperatorData(info=self.info, mat=self.weighted_matrix)
        symbolic_adjacency = self._build_symbolic_adjacency(body_edges) if body_edges is not None else None
        # Factorization benefits from fewer wide dense panels, while repeated
        # single-RHS solves retain the smaller tile for better occupancy.
        self.linear_solver = HybridLLTBlockedSolver(
            operator=self.operator,
            factorize_block_size=64,
            solve_block_dim=256,
            rcm_min_dimension=_RCM_MIN_DIMENSION,
            symbolic_adjacency=symbolic_adjacency,
            dtype=wp.float32,
            device=self.device,
        )
        self._mass: wp.array[wp.float32] | None = None
        self._inertia_world: wp.array[wp.mat33f] | None = None

    def _build_symbolic_adjacency(
        self, body_edges: Sequence[tuple[int, int]]
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Expand fixed body topology into scalar adjacency per factor block."""
        edges = np.asarray(body_edges, dtype=np.int32).reshape((-1, 2))
        if np.any((edges < 0) | (edges >= self.num_bodies)):
            raise ValueError("body_edges must reference packed bodies.")
        body_block = np.asarray(self.body_block_host, dtype=np.int32)
        body_local = np.asarray(self.body_local_host, dtype=np.int32)
        if edges.size:
            edge_blocks = body_block[edges]
            dynamic_edge = np.all(edge_blocks >= 0, axis=1)
            if np.any(edge_blocks[dynamic_edge, 0] != edge_blocks[dynamic_edge, 1]):
                raise ValueError("A body edge cannot span factor blocks.")
            edges = edges[dynamic_edge]
            edge_block = edge_blocks[dynamic_edge, 0]
            local_edges = body_local[edges]
            local_edges.sort(axis=1)
            order = np.lexsort((local_edges[:, 1], local_edges[:, 0], edge_block))
            edge_block = edge_block[order]
            local_edges = local_edges[order]
            edge_counts = np.bincount(edge_block, minlength=self.num_blocks).astype(np.int32, copy=False)
            edge_offsets = np.empty(self.num_blocks + 1, dtype=np.int32)
            edge_offsets[0] = 0
            np.cumsum(edge_counts, out=edge_offsets[1:])
        else:
            local_edges = edges
            edge_offsets = np.zeros(self.num_blocks + 1, dtype=np.int32)

        adjacency_blocks = []
        for block, body_count in enumerate(self.block_body_counts):
            if 6 * body_count < _RCM_MIN_DIMENSION:
                adjacency_blocks.append(())
                continue
            begin = edge_offsets[block]
            end = edge_offsets[block + 1]
            block_edges = tuple(map(tuple, local_edges[begin:end].tolist()))
            adjacency_blocks.append(_expand_body_symbolic_adjacency(body_count, block_edges))
        return tuple(adjacency_blocks)

    def validate_body_pairs(
        self,
        name: str,
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
    ) -> None:
        """Verify fixed topology covers every dynamic off-diagonal assembly pair."""
        first_values = body_first.numpy().astype(np.int32, copy=False)
        second_values = body_second.numpy().astype(np.int32, copy=False)
        if first_values.size != second_values.size:
            raise ValueError(f"{name} endpoint arrays must have identical lengths.")
        body_block = np.asarray(self.body_block_host, dtype=np.int32)
        valid = (first_values >= 0) & (second_values >= 0)
        valid &= body_block[first_values.clip(min=0)] >= 0
        valid &= body_block[second_values.clip(min=0)] >= 0
        pairs = np.column_stack((first_values[valid], second_values[valid]))
        pairs.sort(axis=1)
        pair_keys = pairs[:, 0].astype(np.int64) * self.num_bodies + pairs[:, 1]
        missing = np.unique(pairs[~np.isin(pair_keys, self._body_edge_keys_host)], axis=0)
        if missing.size:
            raise ValueError(f"Fixed body topology does not cover {name} pairs: {list(map(tuple, missing.tolist()))}.")

    def reset(self) -> None:
        """Clear assembled matrices, vectors, weights, and factorization state."""
        self.smooth_matrix.zero_()
        self.weighted_matrix.zero_()
        self.right_hand_side.zero_()
        self.candidate_right_hand_side.zero_()
        self._packed_solution.zero_()
        self.body_solution.zero_()
        self.weight.zero_()
        self.inverse_weight.zero_()
        self.weight_eta.zero_()
        self.weight_alpha.zero_()
        self.weight_status.zero_()
        self.body_weight_enabled.zero_()
        self.block_has_unilateral.zero_()
        self.linear_solver.reset()

    def assemble_bodies(
        self,
        mass: wp.array[wp.float32],
        inertia_world: wp.array[wp.mat33f],
        velocity_previous: wp.array[vec6f],
        evaluation_velocity: wp.array[vec6f],
        external_wrench: wp.array[vec6f],
        actuation_wrench: wp.array[vec6f],
        gravity: wp.array[wp.vec3f],
        time_step: wp.array[wp.float32],
    ) -> None:
        """Reset and assemble body inertia and explicit-force terms."""
        if any(
            array.shape[0] != self.num_bodies
            for array in (
                mass,
                inertia_world,
                velocity_previous,
                evaluation_velocity,
                external_wrench,
                actuation_wrench,
            )
        ):
            raise ValueError("Body input arrays must contain one entry per packed active body.")
        self.reset()
        self._mass = mass
        self._inertia_world = inertia_world
        wp.launch(
            _assemble_body_inertial_systems,
            dim=self.num_bodies,
            inputs=[
                self.body_world,
                self.body_block,
                self.body_local,
                self.info.dim,
                self.info.mio,
                self.info.vio,
                mass,
                inertia_world,
                velocity_previous,
                evaluation_velocity,
                external_wrench,
                actuation_wrench,
                gravity,
                time_step,
            ],
            outputs=[self.smooth_matrix, self.right_hand_side],
            device=self.device,
        )

    def add_dynamic_rows(
        self,
        row_world: wp.array[wp.int32],
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        effective_inertia: wp.array[wp.float32],
        free_velocity: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add implicit joint-dynamics rows to the smooth system."""
        row_count = row_world.shape[0]
        if row_count == 0:
            return
        if any(
            array.shape[0] != row_count
            for array in (
                body_first,
                body_second,
                jacobian_first,
                jacobian_second,
                effective_inertia,
                free_velocity,
            )
        ):
            raise ValueError("Dynamic-row arrays must have identical lengths.")
        if prescribed_twist is not None and prescribed_twist.shape[0] != self.num_bodies:
            raise ValueError("prescribed_twist must contain one entry per packed body.")
        wp.launch(
            _assemble_dynamic_joint_rows,
            dim=row_count,
            inputs=[
                self.info.dim,
                self.info.mio,
                self.info.vio,
                body_first,
                body_second,
                self.body_block,
                self.body_local,
                jacobian_first,
                jacobian_second,
                effective_inertia,
                free_velocity,
                prescribed_twist,
            ],
            outputs=[self.smooth_matrix, self.right_hand_side],
            device=self.device,
        )

    def add_structural_rows(
        self,
        row_world: wp.array[wp.int32],
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        residual: wp.array[wp.float32],
        reaction: wp.array[wp.float32],
        effective_mass: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
        time_step: wp.array[wp.float32],
        joint_penalty_scale: wp.array[wp.float32],
        penalty: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add augmented structural rows and write their derived penalties."""
        row_count = row_world.shape[0]
        if row_count == 0:
            return
        if joint_penalty_scale.shape[0] != self.num_worlds:
            raise ValueError("joint_penalty_scale must contain one entry per world.")
        if linearization_twist.shape[0] != self.num_bodies:
            raise ValueError("linearization_twist must contain one entry per packed body.")
        if prescribed_twist is not None and prescribed_twist.shape[0] != self.num_bodies:
            raise ValueError("prescribed_twist must contain one entry per packed body.")
        if any(
            array.shape[0] != row_count
            for array in (
                body_first,
                body_second,
                jacobian_first,
                jacobian_second,
                residual,
                reaction,
                effective_mass,
                penalty,
            )
        ):
            raise ValueError("Structural-row arrays must have identical lengths.")
        wp.launch(
            _assemble_structural_joint_rows,
            dim=row_count,
            inputs=[
                self.info.dim,
                self.info.mio,
                self.info.vio,
                row_world,
                body_first,
                body_second,
                self.body_block,
                self.body_local,
                jacobian_first,
                jacobian_second,
                residual,
                reaction,
                effective_mass,
                linearization_twist,
                prescribed_twist,
                time_step,
                joint_penalty_scale,
            ],
            outputs=[penalty, self.smooth_matrix, self.right_hand_side],
            device=self.device,
        )

    def _mark_weighted_bodies(self, body_has_unilateral: wp.array[wp.int32] | None) -> None:
        if body_has_unilateral is None:
            self.body_weight_enabled.fill_(1)
            return
        if body_has_unilateral.shape[0] != self.num_bodies:
            raise ValueError("body_has_unilateral must contain one entry per body.")
        if self.selective_body_weights:
            wp.copy(self.body_weight_enabled, body_has_unilateral)
            return
        self.block_has_unilateral.zero_()
        wp.launch(
            _mark_blocks_with_unilaterals,
            dim=self.num_bodies,
            inputs=[self.body_block, body_has_unilateral],
            outputs=[self.block_has_unilateral],
            device=self.device,
        )
        wp.launch(
            _enable_bodies_in_weighted_blocks,
            dim=self.num_bodies,
            inputs=[self.body_block, self.block_has_unilateral],
            outputs=[self.body_weight_enabled],
            device=self.device,
        )

    def build_weighted_matrix(
        self,
        *,
        sigma: float,
        beta: float,
        metric_matrix: wp.array[wp.float32] | None = None,
        body_has_unilateral: wp.array[wp.int32] | None = None,
    ) -> None:
        """Compute body weights and assemble ``A + W``."""
        if self._mass is None or self._inertia_world is None:
            raise ValueError("assemble_bodies() must be called before build_weighted_matrix().")
        if metric_matrix is None:
            metric_matrix = self.smooth_matrix
        elif metric_matrix.shape[0] != self.info.total_mat_size:
            raise ValueError("metric_matrix must match the packed body matrix storage.")
        self._mark_weighted_bodies(body_has_unilateral)
        wp.copy(self.weighted_matrix, self.smooth_matrix)
        wp.launch(
            _compute_body_weights_and_add,
            dim=self.num_bodies,
            inputs=[
                self.body_block,
                self.body_local,
                self.body_weight_enabled,
                self.info.dim,
                self.info.mio,
                metric_matrix,
                self._mass,
                self._inertia_world,
                sigma,
                beta,
            ],
            outputs=[
                self.weight,
                self.inverse_weight,
                self.weight_eta,
                self.weight_alpha,
                self.weight_status,
                self.weighted_matrix,
            ],
            device=self.device,
        )

    def factorize(self) -> None:
        """Factorize the current weighted matrices with batched dense LLT."""
        self.linear_solver.compute(self.weighted_matrix)

    def build_candidate_right_hand_side(
        self,
        projected_twist: wp.array[vec6f],
        splitting_dual: wp.array[vec6f],
    ) -> None:
        """Build ``f + W (p + lambda)`` for one splitting iteration."""
        if projected_twist.shape[0] != self.num_bodies or splitting_dual.shape[0] != self.num_bodies:
            raise ValueError("Splitting vectors must contain one entry per packed active body.")
        wp.launch(
            _build_candidate_right_hand_side,
            dim=self.num_bodies,
            inputs=[
                self.body_vector_index,
                self.right_hand_side,
                self.weight,
                projected_twist,
                splitting_dual,
            ],
            outputs=[self.candidate_right_hand_side],
            device=self.device,
        )

    def build_candidate_right_hand_side_with_effort(
        self,
        projected_twist: wp.array[vec6f],
        splitting_dual: wp.array[vec6f],
        body_effort_offset: wp.array[wp.int32],
        body_effort_index: wp.array[wp.int32],
        body_effort_side: wp.array[wp.int32],
        effort_dynamic_row: wp.array[wp.int32],
        dynamic_jacobian_first: wp.array[vec6f],
        dynamic_jacobian_second: wp.array[vec6f],
        effort_counter_applied: wp.array[wp.float32],
    ) -> None:
        """Build a candidate right-hand side with finite-drive counter-impulses."""
        if projected_twist.shape[0] != self.num_bodies or splitting_dual.shape[0] != self.num_bodies:
            raise ValueError("Splitting vectors must contain one entry per packed active body.")
        if body_effort_offset.shape[0] != self.num_bodies + 1:
            raise ValueError("Effort offsets must contain one interval per packed active body.")
        wp.launch(
            _build_candidate_right_hand_side_with_effort,
            dim=self.num_bodies,
            inputs=[
                self.body_vector_index,
                self.right_hand_side,
                self.weight,
                projected_twist,
                splitting_dual,
                body_effort_offset,
                body_effort_index,
                body_effort_side,
                effort_dynamic_row,
                dynamic_jacobian_first,
                dynamic_jacobian_second,
                effort_counter_applied,
            ],
            outputs=[self.candidate_right_hand_side],
            device=self.device,
        )

    def solve_candidate(
        self,
        prescribed_twist: wp.array[vec6f],
    ) -> None:
        """Solve the prepared candidate right-hand side and unpack body velocities."""
        if prescribed_twist.shape[0] != self.num_bodies:
            raise ValueError("prescribed_twist must contain one entry per packed body.")
        self.linear_solver.solve(self.candidate_right_hand_side, self._packed_solution)
        wp.launch(
            _unpack_body_solution,
            dim=self.num_bodies,
            inputs=[self.body_vector_index, self._packed_solution, prescribed_twist],
            outputs=[self.body_solution],
            device=self.device,
        )
