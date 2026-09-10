# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared unilateral preparation, projection kernels, and residuals.

Contact-facing functions use Kamino's normal-last order
``(tangent_x, tangent_y, normal_z)``. Body twists and Jacobian columns use
Kamino's linear-first 6D convention.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import compute_contact_scaled_alart_curnier_residual, solve_contact_coulomb_newton

__all__ = [
    "PROJECTION_STATUS_INVALID",
    "PROJECTION_STATUS_VALID",
    "compute_contact_delassus",
    "compute_limit_delassus",
    "prepare_contact_coulomb_delassus",
    "project_contact_coulomb_local",
    "project_friction_local",
    "project_limit_local",
]

PROJECTION_STATUS_INVALID = 0
"""The local block or an input value was non-finite or non-positive."""

PROJECTION_STATUS_VALID = 1
"""The one-constraint update completed successfully."""

wp.set_module_options({"enable_backward": False})

_FUSED_RIGID_WORLD_MIN_BLOCKS_PER_SM = 4


def _can_fuse_rigid_projection_by_world(
    device: wp.Device,
    world_count: int,
    *,
    required_world_arrays: tuple[wp.array[Any] | None, ...],
    parallel_constraint_capacity: int | None = None,
    world_block_dim: int | None = None,
    minimum_blocks_per_sm: int = _FUSED_RIGID_WORLD_MIN_BLOCKS_PER_SM,
) -> bool:
    """Return whether launch fusion or world-level occupancy favors one block per world."""
    enough_world_blocks = device.is_cuda and world_count >= device.sm_count * minimum_blocks_per_sm
    small_world_work = (
        parallel_constraint_capacity is not None
        and world_block_dim is not None
        and world_count > 0
        and parallel_constraint_capacity <= world_count * world_block_dim
    )
    return (
        device.is_cuda
        and (enough_world_blocks or small_world_work)
        and all(array is not None for array in required_world_arrays)
    )


@wp.struct
class ContactProjectionData:
    """Contact data that is invariant throughout a body-space solve."""

    delassus: wp.mat33f
    status: wp.int32


@wp.struct
class ScalarProjectionResult:
    """Minimal result of a prepared scalar projection."""

    reaction: wp.float32
    reaction_delta: wp.float32
    status: wp.int32


@wp.struct
class ContactProjectionResult:
    """Minimal result of a prepared Coulomb projection."""

    reaction: wp.vec3f
    reaction_delta: wp.vec3f
    status: wp.int32


@wp.func
def compute_contact_delassus(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
) -> wp.mat33f:
    """Compute the full normal-last local contact Delassus block."""
    return jacobian_first @ inverse_weight_first @ wp.transpose(
        jacobian_first
    ) + jacobian_second @ inverse_weight_second @ wp.transpose(jacobian_second)


@wp.func
def prepare_contact_coulomb_delassus(
    delassus: wp.mat33f,
    velocity_bias: wp.vec3f,
    friction: wp.float32,
) -> ContactProjectionData:
    """Validate and, when numerically marginal, regularize a contact block."""
    result = ContactProjectionData()
    result.delassus = delassus
    result.status = PROJECTION_STATUS_INVALID
    if not wp.isfinite(velocity_bias) or not wp.isfinite(friction) or friction < 0.0:
        return result
    if not wp.isfinite(delassus):
        return result

    scale = wp.float32(0.0)
    asymmetry = wp.float32(0.0)
    symmetric = wp.mat33f(0.0)
    for row in range(3):
        for col in range(3):
            scale = wp.max(scale, wp.abs(delassus[row, col]))
            asymmetry = wp.max(asymmetry, wp.abs(delassus[row, col] - delassus[col, row]))
            symmetric[row, col] = 0.5 * (delassus[row, col] + delassus[col, row])
    if scale <= 0.0 or asymmetry > 1.0e-5 * scale:
        return result

    eigenvectors, eigenvalues = wp.eig3(symmetric)
    if not wp.isfinite(eigenvalues):
        return result
    minimum_eigenvalue = wp.min(eigenvalues)
    if minimum_eigenvalue < -1.0e-5 * scale:
        return result

    eigenvalue_floor = 1.0e-6 * scale
    clamped_eigenvalues = wp.max(eigenvalues, wp.vec3f(eigenvalue_floor))
    regularized = minimum_eigenvalue < eigenvalue_floor
    if regularized:
        symmetric = eigenvectors @ wp.diag(clamped_eigenvalues) @ wp.transpose(eigenvectors)
    result.status = PROJECTION_STATUS_VALID
    result.delassus = symmetric
    return result


@wp.func
def _contact_projection_inputs_are_finite(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
) -> wp.bool:
    return (
        wp.isfinite(jacobian_first)
        and wp.isfinite(inverse_weight_first)
        and wp.isfinite(jacobian_second)
        and wp.isfinite(inverse_weight_second)
    )


@wp.func
def prepare_contact_coulomb(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
    velocity_bias: wp.vec3f,
    friction: wp.float32,
) -> ContactProjectionData:
    """Validate fixed inputs and prepare the normal-last contact block."""
    result = prepare_contact_coulomb_delassus(
        compute_contact_delassus(
            jacobian_first,
            inverse_weight_first,
            jacobian_second,
            inverse_weight_second,
        ),
        velocity_bias,
        friction,
    )
    if not _contact_projection_inputs_are_finite(
        jacobian_first,
        inverse_weight_first,
        jacobian_second,
        inverse_weight_second,
    ):
        result.status = PROJECTION_STATUS_INVALID
    return result


@wp.func
def compute_limit_delassus(
    jacobian_first: vec6f,
    inverse_weight_first: mat66f,
    jacobian_second: vec6f,
    inverse_weight_second: mat66f,
) -> wp.float32:
    """Compute the scalar local joint-limit Delassus coefficient."""
    return wp.dot(jacobian_first, inverse_weight_first @ jacobian_first) + wp.dot(
        jacobian_second, inverse_weight_second @ jacobian_second
    )


@wp.func
def project_limit_local(
    current_velocity: wp.float32,
    reaction_old: wp.float32,
    delassus: wp.float32,
) -> ScalarProjectionResult:
    """Project one prepared scalar unilateral constraint."""
    result = ScalarProjectionResult()
    result.reaction = reaction_old
    result.reaction_delta = 0.0
    result.status = PROJECTION_STATUS_INVALID
    if (
        not wp.isfinite(current_velocity)
        or not wp.isfinite(reaction_old)
        or not wp.isfinite(delassus)
        or delassus <= 0.0
    ):
        return result

    free_velocity = current_velocity - delassus * reaction_old
    reaction_new = wp.max(-free_velocity / delassus, 0.0)
    reaction_delta = reaction_new - reaction_old
    if not wp.isfinite(reaction_new) or not wp.isfinite(reaction_delta):
        return result
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.status = PROJECTION_STATUS_VALID
    return result


@wp.func
def project_friction_local(
    current_velocity: wp.float32,
    reaction_old: wp.float32,
    delassus: wp.float32,
    impulse_bound: wp.float32,
) -> ScalarProjectionResult:
    """Project one prepared bounded scalar friction constraint."""
    result = ScalarProjectionResult()
    result.reaction = reaction_old
    result.reaction_delta = 0.0
    result.status = PROJECTION_STATUS_INVALID
    if (
        not wp.isfinite(current_velocity)
        or not wp.isfinite(reaction_old)
        or not wp.isfinite(delassus)
        or delassus <= 0.0
        or not wp.isfinite(impulse_bound)
        or impulse_bound < 0.0
    ):
        return result

    free_velocity = current_velocity - delassus * reaction_old
    reaction_new = wp.clamp(-free_velocity / delassus, -impulse_bound, impulse_bound)
    reaction_delta = reaction_new - reaction_old
    if not wp.isfinite(reaction_new) or not wp.isfinite(reaction_delta):
        return result
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.status = PROJECTION_STATUS_VALID
    return result


@wp.func
def project_contact_coulomb_local(
    current_velocity: wp.vec3f,
    reaction_old: wp.vec3f,
    delassus: wp.mat33f,
    friction: wp.float32,
) -> ContactProjectionResult:
    """Project one prepared normal-last Coulomb contact."""
    result = ContactProjectionResult()
    result.reaction = reaction_old
    result.reaction_delta = wp.vec3f(0.0)
    result.status = PROJECTION_STATUS_INVALID
    free_velocity = current_velocity - delassus @ reaction_old
    if not wp.isfinite(free_velocity):
        return result

    reaction_new = solve_contact_coulomb_newton(delassus, free_velocity, friction)
    reaction_delta = reaction_new - reaction_old
    if not wp.isfinite(reaction_new) or not wp.isfinite(reaction_delta):
        return result
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.status = PROJECTION_STATUS_VALID
    return result


@wp.struct
class _ProjectionIndexData:
    constraint_world: wp.array[wp.int32]
    constraint_local: wp.array[wp.int32]
    world_constraint_count: wp.array[wp.int32]
    color_counts: wp.array[wp.int32]
    color_offsets: wp.array[wp.int32]
    order: wp.array[wp.int32]


@wp.struct
class _RigidProjectionState:
    """Array descriptors; body values remain in their original SoA buffers."""

    world_active: wp.array[wp.bool]
    occupancy: wp.array2d[wp.int32]
    inverse_weight: wp.array[mat66f]
    projected_twist: wp.array[vec6f]
    twist_delta: wp.array[vec6f]
    world_status: wp.array[wp.int32]


@wp.struct
class _RigidContactProjectionData:
    """Array descriptors; constructing this view does not pack contact values."""

    body_first: wp.array[wp.int32]
    body_second: wp.array[wp.int32]
    jacobian_first: wp.array[mat36f]
    jacobian_second: wp.array[mat36f]
    delassus: wp.array[wp.mat33f]
    bias: wp.array[wp.vec3f]
    friction: wp.array[wp.float32]
    reaction: wp.array[wp.vec3f]


def _make_direct_projection_index(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32] | None = None,
    world_constraint_count: wp.array[wp.int32] | None = None,
) -> _ProjectionIndexData:
    index = _ProjectionIndexData()
    index.constraint_world = constraint_world
    if constraint_local is not None:
        index.constraint_local = constraint_local
    if world_constraint_count is not None:
        index.world_constraint_count = world_constraint_count
    return index


def _make_colored_projection_index(
    constraint_world: wp.array[wp.int32],
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
) -> _ProjectionIndexData:
    index = _ProjectionIndexData()
    index.constraint_world = constraint_world
    index.color_counts = color_counts
    index.color_offsets = color_offsets
    index.order = order
    return index


def _make_rigid_projection_state(
    world_active: wp.array[wp.bool],
    projected_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32] | None = None,
    inverse_weight: wp.array[mat66f] | None = None,
) -> _RigidProjectionState:
    state = _RigidProjectionState()
    state.world_active = world_active
    state.projected_twist = projected_twist
    state.twist_delta = twist_delta
    state.world_status = world_status
    if occupancy is not None:
        state.occupancy = occupancy
    if inverse_weight is not None:
        state.inverse_weight = inverse_weight
    return state


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.kernel
def _prepare_contact_physical_projection_data(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    physical_delassus: wp.array[wp.mat33f],
    prepared_delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if contact_local[contact] >= world_contact_count[world]:
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        physical_delassus[contact] = wp.mat33f(0.0)
        prepared_delassus[contact] = wp.mat33f(0.0)
        return
    if first >= 0:
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        inverse_weight_second = inverse_weight[second]
    jacobian_first = contact_jacobian_first[contact]
    jacobian_second = contact_jacobian_second[contact]
    physical = compute_contact_delassus(
        jacobian_first,
        inverse_weight_first,
        jacobian_second,
        inverse_weight_second,
    )
    data = prepare_contact_coulomb_delassus(
        physical,
        contact_bias[contact],
        contact_friction[contact],
    )
    if not _contact_projection_inputs_are_finite(
        jacobian_first,
        inverse_weight_first,
        jacobian_second,
        inverse_weight_second,
    ):
        data.status = PROJECTION_STATUS_INVALID
    physical_delassus[contact] = physical
    prepared_delassus[contact] = data.delassus
    if data.status == PROJECTION_STATUS_INVALID:
        world_status[world] = data.status


@wp.kernel
def _prepare_scalar_projection_data(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        return

    first = body_first[constraint]
    second = body_second[constraint]
    if first < 0 and second < 0:
        delassus[constraint] = 0.0
        return
    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    if first >= 0:
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        inverse_weight_second = inverse_weight[second]
    value = compute_limit_delassus(
        jacobian_first[constraint],
        inverse_weight_first,
        jacobian_second[constraint],
        inverse_weight_second,
    )
    delassus[constraint] = value
    if not wp.isfinite(value) or value <= 0.0:
        world_status[world] = PROJECTION_STATUS_INVALID


@wp.func
def _atomic_add_twist(values: wp.array[vec6f], body: wp.int32, increment: vec6f):
    if body >= 0:
        wp.atomic_add(values, body, increment)


@wp.func
def _accumulate_twist_by_occupancy(
    values: wp.array[vec6f],
    occupancy: wp.array2d[wp.int32],
    body: wp.int32,
    color: wp.int32,
    increment: vec6f,
):
    if occupancy[body, color] == 1:
        values[body] += increment
    else:
        wp.atomic_add(values, body, increment)


@wp.func
def _is_zero_vec3(value: wp.vec3f) -> wp.bool:
    return value[0] == 0.0 and value[1] == 0.0 and value[2] == 0.0


@wp.func
def _compute_contact_velocity(
    contact: wp.int32,
    first: wp.int32,
    second: wp.int32,
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    projected_twist: wp.array[vec6f],
) -> wp.vec3f:
    velocity = contact_bias[contact]
    if first >= 0:
        velocity += contact_jacobian_first[contact] @ projected_twist[first]
    if second >= 0:
        velocity += contact_jacobian_second[contact] @ projected_twist[second]
    return velocity


@cache
def _make_accumulate_projection_delta(colored: bool):
    """Apply weights per color, or defer them until Jacobi body accumulation."""

    @wp.func
    def accumulate(
        first: wp.int32,
        second: wp.int32,
        target_color: wp.int32,
        correction_first: vec6f,
        correction_second: vec6f,
        state: _RigidProjectionState,
    ):
        if wp.static(colored):
            if first >= 0:
                correction_first = state.inverse_weight[first] @ correction_first
            if second >= 0:
                correction_second = state.inverse_weight[second] @ correction_second
            if first >= 0:
                if first == second:
                    _accumulate_twist_by_occupancy(
                        state.twist_delta,
                        state.occupancy,
                        first,
                        target_color,
                        correction_first + correction_second,
                    )
                else:
                    _accumulate_twist_by_occupancy(
                        state.twist_delta,
                        state.occupancy,
                        first,
                        target_color,
                        correction_first,
                    )
            if second >= 0 and second != first:
                _accumulate_twist_by_occupancy(
                    state.twist_delta,
                    state.occupancy,
                    second,
                    target_color,
                    correction_second,
                )
        else:
            _atomic_add_twist(state.twist_delta, first, correction_first)
            _atomic_add_twist(state.twist_delta, second, correction_second)

    return accumulate


@cache
def _make_project_rigid_contact(colored: bool):
    """Share local projection and specialize only accumulation."""
    accumulate = _make_accumulate_projection_delta(colored)

    @wp.func
    def project_contact(
        contact: wp.int32,
        world: wp.int32,
        target_color: wp.int32,
        data: _RigidContactProjectionData,
        delassus: wp.array[wp.mat33f],
        state: _RigidProjectionState,
    ):
        first = data.body_first[contact]
        second = data.body_second[contact]
        if first < 0 and second < 0:
            data.reaction[contact] = wp.vec3f(0.0)
            return
        velocity = _compute_contact_velocity(
            contact,
            first,
            second,
            data.jacobian_first,
            data.jacobian_second,
            data.bias,
            state.projected_twist,
        )
        reaction_old = data.reaction[contact]
        if _is_zero_vec3(reaction_old) and velocity[2] >= 0.0 and wp.isfinite(velocity):
            return
        projection = project_contact_coulomb_local(
            velocity,
            reaction_old,
            delassus[contact],
            data.friction[contact],
        )
        if projection.status == PROJECTION_STATUS_INVALID:
            state.world_status[world] = PROJECTION_STATUS_INVALID
            return
        data.reaction[contact] = projection.reaction
        if _is_zero_vec3(projection.reaction_delta):
            return
        correction_first = vec6f(0.0)
        correction_second = vec6f(0.0)
        if first >= 0:
            correction_first = wp.transpose(data.jacobian_first[contact]) @ projection.reaction_delta
        if second >= 0:
            correction_second = wp.transpose(data.jacobian_second[contact]) @ projection.reaction_delta
        accumulate(first, second, target_color, correction_first, correction_second, state)

    return project_contact


@cache
def _make_project_rigid_scalar(colored: bool, limit: bool):
    """Share local projection and specialize only accumulation."""
    accumulate = _make_accumulate_projection_delta(colored)

    @wp.func
    def project_scalar(
        constraint: wp.int32,
        world: wp.int32,
        target_color: wp.int32,
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        bias: wp.array[wp.float32],
        bound: wp.array[wp.float32],
        delassus: wp.array[wp.float32],
        reaction: wp.array[wp.float32],
        state: _RigidProjectionState,
    ):
        first = body_first[constraint]
        second = body_second[constraint]
        if first < 0 and second < 0:
            reaction[constraint] = 0.0
            return
        velocity = wp.float32(0.0)
        if wp.static(limit):
            velocity = bias[constraint]
        if first >= 0:
            velocity += wp.dot(jacobian_first[constraint], state.projected_twist[first])
        if second >= 0:
            velocity += wp.dot(jacobian_second[constraint], state.projected_twist[second])
        if wp.static(limit):
            projection = project_limit_local(velocity, reaction[constraint], delassus[constraint])
        else:
            projection = project_friction_local(
                velocity,
                reaction[constraint],
                delassus[constraint],
                bound[constraint],
            )
        if projection.status == PROJECTION_STATUS_INVALID:
            state.world_status[world] = PROJECTION_STATUS_INVALID
            return
        correction_first = vec6f(0.0)
        correction_second = vec6f(0.0)
        if first >= 0:
            correction_first = jacobian_first[constraint] * projection.reaction_delta
        if second >= 0:
            correction_second = jacobian_second[constraint] * projection.reaction_delta
        accumulate(first, second, target_color, correction_first, correction_second, state)
        reaction[constraint] = projection.reaction

    return project_scalar


_project_rigid_contact_colored = _make_project_rigid_contact(True)
_project_rigid_contact_jacobi = _make_project_rigid_contact(False)
_project_rigid_friction_colored = _make_project_rigid_scalar(True, False)
_project_rigid_limit_colored = _make_project_rigid_scalar(True, True)
_project_rigid_friction_jacobi = _make_project_rigid_scalar(False, False)
_project_rigid_limit_jacobi = _make_project_rigid_scalar(False, True)


@wp.kernel
def _prepare_contacts_jacobi(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if contact_local[contact] >= world_contact_count[world]:
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        delassus[contact] = wp.mat33f(0.0)
        return
    if first >= 0:
        multiplicity = wp.max(1, body_constraint_count[first] - static_body_constraint_count[first])
        inverse_weight_first = wp.float32(multiplicity) * inverse_weight[first]
    if second >= 0:
        multiplicity = wp.max(1, body_constraint_count[second] - static_body_constraint_count[second])
        inverse_weight_second = wp.float32(multiplicity) * inverse_weight[second]
    data = prepare_contact_coulomb(
        contact_jacobian_first[contact],
        inverse_weight_first,
        contact_jacobian_second[contact],
        inverse_weight_second,
        contact_bias[contact],
        contact_friction[contact],
    )
    delassus[contact] = data.delassus
    if data.status == PROJECTION_STATUS_INVALID:
        world_status[world] = data.status


@wp.kernel
def _prepare_limits_jacobi(
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    limit = wp.tid()
    world = limit_world[limit]
    if limit_local[limit] >= world_limit_count[world]:
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = limit_body_first[limit]
    second = limit_body_second[limit]
    if first < 0 and second < 0:
        delassus[limit] = 0.0
        return
    if first >= 0:
        multiplicity = wp.max(1, body_constraint_count[first] - static_body_constraint_count[first])
        inverse_weight_first = wp.float32(multiplicity) * inverse_weight[first]
    if second >= 0:
        multiplicity = wp.max(1, body_constraint_count[second] - static_body_constraint_count[second])
        inverse_weight_second = wp.float32(multiplicity) * inverse_weight[second]
    value = compute_limit_delassus(
        limit_jacobian_first[limit],
        inverse_weight_first,
        limit_jacobian_second[limit],
        inverse_weight_second,
    )
    delassus[limit] = value
    if not wp.isfinite(value) or value <= 0.0:
        world_status[world] = PROJECTION_STATUS_INVALID


@wp.kernel
def _prepare_frictions_jacobi(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    friction = wp.tid()
    world = friction_world[friction]
    if friction_local[friction] >= world_friction_count[world]:
        return
    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = friction_body_first[friction]
    second = friction_body_second[friction]
    if first >= 0:
        multiplicity = wp.max(1, body_constraint_count[first] - static_body_constraint_count[first])
        inverse_weight_first = wp.float32(multiplicity) * inverse_weight[first]
    if second >= 0:
        multiplicity = wp.max(1, body_constraint_count[second] - static_body_constraint_count[second])
        inverse_weight_second = wp.float32(multiplicity) * inverse_weight[second]
    value = compute_limit_delassus(
        friction_jacobian_first[friction],
        inverse_weight_first,
        friction_jacobian_second[friction],
        inverse_weight_second,
    )
    delassus[friction] = value
    if not wp.isfinite(value) or value <= 0.0:
        world_status[world] = PROJECTION_STATUS_INVALID


@wp.kernel
def _initialize_jacobi_projection_status(
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if world_active[world]:
        world_status[world] = prepared_status[world]


@wp.kernel
def _warmstart_contacts_jacobi(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    inverse_weight: wp.array[mat66f],
    apply_inverse_weight: wp.bool,
    reaction: wp.array[wp.vec3f],
    twist_delta: wp.array[vec6f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_local[contact] >= world_contact_count[world]
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    impulse = reaction[contact]
    if not _is_zero_vec3(impulse):
        if first >= 0:
            wrench = wp.transpose(contact_jacobian_first[contact]) @ impulse
            if apply_inverse_weight:
                wrench = inverse_weight[first] @ wrench
            _atomic_add_twist(twist_delta, first, wrench)
        if second >= 0:
            wrench = wp.transpose(contact_jacobian_second[contact]) @ impulse
            if apply_inverse_weight:
                wrench = inverse_weight[second] @ wrench
            _atomic_add_twist(twist_delta, second, wrench)


@wp.kernel
def _warmstart_limits_jacobi(
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    apply_inverse_weight: wp.bool,
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    limit = wp.tid()
    world = limit_world[limit]
    if (
        limit_local[limit] >= world_limit_count[world]
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = limit_body_first[limit]
    second = limit_body_second[limit]
    impulse = reaction[limit]
    if first >= 0:
        wrench = limit_jacobian_first[limit] * impulse
        if apply_inverse_weight:
            wrench = inverse_weight[first] @ wrench
        _atomic_add_twist(twist_delta, first, wrench)
    if second >= 0:
        wrench = limit_jacobian_second[limit] * impulse
        if apply_inverse_weight:
            wrench = inverse_weight[second] @ wrench
        _atomic_add_twist(twist_delta, second, wrench)


@wp.kernel
def _warmstart_frictions_jacobi(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    apply_inverse_weight: wp.bool,
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    friction = wp.tid()
    world = friction_world[friction]
    if (
        friction_local[friction] >= world_friction_count[world]
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = friction_body_first[friction]
    second = friction_body_second[friction]
    impulse = reaction[friction]
    if first >= 0:
        wrench = friction_jacobian_first[friction] * impulse
        if apply_inverse_weight:
            wrench = inverse_weight[first] @ wrench
        _atomic_add_twist(twist_delta, first, wrench)
    if second >= 0:
        wrench = friction_jacobian_second[friction] * impulse
        if apply_inverse_weight:
            wrench = inverse_weight[second] @ wrench
        _atomic_add_twist(twist_delta, second, wrench)


@cache
def _make_project_contacts_kernel(colored: bool):
    """Specialize local-frame Coulomb projection by indexing."""
    project_contact = _make_project_rigid_contact(colored)

    @wp.kernel(module="unique", enable_backward=False)
    def project_contacts_kernel(
        launch_dim: wp.int32,
        target_color: wp.int32,
        index: _ProjectionIndexData,
        contact_data: _RigidContactProjectionData,
        state: _RigidProjectionState,
    ):
        lane = wp.tid()
        begin = lane
        end = lane + 1
        stride = 1
        if wp.static(colored):
            begin = index.color_offsets[target_color] + lane
            end = index.color_offsets[target_color] + index.color_counts[target_color]
            stride = launch_dim
        for ordered in range(begin, end, stride):
            contact = ordered
            if wp.static(colored):
                contact = index.order[ordered]
            world = index.constraint_world[contact]
            if wp.static(not colored):
                if index.constraint_local[contact] >= index.world_constraint_count[world]:
                    continue
            if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
                continue
            project_contact(contact, world, target_color, contact_data, contact_data.delassus, state)

    return project_contacts_kernel


@cache
def _make_project_scalar_kernel(colored: bool, limit: bool):
    """Specialize scalar unilateral projection by indexing and projection law."""
    project_scalar = _make_project_rigid_scalar(colored, limit)

    @wp.kernel(module="unique", enable_backward=False)
    def project_scalar_kernel(
        launch_dim: wp.int32,
        target_color: wp.int32,
        index: _ProjectionIndexData,
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        bias: wp.array[wp.float32],
        impulse_bound: wp.array[wp.float32],
        delassus: wp.array[wp.float32],
        reaction: wp.array[wp.float32],
        state: _RigidProjectionState,
    ):
        lane = wp.tid()
        begin = lane
        end = lane + 1
        stride = 1
        if wp.static(colored):
            begin = index.color_offsets[target_color] + lane
            end = index.color_offsets[target_color] + index.color_counts[target_color]
            stride = launch_dim

        for ordered in range(begin, end, stride):
            constraint = ordered
            if wp.static(colored):
                constraint = index.order[ordered]
            world = index.constraint_world[constraint]
            if wp.static(not colored):
                if index.constraint_local[constraint] >= index.world_constraint_count[world]:
                    continue
            if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
                continue
            project_scalar(
                constraint,
                world,
                target_color,
                body_first,
                body_second,
                jacobian_first,
                jacobian_second,
                bias,
                impulse_bound,
                delassus,
                reaction,
                state,
            )

    return project_scalar_kernel


@wp.kernel
def _initialize_projection_residuals(
    world_contact_residual_max: wp.array[wp.float32],
    world_limit_residual_max: wp.array[wp.float32],
    world_friction_residual_max: wp.array[wp.float32],
):
    world = wp.tid()
    world_contact_residual_max[world] = 0.0
    world_limit_residual_max[world] = 0.0
    world_friction_residual_max[world] = 0.0


@wp.kernel
def _compute_friction_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    friction_velocity: wp.array[wp.float32],
    friction_residual: wp.array[wp.float32],
    world_friction_residual_max: wp.array[wp.float32],
):
    friction = wp.tid()
    world = friction_world[friction]
    if (
        friction_local[friction] >= world_friction_count[world]
        or not world_mask[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    first = friction_body_first[friction]
    second = friction_body_second[friction]
    value = wp.float32(0.0)
    if first >= 0:
        value += wp.dot(friction_jacobian_first[friction], projected_twist[first])
    if second >= 0:
        value += wp.dot(friction_jacobian_second[friction], projected_twist[second])
    friction_velocity[friction] = value
    delassus = friction_delassus[friction]
    scale = wp.sqrt(delassus)
    scaled_reaction = scale * friction_reaction[friction]
    scaled_velocity = value / scale
    scaled_bound = scale * friction_impulse_bound[friction]
    residual = wp.abs(scaled_reaction - wp.clamp(scaled_reaction - scaled_velocity, -scaled_bound, scaled_bound))
    friction_residual[friction] = residual
    wp.atomic_max(world_friction_residual_max, world, residual)


@wp.kernel
def _compute_contact_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_delassus: wp.array[wp.mat33f],
    projected_twist: wp.array[vec6f],
    contact_velocity: wp.array[wp.vec3f],
    contact_residual: wp.array[wp.float32],
    world_contact_residual_max: wp.array[wp.float32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_local[contact] >= world_contact_count[world]
        or not world_mask[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        contact_velocity[contact] = wp.vec3f(0.0)
        contact_residual[contact] = 0.0
        return
    contact_velocity_final = wp.vec3f(0.0)
    if first >= 0:
        contact_velocity_final += contact_jacobian_first[contact] @ projected_twist[first]
    if second >= 0:
        contact_velocity_final += contact_jacobian_second[contact] @ projected_twist[second]
    contact_velocity[contact] = contact_velocity_final
    contact_residual_vector = compute_contact_scaled_alart_curnier_residual(
        contact_delassus[contact],
        contact_reaction[contact],
        contact_velocity_final + contact_bias[contact],
        contact_friction[contact],
    )
    residual_max = wp.max(wp.abs(contact_residual_vector))
    contact_residual[contact] = residual_max
    wp.atomic_max(world_contact_residual_max, world, residual_max)


@wp.kernel
def _compute_limit_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    limit_velocity: wp.array[wp.float32],
    limit_residual: wp.array[wp.float32],
    world_limit_residual_max: wp.array[wp.float32],
):
    limit = wp.tid()
    world = limit_world[limit]
    if (
        limit_local[limit] >= world_limit_count[world]
        or not world_mask[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    first = limit_body_first[limit]
    second = limit_body_second[limit]
    if first < 0 and second < 0:
        limit_velocity[limit] = 0.0
        limit_residual[limit] = 0.0
        return
    limit_velocity_final = wp.float32(0.0)
    if first >= 0:
        limit_velocity_final += wp.dot(limit_jacobian_first[limit], projected_twist[first])
    if second >= 0:
        limit_velocity_final += wp.dot(limit_jacobian_second[limit], projected_twist[second])
    limit_velocity[limit] = limit_velocity_final
    scale = wp.sqrt(limit_delassus[limit])
    scaled_reaction = scale * limit_reaction[limit]
    scaled_velocity = (limit_velocity_final + limit_bias[limit]) / scale
    limit_residual_value = wp.abs(scaled_reaction - wp.max(scaled_reaction - scaled_velocity, 0.0))
    limit_residual[limit] = limit_residual_value
    wp.atomic_max(world_limit_residual_max, world, limit_residual_value)


def compute_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_delassus: wp.array[wp.mat33f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    friction_velocity: wp.array[wp.float32],
    contact_velocity: wp.array[wp.vec3f],
    limit_velocity: wp.array[wp.float32],
    contact_residual: wp.array[wp.float32],
    limit_residual: wp.array[wp.float32],
    friction_residual: wp.array[wp.float32],
    world_contact_residual_max: wp.array[wp.float32],
    world_limit_residual_max: wp.array[wp.float32],
    world_friction_residual_max: wp.array[wp.float32],
) -> None:
    """Recompute final unilateral velocities and per-world natural-map residual maxima."""
    world_count = world_mask.shape[0]
    if (
        projection_status.shape[0] != world_count
        or world_friction_count.shape[0] != world_count
        or world_contact_count.shape[0] != world_count
        or world_limit_count.shape[0] != world_count
        or world_contact_residual_max.shape[0] != world_count
        or world_limit_residual_max.shape[0] != world_count
        or world_friction_residual_max.shape[0] != world_count
    ):
        raise ValueError("Projection diagnostic world arrays must have identical lengths.")
    if (
        friction_world.shape[0] != friction_local.shape[0]
        or contact_world.shape[0] != contact_local.shape[0]
        or limit_world.shape[0] != limit_local.shape[0]
    ):
        raise ValueError("Projection diagnostic world and local arrays must have identical lengths.")
    wp.launch(
        _initialize_projection_residuals,
        dim=world_count,
        inputs=[],
        outputs=[
            world_contact_residual_max,
            world_limit_residual_max,
            world_friction_residual_max,
        ],
        device=projected_twist.device,
    )
    if friction_world.shape[0] > 0:
        wp.launch(
            _compute_friction_projection_residuals,
            dim=friction_world.shape[0],
            inputs=[
                world_mask,
                projection_status,
                friction_world,
                friction_local,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                friction_impulse_bound,
                friction_reaction,
                friction_delassus,
                projected_twist,
            ],
            outputs=[friction_velocity, friction_residual, world_friction_residual_max],
            device=projected_twist.device,
        )
    if contact_world.shape[0] > 0:
        wp.launch(
            _compute_contact_projection_residuals,
            dim=contact_world.shape[0],
            inputs=[
                world_mask,
                projection_status,
                contact_world,
                contact_local,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                contact_reaction,
                contact_delassus,
                projected_twist,
            ],
            outputs=[contact_velocity, contact_residual, world_contact_residual_max],
            device=projected_twist.device,
        )
    if limit_world.shape[0] > 0:
        wp.launch(
            _compute_limit_projection_residuals,
            dim=limit_world.shape[0],
            inputs=[
                world_mask,
                projection_status,
                limit_world,
                limit_local,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                limit_bias,
                limit_reaction,
                limit_delassus,
                projected_twist,
            ],
            outputs=[limit_velocity, limit_residual, world_limit_residual_max],
            device=projected_twist.device,
        )


def prepare_physical_projection_data(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    friction_delassus: wp.array[wp.float32],
    contact_physical_delassus: wp.array[wp.mat33f],
    contact_prepared_delassus: wp.array[wp.mat33f],
    limit_delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
) -> None:
    """Cache physical local Delassus blocks once for fixed body-space data."""
    world_count = world_status.shape[0]
    if (
        world_friction_count.shape[0] != world_count
        or world_contact_count.shape[0] != world_count
        or world_limit_count.shape[0] != world_count
    ):
        raise ValueError("Friction, contact, limit, and status world arrays must have identical lengths.")
    world_status.fill_(PROJECTION_STATUS_VALID)
    for world, local, count, first, second, jacobian_first, jacobian_second, output in (
        (
            friction_world,
            friction_local,
            world_friction_count,
            friction_body_first,
            friction_body_second,
            friction_jacobian_first,
            friction_jacobian_second,
            friction_delassus,
        ),
        (
            limit_world,
            limit_local,
            world_limit_count,
            limit_body_first,
            limit_body_second,
            limit_jacobian_first,
            limit_jacobian_second,
            limit_delassus,
        ),
    ):
        if world.shape[0] > 0:
            wp.launch(
                _prepare_scalar_projection_data,
                dim=world.shape[0],
                inputs=[world, local, count, first, second, jacobian_first, jacobian_second, inverse_weight],
                outputs=[output, world_status],
                device=inverse_weight.device,
            )
    if contact_world.shape[0] > 0:
        wp.launch(
            _prepare_contact_physical_projection_data,
            dim=contact_world.shape[0],
            inputs=[
                contact_world,
                contact_local,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                inverse_weight,
            ],
            outputs=[contact_physical_delassus, contact_prepared_delassus, world_status],
            device=inverse_weight.device,
        )


def prepare_jacobi_projection_data(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    friction_delassus: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    limit_delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
) -> None:
    """Prepare mass-split body-space blocks for true Jacobi projection."""
    world_count = world_status.shape[0]
    if (
        world_friction_count.shape[0] != world_count
        or world_contact_count.shape[0] != world_count
        or world_limit_count.shape[0] != world_count
    ):
        raise ValueError("Friction, contact, limit, and status world arrays must have identical lengths.")
    if body_constraint_count.shape[0] != static_body_constraint_count.shape[0]:
        raise ValueError("Body incidence arrays must have identical lengths.")
    world_status.fill_(PROJECTION_STATUS_VALID)
    if friction_world.shape[0] > 0:
        wp.launch(
            _prepare_frictions_jacobi,
            dim=friction_world.shape[0],
            inputs=[
                friction_world,
                friction_local,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                body_constraint_count,
                static_body_constraint_count,
                inverse_weight,
            ],
            outputs=[friction_delassus, world_status],
            device=inverse_weight.device,
        )
    if contact_world.shape[0] > 0:
        wp.launch(
            _prepare_contacts_jacobi,
            dim=contact_world.shape[0],
            inputs=[
                contact_world,
                contact_local,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                body_constraint_count,
                static_body_constraint_count,
                inverse_weight,
            ],
            outputs=[contact_delassus, world_status],
            device=inverse_weight.device,
        )
    if limit_world.shape[0] > 0:
        wp.launch(
            _prepare_limits_jacobi,
            dim=limit_world.shape[0],
            inputs=[
                limit_world,
                limit_local,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                body_constraint_count,
                static_body_constraint_count,
                inverse_weight,
            ],
            outputs=[limit_delassus, world_status],
            device=inverse_weight.device,
        )


@wp.func
def _check_projected_twist(
    body: wp.int32,
    world: wp.int32,
    projected_twist: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    if not wp.isfinite(projected_twist[body]):
        wp.atomic_min(world_status, world, PROJECTION_STATUS_INVALID)


@wp.kernel
def _validate_projected_twists(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projected_twist: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    body = wp.tid()
    world = body_world[body]
    if world_active[world]:
        _check_projected_twist(body, world, projected_twist, world_status)
