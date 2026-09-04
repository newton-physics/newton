# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-batched Gauss--Seidel projection with final Jacobi smoothing."""

from __future__ import annotations

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .jacobi import project_constraints_jacobi
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    _atomic_add_twist,
    _can_fuse_rigid_projection_by_world,
    _check_projected_twist,
    _initialize_jacobi_projection_status,
    _make_colored_projection_index,
    _make_project_contacts_kernel,
    _make_project_scalar_kernel,
    _make_rigid_projection_state,
    _project_rigid_contact_colored,
    _project_rigid_contact_jacobi,
    _project_rigid_friction_colored,
    _project_rigid_friction_jacobi,
    _project_rigid_limit_colored,
    _project_rigid_limit_jacobi,
    _RigidContactProjectionData,
    _RigidProjectionState,
    _sync_threads,
    _warmstart_contacts_jacobi,
    _warmstart_frictions_jacobi,
    _warmstart_limits_jacobi,
    compute_limit_delassus,
    prepare_contact_coulomb,
    prepare_jacobi_projection_data,
)

wp.set_module_options({"enable_backward": False})

_COLOR_REPAIR_PASSES = 3
_COLOR_BLOCK_DIM = 128
_WORLD_COLOR_BLOCK_DIM = 64
_COLOR_BLOCKS_PER_SM = 2
# Avoid parallel-scan setup for the small color counts used by normal workloads.
_SERIAL_COLOR_PREFIX_LIMIT = 64
_NO_PROPOSAL = -1
_LOCK_FREE = 0x7FFFFFFF


@wp.struct
class _RigidColoredWorldData:
    world_color_count: wp.array2d[wp.int32]
    world_friction_offset: wp.array[wp.int32]
    world_friction_count: wp.array[wp.int32]
    friction_colors: wp.array[wp.int32]
    friction_body_first: wp.array[wp.int32]
    friction_body_second: wp.array[wp.int32]
    friction_jacobian_first: wp.array[vec6f]
    friction_jacobian_second: wp.array[vec6f]
    friction_bound: wp.array[wp.float32]
    friction_colored_delassus: wp.array[wp.float32]
    friction_jacobi_delassus: wp.array[wp.float32]
    friction_reaction: wp.array[wp.float32]
    world_contact_offset: wp.array[wp.int32]
    world_contact_count: wp.array[wp.int32]
    contact_colors: wp.array[wp.int32]
    contact_jacobi_delassus: wp.array[wp.mat33f]
    world_limit_offset: wp.array[wp.int32]
    world_limit_count: wp.array[wp.int32]
    limit_colors: wp.array[wp.int32]
    limit_body_first: wp.array[wp.int32]
    limit_body_second: wp.array[wp.int32]
    limit_jacobian_first: wp.array[vec6f]
    limit_jacobian_second: wp.array[vec6f]
    limit_bias: wp.array[wp.float32]
    limit_colored_delassus: wp.array[wp.float32]
    limit_jacobi_delassus: wp.array[wp.float32]
    limit_reaction: wp.array[wp.float32]


def _bounded_worker_count(capacity: int, device) -> int:
    if capacity <= 0:
        return 0
    if device.is_cuda:
        return min(capacity, max(_COLOR_BLOCK_DIM, device.sm_count * _COLOR_BLOCKS_PER_SM * _COLOR_BLOCK_DIM))
    return capacity


@wp.func
def _mix_color_key(value: wp.uint32) -> wp.uint32:
    value = (value ^ (value >> wp.uint32(16))) * wp.static(wp.uint32(0x7FEB352D))
    value = (value ^ (value >> wp.uint32(15))) * wp.static(wp.uint32(0x846CA68B))
    return value ^ (value >> wp.uint32(16))


@wp.func
def _initial_color(world: int, local: int, first: int, second: int, family: int, color_count: int) -> int:
    key = wp.uint32(world + 1) * wp.static(wp.uint32(0x9E3779B9))
    key = key ^ (wp.uint32(local + 1) * wp.static(wp.uint32(0x85EBCA6B)))
    key = key ^ (wp.uint32(first + 2) * wp.static(wp.uint32(0xC2B2AE35)))
    key = key ^ (wp.uint32(second + 2) * wp.static(wp.uint32(0x27D4EB2F)))
    key = key ^ (wp.uint32(family) * wp.static(wp.uint32(0x165667B1)))
    return int(_mix_color_key(key) % wp.uint32(color_count))


@wp.func
def _occupancy(occupancy: wp.array2d[wp.int32], endpoint: int, color: int) -> int:
    value = int(0)
    if endpoint >= 0:
        value = occupancy[endpoint, color]
    return value


@wp.func
def _choose_two_endpoint_color(
    first: int,
    second: int,
    current: int,
    color_count: int,
    occupancy: wp.array2d[wp.int32],
) -> int:
    current_sum = _occupancy(occupancy, first, current)
    endpoint_count = int(0)
    if first >= 0:
        endpoint_count += 1
    if second >= 0 and second != first:
        current_sum += _occupancy(occupancy, second, current)
        endpoint_count += 1
    best = current
    best_sum = current_sum
    best_max = wp.max(_occupancy(occupancy, first, current), _occupancy(occupancy, second, current))
    for color in range(color_count):
        candidate_sum = _occupancy(occupancy, first, color)
        if second >= 0 and second != first:
            candidate_sum += _occupancy(occupancy, second, color)
        candidate_max = wp.max(_occupancy(occupancy, first, color), _occupancy(occupancy, second, color))
        if candidate_sum < best_sum or (candidate_sum == best_sum and candidate_max < best_max):
            best = color
            best_sum = candidate_sum
            best_max = candidate_max
    # Moving one incidence adds one to the destination and removes one from
    # the source. This is exactly the strict-improvement condition for sum(m^2).
    if best != current and best_sum + endpoint_count < current_sum:
        return best
    return _NO_PROPOSAL


@wp.kernel
def _assign_two_endpoint_colors(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    family: int,
    color_count: int,
    colors: wp.array[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        colors[constraint] = _NO_PROPOSAL
        return
    colors[constraint] = _initial_color(
        world,
        constraint_local[constraint],
        endpoint_first[constraint],
        endpoint_second[constraint],
        family,
        color_count,
    )


@wp.kernel
def _count_two_endpoint_occupancy(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    colors: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    world_color_count: wp.array2d[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        return
    color = colors[constraint]
    wp.atomic_add(world_color_count, world, color, 1)
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if first >= 0:
        wp.atomic_add(occupancy, first, color, 1)
    if second >= 0 and second != first:
        wp.atomic_add(occupancy, second, color, 1)


@wp.kernel
def _propose_and_claim_two_endpoint_repairs(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    color_count: int,
    key_offset: int,
    colors: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    proposals: wp.array[wp.int32],
    locks: wp.array[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        proposals[constraint] = _NO_PROPOSAL
        return
    proposal = _choose_two_endpoint_color(
        endpoint_first[constraint],
        endpoint_second[constraint],
        colors[constraint],
        color_count,
        occupancy,
    )
    proposals[constraint] = proposal
    if proposal < 0:
        return
    key = key_offset + constraint
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if first >= 0:
        wp.atomic_min(locks, first, key)
    if second >= 0 and second != first:
        wp.atomic_min(locks, second, key)


@wp.kernel
def _commit_two_endpoint_repairs(
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    key_offset: int,
    colors: wp.array[wp.int32],
    proposals: wp.array[wp.int32],
    locks: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    world_color_count: wp.array2d[wp.int32],
):
    constraint = wp.tid()
    candidate = proposals[constraint]
    if candidate < 0:
        return
    key = key_offset + constraint
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if (first >= 0 and locks[first] != key) or (second >= 0 and second != first and locks[second] != key):
        return
    current = colors[constraint]
    world = constraint_world[constraint]
    wp.atomic_add(world_color_count, world, current, -1)
    wp.atomic_add(world_color_count, world, candidate, 1)
    if first >= 0:
        wp.atomic_add(occupancy, first, current, -1)
        wp.atomic_add(occupancy, first, candidate, 1)
    if second >= 0 and second != first:
        wp.atomic_add(occupancy, second, current, -1)
        wp.atomic_add(occupancy, second, candidate, 1)
    colors[constraint] = candidate


@wp.kernel
def _count_colors(colors: wp.array[wp.int32], color_count: int, counts: wp.array[wp.int32]):
    item = wp.tid()
    color = colors[item]
    if color >= 0 and color < color_count:
        wp.atomic_add(counts, color, 1)


@wp.kernel
def _prefix_color_counts(
    color_count: int, counts: wp.array[wp.int32], offsets: wp.array[wp.int32], cursors: wp.array[wp.int32]
):
    offset = int(0)
    for color in range(color_count):
        offsets[color] = offset
        cursors[color] = offset
        offset += counts[color]


@wp.kernel
def _scatter_color_order(
    colors: wp.array[wp.int32],
    color_count: int,
    cursors: wp.array[wp.int32],
    order: wp.array[wp.int32],
):
    item = wp.tid()
    color = colors[item]
    if color >= 0 and color < color_count:
        ordered = wp.atomic_add(cursors, color, 1)
        order[ordered] = item


@wp.kernel
def _prepare_frictions_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        constraint = order[ordered]
        first = endpoint_first[constraint]
        second = endpoint_second[constraint]
        if first < 0 and second < 0:
            delassus[constraint] = 0.0
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        value = compute_limit_delassus(
            jacobian_first[constraint], inverse_first, jacobian_second[constraint], inverse_second
        )
        delassus[constraint] = value
        if not wp.isfinite(value) or value <= 0.0:
            world_status[constraint_world[constraint]] = PROJECTION_STATUS_INVALID


@wp.kernel
def _prepare_contacts_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    jacobian_first: wp.array[mat36f],
    jacobian_second: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        constraint = order[ordered]
        first = endpoint_first[constraint]
        second = endpoint_second[constraint]
        if first < 0 and second < 0:
            delassus[constraint] = wp.mat33f(0.0)
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        data = prepare_contact_coulomb(
            jacobian_first[constraint],
            inverse_first,
            jacobian_second[constraint],
            inverse_second,
            bias[constraint],
            friction[constraint],
        )
        delassus[constraint] = data.delassus
        if data.status == PROJECTION_STATUS_INVALID:
            world_status[constraint_world[constraint]] = data.status


@wp.kernel
def _prepare_rigid_colored(
    launch_dim: int,
    target_color: int,
    friction_counts: wp.array[wp.int32],
    friction_offsets: wp.array[wp.int32],
    friction_order: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_first: wp.array[wp.int32],
    friction_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_counts: wp.array[wp.int32],
    contact_offsets: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_first: wp.array[wp.int32],
    contact_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    limit_counts: wp.array[wp.int32],
    limit_offsets: wp.array[wp.int32],
    limit_order: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_first: wp.array[wp.int32],
    limit_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    friction_delassus: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    limit_delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    friction_begin = friction_offsets[target_color] + lane
    friction_end = friction_offsets[target_color] + friction_counts[target_color]
    for ordered in range(friction_begin, friction_end, launch_dim):
        constraint = friction_order[ordered]
        first = friction_first[constraint]
        second = friction_second[constraint]
        if first < 0 and second < 0:
            friction_delassus[constraint] = 0.0
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        value = compute_limit_delassus(
            friction_jacobian_first[constraint],
            inverse_first,
            friction_jacobian_second[constraint],
            inverse_second,
        )
        friction_delassus[constraint] = value
        if not wp.isfinite(value) or value <= 0.0:
            world_status[friction_world[constraint]] = PROJECTION_STATUS_INVALID

    contact_begin = contact_offsets[target_color] + lane
    contact_end = contact_offsets[target_color] + contact_counts[target_color]
    for ordered in range(contact_begin, contact_end, launch_dim):
        constraint = contact_order[ordered]
        first = contact_first[constraint]
        second = contact_second[constraint]
        if first < 0 and second < 0:
            contact_delassus[constraint] = wp.mat33f(0.0)
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        data = prepare_contact_coulomb(
            contact_jacobian_first[constraint],
            inverse_first,
            contact_jacobian_second[constraint],
            inverse_second,
            contact_bias[constraint],
            contact_friction[constraint],
        )
        contact_delassus[constraint] = data.delassus
        if data.status == PROJECTION_STATUS_INVALID:
            world_status[contact_world[constraint]] = data.status

    limit_begin = limit_offsets[target_color] + lane
    limit_end = limit_offsets[target_color] + limit_counts[target_color]
    for ordered in range(limit_begin, limit_end, launch_dim):
        constraint = limit_order[ordered]
        first = limit_first[constraint]
        second = limit_second[constraint]
        if first < 0 and second < 0:
            limit_delassus[constraint] = 0.0
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        value = compute_limit_delassus(
            limit_jacobian_first[constraint], inverse_first, limit_jacobian_second[constraint], inverse_second
        )
        limit_delassus[constraint] = value
        if not wp.isfinite(value) or value <= 0.0:
            world_status[limit_world[constraint]] = PROJECTION_STATUS_INVALID


@wp.kernel
def _project_rigid_colored(
    launch_dim: int,
    target_color: int,
    friction_counts: wp.array[wp.int32],
    friction_offsets: wp.array[wp.int32],
    friction_order: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_first: wp.array[wp.int32],
    friction_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_bound: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    contact_counts: wp.array[wp.int32],
    contact_offsets: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_data: _RigidContactProjectionData,
    limit_counts: wp.array[wp.int32],
    limit_offsets: wp.array[wp.int32],
    limit_order: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_first: wp.array[wp.int32],
    limit_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    state: _RigidProjectionState,
    friction_reaction: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
):
    lane = wp.tid()
    friction_begin = friction_offsets[target_color] + lane
    friction_end = friction_offsets[target_color] + friction_counts[target_color]
    for ordered in range(friction_begin, friction_end, launch_dim):
        constraint = friction_order[ordered]
        world = friction_world[constraint]
        if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
            continue
        _project_rigid_friction_colored(
            constraint,
            world,
            target_color,
            friction_first,
            friction_second,
            friction_jacobian_first,
            friction_jacobian_second,
            friction_bound,
            friction_bound,
            friction_delassus,
            friction_reaction,
            state,
        )

    contact_begin = contact_offsets[target_color] + lane
    contact_end = contact_offsets[target_color] + contact_counts[target_color]
    for ordered in range(contact_begin, contact_end, launch_dim):
        constraint = contact_order[ordered]
        world = contact_world[constraint]
        if state.world_active[world] and state.world_status[world] == PROJECTION_STATUS_VALID:
            _project_rigid_contact_colored(constraint, world, target_color, contact_data, contact_data.delassus, state)

    limit_begin = limit_offsets[target_color] + lane
    limit_end = limit_offsets[target_color] + limit_counts[target_color]
    for ordered in range(limit_begin, limit_end, launch_dim):
        constraint = limit_order[ordered]
        world = limit_world[constraint]
        if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
            continue
        _project_rigid_limit_colored(
            constraint,
            world,
            target_color,
            limit_first,
            limit_second,
            limit_jacobian_first,
            limit_jacobian_second,
            limit_bias,
            limit_bias,
            limit_delassus,
            limit_reaction,
            state,
        )


@wp.kernel
def _project_rigid_colored_by_world(
    iterations: wp.int32,
    color_count: wp.int32,
    world_body_offset: wp.array[wp.int32],
    world_body_count: wp.array[wp.int32],
    data: _RigidColoredWorldData,
    contact_data: _RigidContactProjectionData,
    state: _RigidProjectionState,
    prepared_status: wp.array[wp.int32],
):
    """Project one rigid world per block, including warm start and smoothing."""
    world, lane = wp.tid()
    thread_count = wp.block_dim()

    local = lane
    while local < world_body_count[world]:
        state.twist_delta[world_body_offset[world] + local] = vec6f(0.0)
        local += thread_count
    if state.world_active[world]:
        state.world_status[world] = prepared_status[world]
    _sync_threads()
    if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
        return

    # Apply the existing reactions before the first colored sweep.
    local = lane
    while local < data.world_friction_count[world]:
        constraint = data.world_friction_offset[world] + local
        first = data.friction_body_first[constraint]
        second = data.friction_body_second[constraint]
        friction_reaction = data.friction_reaction[constraint]
        correction_first = vec6f(0.0)
        correction_second = vec6f(0.0)
        if first >= 0:
            correction_first = data.friction_jacobian_first[constraint] * friction_reaction
        if second >= 0:
            correction_second = data.friction_jacobian_second[constraint] * friction_reaction
        _atomic_add_twist(state.twist_delta, first, correction_first)
        _atomic_add_twist(state.twist_delta, second, correction_second)
        local += thread_count

    local = lane
    while local < data.world_contact_count[world]:
        constraint = data.world_contact_offset[world] + local
        first = contact_data.body_first[constraint]
        second = contact_data.body_second[constraint]
        contact_reaction = contact_data.reaction[constraint]
        correction_first = vec6f(0.0)
        correction_second = vec6f(0.0)
        if first >= 0:
            correction_first = wp.transpose(contact_data.jacobian_first[constraint]) @ contact_reaction
        if second >= 0:
            correction_second = wp.transpose(contact_data.jacobian_second[constraint]) @ contact_reaction
        _atomic_add_twist(state.twist_delta, first, correction_first)
        _atomic_add_twist(state.twist_delta, second, correction_second)
        local += thread_count

    local = lane
    while local < data.world_limit_count[world]:
        constraint = data.world_limit_offset[world] + local
        first = data.limit_body_first[constraint]
        second = data.limit_body_second[constraint]
        limit_reaction = data.limit_reaction[constraint]
        correction_first = vec6f(0.0)
        correction_second = vec6f(0.0)
        if first >= 0:
            correction_first = data.limit_jacobian_first[constraint] * limit_reaction
        if second >= 0:
            correction_second = data.limit_jacobian_second[constraint] * limit_reaction
        _atomic_add_twist(state.twist_delta, first, correction_first)
        _atomic_add_twist(state.twist_delta, second, correction_second)
        local += thread_count

    _sync_threads()
    local = lane
    while local < world_body_count[world]:
        body = world_body_offset[world] + local
        if state.world_status[world] == PROJECTION_STATUS_VALID:
            correction = state.inverse_weight[body] @ state.twist_delta[body]
            state.projected_twist[body] += correction
        state.twist_delta[body] = vec6f(0.0)
        local += thread_count
    _sync_threads()

    # Colors and iterations are block-local for independent rigid worlds.
    for _iteration in range(iterations):
        for color in range(color_count):
            if data.world_color_count[world, color] != 0:
                if state.world_status[world] == PROJECTION_STATUS_VALID:
                    local = lane
                    while local < data.world_friction_count[world]:
                        constraint = data.world_friction_offset[world] + local
                        if data.friction_colors[constraint] == color:
                            _project_rigid_friction_colored(
                                constraint,
                                world,
                                color,
                                data.friction_body_first,
                                data.friction_body_second,
                                data.friction_jacobian_first,
                                data.friction_jacobian_second,
                                data.friction_bound,
                                data.friction_bound,
                                data.friction_colored_delassus,
                                data.friction_reaction,
                                state,
                            )
                        local += thread_count

                    local = lane
                    while local < data.world_contact_count[world]:
                        constraint = data.world_contact_offset[world] + local
                        if data.contact_colors[constraint] == color:
                            _project_rigid_contact_colored(
                                constraint, world, color, contact_data, contact_data.delassus, state
                            )
                        local += thread_count

                    local = lane
                    while local < data.world_limit_count[world]:
                        constraint = data.world_limit_offset[world] + local
                        if data.limit_colors[constraint] == color:
                            _project_rigid_limit_colored(
                                constraint,
                                world,
                                color,
                                data.limit_body_first,
                                data.limit_body_second,
                                data.limit_jacobian_first,
                                data.limit_jacobian_second,
                                data.limit_bias,
                                data.limit_bias,
                                data.limit_colored_delassus,
                                data.limit_reaction,
                                state,
                            )
                        local += thread_count

                _sync_threads()
                local = lane
                while local < world_body_count[world]:
                    body = world_body_offset[world] + local
                    if state.world_status[world] == PROJECTION_STATUS_VALID:
                        state.projected_twist[body] += state.twist_delta[body]
                    state.twist_delta[body] = vec6f(0.0)
                    local += thread_count
                _sync_threads()

    # Finish with the same mass-split Jacobi smoothing as the global path.
    if state.world_status[world] == PROJECTION_STATUS_VALID:
        local = lane
        while local < data.world_friction_count[world]:
            constraint = data.world_friction_offset[world] + local
            _project_rigid_friction_jacobi(
                constraint,
                world,
                0,
                data.friction_body_first,
                data.friction_body_second,
                data.friction_jacobian_first,
                data.friction_jacobian_second,
                data.friction_bound,
                data.friction_bound,
                data.friction_jacobi_delassus,
                data.friction_reaction,
                state,
            )
            local += thread_count

        local = lane
        while local < data.world_contact_count[world]:
            constraint = data.world_contact_offset[world] + local
            _project_rigid_contact_jacobi(
                constraint,
                world,
                0,
                contact_data,
                data.contact_jacobi_delassus,
                state,
            )
            local += thread_count

        local = lane
        while local < data.world_limit_count[world]:
            constraint = data.world_limit_offset[world] + local
            _project_rigid_limit_jacobi(
                constraint,
                world,
                0,
                data.limit_body_first,
                data.limit_body_second,
                data.limit_jacobian_first,
                data.limit_jacobian_second,
                data.limit_bias,
                data.limit_bias,
                data.limit_jacobi_delassus,
                data.limit_reaction,
                state,
            )
            local += thread_count

    _sync_threads()
    local = lane
    while local < world_body_count[world]:
        body = world_body_offset[world] + local
        if state.world_status[world] == PROJECTION_STATUS_VALID:
            correction = state.inverse_weight[body] @ state.twist_delta[body]
            state.projected_twist[body] += correction
        _check_projected_twist(body, world, state.projected_twist, state.world_status)
        state.twist_delta[body] = vec6f(0.0)
        local += thread_count


@wp.kernel
def _apply_body_delta(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
        projected_twist[body] += delta[body]
    delta[body] = vec6f(0.0)


class _ColorFamily:
    def __init__(self, capacity: int, color_count: int, device):
        self.capacity = capacity
        self.worker_count = _bounded_worker_count(capacity, device)
        self.colors = wp.full(capacity, _NO_PROPOSAL, dtype=wp.int32, device=device)
        self.proposals = wp.full(capacity, _NO_PROPOSAL, dtype=wp.int32, device=device)
        self.counts = wp.zeros(color_count, dtype=wp.int32, device=device)
        self.offsets = wp.zeros(color_count, dtype=wp.int32, device=device)
        self.cursors = wp.zeros(color_count, dtype=wp.int32, device=device)
        self.order = wp.full(capacity, -1, dtype=wp.int32, device=device)

    def _prefix_counts(self, color_count: int, device) -> None:
        if color_count <= _SERIAL_COLOR_PREFIX_LIMIT:
            wp.launch(
                _prefix_color_counts,
                dim=1,
                inputs=[color_count, self.counts],
                outputs=[self.offsets, self.cursors],
                device=device,
            )
        else:
            wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
            wp.copy(self.cursors, self.offsets)

    def compact(self, color_count: int, device) -> None:
        self.counts.zero_()
        if self.capacity == 0:
            return
        wp.launch(
            _count_colors, dim=self.capacity, inputs=[self.colors, color_count], outputs=[self.counts], device=device
        )
        self._prefix_counts(color_count, device)
        wp.launch(
            _scatter_color_order,
            dim=self.capacity,
            inputs=[self.colors, color_count],
            outputs=[self.cursors, self.order],
            device=device,
        )


class ColoredGaussSeidelProjection:
    """Own fixed-capacity coloring and projection scratch for one LOX solver.

    The effective color count is the requested maximum bounded by total
    allocated unilateral capacity, with one internal color retained for empty
    or single-constraint systems. The endpoint occupancy tables use int32
    atomics. Scratch is allocated only for requested counts of at least two;
    the one-color endpoint uses the existing Jacobi
    implementation directly.
    """

    def __init__(self, problem, color_count: int):
        if color_count < 2:
            raise ValueError("Colored Gauss-Seidel requires at least two colors.")
        self.problem = problem
        rigid_capacity = problem.friction_capacity + problem.contact_capacity + problem.limit_capacity
        self.color_count = max(1, min(color_count, rigid_capacity))
        self.device = problem.device
        body_count = problem.body_constraint_count.shape[0]
        self.body_occupancy = wp.zeros((body_count, self.color_count), dtype=wp.int32, device=self.device)
        self.rigid_world_color_count = wp.zeros(
            (problem.world_contact_count.shape[0], self.color_count),
            dtype=wp.int32,
            device=self.device,
        )
        self.body_locks = wp.full(body_count, _LOCK_FREE, dtype=wp.int32, device=self.device)
        self.friction = _ColorFamily(problem.friction_capacity, self.color_count, self.device)
        self.contact = _ColorFamily(problem.contact_capacity, self.color_count, self.device)
        self.limit = _ColorFamily(problem.limit_capacity, self.color_count, self.device)
        self.friction_delassus = wp.zeros(self.friction.capacity, dtype=wp.float32, device=self.device)
        self.contact_delassus = wp.zeros(self.contact.capacity, dtype=wp.mat33f, device=self.device)
        self.limit_delassus = wp.zeros(self.limit.capacity, dtype=wp.float32, device=self.device)
        self._families = (self.friction, self.contact, self.limit)
        self.rigid_worker_count = _bounded_worker_count(rigid_capacity, self.device)
        self._fuse_rigid_families = sum(family.capacity > 0 for family in self._families) > 1

    def _launch_rigid_families(self, kernel, extra_inputs: tuple = ()) -> None:
        problem = self.problem
        if problem is None:
            return
        entries = (
            (
                self.friction,
                problem.friction_world,
                problem.friction_local,
                problem.world_friction_count,
                problem.friction_body_first,
                problem.friction_body_second,
                0,
            ),
            (
                self.contact,
                problem.contact_world,
                problem.contact_local,
                problem.world_contact_count,
                problem.contact_body_first,
                problem.contact_body_second,
                1,
            ),
            (
                self.limit,
                problem.limit_world,
                problem.limit_local,
                problem.world_limit_count,
                problem.limit_body_first,
                problem.limit_body_second,
                2,
            ),
        )
        key_offset = 0
        for family, worlds, local, counts, first, second, salt in entries:
            if family.capacity > 0:
                if kernel is _assign_two_endpoint_colors:
                    inputs = [worlds, local, counts, first, second, salt, self.color_count]
                    outputs = [family.colors]
                elif kernel is _count_two_endpoint_occupancy:
                    inputs = [worlds, local, counts, first, second, family.colors]
                    outputs = [self.body_occupancy, self.rigid_world_color_count]
                elif kernel is _propose_and_claim_two_endpoint_repairs:
                    inputs = [
                        worlds,
                        local,
                        counts,
                        first,
                        second,
                        self.color_count,
                        key_offset,
                        family.colors,
                        self.body_occupancy,
                    ]
                    outputs = [family.proposals, self.body_locks]
                else:
                    inputs = [worlds, first, second, key_offset, family.colors, family.proposals, self.body_locks]
                    outputs = [self.body_occupancy, self.rigid_world_color_count]
                wp.launch(kernel, dim=family.capacity, inputs=inputs, outputs=outputs, device=self.device)
            key_offset += family.capacity

    def build_colors(self) -> None:
        self.body_occupancy.zero_()
        self.rigid_world_color_count.zero_()
        self._launch_rigid_families(_assign_two_endpoint_colors)
        self._launch_rigid_families(_count_two_endpoint_occupancy)
        for _repair in range(_COLOR_REPAIR_PASSES):
            self.body_locks.fill_(_LOCK_FREE)
            self._launch_rigid_families(_propose_and_claim_two_endpoint_repairs)
            self._launch_rigid_families(_commit_two_endpoint_repairs)
        for family in self._families:
            family.compact(self.color_count, self.device)

    def prepare(self, inverse_weight: wp.array[mat66f] | None, prepared_status: wp.array[wp.int32]) -> None:
        self.build_colors()
        problem = self.problem
        if problem is not None:
            prepare_jacobi_projection_data(
                problem.friction_world,
                problem.friction_local,
                problem.world_friction_count,
                problem.friction_body_first,
                problem.friction_body_second,
                problem.friction_jacobian_first,
                problem.friction_jacobian_second,
                problem.contact_world,
                problem.contact_local,
                problem.world_contact_count,
                problem.contact_body_first,
                problem.contact_body_second,
                problem.contact_jacobian_first,
                problem.contact_jacobian_second,
                problem.contact_bias,
                problem.contact_friction,
                problem.limit_world,
                problem.limit_local,
                problem.world_limit_count,
                problem.limit_body_first,
                problem.limit_body_second,
                problem.limit_jacobian_first,
                problem.limit_jacobian_second,
                problem.body_constraint_count,
                problem.static_body_constraint_count,
                inverse_weight,
                problem.friction_projection_delassus,
                problem.contact_projection_delassus,
                problem.limit_projection_delassus,
                prepared_status,
            )
            if self.rigid_worker_count > 0:
                for color in range(self.color_count):
                    if self._fuse_rigid_families:
                        wp.launch(
                            _prepare_rigid_colored,
                            dim=self.rigid_worker_count,
                            inputs=[
                                self.rigid_worker_count,
                                color,
                                self.friction.counts,
                                self.friction.offsets,
                                self.friction.order,
                                problem.friction_world,
                                problem.friction_body_first,
                                problem.friction_body_second,
                                problem.friction_jacobian_first,
                                problem.friction_jacobian_second,
                                self.contact.counts,
                                self.contact.offsets,
                                self.contact.order,
                                problem.contact_world,
                                problem.contact_body_first,
                                problem.contact_body_second,
                                problem.contact_jacobian_first,
                                problem.contact_jacobian_second,
                                problem.contact_bias,
                                problem.contact_friction,
                                self.limit.counts,
                                self.limit.offsets,
                                self.limit.order,
                                problem.limit_world,
                                problem.limit_body_first,
                                problem.limit_body_second,
                                problem.limit_jacobian_first,
                                problem.limit_jacobian_second,
                                self.body_occupancy,
                                inverse_weight,
                            ],
                            outputs=[
                                self.friction_delassus,
                                self.contact_delassus,
                                self.limit_delassus,
                                prepared_status,
                            ],
                            device=self.device,
                            block_dim=_COLOR_BLOCK_DIM,
                        )
                    else:
                        if self.friction.capacity > 0:
                            wp.launch(
                                _prepare_frictions_colored,
                                dim=self.friction.worker_count,
                                inputs=[
                                    self.friction.worker_count,
                                    color,
                                    self.friction.counts,
                                    self.friction.offsets,
                                    self.friction.order,
                                    problem.friction_world,
                                    problem.friction_body_first,
                                    problem.friction_body_second,
                                    problem.friction_jacobian_first,
                                    problem.friction_jacobian_second,
                                    self.body_occupancy,
                                    inverse_weight,
                                ],
                                outputs=[self.friction_delassus, prepared_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        elif self.contact.capacity > 0:
                            wp.launch(
                                _prepare_contacts_colored,
                                dim=self.contact.worker_count,
                                inputs=[
                                    self.contact.worker_count,
                                    color,
                                    self.contact.counts,
                                    self.contact.offsets,
                                    self.contact.order,
                                    problem.contact_world,
                                    problem.contact_body_first,
                                    problem.contact_body_second,
                                    problem.contact_jacobian_first,
                                    problem.contact_jacobian_second,
                                    problem.contact_bias,
                                    problem.contact_friction,
                                    self.body_occupancy,
                                    inverse_weight,
                                ],
                                outputs=[self.contact_delassus, prepared_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        else:
                            wp.launch(
                                _prepare_frictions_colored,
                                dim=self.limit.worker_count,
                                inputs=[
                                    self.limit.worker_count,
                                    color,
                                    self.limit.counts,
                                    self.limit.offsets,
                                    self.limit.order,
                                    problem.limit_world,
                                    problem.limit_body_first,
                                    problem.limit_body_second,
                                    problem.limit_jacobian_first,
                                    problem.limit_jacobian_second,
                                    self.body_occupancy,
                                    inverse_weight,
                                ],
                                outputs=[self.limit_delassus, prepared_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
        else:
            prepared_status.fill_(PROJECTION_STATUS_VALID)

    def project(
        self,
        iterations: int,
        world_active: wp.array[wp.bool],
        body_world: wp.array[wp.int32] | None,
        inverse_weight: wp.array[mat66f] | None,
        projected_twist: wp.array[vec6f] | None,
        twist_delta: wp.array[vec6f] | None,
        prepared_status: wp.array[wp.int32],
        projection_status: wp.array[wp.int32],
    ) -> None:
        problem = self.problem
        world_count = world_active.shape[0]
        world_body_offset = None
        world_body_count = None
        world_friction_offset = None
        world_contact_offset = None
        world_limit_offset = None
        if problem is not None:
            model_info = getattr(getattr(problem, "model", None), "info", None)
            if model_info is not None:
                world_body_offset = model_info.bodies_offset
                world_body_count = model_info.num_bodies
            world_friction_offset = getattr(problem, "world_friction_offset", None)
            world_contact_offset = getattr(problem, "world_contact_offset", None)
            world_limit_offset = getattr(problem, "world_limit_offset", None)
        use_world_projection = _can_fuse_rigid_projection_by_world(
            self.device,
            world_count,
            required_world_arrays=(
                world_body_offset,
                world_body_count,
                world_friction_offset,
                world_contact_offset,
                world_limit_offset,
            ),
            parallel_constraint_capacity=(
                problem.friction_capacity + problem.contact_capacity + problem.limit_capacity + self.color_count - 1
            )
            // self.color_count
            if problem is not None
            else None,
            world_block_dim=_WORLD_COLOR_BLOCK_DIM,
            minimum_blocks_per_sm=1,
        )
        if use_world_projection:
            state = _make_rigid_projection_state(
                world_active, projected_twist, twist_delta, projection_status, self.body_occupancy, inverse_weight
            )
            contact_data = _RigidContactProjectionData()
            contact_data.body_first = problem.contact_body_first
            contact_data.body_second = problem.contact_body_second
            contact_data.jacobian_first = problem.contact_jacobian_first
            contact_data.jacobian_second = problem.contact_jacobian_second
            contact_data.delassus = self.contact_delassus
            contact_data.bias = problem.contact_bias
            contact_data.friction = problem.contact_friction
            contact_data.reaction = problem.contact_reaction
            data = _RigidColoredWorldData()
            data.world_color_count = self.rigid_world_color_count
            data.world_friction_offset = world_friction_offset
            data.world_friction_count = problem.world_friction_count
            data.friction_colors = self.friction.colors
            data.friction_body_first = problem.friction_body_first
            data.friction_body_second = problem.friction_body_second
            data.friction_jacobian_first = problem.friction_jacobian_first
            data.friction_jacobian_second = problem.friction_jacobian_second
            data.friction_bound = problem.friction_impulse_bound
            data.friction_colored_delassus = self.friction_delassus
            data.friction_jacobi_delassus = problem.friction_projection_delassus
            data.friction_reaction = problem.friction_reaction
            data.world_contact_offset = world_contact_offset
            data.world_contact_count = problem.world_contact_count
            data.contact_colors = self.contact.colors
            data.contact_jacobi_delassus = problem.contact_projection_delassus
            data.world_limit_offset = world_limit_offset
            data.world_limit_count = problem.world_limit_count
            data.limit_colors = self.limit.colors
            data.limit_body_first = problem.limit_body_first
            data.limit_body_second = problem.limit_body_second
            data.limit_jacobian_first = problem.limit_jacobian_first
            data.limit_jacobian_second = problem.limit_jacobian_second
            data.limit_bias = problem.limit_bias
            data.limit_colored_delassus = self.limit_delassus
            data.limit_jacobi_delassus = problem.limit_projection_delassus
            data.limit_reaction = problem.limit_reaction
            wp.launch(
                _project_rigid_colored_by_world,
                dim=(world_count, _WORLD_COLOR_BLOCK_DIM),
                block_dim=_WORLD_COLOR_BLOCK_DIM,
                inputs=[
                    iterations,
                    self.color_count,
                    world_body_offset,
                    world_body_count,
                    data,
                    contact_data,
                    state,
                    prepared_status,
                ],
                device=self.device,
            )
            return
        if problem is not None:
            wp.launch(
                _initialize_jacobi_projection_status,
                dim=world_active.shape[0],
                inputs=[world_active, prepared_status],
                outputs=[projection_status],
                device=self.device,
            )
            twist_delta.zero_()
            if self.friction.capacity > 0:
                wp.launch(
                    _warmstart_frictions_jacobi,
                    dim=self.friction.capacity,
                    inputs=[
                        problem.friction_world,
                        problem.friction_local,
                        world_active,
                        prepared_status,
                        problem.world_friction_count,
                        problem.friction_body_first,
                        problem.friction_body_second,
                        problem.friction_jacobian_first,
                        problem.friction_jacobian_second,
                        inverse_weight,
                        True,
                        problem.friction_reaction,
                    ],
                    outputs=[twist_delta],
                    device=self.device,
                )
            if self.contact.capacity > 0:
                wp.launch(
                    _warmstart_contacts_jacobi,
                    dim=self.contact.capacity,
                    inputs=[
                        problem.contact_world,
                        problem.contact_local,
                        world_active,
                        prepared_status,
                        problem.world_contact_count,
                        problem.contact_body_first,
                        problem.contact_body_second,
                        problem.contact_jacobian_first,
                        problem.contact_jacobian_second,
                        inverse_weight,
                        True,
                        problem.contact_reaction,
                    ],
                    outputs=[twist_delta],
                    device=self.device,
                )
            if self.limit.capacity > 0:
                wp.launch(
                    _warmstart_limits_jacobi,
                    dim=self.limit.capacity,
                    inputs=[
                        problem.limit_world,
                        problem.limit_local,
                        world_active,
                        prepared_status,
                        problem.world_limit_count,
                        problem.limit_body_first,
                        problem.limit_body_second,
                        problem.limit_jacobian_first,
                        problem.limit_jacobian_second,
                        inverse_weight,
                        True,
                        problem.limit_reaction,
                    ],
                    outputs=[twist_delta],
                    device=self.device,
                )
            wp.launch(
                _apply_body_delta,
                dim=projected_twist.shape[0],
                inputs=[body_world, world_active, projection_status],
                outputs=[twist_delta, projected_twist],
                device=self.device,
            )
        else:
            wp.copy(projection_status, prepared_status)
        rigid_projection_state = None
        rigid_contact_data = None
        friction_index = None
        contact_index = None
        limit_index = None
        if problem is not None:
            rigid_projection_state = _make_rigid_projection_state(
                world_active, projected_twist, twist_delta, projection_status, self.body_occupancy, inverse_weight
            )
            rigid_contact_data = _RigidContactProjectionData()
            rigid_contact_data.body_first = problem.contact_body_first
            rigid_contact_data.body_second = problem.contact_body_second
            rigid_contact_data.jacobian_first = problem.contact_jacobian_first
            rigid_contact_data.jacobian_second = problem.contact_jacobian_second
            rigid_contact_data.delassus = self.contact_delassus
            rigid_contact_data.bias = problem.contact_bias
            rigid_contact_data.friction = problem.contact_friction
            rigid_contact_data.reaction = problem.contact_reaction
            friction_index = _make_colored_projection_index(
                problem.friction_world, self.friction.counts, self.friction.offsets, self.friction.order
            )
            contact_index = _make_colored_projection_index(
                problem.contact_world, self.contact.counts, self.contact.offsets, self.contact.order
            )
            limit_index = _make_colored_projection_index(
                problem.limit_world, self.limit.counts, self.limit.offsets, self.limit.order
            )
        for _iteration in range(iterations):
            for color in range(self.color_count):
                if problem is not None:
                    if self.rigid_worker_count > 0:
                        if self._fuse_rigid_families:
                            wp.launch(
                                _project_rigid_colored,
                                dim=self.rigid_worker_count,
                                inputs=[
                                    self.rigid_worker_count,
                                    color,
                                    self.friction.counts,
                                    self.friction.offsets,
                                    self.friction.order,
                                    problem.friction_world,
                                    problem.friction_body_first,
                                    problem.friction_body_second,
                                    problem.friction_jacobian_first,
                                    problem.friction_jacobian_second,
                                    problem.friction_impulse_bound,
                                    self.friction_delassus,
                                    self.contact.counts,
                                    self.contact.offsets,
                                    self.contact.order,
                                    problem.contact_world,
                                    rigid_contact_data,
                                    self.limit.counts,
                                    self.limit.offsets,
                                    self.limit.order,
                                    problem.limit_world,
                                    problem.limit_body_first,
                                    problem.limit_body_second,
                                    problem.limit_jacobian_first,
                                    problem.limit_jacobian_second,
                                    problem.limit_bias,
                                    self.limit_delassus,
                                    rigid_projection_state,
                                ],
                                outputs=[problem.friction_reaction, problem.limit_reaction],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        elif self.friction.capacity > 0:
                            wp.launch(
                                _make_project_scalar_kernel(True, False),
                                dim=self.friction.worker_count,
                                inputs=[
                                    self.friction.worker_count,
                                    color,
                                    friction_index,
                                    problem.friction_body_first,
                                    problem.friction_body_second,
                                    problem.friction_jacobian_first,
                                    problem.friction_jacobian_second,
                                    problem.friction_impulse_bound,
                                    problem.friction_impulse_bound,
                                    self.friction_delassus,
                                    problem.friction_reaction,
                                    rigid_projection_state,
                                ],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        elif self.contact.capacity > 0:
                            wp.launch(
                                _make_project_contacts_kernel(True),
                                dim=self.contact.worker_count,
                                inputs=[
                                    self.contact.worker_count,
                                    color,
                                    contact_index,
                                    rigid_contact_data,
                                    rigid_projection_state,
                                ],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        else:
                            wp.launch(
                                _make_project_scalar_kernel(True, True),
                                dim=self.limit.worker_count,
                                inputs=[
                                    self.limit.worker_count,
                                    color,
                                    limit_index,
                                    problem.limit_body_first,
                                    problem.limit_body_second,
                                    problem.limit_jacobian_first,
                                    problem.limit_jacobian_second,
                                    problem.limit_bias,
                                    problem.limit_bias,
                                    self.limit_delassus,
                                    problem.limit_reaction,
                                    rigid_projection_state,
                                ],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                if problem is not None:
                    wp.launch(
                        _apply_body_delta,
                        dim=projected_twist.shape[0],
                        inputs=[body_world, world_active, projection_status],
                        outputs=[twist_delta, projected_twist],
                        device=self.device,
                    )
        if problem is not None:
            project_constraints_jacobi(
                1,
                world_active,
                body_world,
                problem.friction_world,
                problem.friction_local,
                problem.world_friction_count,
                problem.friction_body_first,
                problem.friction_body_second,
                problem.friction_jacobian_first,
                problem.friction_jacobian_second,
                problem.friction_impulse_bound,
                problem.friction_projection_delassus,
                problem.contact_world,
                problem.contact_local,
                problem.world_contact_count,
                problem.contact_body_first,
                problem.contact_body_second,
                problem.contact_jacobian_first,
                problem.contact_jacobian_second,
                problem.contact_bias,
                problem.contact_friction,
                problem.contact_projection_delassus,
                problem.limit_world,
                problem.limit_local,
                problem.world_limit_count,
                problem.limit_body_first,
                problem.limit_body_second,
                problem.limit_jacobian_first,
                problem.limit_jacobian_second,
                problem.limit_bias,
                problem.limit_projection_delassus,
                inverse_weight,
                projected_twist,
                twist_delta,
                problem.contact_reaction,
                problem.limit_reaction,
                problem.friction_reaction,
                prepared_status,
                projection_status,
                warm_start=False,
            )
