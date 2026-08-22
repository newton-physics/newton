# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import vec6f
from ...geometry.keying import build_pair_key2, uint64_sentinel_value
from .kernels import _FUSED_BILATERAL_BLOCK, _FUSED_INEQUALITY_BLOCK, _sync_threads
from .projections import (
    contact_friction_normal_load as _contact_friction_normal_load,
)
from .projections import (
    project_contact_normal_update as _project_contact_normal_update,
)
from .projections import (
    project_contact_tangent_update as _project_contact_tangent_update,
)
from .types import DVIConfigStruct, DVIStatus

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32
mat33f = wp.mat33f
vec3f = wp.vec3f


@wp.kernel
def _zero_bilateral_lambdas(
    # Inputs:
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    solution_lambdas[problem_vio[wid] + row] = 0.0


@wp.kernel
def _reset_active_bilateral_delta(
    active_dim: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_delta: wp.array[float32],
):
    """Reset the implicit bilateral response materialized by a direct solve."""
    wid, row = wp.tid()
    if row < active_dim[wid]:
        bilateral_delta[bilateral_vio[wid] + row] = 0.0


@wp.kernel
def _reconstruct_fused_bilateral_solution(
    problem_dim: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    response_mio: wp.array[int32],
    response_stride: wp.array[int32],
    bilateral_response: wp.array[float32],
    initial_unilateral_lambdas: wp.array[float32],
    bilateral_delta: wp.array[float32],
    enable_compact_schur: wp.bool,
    solution_lambdas: wp.array[float32],
):
    """Recover the exact final bilateral iterate from a fused unilateral solve."""
    wid, row = wp.tid()
    njc = problem_njc[wid]
    if row >= njc:
        return
    vio = problem_vio[wid]
    bvio = bilateral_vio[wid]
    nu = problem_dim[wid] - njc
    value = solution_lambdas[vio + row]
    if enable_compact_schur and nu <= njc:
        offset = response_mio[wid]
        stride = response_stride[wid]
        for unilateral in range(nu):
            delta = solution_lambdas[vio + njc + unilateral] - initial_unilateral_lambdas[vio + njc + unilateral]
            value -= bilateral_response[offset + row * stride + unilateral] * delta
    else:
        value += bilateral_delta[bvio + row]
    solution_lambdas[vio + row] = value


@wp.kernel
def _build_sparse_bilateral_rhs(
    # Inputs:
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_v_f: wp.array[float32],
    state_v_aug: wp.array[float32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    # Outputs:
    bilateral_rhs: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]
    rhs = -(state_v_aug[pvio + row] + problem_v_f[pvio + row])
    bilateral_rhs[bvio + row] = bilateral_P[bvio + row] * rhs


@wp.kernel
def _sparse_delassus_gemv_rows(
    # Matrix data:
    dims: wp.array2d[int32],
    num_nzb: wp.array[int32],
    nzb_start: wp.array[int32],
    nzb_coords: wp.array2d[int32],
    nzb_values: wp.array[vec6f],
    row_start: wp.array[int32],
    col_start: wp.array[int32],
    # Row ranges:
    problem_dim: wp.array[int32],
    problem_njc: wp.array[int32],
    row_kind: int32,
    # Regularization:
    eta: wp.array[float32],
    # Vectors:
    body_space: wp.array[float32],
    y: wp.array[float32],
    lambdas: wp.array[float32],
    # Mask:
    world_mask: wp.array[bool],
):
    wid, block_idx = wp.tid()

    if not world_mask[wid]:
        return

    dim = problem_dim[wid]
    njc = problem_njc[wid]

    if block_idx < dim:
        row = block_idx
        row_active = row < njc
        if row_kind == int32(1):
            row_active = row >= njc
        if row_active:
            vec_idx = row_start[wid] + row
            wp.atomic_add(y, vec_idx, eta[vec_idx] * lambdas[vec_idx])

    if block_idx >= num_nzb[wid]:
        return

    global_block_idx = nzb_start[wid] + block_idx
    block_coord = nzb_coords[global_block_idx]
    row = block_coord[0]
    if row < 0 or row >= dim:
        return

    row_active = row < njc
    if row_kind == int32(1):
        row_active = row >= njc
    if not row_active:
        return

    # The body-space input already contains M^-1 * J^T * lambda. Accumulate
    # selected rows of J times that vector; eta * lambda supplies R * lambda.
    block = nzb_values[global_block_idx]
    x_idx_base = col_start[wid] + block_coord[1]
    acc = float32(0.0)
    for j in range(6):
        acc += block[j] * body_space[x_idx_base + j]

    wp.atomic_add(y, row_start[wid] + row, acc)


@wp.kernel
def _map_active_limits(
    limits_model_active: wp.array[int32],
    limits_wid: wp.array[int32],
    limits_lid: wp.array[int32],
    limits_bids: wp.array[wp.vec2i],
    model_body_inv_mass: wp.array[float32],
    problem_lio: wp.array[int32],
    problem_uio: wp.array[int32],
    limit_indices: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
):
    """Map active limits into the unified inequality topology."""
    limit_id = wp.tid()
    if limit_id < limits_model_active[0]:
        wid = limits_wid[limit_id]
        lid = limits_lid[limit_id]
        limit_indices[problem_lio[wid] + lid] = limit_id
        bids = limits_bids[limit_id]
        bid_a = bids[0]
        bid_b = bids[1]
        if bid_a >= int32(0) and model_body_inv_mass[bid_a] <= float32(0.0):
            bid_a = int32(-1)
        if bid_b >= int32(0) and model_body_inv_mass[bid_b] <= float32(0.0):
            bid_b = int32(-1)
        inequality_bodies[problem_uio[wid] + lid] = wp.vec2i(bid_a, bid_b)


@wp.kernel
def _map_active_contacts(
    contacts_model_active: wp.array[int32],
    contacts_wid: wp.array[int32],
    contacts_cid: wp.array[int32],
    contacts_bid_AB: wp.array[wp.vec2i],
    model_body_inv_mass: wp.array[float32],
    problem_nl: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_uio: wp.array[int32],
    contact_indices: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
):
    """Map active contacts into the unified inequality topology."""
    contact_id = wp.tid()
    if contact_id < contacts_model_active[0]:
        wid = contacts_wid[contact_id]
        cid = contacts_cid[contact_id]
        contact_indices[problem_cio[wid] + cid] = contact_id
        bids = contacts_bid_AB[contact_id]
        bid_a = bids[0]
        bid_b = bids[1]
        if bid_a >= int32(0) and model_body_inv_mass[bid_a] <= float32(0.0):
            bid_a = int32(-1)
        if bid_b >= int32(0) and model_body_inv_mass[bid_b] <= float32(0.0):
            bid_b = int32(-1)
        inequality_bodies[problem_uio[wid] + problem_nl[wid] + cid] = wp.vec2i(bid_a, bid_b)


@wp.kernel
def _prepare_contact_pair_sort(
    contacts_model_active: wp.array[int32],
    contacts_gid_AB: wp.array[wp.vec2i],
    sorted_keys: wp.array[wp.uint64],
    sorted_to_unsorted_map: wp.array[int32],
):
    """Prepare explicit geometry-pair keys and source indices for radix sort."""
    contact_id = wp.tid()
    if contact_id < contacts_model_active[0]:
        gids = contacts_gid_AB[contact_id]
        sorted_keys[contact_id] = build_pair_key2(wp.uint32(gids[0]), wp.uint32(gids[1]))
        sorted_to_unsorted_map[contact_id] = contact_id
    else:
        sorted_keys[contact_id] = uint64_sentinel_value()


@wp.kernel
def _prepare_contact_world_sort(
    contacts_model_active: wp.array[int32],
    contacts_wid: wp.array[int32],
    sorted_to_unsorted_map: wp.array[int32],
    sorted_keys: wp.array[wp.uint64],
):
    """Make a second stable contact sort world-major without losing pair order."""
    sorted_id = wp.tid()
    if sorted_id < contacts_model_active[0]:
        contact_id = sorted_to_unsorted_map[sorted_id]
        sorted_keys[sorted_id] = build_pair_key2(wp.uint32(contacts_wid[contact_id]), wp.uint32(sorted_id))
    else:
        sorted_keys[sorted_id] = uint64_sentinel_value()


@wp.kernel
def _prefix_active_contacts_by_world(
    num_worlds: int32,
    problem_nc: wp.array[int32],
    contact_world_starts: wp.array[int32],
):
    """Compute model-active contact offsets after sorting contacts by world."""
    if wp.tid() == 0:
        contact_world_starts[0] = int32(0)
        for wid in range(num_worlds):
            contact_world_starts[wid + int32(1)] = contact_world_starts[wid] + problem_nc[wid]


@wp.kernel
def _map_ordered_active_contacts(
    contacts_model_active: wp.array[int32],
    contacts_wid: wp.array[int32],
    contacts_cid: wp.array[int32],
    contacts_bid_AB: wp.array[wp.vec2i],
    sorted_to_unsorted_map: wp.array[int32],
    contact_world_starts: wp.array[int32],
    model_body_inv_mass: wp.array[float32],
    problem_nl: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_uio: wp.array[int32],
    contact_indices: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
    inequality_order: wp.array[int32],
):
    """Map contacts while retaining geometry-pair order in private schedule scratch."""
    sorted_id = wp.tid()
    if sorted_id < contacts_model_active[0]:
        contact_id = sorted_to_unsorted_map[sorted_id]
        wid = contacts_wid[contact_id]
        cid = contacts_cid[contact_id]
        contact_indices[problem_cio[wid] + cid] = contact_id
        bids = contacts_bid_AB[contact_id]
        bid_a = bids[0]
        bid_b = bids[1]
        if bid_a >= int32(0) and model_body_inv_mass[bid_a] <= float32(0.0):
            bid_a = int32(-1)
        if bid_b >= int32(0) and model_body_inv_mass[bid_b] <= float32(0.0):
            bid_b = int32(-1)
        nl = problem_nl[wid]
        uio = problem_uio[wid]
        uid = nl + cid
        inequality_bodies[uio + uid] = wp.vec2i(bid_a, bid_b)
        local_sorted_id = sorted_id - contact_world_starts[wid]
        inequality_order[uio + wid + nl + local_sorted_id] = uid


@wp.kernel
def _mark_contact_group_boundaries(
    contacts_model_active: wp.array[int32],
    sorted_to_unsorted_map: wp.array[int32],
    contacts_cid: wp.array[int32],
    problem_uio: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
    group_flags: wp.array[int32],
):
    sorted_id = wp.tid()
    boundary = int32(0)
    if sorted_id < contacts_model_active[0]:
        boundary = int32(1)
        if sorted_id > int32(0):
            contact_id = sorted_to_unsorted_map[sorted_id]
            previous_id = sorted_to_unsorted_map[sorted_id - int32(1)]
            pair = inequality_bodies[problem_uio[0] + contacts_cid[contact_id]]
            previous_pair = inequality_bodies[problem_uio[0] + contacts_cid[previous_id]]
            boundary = int32(pair[0] != previous_pair[0] or pair[1] != previous_pair[1])
    group_flags[sorted_id] = boundary


@wp.kernel
def _compact_contact_group_starts(
    contacts_model_active: wp.array[int32],
    group_flags: wp.array[int32],
    group_prefix: wp.array[int32],
    compact_group_starts: wp.array[int32],
):
    sorted_id = wp.tid()
    if sorted_id < contacts_model_active[0] and group_flags[sorted_id] != int32(0):
        compact_group_starts[group_prefix[sorted_id] - int32(1)] = sorted_id


@wp.kernel
def _compare_compact_contact_topology(
    contacts_model_active: wp.array[int32],
    group_prefix: wp.array[int32],
    problem_uio: wp.array[int32],
    sorted_to_unsorted_map: wp.array[int32],
    contacts_cid: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
    compact_group_starts: wp.array[int32],
    cached_group_pairs: wp.array[wp.vec2i],
    cached_group_count: wp.array[int32],
    cache_valid: wp.array[int32],
    topology_changed: wp.array[int32],
):
    group = wp.tid()
    nc = contacts_model_active[0]
    num_groups = group_prefix[wp.max(nc - int32(1), int32(0))]
    if nc == int32(0):
        num_groups = int32(0)
    if group == int32(0) and (cache_valid[0] == int32(0) or cached_group_count[0] != num_groups):
        wp.atomic_max(topology_changed, 0, int32(1))
    if cache_valid[0] != int32(0) and cached_group_count[0] == num_groups and group < num_groups:
        contact_id = sorted_to_unsorted_map[compact_group_starts[group]]
        pair = inequality_bodies[problem_uio[0] + contacts_cid[contact_id]]
        cached = cached_group_pairs[group]
        if pair[0] != cached[0] or pair[1] != cached[1]:
            wp.atomic_max(topology_changed, 0, int32(1))


@wp.kernel
def _color_compact_contact_groups(
    contacts_model_active: wp.array[int32],
    group_prefix: wp.array[int32],
    problem_uio: wp.array[int32],
    sorted_to_unsorted_map: wp.array[int32],
    contacts_cid: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
    body_color_masks: wp.array[wp.uint64],
    compact_group_starts: wp.array[int32],
    group_colors: wp.array[int32],
    inequality_num_colors: wp.array[int32],
    contact_group_count: wp.array[int32],
    inequality_color_starts: wp.array[int32],
    groups_by_color: wp.array[int32],
    cached_group_pairs: wp.array[wp.vec2i],
    cached_group_count: wp.array[int32],
    cached_num_colors: wp.array[int32],
    cached_color_starts: wp.array[int32],
    cache_valid: wp.array[int32],
    topology_changed: wp.array[int32],
):
    if wp.tid() != 0:
        return
    num_groups = group_prefix[wp.max(contacts_model_active[0] - int32(1), int32(0))]
    if contacts_model_active[0] == int32(0):
        num_groups = int32(0)
    contact_group_count[0] = num_groups
    uio = problem_uio[0]
    if cache_valid[0] != int32(0) and topology_changed[0] == int32(0):
        num_colors = cached_num_colors[0]
        inequality_num_colors[0] = num_colors
        for color in range(num_colors + int32(1)):
            inequality_color_starts[uio + color] = cached_color_starts[color]
        return
    num_colors = int32(0)
    for group in range(num_groups):
        contact_id = sorted_to_unsorted_map[compact_group_starts[group]]
        pair = inequality_bodies[uio + contacts_cid[contact_id]]
        cached_group_pairs[group] = pair
        forbidden = wp.uint64(0)
        if pair[0] >= int32(0):
            forbidden |= body_color_masks[pair[0]]
        if pair[1] >= int32(0):
            forbidden |= body_color_masks[pair[1]]
        color = _lowest_set_color(wp.int64(forbidden) ^ wp.int64(-1))
        if color < int32(0):
            color = num_colors
        group_colors[group] = color
        num_colors = wp.max(num_colors, color + int32(1))
        if color < int32(64):
            color_bit = wp.uint64(1) << wp.uint64(color)
            if pair[0] >= int32(0):
                body_color_masks[pair[0]] |= color_bit
            if pair[1] >= int32(0):
                body_color_masks[pair[1]] |= color_bit
    inequality_num_colors[0] = num_colors
    for color in range(num_colors + int32(1)):
        inequality_color_starts[uio + color] = int32(0)
    for group in range(num_groups):
        inequality_color_starts[uio + group_colors[group] + int32(1)] += int32(1)
    for color in range(num_colors):
        inequality_color_starts[uio + color + int32(1)] += inequality_color_starts[uio + color]
    for group in range(num_groups):
        color = group_colors[group]
        slot = inequality_color_starts[uio + color]
        groups_by_color[slot] = group
        inequality_color_starts[uio + color] = slot + int32(1)
    previous = int32(0)
    for color in range(num_colors + int32(1)):
        cursor = inequality_color_starts[uio + color]
        inequality_color_starts[uio + color] = previous
        cached_color_starts[color] = previous
        previous = cursor
    cached_group_count[0] = num_groups
    cached_num_colors[0] = num_colors
    cache_valid[0] = int32(1)


@wp.kernel
def _prepare_colored_contact_group_sizes(
    contacts_model_active: wp.array[int32],
    contact_group_count: wp.array[int32],
    compact_group_starts: wp.array[int32],
    groups_by_color: wp.array[int32],
    colored_group_sizes: wp.array[int32],
):
    scheduled_group = wp.tid()
    num_groups = contact_group_count[0]
    nc = contacts_model_active[0]
    if scheduled_group < num_groups:
        group = groups_by_color[scheduled_group]
        start = compact_group_starts[group]
        end = compact_group_starts[group + int32(1)] if group + int32(1) < num_groups else nc
        colored_group_sizes[scheduled_group] = end - start


@wp.kernel
def _expand_colored_contact_groups(
    contacts_model_active: wp.array[int32],
    contact_group_count: wp.array[int32],
    problem_uio: wp.array[int32],
    sorted_to_unsorted_map: wp.array[int32],
    contacts_cid: wp.array[int32],
    compact_group_starts: wp.array[int32],
    groups_by_color: wp.array[int32],
    colored_group_prefix: wp.array[int32],
    inequality_ids_by_color: wp.array[int32],
    inequality_group_starts: wp.array[int32],
):
    scheduled_group = wp.tid()
    nc = contacts_model_active[0]
    num_groups = contact_group_count[0]
    uio = problem_uio[0]
    if scheduled_group < num_groups:
        group = groups_by_color[scheduled_group]
        ordered_start = compact_group_starts[group]
        ordered_end = compact_group_starts[group + int32(1)] if group + int32(1) < num_groups else nc
        slot_end = colored_group_prefix[scheduled_group]
        slot_start = slot_end - (ordered_end - ordered_start)
        inequality_group_starts[uio + scheduled_group] = slot_start
        for ordered_id in range(ordered_start, ordered_end):
            contact_id = sorted_to_unsorted_map[ordered_id]
            inequality_ids_by_color[uio + slot_start + ordered_id - ordered_start] = contacts_cid[contact_id]
    if scheduled_group == num_groups:
        inequality_group_starts[uio + num_groups] = nc


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return ((int)__ffsll((long long)mask)) - 1;
#else
if (mask == 0) return -1;
int position = 0;
while ((mask & 1LL) == 0LL) { mask >>= 1; position++; }
return position;
#endif
""")
def _lowest_set_color(mask: wp.int64) -> wp.int32:
    """Return the lowest set bit, or -1 when no bit is set."""
    ...


@wp.kernel
def _color_mapped_dvi_inequalities(
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_uio: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
    body_color_masks: wp.array[wp.uint64],
    inequality_colors: wp.array[int32],
    inequality_num_colors: wp.array[int32],
    inequality_ids_by_color: wp.array[int32],
    inequality_color_starts: wp.array[int32],
):
    """Greedily color one world per thread using per-body 64-bit masks.

    This favors the many-small-world workload. Unusually high-degree graphs
    that exhaust 64 colors assign fresh colors without a color cap. The same
    pass emits compact color ranges shared by dense and sparse PGS.
    """
    wid = wp.tid()
    nu = problem_nl[wid] + problem_nc[wid]
    uio = problem_uio[wid]
    num_colors = int32(0)
    for uid in range(nu):
        pair = inequality_bodies[uio + uid]
        forbidden = wp.uint64(0)
        if pair[0] >= int32(0):
            forbidden |= body_color_masks[pair[0]]
        if pair[1] >= int32(0):
            forbidden |= body_color_masks[pair[1]]

        color = _lowest_set_color(wp.int64(forbidden) ^ wp.int64(-1))
        if color < int32(0):
            # A fresh color is always conflict-free and avoids a superlinear
            # search in dense manifolds that share a body.
            color = num_colors
        inequality_colors[uio + uid] = color
        num_colors = wp.max(num_colors, color + int32(1))
        if color < int32(64):
            color_bit = wp.uint64(1) << wp.uint64(color)
            if pair[0] >= int32(0):
                body_color_masks[pair[0]] |= color_bit
            if pair[1] >= int32(0):
                body_color_masks[pair[1]] |= color_bit

    inequality_num_colors[wid] = num_colors

    schedule_offset = uio + wid
    for color in range(num_colors + int32(1)):
        inequality_color_starts[schedule_offset + color] = int32(0)
    for uid in range(nu):
        color = inequality_colors[uio + uid]
        inequality_color_starts[schedule_offset + color + int32(1)] += int32(1)
    for color in range(num_colors):
        inequality_color_starts[schedule_offset + color + int32(1)] += inequality_color_starts[schedule_offset + color]
    for uid in range(nu):
        color = inequality_colors[uio + uid]
        slot = inequality_color_starts[schedule_offset + color]
        inequality_ids_by_color[uio + slot] = uid
        inequality_color_starts[schedule_offset + color] = slot + int32(1)
    previous = int32(0)
    for color in range(num_colors + int32(1)):
        cursor = inequality_color_starts[schedule_offset + color]
        inequality_color_starts[schedule_offset + color] = previous
        previous = cursor


@wp.kernel
def _group_mapped_dvi_inequalities(
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_uio: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
    body_color_masks: wp.array[wp.uint64],
    inequality_colors: wp.array[int32],
    inequality_num_colors: wp.array[int32],
    inequality_ids_by_color: wp.array[int32],
    inequality_color_starts: wp.array[int32],
    inequality_group_starts: wp.array[int32],
    inequality_order: wp.array[int32],
    use_contact_order: wp.bool,
):
    """Color consecutive contact groups and emit group ranges for sparse PGS."""
    wid = wp.tid()
    nl = problem_nl[wid]
    nu = nl + problem_nc[wid]
    uio = problem_uio[wid]
    num_colors = int32(0)
    previous_color = int32(-1)
    previous_pair = wp.vec2i(-1, -1)
    order_offset = uio + wid
    for ordered_id in range(nu):
        uid = ordered_id
        if use_contact_order and ordered_id >= nl:
            uid = inequality_order[order_offset + ordered_id]
        pair = inequality_bodies[uio + uid]
        grouped = ordered_id > nl and pair[0] == previous_pair[0] and pair[1] == previous_pair[1]
        color = previous_color
        if not grouped:
            forbidden = wp.uint64(0)
            if pair[0] >= int32(0):
                forbidden |= body_color_masks[pair[0]]
            if pair[1] >= int32(0):
                forbidden |= body_color_masks[pair[1]]
            color = _lowest_set_color(wp.int64(forbidden) ^ wp.int64(-1))
            if color < int32(0):
                color = num_colors
            num_colors = wp.max(num_colors, color + int32(1))
            if color < int32(64):
                color_bit = wp.uint64(1) << wp.uint64(color)
                if pair[0] >= int32(0):
                    body_color_masks[pair[0]] |= color_bit
                if pair[1] >= int32(0):
                    body_color_masks[pair[1]] |= color_bit
        inequality_colors[uio + uid] = color
        previous_color = color
        previous_pair = pair
    inequality_num_colors[wid] = num_colors
    schedule_offset = uio + wid
    for color in range(num_colors + int32(1)):
        inequality_color_starts[schedule_offset + color] = int32(0)
    for ordered_id in range(nu):
        uid = ordered_id
        if use_contact_order and ordered_id >= nl:
            uid = inequality_order[order_offset + ordered_id]
        color = inequality_colors[uio + uid]
        inequality_color_starts[schedule_offset + color + int32(1)] += int32(1)
    for color in range(num_colors):
        inequality_color_starts[schedule_offset + color + int32(1)] += inequality_color_starts[schedule_offset + color]
    for ordered_id in range(nu):
        uid = ordered_id
        if use_contact_order and ordered_id >= nl:
            uid = inequality_order[order_offset + ordered_id]
        color = inequality_colors[uio + uid]
        slot = inequality_color_starts[schedule_offset + color]
        inequality_ids_by_color[uio + slot] = uid
        inequality_color_starts[schedule_offset + color] = slot + int32(1)
    previous = int32(0)
    for color in range(num_colors + int32(1)):
        cursor = inequality_color_starts[schedule_offset + color]
        inequality_color_starts[schedule_offset + color] = previous
        previous = cursor
    group_schedule_offset = uio + wid
    group_count = int32(0)
    contact_start = int32(0)
    for color in range(num_colors):
        contact_end = inequality_color_starts[schedule_offset + color + int32(1)]
        inequality_color_starts[schedule_offset + color] = group_count
        previous_uid = int32(-1)
        for slot in range(contact_start, contact_end):
            uid = inequality_ids_by_color[uio + slot]
            pair = inequality_bodies[uio + uid]
            previous_pair = wp.vec2i(-1, -1)
            if previous_uid >= int32(0):
                previous_pair = inequality_bodies[uio + previous_uid]
            new_group = uid < nl or previous_uid < nl or pair[0] != previous_pair[0] or pair[1] != previous_pair[1]
            if new_group:
                inequality_group_starts[group_schedule_offset + group_count] = slot
                group_count += int32(1)
            previous_uid = uid
        contact_start = contact_end
    inequality_color_starts[schedule_offset + num_colors] = group_count
    inequality_group_starts[group_schedule_offset + group_count] = nu


@wp.kernel
def _assemble_sparse_bilateral_unilateral_coupling(
    bsm_num_nzb: wp.array[int32],
    bsm_nzb_start: wp.array[int32],
    bsm_nzb_coords: wp.array2d[int32],
    mass_weighted_nzb_values: wp.array[vec6f],
    jacobian_nzb_values: wp.array[vec6f],
    problem_dim: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lio: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_P: wp.array[float32],
    limit_indices: wp.array[int32],
    contact_indices: wp.array[int32],
    limit_nzb_offsets: wp.array[int32],
    contact_nzb_offsets: wp.array[int32],
    bilateral_world_row_offsets: wp.array[int32],
    bilateral_row_starts: wp.array[int32],
    bilateral_row_nzb_indices: wp.array[int32],
    response_mio: wp.array[int32],
    response_stride: wp.array[int32],
    coupling: wp.array[float32],
):
    wid, row, unilateral = wp.tid()
    njc = problem_njc[wid]
    col = njc + unilateral
    if row >= njc or col >= problem_dim[wid]:
        return

    matrix_end = bsm_nzb_start[wid] + bsm_num_nzb[wid]
    col_block_0 = int32(-1)
    col_block_1 = int32(-1)
    nl = problem_nl[wid]
    if unilateral < nl:
        mapped_limit = limit_indices[problem_lio[wid] + unilateral]
        if mapped_limit >= int32(0):
            col_block_0 = limit_nzb_offsets[mapped_limit]
            candidate = col_block_0 + int32(1)
            if candidate < matrix_end and bsm_nzb_coords[candidate, 0] == col:
                col_block_1 = candidate
    else:
        contact_component = unilateral - nl
        cid = contact_component / int32(3)
        if cid < problem_nc[wid]:
            component = contact_component - int32(3) * cid
            mapped_contact = contact_indices[problem_cio[wid] + cid]
            if mapped_contact >= int32(0):
                col_block_0 = contact_nzb_offsets[mapped_contact] + component
                candidate = col_block_0 + int32(3)
                if candidate < matrix_end and bsm_nzb_coords[candidate, 0] == col:
                    col_block_1 = candidate

    value = float32(0.0)
    cached_row = bilateral_world_row_offsets[wid] + row
    for entry in range(bilateral_row_starts[cached_row], bilateral_row_starts[cached_row + int32(1)]):
        row_block = bilateral_row_nzb_indices[entry]
        row_body = bsm_nzb_coords[row_block, 1]
        mass_weighted = mass_weighted_nzb_values[row_block]
        if col_block_0 >= int32(0) and bsm_nzb_coords[col_block_0, 1] == row_body:
            jacobian = jacobian_nzb_values[col_block_0]
            for component in range(6):
                value += mass_weighted[component] * jacobian[component]
        if col_block_1 >= int32(0) and bsm_nzb_coords[col_block_1, 1] == row_body:
            jacobian = jacobian_nzb_values[col_block_1]
            for component in range(6):
                value += mass_weighted[component] * jacobian[component]
    value *= problem_P[problem_vio[wid] + col]
    offset = response_mio[wid]
    coupling[offset + row * response_stride[wid] + unilateral] = value


@wp.kernel
def _cache_sparse_contact_diagonal(
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_P: wp.array[float32],
    problem_diag: wp.array[float32],
    projected_diag: wp.array[float32],
):
    """Cache the invariant projected diagonal of contact rows."""
    wid, cid = wp.tid()
    if cid >= problem_nc[wid]:
        return
    vec_idx = problem_vio[wid] + problem_ccgo[wid] + int32(3) * cid
    for component in range(3):
        P_i = problem_P[vec_idx + component]
        projected_diag[vec_idx + component] = wp.abs(problem_diag[vec_idx + component]) * P_i * P_i


@wp.kernel
def _cache_sparse_projected_diagonal(
    problem_dim: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_P: wp.array[float32],
    problem_diag: wp.array[float32],
    response_mio: wp.array[int32],
    response_stride: wp.array[int32],
    bilateral_coupling: wp.array[float32],
    bilateral_response: wp.array[float32],
    solution_lambdas: wp.array[float32],
    initial_unilateral_lambdas: wp.array[float32],
    projected_diag: wp.array[float32],
):
    """Cache the Schur diagonal and snapshot the warm-started unilateral iterate."""
    wid, unilateral_row = wp.tid()
    njc = problem_njc[wid]
    row = njc + unilateral_row
    if row >= problem_dim[wid]:
        return
    vio = problem_vio[wid]
    vec_idx = vio + row
    P_i = problem_P[vec_idx]
    value = wp.abs(problem_diag[vec_idx]) * P_i * P_i
    offset = response_mio[wid]
    stride = response_stride[wid]
    for bilateral_row in range(njc):
        index = offset + bilateral_row * stride + unilateral_row
        value -= bilateral_coupling[index] * bilateral_response[index]
    projected_diag[vec_idx] = value
    initial_unilateral_lambdas[vec_idx] = solution_lambdas[vec_idx]


@wp.kernel
def _select_parallel_contact_colors(
    problem_nc: wp.array[int32],
    inequality_num_colors: wp.array[int32],
    min_contacts: int32,
    max_colors: int32,
    parallel_contact_colors: wp.array[int32],
):
    """Select the bounded multi-block contact schedule without a host readback."""
    if wp.tid() == 0:
        num_colors = inequality_num_colors[0]
        parallel_contact_colors[0] = int32(
            problem_nc[0] >= min_contacts and num_colors > int32(0) and num_colors <= max_colors
        )


@wp.kernel
def _solve_dvi_sparse_contacts_pgs(
    bsm_num_nzb: wp.array[int32],
    bsm_nzb_start: wp.array[int32],
    bsm_nzb_coords: wp.array2d[int32],
    bsm_nzb_values: wp.array[vec6f],
    jacobian_nzb_values: wp.array[vec6f],
    bsm_row_start: wp.array[int32],
    bsm_col_start: wp.array[int32],
    contact_nzb_offsets: wp.array[int32],
    contact_indices: wp.array[int32],
    contact_bid_AB: wp.array[wp.vec2i],
    model_bodies_offset: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_uio: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    problem_v_b: wp.array[float32],
    projected_diag: wp.array[float32],
    eta: wp.array[float32],
    inequality_num_colors: wp.array[int32],
    inequality_ids_by_color: wp.array[int32],
    inequality_color_starts: wp.array[int32],
    inequality_group_starts: wp.array[int32],
    inequality_tangent_cross: wp.array[float32],
    parallel_contact_colors: wp.array[int32],
    parallel_color_node: bool,
    selected_sweep: int32,
    selected_color_ordinal: int32,
    parallel_group_stride: int32,
    parallel_group_width: int32,
    block_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    body_space: wp.array[float32],
    solution_lambdas: wp.array[float32],
):
    """Apply sparse PGS specialized for contact-only body-space systems."""
    tid = wp.tid()
    threads_per_world = int32(wp.block_dim())
    lane = tid % threads_per_world
    wid = tid / threads_per_world
    if parallel_color_node:
        if parallel_contact_colors[0] == int32(0):
            return
        # A color node is one-world only; global thread ids own disjoint groups.
        wid = int32(0)
        # Keep the width-selected useful-warp count, but pack each warp's
        # independent group owners into adjacent lanes so their schedule and
        # contact-row reads can share memory transactions.
        warp_lane = tid % int32(32)
        groups_per_warp = int32(32) / parallel_group_width
        if warp_lane >= groups_per_warp:
            return
        lane = (tid / int32(32)) * groups_per_warp + warp_lane
        threads_per_world = parallel_group_stride
    elif parallel_contact_colors[0] != int32(0):
        return
    cfg = solver_config[wid]
    if block_iteration >= int32(0) and block_iteration >= cfg.max_alternating_iterations:
        return
    nc = problem_nc[wid]
    if nc == 0:
        return
    cio = problem_cio[wid]
    uio = problem_uio[wid]
    schedule_offset = uio + wid
    ccgo = problem_ccgo[wid]
    vio = problem_vio[wid]
    row_start = bsm_row_start[wid]
    col_start = bsm_col_start[wid]
    bodies_offset = model_bodies_offset[wid]
    sweep_count = cfg.inequality_sweeps_per_iteration
    first_tangent_sweep = int32(0)
    if block_iteration == int32(_FUSED_INEQUALITY_BLOCK):
        tangent_sweep_count = sweep_count * cfg.max_alternating_iterations / int32(2)
        sweep_count = (sweep_count + int32(1)) * cfg.max_alternating_iterations
        first_tangent_sweep = sweep_count - tangent_sweep_count
        # Contact-block sweeps are cheaper than separate normal/tangent phases,
        # so spend the saved body reload on a full frictional pass budget. One
        # final polishing sweep preserves the legacy stack convergence budget.
        sweep_count += tangent_sweep_count
        sweep_count += int32(1)
    elif block_iteration == int32(_FUSED_BILATERAL_BLOCK):
        sweep_count *= cfg.max_alternating_iterations
    sweep_begin = int32(0)
    sweep_end = sweep_count
    if parallel_color_node:
        if selected_sweep >= sweep_count:
            return
        sweep_begin = selected_sweep
        sweep_end = selected_sweep + int32(1)
    for sweep in range(sweep_begin, sweep_end):
        tangent_pass = sweep >= first_tangent_sweep and (sweep - first_tangent_sweep) % int32(2) != int32(0)
        solve_normal = not tangent_pass
        tangent_ordinal = int32(0)
        if tangent_pass:
            tangent_ordinal = (sweep - first_tangent_sweep) / int32(2)
        # Only reverse friction passes; normal support loads use a stable order.
        reverse_colors = tangent_pass and tangent_ordinal % int32(2) != int32(0)
        num_colors = inequality_num_colors[wid]
        color_count = num_colors
        if parallel_color_node:
            if selected_color_ordinal >= num_colors:
                return
            color_count = int32(1)
        for local_color_index in range(color_count):
            color = local_color_index
            if parallel_color_node:
                color = selected_color_ordinal
            if reverse_colors:
                color = num_colors - int32(1) - color
            group_start = inequality_color_starts[schedule_offset + color]
            group_end = inequality_color_starts[schedule_offset + color + int32(1)]
            group = group_start + lane
            while group < group_end:
                color_start = inequality_group_starts[schedule_offset + group]
                color_end = inequality_group_starts[schedule_offset + group + int32(1)]
                color_slot = color_start
                color_step = int32(1)
                if reverse_colors:
                    color_slot = color_end - int32(1)
                    color_step = int32(-1)
                local_x_idx_0 = int32(-1)
                local_x_idx_1 = int32(-1)
                local_body_0 = vec6f(0.0)
                local_body_1 = vec6f(0.0)
                first_cid = inequality_ids_by_color[uio + color_slot]
                first_contact_id = contact_indices[cio + first_cid]
                block_count = int32(3)
                if first_contact_id >= int32(0):
                    # Contact Jacobians store B's three rows first, followed by A's when present.
                    first_bids = contact_bid_AB[first_contact_id]
                    local_x_idx_0 = col_start + int32(6) * (first_bids[1] - bodies_offset)
                    for j in range(6):
                        local_body_0[j] = body_space[local_x_idx_0 + j]
                    if first_bids[0] >= int32(0):
                        block_count = int32(6)
                        local_x_idx_1 = col_start + int32(6) * (first_bids[0] - bodies_offset)
                        for j in range(6):
                            local_body_1[j] = body_space[local_x_idx_1 + j]
                while color_slot >= color_start and color_slot < color_end:
                    cid = inequality_ids_by_color[uio + color_slot]
                    contact_id = contact_indices[cio + cid]
                    if contact_id >= int32(0):
                        row = ccgo + int32(3) * cid
                        vec_idx = vio + row
                        nzb_offset = contact_nzb_offsets[contact_id]

                        # Contact rows are B(t0, t1, n), optionally followed by A.
                        # Keep the fixed topology explicit so Jacobian rows load with
                        # their mass-weighted rows and remain live through the update.
                        normal_value = eta[row_start + row + int32(2)] * solution_lambdas[vec_idx + int32(2)]
                        block_n_0 = bsm_nzb_values[nzb_offset + int32(2)]
                        row_n_0 = jacobian_nzb_values[nzb_offset + int32(2)]
                        for j in range(6):
                            normal_value += block_n_0[j] * local_body_0[j]
                        block_n_1 = vec6f(0.0)
                        row_n_1 = vec6f(0.0)
                        if block_count == int32(6):
                            block_n_1 = bsm_nzb_values[nzb_offset + int32(5)]
                            row_n_1 = jacobian_nzb_values[nzb_offset + int32(5)]
                            for j in range(6):
                                normal_value += block_n_1[j] * local_body_1[j]
                        normal_value += problem_v_f[vec_idx + int32(2)]
                        lambda_n_old = solution_lambdas[vec_idx + int32(2)]
                        P_n = problem_P[vec_idx + int32(2)]
                        diagonal_n = projected_diag[vec_idx + int32(2)]
                        lambda_n_new = lambda_n_old
                        if solve_normal:
                            lambda_n_new = _project_contact_normal_update(
                                lambda_n_old, normal_value, diagonal_n, cfg.regularization, cfg.omega
                            )
                        solution_lambdas[vec_idx + int32(2)] = lambda_n_new
                        normal_delta_body = P_n * (lambda_n_new - lambda_n_old)
                        for j in range(6):
                            local_body_0[j] += row_n_0[j] * normal_delta_body
                        if block_count == int32(6):
                            for j in range(6):
                                local_body_1[j] += row_n_1[j] * normal_delta_body

                        if not tangent_pass:
                            color_slot += color_step
                            continue

                        tangent_value = wp.vec2f(
                            eta[row_start + row] * solution_lambdas[vec_idx],
                            eta[row_start + row + int32(1)] * solution_lambdas[vec_idx + int32(1)],
                        )
                        block_t0_0 = bsm_nzb_values[nzb_offset]
                        block_t1_0 = bsm_nzb_values[nzb_offset + int32(1)]
                        row_t0_0 = jacobian_nzb_values[nzb_offset]
                        row_t1_0 = jacobian_nzb_values[nzb_offset + int32(1)]
                        for j in range(6):
                            tangent_value[0] += block_t0_0[j] * local_body_0[j]
                            tangent_value[1] += block_t1_0[j] * local_body_0[j]
                        block_t0_1 = vec6f(0.0)
                        block_t1_1 = vec6f(0.0)
                        row_t0_1 = vec6f(0.0)
                        row_t1_1 = vec6f(0.0)
                        if block_count == int32(6):
                            block_t0_1 = bsm_nzb_values[nzb_offset + int32(3)]
                            block_t1_1 = bsm_nzb_values[nzb_offset + int32(4)]
                            row_t0_1 = jacobian_nzb_values[nzb_offset + int32(3)]
                            row_t1_1 = jacobian_nzb_values[nzb_offset + int32(4)]
                            for j in range(6):
                                tangent_value[0] += block_t0_1[j] * local_body_1[j]
                                tangent_value[1] += block_t1_1[j] * local_body_1[j]
                        tangent_value += wp.vec2f(problem_v_f[vec_idx], problem_v_f[vec_idx + int32(1)])
                        lambda_t_old = wp.vec2f(solution_lambdas[vec_idx], solution_lambdas[vec_idx + int32(1)])
                        P_t0 = problem_P[vec_idx]
                        P_t1 = problem_P[vec_idx + int32(1)]
                        diagonal_t0 = projected_diag[vec_idx]
                        diagonal_t1 = projected_diag[vec_idx + int32(1)]
                        off_diagonal = inequality_tangent_cross[uio + cid]
                        if tangent_ordinal == int32(0):
                            off_diagonal = float32(0.0)
                            for j in range(6):
                                off_diagonal += block_t0_0[j] * row_t1_0[j]
                            if block_count == int32(6):
                                for j in range(6):
                                    off_diagonal += block_t0_1[j] * row_t1_1[j]
                            off_diagonal *= P_t1
                            inequality_tangent_cross[uio + cid] = off_diagonal
                        lambda_t_new = _project_contact_tangent_update(
                            lambda_t_old,
                            tangent_value,
                            wp.vec2f(diagonal_t0, diagonal_t1),
                            off_diagonal,
                            cfg.regularization,
                            cfg.omega,
                            problem_mu[cio + cid]
                            * _contact_friction_normal_load(
                                lambda_n_new,
                                problem_v_b[vec_idx + int32(2)],
                                P_n,
                                diagonal_n,
                                cfg.regularization,
                                cfg.omega,
                            ),
                        )
                        solution_lambdas[vec_idx] = lambda_t_new.x
                        solution_lambdas[vec_idx + int32(1)] = lambda_t_new.y
                        tangent_delta_body = wp.vec2f(
                            P_t0 * (lambda_t_new.x - lambda_t_old.x),
                            P_t1 * (lambda_t_new.y - lambda_t_old.y),
                        )
                        for j in range(6):
                            local_body_0[j] += row_t0_0[j] * tangent_delta_body[0] + row_t1_0[j] * tangent_delta_body[1]
                        if block_count == int32(6):
                            for j in range(6):
                                local_body_1[j] += (
                                    row_t0_1[j] * tangent_delta_body[0] + row_t1_1[j] * tangent_delta_body[1]
                                )
                    color_slot += color_step
                if local_x_idx_0 >= int32(0):
                    for j in range(6):
                        body_space[local_x_idx_0 + j] = local_body_0[j]
                if local_x_idx_1 >= int32(0):
                    for j in range(6):
                        body_space[local_x_idx_1 + j] = local_body_1[j]
                group += threads_per_world
            if not parallel_color_node:
                _sync_threads()


@wp.kernel
def _solve_dvi_sparse_inequalities_pgs(
    bsm_num_nzb: wp.array[int32],
    bsm_nzb_start: wp.array[int32],
    bsm_nzb_coords: wp.array2d[int32],
    bsm_nzb_values: wp.array[vec6f],
    jacobian_nzb_values: wp.array[vec6f],
    bsm_row_start: wp.array[int32],
    bsm_col_start: wp.array[int32],
    limit_nzb_offsets: wp.array[int32],
    contact_nzb_offsets: wp.array[int32],
    limit_indices: wp.array[int32],
    contact_indices: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lio: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_uio: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    problem_v_b: wp.array[float32],
    problem_diag: wp.array[float32],
    projected_diag: wp.array[float32],
    eta: wp.array[float32],
    problem_njc: wp.array[int32],
    bilateral_vio: wp.array[int32],
    response_mio: wp.array[int32],
    response_stride: wp.array[int32],
    bilateral_coupling: wp.array[float32],
    bilateral_response: wp.array[float32],
    bilateral_delta: wp.array[float32],
    inequality_num_colors: wp.array[int32],
    inequality_ids_by_color: wp.array[int32],
    inequality_color_starts: wp.array[int32],
    inequality_group_starts: wp.array[int32],
    inequality_tangent_cross: wp.array[float32],
    block_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    body_space: wp.array[float32],
    solution_lambdas: wp.array[float32],
):
    """Apply one conflict-free sparse PGS schedule to every inequality."""
    tid = wp.tid()
    threads_per_world = int32(wp.block_dim())
    lane = tid % threads_per_world
    wid = tid / threads_per_world
    cfg = solver_config[wid]
    if block_iteration >= int32(0) and block_iteration >= cfg.max_alternating_iterations:
        return
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    nu = nl + nc
    if nu == 0:
        return
    lio = problem_lio[wid]
    cio = problem_cio[wid]
    uio = problem_uio[wid]
    schedule_offset = uio + wid
    group_schedule_offset = uio + wid
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    vio = problem_vio[wid]
    njc = problem_njc[wid]
    bilateral_offset = response_mio[wid]
    max_unilateral_rows = response_stride[wid]
    bvio = bilateral_vio[wid]
    row_start = bsm_row_start[wid]
    col_start = bsm_col_start[wid]
    matrix_end = bsm_nzb_start[wid] + bsm_num_nzb[wid]
    sweep_count = cfg.inequality_sweeps_per_iteration
    first_tangent_sweep = int32(0)
    if block_iteration == int32(_FUSED_INEQUALITY_BLOCK):
        tangent_sweep_count = sweep_count * cfg.max_alternating_iterations / int32(2)
        sweep_count = (sweep_count + int32(1)) * cfg.max_alternating_iterations
        first_tangent_sweep = sweep_count - tangent_sweep_count
    elif block_iteration == int32(_FUSED_BILATERAL_BLOCK):
        sweep_count *= cfg.max_alternating_iterations
    for _sweep in range(sweep_count):
        phase_count = int32(2)
        if block_iteration == int32(_FUSED_INEQUALITY_BLOCK) and _sweep < first_tangent_sweep:
            # Match the dense path's inequality-only normal-load warmup.
            phase_count = int32(1)
        for phase in range(phase_count):
            # Symmetric tangent ordering reduces load bias in redundant sticking patches.
            schedule_sweep = _sweep
            if block_iteration == int32(_FUSED_BILATERAL_BLOCK):
                schedule_sweep = _sweep % cfg.inequality_sweeps_per_iteration
            reverse_colors = phase == int32(1) and schedule_sweep % int32(2) != int32(0)
            num_colors = inequality_num_colors[wid]
            for color_index in range(num_colors):
                color = color_index
                if reverse_colors:
                    color = num_colors - int32(1) - color_index
                group_start = inequality_color_starts[schedule_offset + color]
                group_end = inequality_color_starts[schedule_offset + color + int32(1)]
                group = group_start + lane
                while group < group_end:
                    color_start = inequality_group_starts[group_schedule_offset + group]
                    color_end = inequality_group_starts[group_schedule_offset + group + int32(1)]
                    color_slot = color_start
                    color_step = int32(1)
                    if reverse_colors:
                        color_slot = color_end - int32(1)
                        color_step = int32(-1)
                    local_x_idx_0 = int32(-1)
                    local_x_idx_1 = int32(-1)
                    local_body_0 = vec6f(0.0)
                    local_body_1 = vec6f(0.0)
                    first_uid = inequality_ids_by_color[uio + color_slot]
                    if first_uid >= nl:
                        first_contact_id = contact_indices[cio + first_uid - nl]
                        if first_contact_id >= int32(0):
                            first_row = ccgo + int32(3) * (first_uid - nl)
                            first_nzb_offset = contact_nzb_offsets[first_contact_id]
                            local_x_idx_0 = col_start + bsm_nzb_coords[first_nzb_offset, 1]
                            for j in range(6):
                                local_body_0[j] = body_space[local_x_idx_0 + j]
                            second_body_offset = first_nzb_offset + int32(3)
                            if second_body_offset < matrix_end and bsm_nzb_coords[second_body_offset, 0] == first_row:
                                local_x_idx_1 = col_start + bsm_nzb_coords[second_body_offset, 1]
                                for j in range(6):
                                    local_body_1[j] = body_space[local_x_idx_1 + j]
                    while color_slot >= color_start and color_slot < color_end:
                        uid = inequality_ids_by_color[uio + color_slot]
                        # An inequality without mapped topology has no Jacobian offsets
                        # to read, so it is skipped rather than dereferenced.
                        mapped_id = int32(-1)
                        if uid < nl:
                            mapped_id = limit_indices[lio + uid]
                        else:
                            mapped_id = contact_indices[cio + uid - nl]
                        if mapped_id >= int32(0):
                            if uid < nl:
                                if phase == int32(0):
                                    limit_id = mapped_id
                                    row = lcgo + uid
                                    vec_idx = vio + row
                                    nzb_offset = limit_nzb_offsets[limit_id]
                                    limit_value = eta[row_start + row] * solution_lambdas[vec_idx]
                                    for k in range(2):
                                        nzb_idx = nzb_offset + k
                                        if nzb_idx < matrix_end and bsm_nzb_coords[nzb_idx, 0] == row:
                                            block = bsm_nzb_values[nzb_idx]
                                            x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                                            for j in range(6):
                                                limit_value += block[j] * body_space[x_idx_base + j]
                                    unilateral_row = row - njc
                                    for bilateral_row in range(njc):
                                        coupling_index = (
                                            bilateral_offset + bilateral_row * max_unilateral_rows + unilateral_row
                                        )
                                        limit_value += (
                                            bilateral_coupling[coupling_index] * bilateral_delta[bvio + bilateral_row]
                                        )
                                    limit_value += problem_v_f[vec_idx]
                                    P_i = problem_P[vec_idx]
                                    diagonal_raw = projected_diag[vec_idx]
                                    lambda_limit_old = solution_lambdas[vec_idx]
                                    lambda_limit_new = lambda_limit_old
                                    if diagonal_raw > FLOAT32_EPS:
                                        lambda_limit_new = wp.max(
                                            float32(0.0),
                                            lambda_limit_old
                                            - cfg.omega
                                            * limit_value
                                            / (diagonal_raw + cfg.regularization + FLOAT32_EPS),
                                        )
                                    lambda_limit_delta = lambda_limit_new - lambda_limit_old
                                    limit_delta_body = P_i * lambda_limit_delta
                                    solution_lambdas[vec_idx] = lambda_limit_new
                                    for bilateral_row in range(njc):
                                        response_index = (
                                            bilateral_offset + bilateral_row * max_unilateral_rows + unilateral_row
                                        )
                                        wp.atomic_sub(
                                            bilateral_delta,
                                            bvio + bilateral_row,
                                            bilateral_response[response_index] * lambda_limit_delta,
                                        )
                                    for k in range(2):
                                        nzb_idx = nzb_offset + k
                                        if nzb_idx < matrix_end and bsm_nzb_coords[nzb_idx, 0] == row:
                                            x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                                            jacobian_row = jacobian_nzb_values[nzb_idx]
                                            for j in range(6):
                                                body_space[x_idx_base + j] += jacobian_row[j] * limit_delta_body
                            else:
                                cid = uid - nl
                                row = ccgo + int32(3) * cid
                                vec_idx = vio + row
                                contact_id = mapped_id
                                nzb_offset = contact_nzb_offsets[contact_id]
                                block_count = int32(3)
                                second_body_offset = nzb_offset + int32(3)
                                if second_body_offset < matrix_end and bsm_nzb_coords[second_body_offset, 0] == row:
                                    block_count = int32(6)

                                contact_value = vec3f(0.0)
                                if phase == int32(0):
                                    contact_value.z = (
                                        eta[row_start + row + int32(2)] * solution_lambdas[vec_idx + int32(2)]
                                    )
                                    local_block = int32(2)
                                    while local_block < block_count:
                                        nzb_idx = nzb_offset + local_block
                                        block_n = bsm_nzb_values[nzb_idx]
                                        x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                                        body_values = local_body_0
                                        if x_idx_base == local_x_idx_1:
                                            body_values = local_body_1
                                        for j in range(6):
                                            contact_value.z += block_n[j] * body_values[j]
                                        local_block += int32(3)
                                else:
                                    contact_value.x = eta[row_start + row] * solution_lambdas[vec_idx]
                                    contact_value.y = (
                                        eta[row_start + row + int32(1)] * solution_lambdas[vec_idx + int32(1)]
                                    )
                                    local_block = int32(0)
                                    while local_block < block_count:
                                        nzb_idx = nzb_offset + local_block
                                        block_t0 = bsm_nzb_values[nzb_idx]
                                        block_t1 = bsm_nzb_values[nzb_idx + int32(1)]
                                        x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                                        body_values = local_body_0
                                        if x_idx_base == local_x_idx_1:
                                            body_values = local_body_1
                                        for j in range(6):
                                            contact_value.x += block_t0[j] * body_values[j]
                                            contact_value.y += block_t1[j] * body_values[j]
                                        local_block += int32(3)

                                contact_delta_body = vec3f(0.0)
                                unilateral_row = row - njc
                                if phase == int32(0):
                                    for bilateral_row in range(njc):
                                        coupling_index = (
                                            bilateral_offset
                                            + bilateral_row * max_unilateral_rows
                                            + unilateral_row
                                            + int32(2)
                                        )
                                        contact_value.z += (
                                            bilateral_coupling[coupling_index] * bilateral_delta[bvio + bilateral_row]
                                        )
                                    contact_value.z += problem_v_f[vec_idx + int32(2)]
                                    lambda_n_old = solution_lambdas[vec_idx + int32(2)]
                                    P_n = problem_P[vec_idx + int32(2)]
                                    diagonal_n = projected_diag[vec_idx + int32(2)]
                                    lambda_n_new = _project_contact_normal_update(
                                        lambda_n_old,
                                        contact_value.z,
                                        diagonal_n,
                                        cfg.regularization,
                                        cfg.omega,
                                    )
                                    lambda_n_delta = lambda_n_new - lambda_n_old
                                    solution_lambdas[vec_idx + int32(2)] = lambda_n_new
                                    contact_delta_body.z = P_n * lambda_n_delta
                                    for bilateral_row in range(njc):
                                        response_index = (
                                            bilateral_offset
                                            + bilateral_row * max_unilateral_rows
                                            + unilateral_row
                                            + int32(2)
                                        )
                                        wp.atomic_sub(
                                            bilateral_delta,
                                            bvio + bilateral_row,
                                            bilateral_response[response_index] * lambda_n_delta,
                                        )
                                else:
                                    for bilateral_row in range(njc):
                                        coupling_index_t0 = (
                                            bilateral_offset + bilateral_row * max_unilateral_rows + unilateral_row
                                        )
                                        coupling_index_t1 = coupling_index_t0 + int32(1)
                                        bilateral_value = bilateral_delta[bvio + bilateral_row]
                                        contact_value.x += bilateral_coupling[coupling_index_t0] * bilateral_value
                                        contact_value.y += bilateral_coupling[coupling_index_t1] * bilateral_value
                                    contact_value.x += problem_v_f[vec_idx]
                                    contact_value.y += problem_v_f[vec_idx + int32(1)]
                                    lambda_t0_old = solution_lambdas[vec_idx]
                                    lambda_t1_old = solution_lambdas[vec_idx + int32(1)]
                                    P_t0 = problem_P[vec_idx]
                                    P_t1 = problem_P[vec_idx + int32(1)]
                                    diagonal_t0 = projected_diag[vec_idx]
                                    diagonal_t1 = projected_diag[vec_idx + int32(1)]
                                    lambda_t_old = wp.vec2f(lambda_t0_old, lambda_t1_old)
                                    off_diagonal = inequality_tangent_cross[uio + uid]
                                    if _sweep == first_tangent_sweep:
                                        off_diagonal = float32(0.0)
                                        body_group = int32(0)
                                        while body_group < block_count:
                                            nzb_idx = nzb_offset + body_group
                                            mass_weighted_t0 = bsm_nzb_values[nzb_idx]
                                            jacobian_t1 = jacobian_nzb_values[nzb_idx + int32(1)]
                                            for j in range(6):
                                                off_diagonal += mass_weighted_t0[j] * jacobian_t1[j]
                                            body_group += int32(3)
                                        off_diagonal *= P_t1
                                        inequality_tangent_cross[uio + uid] = off_diagonal
                                    for bilateral_row in range(njc):
                                        coupling_index_t0 = (
                                            bilateral_offset + bilateral_row * max_unilateral_rows + unilateral_row
                                        )
                                        coupling_index_t1 = coupling_index_t0 + int32(1)
                                        off_diagonal -= (
                                            bilateral_coupling[coupling_index_t0]
                                            * bilateral_response[coupling_index_t1]
                                        )
                                    lambda_t_new = _project_contact_tangent_update(
                                        lambda_t_old,
                                        wp.vec2f(contact_value.x, contact_value.y),
                                        wp.vec2f(diagonal_t0, diagonal_t1),
                                        off_diagonal,
                                        cfg.regularization,
                                        cfg.omega,
                                        problem_mu[cio + cid]
                                        * _contact_friction_normal_load(
                                            solution_lambdas[vec_idx + int32(2)],
                                            problem_v_b[vec_idx + int32(2)],
                                            problem_P[vec_idx + int32(2)],
                                            wp.abs(problem_diag[vec_idx + int32(2)])
                                            * problem_P[vec_idx + int32(2)]
                                            * problem_P[vec_idx + int32(2)],
                                            cfg.regularization,
                                            cfg.omega,
                                        ),
                                    )
                                    solution_lambdas[vec_idx] = lambda_t_new.x
                                    solution_lambdas[vec_idx + int32(1)] = lambda_t_new.y
                                    lambda_t0_delta = lambda_t_new.x - lambda_t_old.x
                                    lambda_t1_delta = lambda_t_new.y - lambda_t_old.y
                                    contact_delta_body.x = P_t0 * lambda_t0_delta
                                    contact_delta_body.y = P_t1 * lambda_t1_delta
                                    for bilateral_row in range(njc):
                                        response_index_t0 = (
                                            bilateral_offset + bilateral_row * max_unilateral_rows + unilateral_row
                                        )
                                        response_index_t1 = response_index_t0 + int32(1)
                                        wp.atomic_sub(
                                            bilateral_delta,
                                            bvio + bilateral_row,
                                            bilateral_response[response_index_t0] * lambda_t0_delta
                                            + bilateral_response[response_index_t1] * lambda_t1_delta,
                                        )

                                body_group = int32(0)
                                while body_group < block_count:
                                    nzb_idx = nzb_offset + body_group
                                    x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                                    if phase == int32(0):
                                        row_n = jacobian_nzb_values[nzb_idx + int32(2)]
                                        for j in range(6):
                                            body_delta = row_n[j] * contact_delta_body.z
                                            if x_idx_base == local_x_idx_0:
                                                local_body_0[j] += body_delta
                                            else:
                                                local_body_1[j] += body_delta
                                    else:
                                        row_t0 = jacobian_nzb_values[nzb_idx]
                                        row_t1 = jacobian_nzb_values[nzb_idx + int32(1)]
                                        for j in range(6):
                                            body_delta = (
                                                row_t0[j] * contact_delta_body.x + row_t1[j] * contact_delta_body.y
                                            )
                                            if x_idx_base == local_x_idx_0:
                                                local_body_0[j] += body_delta
                                            else:
                                                local_body_1[j] += body_delta
                                    body_group += int32(3)
                        color_slot += color_step
                    if local_x_idx_0 >= int32(0):
                        for j in range(6):
                            body_space[local_x_idx_0 + j] = local_body_0[j]
                    if local_x_idx_1 >= int32(0):
                        for j in range(6):
                            body_space[local_x_idx_1 + j] = local_body_1[j]
                    group += threads_per_world
                _sync_threads()


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    float r = value;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        r += __shfl_xor_sync(0xffffffffu, r, offset, 32);
    return r;
#else
    return value;
#endif
    """
)
def _subgroup_sum_32(value: float32) -> float32: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, 0, 32);
#else
    return value;
#endif
    """
)
def _broadcast_lane_0_32(value: float32) -> float32: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    """
)
def _sync_warp_32(): ...


@wp.func
def _cooperative_unilateral_component(uid: int32, nl: int32, phase: int32) -> int32:
    component = int32(0)
    if uid >= nl and phase == int32(0):
        component = int32(2)
    return component


@wp.func
def _compact_unilateral_correction(
    compact_q: wp.array[float32], offset: int32, unilateral_row: int32, uid: int32, nl: int32, phase: int32
) -> wp.vec2f:
    component = _cooperative_unilateral_component(uid, nl, phase)
    correction = wp.vec2f(compact_q[offset + unilateral_row + component], float32(0.0))
    if uid >= nl and phase != int32(0):
        correction.y = compact_q[offset + unilateral_row + int32(1)]
    return correction


@wp.kernel
def _assemble_compact_unilateral_schur(
    problem_dim: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    response_mio: wp.array[int32],
    response_stride: wp.array[int32],
    coupling: wp.array[float32],
    response: wp.array[float32],
    compact_schur: wp.array[float32],
    compact_q: wp.array[float32],
):
    """Assemble ``C.T * response`` for worlds with no more unilateral than bilateral rows."""
    tid = wp.tid()
    threads_per_world = int32(wp.block_dim())
    lane = tid % threads_per_world
    wid = tid / threads_per_world
    njc = problem_njc[wid]
    nu = problem_dim[wid] - njc
    if nu > njc:
        return
    offset = response_mio[wid]
    stride = response_stride[wid]
    for unilateral in range(lane, nu, threads_per_world):
        compact_q[problem_vio[wid] + njc + unilateral] = float32(0.0)
    # Store the Schur matrix transposed so a warp updating consecutive target
    # rows reads consecutive values for a fixed source constraint.
    for entry in range(lane, nu * nu, threads_per_world):
        column = entry / nu
        row = entry - column * nu
        value = float32(0.0)
        for bilateral in range(njc):
            value += coupling[offset + bilateral * stride + row] * response[offset + bilateral * stride + column]
        compact_schur[offset + column * stride + row] = value


@wp.func
def _cooperative_sparse_limit_update(
    mapped_id: int32,
    row: int32,
    vec_idx: int32,
    row_start: int32,
    col_start: int32,
    matrix_end: int32,
    bilateral_value: float32,
    cfg: DVIConfigStruct,
    bsm_nzb_coords: wp.array2d[int32],
    bsm_nzb_values: wp.array[vec6f],
    jacobian_nzb_values: wp.array[vec6f],
    limit_nzb_offsets: wp.array[int32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    projected_diag: wp.array[float32],
    eta: wp.array[float32],
    body_space: wp.array[float32],
    solution_lambdas: wp.array[float32],
) -> float32:
    nzb_offset = limit_nzb_offsets[mapped_id]
    value = eta[row_start + row] * solution_lambdas[vec_idx] + bilateral_value
    for k in range(2):
        nzb_idx = nzb_offset + k
        if nzb_idx < matrix_end and bsm_nzb_coords[nzb_idx, 0] == row:
            block = bsm_nzb_values[nzb_idx]
            x_idx = col_start + bsm_nzb_coords[nzb_idx, 1]
            for j in range(6):
                value += block[j] * body_space[x_idx + j]
    value += problem_v_f[vec_idx]
    old_lambda = solution_lambdas[vec_idx]
    new_lambda = old_lambda
    diagonal = projected_diag[vec_idx]
    if diagonal > FLOAT32_EPS:
        new_lambda = wp.max(
            float32(0.0),
            old_lambda - cfg.omega * value / (diagonal + cfg.regularization + FLOAT32_EPS),
        )
    lambda_delta = new_lambda - old_lambda
    solution_lambdas[vec_idx] = new_lambda
    body_delta = problem_P[vec_idx] * lambda_delta
    for k in range(2):
        nzb_idx = nzb_offset + k
        if nzb_idx < matrix_end and bsm_nzb_coords[nzb_idx, 0] == row:
            x_idx = col_start + bsm_nzb_coords[nzb_idx, 1]
            jacobian_row = jacobian_nzb_values[nzb_idx]
            for j in range(6):
                body_space[x_idx + j] += jacobian_row[j] * body_delta
    return lambda_delta


@wp.func
def _cooperative_sparse_contact_normal_update(
    mapped_id: int32,
    row: int32,
    vec_idx: int32,
    row_start: int32,
    col_start: int32,
    matrix_end: int32,
    bilateral_value: float32,
    cfg: DVIConfigStruct,
    bsm_nzb_coords: wp.array2d[int32],
    bsm_nzb_values: wp.array[vec6f],
    jacobian_nzb_values: wp.array[vec6f],
    contact_nzb_offsets: wp.array[int32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    projected_diag: wp.array[float32],
    eta: wp.array[float32],
    body_space: wp.array[float32],
    solution_lambdas: wp.array[float32],
) -> float32:
    nzb_offset = contact_nzb_offsets[mapped_id]
    block_count = int32(3)
    if nzb_offset + int32(3) < matrix_end and bsm_nzb_coords[nzb_offset + int32(3), 0] == row:
        block_count = int32(6)
    value = eta[row_start + row + int32(2)] * solution_lambdas[vec_idx + int32(2)] + bilateral_value
    block_offset = int32(2)
    while block_offset < block_count:
        nzb_idx = nzb_offset + block_offset
        block = bsm_nzb_values[nzb_idx]
        x_idx = col_start + bsm_nzb_coords[nzb_idx, 1]
        for j in range(6):
            value += block[j] * body_space[x_idx + j]
        block_offset += int32(3)
    value += problem_v_f[vec_idx + int32(2)]
    old_lambda = solution_lambdas[vec_idx + int32(2)]
    new_lambda = _project_contact_normal_update(
        old_lambda, value, projected_diag[vec_idx + int32(2)], cfg.regularization, cfg.omega
    )
    lambda_delta = new_lambda - old_lambda
    solution_lambdas[vec_idx + int32(2)] = new_lambda
    body_delta = problem_P[vec_idx + int32(2)] * lambda_delta
    block_offset = int32(2)
    while block_offset < block_count:
        nzb_idx = nzb_offset + block_offset
        x_idx = col_start + bsm_nzb_coords[nzb_idx, 1]
        jacobian_row = jacobian_nzb_values[nzb_idx]
        for j in range(6):
            body_space[x_idx + j] += jacobian_row[j] * body_delta
        block_offset += int32(3)
    return lambda_delta


@wp.func
def _cooperative_sparse_contact_tangent_update(
    mapped_id: int32,
    cid: int32,
    uid: int32,
    row: int32,
    vec_idx: int32,
    row_start: int32,
    col_start: int32,
    matrix_end: int32,
    value_correction_0: float32,
    value_correction_1: float32,
    projected_cross: float32,
    first_tangent_sweep: bool,
    uio: int32,
    cio: int32,
    cfg: DVIConfigStruct,
    bsm_nzb_coords: wp.array2d[int32],
    bsm_nzb_values: wp.array[vec6f],
    jacobian_nzb_values: wp.array[vec6f],
    contact_nzb_offsets: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    problem_v_b: wp.array[float32],
    problem_diag: wp.array[float32],
    projected_diag: wp.array[float32],
    eta: wp.array[float32],
    inequality_tangent_cross: wp.array[float32],
    body_space: wp.array[float32],
    solution_lambdas: wp.array[float32],
) -> vec3f:
    nzb_offset = contact_nzb_offsets[mapped_id]
    block_count = int32(3)
    if nzb_offset + int32(3) < matrix_end and bsm_nzb_coords[nzb_offset + int32(3), 0] == row:
        block_count = int32(6)
    value_0 = eta[row_start + row] * solution_lambdas[vec_idx] + value_correction_0
    value_1 = eta[row_start + row + int32(1)] * solution_lambdas[vec_idx + int32(1)] + value_correction_1
    block_offset = int32(0)
    while block_offset < block_count:
        nzb_idx = nzb_offset + block_offset
        block_0 = bsm_nzb_values[nzb_idx]
        block_1 = bsm_nzb_values[nzb_idx + int32(1)]
        x_idx = col_start + bsm_nzb_coords[nzb_idx, 1]
        for j in range(6):
            body_value = body_space[x_idx + j]
            value_0 += block_0[j] * body_value
            value_1 += block_1[j] * body_value
        block_offset += int32(3)
    value_0 += problem_v_f[vec_idx]
    value_1 += problem_v_f[vec_idx + int32(1)]
    old_lambda = wp.vec2f(solution_lambdas[vec_idx], solution_lambdas[vec_idx + int32(1)])
    off_diagonal = inequality_tangent_cross[uio + uid]
    if first_tangent_sweep:
        off_diagonal = float32(0.0)
        block_offset = int32(0)
        while block_offset < block_count:
            nzb_idx = nzb_offset + block_offset
            mass_weighted_t0 = bsm_nzb_values[nzb_idx]
            jacobian_t1 = jacobian_nzb_values[nzb_idx + int32(1)]
            for j in range(6):
                off_diagonal += mass_weighted_t0[j] * jacobian_t1[j]
            block_offset += int32(3)
        off_diagonal *= problem_P[vec_idx + int32(1)]
        inequality_tangent_cross[uio + uid] = off_diagonal
    off_diagonal -= projected_cross
    new_lambda = _project_contact_tangent_update(
        old_lambda,
        wp.vec2f(value_0, value_1),
        wp.vec2f(projected_diag[vec_idx], projected_diag[vec_idx + int32(1)]),
        off_diagonal,
        cfg.regularization,
        cfg.omega,
        problem_mu[cio + cid]
        * _contact_friction_normal_load(
            solution_lambdas[vec_idx + int32(2)],
            problem_v_b[vec_idx + int32(2)],
            problem_P[vec_idx + int32(2)],
            wp.abs(problem_diag[vec_idx + int32(2)]) * problem_P[vec_idx + int32(2)] * problem_P[vec_idx + int32(2)],
            cfg.regularization,
            cfg.omega,
        ),
    )
    delta_0 = new_lambda.x - old_lambda.x
    delta_1 = new_lambda.y - old_lambda.y
    solution_lambdas[vec_idx] = new_lambda.x
    solution_lambdas[vec_idx + int32(1)] = new_lambda.y
    body_delta_0 = problem_P[vec_idx] * delta_0
    body_delta_1 = problem_P[vec_idx + int32(1)] * delta_1
    block_offset = int32(0)
    while block_offset < block_count:
        nzb_idx = nzb_offset + block_offset
        x_idx = col_start + bsm_nzb_coords[nzb_idx, 1]
        jacobian_0 = jacobian_nzb_values[nzb_idx]
        jacobian_1 = jacobian_nzb_values[nzb_idx + int32(1)]
        for j in range(6):
            body_space[x_idx + j] += jacobian_0[j] * body_delta_0 + jacobian_1[j] * body_delta_1
        block_offset += int32(3)
    return vec3f(delta_0, delta_1, 0.0)


@wp.kernel
def _solve_dvi_sparse_inequalities_pgs_cooperative(
    bsm_num_nzb: wp.array[int32],
    bsm_nzb_start: wp.array[int32],
    bsm_nzb_coords: wp.array2d[int32],
    bsm_nzb_values: wp.array[vec6f],
    jacobian_nzb_values: wp.array[vec6f],
    bsm_row_start: wp.array[int32],
    bsm_col_start: wp.array[int32],
    limit_nzb_offsets: wp.array[int32],
    contact_nzb_offsets: wp.array[int32],
    limit_indices: wp.array[int32],
    contact_indices: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lio: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_uio: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    problem_v_b: wp.array[float32],
    problem_diag: wp.array[float32],
    projected_diag: wp.array[float32],
    eta: wp.array[float32],
    problem_njc: wp.array[int32],
    bilateral_vio: wp.array[int32],
    response_mio: wp.array[int32],
    response_stride: wp.array[int32],
    bilateral_coupling: wp.array[float32],
    bilateral_response: wp.array[float32],
    bilateral_delta: wp.array[float32],
    compact_schur: wp.array[float32],
    compact_q: wp.array[float32],
    enable_compact_schur: wp.bool,
    inequality_num_colors: wp.array[int32],
    inequality_ids_by_color: wp.array[int32],
    inequality_color_starts: wp.array[int32],
    inequality_group_starts: wp.array[int32],
    inequality_tangent_cross: wp.array[float32],
    block_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    solver_status: wp.array[DVIStatus],
    body_space: wp.array[float32],
    solution_lambdas: wp.array[float32],
):
    """Apply sparse PGS with one warp cooperating on each articulated world."""
    tid = wp.tid()
    lane = tid % int32(32)
    wid = tid / int32(32)
    cfg = solver_config[wid]
    if block_iteration >= int32(0) and block_iteration >= cfg.max_alternating_iterations:
        return
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    if nl + nc == int32(0):
        if lane == int32(0) and block_iteration == int32(_FUSED_BILATERAL_BLOCK):
            status = solver_status[wid]
            status.iterations = int32(1)
            solver_status[wid] = status
        return
    lio = problem_lio[wid]
    cio = problem_cio[wid]
    uio = problem_uio[wid]
    schedule_offset = uio + wid
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    vio = problem_vio[wid]
    njc = problem_njc[wid]
    response_offset = response_mio[wid]
    response_row_stride = response_stride[wid]
    num_unilateral_rows = nl + int32(3) * nc
    use_compact_schur = enable_compact_schur and num_unilateral_rows <= njc
    bvio = bilateral_vio[wid]
    row_start = bsm_row_start[wid]
    col_start = bsm_col_start[wid]
    matrix_end = bsm_nzb_start[wid] + bsm_num_nzb[wid]
    sweep_count = cfg.inequality_sweeps_per_iteration
    first_tangent_sweep = int32(0)
    if block_iteration == int32(_FUSED_INEQUALITY_BLOCK):
        tangent_sweep_count = sweep_count * cfg.max_alternating_iterations / int32(2)
        sweep_count = (sweep_count + int32(1)) * cfg.max_alternating_iterations
        first_tangent_sweep = sweep_count - tangent_sweep_count
    elif block_iteration == int32(_FUSED_BILATERAL_BLOCK):
        sweep_count *= cfg.max_alternating_iterations

    for sweep in range(sweep_count):
        phase_count = int32(2)
        if block_iteration == int32(_FUSED_INEQUALITY_BLOCK) and sweep < first_tangent_sweep:
            phase_count = int32(1)
        for phase in range(phase_count):
            schedule_sweep = sweep
            if block_iteration == int32(_FUSED_BILATERAL_BLOCK):
                schedule_sweep = sweep % cfg.inequality_sweeps_per_iteration
            reverse_colors = phase == int32(1) and schedule_sweep % int32(2) != int32(0)
            num_colors = inequality_num_colors[wid]
            for color_index in range(num_colors):
                color = color_index
                if reverse_colors:
                    color = num_colors - int32(1) - color_index
                group_start = inequality_color_starts[schedule_offset + color]
                group_end = inequality_color_starts[schedule_offset + color + int32(1)]
                for group in range(group_start, group_end):
                    slot_start = inequality_group_starts[schedule_offset + group]
                    slot_end = inequality_group_starts[schedule_offset + group + int32(1)]
                    for slot_iteration in range(slot_end - slot_start):
                        slot = slot_start + slot_iteration
                        if reverse_colors:
                            slot = slot_end - int32(1) - slot_iteration
                        uid = inequality_ids_by_color[uio + slot]
                        mapped_id = int32(-1)
                        if uid < nl and phase != int32(0):
                            continue
                        if uid < nl:
                            mapped_id = limit_indices[lio + uid]
                        else:
                            mapped_id = contact_indices[cio + uid - nl]

                        row = lcgo + uid
                        if uid >= nl:
                            row = ccgo + int32(3) * (uid - nl)
                        vec_idx = vio + row
                        unilateral_row = row - njc
                        correction_0 = float32(0.0)
                        correction_1 = float32(0.0)
                        projected_cross = float32(0.0)
                        if mapped_id >= int32(0):
                            if use_compact_schur:
                                if lane == int32(0):
                                    correction = _compact_unilateral_correction(
                                        compact_q, vio + njc, unilateral_row, uid, nl, phase
                                    )
                                    correction_0 = correction.x
                                    correction_1 = correction.y
                                    if uid >= nl and phase != int32(0):
                                        projected_cross = compact_schur[
                                            response_offset
                                            + (unilateral_row + int32(1)) * response_row_stride
                                            + unilateral_row
                                        ]
                            else:
                                partial_0 = float32(0.0)
                                partial_1 = float32(0.0)
                                partial_cross = float32(0.0)
                                for bilateral_row in range(lane, njc, int32(32)):
                                    index = response_offset + bilateral_row * response_row_stride + unilateral_row
                                    bilateral_value = bilateral_delta[bvio + bilateral_row]
                                    if uid < nl or phase == int32(0):
                                        component = int32(0)
                                        if uid >= nl:
                                            component = int32(2)
                                        partial_0 += bilateral_coupling[index + component] * bilateral_value
                                    else:
                                        partial_0 += bilateral_coupling[index] * bilateral_value
                                        partial_1 += bilateral_coupling[index + int32(1)] * bilateral_value
                                        partial_cross += (
                                            bilateral_coupling[index] * bilateral_response[index + int32(1)]
                                        )
                                correction_0 = _subgroup_sum_32(partial_0)
                                correction_1 = _subgroup_sum_32(partial_1)
                                projected_cross = _subgroup_sum_32(partial_cross)

                        delta_0 = float32(0.0)
                        delta_1 = float32(0.0)
                        if lane == int32(0):
                            if mapped_id >= int32(0):
                                if uid < nl:
                                    if phase == int32(0):
                                        delta_0 = _cooperative_sparse_limit_update(
                                            mapped_id,
                                            row,
                                            vec_idx,
                                            row_start,
                                            col_start,
                                            matrix_end,
                                            correction_0,
                                            cfg,
                                            bsm_nzb_coords,
                                            bsm_nzb_values,
                                            jacobian_nzb_values,
                                            limit_nzb_offsets,
                                            problem_P,
                                            problem_v_f,
                                            projected_diag,
                                            eta,
                                            body_space,
                                            solution_lambdas,
                                        )
                                elif phase == int32(0):
                                    delta_0 = _cooperative_sparse_contact_normal_update(
                                        mapped_id,
                                        row,
                                        vec_idx,
                                        row_start,
                                        col_start,
                                        matrix_end,
                                        correction_0,
                                        cfg,
                                        bsm_nzb_coords,
                                        bsm_nzb_values,
                                        jacobian_nzb_values,
                                        contact_nzb_offsets,
                                        problem_P,
                                        problem_v_f,
                                        projected_diag,
                                        eta,
                                        body_space,
                                        solution_lambdas,
                                    )
                                else:
                                    tangent_delta = _cooperative_sparse_contact_tangent_update(
                                        mapped_id,
                                        uid - nl,
                                        uid,
                                        row,
                                        vec_idx,
                                        row_start,
                                        col_start,
                                        matrix_end,
                                        correction_0,
                                        correction_1,
                                        projected_cross,
                                        sweep == first_tangent_sweep,
                                        uio,
                                        cio,
                                        cfg,
                                        bsm_nzb_coords,
                                        bsm_nzb_values,
                                        jacobian_nzb_values,
                                        contact_nzb_offsets,
                                        problem_mu,
                                        problem_P,
                                        problem_v_f,
                                        problem_v_b,
                                        problem_diag,
                                        projected_diag,
                                        eta,
                                        inequality_tangent_cross,
                                        body_space,
                                        solution_lambdas,
                                    )
                                    delta_0 = tangent_delta.x
                                    delta_1 = tangent_delta.y
                        delta_0 = _broadcast_lane_0_32(delta_0)
                        delta_1 = _broadcast_lane_0_32(delta_1)
                        if mapped_id >= int32(0) and (delta_0 != float32(0.0) or delta_1 != float32(0.0)):
                            component = _cooperative_unilateral_component(uid, nl, phase)
                            if use_compact_schur:
                                for target in range(lane, num_unilateral_rows, int32(32)):
                                    value = (
                                        compact_schur[
                                            response_offset
                                            + (unilateral_row + component) * response_row_stride
                                            + target
                                        ]
                                        * delta_0
                                    )
                                    if uid >= nl and phase != int32(0):
                                        value += (
                                            compact_schur[
                                                response_offset
                                                + (unilateral_row + int32(1)) * response_row_stride
                                                + target
                                            ]
                                            * delta_1
                                        )
                                    compact_q[vio + njc + target] -= value
                            else:
                                for bilateral_row in range(lane, njc, int32(32)):
                                    index = response_offset + bilateral_row * response_row_stride + unilateral_row
                                    if uid < nl:
                                        if phase == int32(0):
                                            bilateral_delta[bvio + bilateral_row] -= bilateral_response[index] * delta_0
                                    elif phase == int32(0):
                                        bilateral_delta[bvio + bilateral_row] -= (
                                            bilateral_response[index + int32(2)] * delta_0
                                        )
                                    else:
                                        bilateral_delta[bvio + bilateral_row] -= (
                                            bilateral_response[index] * delta_0
                                            + bilateral_response[index + int32(1)] * delta_1
                                        )
                        _sync_warp_32()

    if lane == int32(0) and block_iteration == int32(_FUSED_BILATERAL_BLOCK):
        status = solver_status[wid]
        status.iterations = cfg.max_alternating_iterations * cfg.inequality_sweeps_per_iteration
        solver_status[wid] = status


@wp.kernel
def _build_sparse_bilateral_block(
    # Inputs:
    model_bodies_inv_m_i: wp.array[float32],
    data_bodies_inv_I_i: wp.array[mat33f],
    pair_wid: wp.array[int32],
    pair_row: wp.array[int32],
    pair_col: wp.array[int32],
    pair_bid: wp.array[int32],
    pair_i: wp.array[int32],
    pair_j: wp.array[int32],
    jacobian_cts_nzb_values: wp.array[vec6f],
    problem_njc: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    # Output:
    bilateral_D: wp.array[float32],
):
    pair_id = wp.tid()
    wid = pair_wid[pair_id]
    njc = problem_njc[wid]
    row = pair_row[pair_id]
    col = pair_col[pair_id]
    block_i = jacobian_cts_nzb_values[pair_i[pair_id]]
    block_j = jacobian_cts_nzb_values[pair_j[pair_id]]
    Jv_i = vec3f(block_i[0], block_i[1], block_i[2])
    Jv_j = vec3f(block_j[0], block_j[1], block_j[2])
    Jw_i = vec3f(block_i[3], block_i[4], block_i[5])
    Jw_j = vec3f(block_j[3], block_j[4], block_j[5])

    bid_k = pair_bid[pair_id]
    inv_m_k = model_bodies_inv_m_i[bid_k]
    inv_I_k = data_bodies_inv_I_i[bid_k]
    D_ij = inv_m_k * wp.dot(Jv_i, Jv_j) + wp.dot(Jw_i, inv_I_k @ Jw_j)

    bvio = bilateral_vio[wid]
    p_row = bilateral_P[bvio + row]
    p_col = bilateral_P[bvio + col]
    val = p_row * D_ij * p_col

    bmio = bilateral_mio[wid]
    wp.atomic_add(bilateral_D, bmio + njc * row + col, val)
    wp.atomic_add(bilateral_D, bmio + njc * col + row, val)


@wp.kernel
def _set_sparse_bilateral_diagonal(
    # Inputs:
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    problem_diag: wp.array[float32],
    # Outputs:
    bilateral_D: wp.array[float32],
    bilateral_P: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if njc == 0:
        if row == 0:
            bilateral_D[bilateral_mio[wid]] = float32(1.0)
            bilateral_P[bilateral_vio[wid]] = float32(1.0)
        return
    if row >= njc:
        return

    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]
    bmio = bilateral_mio[wid]
    diag = wp.abs(problem_diag[pvio + row])
    p = wp.sqrt(1.0 / (diag + FLOAT32_EPS))
    bilateral_P[bvio + row] = p
    bilateral_D[bmio + njc * row + row] = p * diag * p + float32(7.0e-7)


@wp.kernel
def _compute_dvi_sparse_solution_vectors(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_v_f: wp.array[float32],
    # Outputs:
    state_s: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    v_i = problem_vio[wid] + tid
    v_plus = state_v_aug[v_i] + problem_v_f[v_i]
    solution_v_plus[v_i] = v_plus
    state_v_aug[v_i] = v_plus
    state_s[v_i] = 0.0
