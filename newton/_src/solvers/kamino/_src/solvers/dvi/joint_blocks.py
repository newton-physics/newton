# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact joint-block Cholesky for homogeneous bilateral systems.

Each joint contributes at most six bilateral rows. Batched solves preserve
zero-fill ordering; persistent small-batch solves also allow bounded symbolic
fill. Identity dummy rows complete short blocks.
Response tiles omit those dummy rows, preserving the compact Schur layout.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import vec6f
from ...linalg.factorize.llt_blocked_rcm import _sync_threads, get_float32_array_offset_ptr
from .coupling import PackedCoupling, read_packed_coupling
from .kernels import BILATERAL_DIAGONAL_FLOOR, _compact_schur_fits
from .sparse_kernels import _shuffle_lane_32, _unilateral_nzb_offsets

if TYPE_CHECKING:
    from ...dynamics.dual import DualProblem
    from .sparse import SparseDVIPath

wp.set_module_options({"enable_backward": False, "enable_mathdx_solver": False, "enable_mathdx_gemm": False})


# Bound fill storage on the small-batch path; larger graphs retain RCM.
_PERSISTENT_MAX_BLOCKS = 4096


def _build_topology(
    path: SparseDVIPath, pair_world: wp.array, pair_row: wp.array, pair_col: wp.array, *, allow_fill: bool = False
):
    """Build a static elimination schedule, optionally allowing bounded fill."""
    joints = path.model.joints
    world = joints.wid.numpy()
    dynamic_counts = joints.num_dynamic_cts.numpy()
    kinematic_counts = joints.num_kinematic_cts.numpy()
    dynamic_offsets = joints.dynamic_cts_offset_total_cts.numpy()
    kinematic_offsets = joints.kinematic_cts_offset_total_cts.numpy()
    origin = path.model.info.total_cts_offset.numpy()
    dimensions = path.data.bilateral_dim.numpy()
    worlds = path.size.num_worlds
    groups_by_world = [[] for _ in range(worlds)]
    for joint, wid in enumerate(world):
        rows = list(
            range(
                int(dynamic_offsets[joint] - origin[wid]),
                int(dynamic_offsets[joint] - origin[wid] + dynamic_counts[joint]),
            )
        )
        rows += list(
            range(
                int(kinematic_offsets[joint] - origin[wid]),
                int(kinematic_offsets[joint] - origin[wid] + kinematic_counts[joint]),
            )
        )
        if rows:
            if len(rows) > 6:
                return None, "joint group exceeds six bilateral rows"
            groups_by_world[wid].append(tuple(rows))
    groups = groups_by_world[0]
    n = int(dimensions[0])
    if n == 0 or any(g != groups for g in groups_by_world) or not np.all(dimensions == n):
        return None, "empty or heterogeneous joint-row topology"
    if sorted(row for group in groups for row in group) != list(range(n)):
        return None, "joint groups do not partition the bilateral rows exactly"
    group_of = np.empty(n, dtype=np.int32)
    for i, group in enumerate(groups):
        group_of[list(group)] = i
    pw, pr, pc = pair_world.numpy(), pair_row.numpy(), pair_col.numpy()
    if np.any(pr < 0) or np.any(pc < 0) or np.any(pr >= n) or np.any(pc >= n):
        return None, "pair row outside static bilateral dimensions"
    ng = len(groups)
    lo = np.minimum(group_of[pr], group_of[pc])
    hi = np.maximum(group_of[pr], group_of[pc])
    # Unique block edges per world verify all replicated worlds share the graph.
    edges = np.unique(pw.astype(np.int64) * ng * ng + lo.astype(np.int64) * ng + hi)
    counts = np.bincount(edges // (ng * ng), minlength=worlds)
    if not np.all(counts == counts[0]):
        return None, "heterogeneous block graphs"
    edge_sets = edges.reshape(worlds, int(counts[0])) % (ng * ng)
    if not np.all(edge_sets == edge_sets[0]):
        return None, "heterogeneous block graphs"
    graph = [set() for _ in groups]
    for encoded in edge_sets[0]:
        a, b = divmod(int(encoded), ng)
        if a != b:
            graph[a].add(b)
            graph[b].add(a)
    stored_blocks = ng + sum(len(adjacent) for adjacent in graph) // 2
    if allow_fill and stored_blocks > _PERSISTENT_MAX_BLOCKS:
        return None, "persistent block storage budget exceeded"
    remaining = set(range(ng))
    order = []
    while remaining:
        choices = []
        for node in remaining:
            adjacent = graph[node] & remaining
            if all(b in graph[a] for a in adjacent for b in adjacent if a != b):
                choices.append((len(adjacent), node))
        if not choices:
            if not allow_fill:
                return None, "nonchordal graph: zero-fill block elimination unavailable"
            node = min(remaining, key=lambda node: (len(graph[node] & remaining), node))
            adjacent = graph[node] & remaining
            added = sum(len(adjacent - graph[a] - {a}) for a in adjacent) // 2
            if stored_blocks + added > _PERSISTENT_MAX_BLOCKS:
                return None, "persistent block storage budget exceeded"
            stored_blocks += added
            # Completing the surviving neighborhood gives exact symbolic
            # Cholesky fill; the corresponding original matrix blocks stay zero.
            for a in adjacent:
                graph[a].update(adjacent - {a})
            choices.append((len(adjacent), node))
        node = min(choices)[1]
        order.append(node)
        remaining.remove(node)
    positions = {node: i for i, node in enumerate(order)}
    pairs = [(i, i) for i in range(ng)]
    for node in order:
        for other in graph[node]:
            if positions[other] > positions[node]:
                pairs.append((positions[other], positions[node]))
    pairs.sort(key=lambda pair: (pair[1], pair[0]))
    index = {pair: i for i, pair in enumerate(pairs)}
    starts, left, right = [0], [], []
    for i, j in pairs:
        for k in range(j):
            if (i, k) in index and (j, k) in index:
                left.append(index[i, k])
                right.append(index[j, k])
        starts.append(len(left))
    previous_start, previous, following_start, following = [0], [], [0], []
    for i in range(ng):
        previous.extend(index[i, k] for k in range(i) if (i, k) in index)
        following.extend(index[k, i] for k in range(i + 1, ng) if (k, i) in index)
        previous_start.append(len(previous))
        following_start.append(len(following))
    lengths = [len(groups[node]) for node in order]
    scalar_order = [row for node in order for row in (*groups[node], *([-1] * (6 - len(groups[node]))))]
    row_offsets = np.cumsum([0, *lengths]).tolist()
    rows, cols = zip(*pairs, strict=False)
    return {
        "n": n,
        "ng": ng,
        "padded_n": ng * 6,
        "nb": len(pairs),
        "order": scalar_order,
        "rows": rows,
        "cols": cols,
        "diagonal": [index[i, i] for i in range(ng)],
        "starts": starts,
        "left": left,
        "right": right,
        "pstart": previous_start,
        "previous": previous,
        "fstart": following_start,
        "following": following,
        "lengths": lengths,
        "row_offsets": row_offsets,
    }, "eligible homogeneous joint-block graph"


def _build_body_masks(path: SparseDVIPath, metadata: dict) -> tuple[wp.array, wp.array] | None:
    """Find coupling seeds and forward-solve support shared by local body labels."""
    groups = metadata["ng"]
    if groups > 64:
        return None
    worlds, bodies = path.size.num_worlds, path.size.max_of_num_bodies
    order = np.asarray(metadata["order"], dtype=np.int32)
    group_of = np.empty(metadata["n"], dtype=np.int32)
    group_of[order[order >= 0]] = np.flatnonzero(order >= 0) // 6
    jacobian = path.jacobians._J_cts.bsm
    counts = path.jacobians.joint_constraint_nzb_count.numpy().astype(np.int64)
    starts = jacobian.nzb_start.numpy().astype(np.int64)
    world = np.repeat(np.arange(worlds, dtype=np.int64), counts)
    origins = np.cumsum(counts) - counts
    indices = np.repeat(starts - origins, counts) + np.arange(counts.sum(), dtype=np.int64)
    coords = jacobian.nzb_coords.numpy()[indices]
    active = (coords[:, 0] >= 0) & (coords[:, 0] < metadata["n"])
    world, coords = world[active], coords[active]
    if np.any(coords[:, 1] % 6) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] // 6 >= bodies):
        return None
    encoded = np.unique(
        world * (bodies * groups) + (coords[:, 1] // 6).astype(np.int64) * groups + group_of[coords[:, 0]]
    )
    per_world = np.bincount(encoded // (bodies * groups), minlength=worlds)
    if not np.all(per_world == per_world[0]):
        return None
    incidence = encoded.reshape(worlds, int(per_world[0])) % (bodies * groups)
    if not np.all(incidence == incidence[0]):
        return None
    masks = [0] * bodies
    for code in incidence[0]:
        body, group = divmod(int(code), groups)
        masks[body] |= 1 << group
    seeds = wp.array(np.asarray(masks, dtype=np.uint64), dtype=wp.uint64, device=path.device)
    # Columns are already in elimination order: propagate each body's seed
    # groups through every potentially nonzero lower-factor block.
    for body in range(bodies):
        for row, col in zip(metadata["rows"], metadata["cols"], strict=True):
            if masks[body] & (1 << int(col)):
                masks[body] |= 1 << int(row)
    return seeds, wp.array(np.asarray(masks, dtype=np.uint64), dtype=wp.uint64, device=path.device)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
unsigned long long bits = (unsigned long long)value;
for (int delta = 16; delta > 0; delta >>= 1) {
    bits |= __shfl_xor_sync(0xffffffff, bits, delta);
}
return bits;
#else
return value;
#endif
""")
def _warp_or_mask(value: wp.uint64) -> wp.uint64: ...


@wp.kernel
def _factor(
    blocks: wp.array[wp.float32],
    factors: wp.array[wp.float32],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    diagonal: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    lefts: wp.array[wp.int32],
    rights: wp.array[wp.int32],
    nb: int,
):
    w, _lane = wp.tid()
    aa = wp.array(ptr=get_float32_array_offset_ptr(blocks, w * nb * 36), shape=(nb * 6, 6), dtype=wp.float32)
    ll = wp.array(ptr=get_float32_array_offset_ptr(factors, w * nb * 36), shape=(nb * 6, 6), dtype=wp.float32)
    for b in range(nb):
        value = wp.tile_load(aa, shape=(6, 6), offset=(b * 6, 0), storage="shared")
        for update in range(starts[b], starts[b + 1]):
            left = wp.tile_load(ll, shape=(6, 6), offset=(lefts[update] * 6, 0))
            right = wp.tile_load(ll, shape=(6, 6), offset=(rights[update] * 6, 0))
            wp.tile_matmul(left, wp.tile_transpose(right), value, alpha=-1.0)
        if rows[b] == cols[b]:
            wp.tile_cholesky_inplace(value, fill_mode="upper")
            wp.tile_store(ll, wp.tile_transpose(value), offset=(b * 6, 0))
        else:
            d = wp.tile_load(ll, shape=(6, 6), offset=(diagonal[cols[b]] * 6, 0))
            transposed = wp.tile_transpose(value)
            wp.tile_lower_solve_inplace(d, transposed)
            wp.tile_store(ll, wp.tile_transpose(transposed), offset=(b * 6, 0))


@wp.kernel
def _factor_persistent(
    matrix: wp.array[wp.float32],
    factors: wp.array[wp.float32],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    diagonal: wp.array[wp.int32],
    updates: wp.array[wp.int32],
    lefts: wp.array[wp.int32],
    rights: wp.array[wp.int32],
    nb: int,
    tasks: wp.array[wp.int32],
    levels: wp.array[wp.int32],
    level_count: int,
):
    """Factor independent blocks with eight warps and a barrier per level."""
    world, thread = wp.tid()
    lane = thread % 32
    warp = thread // 32
    origin = world * nb * 36
    for level in range(level_count):
        for task in range(levels[level] + warp, levels[level + 1], 8):
            block = tasks[task]
            value = vec6f(0.0)
            result = vec6f(0.0)
            # Distribute all 36 entries across the warp during block updates.
            entry0 = matrix[origin + block * 36 + lane]
            entry1 = wp.float32(0.0)
            if lane < 4:
                entry1 = matrix[origin + block * 36 + lane + 32]
            for update in range(updates[block], updates[block + 1]):
                left = lefts[update]
                right = rights[update]
                accum0 = wp.float32(0.0)
                for k in range(6):
                    accum0 += (
                        factors[origin + left * 36 + (lane // 6) * 6 + k]
                        * factors[origin + right * 36 + (lane % 6) * 6 + k]
                    )
                entry0 -= accum0
                if lane < 4:
                    accum1 = wp.float32(0.0)
                    for k in range(6):
                        accum1 += (
                            factors[origin + left * 36 + ((lane + 32) // 6) * 6 + k]
                            * factors[origin + right * 36 + ((lane + 32) % 6) * 6 + k]
                        )
                    entry1 -= accum1
            # Gather rows for the six-lane triangular solve. Both shuffles
            # require participation from every lane, including unused rows.
            for col in range(6):
                source = wp.min(lane, 5) * 6 + col
                first = _shuffle_lane_32(entry0, source % 32)
                tail = _shuffle_lane_32(entry1, source % 32)
                value[col] = wp.where(source < 32, first, tail)
            if rows[block] == cols[block]:
                for pivot in range(6):
                    accum = wp.float32(0.0)
                    for k in range(pivot):
                        component = _shuffle_lane_32(result[k], pivot)
                        accum += result[k] * component
                    numerator = value[pivot] - accum
                    root = wp.float32(0.0)
                    if lane == pivot:
                        root = wp.sqrt(numerator)
                    pivot_diagonal = _shuffle_lane_32(root, pivot)
                    if lane == pivot:
                        result[pivot] = pivot_diagonal
                    elif lane > pivot and lane < 6:
                        result[pivot] = numerator / pivot_diagonal
            elif lane < 6:
                d = diagonal[cols[block]]
                for pivot in range(6):
                    numerator = value[pivot]
                    for k in range(pivot):
                        numerator -= factors[origin + d * 36 + pivot * 6 + k] * result[k]
                    result[pivot] = numerator / factors[origin + d * 36 + pivot * 6 + pivot]
            if lane < 6:
                for col in range(6):
                    factors[origin + block * 36 + lane * 6 + col] = result[col]
        _sync_threads()


@wp.kernel
def _solve_persistent(
    factors: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
    active: wp.array[wp.int32],
    diagonal: wp.array[wp.int32],
    pstart: wp.array[wp.int32],
    previous: wp.array[wp.int32],
    fstart: wp.array[wp.int32],
    following: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    nb: int,
    padded: int,
    forward_tasks: wp.array[wp.int32],
    forward_levels: wp.array[wp.int32],
    forward_count: int,
    backward_tasks: wp.array[wp.int32],
    backward_levels: wp.array[wp.int32],
    backward_count: int,
):
    """Solve independent group rows with one persistent block per world."""
    world, thread = wp.tid()
    if active[world] == 0:
        return
    lane = thread % 32
    warp = thread // 32
    origin = world * nb * 36
    vector = world * padded
    for level in range(forward_count):
        for task in range(forward_levels[level] + warp, forward_levels[level + 1], 8):
            row = forward_tasks[task]
            value = wp.float32(0.0)
            if lane < 6:
                value = scratch[vector + row * 6 + lane]
                for entry in range(pstart[row], pstart[row + 1]):
                    block = previous[entry]
                    accum = wp.float32(0.0)
                    for k in range(6):
                        accum += factors[origin + block * 36 + lane * 6 + k] * scratch[vector + cols[block] * 6 + k]
                    value -= accum
            d = diagonal[row]
            for pivot in range(6):
                solved = wp.float32(0.0)
                if lane == pivot:
                    solved = value / factors[origin + d * 36 + pivot * 6 + pivot]
                solved = _shuffle_lane_32(solved, pivot)
                if lane == pivot:
                    value = solved
                elif lane > pivot and lane < 6:
                    value -= factors[origin + d * 36 + lane * 6 + pivot] * solved
            if lane < 6:
                scratch[vector + row * 6 + lane] = value
        _sync_threads()
    for level in range(backward_count):
        for task in range(backward_levels[level] + warp, backward_levels[level + 1], 8):
            row = backward_tasks[task]
            value = wp.float32(0.0)
            if lane < 6:
                value = scratch[vector + row * 6 + lane]
                for entry in range(fstart[row], fstart[row + 1]):
                    block = following[entry]
                    accum = wp.float32(0.0)
                    for k in range(6):
                        accum += factors[origin + block * 36 + k * 6 + lane] * scratch[vector + rows[block] * 6 + k]
                    value -= accum
            d = diagonal[row]
            for pivot in range(5, -1, -1):
                solved = wp.float32(0.0)
                if lane == pivot:
                    solved = value / factors[origin + d * 36 + pivot * 6 + pivot]
                solved = _shuffle_lane_32(solved, pivot)
                if lane == pivot:
                    value = solved
                elif lane < pivot:
                    value -= factors[origin + d * 36 + pivot * 6 + lane] * solved
            if lane < 6:
                scratch[vector + row * 6 + lane] = value
        _sync_threads()


@wp.kernel
def _initialize_dummy_rows(
    order: wp.array[wp.int32], diagonal: wp.array[wp.int32], matrix: wp.array[wp.float32], num_blocks: int
):
    world, row = wp.tid()
    if order[row] < 0:
        local = row % 6
        matrix[world * num_blocks * 36 + diagonal[row // 6] * 36 + local * 7] = 1.0


@wp.kernel
def _assemble_diagonal(
    njc: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    bilateral_vio: wp.array[wp.int32],
    problem_diag: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    row_diagonal: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    num_blocks: int,
):
    world, row = wp.tid()
    if row >= njc[world]:
        return
    diag = wp.abs(problem_diag[problem_vio[world] + row])
    p = wp.sqrt(1.0 / (diag + FLOAT32_EPS))
    scale[bilateral_vio[world] + row] = p
    matrix[world * num_blocks * 36 + row_diagonal[row]] = p * diag * p + wp.float32(BILATERAL_DIAGONAL_FLOOR)


@cache
def _make_assemble_entries(shared: bool):
    """Specialize assembly for shared local or original global indices."""

    @wp.kernel(
        module="unique",
        module_options={"enable_backward": False, "enable_mathdx_solver": False, "enable_mathdx_gemm": False},
    )
    def assemble_entries(
        inv_mass: wp.array[wp.float32],
        inv_inertia: wp.array[wp.mat33f],
        entry_starts: wp.array[wp.int32],
        pair_world: wp.array[wp.int32],
        body_offsets: wp.array[wp.int32],
        nzb_offsets: wp.array[wp.int32],
        pair_row: wp.array[wp.int32],
        pair_col: wp.array[wp.int32],
        pair_body: wp.array[wp.int32],
        pair_i: wp.array[wp.int32],
        pair_j: wp.array[wp.int32],
        jacobian: wp.array[vec6f],
        bilateral_vio: wp.array[wp.int32],
        scale: wp.array[wp.float32],
        entry_target: wp.array[wp.int32],
        block_rows: wp.array[wp.int32],
        block_cols: wp.array[wp.int32],
        num_blocks: int,
        matrix: wp.array[wp.float32],
    ):
        world = int(0)
        entry = int(0)
        if wp.static(shared):
            world, entry = wp.tid()
        else:
            entry = wp.tid()
        first = entry_starts[entry]
        body_offset = int(0)
        nzb_offset = int(0)
        if wp.static(shared):
            body_offset = body_offsets[world]
            nzb_offset = nzb_offsets[world]
        else:
            world = pair_world[first]
        value = wp.float32(0.0)
        # Keep body accumulation and normalization in the dense assembly's order.
        for pair in range(first, entry_starts[entry + 1]):
            block_i = jacobian[nzb_offset + pair_i[pair]]
            block_j = jacobian[nzb_offset + pair_j[pair]]
            Jv_i = wp.vec3f(block_i[0], block_i[1], block_i[2])
            Jv_j = wp.vec3f(block_j[0], block_j[1], block_j[2])
            Jw_i = wp.vec3f(block_i[3], block_i[4], block_i[5])
            Jw_j = wp.vec3f(block_j[3], block_j[4], block_j[5])
            body = body_offset + pair_body[pair]
            value += inv_mass[body] * wp.dot(Jv_i, Jv_j) + wp.dot(Jw_i, inv_inertia[body] @ Jw_j)
        bvio = bilateral_vio[world]
        value = scale[bvio + pair_row[first]] * value * scale[bvio + pair_col[first]]
        target = entry_target[entry]
        if wp.static(shared):
            target += world * num_blocks * 36
        matrix[target] = value
        local = target % (num_blocks * 36)
        block = local // 36
        if block_rows[block] == block_cols[block]:
            # Diagonal tiles need both triangles; off-diagonal tiles have one owner.
            cell = local % 36
            matrix[target - cell + (cell % 6) * 6 + cell // 6] = value

    return assemble_entries


@wp.kernel
def _check_factor(
    factors: wp.array[wp.float32], diagonal: wp.array[wp.int32], failure: wp.array[wp.int32], nb: int, ng: int
):
    w = wp.tid()
    failed = int(0)
    for e in range(nb * 36):
        if not wp.isfinite(factors[w * nb * 36 + e]):
            failed = 1
    for row in range(ng):
        for i in range(6):
            if factors[w * nb * 36 + diagonal[row] * 36 + i * 6 + i] <= 0.0:
                failed = 1
    wp.atomic_max(failure, 0, failed)


@wp.kernel
def _check_factor_parallel(
    factors: wp.array[wp.float32],
    diagonal: wp.array[wp.int32],
    failure: wp.array[wp.int32],
    nb: int,
    ng: int,
):
    w, lane = wp.tid()
    failed = int(0)
    for entry in range(lane, nb * 36, 256):
        if not wp.isfinite(factors[w * nb * 36 + entry]):
            failed = 1
    for entry in range(lane, ng * 6, 256):
        row, col = entry // 6, entry % 6
        if factors[w * nb * 36 + diagonal[row] * 36 + col * 7] <= 0.0:
            failed = 1
    if failed:
        wp.atomic_max(failure, 0, failed)


@wp.kernel
def _check_dimensions(
    dim: wp.array[wp.int32], njc: wp.array[wp.int32], stride: wp.array[wp.int32], n: int, failure: wp.array[wp.int32]
):
    w = wp.tid()
    nu = dim[w] - njc[w]
    if njc[w] != n or nu < 0 or nu > n or not _compact_schur_fits(njc[w], nu, stride[w]):
        wp.atomic_max(failure, 0, 1)


@wp.kernel
def _pack_rhs(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    order: wp.array[wp.int32],
    rhs: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
    padded: int,
):
    w, row = wp.tid()
    if dim[w] == 0:
        return
    value = wp.float32(0)
    if order[row] >= 0:
        value = rhs[vio[w] + order[row]]
    scratch[w * padded + row] = value


@wp.kernel
def _solve(
    factors: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
    active: wp.array[wp.int32],
    diagonal: wp.array[wp.int32],
    previous_start: wp.array[wp.int32],
    previous: wp.array[wp.int32],
    following_start: wp.array[wp.int32],
    following: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    nb: int,
    ng: int,
    padded: int,
):
    w, _lane = wp.tid()
    if active[w] == 0:
        return
    ll = wp.array(ptr=get_float32_array_offset_ptr(factors, w * nb * 36), shape=(nb * 6, 6), dtype=wp.float32)
    yy = wp.array(ptr=get_float32_array_offset_ptr(scratch, w * padded), shape=(padded, 1), dtype=wp.float32)
    for row in range(ng):
        value = wp.tile_load(yy, shape=(6, 1), offset=(row * 6, 0), storage="shared")
        for entry in range(previous_start[row], previous_start[row + 1]):
            block = previous[entry]
            left = wp.tile_load(ll, shape=(6, 6), offset=(block * 6, 0))
            right = wp.tile_load(yy, shape=(6, 1), offset=(cols[block] * 6, 0))
            wp.tile_matmul(left, right, value, alpha=-1.0)
        d = wp.tile_load(ll, shape=(6, 6), offset=(diagonal[row] * 6, 0))
        wp.tile_lower_solve_inplace(d, value)
        wp.tile_store(yy, value, offset=(row * 6, 0))
    for row in range(ng - 1, -1, -1):
        value = wp.tile_load(yy, shape=(6, 1), offset=(row * 6, 0), storage="shared")
        for entry in range(following_start[row], following_start[row + 1]):
            block = following[entry]
            left = wp.tile_load(ll, shape=(6, 6), offset=(block * 6, 0))
            right = wp.tile_load(yy, shape=(6, 1), offset=(rows[block] * 6, 0))
            wp.tile_matmul(wp.tile_transpose(left), right, value, alpha=-1.0)
        d = wp.tile_load(ll, shape=(6, 6), offset=(diagonal[row] * 6, 0))
        wp.tile_upper_solve_inplace(wp.tile_transpose(d), value)
        wp.tile_store(yy, value, offset=(row * 6, 0))


@wp.kernel
def _scatter(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    order: wp.array[wp.int32],
    scratch: wp.array[wp.float32],
    out: wp.array[wp.float32],
    padded: int,
):
    w, row = wp.tid()
    if dim[w] != 0 and order[row] >= 0:
        out[vio[w] + order[row]] = scratch[w * padded + row]


@cache
def make_response_kernel(use_packed: bool):
    """Specialize coefficient loads while sharing the block-whitening algorithm."""

    @wp.kernel(module="unique")
    def response(
        packed_masks: wp.array[wp.uint64],
        packed_values: wp.array[wp.float32],
        packed_width: int,
        packed_slots: int,
        factors: wp.array[wp.float32],
        diagonal: wp.array[wp.int32],
        starts: wp.array[wp.int32],
        previous: wp.array[wp.int32],
        cols: wp.array[wp.int32],
        nb: int,
        ng: int,
        dim: wp.array[wp.int32],
        njc: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        rio: wp.array[wp.int32],
        order: wp.array[wp.int32],
        scale: wp.array[wp.float32],
        coupling: wp.array[wp.float32],
        response: wp.array[wp.float32],
        lengths: wp.array[wp.int32],
        offsets: wp.array[wp.int32],
        body_reach: wp.array[wp.uint64],
        bsm_num_nzb: wp.array[wp.int32],
        bsm_nzb_start: wp.array[wp.int32],
        bsm_nzb_coords: wp.array2d[wp.int32],
        nbc: wp.array[wp.int32],
        nl: wp.array[wp.int32],
        bcio: wp.array[wp.int32],
        lio: wp.array[wp.int32],
        cio: wp.array[wp.int32],
        limit_indices: wp.array[wp.int32],
        contact_indices: wp.array[wp.int32],
        bounded_offsets: wp.array[wp.vec2i],
        limit_offsets: wp.array[wp.int32],
        contact_offsets: wp.array[wp.int32],
    ):
        w, group, lane = wp.tid()
        nu = dim[w] - njc[w]
        if group * 16 >= nu:
            return
        ll = wp.array(ptr=get_float32_array_offset_ptr(factors, w * nb * 36), shape=(nb * 6, 6), dtype=wp.float32)
        reach = wp.uint64(0)
        if body_reach:
            unilateral = group * 16 + lane % 16
            if unilateral < nu:
                blocks = _unilateral_nzb_offsets(
                    unilateral,
                    njc[w],
                    nbc[w],
                    nbc[w] + nl[w],
                    bcio[w],
                    lio[w],
                    cio[w],
                    bsm_nzb_start[w] + bsm_num_nzb[w],
                    bsm_nzb_coords,
                    limit_indices,
                    contact_indices,
                    bounded_offsets,
                    limit_offsets,
                    contact_offsets,
                )
                for side in range(2):
                    if blocks[side] >= 0:
                        reach = reach | body_reach[bsm_nzb_coords[blocks[side], 1] // 6]
            reach = _warp_or_mask(reach)
        for row in range(ng):
            value = wp.tile_zeros(shape=(6, 16), dtype=wp.float32, storage="shared")
            if body_reach and (reach & (wp.uint64(1) << wp.uint64(row))) == wp.uint64(0):
                # Contacts can move to another branch: overwrite formerly active
                # response rows instead of retaining values from the previous solve.
                target = wp.array(
                    ptr=get_float32_array_offset_ptr(response, rio[w] + offsets[row] * nu),
                    shape=(lengths[row], nu),
                    dtype=wp.float32,
                )
                wp.tile_store(target, value, offset=(0, group * 16))
                continue
            for entry in range(lane, 96, 32):
                local = entry // 16
                col = group * 16 + entry % 16
                original = order[row * 6 + local]
                v = wp.float32(0)
                if original >= 0 and col < nu:
                    if wp.static(use_packed):
                        v = scale[vio[w] + original] * read_packed_coupling(
                            packed_values, packed_masks, w, col, row, local, packed_width, packed_slots
                        )
                    else:
                        v = scale[vio[w] + original] * coupling[rio[w] + original * nu + col]
                value[local, entry % 16] = v
            _sync_threads()
            for entry in range(starts[row], starts[row + 1]):
                block = previous[entry]
                left = wp.tile_load(ll, shape=(6, 6), offset=(block * 6, 0))
                previous_row = cols[block]
                source = wp.array(
                    ptr=get_float32_array_offset_ptr(response, rio[w] + offsets[previous_row] * nu),
                    shape=(lengths[previous_row], nu),
                    dtype=wp.float32,
                )
                right = wp.tile_load(source, shape=(6, 16), offset=(0, group * 16))
                wp.tile_matmul(left, right, value, alpha=-1.0)
            d = wp.tile_load(ll, shape=(6, 6), offset=(diagonal[row] * 6, 0))
            wp.tile_lower_solve_inplace(d, value)
            target = wp.array(
                ptr=get_float32_array_offset_ptr(response, rio[w] + offsets[row] * nu),
                shape=(lengths[row], nu),
                dtype=wp.float32,
            )
            wp.tile_store(target, value, offset=(0, group * 16))

    return response


def _build_levels(metadata: dict) -> dict:
    """Group independent factor blocks and triangular rows into barrier levels."""
    factor = np.zeros(metadata["nb"], dtype=np.int32)
    for block, (row, col) in enumerate(zip(metadata["rows"], metadata["cols"], strict=True)):
        start, end = metadata["starts"][block : block + 2]
        dependencies = metadata["left"][start:end] + metadata["right"][start:end]
        if row != col:
            dependencies = [*dependencies, metadata["diagonal"][col]]
        factor[block] = 1 + max((int(factor[index]) for index in dependencies), default=0)
    forward = np.zeros(metadata["ng"], dtype=np.int32)
    backward = np.zeros_like(forward)
    for row in range(metadata["ng"]):
        entries = metadata["previous"][metadata["pstart"][row] : metadata["pstart"][row + 1]]
        forward[row] = 1 + max((int(forward[metadata["cols"][entry]]) for entry in entries), default=0)
    for row in range(metadata["ng"] - 1, -1, -1):
        entries = metadata["following"][metadata["fstart"][row] : metadata["fstart"][row + 1]]
        backward[row] = 1 + max((int(backward[metadata["rows"][entry]]) for entry in entries), default=0)
    levels = {}
    for name, values in (("factor", factor), ("forward", forward), ("backward", backward)):
        tasks = np.argsort(values, kind="stable").astype(np.int32)
        offsets = np.cumsum(np.r_[0, np.bincount(values)[1:]], dtype=np.int32)
        levels[name] = (tasks, offsets)
    return levels


def _build_assembly_template(path: SparseDVIPath, targets: np.ndarray, num_blocks: int) -> tuple | None:
    """Share metadata only when every world's complete ordered contribution list agrees."""
    if not targets.size:
        return None
    starts = path.bilateral_entry_starts.numpy()
    pairs = [array.numpy() for array in path.bilateral_nzb_pairs]
    owners = pairs[0][starts[:-1]]
    worlds = path.size.num_worlds
    if np.any(np.diff(owners) < 0) or owners[0] < 0 or owners[-1] >= worlds:
        return None
    bounds = np.searchsorted(owners, np.arange(worlds + 1))
    body_offsets = path.model.info.bodies_offset.numpy()
    # Jacobian regions use fixed capacities; dynamic contacts change num_nzb,
    # not these starts or the joint contributions at the front of each region.
    nzb_offsets = path.jacobians._J_cts.bsm.nzb_start.numpy()
    reference = None
    for world in range(worlds):
        entry_begin, entry_end = bounds[world : world + 2]
        if entry_begin == entry_end:
            return None
        begin, end = starts[entry_begin], starts[entry_end]
        boundaries = starts[entry_begin : entry_end + 1] - begin
        if np.any(np.diff(boundaries) <= 0) or not np.all(pairs[0][begin:end] == world):
            return None
        contributions = np.column_stack(
            [
                pairs[1][begin:end],
                pairs[2][begin:end],
                pairs[3][begin:end] - body_offsets[world],
                pairs[4][begin:end] - nzb_offsets[world],
                pairs[5][begin:end] - nzb_offsets[world],
            ]
        )
        candidate = (boundaries, contributions, targets[entry_begin:entry_end] - world * num_blocks * 36)
        if reference is None:
            reference = candidate
        elif any(not np.array_equal(a, b) for a, b in zip(reference, candidate, strict=True)):
            return None
    boundaries, contributions, local_targets = reference
    arrays = (boundaries, *contributions.T, local_targets)
    return (len(local_targets), *(wp.array(a, dtype=wp.int32, device=path.device) for a in arrays))


class JointBlockSolver:
    """Own a fixed block schedule and scratch without modifying the RCM solver.

    ``failure`` is a device scalar. After ``prepare``, the caller must select
    the original complete solve whenever it is nonzero, before changing impulses.
    """

    @classmethod
    def create(cls, path: SparseDVIPath, *, persistent: bool = False) -> JointBlockSolver | None:
        """Allocate a fixed schedule, allowing bounded fill for persistent solves."""
        if path.bilateral_nzb_pairs is None or not path.device.is_cuda:
            return None
        metadata, _ = _build_topology(path, *path.bilateral_nzb_pairs[:3], allow_fill=persistent)
        if metadata is None:
            return None
        return cls(path, metadata, persistent=persistent)

    def __init__(self, path: SparseDVIPath, metadata: dict, *, persistent: bool = False):
        self._persistent = persistent
        self._device = path.device
        self._info = path.data.bilateral_operator.info
        self._worlds = path.size.num_worlds
        self._n = metadata["n"]
        self._groups = metadata["ng"]
        self._padded = metadata["padded_n"]
        self._blocks = metadata["nb"]
        self._schedule = {
            name: wp.array(np.asarray(value, dtype=np.int32), dtype=wp.int32, device=self._device)
            for name, value in metadata.items()
            if name not in ("n", "ng", "padded_n", "nb")
        }
        self._levels = {}
        if persistent:
            for name, (tasks, offsets) in _build_levels(metadata).items():
                self._levels[name] = (
                    wp.array(tasks, dtype=wp.int32, device=self._device),
                    wp.array(offsets, dtype=wp.int32, device=self._device),
                    len(offsets) - 1,
                )
        self._matrix = wp.zeros(self._worlds * self._blocks * 36, dtype=wp.float32, device=self._device)
        self._factor = wp.empty_like(self._matrix)
        self._vector = wp.empty(self._worlds * self._padded, dtype=wp.float32, device=self._device)
        self.failure = wp.zeros(1, dtype=wp.int32, device=self._device)
        masks = _build_body_masks(path, metadata)
        self._body_seeds, self._body_reach = (None, None) if masks is None else masks

        order = np.asarray(metadata["order"], dtype=np.int32)
        inverse = np.empty(self._n, dtype=np.int32)
        inverse[order[order >= 0]] = np.flatnonzero(order >= 0)
        diagonal = np.asarray(metadata["diagonal"], dtype=np.int32)
        self._row_diagonal = wp.array(
            diagonal[inverse // 6] * 36 + (inverse % 6) * 7, dtype=wp.int32, device=self._device
        )
        lookup = np.full((self._groups, self._groups), -1, dtype=np.int32)
        lookup[np.asarray(metadata["rows"]), np.asarray(metadata["cols"])] = np.arange(self._blocks, dtype=np.int32)
        first = path.bilateral_entry_starts.numpy()[:-1]
        worlds = path.bilateral_nzb_pairs[0].numpy()[first]
        i = inverse[path.bilateral_nzb_pairs[1].numpy()[first]]
        j = inverse[path.bilateral_nzb_pairs[2].numpy()[first]]
        swap = i // 6 < j // 6
        rows, cols = np.where(swap, j, i), np.where(swap, i, j)
        blocks = lookup[rows // 6, cols // 6]
        offsets = worlds * (self._blocks * 36) + blocks * 36 + (rows % 6) * 6 + cols % 6
        self._assembly_template = None
        if not persistent and self._worlds > 1:
            self._assembly_template = _build_assembly_template(path, offsets, self._blocks)
        self._entry_target = (
            self._assembly_template[-1]
            if self._assembly_template is not None
            else wp.array(offsets, dtype=wp.int32, device=self._device)
        )
        # Numeric factorization has separate storage, so structural zeros and
        # dummy identities need initialization only when the topology is built.
        wp.launch(
            _initialize_dummy_rows,
            dim=(self._worlds, self._padded),
            inputs=[self._schedule["order"], self._schedule["diagonal"], self._matrix, self._blocks],
            device=self._device,
        )

    def assemble(self, path: SparseDVIPath, problem: DualProblem) -> None:
        """Assemble normalized joint blocks directly from their body contributions."""
        state = path.data.state
        state.bilateral_preconditioner.zero_()
        problem.delassus.diagonal(state.scratch)
        wp.launch(
            _assemble_diagonal,
            dim=(self._worlds, self._n),
            inputs=[
                problem.data.njc,
                problem.data.vio,
                self._info.vio,
                state.scratch,
                state.bilateral_preconditioner,
                self._row_diagonal,
                self._matrix,
                self._blocks,
            ],
            device=self._device,
        )
        if self._entry_target.size:
            template = self._assembly_template
            shared = template is not None
            if shared:
                count, starts, *pairs, targets = template
                pair_world = None
                body_offsets = path.model.info.bodies_offset
                nzb_offsets = path.jacobians._J_cts.bsm.nzb_start
                dim = (self._worlds, count)
            else:
                starts = path.bilateral_entry_starts
                pair_world, *pairs = path.bilateral_nzb_pairs
                targets = self._entry_target
                body_offsets = nzb_offsets = None
                dim = self._entry_target.size
            wp.launch(
                _make_assemble_entries(shared),
                dim=dim,
                inputs=[
                    path.model.bodies.inv_m_i,
                    path.model_data.bodies.inv_I_i,
                    starts,
                    pair_world,
                    body_offsets,
                    nzb_offsets,
                    *pairs,
                    problem.delassus.constraint_jacobian.nzb_values,
                    self._info.vio,
                    state.bilateral_preconditioner,
                    targets,
                    self._schedule["rows"],
                    self._schedule["cols"],
                    self._blocks,
                    self._matrix,
                ],
                device=self._device,
            )

    def prepare(self, path: SparseDVIPath, problem: DualProblem) -> None:
        """Factor the already assembled, normalized joint blocks.

        Preserve its diagonal and regularization exactly. Unsupported dynamic
        dimensions or failed factors request the caller's original solve.
        """
        self.failure.zero_()
        schedule = self._schedule
        wp.launch(
            _check_dimensions,
            dim=self._worlds,
            inputs=[
                problem.data.dim,
                problem.data.njc,
                path.data.state.bilateral_response_stride,
                self._n,
                self.failure,
            ],
            device=self._device,
        )
        if self._persistent:
            wp.launch(
                _factor_persistent,
                dim=(self._worlds, 256),
                inputs=[
                    self._matrix,
                    self._factor,
                    schedule["rows"],
                    schedule["cols"],
                    schedule["diagonal"],
                    schedule["starts"],
                    schedule["left"],
                    schedule["right"],
                    self._blocks,
                    *self._levels["factor"],
                ],
                block_dim=256,
                device=self._device,
            )
        else:
            wp.launch_tiled(
                _factor,
                dim=self._worlds,
                inputs=[
                    self._matrix,
                    self._factor,
                    schedule["rows"],
                    schedule["cols"],
                    schedule["diagonal"],
                    schedule["starts"],
                    schedule["left"],
                    schedule["right"],
                    self._blocks,
                ],
                block_dim=32,
                device=self._device,
            )
        wp.launch(
            _check_factor_parallel if self._persistent else _check_factor,
            dim=(self._worlds, 256) if self._persistent else self._worlds,
            inputs=[self._factor, schedule["diagonal"], self.failure, self._blocks, self._groups],
            block_dim=256,
            device=self._device,
        )

    def solve(
        self,
        rhs: wp.array[wp.float32],
        x: wp.array[wp.float32],
        active_dim: wp.array[wp.int32] | None = None,
    ) -> None:
        """Solve in original row order; inactive worlds retain their outputs."""
        info, schedule = self._info, self._schedule
        dimensions = info.dim if active_dim is None else active_dim
        wp.launch(
            _pack_rhs,
            dim=(self._worlds, self._padded),
            inputs=[dimensions, info.vio, schedule["order"], rhs, self._vector, self._padded],
            device=self._device,
        )
        if self._persistent:
            wp.launch(
                _solve_persistent,
                dim=(self._worlds, 256),
                inputs=[
                    self._factor,
                    self._vector,
                    dimensions,
                    schedule["diagonal"],
                    schedule["pstart"],
                    schedule["previous"],
                    schedule["fstart"],
                    schedule["following"],
                    schedule["rows"],
                    schedule["cols"],
                    self._blocks,
                    self._padded,
                    *self._levels["forward"],
                    *self._levels["backward"],
                ],
                block_dim=256,
                device=self._device,
            )
        else:
            wp.launch_tiled(
                _solve,
                dim=self._worlds,
                inputs=[
                    self._factor,
                    self._vector,
                    dimensions,
                    schedule["diagonal"],
                    schedule["pstart"],
                    schedule["previous"],
                    schedule["fstart"],
                    schedule["following"],
                    schedule["rows"],
                    schedule["cols"],
                    self._blocks,
                    self._groups,
                    self._padded,
                ],
                block_dim=32,
                device=self._device,
            )
        wp.launch(
            _scatter,
            dim=(self._worlds, self._padded),
            inputs=[dimensions, info.vio, schedule["order"], self._vector, x, self._padded],
            device=self._device,
        )

    def response(self, path: SparseDVIPath, problem: DualProblem, *, coupling: PackedCoupling | None = None) -> None:
        """Whiten into compact response storage, omitting identity dummy rows."""
        state, schedule = path.data.state, self._schedule
        reach_inputs = [None] * 14
        if self._body_reach is not None:
            bsm = problem.delassus.bsm
            reach_inputs = [
                self._body_reach,
                bsm.num_nzb,
                bsm.nzb_start,
                bsm.nzb_coords,
                problem.data.nbc,
                problem.data.nl,
                problem.data.bcio,
                problem.data.lio,
                problem.data.cio,
                state.limit_indices,
                state.contact_indices,
                path.jacobians.bounded_constraint_nzb_offsets,
                path.jacobians.limit_constraint_nzb_offsets,
                path.jacobians.contact_constraint_nzb_offsets,
            ]
        wp.launch_tiled(
            make_response_kernel(coupling is not None),
            dim=(self._worlds, (self._n + 15) // 16),
            inputs=[
                coupling.masks if coupling is not None else None,
                coupling.values if coupling is not None else None,
                coupling.width if coupling is not None else 0,
                coupling.slots if coupling is not None else 0,
                self._factor,
                schedule["diagonal"],
                schedule["pstart"],
                schedule["previous"],
                schedule["cols"],
                self._blocks,
                self._groups,
                problem.data.dim,
                problem.data.njc,
                self._info.vio,
                state.bilateral_response_mio,
                schedule["order"],
                state.bilateral_preconditioner,
                state.bilateral_coupling,
                state.bilateral_response,
                schedule["lengths"],
                schedule["row_offsets"],
                *reach_inputs,
            ],
            block_dim=32,
            device=self._device,
        )
