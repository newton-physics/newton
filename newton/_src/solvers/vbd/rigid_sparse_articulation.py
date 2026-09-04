# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-side sparse articulation layout for the VBD rigid solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class RigidArticulationSparseLayout:
    """Static block-sparse articulation layout for rigid VBD solves."""

    articulation_body_offsets: wp.array[wp.int32]
    articulation_joint_offsets: wp.array[wp.int32]
    articulation_bodies: wp.array[wp.int32]
    articulation_joints: wp.array[wp.int32]
    articulation_joint_body_start: wp.array[wp.int32]
    articulation_block_row_offsets: wp.array[wp.int32]
    articulation_block_cols: wp.array[wp.int32]
    articulation_block_col_offsets: wp.array[wp.int32]
    articulation_block_col_rows: wp.array[wp.int32]
    articulation_block_col_slots: wp.array[wp.int32]
    articulation_factor_update_offsets: wp.array[wp.int32]
    articulation_factor_update_dst_slots: wp.array[wp.int32]
    articulation_factor_update_left_slots: wp.array[wp.int32]
    articulation_factor_update_right_slots: wp.array[wp.int32]
    articulation_diag_slots: wp.array[wp.int32]
    body_articulation_sparse: wp.array[wp.int32]
    body_articulation_local: wp.array[wp.int32]
    articulation_count: int
    articulation_body_count: int
    articulation_joint_count: int
    block_count: int


def _symbolic_cholesky_pattern(body_count: int, edges: set[tuple[int, int]]) -> list[list[int]]:
    rows = [{i} for i in range(body_count)]
    for a, b in edges:
        row = max(a, b)
        col = min(a, b)
        rows[row].add(col)

    for k in range(body_count):
        later = [i for i in range(k + 1, body_count) if k in rows[i]]
        for i_index, i in enumerate(later):
            for j in later[:i_index]:
                rows[max(i, j)].add(min(i, j))

    return [sorted(row) for row in rows]


def _minimum_degree_order(body_count: int, edges: set[tuple[int, int]]) -> list[int]:
    adjacency = [set() for _ in range(body_count)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    remaining = set(range(body_count))
    order: list[int] = []
    for _ in range(body_count):
        pivot = min(remaining, key=lambda i: (len(adjacency[i] & remaining), i))
        neighbors = sorted((adjacency[pivot] & remaining) - {pivot})
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                adjacency[a].add(b)
                adjacency[b].add(a)
        remaining.remove(pivot)
        order.append(pivot)

    return order


def build_rigid_articulation_sparse_layout(
    model, device: wp.context.Devicelike
) -> RigidArticulationSparseLayout | None:
    """Build the static sparse layout from rigid-body joint connectivity.

    This host analysis runs once from :class:`SolverVBD` construction, never
    from the per-step or per-iteration path. Repeated articulation topologies,
    such as replicated RL environments, share cached ordering and fill analysis.

    Newton articulation ranges contain only tree joints. Loop-closing joints,
    world-root joints, and orphan joints may legally live outside those ranges,
    so solver groups are connected components over all model joints instead.
    """

    if model.body_count == 0:
        return None

    joint_parent = np.asarray(model.joint_parent.to("cpu").numpy(), dtype=np.int32)
    joint_child = np.asarray(model.joint_child.to("cpu").numpy(), dtype=np.int32)

    articulation_bodies_host: list[int] = []
    articulation_joints_host: list[int] = []
    articulation_joint_body_start_host: list[int] = []
    articulation_body_offsets_host = [0]
    articulation_joint_offsets_host = [0]
    block_row_offsets_host = [0]
    block_cols_host: list[int] = []
    block_col_rows_host: list[int] = []
    block_col_slots_host: list[int] = []
    block_col_offsets_host: list[int] = []
    factor_update_dst_slots_host: list[int] = []
    factor_update_left_slots_host: list[int] = []
    factor_update_right_slots_host: list[int] = []
    factor_update_offsets_host: list[int] = []
    diag_slots_host: list[int] = []

    body_articulation_sparse_host = np.full((model.body_count,), -1, dtype=np.int32)
    body_articulation_local_host = np.full((model.body_count,), -1, dtype=np.int32)
    component_parent = np.arange(model.body_count, dtype=np.int32)

    def find(body: int) -> int:
        root = body
        while int(component_parent[root]) != root:
            root = int(component_parent[root])
        while body != root:
            next_body = int(component_parent[body])
            component_parent[body] = root
            body = next_body
        return root

    def union(body_a: int, body_b: int):
        root_a = find(body_a)
        root_b = find(body_b)
        if root_a != root_b:
            component_parent[max(root_a, root_b)] = min(root_a, root_b)

    for joint_idx in range(model.joint_count):
        parent = int(joint_parent[joint_idx])
        child = int(joint_child[joint_idx])
        if parent >= model.body_count or child >= model.body_count:
            raise ValueError(
                f"Joint {joint_idx} references an out-of-range body: parent={parent}, child={child}, "
                f"body_count={model.body_count}."
            )
        if parent >= 0 and child >= 0:
            union(parent, child)
        elif parent < 0 and child < 0:
            raise ValueError(f"Joint {joint_idx} has no rigid-body endpoint.")

    component_bodies: dict[int, list[int]] = {}
    for body in range(model.body_count):
        component_bodies.setdefault(find(body), []).append(body)

    component_joints: dict[int, list[int]] = {root: [] for root in component_bodies}
    for joint_idx in range(model.joint_count):
        parent = int(joint_parent[joint_idx])
        child = int(joint_child[joint_idx])
        endpoint = child if child >= 0 else parent
        root = find(endpoint)
        if parent >= 0 and find(parent) != root:
            raise ValueError(f"Joint {joint_idx} parent and child resolve to different sparse solver groups.")
        component_joints[root].append(joint_idx)

    articulation_groups = [
        (bodies, component_joints[root]) for root, bodies in sorted(component_bodies.items(), key=lambda item: item[0])
    ]

    symbolic_cache: dict[
        tuple[int, tuple[tuple[int, int], ...]],
        tuple[list[int], list[list[int]]],
    ] = {}
    for articulation_sparse, (bodies, joints) in enumerate(articulation_groups):
        local_index = {body: i for i, body in enumerate(bodies)}
        edges: set[tuple[int, int]] = set()
        for joint_idx in joints:
            parent = int(joint_parent[joint_idx])
            child = int(joint_child[joint_idx])
            if parent >= 0 and child >= 0 and parent in local_index and child in local_index:
                edges.add((local_index[parent], local_index[child]))

        topology_key = (
            len(bodies),
            tuple(sorted((min(a, b), max(a, b)) for a, b in edges)),
        )
        symbolic = symbolic_cache.get(topology_key)
        if symbolic is None:
            order = _minimum_degree_order(len(bodies), edges) if len(bodies) > 1 else [0]
            old_to_new = {old: new for new, old in enumerate(order)}
            ordered_edges = {(old_to_new[a], old_to_new[b]) for a, b in edges}
            pattern = _symbolic_cholesky_pattern(len(bodies), ordered_edges)
            symbolic = (order, pattern)
            symbolic_cache[topology_key] = symbolic
        order, pattern = symbolic
        ordered_bodies = [bodies[old] for old in order]

        body_start = len(articulation_bodies_host)
        articulation_bodies_host.extend(ordered_bodies)
        articulation_joints_host.extend(joints)
        articulation_joint_body_start_host.extend([body_start] * len(joints))
        articulation_body_offsets_host.append(len(articulation_bodies_host))
        articulation_joint_offsets_host.append(len(articulation_joints_host))

        for local_body, body in enumerate(ordered_bodies):
            if body_articulation_sparse_host[body] >= 0:
                raise ValueError(f"Body {body} belongs to more than one sparse solver group.")
            body_articulation_sparse_host[body] = articulation_sparse
            body_articulation_local_host[body] = local_body

        for local_row, row_cols in enumerate(pattern):
            row_start = len(block_cols_host)
            block_cols_host.extend(row_cols)
            diag_slots_host.append(row_start + row_cols.index(local_row))
            block_row_offsets_host.append(len(block_cols_host))

        block_lookup: dict[tuple[int, int], int] = {}
        local_row_offset_start = len(block_row_offsets_host) - len(pattern) - 1
        for local_row, row_cols in enumerate(pattern):
            row_start = block_row_offsets_host[local_row_offset_start + local_row]
            for local_col_index, local_col in enumerate(row_cols):
                block_lookup[(local_row, local_col)] = row_start + local_col_index

        for local_col in range(len(ordered_bodies)):
            block_col_offsets_host.append(len(block_col_rows_host))
            later_rows: list[tuple[int, int]] = []
            for local_row in range(local_col + 1, len(ordered_bodies)):
                slot = block_lookup.get((local_row, local_col))
                if slot is not None:
                    later_rows.append((local_row, slot))
                    block_col_rows_host.append(local_row)
                    block_col_slots_host.append(slot)

            factor_update_offsets_host.append(len(factor_update_dst_slots_host))
            for left_index, (left_row, left_slot) in enumerate(later_rows):
                for right_row, right_slot in later_rows[: left_index + 1]:
                    dst_slot = block_lookup[(left_row, right_row)]
                    factor_update_dst_slots_host.append(dst_slot)
                    factor_update_left_slots_host.append(left_slot)
                    factor_update_right_slots_host.append(right_slot)

    articulation_bodies_np = np.asarray(articulation_bodies_host, dtype=np.int32)
    articulation_joints_np = np.asarray(articulation_joints_host, dtype=np.int32)
    articulation_joint_body_start_np = np.asarray(articulation_joint_body_start_host, dtype=np.int32)
    articulation_body_offsets_np = np.asarray(articulation_body_offsets_host, dtype=np.int32)
    articulation_joint_offsets_np = np.asarray(articulation_joint_offsets_host, dtype=np.int32)
    block_row_offsets_np = np.asarray(block_row_offsets_host, dtype=np.int32)
    block_cols_np = np.asarray(block_cols_host, dtype=np.int32)
    block_col_offsets_host.append(len(block_col_rows_host))
    factor_update_offsets_host.append(len(factor_update_dst_slots_host))
    block_col_offsets_np = np.asarray(block_col_offsets_host, dtype=np.int32)
    block_col_rows_np = np.asarray(block_col_rows_host, dtype=np.int32)
    block_col_slots_np = np.asarray(block_col_slots_host, dtype=np.int32)
    factor_update_offsets_np = np.asarray(factor_update_offsets_host, dtype=np.int32)
    factor_update_dst_slots_np = np.asarray(factor_update_dst_slots_host, dtype=np.int32)
    factor_update_left_slots_np = np.asarray(factor_update_left_slots_host, dtype=np.int32)
    factor_update_right_slots_np = np.asarray(factor_update_right_slots_host, dtype=np.int32)
    diag_slots_np = np.asarray(diag_slots_host, dtype=np.int32)

    return RigidArticulationSparseLayout(
        articulation_body_offsets=wp.array(articulation_body_offsets_np, dtype=wp.int32, device=device),
        articulation_joint_offsets=wp.array(articulation_joint_offsets_np, dtype=wp.int32, device=device),
        articulation_bodies=wp.array(articulation_bodies_np, dtype=wp.int32, device=device),
        articulation_joints=wp.array(articulation_joints_np, dtype=wp.int32, device=device),
        articulation_joint_body_start=wp.array(articulation_joint_body_start_np, dtype=wp.int32, device=device),
        articulation_block_row_offsets=wp.array(block_row_offsets_np, dtype=wp.int32, device=device),
        articulation_block_cols=wp.array(block_cols_np, dtype=wp.int32, device=device),
        articulation_block_col_offsets=wp.array(block_col_offsets_np, dtype=wp.int32, device=device),
        articulation_block_col_rows=wp.array(block_col_rows_np, dtype=wp.int32, device=device),
        articulation_block_col_slots=wp.array(block_col_slots_np, dtype=wp.int32, device=device),
        articulation_factor_update_offsets=wp.array(factor_update_offsets_np, dtype=wp.int32, device=device),
        articulation_factor_update_dst_slots=wp.array(factor_update_dst_slots_np, dtype=wp.int32, device=device),
        articulation_factor_update_left_slots=wp.array(factor_update_left_slots_np, dtype=wp.int32, device=device),
        articulation_factor_update_right_slots=wp.array(factor_update_right_slots_np, dtype=wp.int32, device=device),
        articulation_diag_slots=wp.array(diag_slots_np, dtype=wp.int32, device=device),
        body_articulation_sparse=wp.array(body_articulation_sparse_host, dtype=wp.int32, device=device),
        body_articulation_local=wp.array(body_articulation_local_host, dtype=wp.int32, device=device),
        articulation_count=len(articulation_groups),
        articulation_body_count=len(articulation_bodies_host),
        articulation_joint_count=len(articulation_joints_host),
        block_count=len(block_cols_host),
    )
