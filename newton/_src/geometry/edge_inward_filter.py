# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative removal of fully inward mesh collision edges."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .types import Mesh

MINVAL = 1.0e-15


def filter_fully_inward_edges(
    mesh: Mesh,
    edge_indices: np.ndarray,
    *,
    canonical_vertex_ids: np.ndarray | None = None,
    edge_slot_topology: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Drop concave edges whose endpoints have fully inward manifold one-rings.

    A removable edge is shared by exactly two non-degenerate triangles. Both
    endpoint vertices must have connected, closed, consistently oriented
    one-rings, and every one-ring neighbor must lie on the inward side of the
    endpoint's angle-weighted tangent plane. Boundary, non-manifold, saddle,
    flat, and ambiguous features are preserved.

    Args:
        mesh: Source mesh with consistently authored triangle winding.
        edge_indices: Candidate collision-edge vertex pairs.
        canonical_vertex_ids: Optional precomputed canonical vertex IDs.
        edge_slot_topology: Optional precomputed edge-slot topology.

    Returns:
        A contiguous subset of ``edge_indices`` with fully inward edges removed.
    """
    if len(edge_indices) == 0 or mesh.indices.size == 0 or mesh.vertices.size == 0:
        return np.ascontiguousarray(edge_indices, dtype=np.int32)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
    canonical = canonical_vertex_ids
    if canonical is None:
        canonical = mesh._canonical_vertex_ids()
    canonical_triangles = canonical[triangles]

    # Winding-number SDF construction uses this volume sign to correct a
    # globally inverted mesh. Apply the same correction to the feature tests.
    volume_origin = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    centered_vertices = vertices - volume_origin
    v0 = centered_vertices[triangles[:, 0]]
    v1 = centered_vertices[triangles[:, 1]]
    v2 = centered_vertices[triangles[:, 2]]
    signed_volume = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
    diagonal = max(mesh._aabb_diagonal(), 1.0)
    volume_tolerance = np.finfo(np.float64).eps * diagonal**3 * max(len(triangles), 1)
    if abs(signed_volume) <= volume_tolerance:
        return np.ascontiguousarray(edge_indices, dtype=np.int32)
    orientation = 1.0 if signed_volume > 0.0 else -1.0

    if edge_slot_topology is None:
        edge_slot_topology = mesh._build_edge_slot_topology(canonical)
    orig_edges, _slot_keys, order, keys_sorted, face_normals, face_norms = edge_slot_topology
    if len(keys_sorted) == 0:
        return np.ascontiguousarray(edge_indices, dtype=np.int32)

    change = np.empty(len(keys_sorted), dtype=bool)
    change[0] = True
    change[1:] = keys_sorted[1:] != keys_sorted[:-1]
    group_starts = np.flatnonzero(change)
    group_ends = np.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:]
    group_ends[-1] = len(keys_sorted)
    group_counts = group_ends - group_starts

    canonical_count = int(canonical.max()) + 1
    canonical_positions = np.empty((canonical_count, 3), dtype=np.float64)
    canonical_positions[canonical] = vertices

    # Compute the per-corner geometry in batches. Keeping NumPy scalar
    # operations in the one-ring loop makes this preprocessing dominate SDF
    # cooking even for modest meshes.
    previous_vertices = np.roll(canonical_triangles, 1, axis=1)
    next_vertices = np.roll(canonical_triangles, -1, axis=1)
    if orientation < 0.0:
        previous_vertices, next_vertices = next_vertices, previous_vertices

    previous_deltas = canonical_positions[previous_vertices] - canonical_positions[canonical_triangles]
    next_deltas = canonical_positions[next_vertices] - canonical_positions[canonical_triangles]
    previous_lengths = np.linalg.norm(previous_deltas, axis=2)
    next_lengths = np.linalg.norm(next_deltas, axis=2)
    corner_valid = (
        (
            (canonical_triangles[:, 0] != canonical_triangles[:, 1])
            & (canonical_triangles[:, 1] != canonical_triangles[:, 2])
            & (canonical_triangles[:, 2] != canonical_triangles[:, 0])
            & (face_norms > MINVAL)
        )[:, None]
        & (previous_lengths > MINVAL)
        & (next_lengths > MINVAL)
    )

    cosine = np.ones_like(previous_lengths)
    np.divide(
        np.einsum("ijk,ijk->ij", previous_deltas, next_deltas),
        previous_lengths * next_lengths,
        out=cosine,
        where=corner_valid,
    )
    np.clip(cosine, -1.0, 1.0, out=cosine)
    corner_angles = np.zeros_like(cosine)
    np.arccos(cosine, out=corner_angles, where=corner_valid)

    unit_face_normals = np.zeros_like(face_normals)
    np.divide(face_normals, face_norms[:, None], out=unit_face_normals, where=(face_norms > MINVAL)[:, None])
    corner_normal_contributions = corner_angles[:, :, None] * orientation * unit_face_normals[:, None, :]

    previous_keys = (np.minimum(canonical_triangles, previous_vertices).astype(np.int64) << 32) | np.maximum(
        canonical_triangles, previous_vertices
    ).astype(np.int64)
    next_keys = (np.minimum(canonical_triangles, next_vertices).astype(np.int64) << 32) | np.maximum(
        canonical_triangles, next_vertices
    ).astype(np.int64)

    corner_vertices = canonical_triangles.reshape(-1)
    corner_order = np.argsort(corner_vertices, kind="stable")
    sorted_corner_vertices = corner_vertices[corner_order]
    vertex_change = np.empty(len(sorted_corner_vertices), dtype=bool)
    vertex_change[0] = True
    vertex_change[1:] = sorted_corner_vertices[1:] != sorted_corner_vertices[:-1]
    vertex_group_starts = np.flatnonzero(vertex_change)
    vertex_group_ends = np.empty_like(vertex_group_starts)
    vertex_group_ends[:-1] = vertex_group_starts[1:]
    vertex_group_ends[-1] = len(sorted_corner_vertices)
    vertex_group_counts = vertex_group_ends - vertex_group_starts
    vertex_ids = sorted_corner_vertices[vertex_group_starts]

    previous_flat = previous_vertices.reshape(-1)
    next_flat = next_vertices.reshape(-1)
    unique_edge_keys = keys_sorted[group_starts]
    previous_share_counts = group_counts[np.searchsorted(unique_edge_keys, previous_keys.reshape(-1))]
    next_share_counts = group_counts[np.searchsorted(unique_edge_keys, next_keys.reshape(-1))]
    corner_topology_valid = corner_valid.reshape(-1) & (previous_share_counts == 2) & (next_share_counts == 2)
    topology_valid = (vertex_group_counts >= 3) & np.logical_and.reduceat(
        corner_topology_valid[corner_order], vertex_group_starts
    )

    vertex_key_prefix = corner_vertices.astype(np.int64) << 32
    previous_pair_keys = np.sort(vertex_key_prefix | previous_flat.astype(np.int64))
    next_pair_keys = np.sort(vertex_key_prefix | next_flat.astype(np.int64))
    invalid_pair_vertices = set((previous_pair_keys[previous_pair_keys != next_pair_keys] >> 32).tolist())
    previous_duplicates = previous_pair_keys[1:][previous_pair_keys[1:] == previous_pair_keys[:-1]]
    next_duplicates = next_pair_keys[1:][next_pair_keys[1:] == next_pair_keys[:-1]]
    invalid_pair_vertices.update((previous_duplicates >> 32).tolist())
    invalid_pair_vertices.update((next_duplicates >> 32).tolist())
    if invalid_pair_vertices:
        topology_valid &= ~np.isin(vertex_ids, np.fromiter(invalid_pair_vertices, dtype=np.int32))

    # A valid oriented one-ring is one cycle, rather than multiple disjoint
    # cycles that happen to share the same center vertex.
    for group_idx in np.flatnonzero(topology_valid):
        start = vertex_group_starts[group_idx]
        end = vertex_group_ends[group_idx]
        corners = corner_order[start:end]
        successor = dict(zip(previous_flat[corners].tolist(), next_flat[corners].tolist(), strict=True))
        first = int(previous_flat[corners[0]])
        current = first
        reached: set[int] = set()
        while current not in reached and current in successor:
            reached.add(current)
            current = successor[current]
        if current != first or len(reached) != vertex_group_counts[group_idx]:
            topology_valid[group_idx] = False

    normal_sums = np.add.reduceat(corner_normal_contributions.reshape(-1, 3)[corner_order], vertex_group_starts, axis=0)
    normal_lengths = np.linalg.norm(normal_sums, axis=1)
    normals = np.zeros_like(normal_sums)
    np.divide(normal_sums, normal_lengths[:, None], out=normals, where=(normal_lengths > MINVAL)[:, None])
    normals_by_vertex = np.zeros((canonical_count, 3), dtype=np.float64)
    normals_by_vertex[vertex_ids] = normals
    heights = np.einsum("ij,ij->i", next_deltas.reshape(-1, 3), normals_by_vertex[corner_vertices])
    sorted_heights = heights[corner_order]
    min_heights = np.minimum.reduceat(sorted_heights, vertex_group_starts)
    max_heights = np.maximum.reduceat(sorted_heights, vertex_group_starts)

    plane_tolerance = 1.0e-7 * diagonal
    inward_mask = (
        topology_valid & (normal_lengths > MINVAL) & (min_heights >= -plane_tolerance) & (max_heights > plane_tolerance)
    )
    inward_vertices = set(vertex_ids[inward_mask].tolist())

    if len(inward_vertices) < 2:
        return np.ascontiguousarray(edge_indices, dtype=np.int32)

    concave_array = np.empty(0, dtype=np.int64)
    manifold_group_indices = np.flatnonzero(group_counts == 2)
    if len(manifold_group_indices) > 0:
        manifold_starts = group_starts[manifold_group_indices]
        slots_a = order[manifold_starts]
        slots_b = order[manifold_starts + 1]
        triangles_a = slots_a // 3
        triangles_b = slots_b // 3
        valid = (face_norms[triangles_a] > MINVAL) & (face_norms[triangles_b] > MINVAL)

        edges = orig_edges[slots_a]
        edge_canonical = canonical[edges]
        canonical_a = np.minimum(edge_canonical[:, 0], edge_canonical[:, 1])
        canonical_b = np.maximum(edge_canonical[:, 0], edge_canonical[:, 1])
        face_vertices_a = triangles[triangles_a]
        face_vertices_b = triangles[triangles_b]
        face_canonical_a = canonical[face_vertices_a]
        face_canonical_b = canonical[face_vertices_b]
        opposite_mask_a = (face_canonical_a != canonical_a[:, None]) & (face_canonical_a != canonical_b[:, None])
        opposite_mask_b = (face_canonical_b != canonical_a[:, None]) & (face_canonical_b != canonical_b[:, None])
        valid &= (opposite_mask_a.sum(axis=1) == 1) & (opposite_mask_b.sum(axis=1) == 1)
        opposite_a = np.take_along_axis(face_vertices_a, np.argmax(opposite_mask_a, axis=1)[:, None], axis=1)[:, 0]
        opposite_b = np.take_along_axis(face_vertices_b, np.argmax(opposite_mask_b, axis=1)[:, None], axis=1)[:, 0]

        edge_points = vertices[edges[:, 0]]
        normal_a = orientation * unit_face_normals[triangles_a]
        normal_b = orientation * unit_face_normals[triangles_b]
        side_b = np.einsum("ij,ij->i", normal_a, vertices[opposite_b] - edge_points)
        side_a = np.einsum("ij,ij->i", normal_b, vertices[opposite_a] - edge_points)
        concave_array = keys_sorted[manifold_starts[valid & (side_a > plane_tolerance) & (side_b > plane_tolerance)]]

    canonical_edges = canonical[edge_indices]
    canonical_a = np.minimum(canonical_edges[:, 0], canonical_edges[:, 1])
    canonical_b = np.maximum(canonical_edges[:, 0], canonical_edges[:, 1])
    edge_keys = (canonical_a.astype(np.int64) << 32) | canonical_b.astype(np.int64)
    inward_array = np.fromiter(inward_vertices, dtype=np.int32)
    keep = ~(
        np.isin(canonical_a, inward_array) & np.isin(canonical_b, inward_array) & np.isin(edge_keys, concave_array)
    )
    return np.ascontiguousarray(edge_indices[keep], dtype=np.int32)
