# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
import warp.utils

from ..utils.mesh import MeshAdjacency
from .bvh import compute_bvh_group_roots
from .kernels import (
    compute_edge_aabbs,
    compute_edge_groups,
    compute_tri_aabbs,
    compute_tri_groups,
    edge_colliding_edges_detection_kernel,
    fill_self_contact_reverse_rows_from_lists,
    fill_self_contact_rows_from_lists,
    finalize_row_offsets,
    sort_self_contact_reverse_rows,
    sort_self_contact_rows,
    triangle_triangle_collision_detection_kernel,
    vertex_triangle_collision_detection_kernel,
)

if TYPE_CHECKING:
    from ..sim import Model


@wp.struct
class TriMeshCollisionInfo:
    """Results of triangle-mesh self-collision queries.

    .. experimental::

        This storage-level result type may change without the normal
        deprecation period while the public self-contact API matures.

    Rows keep the historical layout: ``vertex_colliding_triangles`` and
    ``edge_colliding_edges`` hold interleaved (element, counterpart) index
    pairs and ``triangle_colliding_vertices`` holds vertex indices, but rows
    are now exact-length CSR (``*_offsets`` are real prefix sums and
    ``*_count`` equals each row's length). Detection appends hits to an
    internal shared scratch pool sized by an average budget per element, so
    memory scales with the actual contact count and a locally dense fold
    cannot overflow a private per-element budget; rows are rebuilt from that
    pool after each detection. The vertex and edge rows come out in
    BVH-traversal order per element and, absent overflow, are deterministic
    run to run (under overflow, which records won a pool slot is an
    inter-thread race; each such row keeps a prefix of its traversal order,
    the matching ``global_pair_counts`` overflow flag is set, and the cursor
    keeps counting total demand). The optional triangle-side reverse rows have
    scheduling-dependent order (many writers); ``sort_contact_rows`` makes
    every row canonical. Kernel code should read results through the internal
    ``get_*`` accessors.
    """

    global_pair_counts: wp.array[wp.int32]
    """Pair cursors and overflow flags: [vt cursor, vt overflow, ee cursor, ee overflow].
    The cursors count total detection demand; stored records are ``min(cursor, capacity)``."""

    vertex_colliding_triangles: wp.array[wp.int32]
    """Interleaved (vertex, triangle) index pairs in exact CSR rows."""
    vertex_colliding_triangles_offsets: wp.array[wp.int32]
    """CSR row offsets for vertices, shape ``[particle_count + 1]``, exact."""
    vertex_colliding_triangles_count: wp.array[wp.int32]
    """Stored collision count for each vertex (the exact CSR row length)."""
    vertex_colliding_triangles_min_dist: wp.array[float]
    """Minimum detected vertex-triangle distance for each vertex [m]."""

    triangle_colliding_vertices: wp.array[wp.int32]
    """Optional contacting-vertex indices in exact CSR rows; empty unless recording is enabled."""
    triangle_colliding_vertices_offsets: wp.array[wp.int32]
    """Optional CSR row offsets for triangles, shape ``[tri_count + 1]``."""
    triangle_colliding_vertices_count: wp.array[wp.int32]
    """Optional stored collision count for each triangle."""
    triangle_colliding_vertices_min_dist: wp.array[float]
    """Minimum detected triangle-vertex distance for each triangle [m]; when
    triangle-side recording is off it keeps the constant query-radius fill."""

    edge_colliding_edges: wp.array[wp.int32]
    """Interleaved (edge, colliding edge) index pairs in exact CSR rows, both directions."""
    edge_colliding_edges_offsets: wp.array[wp.int32]
    """CSR row offsets for edges, shape ``[edge_count + 1]``, exact."""
    edge_colliding_edges_count: wp.array[wp.int32]
    """Stored collision count for each edge (the exact CSR row length)."""
    edge_colliding_edges_min_dist: wp.array[float]
    """Minimum detected edge-edge distance for each edge [m]."""


@wp.func
def get_vertex_colliding_triangles_count(collision_info: TriMeshCollisionInfo, vertex: int):
    """Return the stored collision count for ``vertex`` (exact CSR row length)."""
    return (
        collision_info.vertex_colliding_triangles_offsets[vertex + 1]
        - collision_info.vertex_colliding_triangles_offsets[vertex]
    )


@wp.func
def get_vertex_colliding_triangles(collision_info: TriMeshCollisionInfo, vertex: int, collision_index: int):
    """Return the triangle index for ``collision_index`` of ``vertex``."""
    offset = collision_info.vertex_colliding_triangles_offsets[vertex]
    return collision_info.vertex_colliding_triangles[2 * (offset + collision_index) + 1]


@wp.func
def get_vertex_collision_buffer_vertex_index(collision_info: TriMeshCollisionInfo, vertex: int, collision_index: int):
    """Return the stored source vertex for ``collision_index`` of ``vertex``."""
    offset = collision_info.vertex_colliding_triangles_offsets[vertex]
    return collision_info.vertex_colliding_triangles[2 * (offset + collision_index)]


@wp.func
def get_triangle_colliding_vertices_count(collision_info: TriMeshCollisionInfo, triangle: int):
    """Return the stored collision count for ``triangle`` (exact CSR row length)."""
    return (
        collision_info.triangle_colliding_vertices_offsets[triangle + 1]
        - collision_info.triangle_colliding_vertices_offsets[triangle]
    )


@wp.func
def get_triangle_colliding_vertices(collision_info: TriMeshCollisionInfo, triangle: int, collision_index: int):
    """Return the vertex index for ``collision_index`` of ``triangle``."""
    offset = collision_info.triangle_colliding_vertices_offsets[triangle]
    return collision_info.triangle_colliding_vertices[offset + collision_index]


@wp.func
def get_edge_colliding_edges_count(collision_info: TriMeshCollisionInfo, edge: int):
    """Return the stored collision count for ``edge`` (exact CSR row length)."""
    return collision_info.edge_colliding_edges_offsets[edge + 1] - collision_info.edge_colliding_edges_offsets[edge]


@wp.func
def get_edge_colliding_edges(collision_info: TriMeshCollisionInfo, edge: int, collision_index: int):
    """Return the target edge for ``collision_index`` of ``edge``."""
    offset = collision_info.edge_colliding_edges_offsets[edge]
    return collision_info.edge_colliding_edges[2 * (offset + collision_index) + 1]


@wp.func
def get_edge_collision_buffer_edge_index(collision_info: TriMeshCollisionInfo, edge: int, collision_index: int):
    """Return the stored source edge for ``collision_index`` of ``edge``."""
    offset = collision_info.edge_colliding_edges_offsets[edge]
    return collision_info.edge_colliding_edges[2 * (offset + collision_index)]


def _as_numpy(arr) -> np.ndarray:
    """Return ``arr`` as NumPy, accepting either a NumPy or a Warp int array."""
    return arr if isinstance(arr, np.ndarray) else arr.numpy()


def _csr_row(vals: np.ndarray, offs: np.ndarray, i: int) -> np.ndarray:
    """Extract row ``i`` from flat CSR arrays."""
    return vals[offs[i] : offs[i + 1]]


def set_to_csr(
    list_of_sets: list[set[int]], dtype: np.dtype = np.int32, sort: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-row integer sets to flat CSR values and offsets."""
    offsets = np.zeros(len(list_of_sets) + 1, dtype=dtype)
    sizes = np.fromiter((len(s) for s in list_of_sets), count=len(list_of_sets), dtype=dtype)
    np.cumsum(sizes, out=offsets[1:])

    flat = np.empty(offsets[-1], dtype=dtype)
    cursor = 0
    for row in list_of_sets:
        values = np.fromiter(sorted(row) if sort else row, count=len(row), dtype=dtype)
        flat[cursor : cursor + len(values)] = values
        cursor += len(values)
    return flat, offsets


def one_ring_vertices(
    vertex: int, edge_indices: np.ndarray, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """Return vertices sharing a collision edge with ``vertex``."""
    edge_v0 = edge_indices[:, 2]
    edge_v1 = edge_indices[:, 3]
    edge_rows = _csr_row(v_adj_edges, v_adj_edges_offsets, vertex)
    edge_ids = edge_rows[::2]
    local_slots = edge_rows[1::2]
    if edge_ids.size == 0:
        return np.empty(0, dtype=np.int32)

    endpoint_edge_ids = edge_ids[np.where(local_slots >= 2)]
    us = edge_v0[endpoint_edge_ids]
    vs = edge_v1[endpoint_edge_ids]
    assert (np.logical_or(us == vertex, vs == vertex)).all()

    neighbors = np.unique(np.concatenate([us, vs]))
    return neighbors[neighbors != vertex]


def leq_n_ring_vertices(
    vertex: int, edge_indices: np.ndarray, n: int, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """Return vertices within ``n`` edge rings of ``vertex``, including itself."""
    visited = {vertex}
    frontier = {vertex}
    for _ in range(n):
        next_frontier = set()
        for current in frontier:
            for neighbor in one_ring_vertices(current, edge_indices, v_adj_edges, v_adj_edges_offsets):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
    return np.fromiter(visited, dtype=np.int32)


def build_vertex_n_ring_tris_collision_filter(
    n: int,
    particle_count: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
    v_adj_tris: np.ndarray,
    v_adj_tris_offsets: np.ndarray,
) -> list[set[int]] | None:
    """Build vertex-triangle filters from adjacency within ``n`` edge rings."""
    if n <= 1:
        return None

    vertex_triangle_sets = [set() for _ in range(particle_count)]
    for vertex in range(particle_count):
        if n == 2:
            neighbor_vertices = one_ring_vertices(vertex, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            neighbor_vertices = leq_n_ring_vertices(vertex, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        incident_tris = set(_csr_row(v_adj_tris, v_adj_tris_offsets, vertex)[::2])
        filter_set = vertex_triangle_sets[vertex]
        for neighbor in neighbor_vertices:
            if neighbor != vertex:
                filter_set.update(_csr_row(v_adj_tris, v_adj_tris_offsets, neighbor)[::2])
        filter_set.difference_update(incident_tris)

    return vertex_triangle_sets


def build_edge_n_ring_edge_collision_filter(
    n: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
) -> list[set[int]] | None:
    """Build edge-edge filters from adjacency within ``n`` edge rings."""
    if n <= 1:
        return None

    edge_sets = [set() for _ in range(edge_indices.shape[0])]
    for edge_id in range(edge_indices.shape[0]):
        v0 = edge_indices[edge_id, 2]
        v1 = edge_indices[edge_id, 3]

        if n == 2:
            v0_neighbors = one_ring_vertices(v0, edge_indices, v_adj_edges, v_adj_edges_offsets)
            v1_neighbors = one_ring_vertices(v1, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            v0_neighbors = leq_n_ring_vertices(v0, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)
            v1_neighbors = leq_n_ring_vertices(v1, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        neighbor_vertices = set(v0_neighbors)
        neighbor_vertices.update(v1_neighbors)

        incident_to_v0 = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v0)[::2])
        incident_to_v1 = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v1)[::2])

        filter_set = edge_sets[edge_id]
        for neighbor in neighbor_vertices:
            if neighbor != v0 and neighbor != v1:
                edge_rows = _csr_row(v_adj_edges, v_adj_edges_offsets, neighbor)
                adj_edges = edge_rows[::2]
                local_slots = edge_rows[1::2]
                filter_set.update(adj_edges[np.where(local_slots >= 2)])

        filter_set.difference_update(incident_to_v0)
        filter_set.difference_update(incident_to_v1)

    return edge_sets


def build_tri_mesh_collision_info(
    particle_count: int,
    tri_count: int,
    edge_count: int,
    *,
    vertex_collision_buffer_pre_alloc: int = 8,
    triangle_collision_buffer_pre_alloc: int = 8,
    edge_collision_buffer_pre_alloc: int = 16,
    record_triangle_contacting_vertices: bool = False,
    device=None,
) -> TriMeshCollisionInfo:
    """Allocate all self-contact result arrays into a :class:`TriMeshCollisionInfo`.

    This is the single allocation path for tri-mesh self-contact results:
    :class:`TriMeshCollisionDetector` calls it when constructed with
    ``init_collision_info=True``, and result-owning containers call it to
    allocate buffers the detector then writes into.

    The ``*_pre_alloc`` values are average contact budgets per element: each
    family's rows hold up to ``pre_alloc x element_count`` stored contacts
    that any element can draw from, so a locally dense fold only overflows
    when the whole mesh's contact demand exceeds the pool.

    When ``record_triangle_contacting_vertices`` is ``False`` the
    triangle-side CSR fields are left at their empty defaults;
    ``triangle_colliding_vertices_min_dist`` is always allocated.

    Args:
        particle_count: Number of mesh vertices.
        tri_count: Number of mesh triangles.
        edge_count: Number of mesh edges.
        vertex_collision_buffer_pre_alloc: Average vertex-triangle contact
            budget per vertex; row capacity = this x ``particle_count`` pairs.
        triangle_collision_buffer_pre_alloc: Unused for sizing (the reverse
            table holds the same stored contacts); kept for signature stability.
        edge_collision_buffer_pre_alloc: Average edge-edge contact budget per
            edge; row capacity = this x ``edge_count`` pairs.
        record_triangle_contacting_vertices: Whether to allocate the reverse
            triangle-to-vertex CSR table.
        device: Warp device on which to allocate the arrays.

    Returns:
        An allocated collision-result struct.
    """
    info = TriMeshCollisionInfo()

    vt_capacity = vertex_collision_buffer_pre_alloc * particle_count
    ee_capacity = edge_collision_buffer_pre_alloc * edge_count

    info.global_pair_counts = wp.zeros(shape=(4,), dtype=wp.int32, device=device)

    # interleaved (element, counterpart) pairs: 2 ints per stored contact
    info.vertex_colliding_triangles = wp.zeros(shape=(max(2 * vt_capacity, 1),), dtype=wp.int32, device=device)
    info.vertex_colliding_triangles_offsets = wp.zeros(shape=(particle_count + 1,), dtype=wp.int32, device=device)
    info.vertex_colliding_triangles_count = wp.zeros(shape=(particle_count,), dtype=wp.int32, device=device)
    info.vertex_colliding_triangles_min_dist = wp.zeros(shape=(particle_count,), dtype=float, device=device)

    if record_triangle_contacting_vertices:
        # reverse rows store the contacting vertex index only: 1 int per contact
        info.triangle_colliding_vertices = wp.zeros(shape=(max(vt_capacity, 1),), dtype=wp.int32, device=device)
        info.triangle_colliding_vertices_offsets = wp.zeros(shape=(tri_count + 1,), dtype=wp.int32, device=device)
        info.triangle_colliding_vertices_count = wp.zeros(shape=(tri_count,), dtype=wp.int32, device=device)

    # needed regardless of whether triangle contacting vertices are recorded
    info.triangle_colliding_vertices_min_dist = wp.zeros(shape=(tri_count,), dtype=float, device=device)

    info.edge_colliding_edges = wp.zeros(shape=(max(2 * ee_capacity, 1),), dtype=wp.int32, device=device)
    info.edge_colliding_edges_offsets = wp.zeros(shape=(edge_count + 1,), dtype=wp.int32, device=device)
    info.edge_colliding_edges_count = wp.zeros(shape=(edge_count,), dtype=wp.int32, device=device)
    info.edge_colliding_edges_min_dist = wp.zeros(shape=(edge_count,), dtype=float, device=device)

    return info


class TriMeshCollisionDetector:
    def __init__(
        self,
        model: Model,
        record_triangle_contacting_vertices=False,
        vertex_positions=None,
        vertex_collision_buffer_pre_alloc=8,
        vertex_triangle_filtering_list=None,
        vertex_triangle_filtering_list_offsets=None,
        triangle_collision_buffer_pre_alloc=8,
        edge_collision_buffer_pre_alloc=16,
        edge_filtering_list=None,
        edge_filtering_list_offsets=None,
        topological_contact_filter_threshold: int = 0,
        external_vertex_triangle_filtering_map: dict | None = None,
        external_edge_edge_filtering_map: dict | None = None,
        triangle_triangle_collision_buffer_pre_alloc=8,
        triangle_triangle_collision_buffer_max_alloc=256,
        edge_edge_parallel_epsilon=1e-5,
        collision_detection_block_size: int | None = None,
        collision_info: TriMeshCollisionInfo | None = None,
        init_collision_info: bool = False,
        sort_contact_rows: bool = False,
    ):
        self.model = model
        self.record_triangle_contacting_vertices = record_triangle_contacting_vertices
        # sort each CSR row by counterpart index after the build: canonical row
        # order for a given contact set (frame-to-frame matching, reproducible
        # consumers); off by default because the solver does not need it
        self.sort_contact_rows = sort_contact_rows
        self.vertex_positions = model.particle_q if vertex_positions is None else vertex_positions
        self.device = model.device
        self.vertex_collision_buffer_pre_alloc = vertex_collision_buffer_pre_alloc
        self.triangle_collision_buffer_pre_alloc = triangle_collision_buffer_pre_alloc
        self.edge_collision_buffer_pre_alloc = edge_collision_buffer_pre_alloc
        self.triangle_triangle_collision_buffer_pre_alloc = triangle_triangle_collision_buffer_pre_alloc
        self.triangle_triangle_collision_buffer_max_alloc = triangle_triangle_collision_buffer_max_alloc

        self.vertex_triangle_filtering_list = vertex_triangle_filtering_list
        self.vertex_triangle_filtering_list_offsets = vertex_triangle_filtering_list_offsets

        self.edge_filtering_list = edge_filtering_list
        self.edge_filtering_list_offsets = edge_filtering_list_offsets

        self.edge_edge_parallel_epsilon = edge_edge_parallel_epsilon
        # The soft-mesh adjacency comes from the model; ensure its vertex-adjacency CSR is built.
        # init_vertex_adjacency is idempotent (vertex_adjacency_initialized flag), so this is a no-op
        # once the solver has built it.
        if model.soft_mesh_adjacency is None:
            raise ValueError("model.soft_mesh_adjacency is missing; finalize the model with ModelBuilder.")
        self.mesh_adjacency = model.soft_mesh_adjacency.init_vertex_adjacency(model.particle_count)

        # None picks the block size per kernel launch (see _edge_collision_block_size).
        self.collision_detection_block_size = collision_detection_block_size

        # Build each filter family independently: generate a side only when the caller did not
        # provide it explicitly and a threshold/external source requests it (so providing one
        # list plus an external map for the other side still generates the missing side).
        need_vertex_triangle = vertex_triangle_filtering_list is None and (
            topological_contact_filter_threshold >= 2 or external_vertex_triangle_filtering_map is not None
        )
        need_edge_edge = edge_filtering_list is None and (
            topological_contact_filter_threshold >= 2 or external_edge_edge_filtering_map is not None
        )
        if (need_vertex_triangle or need_edge_edge) and self.model.tri_count > 0:
            # Extract the shared vertex adjacency once, then build each family with its own builder.
            adjacency = None
            if topological_contact_filter_threshold >= 2 and self.model.edge_indices is not None:
                adjacency = self._extract_filter_adjacency()
            if need_vertex_triangle:
                self._build_vertex_triangle_filter(
                    topological_contact_filter_threshold, external_vertex_triangle_filtering_map, adjacency
                )
            if need_edge_edge:
                self._build_edge_edge_filter(
                    topological_contact_filter_threshold, external_edge_edge_filtering_map, adjacency
                )

        self.lower_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        self.tri_groups = wp.array(shape=(model.tri_count,), dtype=wp.int32, device=model.device)
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=model.tri_count,
            device=model.device,
        )
        wp.launch(
            kernel=compute_tri_groups,
            inputs=[model.tri_indices, model.particle_world, model.world_count, self.tri_groups],
            dim=model.tri_count,
            device=model.device,
        )

        self.bvh_tris = wp.Bvh(self.lower_bounds_tris, self.upper_bounds_tris, groups=self.tri_groups)
        self.bvh_tris_group_roots = wp.zeros(model.world_count + 1, dtype=wp.int32, device=model.device)
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=model.world_count + 1,
            inputs=[self.bvh_tris.id, self.bvh_tris_group_roots],
            device=model.device,
        )

        # Collision detection results live in a TriMeshCollisionInfo owned outside
        # the detector. Explicitly one of: injected (collision_info=...), self-built
        # at construction (init_collision_info=True), or absent until a result
        # struct is bound via _bind_external_buffers.
        if collision_info is not None and init_collision_info:
            raise ValueError("pass either collision_info or init_collision_info=True, not both")
        if init_collision_info:
            collision_info = build_tri_mesh_collision_info(
                model.particle_count,
                model.tri_count,
                model.edge_count,
                vertex_collision_buffer_pre_alloc=vertex_collision_buffer_pre_alloc,
                triangle_collision_buffer_pre_alloc=triangle_collision_buffer_pre_alloc,
                edge_collision_buffer_pre_alloc=edge_collision_buffer_pre_alloc,
                record_triangle_contacting_vertices=record_triangle_contacting_vertices,
                device=self.device,
            )
        if collision_info is not None:
            self._validate_collision_info(collision_info)
        self.collision_info = collision_info

        self.lower_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        self.edge_groups = wp.array(shape=(model.edge_count,), dtype=wp.int32, device=model.device)
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=model.edge_count,
            device=model.device,
        )
        wp.launch(
            kernel=compute_edge_groups,
            inputs=[model.edge_indices, model.particle_world, model.world_count, self.edge_groups],
            dim=model.edge_count,
            device=model.device,
        )

        self.bvh_edges = wp.Bvh(self.lower_bounds_edges, self.upper_bounds_edges, groups=self.edge_groups)
        self.bvh_edges_group_roots = wp.zeros(model.world_count + 1, dtype=wp.int32, device=model.device)
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=model.world_count + 1,
            inputs=[self.bvh_edges.id, self.bvh_edges_group_roots],
            device=model.device,
        )

        # resize_flags only serves the on-demand triangle-triangle intersection
        # buffers now; self-contact overflow lives in collision_info.global_pair_counts
        self.resize_flags = wp.zeros(shape=(4,), dtype=wp.int32, device=self.device)
        # stand-in for the optional per-triangle min-dist output when the
        # triangle-side recording is off (parity with the historical behavior:
        # the array then keeps its constant query-radius fill)
        self._empty_min_dist = wp.empty(shape=(0,), dtype=float, device=self.device)
        self._empty_int32 = wp.empty(shape=(0,), dtype=wp.int32, device=self.device)

        # Internal append log + linked-list scratch for the CSR build. Heads
        # are per element (fixed for the model); the pair/next pool is sized by
        # the current budgets and shared by both families: vt and ee detection
        # run back-to-back on one stream and each family's log is dead as soon
        # as its rows are filled.
        self._vt_list_heads = wp.empty(shape=(max(model.particle_count, 1),), dtype=wp.int32, device=self.device)
        self._ee_list_heads = wp.empty(shape=(max(model.edge_count, 1),), dtype=wp.int32, device=self.device)
        self._tri_list_heads = (
            wp.empty(shape=(max(model.tri_count, 1),), dtype=wp.int32, device=self.device)
            if record_triangle_contacting_vertices
            else self._empty_int32
        )
        self._pair_scratch = None
        self._list_next_scratch = None
        self._tri_list_next_scratch = self._empty_int32
        self._ensure_scratch()

        # data for triangle-triangle intersection; they will only be initialized on demand, as triangle-triangle intersection is not needed for simulation
        self.triangle_intersecting_triangles = None
        self.triangle_intersecting_triangles_count = None
        self.triangle_intersecting_triangles_offsets = None

    def _ensure_scratch(self) -> None:
        """(Re)allocate the shared append log for the current budgets.

        One ``vec2i`` pool plus one next-slot pool serve both families; the
        optional triangle-side reverse lists chain the same vt records through
        their own next pool.
        """
        capacity = max(
            self.vertex_collision_buffer_pre_alloc * self.model.particle_count,
            self.edge_collision_buffer_pre_alloc * self.model.edge_count,
            1,
        )
        if self._pair_scratch is None or self._pair_scratch.shape[0] < capacity:
            self._pair_scratch = wp.empty(shape=(capacity,), dtype=wp.vec2i, device=self.device)
            self._list_next_scratch = wp.empty(shape=(capacity,), dtype=wp.int32, device=self.device)
            if self.record_triangle_contacting_vertices:
                self._tri_list_next_scratch = wp.empty(shape=(capacity,), dtype=wp.int32, device=self.device)

    def _validate_collision_info(self, collision_info: TriMeshCollisionInfo) -> None:
        """Validate a result struct (injected or self-built) against this detector."""

        def validate_array(name, array, size, dtype):
            if array is None:
                raise ValueError(f"collision_info.{name} is required")
            if array.device != self.device:
                raise ValueError(f"collision_info.{name} is on {array.device}, but the detector is on {self.device}")
            if array.size != size:
                raise ValueError(f"collision_info.{name} has size {array.size}, expected {size}")
            if array.dtype != dtype:
                raise ValueError(f"collision_info.{name} has dtype {array.dtype}, expected {dtype}")

        particle_count = self.model.particle_count
        tri_count = self.model.tri_count
        edge_count = self.model.edge_count
        # expected sizes must mirror build_tri_mesh_collision_info exactly,
        # including the degenerate zero-count clamps
        vt_capacity = self.vertex_collision_buffer_pre_alloc * particle_count
        ee_capacity = self.edge_collision_buffer_pre_alloc * edge_count
        arrays = (
            (
                "vertex_colliding_triangles",
                collision_info.vertex_colliding_triangles,
                max(2 * vt_capacity, 1),
                wp.int32,
            ),
            ("global_pair_counts", collision_info.global_pair_counts, 4, wp.int32),
            (
                "vertex_colliding_triangles_offsets",
                collision_info.vertex_colliding_triangles_offsets,
                particle_count + 1,
                wp.int32,
            ),
            (
                "vertex_colliding_triangles_count",
                collision_info.vertex_colliding_triangles_count,
                particle_count,
                wp.int32,
            ),
            (
                "vertex_colliding_triangles_min_dist",
                collision_info.vertex_colliding_triangles_min_dist,
                particle_count,
                wp.float32,
            ),
            (
                "triangle_colliding_vertices_min_dist",
                collision_info.triangle_colliding_vertices_min_dist,
                tri_count,
                wp.float32,
            ),
            ("edge_colliding_edges", collision_info.edge_colliding_edges, max(2 * ee_capacity, 1), wp.int32),
            (
                "edge_colliding_edges_offsets",
                collision_info.edge_colliding_edges_offsets,
                edge_count + 1,
                wp.int32,
            ),
            (
                "edge_colliding_edges_count",
                collision_info.edge_colliding_edges_count,
                edge_count,
                wp.int32,
            ),
            (
                "edge_colliding_edges_min_dist",
                collision_info.edge_colliding_edges_min_dist,
                edge_count,
                wp.float32,
            ),
        )
        if self.record_triangle_contacting_vertices:
            arrays += (
                (
                    "triangle_colliding_vertices",
                    collision_info.triangle_colliding_vertices,
                    max(vt_capacity, 1),
                    wp.int32,
                ),
                (
                    "triangle_colliding_vertices_offsets",
                    collision_info.triangle_colliding_vertices_offsets,
                    tri_count + 1,
                    wp.int32,
                ),
                (
                    "triangle_colliding_vertices_count",
                    collision_info.triangle_colliding_vertices_count,
                    tri_count,
                    wp.int32,
                ),
            )
        for array in arrays:
            validate_array(*array)

    def _bind_external_buffers(self, collision_info: TriMeshCollisionInfo):
        """Re-point result reads/writes at another externally-owned struct.

        Plain reference assignment: the detection launches and the forwarding
        properties below always go through ``self.collision_info``, so one
        detector (one BVH set) can serve any number of result buffers.
        """
        self._validate_collision_info(collision_info)
        self.collision_info = collision_info
        # budgets may have grown since construction; the log must cover them
        self._ensure_scratch()

    def _require_collision_info(self) -> None:
        if self.collision_info is None:
            raise ValueError(
                "TriMeshCollisionDetector has no result buffers; construct it with "
                "init_collision_info=True, pass collision_info=..., or bind a result "
                "struct before detecting."
            )

    # Result-array views into the owned/injected ``collision_info`` (D21: the
    # detector owns no result buffers). Read-only properties preserve the
    # historical attribute surface, including ``None`` for the optional
    # triangle-side buffers when ``record_triangle_contacting_vertices`` is off.

    @property
    def global_pair_counts(self):
        return self.collision_info.global_pair_counts

    @property
    def vertex_colliding_triangles(self):
        return self.collision_info.vertex_colliding_triangles

    @property
    def vertex_colliding_triangles_offsets(self):
        return self.collision_info.vertex_colliding_triangles_offsets

    @property
    def vertex_colliding_triangles_count(self):
        return self.collision_info.vertex_colliding_triangles_count

    @property
    def vertex_colliding_triangles_min_dist(self):
        return self.collision_info.vertex_colliding_triangles_min_dist

    @property
    def triangle_colliding_vertices(self):
        return self.collision_info.triangle_colliding_vertices if self.record_triangle_contacting_vertices else None

    @property
    def triangle_colliding_vertices_offsets(self):
        return (
            self.collision_info.triangle_colliding_vertices_offsets
            if self.record_triangle_contacting_vertices
            else None
        )

    @property
    def triangle_colliding_vertices_count(self):
        return (
            self.collision_info.triangle_colliding_vertices_count if self.record_triangle_contacting_vertices else None
        )

    @property
    def triangle_colliding_vertices_min_dist(self):
        return self.collision_info.triangle_colliding_vertices_min_dist

    @property
    def edge_colliding_edges(self):
        return self.collision_info.edge_colliding_edges

    @property
    def edge_colliding_edges_offsets(self):
        return self.collision_info.edge_colliding_edges_offsets

    @property
    def edge_colliding_edges_count(self):
        return self.collision_info.edge_colliding_edges_count

    @property
    def edge_colliding_edges_min_dist(self):
        return self.collision_info.edge_colliding_edges_min_dist

    def set_collision_filter_list(
        self,
        vertex_triangle_filtering_list,
        vertex_triangle_filtering_list_offsets,
        edge_filtering_list,
        edge_filtering_list_offsets,
    ):
        self.vertex_triangle_filtering_list = vertex_triangle_filtering_list
        self.vertex_triangle_filtering_list_offsets = vertex_triangle_filtering_list_offsets

        self.edge_filtering_list = edge_filtering_list
        self.edge_filtering_list_offsets = edge_filtering_list_offsets

    def _extract_filter_adjacency(self):
        """Return ``(edge_indices, v_adj_edges, v_adj_edges_offsets, v_adj_tris, v_adj_tris_offsets)`` as
        NumPy for the topological filter builders.

        Reuses the model's vertex-adjacency CSR when it is already populated, otherwise computes it on
        demand. Shared by the vertex-triangle and edge-edge builders so the adjacency is extracted once.
        """
        edge_indices = self.model.edge_indices.numpy()
        adjacency = self.mesh_adjacency
        if (
            adjacency is not None
            and adjacency.v_adj_edges is not None
            and adjacency.v_adj_edges.size > 0
            and adjacency.v_adj_edges_offsets.size > 0
            and adjacency.v_adj_tris_offsets.size > 0
        ):
            source = adjacency
        else:
            source = MeshAdjacency.compute_vertex_adjacency(
                self.model.particle_count,
                edge_indices=self.model.edge_indices,
                tri_indices=self.model.tri_indices,
            )
        return (
            edge_indices,
            _as_numpy(source.v_adj_edges),
            _as_numpy(source.v_adj_edges_offsets),
            _as_numpy(source.v_adj_tris),
            _as_numpy(source.v_adj_tris_offsets),
        )

    def _build_vertex_triangle_filter(
        self,
        topological_contact_filter_threshold: int,
        external_vertex_triangle_filtering_map: dict | None,
        adjacency: tuple | None,
    ) -> None:
        """Build the detector-owned vertex-triangle filter list from the n-ring topology and the optional
        external map. The caller decides whether this side is needed (an explicitly-provided list is left
        untouched); ``adjacency`` is the shared :meth:`_extract_filter_adjacency` result or ``None``.
        """
        filter_sets = None
        if topological_contact_filter_threshold >= 2 and adjacency is not None:
            edge_indices, v_adj_edges, v_adj_edges_offsets, v_adj_tris, v_adj_tris_offsets = adjacency
            filter_sets = build_vertex_n_ring_tris_collision_filter(
                topological_contact_filter_threshold,
                self.model.particle_count,
                edge_indices,
                v_adj_edges,
                v_adj_edges_offsets,
                v_adj_tris,
                v_adj_tris_offsets,
            )
        if external_vertex_triangle_filtering_map is not None:
            if filter_sets is None:
                filter_sets = [set() for _ in range(self.model.particle_count)]
            for vertex_id, filter_set in external_vertex_triangle_filtering_map.items():
                filter_sets[vertex_id].update(filter_set)

        if filter_sets is not None:
            filtering_list, filtering_list_offsets = set_to_csr(filter_sets)
            self.vertex_triangle_filtering_list = wp.array(filtering_list, dtype=wp.int32, device=self.device)
            self.vertex_triangle_filtering_list_offsets = wp.array(
                filtering_list_offsets, dtype=wp.int32, device=self.device
            )

    def _build_edge_edge_filter(
        self,
        topological_contact_filter_threshold: int,
        external_edge_edge_filtering_map: dict | None,
        adjacency: tuple | None,
    ) -> None:
        """Build the detector-owned edge-edge filter list from the n-ring topology and the optional
        external map. The caller decides whether this side is needed (an explicitly-provided list is left
        untouched); ``adjacency`` is the shared :meth:`_extract_filter_adjacency` result or ``None``.
        """
        filter_sets = None
        if topological_contact_filter_threshold >= 2 and adjacency is not None:
            edge_indices, v_adj_edges, v_adj_edges_offsets, _, _ = adjacency
            filter_sets = build_edge_n_ring_edge_collision_filter(
                topological_contact_filter_threshold,
                edge_indices,
                v_adj_edges,
                v_adj_edges_offsets,
            )
        if external_edge_edge_filtering_map is not None:
            if filter_sets is None:
                filter_sets = [set() for _ in range(self.model.edge_count)]
            for edge_id, filter_set in external_edge_edge_filtering_map.items():
                filter_sets[edge_id].update(filter_set)

        if filter_sets is not None:
            filtering_list, filtering_list_offsets = set_to_csr(filter_sets)
            self.edge_filtering_list = wp.array(filtering_list, dtype=wp.int32, device=self.device)
            self.edge_filtering_list_offsets = wp.array(filtering_list_offsets, dtype=wp.int32, device=self.device)

    def get_collision_data(self):
        """Return the result struct; results live in :attr:`collision_info` (D27)."""
        return self.collision_info

    def _build_pair_rows(self, fill_kernel, row_counts, row_offsets, list_heads, list_next, row_values):
        """Build one family's exact CSR rows from its per-element record lists.

        Graph-capturable passes: exclusive-scan the stored counts (written by
        detection) into the row offsets; one thread per element walks its
        chain in the append log and writes its row content back to front
        (restoring forward BVH-traversal order for the single-writer vertex
        and edge lists; the many-writer triangle reverse lists keep a
        scheduling-dependent order); optionally one in-place insertion sort
        per row when ``sort_contact_rows`` is on. Lists link only stored
        records, so rows stay consistent when detection dropped records on
        overflow.
        """
        element_count = row_counts.shape[0]
        if element_count == 0:
            return
        warp.utils.array_scan(row_counts, row_offsets[:element_count], inclusive=False)
        wp.launch(
            kernel=finalize_row_offsets,
            dim=1,
            inputs=[row_counts],
            outputs=[row_offsets],
            device=self.device,
        )
        wp.launch(
            kernel=fill_kernel,
            dim=element_count,
            inputs=[self._pair_scratch, list_heads, list_next, row_offsets],
            outputs=[row_values],
            device=self.device,
        )
        if self.sort_contact_rows:
            sort_kernel = (
                sort_self_contact_reverse_rows
                if fill_kernel is fill_self_contact_reverse_rows_from_lists
                else sort_self_contact_rows
            )
            wp.launch(
                kernel=sort_kernel,
                dim=element_count,
                inputs=[row_offsets],
                outputs=[row_values],
                device=self.device,
            )

    def check_self_contact_overflow(self, warn: bool = True) -> tuple[int, int, bool, bool]:
        """Read back pair demand and overflow flags (synchronizes the device).

        Returns ``(vt_demand, ee_demand, vt_overflowed, ee_overflowed)``. The
        demands are the total pair counts detection tried to store; when an
        overflow flag is set the corresponding pair array kept only its first
        ``capacity`` records and the CSR rows cover only those.
        """
        self._require_collision_info()
        global_pair_counts = self.collision_info.global_pair_counts.numpy()
        vt_demand, vt_overflow = int(global_pair_counts[0]), bool(global_pair_counts[1])
        ee_demand, ee_overflow = int(global_pair_counts[2]), bool(global_pair_counts[3])
        if warn and (vt_overflow or ee_overflow):
            warnings.warn(
                f"tri-mesh self-contact pair arrays overflowed "
                f"(vertex-triangle demand {vt_demand} / capacity "
                f"{self.collision_info.vertex_colliding_triangles.shape[0] // 2}, "
                f"edge-edge demand {ee_demand} / capacity "
                f"{self.collision_info.edge_colliding_edges.shape[0] // 2}); "
                "excess contacts were dropped this detection. Raise the average "
                "per-element budgets (SolverVBD: particle_vertex_contact_buffer_size / "
                "particle_edge_contact_buffer_size; detector or pipeline: the "
                "vertex/edge *_pre_alloc parameters), or call "
                "check_and_grow_collision_buffers() between detections "
                "(SolverVBD users: check_and_grow_self_contact_buffers()).",
                stacklevel=2,
            )
        return vt_demand, ee_demand, vt_overflow, ee_overflow

    def check_and_grow_collision_buffers(self, warn: bool = True) -> bool:
        """Check the last detection's overflow flags and grow the result storage.

        Reads the global_pair_counts back (synchronizes the device). When a family
        overflowed, its per-element budget is raised to cover 1.5x the measured
        demand and the storage grows IN PLACE: the bound struct stays the same
        object and only its capacity-sized row arrays are replaced (per-element
        arrays never change size), so every owner of the struct keeps a valid
        reference. The internal append log is resized to match. The grown rows
        are empty until the next detection fills them, so call this between
        detections, not between a detection and a consumer of its results.

        This call must not be captured into a CUDA graph: it synchronizes,
        and growth reallocates arrays. While capture is active it returns
        ``False`` without checking anything. After a ``True`` return, any
        PREVIOUSLY captured graph that contains detection or contact kernels
        is invalid: it still operates on the old row arrays, so replaying it
        silently produces stale results. Re-create such graphs after growth.
        Device-side COPIES of the struct also go stale (``SolverVBD`` keeps
        one for its contact kernels; its
        ``check_and_grow_self_contact_buffers`` wrapper refreshes it).

        Args:
            warn: Emit a ``UserWarning`` describing the overflow (per-family
                demand versus capacity and how to raise the budgets) before
                growing. Pass ``False`` to grow silently and rely on the
                return value instead.

        Returns:
            True if the storage was reallocated.
        """
        if self.device.is_capturing:
            return False
        self._require_collision_info()
        vt_demand, ee_demand, vt_overflow, ee_overflow = self.check_self_contact_overflow(warn=warn)
        if not (vt_overflow or ee_overflow):
            return False

        particle_count = max(self.model.particle_count, 1)
        edge_count = max(self.model.edge_count, 1)
        info = self.collision_info
        if vt_overflow:
            self.vertex_collision_buffer_pre_alloc = max(
                self.vertex_collision_buffer_pre_alloc + 1, -(-3 * vt_demand // (2 * particle_count))
            )
            vt_capacity = self.vertex_collision_buffer_pre_alloc * self.model.particle_count
            info.vertex_colliding_triangles = wp.zeros(
                shape=(max(2 * vt_capacity, 1),), dtype=wp.int32, device=self.device
            )
            if self.record_triangle_contacting_vertices:
                info.triangle_colliding_vertices = wp.zeros(
                    shape=(max(vt_capacity, 1),), dtype=wp.int32, device=self.device
                )
        if ee_overflow:
            self.edge_collision_buffer_pre_alloc = max(
                self.edge_collision_buffer_pre_alloc + 1, -(-3 * ee_demand // (2 * edge_count))
            )
            ee_capacity = self.edge_collision_buffer_pre_alloc * self.model.edge_count
            info.edge_colliding_edges = wp.zeros(shape=(max(2 * ee_capacity, 1),), dtype=wp.int32, device=self.device)

        # leave the rows in the same state a fresh allocation would have:
        # empty rows described by empty offsets/counts, cleared demand cursors
        # (min_dist keeps its last values; detection rewrites it anyway)
        info.global_pair_counts.zero_()
        info.vertex_colliding_triangles_count.zero_()
        info.vertex_colliding_triangles_offsets.zero_()
        info.edge_colliding_edges_count.zero_()
        info.edge_colliding_edges_offsets.zero_()
        if self.record_triangle_contacting_vertices:
            info.triangle_colliding_vertices_count.zero_()
            info.triangle_colliding_vertices_offsets.zero_()

        self._ensure_scratch()
        return True

    def rebuild(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = new_pos

        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[
                self.vertex_positions,
                self.model.tri_indices,
            ],
            outputs=[self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris.rebuild()
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=self.model.world_count + 1,
            inputs=[self.bvh_tris.id, self.bvh_tris_group_roots],
            device=self.model.device,
        )

        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices],
            outputs=[self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.rebuild()
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=self.model.world_count + 1,
            inputs=[self.bvh_edges.id, self.bvh_edges_group_roots],
            device=self.model.device,
        )

    def refit(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = new_pos

        self.refit_triangles()
        self.refit_edges()

    def refit_triangles(self):
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, self.model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris.refit()

    def refit_edges(self):
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.refit()

    def vertex_triangle_collision_detection(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        self._require_collision_info()
        info = self.collision_info
        # clear this family's (cursor, overflow) slice; the other family's slots are untouched
        info.global_pair_counts[0:2].zero_()
        info.triangle_colliding_vertices_min_dist.fill_(max_query_radius)
        if self.record_triangle_contacting_vertices:
            # triangle-keyed reverse lists have many writers, so heads and
            # stored counts need explicit resets (the vertex/edge lists do not:
            # each element's own thread writes its head and count unconditionally)
            self._tri_list_heads.fill_(-1)
            info.triangle_colliding_vertices_count.zero_()
        vt_capacity = info.vertex_colliding_triangles.shape[0] // 2

        wp.launch(
            kernel=vertex_triangle_collision_detection_kernel,
            inputs=[
                max_query_radius,
                min_query_radius,
                self.bvh_tris.id,
                self.bvh_tris_group_roots,
                self.vertex_positions,
                self.model.tri_indices,
                self.model.particle_world,
                self.model.world_count,
                self.vertex_triangle_filtering_list,
                self.vertex_triangle_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
                vt_capacity,
            ],
            outputs=[
                self._pair_scratch,
                info.global_pair_counts,
                self._vt_list_heads,
                self._list_next_scratch,
                info.vertex_colliding_triangles_count,
                info.vertex_colliding_triangles_min_dist,
                # gate the triangle-side outputs on the DETECTOR's flag so a
                # recording-enabled struct served by a non-recording detector
                # is never written without its per-detection resets
                self._tri_list_heads if self.record_triangle_contacting_vertices else self._empty_int32,
                self._tri_list_next_scratch if self.record_triangle_contacting_vertices else self._empty_int32,
                info.triangle_colliding_vertices_count
                if self.record_triangle_contacting_vertices
                else self._empty_int32,
                info.triangle_colliding_vertices_min_dist
                if self.record_triangle_contacting_vertices
                else self._empty_min_dist,
            ],
            dim=self.model.particle_count,
            device=self.model.device,
            block_dim=self._vertex_collision_block_size(),
        )

        self._build_pair_rows(
            fill_self_contact_rows_from_lists,
            info.vertex_colliding_triangles_count,
            info.vertex_colliding_triangles_offsets,
            self._vt_list_heads,
            self._list_next_scratch,
            info.vertex_colliding_triangles,
        )
        if self.record_triangle_contacting_vertices:
            self._build_pair_rows(
                fill_self_contact_reverse_rows_from_lists,
                info.triangle_colliding_vertices_count,
                info.triangle_colliding_vertices_offsets,
                self._tri_list_heads,
                self._tri_list_next_scratch,
                info.triangle_colliding_vertices,
            )

    def _vertex_collision_block_size(self) -> int:
        if self.collision_detection_block_size is None:
            return 16
        return self.collision_detection_block_size

    def edge_edge_collision_detection(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        self._require_collision_info()
        info = self.collision_info
        info.global_pair_counts[2:4].zero_()
        ee_capacity = info.edge_colliding_edges.shape[0] // 2
        wp.launch(
            kernel=edge_colliding_edges_detection_kernel,
            inputs=[
                max_query_radius,
                min_query_radius,
                self.bvh_edges.id,
                self.bvh_edges_group_roots,
                self.vertex_positions,
                self.model.edge_indices,
                self.model.particle_world,
                self.model.world_count,
                self.edge_edge_parallel_epsilon,
                self.edge_filtering_list,
                self.edge_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
                ee_capacity,
            ],
            outputs=[
                self._pair_scratch,
                info.global_pair_counts,
                self._ee_list_heads,
                self._list_next_scratch,
                info.edge_colliding_edges_count,
                info.edge_colliding_edges_min_dist,
            ],
            dim=self.model.edge_count,
            device=self.model.device,
            block_dim=self._edge_collision_block_size(),
        )

        self._build_pair_rows(
            fill_self_contact_rows_from_lists,
            info.edge_colliding_edges_count,
            info.edge_colliding_edges_offsets,
            self._ee_list_heads,
            self._list_next_scratch,
            info.edge_colliding_edges,
        )

    def _edge_collision_block_size(self) -> int:
        if self.collision_detection_block_size is not None:
            return self.collision_detection_block_size

        # The per-edge BVH traversal diverges heavily within a warp. Launches too small to fill the
        # GPU are latency bound and run fastest with few threads per block, while large launches
        # need full warps for throughput. Aim for about 16 blocks per SM, clamped to [8, 32].
        if not self.device.is_cuda:
            return 16
        blocks_per_sm = max(self.model.edge_count / (16 * self.device.sm_count), 1.0)
        return int(min(32, max(8, 2 ** round(math.log2(blocks_per_sm)))))

    def triangle_triangle_intersection_detection(self):
        # resize_flags only reports this on-demand query now; clear it per call
        # so a past overflow does not read as a current one
        self.resize_flags.zero_()
        if self.triangle_intersecting_triangles is None:
            self.triangle_intersecting_triangles = wp.zeros(
                shape=(self.model.tri_count * self.triangle_triangle_collision_buffer_pre_alloc,),
                dtype=wp.int32,
                device=self.device,
            )

        if self.triangle_intersecting_triangles_count is None:
            self.triangle_intersecting_triangles_count = wp.array(
                shape=(self.model.tri_count,), dtype=wp.int32, device=self.device
            )

        if self.triangle_intersecting_triangles_offsets is None:
            buffer_sizes = np.full((self.model.tri_count,), self.triangle_triangle_collision_buffer_pre_alloc)
            offsets = np.zeros((self.model.tri_count + 1,), dtype=np.int32)
            offsets[1:] = np.cumsum(buffer_sizes)

            self.triangle_intersecting_triangles_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=triangle_triangle_collision_detection_kernel,
            inputs=[
                self.bvh_tris.id,
                self.vertex_positions,
                self.model.tri_indices,
                self.triangle_intersecting_triangles_offsets,
            ],
            outputs=[
                self.triangle_intersecting_triangles,
                self.triangle_intersecting_triangles_count,
                self.resize_flags,
            ],
            dim=self.model.tri_count,
            device=self.model.device,
        )
