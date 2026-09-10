# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
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
    count_self_contact_pair_rows,
    edge_colliding_edges_detection_kernel,
    fill_self_contact_pair_rows,
    finalize_row_offsets,
    triangle_triangle_collision_detection_kernel,
    vertex_triangle_collision_detection_kernel,
)

if TYPE_CHECKING:
    from ..sim import Model

# cap for the strided CSR-build launches: dims stay host-static (graph-safe),
# threads loop when a grown pair array exceeds the cap
_CSR_BUILD_MAX_LAUNCH_DIM = 2**21


@wp.struct
class TriMeshCollisionInfo:
    """Results of triangle-mesh self-collision queries.

    .. experimental::

        This storage-level result type may change without the normal
        deprecation period while the public self-contact API matures.

    Detection appends every found pair to one shared array per family
    (``vt_pairs`` for vertex-triangle, ``ee_pairs`` for edge-edge) through the
    cursors in ``counters``; memory scales with the actual contact count and a
    hot element cannot overflow a private budget. The per-element row arrays
    (``vertex_colliding_triangles``, ``edge_colliding_edges``) are exact CSR
    tables over the pair arrays, rebuilt after each detection: row ``i`` of a
    family lists the indices of the pairs owned by element ``i``. Row order
    within an element is scheduling-dependent. Counts record all pairs found
    and may exceed what fit into a pair array when it overflows (the matching
    ``counters`` overflow flag is set). Kernel code should read results through
    the internal ``get_*`` accessors.
    """

    vt_pairs: wp.array[wp.vec2i]
    """Shared (vertex, triangle) pair records, valid in ``[0, min(counters[VT_PAIR_CURSOR], capacity))``."""
    ee_pairs: wp.array[wp.vec2i]
    """Shared (edge, colliding edge) pair records, both directions, valid below the cursor."""
    counters: wp.array[wp.int32]
    """Pair cursors and overflow flags: [vt cursor, vt overflow, ee cursor, ee overflow]."""

    vertex_colliding_triangles: wp.array[wp.int32]
    """CSR row values for vertices: indices into ``vt_pairs``."""
    vertex_colliding_triangles_offsets: wp.array[wp.int32]
    """CSR row offsets for vertices, shape ``[particle_count + 1]``, exact."""
    vertex_colliding_triangles_count: wp.array[wp.int32]
    """Detected collision count for each vertex; may exceed the stored count on overflow."""
    vertex_colliding_triangles_min_dist: wp.array[float]
    """Minimum detected vertex-triangle distance for each vertex [m]."""

    triangle_colliding_vertices: wp.array[wp.int32]
    """Optional CSR row values for triangles (indices into ``vt_pairs``); empty unless recording is enabled."""
    triangle_colliding_vertices_offsets: wp.array[wp.int32]
    """Optional CSR row offsets for triangles, shape ``[tri_count + 1]``."""
    triangle_colliding_vertices_count: wp.array[wp.int32]
    """Optional stored collision count for each triangle."""
    triangle_colliding_vertices_min_dist: wp.array[float]
    """Minimum detected triangle-vertex distance for each triangle [m]."""

    edge_colliding_edges: wp.array[wp.int32]
    """CSR row values for edges: indices into ``ee_pairs``."""
    edge_colliding_edges_offsets: wp.array[wp.int32]
    """CSR row offsets for edges, shape ``[edge_count + 1]``, exact."""
    edge_colliding_edges_count: wp.array[wp.int32]
    """Detected collision count for each edge; may exceed the stored count on overflow."""
    edge_colliding_edges_min_dist: wp.array[float]
    """Minimum detected edge-edge distance for each edge [m]."""

    _vertex_row_cursors: wp.array[wp.int32]
    """Internal scratch for the CSR fill pass (per-vertex write cursors)."""
    _triangle_row_cursors: wp.array[wp.int32]
    """Internal scratch for the optional triangle-side CSR fill pass."""
    _edge_row_cursors: wp.array[wp.int32]
    """Internal scratch for the CSR fill pass (per-edge write cursors)."""


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
    pair_index = collision_info.vertex_colliding_triangles[offset + collision_index]
    return collision_info.vt_pairs[pair_index][1]


@wp.func
def get_vertex_collision_buffer_vertex_index(collision_info: TriMeshCollisionInfo, vertex: int, collision_index: int):
    """Return the stored source vertex for ``collision_index`` of ``vertex``."""
    offset = collision_info.vertex_colliding_triangles_offsets[vertex]
    pair_index = collision_info.vertex_colliding_triangles[offset + collision_index]
    return collision_info.vt_pairs[pair_index][0]


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
    pair_index = collision_info.triangle_colliding_vertices[offset + collision_index]
    return collision_info.vt_pairs[pair_index][0]


@wp.func
def get_edge_colliding_edges_count(collision_info: TriMeshCollisionInfo, edge: int):
    """Return the stored collision count for ``edge`` (exact CSR row length)."""
    return (
        collision_info.edge_colliding_edges_offsets[edge + 1] - collision_info.edge_colliding_edges_offsets[edge]
    )


@wp.func
def get_edge_colliding_edges(collision_info: TriMeshCollisionInfo, edge: int, collision_index: int):
    """Return the target edge for ``collision_index`` of ``edge``."""
    offset = collision_info.edge_colliding_edges_offsets[edge]
    pair_index = collision_info.edge_colliding_edges[offset + collision_index]
    return collision_info.ee_pairs[pair_index][1]


@wp.func
def get_edge_collision_buffer_edge_index(collision_info: TriMeshCollisionInfo, edge: int, collision_index: int):
    """Return the stored source edge for ``collision_index`` of ``edge``."""
    offset = collision_info.edge_colliding_edges_offsets[edge]
    pair_index = collision_info.edge_colliding_edges[offset + collision_index]
    return collision_info.ee_pairs[pair_index][0]


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
    :class:`TriMeshCollisionDetector` calls it when no external struct is
    injected, and result-owning containers call it to allocate buffers the
    detector then writes into.

    The ``*_pre_alloc`` values are average contact budgets per element: the
    shared pair arrays hold ``pre_alloc x element_count`` records that any
    element can draw from, so a locally dense fold only overflows when the
    whole mesh's contact demand exceeds the pool.

    When ``record_triangle_contacting_vertices`` is ``False`` the
    triangle-side CSR fields are left at their empty defaults;
    ``triangle_colliding_vertices_min_dist`` is always allocated.

    Args:
        particle_count: Number of mesh vertices.
        tri_count: Number of mesh triangles.
        edge_count: Number of mesh edges.
        vertex_collision_buffer_pre_alloc: Average vertex-triangle contact
            budget per vertex; ``vt_pairs`` capacity = this x ``particle_count``.
        triangle_collision_buffer_pre_alloc: Unused for sizing (the reverse
            table indexes the same ``vt_pairs``); kept for signature stability.
        edge_collision_buffer_pre_alloc: Average edge-edge contact budget per
            edge; ``ee_pairs`` capacity = this x ``edge_count``.
        record_triangle_contacting_vertices: Whether to allocate the reverse
            triangle-to-vertex CSR table.
        device: Warp device on which to allocate the arrays.

    Returns:
        An allocated collision-result struct.
    """
    info = TriMeshCollisionInfo()

    vt_capacity = vertex_collision_buffer_pre_alloc * particle_count
    ee_capacity = edge_collision_buffer_pre_alloc * edge_count

    info.vt_pairs = wp.empty(shape=(max(vt_capacity, 1),), dtype=wp.vec2i, device=device)
    info.ee_pairs = wp.empty(shape=(max(ee_capacity, 1),), dtype=wp.vec2i, device=device)
    info.counters = wp.zeros(shape=(4,), dtype=wp.int32, device=device)

    info.vertex_colliding_triangles = wp.zeros(shape=(max(vt_capacity, 1),), dtype=wp.int32, device=device)
    info.vertex_colliding_triangles_offsets = wp.zeros(shape=(particle_count + 1,), dtype=wp.int32, device=device)
    info.vertex_colliding_triangles_count = wp.zeros(shape=(particle_count,), dtype=wp.int32, device=device)
    info.vertex_colliding_triangles_min_dist = wp.zeros(shape=(particle_count,), dtype=float, device=device)
    info._vertex_row_cursors = wp.zeros(shape=(particle_count,), dtype=wp.int32, device=device)

    if record_triangle_contacting_vertices:
        info.triangle_colliding_vertices = wp.zeros(shape=(max(vt_capacity, 1),), dtype=wp.int32, device=device)
        info.triangle_colliding_vertices_offsets = wp.zeros(shape=(tri_count + 1,), dtype=wp.int32, device=device)
        info.triangle_colliding_vertices_count = wp.zeros(shape=(tri_count,), dtype=wp.int32, device=device)
        info._triangle_row_cursors = wp.zeros(shape=(tri_count,), dtype=wp.int32, device=device)

    # needed regardless of whether triangle contacting vertices are recorded
    info.triangle_colliding_vertices_min_dist = wp.zeros(shape=(tri_count,), dtype=float, device=device)

    info.edge_colliding_edges = wp.zeros(shape=(max(ee_capacity, 1),), dtype=wp.int32, device=device)
    info.edge_colliding_edges_offsets = wp.zeros(shape=(edge_count + 1,), dtype=wp.int32, device=device)
    info.edge_colliding_edges_count = wp.zeros(shape=(edge_count,), dtype=wp.int32, device=device)
    info.edge_colliding_edges_min_dist = wp.zeros(shape=(edge_count,), dtype=float, device=device)
    info._edge_row_cursors = wp.zeros(shape=(edge_count,), dtype=wp.int32, device=device)

    return info


class TriMeshCollisionDetector:
    def __init__(
        self,
        model: Model,
        record_triangle_contacting_vertices=False,
        vertex_positions=None,
        vertex_collision_buffer_pre_alloc=16,
        vertex_collision_buffer_max_alloc=256,
        vertex_triangle_filtering_list=None,
        vertex_triangle_filtering_list_offsets=None,
        triangle_collision_buffer_pre_alloc=16,
        triangle_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=32,
        edge_collision_buffer_max_alloc=256,
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
    ):
        self.model = model
        self.record_triangle_contacting_vertices = record_triangle_contacting_vertices
        self.vertex_positions = model.particle_q if vertex_positions is None else vertex_positions
        self.device = model.device
        self.vertex_collision_buffer_pre_alloc = vertex_collision_buffer_pre_alloc
        self.vertex_collision_buffer_max_alloc = vertex_collision_buffer_max_alloc
        self.triangle_collision_buffer_pre_alloc = triangle_collision_buffer_pre_alloc
        self.triangle_collision_buffer_max_alloc = triangle_collision_buffer_max_alloc
        self.edge_collision_buffer_pre_alloc = edge_collision_buffer_pre_alloc
        self.edge_collision_buffer_max_alloc = edge_collision_buffer_max_alloc
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
        # buffers now; self-contact overflow lives in collision_info.counters
        self.resize_flags = wp.zeros(shape=(4,), dtype=wp.int32, device=self.device)
        # stand-in for the optional per-triangle min-dist output when the
        # triangle-side recording is off (parity with the historical behavior:
        # the array then keeps its constant query-radius fill)
        self._empty_min_dist = wp.empty(shape=(0,), dtype=float, device=self.device)

        # data for triangle-triangle intersection; they will only be initialized on demand, as triangle-triangle intersection is not needed for simulation
        self.triangle_intersecting_triangles = None
        self.triangle_intersecting_triangles_count = None
        self.triangle_intersecting_triangles_offsets = None

    def _validate_collision_info(self, collision_info: TriMeshCollisionInfo) -> None:
        """Validate externally owned result buffers against this detector."""

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
        vt_capacity = max(self.vertex_collision_buffer_pre_alloc * particle_count, 1)
        ee_capacity = max(self.edge_collision_buffer_pre_alloc * edge_count, 1)
        arrays = (
            ("vt_pairs", collision_info.vt_pairs, vt_capacity, wp.vec2i),
            ("ee_pairs", collision_info.ee_pairs, ee_capacity, wp.vec2i),
            ("counters", collision_info.counters, 4, wp.int32),
            ("vertex_colliding_triangles", collision_info.vertex_colliding_triangles, vt_capacity, wp.int32),
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
            ("_vertex_row_cursors", collision_info._vertex_row_cursors, particle_count, wp.int32),
            (
                "triangle_colliding_vertices_min_dist",
                collision_info.triangle_colliding_vertices_min_dist,
                tri_count,
                wp.float32,
            ),
            ("edge_colliding_edges", collision_info.edge_colliding_edges, ee_capacity, wp.int32),
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
            ("_edge_row_cursors", collision_info._edge_row_cursors, edge_count, wp.int32),
        )
        if self.record_triangle_contacting_vertices:
            arrays += (
                (
                    "triangle_colliding_vertices",
                    collision_info.triangle_colliding_vertices,
                    vt_capacity,
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
                ("_triangle_row_cursors", collision_info._triangle_row_cursors, tri_count, wp.int32),
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
    def vt_pairs(self):
        return self.collision_info.vt_pairs

    @property
    def ee_pairs(self):
        return self.collision_info.ee_pairs

    @property
    def counters(self):
        return self.collision_info.counters

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

    def _build_pair_rows(self, pairs, cursor_slot, pair_capacity, owner_component, row_counts, row_offsets, row_cursors, row_values):
        """Build one family's exact CSR over its shared pair array.

        Three graph-capturable passes: count stored pairs per owning element,
        exclusive-scan the counts into the row offsets, then scatter each
        pair's index into its owner's row. Counting reads only stored pairs
        (bounded by capacity), so rows stay consistent when detection dropped
        records on overflow.
        """
        element_count = row_counts.shape[0]
        if element_count == 0:
            return
        build_dim = min(pair_capacity, _CSR_BUILD_MAX_LAUNCH_DIM)
        row_counts.zero_()
        wp.launch(
            kernel=count_self_contact_pair_rows,
            dim=build_dim,
            inputs=[pairs, self.collision_info.counters, cursor_slot, pair_capacity, owner_component, build_dim],
            outputs=[row_counts],
            device=self.device,
        )
        warp.utils.array_scan(row_counts, row_offsets[:element_count], inclusive=False)
        wp.launch(
            kernel=finalize_row_offsets,
            dim=1,
            inputs=[row_counts],
            outputs=[row_offsets],
            device=self.device,
        )
        row_cursors.zero_()
        wp.launch(
            kernel=fill_self_contact_pair_rows,
            dim=build_dim,
            inputs=[
                pairs,
                self.collision_info.counters,
                cursor_slot,
                pair_capacity,
                owner_component,
                build_dim,
                row_offsets,
            ],
            outputs=[row_cursors, row_values],
            device=self.device,
        )

    def check_self_contact_overflow(self, warn: bool = True) -> tuple[int, int, bool, bool]:
        """Read back pair demand and overflow flags (synchronizes the device).

        Returns ``(vt_demand, ee_demand, vt_overflowed, ee_overflowed)``. The
        demands are the total pair counts detection tried to store; when an
        overflow flag is set the corresponding pair array kept only its first
        ``capacity`` records and the CSR rows cover only those.
        """
        counters = self.collision_info.counters.numpy()
        vt_demand, vt_overflow = int(counters[0]), bool(counters[1])
        ee_demand, ee_overflow = int(counters[2]), bool(counters[3])
        if warn and (vt_overflow or ee_overflow):
            import warnings

            warnings.warn(
                f"tri-mesh self-contact pair arrays overflowed "
                f"(vertex-triangle demand {vt_demand} / capacity {self.vt_pairs.shape[0]}, "
                f"edge-edge demand {ee_demand} / capacity {self.ee_pairs.shape[0]}); "
                "excess contacts were dropped this detection. Increase the "
                "*_collision_buffer_pre_alloc budgets, or rely on the solver's "
                "automatic growth outside CUDA graph capture.",
                stacklevel=2,
            )
        return vt_demand, ee_demand, vt_overflow, ee_overflow

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
        info.counters[0:2].zero_()
        info.triangle_colliding_vertices_min_dist.fill_(max_query_radius)
        vt_capacity = info.vt_pairs.shape[0]

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
                info.vt_pairs,
                info.counters,
                info.vertex_colliding_triangles_count,
                info.vertex_colliding_triangles_min_dist,
                info.triangle_colliding_vertices_min_dist
                if self.record_triangle_contacting_vertices
                else self._empty_min_dist,
            ],
            dim=self.model.particle_count,
            device=self.model.device,
            block_dim=self._vertex_collision_block_size(),
        )

        self._build_pair_rows(
            info.vt_pairs,
            0,  # VT_PAIR_CURSOR
            vt_capacity,
            0,
            info._vertex_row_cursors,
            info.vertex_colliding_triangles_offsets,
            info._vertex_row_cursors,
            info.vertex_colliding_triangles,
        )
        if self.record_triangle_contacting_vertices:
            self._build_pair_rows(
                info.vt_pairs,
                0,  # VT_PAIR_CURSOR
                vt_capacity,
                1,
                info.triangle_colliding_vertices_count,
                info.triangle_colliding_vertices_offsets,
                info._triangle_row_cursors,
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
        info.counters[2:4].zero_()
        ee_capacity = info.ee_pairs.shape[0]
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
                info.ee_pairs,
                info.counters,
                info.edge_colliding_edges_count,
                info.edge_colliding_edges_min_dist,
            ],
            dim=self.model.edge_count,
            device=self.model.device,
            block_dim=self._edge_collision_block_size(),
        )

        self._build_pair_rows(
            info.ee_pairs,
            2,  # EE_PAIR_CURSOR
            ee_capacity,
            0,
            info._edge_row_cursors,
            info.edge_colliding_edges_offsets,
            info._edge_row_cursors,
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
