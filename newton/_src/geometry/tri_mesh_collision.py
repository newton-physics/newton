# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..utils.mesh import MeshAdjacency
from .bvh import compute_bvh_group_roots
from .kernels import (
    compute_edge_aabbs,
    compute_edge_groups,
    compute_tri_aabbs,
    compute_tri_groups,
    edge_colliding_edges_detection_kernel,
    init_triangle_collision_data_kernel,
    triangle_triangle_collision_detection_kernel,
    vertex_triangle_collision_detection_kernel,
)

if TYPE_CHECKING:
    from ..sim import Model


@wp.struct
class TriMeshCollisionInfo:
    """Bounded buffers produced by triangle-mesh self-collision queries.

    .. experimental::

        This storage-level result type may change without the normal
        deprecation period while the public self-contact API matures.

    Vertex-triangle and edge-edge results use interleaved source/target pairs;
    triangle-vertex results use plain target indices. Counts record all pairs
    found and may exceed a row's capacity when a buffer overflows. Kernel code
    should therefore read results through the internal ``get_*`` accessors,
    which clamp counts and apply the correct packed indexing.
    """

    vertex_colliding_triangles: wp.array[wp.int32]
    """Interleaved vertex/triangle indices, shape ``[2 * sum(vertex row capacities)]``."""
    vertex_colliding_triangles_offsets: wp.array[wp.int32]
    """Offsets into vertex-triangle rows before interleaved-pair indexing."""
    vertex_colliding_triangles_buffer_sizes: wp.array[wp.int32]
    """Maximum stored collision count for each vertex row."""
    vertex_colliding_triangles_count: wp.array[wp.int32]
    """Detected collision count for each vertex; values may exceed row capacity."""
    vertex_colliding_triangles_min_dist: wp.array[float]
    """Minimum detected vertex-triangle distance for each vertex [m]."""

    triangle_colliding_vertices: wp.array[wp.int32]
    """Vertex indices grouped into rows for each triangle."""
    triangle_colliding_vertices_offsets: wp.array[wp.int32]
    """Offsets into the plain-index triangle-vertex rows."""
    triangle_colliding_vertices_buffer_sizes: wp.array[wp.int32]
    """Maximum stored collision count for each triangle row."""
    triangle_colliding_vertices_count: wp.array[wp.int32]
    """Detected collision count for each triangle; values may exceed row capacity."""
    triangle_colliding_vertices_min_dist: wp.array[float]
    """Minimum detected triangle-vertex distance for each triangle [m]."""

    edge_colliding_edges: wp.array[wp.int32]
    """Interleaved source/target edge indices, shape ``[2 * sum(edge row capacities)]``."""
    edge_colliding_edges_offsets: wp.array[wp.int32]
    """Offsets into edge-edge rows before interleaved-pair indexing."""
    edge_colliding_edges_buffer_sizes: wp.array[wp.int32]
    """Maximum stored collision count for each edge row."""
    edge_colliding_edges_count: wp.array[wp.int32]
    """Detected collision count for each edge; values may exceed row capacity."""
    edge_colliding_edges_min_dist: wp.array[float]
    """Minimum detected edge-edge distance for each edge [m]."""


@wp.func
def get_vertex_colliding_triangles_count(collision_info: TriMeshCollisionInfo, vertex: int):
    """Return the stored collision count for ``vertex``, clamped to capacity."""
    return wp.min(
        collision_info.vertex_colliding_triangles_count[vertex],
        collision_info.vertex_colliding_triangles_buffer_sizes[vertex],
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
    """Return the stored collision count for ``triangle``, clamped to capacity."""
    return wp.min(
        collision_info.triangle_colliding_vertices_count[triangle],
        collision_info.triangle_colliding_vertices_buffer_sizes[triangle],
    )


@wp.func
def get_triangle_colliding_vertices(collision_info: TriMeshCollisionInfo, triangle: int, collision_index: int):
    """Return the vertex index for ``collision_index`` of ``triangle``."""
    offset = collision_info.triangle_colliding_vertices_offsets[triangle]
    return collision_info.triangle_colliding_vertices[offset + collision_index]


@wp.func
def get_edge_colliding_edges_count(collision_info: TriMeshCollisionInfo, edge: int):
    """Return the stored collision count for ``edge``, clamped to capacity."""
    return wp.min(
        collision_info.edge_colliding_edges_count[edge], collision_info.edge_colliding_edges_buffer_sizes[edge]
    )


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


def _compute_collision_buffer_offsets(buffer_sizes: wp.array[wp.int32], offsets: wp.array[wp.int32]):
    """Fill CSR ``offsets`` (size N+1) from per-element ``buffer_sizes`` (size N)."""
    assert offsets.size == buffer_sizes.size + 1
    offsets_np = np.empty(shape=(offsets.size,), dtype=np.int32)
    offsets_np[1:] = np.cumsum(buffer_sizes.numpy())[:]
    offsets_np[0] = 0

    offsets.assign(offsets_np)


def build_tri_mesh_collision_info(
    particle_count: int,
    tri_count: int,
    edge_count: int,
    *,
    vertex_collision_buffer_pre_alloc: int = 16,
    triangle_collision_buffer_pre_alloc: int = 16,
    edge_collision_buffer_pre_alloc: int = 32,
    record_triangle_contacting_vertices: bool = False,
    device=None,
) -> TriMeshCollisionInfo:
    """Allocate all self-contact result arrays into a :class:`TriMeshCollisionInfo`.

    This is the single allocation path for tri-mesh self-contact results:
    :class:`TriMeshCollisionDetector` calls it when no external struct is
    injected, and result-owning containers call it to allocate buffers the
    detector then writes into. Mirrors the adjacency pattern of one
    ``@wp.struct`` plus one builder.

    When ``record_triangle_contacting_vertices`` is ``False`` the
    triangle-side list fields are left at their empty defaults;
    ``triangle_colliding_vertices_min_dist`` is always allocated.

    Args:
        particle_count: Number of mesh vertices.
        tri_count: Number of mesh triangles.
        edge_count: Number of mesh edges.
        vertex_collision_buffer_pre_alloc: Initial collision capacity per vertex.
        triangle_collision_buffer_pre_alloc: Initial collision capacity per triangle.
        edge_collision_buffer_pre_alloc: Initial collision capacity per edge.
        record_triangle_contacting_vertices: Whether to allocate the reverse
            triangle-to-vertex result lists.
        device: Warp device on which to allocate the arrays.

    Returns:
        An allocated collision-result struct.
    """
    info = TriMeshCollisionInfo()

    info.vertex_colliding_triangles = wp.zeros(
        shape=(2 * particle_count * vertex_collision_buffer_pre_alloc,), dtype=wp.int32, device=device
    )
    info.vertex_colliding_triangles_count = wp.zeros(shape=(particle_count,), dtype=wp.int32, device=device)
    info.vertex_colliding_triangles_min_dist = wp.zeros(shape=(particle_count,), dtype=float, device=device)
    info.vertex_colliding_triangles_buffer_sizes = wp.full(
        shape=(particle_count,), value=vertex_collision_buffer_pre_alloc, dtype=wp.int32, device=device
    )
    info.vertex_colliding_triangles_offsets = wp.array(shape=(particle_count + 1,), dtype=wp.int32, device=device)
    _compute_collision_buffer_offsets(
        info.vertex_colliding_triangles_buffer_sizes, info.vertex_colliding_triangles_offsets
    )

    if record_triangle_contacting_vertices:
        info.triangle_colliding_vertices = wp.zeros(
            shape=(tri_count * triangle_collision_buffer_pre_alloc,), dtype=wp.int32, device=device
        )
        info.triangle_colliding_vertices_count = wp.zeros(shape=(tri_count,), dtype=wp.int32, device=device)
        info.triangle_colliding_vertices_buffer_sizes = wp.full(
            shape=(tri_count,), value=triangle_collision_buffer_pre_alloc, dtype=wp.int32, device=device
        )
        info.triangle_colliding_vertices_offsets = wp.array(shape=(tri_count + 1,), dtype=wp.int32, device=device)
        _compute_collision_buffer_offsets(
            info.triangle_colliding_vertices_buffer_sizes, info.triangle_colliding_vertices_offsets
        )

    # needed regardless of whether triangle contacting vertices are recorded
    info.triangle_colliding_vertices_min_dist = wp.zeros(shape=(tri_count,), dtype=float, device=device)

    info.edge_colliding_edges = wp.zeros(
        shape=(2 * edge_count * edge_collision_buffer_pre_alloc,), dtype=wp.int32, device=device
    )
    info.edge_colliding_edges_count = wp.zeros(shape=(edge_count,), dtype=wp.int32, device=device)
    info.edge_colliding_edges_buffer_sizes = wp.full(
        shape=(edge_count,), value=edge_collision_buffer_pre_alloc, dtype=wp.int32, device=device
    )
    info.edge_colliding_edges_offsets = wp.array(shape=(edge_count + 1,), dtype=wp.int32, device=device)
    _compute_collision_buffer_offsets(info.edge_colliding_edges_buffer_sizes, info.edge_colliding_edges_offsets)
    info.edge_colliding_edges_min_dist = wp.zeros(shape=(edge_count,), dtype=float, device=device)

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
        collision_detection_block_size=16,
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

        self.resize_flags = wp.zeros(shape=(4,), dtype=wp.int32, device=self.device)

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
        arrays = (
            (
                "vertex_colliding_triangles",
                collision_info.vertex_colliding_triangles,
                2 * particle_count * self.vertex_collision_buffer_pre_alloc,
                wp.int32,
            ),
            (
                "vertex_colliding_triangles_offsets",
                collision_info.vertex_colliding_triangles_offsets,
                particle_count + 1,
                wp.int32,
            ),
            (
                "vertex_colliding_triangles_buffer_sizes",
                collision_info.vertex_colliding_triangles_buffer_sizes,
                particle_count,
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
            (
                "edge_colliding_edges",
                collision_info.edge_colliding_edges,
                2 * edge_count * self.edge_collision_buffer_pre_alloc,
                wp.int32,
            ),
            (
                "edge_colliding_edges_offsets",
                collision_info.edge_colliding_edges_offsets,
                edge_count + 1,
                wp.int32,
            ),
            (
                "edge_colliding_edges_buffer_sizes",
                collision_info.edge_colliding_edges_buffer_sizes,
                edge_count,
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
                    tri_count * self.triangle_collision_buffer_pre_alloc,
                    wp.int32,
                ),
                (
                    "triangle_colliding_vertices_offsets",
                    collision_info.triangle_colliding_vertices_offsets,
                    tri_count + 1,
                    wp.int32,
                ),
                (
                    "triangle_colliding_vertices_buffer_sizes",
                    collision_info.triangle_colliding_vertices_buffer_sizes,
                    tri_count,
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
    def vertex_colliding_triangles(self):
        return self.collision_info.vertex_colliding_triangles

    @property
    def vertex_colliding_triangles_offsets(self):
        return self.collision_info.vertex_colliding_triangles_offsets

    @property
    def vertex_colliding_triangles_buffer_sizes(self):
        return self.collision_info.vertex_colliding_triangles_buffer_sizes

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
    def triangle_colliding_vertices_buffer_sizes(self):
        return (
            self.collision_info.triangle_colliding_vertices_buffer_sizes
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
    def edge_colliding_edges_buffer_sizes(self):
        return self.collision_info.edge_colliding_edges_buffer_sizes

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

    def compute_collision_buffer_offsets(self, buffer_sizes: wp.array[wp.int32], offsets: wp.array[wp.int32]):
        _compute_collision_buffer_offsets(buffer_sizes, offsets)

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
        self.vertex_colliding_triangles.fill_(-1)

        if self.record_triangle_contacting_vertices:
            wp.launch(
                kernel=init_triangle_collision_data_kernel,
                inputs=[
                    max_query_radius,
                ],
                outputs=[
                    self.triangle_colliding_vertices_count,
                    self.triangle_colliding_vertices_min_dist,
                    self.resize_flags,
                ],
                dim=self.model.tri_count,
                device=self.model.device,
            )
        else:
            self.triangle_colliding_vertices_min_dist.fill_(max_query_radius)

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
                self.vertex_colliding_triangles_offsets,
                self.vertex_colliding_triangles_buffer_sizes,
                self.triangle_colliding_vertices_offsets,
                self.triangle_colliding_vertices_buffer_sizes,
                self.vertex_triangle_filtering_list,
                self.vertex_triangle_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
            ],
            outputs=[
                self.vertex_colliding_triangles,
                self.vertex_colliding_triangles_count,
                self.vertex_colliding_triangles_min_dist,
                self.triangle_colliding_vertices,
                self.triangle_colliding_vertices_count,
                self.triangle_colliding_vertices_min_dist,
                self.resize_flags,
            ],
            dim=self.model.particle_count,
            device=self.model.device,
            block_dim=self.collision_detection_block_size,
        )

    def edge_edge_collision_detection(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        self._require_collision_info()
        self.edge_colliding_edges.fill_(-1)
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
                self.edge_colliding_edges_offsets,
                self.edge_colliding_edges_buffer_sizes,
                self.edge_edge_parallel_epsilon,
                self.edge_filtering_list,
                self.edge_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
            ],
            outputs=[
                self.edge_colliding_edges,
                self.edge_colliding_edges_count,
                self.edge_colliding_edges_min_dist,
                self.resize_flags,
            ],
            dim=self.model.edge_count,
            device=self.model.device,
            block_dim=self.collision_detection_block_size,
        )

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
