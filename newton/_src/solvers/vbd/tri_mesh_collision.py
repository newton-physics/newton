# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# The tri-mesh collision module moved to newton/_src/geometry/tri_mesh_collision.py
# so that the sim layer (CollisionPipeline, Contacts) can reference it without
# importing from the solvers layer. This module re-exports the moved names.

from ...geometry.tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
    build_edge_n_ring_edge_collision_filter,
    build_vertex_n_ring_tris_collision_filter,
    get_edge_colliding_edges,
    get_edge_colliding_edges_count,
    get_edge_collision_buffer_edge_index,
    get_triangle_colliding_vertices,
    get_triangle_colliding_vertices_count,
    get_vertex_colliding_triangles,
    get_vertex_colliding_triangles_count,
    get_vertex_collision_buffer_vertex_index,
    leq_n_ring_vertices,
    one_ring_vertices,
    set_to_csr,
)

__all__ = [
    "TriMeshCollisionDetector",
    "TriMeshCollisionInfo",
    "build_edge_n_ring_edge_collision_filter",
    "build_vertex_n_ring_tris_collision_filter",
    "get_edge_colliding_edges",
    "get_edge_colliding_edges_count",
    "get_edge_collision_buffer_edge_index",
    "get_triangle_colliding_vertices",
    "get_triangle_colliding_vertices_count",
    "get_vertex_colliding_triangles",
    "get_vertex_colliding_triangles_count",
    "get_vertex_collision_buffer_vertex_index",
    "leq_n_ring_vertices",
    "one_ring_vertices",
    "set_to_csr",
]
