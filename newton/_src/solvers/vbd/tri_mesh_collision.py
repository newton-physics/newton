# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warp as wp

from ...geometry.kernels import (
    compute_edge_aabbs,
    compute_tri_aabbs,
    edge_colliding_edges_detection_kernel,
    init_triangle_collision_data_kernel,
    triangle_triangle_collision_detection_kernel,
    vertex_triangle_collision_detection_kernel,
)
from ...sim import Model


@wp.struct
class TriMeshCollisionInfo:
    # size: 2 x sum(vertex_colliding_triangles_buffer_sizes)
    # every two elements records the vertex index and a triangle index it collides to
    vertex_colliding_triangles: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_min_dist: wp.array(dtype=float)

    triangle_colliding_vertices: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_offsets: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_buffer_sizes: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_min_dist: wp.array(dtype=float)

    # size: 2 x sum(edge_colliding_edges_buffer_sizes)
    # every two elements records the edge index and an edge index it collides to
    edge_colliding_edges: wp.array(dtype=wp.int32)
    edge_colliding_edges_offsets: wp.array(dtype=wp.int32)
    edge_colliding_edges_buffer_sizes: wp.array(dtype=wp.int32)
    edge_colliding_edges_count: wp.array(dtype=wp.int32)
    edge_colliding_edges_min_dist: wp.array(dtype=float)


@wp.func
def get_vertex_colliding_triangles_count(col_info: TriMeshCollisionInfo, v: int):
    return wp.min(col_info.vertex_colliding_triangles_count[v], col_info.vertex_colliding_triangles_buffer_sizes[v])


@wp.func
def get_vertex_colliding_triangles(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision) + 1]


@wp.func
def get_vertex_collision_buffer_vertex_index(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision)]


@wp.func
def get_triangle_colliding_vertices_count(col_info: TriMeshCollisionInfo, tri: int):
    return wp.min(
        col_info.triangle_colliding_vertices_count[tri], col_info.triangle_colliding_vertices_buffer_sizes[tri]
    )


@wp.func
def get_triangle_colliding_vertices(col_info: TriMeshCollisionInfo, tri: int, i_collision: int):
    offset = col_info.triangle_colliding_vertices_offsets[tri]
    return col_info.triangle_colliding_vertices[offset + i_collision]


@wp.func
def get_edge_colliding_edges_count(col_info: TriMeshCollisionInfo, e: int):
    return wp.min(col_info.edge_colliding_edges_count[e], col_info.edge_colliding_edges_buffer_sizes[e])


@wp.func
def get_edge_colliding_edges(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision) + 1]


@wp.func
def get_edge_collision_buffer_edge_index(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision)]


class TriMeshCollisionDetector:
    def __init__(
        self,
        model: Model,
        record_triangle_contacting_vertices=False,
        vertex_positions=None,
        vertex_collision_buffer_pre_alloc=8,
        vertex_collision_buffer_max_alloc=256,
        vertex_triangle_filtering_list=None,
        vertex_triangle_filtering_list_offsets=None,
        triangle_collision_buffer_pre_alloc=16,
        triangle_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=8,
        edge_collision_buffer_max_alloc=256,
        edge_filtering_list=None,
        edge_filtering_list_offsets=None,
        triangle_triangle_collision_buffer_pre_alloc=8,
        triangle_triangle_collision_buffer_max_alloc=256,
        edge_edge_parallel_epsilon=1e-5,
        collision_detection_block_size=16,
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

        self.collision_detection_block_size = collision_detection_block_size

        self.lower_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=model.tri_count,
            device=model.device,
        )

        self.bvh_tris = wp.Bvh(self.lower_bounds_tris, self.upper_bounds_tris)

        # collision detections results

        # vertex collision buffers
        self.vertex_colliding_triangles = wp.zeros(
            shape=(2 * model.particle_count * self.vertex_collision_buffer_pre_alloc,),
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_count = wp.array(
            shape=(model.particle_count,), dtype=wp.int32, device=self.device
        )
        self.vertex_colliding_triangles_min_dist = wp.array(
            shape=(model.particle_count,), dtype=float, device=self.device
        )
        self.vertex_colliding_triangles_buffer_sizes = wp.full(
            shape=(model.particle_count,),
            value=self.vertex_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_offsets = wp.array(
            shape=(model.particle_count + 1,), dtype=wp.int32, device=self.device
        )
        self.compute_collision_buffer_offsets(
            self.vertex_colliding_triangles_buffer_sizes, self.vertex_colliding_triangles_offsets
        )

        if record_triangle_contacting_vertices:
            # triangle collision buffers
            self.triangle_colliding_vertices = wp.zeros(
                shape=(model.tri_count * self.triangle_collision_buffer_pre_alloc,), dtype=wp.int32, device=self.device
            )
            self.triangle_colliding_vertices_count = wp.zeros(
                shape=(model.tri_count,), dtype=wp.int32, device=self.device
            )
            self.triangle_colliding_vertices_buffer_sizes = wp.full(
                shape=(model.tri_count,),
                value=self.triangle_collision_buffer_pre_alloc,
                dtype=wp.int32,
                device=self.device,
            )

            self.triangle_colliding_vertices_offsets = wp.array(
                shape=(model.tri_count + 1,), dtype=wp.int32, device=self.device
            )
            self.compute_collision_buffer_offsets(
                self.triangle_colliding_vertices_buffer_sizes, self.triangle_colliding_vertices_offsets
            )
        else:
            self.triangle_colliding_vertices = None
            self.triangle_colliding_vertices_count = None
            self.triangle_colliding_vertices_buffer_sizes = None
            self.triangle_colliding_vertices_offsets = None

        # this is need regardless of whether we record triangle contacting vertices
        self.triangle_colliding_vertices_min_dist = wp.array(shape=(model.tri_count,), dtype=float, device=self.device)

        # edge collision buffers
        self.edge_colliding_edges = wp.zeros(
            shape=(2 * model.edge_count * self.edge_collision_buffer_pre_alloc,), dtype=wp.int32, device=self.device
        )
        self.edge_colliding_edges_count = wp.zeros(shape=(model.edge_count,), dtype=wp.int32, device=self.device)
        self.edge_colliding_edges_buffer_sizes = wp.full(
            shape=(model.edge_count,),
            value=self.edge_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.edge_colliding_edges_offsets = wp.array(shape=(model.edge_count + 1,), dtype=wp.int32, device=self.device)
        self.compute_collision_buffer_offsets(self.edge_colliding_edges_buffer_sizes, self.edge_colliding_edges_offsets)
        self.edge_colliding_edges_min_dist = wp.array(shape=(model.edge_count,), dtype=float, device=self.device)

        self.lower_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=model.edge_count,
            device=model.device,
        )

        self.bvh_edges = wp.Bvh(self.lower_bounds_edges, self.upper_bounds_edges)

        self.resize_flags = wp.zeros(shape=(4,), dtype=wp.int32, device=self.device)

        self.collision_info = self.get_collision_data()

        # data for triangle-triangle intersection; they will only be initialized on demand, as triangle-triangle intersection is not needed for simulation
        self.triangle_intersecting_triangles = None
        self.triangle_intersecting_triangles_count = None
        self.triangle_intersecting_triangles_offsets = None

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

    def get_collision_data(self):
        collision_info = TriMeshCollisionInfo()

        collision_info.vertex_colliding_triangles = self.vertex_colliding_triangles
        collision_info.vertex_colliding_triangles_offsets = self.vertex_colliding_triangles_offsets
        collision_info.vertex_colliding_triangles_buffer_sizes = self.vertex_colliding_triangles_buffer_sizes
        collision_info.vertex_colliding_triangles_count = self.vertex_colliding_triangles_count
        collision_info.vertex_colliding_triangles_min_dist = self.vertex_colliding_triangles_min_dist

        if self.record_triangle_contacting_vertices:
            collision_info.triangle_colliding_vertices = self.triangle_colliding_vertices
            collision_info.triangle_colliding_vertices_offsets = self.triangle_colliding_vertices_offsets
            collision_info.triangle_colliding_vertices_buffer_sizes = self.triangle_colliding_vertices_buffer_sizes
            collision_info.triangle_colliding_vertices_count = self.triangle_colliding_vertices_count

        collision_info.triangle_colliding_vertices_min_dist = self.triangle_colliding_vertices_min_dist

        collision_info.edge_colliding_edges = self.edge_colliding_edges
        collision_info.edge_colliding_edges_offsets = self.edge_colliding_edges_offsets
        collision_info.edge_colliding_edges_buffer_sizes = self.edge_colliding_edges_buffer_sizes
        collision_info.edge_colliding_edges_count = self.edge_colliding_edges_count
        collision_info.edge_colliding_edges_min_dist = self.edge_colliding_edges_min_dist

        return collision_info

    def compute_collision_buffer_offsets(
        self, buffer_sizes: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32)
    ):
        assert offsets.size == buffer_sizes.size + 1
        offsets_np = np.empty(shape=(offsets.size,), dtype=np.int32)
        offsets_np[1:] = np.cumsum(buffer_sizes.numpy())[:]
        offsets_np[0] = 0

        offsets.assign(offsets_np)

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
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices],
            outputs=[self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.rebuild()

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
                self.vertex_positions,
                self.model.tri_indices,
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
        self.edge_colliding_edges.fill_(-1)
        wp.launch(
            kernel=edge_colliding_edges_detection_kernel,
            inputs=[
                max_query_radius,
                min_query_radius,
                self.bvh_edges.id,
                self.vertex_positions,
                self.model.edge_indices,
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
