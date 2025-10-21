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

import warp as wp

from newton._src.geometry.kernels import (
    triangle_closest_point,
    vertex_adjacent_to_triangle,
)


@wp.func
def line_intersects_aabb(v0: wp.vec3, v1: wp.vec3, lower: wp.vec3, upper: wp.vec3):
    # Slab method
    dir = v1 - v0
    tmin = 0.0
    tmax = 1.0

    for i in range(3):
        if wp.abs(dir[i]) < 1.0e-8:
            # Segment is parallel to slab. Reject if origin not within slab
            if v0[i] < lower[i] or v0[i] > upper[i]:
                return False
        else:
            invD = 1.0 / dir[i]
            t1 = (lower[i] - v0[i]) * invD
            t2 = (upper[i] - v0[i]) * invD

            tmin = wp.max(tmin, wp.min(t1, t2))
            tmax = wp.min(tmax, wp.max(t1, t2))
            if tmax < tmin:
                return False

    return True


@wp.kernel
def compute_tri_aabbs_kernel(
    enlarge: float,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    v1 = pos[tri_indices[t_id, 0]]
    v2 = pos[tri_indices[t_id, 1]]
    v3 = pos[tri_indices[t_id, 2]]

    lower = wp.min(wp.min(v1, v2), v3)
    upper = wp.max(wp.max(v1, v2), v3)

    lower_bounds[t_id] = lower - wp.vec3(enlarge)
    upper_bounds[t_id] = upper + wp.vec3(enlarge)


@wp.kernel
def compute_edge_aabbs_kernel(
    enlarge: float,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    e_id = wp.tid()

    v1 = pos[edge_indices[e_id, 2]]
    v2 = pos[edge_indices[e_id, 3]]

    lower_bounds[e_id] = wp.min(v1, v2) - wp.vec3(enlarge)
    upper_bounds[e_id] = wp.max(v1, v2) + wp.vec3(enlarge)


@wp.kernel
def aabb_vs_aabb_kernel(
    bvh_id: wp.uint64,
    query_list_rows: int,
    query_radius: float,
    ignore_self_hits: bool,
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
    # outputs
    query_results: wp.array(dtype=int, ndim=2),
):
    tid = wp.int32(wp.tid())
    lower = lower_bounds[tid] - wp.vec3(query_radius)
    upper = upper_bounds[tid] + wp.vec3(query_radius)

    query_count = wp.int32(0)
    query_index = wp.int32(-1)
    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    while (query_count < query_list_rows - 1) and wp.bvh_query_next(query, query_index):
        if not (ignore_self_hits and query_index <= tid):
            query_results[query_count + 1, tid] = query_index
            query_count += 1

    query_results[0, tid] = query_count


@wp.kernel
def aabb_vs_line_kernel(
    bvh_id: wp.uint64,
    query_list_rows: int,
    ignore_self_hits: bool,
    vertices: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
    # outputs
    query_results: wp.array(dtype=int, ndim=2),
):
    eid = wp.int32(wp.tid())
    v1 = vertices[edge_indices[eid, 2]]
    v2 = vertices[edge_indices[eid, 3]]

    query_count = wp.int32(0)
    query_index = wp.int32(-1)
    query = wp.bvh_query_ray(bvh_id, v1, v2 - v1)

    while (query_count < query_list_rows - 1) and wp.bvh_query_next(query, query_index):
        if not (ignore_self_hits and query_index <= eid):
            if line_intersects_aabb(v1, v2, lower_bounds[query_index], upper_bounds[query_index]):
                query_results[query_count + 1, eid] = query_index
                query_count += 1

    query_results[0, eid] = query_count


@wp.kernel
def triangle_vs_point_kernel(
    bvh_id: wp.uint64,
    query_list_rows: int,
    query_radius: float,
    max_dist: float,
    ignore_self_hits: bool,
    pos: wp.array(dtype=wp.vec3),
    tri_pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=int, ndim=2),
    # outputs
    query_results: wp.array(dtype=int, ndim=2),
):
    vid = wp.tid()

    x0 = pos[vid]
    lower = x0 - wp.vec3(query_radius)
    upper = x0 + wp.vec3(query_radius)

    tri_index = wp.int32(-1)
    query_count = wp.int32(0)
    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    while (query_count < query_list_rows - 1) and wp.bvh_query_next(query, tri_index):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]
        if ignore_self_hits and vertex_adjacent_to_triangle(vid, t1, t2, t3):
            continue

        closest_p, _bary, _feature_type = triangle_closest_point(tri_pos[t1], tri_pos[t2], tri_pos[t3], x0)

        dist = wp.length(closest_p - x0)

        if dist < max_dist:
            query_results[query_count + 1, vid] = tri_index
            query_count += 1

    query_results[0, vid] = query_count


@wp.kernel
def edge_vs_edge_kernel(
    bvh_id: wp.uint64,
    query_list_rows: int,
    query_radius: float,
    max_dist: float,
    ignore_self_hits: bool,
    test_pos: wp.array(dtype=wp.vec3),
    test_edge_indices: wp.array(dtype=int, ndim=2),
    edge_pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=int, ndim=2),
    # outputs
    query_results: wp.array(dtype=int, ndim=2),
):
    eid = wp.int32(wp.tid())

    v0 = test_edge_indices[eid, 2]
    v1 = test_edge_indices[eid, 3]

    x0 = test_pos[v0]
    x1 = test_pos[v1]

    lower = wp.min(x0, x1) - wp.vec3(query_radius)
    upper = wp.max(x0, x1) + wp.vec3(query_radius)

    edge_index = wp.int32(-1)
    query_count = wp.int32(0)
    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    while (query_count < query_list_rows - 1) and wp.bvh_query_next(query, edge_index):
        if ignore_self_hits and edge_index <= eid:
            continue
        v2 = edge_indices[edge_index, 2]
        v3 = edge_indices[edge_index, 3]
        if ignore_self_hits and (v0 == v2 or v0 == v3 or v1 == v2 or v1 == v3):
            continue

        x2, x3 = edge_pos[v2], edge_pos[v3]
        edge_edge_parallel_epsilon = wp.float32(1e-5)
        st = wp.closest_point_edge_edge(x0, x1, x2, x3, edge_edge_parallel_epsilon)
        s = st[0]
        t = st[1]
        c1 = wp.lerp(x0, x1, s)
        c2 = wp.lerp(x2, x3, t)
        dist = wp.length(c1 - c2)

        if dist < max_dist:
            query_results[query_count + 1, eid] = edge_index
            query_count += 1

    query_results[0, eid] = query_count
