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

from newton.geometry.kernels import compute_edge_aabbs


@wp.kernel
def compute_sew_v(
    sew_dist: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    vert_indices: wp.array(dtype=wp.int32),
    # outputs
    sew_vinds: wp.array(dtype=wp.vec2i, ndim=2),
    sew_vdists: wp.array(dtype=wp.float32, ndim=2),
):
    v_index = vert_indices[wp.tid()]
    v = pos[v_index]
    lower = wp.vec3(v[0] - sew_dist, v[1] - sew_dist, v[2] - sew_dist)
    upper = wp.vec3(v[0] + sew_dist, v[1] + sew_dist, v[2] + sew_dist)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    edge_index = wp.int32(-1)
    vertex_num_sew = wp.int32(0)
    max_num_sew = sew_vinds.shape[1]
    while wp.bvh_query_next(query, edge_index):
        va_ind = edge_indices[edge_index, 2]
        vb_ind = edge_indices[edge_index, 3]
        if v_index == va_ind or v_index == vb_ind:
            continue
        va = pos[va_ind]
        vb = pos[vb_ind]
        check_va = bool(True)
        check_vb = bool(True)
        for i in range(vertex_num_sew):
            if sew_vinds[wp.tid()][i][1] == va_ind:
                check_va = False
                break

        for i in range(vertex_num_sew):
            if sew_vinds[wp.tid()][i][1] == vb_ind:
                check_vb = False
                break

        if v_index < va_ind and check_va:
            da = wp.length(va - v)
            if da <= sew_dist:
                if vertex_num_sew < max_num_sew:
                    sew_vinds[wp.tid()][vertex_num_sew][0] = v_index
                    sew_vinds[wp.tid()][vertex_num_sew][1] = va_ind
                    sew_vdists[wp.tid()][vertex_num_sew] = da
                    vertex_num_sew = vertex_num_sew + 1
                else:
                    for i in range(max_num_sew):
                        if da < sew_vdists[wp.tid()][i]:
                            sew_vinds[wp.tid()][i][0] = v_index
                            sew_vinds[wp.tid()][i][1] = va_ind
                            sew_vdists[wp.tid()][i] = da
                            break
        if v_index < vb_ind and check_vb:
            db = wp.length(vb - v)
            if db <= sew_dist:
                if vertex_num_sew < max_num_sew:
                    sew_vinds[wp.tid()][vertex_num_sew][0] = v_index
                    sew_vinds[wp.tid()][vertex_num_sew][1] = vb_ind
                    sew_vdists[wp.tid()][vertex_num_sew] = db
                    vertex_num_sew = vertex_num_sew + 1
                else:
                    for i in range(max_num_sew):
                        if db < sew_vdists[wp.tid()][i]:
                            sew_vinds[wp.tid()][i][0] = v_index
                            sew_vinds[wp.tid()][i][1] = vb_ind
                            sew_vdists[wp.tid()][i] = db
                            break


def create_trimesh_sew_springs(
    particle_q,
    edge_indices,
    sew_distance=1.0e-3,
    sew_interior=False,
):
    """
    A function that create sew springs for trimesh.
    It will sew vertices within sew_distance.
    It returns sew spring indices.

    Args:
        sew_distance: Vertices within sew_distance will be connected by springs.
        sew_interior: If True, can sew between interior vertices, otherwise only sew boundary-interior or boundary-boundary vertices

    """

    trimesh_edge_indices = np.array(edge_indices)
    # compute unique vert indices
    flat_inds = trimesh_edge_indices.flatten()
    flat_inds = flat_inds[flat_inds >= 0]
    vert_inds = np.unique(flat_inds)
    # compute unique boundary vert indices
    bound_condition = trimesh_edge_indices[:, 1] < 0
    bound_edge_inds = trimesh_edge_indices[bound_condition]
    bound_edge_inds = bound_edge_inds[:, 2:4]
    bound_vert_inds = np.unique(bound_edge_inds.flatten())
    # compute edge bvh
    num_edge = trimesh_edge_indices.shape[0]
    lower_bounds_edges = wp.array(shape=(num_edge,), dtype=wp.vec3, device="cpu")
    upper_bounds_edges = wp.array(shape=(num_edge,), dtype=wp.vec3, device="cpu")
    wp_edge_indices = wp.array(edge_indices, dtype=wp.int32, device="cpu")
    wp_vert_pos = wp.array(particle_q, dtype=wp.vec3, device="cpu")
    wp.launch(
        kernel=compute_edge_aabbs,
        inputs=[wp_vert_pos, wp_edge_indices],
        outputs=[lower_bounds_edges, upper_bounds_edges],
        dim=num_edge,
        device="cpu",
    )
    bvh_edges = wp.Bvh(lower_bounds_edges, upper_bounds_edges)

    # compute sew springs
    max_num_sew = 5
    if sew_interior:
        num_vert = vert_inds.shape[0]
        wp_vert_inds = wp.array(vert_inds, dtype=wp.int32, device="cpu")
    else:
        num_vert = bound_vert_inds.shape[0]
        wp_vert_inds = wp.array(bound_vert_inds, dtype=wp.int32, device="cpu")

    wp_sew_vinds = wp.full(
        shape=(num_vert, max_num_sew), value=wp.vec2i(-1, -1), dtype=wp.vec2i, device="cpu"
    )  # each vert sew max 5 other verts
    wp_sew_vdists = wp.full(shape=(num_vert, max_num_sew), value=sew_distance, dtype=wp.float32, device="cpu")
    wp.launch(
        kernel=compute_sew_v,
        inputs=[sew_distance, bvh_edges.id, wp_vert_pos, wp_edge_indices, wp_vert_inds],
        outputs=[wp_sew_vinds, wp_sew_vdists],
        dim=num_vert,
        device="cpu",
    )

    np_sew_vinds = wp_sew_vinds.numpy().reshape(num_vert * max_num_sew, 2)
    np_sew_vinds = np_sew_vinds[np_sew_vinds[:, 0] >= 0]

    return np_sew_vinds
