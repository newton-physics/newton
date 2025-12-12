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

vec8f = wp.types.vector(length=8, dtype=wp.float32)


def get_mc_tables(device):
    edge_to_verts = np.array(
        [
            [0, 1],  # 0
            [1, 2],  # 1
            [2, 3],  # 2
            [3, 0],  # 3
            [4, 5],  # 4
            [5, 6],  # 5
            [6, 7],  # 6
            [7, 4],  # 7
            [0, 4],  # 8
            [1, 5],  # 9
            [2, 6],  # 10
            [3, 7],  # 11
        ]
    )

    tri_range_table = wp.marching_cubes._get_mc_case_to_tri_range_table(device)
    tri_local_inds_table = wp.marching_cubes._get_mc_tri_local_inds_table(device)
    corner_offsets_table = wp.array(wp.marching_cubes.mc_cube_corner_offsets, dtype=wp.vec3ub, device=device)
    edge_to_verts_table = wp.array(edge_to_verts, dtype=wp.vec2ub, device=device)

    # Create flattened table:
    # Instead of tri_local_inds_table[i] -> edge_to_verts_table[edge_idx, 0/1],
    # we directly map tri_local_inds_table[i] -> vec2i(v_from, v_to)
    tri_local_inds_np = tri_local_inds_table.numpy()
    flat_edge_verts = np.zeros((len(tri_local_inds_np), 2), dtype=np.uint8)

    for i, edge_idx in enumerate(tri_local_inds_np):
        flat_edge_verts[i, 0] = edge_to_verts[edge_idx, 0]
        flat_edge_verts[i, 1] = edge_to_verts[edge_idx, 1]

    flat_edge_verts_table = wp.array(flat_edge_verts, dtype=wp.vec2ub, device=device)

    return (
        tri_range_table,
        tri_local_inds_table,
        edge_to_verts_table,
        corner_offsets_table,
        flat_edge_verts_table,
    )


@wp.func
def int_to_vec3f(x: wp.int32, y: wp.int32, z: wp.int32):
    return wp.vec3f(float(x), float(y), float(z))


@wp.func
def get_triangle_fraction(vert_depths: wp.vec3f, num_inside: wp.int32) -> wp.float32:
    """Compute the fraction of a triangle that lies inside the object based on vertex depths."""
    if num_inside == 3:
        return 1.0

    if num_inside == 0:
        return 0.0

    idx = wp.int32(0)
    if num_inside == 1:
        if vert_depths[1] > 0.0:
            idx = 1
        elif vert_depths[2] > 0.0:
            idx = 2
    else:  # num_inside == 2
        if vert_depths[1] <= 0.0:
            idx = 1
        elif vert_depths[2] <= 0.0:
            idx = 2

    d0 = vert_depths[idx]
    d1 = vert_depths[(idx + 1) % 3]
    d2 = vert_depths[(idx + 2) % 3]

    fraction = (d0 * d0) / ((d0 - d1) * (d0 - d2))
    if num_inside == 2:
        return 1.0 - fraction
    else:
        return fraction


@wp.func
def clip_triangle_to_inside(
    face_verts: wp.mat33f,
    vert_depths: wp.vec3f,
    original_cross: wp.vec3,
    num_inside: wp.int32,
) -> tuple[wp.float32, wp.mat33f, wp.vec3, wp.float32]:
    """
    Clip triangle vertices so all are inside the object (depth > 0).
    Vertices outside are moved to the surface (depth = 0).
    Args:
        original_cross: Pre-computed cross product of original triangle edges.
        num_inside: Pre-computed count of vertices with depth > 0.
    Returns: (area, clipped_face_verts, center, pen_depth)
    """

    original_area = wp.length(original_cross) / 2.0

    if num_inside == 3 or num_inside == 0:
        center = (face_verts[0] + face_verts[1] + face_verts[2]) / 3.0
        pen_depth = (vert_depths[0] + vert_depths[1] + vert_depths[2]) / 3.0
        area = original_area if num_inside == 3 else 0.0
        return area, face_verts, center, pen_depth

    if num_inside == 1:
        inside_idx = wp.int32(0)
        if vert_depths[1] > 0.0:
            inside_idx = 1
        elif vert_depths[2] > 0.0:
            inside_idx = 2

        v_in = face_verts[inside_idx]
        d_in = vert_depths[inside_idx]

        idx1 = (inside_idx + 1) % 3
        idx2 = (inside_idx + 2) % 3
        v1 = face_verts[idx1]
        d1 = vert_depths[idx1]
        v2 = face_verts[idx2]
        d2 = vert_depths[idx2]

        t1 = d_in / (d_in - d1)
        p1 = v_in + t1 * (v1 - v_in)

        t2 = d_in / (d_in - d2)
        p2 = v_in + t2 * (v2 - v_in)

        clipped_verts = wp.mat33f()
        clipped_verts[0] = v_in
        clipped_verts[1] = p1
        clipped_verts[2] = p2

        # Area = original_area * fraction, where fraction = t1 * t2
        area = original_area * t1 * t2
        center = (v_in + p1 + p2) / 3.0
        pen_depth = d_in / 3.0

        return area, clipped_verts, center, pen_depth

    # num_inside == 2: quadrilateral case
    outside_idx = wp.int32(0)
    if vert_depths[1] <= 0.0:
        outside_idx = 1
    elif vert_depths[2] <= 0.0:
        outside_idx = 2

    v_out = face_verts[outside_idx]
    d_out = vert_depths[outside_idx]

    idx1 = (outside_idx + 1) % 3
    idx2 = (outside_idx + 2) % 3
    v1 = face_verts[idx1]
    d1 = vert_depths[idx1]
    v2 = face_verts[idx2]
    d2 = vert_depths[idx2]

    t1 = d_out / (d_out - d1)
    p1 = v_out + t1 * (v1 - v_out)

    t2 = d_out / (d_out - d2)
    p2 = v_out + t2 * (v2 - v_out)

    # Area = original_area * (1 - t1 * t2) since quad = original - cut triangle
    area = original_area * (1.0 - t1 * t2)

    center = (p1 + v1 + v2 + p2) / 4.0
    pen_depth = (d1 + d2) / 4.0

    clipped_verts = wp.mat33f()
    clipped_verts[0] = p1
    clipped_verts[1] = v1
    clipped_verts[2] = v2

    return area, clipped_verts, center, pen_depth


@wp.func
def mc_calc_face(
    flat_edge_verts_table: wp.array(dtype=wp.vec2ub),
    corner_offsets_table: wp.array(dtype=wp.vec3ub),
    tri_range_start: wp.int32,
    corner_vals: vec8f,
    sdf_a: wp.uint64,
    x_id: wp.int32,
    y_id: wp.int32,
    z_id: wp.int32,
    clip_triangles: bool,
) -> tuple[float, wp.vec3, wp.vec3, float, wp.mat33f]:
    """Calculate a marching cubes triangle face with area, normal, center, and penetration depth."""
    face_verts = wp.mat33f()
    vert_depths = wp.vec3f()
    num_inside = wp.int32(0)
    for vi in range(3):
        edge_verts = wp.vec2i(flat_edge_verts_table[tri_range_start + vi])
        v_idx_from = edge_verts[0]
        v_idx_to = edge_verts[1]
        val_0 = wp.float32(corner_vals[v_idx_from])
        val_1 = wp.float32(corner_vals[v_idx_to])

        p_0 = wp.vec3f(corner_offsets_table[v_idx_from])
        p_1 = wp.vec3f(corner_offsets_table[v_idx_to])
        val_diff = val_1 - val_0
        p = p_0 + (0.0 - val_0) * (p_1 - p_0) / val_diff
        vol_idx = p + int_to_vec3f(x_id, y_id, z_id)
        p_scaled = wp.volume_index_to_world(sdf_a, vol_idx)
        face_verts[vi] = p_scaled
        depth = -wp.volume_sample_f(sdf_a, vol_idx, wp.Volume.LINEAR)
        vert_depths[vi] = depth
        if depth > 0.0:
            num_inside += 1

    n = wp.cross(face_verts[1] - face_verts[0], face_verts[2] - face_verts[0])
    normal = wp.normalize(n)

    if clip_triangles:
        area, face_verts, center, pen_depth = clip_triangle_to_inside(face_verts, vert_depths, n, num_inside)
        return area, normal, center, pen_depth, face_verts

    area = wp.length(n) / 2.0
    center = (face_verts[0] + face_verts[1] + face_verts[2]) / 3.0
    pen_depth = (vert_depths[0] + vert_depths[1] + vert_depths[2]) / 3.0
    area *= get_triangle_fraction(vert_depths, num_inside)
    return area, normal, center, pen_depth, face_verts
