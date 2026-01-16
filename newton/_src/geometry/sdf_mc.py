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

"""Marching Cubes utilities for SDF isosurface extraction.

Provides lookup tables and GPU functions for extracting triangular faces
from voxels that contain the zero-isosurface of an SDF. Used by hydroelastic
contact generation to find the contact surface between two colliding SDFs.

The marching cubes algorithm classifies each voxel by which of its 8 corners
are inside (negative SDF) vs outside (positive SDF), producing up to 5
triangles per voxel along the zero-crossing.
"""

import numpy as np
import warp as wp

#: Corner values for a single voxel (8 corners)
vec8f = wp.types.vector(length=8, dtype=wp.float32)


def get_mc_tables(device):
    """Create marching cubes lookup tables on the specified device.

    Returns:
        Tuple of 5 warp arrays:
        - tri_range_table: Start/end indices into triangle list per cube case (256 cases)
        - tri_local_inds_table: Edge indices for each triangle vertex
        - edge_to_verts_table: Corner vertex pairs for each of 12 edges
        - corner_offsets_table: 3D offsets for 8 cube corners
        - flat_edge_verts_table: Pre-flattened edge→vertex mapping for efficiency
    """
    # 12 edges of a cube, each connecting two corner vertices
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

    tri_range_table = wp._src.marching_cubes._get_mc_case_to_tri_range_table(device)
    tri_local_inds_table = wp._src.marching_cubes._get_mc_tri_local_inds_table(device)
    corner_offsets_table = wp.array(wp._src.marching_cubes.mc_cube_corner_offsets, dtype=wp.vec3ub, device=device)
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
    """Convert integer voxel coordinates to float vector."""
    return wp.vec3f(float(x), float(y), float(z))


@wp.func
def get_triangle_fraction(vert_depths: wp.vec3f, num_inside: wp.int32) -> wp.float32:
    """Compute the fraction of a triangle's area that lies inside the object.

    Uses linear interpolation along edges to estimate where the zero-crossing
    occurs, then computes the area ratio. Returns 1.0 if all vertices inside,
    0.0 if all outside, or a proportional fraction for partial intersections.
    """
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

    denom = (d0 - d1) * (d0 - d2)
    eps = wp.float32(1e-8)
    if wp.abs(denom) < eps:
        if num_inside == 1:
            return 0.0
        else:
            return 1.0

    fraction = wp.clamp((d0 * d0) / denom, 0.0, 1.0)
    if num_inside == 2:
        return 1.0 - fraction
    else:
        return fraction


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
) -> tuple[float, wp.vec3, wp.vec3, float, wp.mat33f]:
    """Extract a triangle face from a marching cubes voxel.

    Interpolates vertex positions along cube edges where the SDF crosses zero,
    then computes face properties for contact generation.

    Returns:
        Tuple of (area, normal, center, penetration_depth, vertices):
        - area: Triangle area scaled by fraction inside the object
        - normal: Outward-facing unit normal
        - center: Triangle centroid in world space
        - penetration_depth: Average depth of vertices below surface
        - vertices: 3x3 matrix with vertex positions as rows
    """
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
        val_diff = wp.float32(val_1 - val_0)
        if wp.abs(val_diff) < 1e-8:
            p = 0.5 * (p_0 + p_1)
        else:
            t = (0.0 - val_0) / val_diff
            p = p_0 + t * (p_1 - p_0)
        vol_idx = p + int_to_vec3f(x_id, y_id, z_id)
        p_scaled = wp.volume_index_to_world(sdf_a, vol_idx)
        face_verts[vi] = p_scaled
        depth = wp.volume_sample_f(sdf_a, vol_idx, wp.Volume.LINEAR)
        if depth >= wp.inf or wp.isnan(depth):
            depth = 0.0
        vert_depths[vi] = -depth
        if depth < 0.0:
            num_inside += 1

    n = wp.cross(face_verts[1] - face_verts[0], face_verts[2] - face_verts[0])
    normal = wp.normalize(n)
    area = wp.length(n) / 2.0
    center = (face_verts[0] + face_verts[1] + face_verts[2]) / 3.0
    pen_depth = (vert_depths[0] + vert_depths[1] + vert_depths[2]) / 3.0
    area *= get_triangle_fraction(vert_depths, num_inside)
    return area, normal, center, pen_depth, face_verts
