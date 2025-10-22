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

"""
Support mapping functions for collision detection primitives.

This module implements support mapping (also called support functions) for various
geometric primitives. A support mapping finds the furthest point of a shape in a
given direction, which is a fundamental operation for collision detection algorithms
like GJK, MPR, and EPA.

The support mapping operates in the shape's local coordinate frame and returns:
- The support point (furthest point in the given direction)
- A feature ID that identifies which geometric feature (vertex, edge, face) the point lies on

Supported primitives:
- Box (axis-aligned rectangular prism)
- Sphere
- Capsule (cylinder with hemispherical caps)
- Ellipsoid
- Cylinder
- Cone
- Plane (finite rectangular plane)
- Convex hull (arbitrary convex mesh)
- Triangle

The module also provides utilities for packing mesh pointers into vectors and
defining generic shape data structures that work across all primitive types.
"""

import enum

import warp as wp

from .types import GeoType


# Is not allowed to share values with GeoType
class GeoTypeEx(enum.IntEnum):
    TRIANGLE = 1000


@wp.struct
class SupportMapDataProvider:
    """
    Placeholder for data access needed by support mapping (e.g., mesh buffers).
    Extend with fields as required by your shapes.
    Not needed for Newton but can be helpful for projects like MuJoCo Warp where
    the convex hull data is stored in warp arrays that would bloat the GenericShapeData struct.
    """

    pass


@wp.func
def pack_mesh_ptr(ptr: wp.uint64) -> wp.vec3:
    """Pack a 64-bit pointer into 3 floats using 22 bits per component"""
    # Extract 22-bit chunks from the pointer
    chunk1 = float(ptr & wp.uint64(0x3FFFFF))  # bits 0-21
    chunk2 = float((ptr >> wp.uint64(22)) & wp.uint64(0x3FFFFF))  # bits 22-43
    chunk3 = float((ptr >> wp.uint64(44)) & wp.uint64(0xFFFFF))  # bits 44-63 (20 bits)

    return wp.vec3(chunk1, chunk2, chunk3)


@wp.func
def unpack_mesh_ptr(arr: wp.vec3) -> wp.uint64:
    """Unpack 3 floats back into a 64-bit pointer"""
    # Convert floats back to integers and combine
    chunk1 = wp.uint64(arr[0]) & wp.uint64(0x3FFFFF)
    chunk2 = (wp.uint64(arr[1]) & wp.uint64(0x3FFFFF)) << wp.uint64(22)
    chunk3 = (wp.uint64(arr[2]) & wp.uint64(0xFFFFF)) << wp.uint64(44)

    return chunk1 | chunk2 | chunk3


@wp.struct
class GenericShapeData:
    """
    Minimal shape descriptor for support mapping.

    Fields:
    - shape_type: matches values from GeoType
    - scale: parameter encoding per primitive
      - BOX: half-extents (x, y, z)
      - SPHERE: radius in x
      - CAPSULE: radius in x, half-height in y (axis +Z)
      - ELLIPSOID: semi-axes (x, y, z)
      - CYLINDER: radius in x, half-height in y (axis +Z)
      - CONE: radius in x, half-height in y (axis +Z, apex at +Z)
      - PLANE: half-width in x, half-length in y (lies in XY plane at z=0, normal along +Z)
      - TRIANGLE: vertex B-A stored in scale, vertex C-A stored in auxiliary
    """

    shape_type: int
    scale: wp.vec3
    auxiliary: wp.vec3


@wp.func
def support_map(
    geom: GenericShapeData, direction: wp.vec3, data_provider: SupportMapDataProvider
) -> tuple[wp.vec3, int]:
    """
    Return the support point of a primitive in its local frame, with a feature id.

    Conventions for `geom.scale` and `geom.auxiliary`:
    - BOX: half-extents in x/y/z
    - SPHERE: radius in x component
    - CAPSULE: radius in x, half-height in y (axis along +Z)
    - ELLIPSOID: semi-axes in x/y/z
    - CYLINDER: radius in x, half-height in y (axis along +Z)
    - CONE: radius in x, half-height in y (axis along +Z, apex at +Z)
    - PLANE: half-width in x, half-length in y (lies in XY plane at z=0, normal along +Z)
    - CONVEX_MESH: scale contains mesh scale, auxiliary contains packed mesh pointer
    - TRIANGLE: scale contains vector B-A, auxiliary contains vector C-A (relative to vertex A at origin)
    """

    # handle zero direction robustly
    eps = 1.0e-12
    dir_len_sq = wp.length_sq(direction)
    dir_safe = wp.vec3(1.0, 0.0, 0.0)
    if dir_len_sq > eps:
        dir_safe = direction

    result = wp.vec3(0.0, 0.0, 0.0)
    feature_id = int(0)

    if geom.shape_type == int(GeoType.CONVEX_MESH):
        # Convex hull support: find the furthest point in the direction
        mesh_ptr = unpack_mesh_ptr(geom.auxiliary)
        mesh = wp.mesh_get(mesh_ptr)

        # The shape scale is stored in geom.scale
        mesh_scale = geom.scale

        # Find the vertex with the maximum dot product with the direction
        max_dot = float(-1.0e10)
        best_vertex = wp.vec3(0.0, 0.0, 0.0)
        best_idx = int(0)

        num_verts = mesh.points.shape[0]

        for i in range(num_verts):
            # Get vertex position (applying scale)
            vertex = wp.cw_mul(mesh.points[i], mesh_scale)

            # Compute dot product with direction
            dot_val = wp.dot(vertex, dir_safe)

            # Track the maximum
            if dot_val > max_dot:
                max_dot = dot_val
                best_vertex = vertex
                best_idx = i
        result = best_vertex
        feature_id = best_idx

    elif geom.shape_type == int(GeoTypeEx.TRIANGLE):
        # Triangle vertices: a at origin, b at scale, c at auxiliary
        tri_a = wp.vec3(0.0, 0.0, 0.0)
        tri_b = geom.scale
        tri_c = geom.auxiliary

        # Compute dot products with direction for each vertex
        dot_a = wp.dot(tri_a, dir_safe)
        dot_b = wp.dot(tri_b, dir_safe)
        dot_c = wp.dot(tri_c, dir_safe)

        # Find the vertex with maximum dot product (furthest in the direction)
        if dot_a >= dot_b and dot_a >= dot_c:
            result = tri_a
            feature_id = 0  # vertex A
        elif dot_b >= dot_c:
            result = tri_b
            feature_id = 1  # vertex B
        else:
            result = tri_c
            feature_id = 2  # vertex C
    elif geom.shape_type == int(GeoType.BOX):
        sx = 1.0 if dir_safe[0] >= 0.0 else -1.0
        sy = 1.0 if dir_safe[1] >= 0.0 else -1.0
        sz = 1.0 if dir_safe[2] >= 0.0 else -1.0

        result = wp.vec3(sx * geom.scale[0], sy * geom.scale[1], sz * geom.scale[2])

        # Bit mask consistent with reference: x->4, y->2, z->1
        feature_id = 0
        if sx >= 0.0:
            feature_id |= 4
        if sy >= 0.0:
            feature_id |= 2
        if sz >= 0.0:
            feature_id |= 1

    elif geom.shape_type == int(GeoType.SPHERE):
        radius = geom.scale[0]
        if dir_len_sq > eps:
            n = wp.normalize(dir_safe)
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        result = n * radius
        feature_id = 0

    elif geom.shape_type == int(GeoType.CAPSULE):
        radius = geom.scale[0]
        half_height = geom.scale[1]

        # Capsule = segment + sphere (adapted from C# code to Z-axis convention)
        # Sphere part: support in normalized direction
        if dir_len_sq > eps:
            n = wp.normalize(dir_safe)
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        result = n * radius

        # Segment endpoints are at (0, 0, +half_height) and (0, 0, -half_height)
        # Use sign of Z-component to pick the correct endpoint
        if dir_safe[2] >= 0.0:
            result = result + wp.vec3(0.0, 0.0, half_height)
            feature_id = 1  # top cap
        else:
            result = result + wp.vec3(0.0, 0.0, -half_height)
            feature_id = 2  # bottom cap

    elif geom.shape_type == int(GeoType.ELLIPSOID):
        # Ellipsoid support for semi-axes a, b, c in direction d:
        # p* = (a^2 dx, b^2 dy, c^2 dz) / sqrt((a dx)^2 + (b dy)^2 + (c dz)^2)
        a = geom.scale[0]
        b = geom.scale[1]
        c = geom.scale[2]
        if dir_len_sq > eps:
            adx = a * dir_safe[0]
            bdy = b * dir_safe[1]
            cdz = c * dir_safe[2]
            denom_sq = adx * adx + bdy * bdy + cdz * cdz
            if denom_sq > eps:
                denom = wp.sqrt(denom_sq)
                result = wp.vec3(
                    (a * a) * dir_safe[0] / denom, (b * b) * dir_safe[1] / denom, (c * c) * dir_safe[2] / denom
                )
            else:
                result = wp.vec3(a, 0.0, 0.0)
        else:
            result = wp.vec3(a, 0.0, 0.0)
        feature_id = 0

    elif geom.shape_type == int(GeoType.CYLINDER):
        radius = geom.scale[0]
        half_height = geom.scale[1]

        # Cylinder support: project direction to XY plane for lateral surface
        dir_xy = wp.vec3(dir_safe[0], dir_safe[1], 0.0)
        dir_xy_len_sq = wp.length_sq(dir_xy)

        if dir_xy_len_sq > eps:
            n_xy = wp.normalize(dir_xy)
            lateral_point = wp.vec3(n_xy[0] * radius, n_xy[1] * radius, 0.0)
        else:
            lateral_point = wp.vec3(radius, 0.0, 0.0)

        # Choose between top cap, bottom cap, or lateral surface
        if dir_safe[2] > 0.0:
            result = wp.vec3(lateral_point[0], lateral_point[1], half_height)
            feature_id = 1  # top cap
        elif dir_safe[2] < 0.0:
            result = wp.vec3(lateral_point[0], lateral_point[1], -half_height)
            feature_id = 2  # bottom cap
        else:
            result = lateral_point
            feature_id = 0  # lateral surface

    elif geom.shape_type == int(GeoType.CONE):
        radius = geom.scale[0]
        half_height = geom.scale[1]

        # Cone support: apex at +Z, base disk at z=-half_height.
        # Using slope k = radius / (2*half_height), the optimal support is:
        #   apex if dz >= k * ||d_xy||, otherwise base rim in d_xy direction.
        apex = wp.vec3(0.0, 0.0, half_height)
        dir_xy = wp.vec3(dir_safe[0], dir_safe[1], 0.0)
        dir_xy_len = wp.length(dir_xy)
        k = radius / (2.0 * half_height) if half_height > eps else 0.0

        if dir_xy_len <= eps:
            # Purely vertical direction
            if dir_safe[2] >= 0.0:
                result = apex
                feature_id = 1  # apex
            else:
                result = wp.vec3(radius, 0.0, -half_height)
                feature_id = 2  # base edge
        else:
            if dir_safe[2] >= k * dir_xy_len:
                result = apex
                feature_id = 1  # apex
            else:
                n_xy = dir_xy / dir_xy_len
                result = wp.vec3(n_xy[0] * radius, n_xy[1] * radius, -half_height)
                feature_id = 2  # base edge

    elif geom.shape_type == int(GeoType.PLANE):
        # Finite plane support: rectangular plane in XY, extents in scale[0] (half-width X) and scale[1] (half-length Y)
        # The plane lies at z=0 with normal along +Z
        half_width = geom.scale[0]
        half_length = geom.scale[1]

        # Clamp the direction to the plane boundaries
        sx = 1.0 if dir_safe[0] >= 0.0 else -1.0
        sy = 1.0 if dir_safe[1] >= 0.0 else -1.0

        # The support point is at the corner in the XY plane (z=0)
        result = wp.vec3(sx * half_width, sy * half_length, 0.0)

        # Feature ID based on which corner
        feature_id = 0
        if sx >= 0.0:
            feature_id |= 1
        if sy >= 0.0:
            feature_id |= 2

    else:
        # Unhandled type: return origin and feature 0
        result = wp.vec3(0.0, 0.0, 0.0)
        feature_id = 0

    return result, feature_id
