# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Support mapping functions for collision detection primitives.

This module implements support mapping (also called support functions) for various
geometric primitives. A support mapping finds the furthest point of a shape in a
given direction, which is a fundamental operation for collision detection algorithms
like GJK, MPR, and EPA.

The support mapping operates in the shape's local coordinate frame and returns the
support point (furthest point in the given direction).

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
    TRIANGLE_PRISM = 1001


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
      - TRIANGLE_PRISM: same as TRIANGLE; support function extrudes 1 m along -Z
    """

    shape_type: int
    scale: wp.vec3
    auxiliary: wp.vec3


@wp.func
def support_map(geom: GenericShapeData, direction: wp.vec3, data_provider: SupportMapDataProvider) -> wp.vec3:
    """
    Return the support point of a primitive in its local frame.

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

    eps = 1.0e-12

    result = wp.vec3(0.0, 0.0, 0.0)

    if geom.shape_type == GeoType.CONVEX_MESH:
        # Convex hull support: find the furthest point in the direction
        mesh_ptr = unpack_mesh_ptr(geom.auxiliary)
        mesh = wp.mesh_get(mesh_ptr)

        mesh_scale = geom.scale
        num_verts = mesh.points.shape[0]

        # Pre-scale direction: dot(scale*v, d) == dot(v, scale*d)
        # This moves the per-vertex cw_mul out of the loop (only 1 at the end)
        scaled_dir = wp.cw_mul(direction, mesh_scale)

        max_dot = float(-1.0e10)
        best_idx = int(0)
        for i in range(num_verts):
            dot_val = wp.dot(mesh.points[i], scaled_dir)
            if dot_val > max_dot:
                max_dot = dot_val
                best_idx = i
        result = wp.cw_mul(mesh.points[best_idx], mesh_scale)

    elif geom.shape_type == GeoTypeEx.TRIANGLE or geom.shape_type == GeoTypeEx.TRIANGLE_PRISM:
        # Triangle vertices: a at origin, b at scale, c at auxiliary
        tri_a = wp.vec3(0.0, 0.0, 0.0)
        tri_b = geom.scale
        tri_c = geom.auxiliary

        # Compute dot products with direction for each vertex
        dot_a = wp.dot(tri_a, direction)
        dot_b = wp.dot(tri_b, direction)
        dot_c = wp.dot(tri_c, direction)

        # Find the vertex with maximum dot product (furthest in the direction)
        if dot_a >= dot_b and dot_a >= dot_c:
            result = tri_a
        elif dot_b >= dot_c:
            result = tri_b
        else:
            result = tri_c

        # TRIANGLE_PRISM: extrude 1 m along -Z to form a solid prism so
        # that GJK/MPR naturally resolves shapes on the back side.
        # The support function is queried in the heightfield's local
        # frame (orientation_a = heightfield rotation), where -Z is
        # always the heightfield's down direction.
        if geom.shape_type == GeoTypeEx.TRIANGLE_PRISM:
            if direction[2] < 0.0:
                result = result + wp.vec3(0.0, 0.0, -1.0)
    elif geom.shape_type == GeoType.BOX:
        sx = 1.0 if direction[0] >= 0.0 else -1.0
        sy = 1.0 if direction[1] >= 0.0 else -1.0
        sz = 1.0 if direction[2] >= 0.0 else -1.0

        result = wp.vec3(sx * geom.scale[0], sy * geom.scale[1], sz * geom.scale[2])

    elif geom.shape_type == GeoType.SPHERE:
        radius = geom.scale[0]
        dir_len_sq = wp.length_sq(direction)
        if dir_len_sq > eps:
            n = wp.normalize(direction)
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        result = n * radius

    elif geom.shape_type == GeoType.CAPSULE:
        radius = geom.scale[0]
        half_height = geom.scale[1]

        # Capsule = segment + sphere (adapted from C# code to Z-axis convention)
        # Sphere part: support in normalized direction
        dir_len_sq = wp.length_sq(direction)
        if dir_len_sq > eps:
            n = wp.normalize(direction)
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        result = n * radius

        # Segment endpoints are at (0, 0, +half_height) and (0, 0, -half_height)
        # Use sign of Z-component to pick the correct endpoint
        if direction[2] >= 0.0:
            result = result + wp.vec3(0.0, 0.0, half_height)
        else:
            result = result + wp.vec3(0.0, 0.0, -half_height)

    elif geom.shape_type == GeoType.ELLIPSOID:
        # Ellipsoid support for semi-axes a, b, c in direction d:
        # p* = (a^2 dx, b^2 dy, c^2 dz) / sqrt((a dx)^2 + (b dy)^2 + (c dz)^2)
        a = geom.scale[0]
        b = geom.scale[1]
        c = geom.scale[2]
        dir_len_sq = wp.length_sq(direction)
        if dir_len_sq > eps:
            adx = a * direction[0]
            bdy = b * direction[1]
            cdz = c * direction[2]
            denom_sq = adx * adx + bdy * bdy + cdz * cdz
            if denom_sq > eps:
                denom = wp.sqrt(denom_sq)
                result = wp.vec3(
                    (a * a) * direction[0] / denom, (b * b) * direction[1] / denom, (c * c) * direction[2] / denom
                )
            else:
                result = wp.vec3(a, 0.0, 0.0)
        else:
            result = wp.vec3(a, 0.0, 0.0)

    elif geom.shape_type == GeoType.CYLINDER:
        radius = geom.scale[0]
        half_height = geom.scale[1]

        # Cylinder support: project direction to XY plane for lateral surface
        dir_xy = wp.vec3(direction[0], direction[1], 0.0)
        dir_xy_len_sq = wp.length_sq(dir_xy)

        if dir_xy_len_sq > eps:
            n_xy = wp.normalize(dir_xy)
            lateral_point = wp.vec3(n_xy[0] * radius, n_xy[1] * radius, 0.0)
        else:
            lateral_point = wp.vec3(radius, 0.0, 0.0)

        # Choose between top cap, bottom cap, or lateral surface
        if direction[2] > 0.0:
            result = wp.vec3(lateral_point[0], lateral_point[1], half_height)
        elif direction[2] < 0.0:
            result = wp.vec3(lateral_point[0], lateral_point[1], -half_height)
        else:
            result = lateral_point

    elif geom.shape_type == GeoType.CONE:
        radius = geom.scale[0]
        half_height = geom.scale[1]

        # Cone support: apex at +Z, base disk at z=-half_height.
        # Using slope k = radius / (2*half_height), the optimal support is:
        #   apex if dz >= k * ||d_xy||, otherwise base rim in d_xy direction.
        apex = wp.vec3(0.0, 0.0, half_height)
        dir_xy = wp.vec3(direction[0], direction[1], 0.0)
        dir_xy_len = wp.length(dir_xy)
        k = radius / (2.0 * half_height) if half_height > eps else 0.0

        if dir_xy_len <= eps:
            # Purely vertical direction
            if direction[2] >= 0.0:
                result = apex
            else:
                result = wp.vec3(radius, 0.0, -half_height)
        else:
            if direction[2] >= k * dir_xy_len:
                result = apex
            else:
                n_xy = dir_xy / dir_xy_len
                result = wp.vec3(n_xy[0] * radius, n_xy[1] * radius, -half_height)

    elif geom.shape_type == GeoType.PLANE:
        # Finite plane support: rectangular plane in XY, extents in scale[0] (half-width X) and scale[1] (half-length Y)
        # The plane lies at z=0 with normal along +Z
        half_width = geom.scale[0]
        half_length = geom.scale[1]

        # Clamp the direction to the plane boundaries
        sx = 1.0 if direction[0] >= 0.0 else -1.0
        sy = 1.0 if direction[1] >= 0.0 else -1.0

        # The support point is at the corner in the XY plane (z=0)
        result = wp.vec3(sx * half_width, sy * half_length, 0.0)

    else:
        # Unhandled type: return origin
        result = wp.vec3(0.0, 0.0, 0.0)

    return result


@wp.func
def support_map_lean(geom: GenericShapeData, direction: wp.vec3, data_provider: SupportMapDataProvider) -> wp.vec3:
    """
    Lean support function for common shape types only: CONVEX_MESH, BOX, SPHERE.

    This is a specialized version of support_map with reduced code size to improve
    GPU instruction cache utilization. It omits support for CAPSULE, ELLIPSOID,
    CYLINDER, CONE, PLANE, and TRIANGLE shapes.
    """
    result = wp.vec3(0.0, 0.0, 0.0)

    if geom.shape_type == GeoType.CONVEX_MESH:
        mesh_ptr = unpack_mesh_ptr(geom.auxiliary)
        mesh = wp.mesh_get(mesh_ptr)
        scaled_dir = wp.cw_mul(direction, geom.scale)
        max_dot = float(-1.0e10)
        best_idx = int(0)
        for i in range(mesh.points.shape[0]):
            dot_val = wp.dot(mesh.points[i], scaled_dir)
            if dot_val > max_dot:
                max_dot = dot_val
                best_idx = i
        result = wp.cw_mul(mesh.points[best_idx], geom.scale)

    elif geom.shape_type == GeoType.BOX:
        sx = 1.0 if direction[0] >= 0.0 else -1.0
        sy = 1.0 if direction[1] >= 0.0 else -1.0
        sz = 1.0 if direction[2] >= 0.0 else -1.0
        result = wp.vec3(sx * geom.scale[0], sy * geom.scale[1], sz * geom.scale[2])

    elif geom.shape_type == GeoType.SPHERE:
        radius = geom.scale[0]
        dir_len_sq = wp.length_sq(direction)
        if dir_len_sq > 1.0e-12:
            n = wp.normalize(direction)
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        result = n * radius

    return result


@wp.func
def extract_shape_data(
    shape_idx: int,
    shape_transform: wp.array[wp.transform],
    shape_types: wp.array[int],
    shape_data: wp.array[wp.vec4],  # scale (xyz), margin_offset (w) or other data
    shape_source: wp.array[wp.uint64],
):
    """
    Extract shape data from the narrow phase API arrays.

    Args:
        shape_idx: Index of the shape
        shape_transform: World space transforms (already computed)
        shape_types: Shape types
        shape_data: Shape data (vec4 - scale xyz, margin_offset w)
        shape_source: Source pointers (mesh IDs etc.)

    Returns:
        tuple: (position, orientation, shape_data, scale, margin_offset)
    """
    # Get shape's world transform (already in world space)
    X_ws = shape_transform[shape_idx]

    position = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    # Extract scale and margin offset from shape_data.
    # shape_data stores scale in xyz and margin offset in w.
    data = shape_data[shape_idx]
    scale = wp.vec3(data[0], data[1], data[2])
    margin_offset = data[3]

    # Create generic shape data
    result = GenericShapeData()
    result.shape_type = shape_types[shape_idx]
    result.scale = scale
    result.auxiliary = wp.vec3(0.0, 0.0, 0.0)

    # For CONVEX_MESH, pack the mesh pointer into auxiliary
    if shape_types[shape_idx] == GeoType.CONVEX_MESH:
        result.auxiliary = pack_mesh_ptr(shape_source[shape_idx])

    return position, orientation, result, scale, margin_offset


@wp.func
def closest_point_on_triangle(
    p: wp.vec3,
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
) -> wp.vec3:
    """
    Closest point on a triangle to a query point.

    Uses Voronoi-region tests with barycentric coordinates to handle
    vertex, edge, and face regions without branching on degenerate normals.

    Args:
        p: Query point
        tri_a: Triangle vertex A
        tri_b: Triangle vertex B
        tri_c: Triangle vertex C

    Returns:
        The closest point on the triangle to *p*.
    """
    ab = tri_b - tri_a
    ac = tri_c - tri_a

    # Guard degenerate triangles: if the triangle has near-zero area, fall
    # back to the closest point on the longest non-degenerate edge (or the
    # nearest vertex when fully collapsed).
    ab_sq = wp.dot(ab, ab)
    ac_sq = wp.dot(ac, ac)
    EPS2 = 1.0e-20
    if wp.dot(wp.cross(ab, ac), wp.cross(ab, ac)) < EPS2:
        bc = tri_c - tri_b
        bc_sq = wp.dot(bc, bc)
        if ab_sq >= ac_sq and ab_sq >= bc_sq:
            if ab_sq < EPS2:
                return tri_a
            t = wp.clamp(wp.dot(p - tri_a, ab) / ab_sq, 0.0, 1.0)
            return tri_a + t * ab
        elif ac_sq >= bc_sq:
            t = wp.clamp(wp.dot(p - tri_a, ac) / ac_sq, 0.0, 1.0)
            return tri_a + t * ac
        else:
            t = wp.clamp(wp.dot(p - tri_b, bc) / bc_sq, 0.0, 1.0)
            return tri_b + t * bc

    ap = p - tri_a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return tri_a

    bp = p - tri_b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return tri_b

    cp = p - tri_c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return tri_c

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return tri_a + v * ab

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return tri_a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return tri_b + w * (tri_c - tri_b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return tri_a + v * ab + w * ac
