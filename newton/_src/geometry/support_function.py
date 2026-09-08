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
from typing import Any

import warp as wp

from .types import GeoType

# Relative deadband factor for box support-map sign decisions.
# Near-zero direction components (e.g. from quaternion rotation noise ~1e-14)
# are treated as non-negative, biasing toward the +1 vertex.
BOX_SUPPORT_DEADBAND = 1.0e-10
_CENTERED_BOX_SUPPORT_TIE_EPSILON = 1.0e-6
TRIANGLE_PRISM_EXTRUSION = 1.0
"""Depth [m] a triangle is extruded along -Z to give a heightfield cell volume."""


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __frsqrt_rn(value);
#else
return 1.0f / sqrtf(value);
#endif
""")
def _support_rsqrt_rn(value: float) -> float:
    """Return a round-to-nearest reciprocal square root of a positive value."""
    ...


# Is not allowed to share values with GeoType
class GeoTypeEx(enum.IntEnum):
    TRIANGLE = 1000
    TRIANGLE_PRISM = 1001


@wp.struct
class SupportMapDataProvider:
    """Optional external data provider for support mapping."""

    pass


@wp.struct
class AcceleratedSupportMapDataProvider:
    """Model-owned directional support-map acceleration data."""

    shape_support_data: wp.array[wp.vec4i]
    support_lut: wp.array[int]
    support_vertex_offsets: wp.array[int]
    support_neighbors: wp.array[int]


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
      - CYLINDER: end radius in x, half-height in y, barrel radius in z (axis +Z)
      - CONE: radius in x, half-height in y (axis +Z, apex at +Z)
      - PLANE: half-width in x, half-length in y (lies in XY plane at z=0, normal along +Z)
      - TRIANGLE: vertex B-A stored in scale, vertex C-A stored in auxiliary
      - TRIANGLE_PRISM: same as TRIANGLE; support function extrudes 1 m along -Z
    """

    shape_type: int
    scale: wp.vec3
    auxiliary: wp.vec3
    center: wp.vec3  # Precomputed local AABB center for convex seed initialization.
    shape_index: int  # Index for optional model-owned support-map acceleration data.


@wp.func
def _octahedral_support_seed(direction: wp.vec3, resolution: int, lut_start: int, lut: wp.array[int]) -> int:
    """Return a directional seed vertex from an octahedral lookup table."""
    length = wp.abs(direction[0]) + wp.abs(direction[1]) + wp.abs(direction[2])
    if length <= 1.0e-20:
        return lut[lut_start]
    p = wp.vec2(direction[0] / length, direction[1] / length)
    if direction[2] < 0.0:
        old = p
        sx = 1.0 if old[0] >= 0.0 else -1.0
        sy = 1.0 if old[1] >= 0.0 else -1.0
        p = wp.vec2((1.0 - wp.abs(old[1])) * sx, (1.0 - wp.abs(old[0])) * sy)
    x = int(wp.round(wp.clamp(0.5 * p[0] + 0.5, 0.0, 1.0) * float(resolution - 1)))
    y = int(wp.round(wp.clamp(0.5 * p[1] + 0.5, 0.0, 1.0) * float(resolution - 1)))
    return lut[lut_start + y * resolution + x]


@wp.func
def _support_map_convex_mesh(
    geom: GenericShapeData, direction: wp.vec3, data_provider: AcceleratedSupportMapDataProvider
) -> wp.vec3:
    """Return convex-mesh support, using an edge walk when acceleration data is available."""
    mesh = wp.mesh_get(unpack_mesh_ptr(geom.auxiliary))
    scaled_dir = wp.cw_mul(direction, geom.scale)
    best_idx = int(0)

    accelerated = geom.shape_index >= 0 and data_provider.shape_support_data.shape[0] > geom.shape_index
    if accelerated:
        support_data = data_provider.shape_support_data[geom.shape_index]
        resolution = support_data[3]
        accelerated = resolution > 0
        if accelerated:
            best_idx = _octahedral_support_seed(scaled_dir, resolution, support_data[0], data_provider.support_lut)
            best_dot = wp.dot(mesh.points[best_idx], scaled_dir)
            improved = int(1)
            iteration = int(0)
            while improved != 0 and iteration < mesh.points.shape[0]:
                improved = 0
                candidate_idx = best_idx
                candidate_dot = best_dot
                begin = data_provider.support_vertex_offsets[support_data[1] + best_idx]
                end = data_provider.support_vertex_offsets[support_data[1] + best_idx + 1]
                for slot in range(begin, end):
                    vertex = data_provider.support_neighbors[support_data[2] + slot]
                    value = wp.dot(mesh.points[vertex], scaled_dir)
                    if value > candidate_dot or (value == candidate_dot and vertex < candidate_idx):
                        candidate_dot = value
                        candidate_idx = vertex
                if candidate_idx != best_idx:
                    best_idx = candidate_idx
                    best_dot = candidate_dot
                    improved = 1
                iteration += 1

    if not accelerated:
        max_dot = float(-1.0e10)
        for i in range(mesh.points.shape[0]):
            dot_val = wp.dot(mesh.points[i], scaled_dir)
            if dot_val > max_dot:
                max_dot = dot_val
                best_idx = i

    return wp.cw_mul(mesh.points[best_idx], geom.scale)


@wp.func
def _support_map_convex_mesh_exhaustive(geom: GenericShapeData, direction: wp.vec3) -> wp.vec3:
    """Return convex-mesh support by scanning all vertices."""
    mesh = wp.mesh_get(unpack_mesh_ptr(geom.auxiliary))
    scaled_dir = wp.cw_mul(direction, geom.scale)
    best_idx = int(0)
    max_dot = float(-1.0e10)
    for i in range(mesh.points.shape[0]):
        dot_val = wp.dot(mesh.points[i], scaled_dir)
        if dot_val > max_dot:
            max_dot = dot_val
            best_idx = i
    return wp.cw_mul(mesh.points[best_idx], geom.scale)


@wp.func
def _support_map_box(geom: GenericShapeData, direction: wp.vec3) -> wp.vec3:
    """Return the support point of a box in its local frame."""
    direction_scale = wp.max(wp.abs(direction[0]), wp.max(wp.abs(direction[1]), wp.abs(direction[2])))
    threshold = BOX_SUPPORT_DEADBAND * direction_scale
    sx = 1.0 if direction[0] >= -threshold else -1.0
    sy = 1.0 if direction[1] >= -threshold else -1.0
    sz = 1.0 if direction[2] >= -threshold else -1.0
    return wp.vec3(sx * geom.scale[0], sy * geom.scale[1], sz * geom.scale[2])


@wp.func
def support_map(geom: GenericShapeData, direction: wp.vec3, data_provider: Any) -> wp.vec3:
    """
    Return the support point of a primitive in its local frame.

    Conventions for `geom.scale` and `geom.auxiliary`:
    - BOX: half-extents in x/y/z
    - SPHERE: radius in x component
    - CAPSULE: radius in x, half-height in y (axis along +Z)
    - ELLIPSOID: semi-axes in x/y/z
    - CYLINDER: end radius in x, half-height in y, barrel radius in z (axis along +Z)
    - CONE: radius in x, half-height in y (axis along +Z, apex at +Z)
    - PLANE: half-width in x, half-length in y (lies in XY plane at z=0, normal along +Z)
    - CONVEX_MESH: scale contains mesh scale, auxiliary contains packed mesh pointer
    - TRIANGLE: scale contains vector B-A, auxiliary contains vector C-A (relative to vertex A at origin)
    """

    eps = 1.0e-12

    result = wp.vec3(0.0, 0.0, 0.0)

    if geom.shape_type == GeoType.CONVEX_MESH:
        result = _support_map_convex_mesh_exhaustive(geom, direction)

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
                result = result + wp.vec3(0.0, 0.0, -TRIANGLE_PRISM_EXTRUSION)
    elif geom.shape_type == GeoType.BOX:
        # Use a relative deadband so near-zero direction components
        # (from solver rotation drift ~1e-7) cannot flip the sign
        # and select a different box vertex.  For face-aligned queries
        # the non-primary components are zero; any vertex on that face
        # is an equally valid support point, so biasing toward +1 is
        # correct and keeps MPR's initial portal construction stable.
        result = _support_map_box(geom, direction)

    elif geom.shape_type == GeoType.SPHERE:
        radius = geom.scale[0]
        dir_len_sq = wp.length_sq(direction)
        if dir_len_sq > eps:
            n = direction * _support_rsqrt_rn(dir_len_sq)
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
            n = direction * _support_rsqrt_rn(dir_len_sq)
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
                inv_denom = _support_rsqrt_rn(denom_sq)
                result = wp.vec3(
                    (a * a) * direction[0] * inv_denom,
                    (b * b) * direction[1] * inv_denom,
                    (c * c) * direction[2] * inv_denom,
                )
            else:
                result = wp.vec3(a, 0.0, 0.0)
        else:
            result = wp.vec3(a, 0.0, 0.0)

    elif geom.shape_type == GeoType.CYLINDER:
        radius = geom.scale[0]
        half_height = geom.scale[1]
        barrel_radius = geom.scale[2]

        dir_xy = wp.vec3(direction[0], direction[1], 0.0)
        dir_xy_len_sq = wp.length_sq(dir_xy)

        if barrel_radius == 0.0:
            # Keep the regular-cylinder path unchanged.
            if dir_xy_len_sq > eps:
                n_xy = dir_xy * _support_rsqrt_rn(dir_xy_len_sq)
                lateral_point = wp.vec3(n_xy[0] * radius, n_xy[1] * radius, 0.0)
            else:
                lateral_point = wp.vec3(radius, 0.0, 0.0)

            if direction[2] > 0.0:
                result = wp.vec3(lateral_point[0], lateral_point[1], half_height)
            elif direction[2] < 0.0:
                result = wp.vec3(lateral_point[0], lateral_point[1], -half_height)
            else:
                result = lateral_point
        else:
            if dir_xy_len_sq > eps:
                dir_xy_len = wp.sqrt(dir_xy_len_sq)
                n_xy = dir_xy / dir_xy_len
            else:
                dir_xy_len = 0.0
                n_xy = wp.vec3(1.0, 0.0, 0.0)

            direction_len = wp.sqrt(dir_xy_len_sq + direction[2] * direction[2])
            support_z = 0.0
            if direction_len > eps:
                support_z = wp.clamp(barrel_radius * direction[2] / direction_len, -half_height, half_height)

            barrel_radius_sq = barrel_radius * barrel_radius
            half_height_sq = half_height * half_height
            support_z_sq = support_z * support_z
            end_offset = wp.sqrt(barrel_radius_sq - half_height_sq)
            support_offset = wp.sqrt(wp.max(barrel_radius_sq - support_z_sq, 0.0))
            offset_sum = support_offset + end_offset
            support_radius = radius
            if offset_sum > eps:
                support_radius += (half_height_sq - support_z_sq) / offset_sum
            result = wp.vec3(n_xy[0] * support_radius, n_xy[1] * support_radius, support_z)

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
def support_map_lean(geom: GenericShapeData, direction: wp.vec3, data_provider: Any) -> wp.vec3:
    """
    Lean support function for common shape types only: CONVEX_MESH, BOX, SPHERE.

    This is a specialized version of support_map with reduced code size to improve
    GPU instruction cache utilization. It omits support for CAPSULE, ELLIPSOID,
    CYLINDER, CONE, PLANE, and TRIANGLE shapes.
    """
    result = wp.vec3(0.0, 0.0, 0.0)

    if geom.shape_type == GeoType.CONVEX_MESH:
        result = _support_map_convex_mesh_exhaustive(geom, direction)

    elif geom.shape_type == GeoType.BOX:
        result = _support_map_box(geom, direction)

    elif geom.shape_type == GeoType.SPHERE:
        radius = geom.scale[0]
        dir_len_sq = wp.length_sq(direction)
        if dir_len_sq > 1.0e-12:
            n = direction * _support_rsqrt_rn(dir_len_sq)
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        result = n * radius

    return result


@wp.func
def support_map_accelerated(geom: GenericShapeData, direction: wp.vec3, data_provider: Any) -> wp.vec3:
    """Support mapping with directional acceleration for eligible convex meshes."""
    if geom.shape_type == GeoType.CONVEX_MESH:
        return _support_map_convex_mesh(geom, direction, data_provider)
    return support_map(geom, direction, data_provider)


@wp.func
def support_map_lean_accelerated(geom: GenericShapeData, direction: wp.vec3, data_provider: Any) -> wp.vec3:
    """Lean support mapping with directional acceleration for eligible convex meshes."""
    if geom.shape_type == GeoType.CONVEX_MESH:
        return _support_map_convex_mesh(geom, direction, data_provider)
    return support_map_lean(geom, direction, data_provider)


def create_shape_support_function(support_func: Any, center_ties: bool = False):
    """Create a support function with built-in shape policies."""
    fuse_builtin_box_support = support_func in (
        support_map,
        support_map_accelerated,
        support_map_lean,
        support_map_lean_accelerated,
    )

    if center_ties:

        @wp.func
        def shape_support(geom: Any, direction: wp.vec3, data_provider: Any) -> wp.vec3:
            result = wp.vec3(0.0, 0.0, 0.0)
            if wp.static(fuse_builtin_box_support):
                if geom.shape_type == GeoType.BOX:
                    abs_direction = wp.vec3(wp.abs(direction[0]), wp.abs(direction[1]), wp.abs(direction[2]))
                    result = _support_map_box(geom, direction)
                    contribution = wp.cw_mul(abs_direction, geom.scale)
                    threshold = _CENTERED_BOX_SUPPORT_TIE_EPSILON * (
                        contribution[0] + contribution[1] + contribution[2]
                    )
                    if contribution[0] <= threshold:
                        result[0] = 0.0
                    if contribution[1] <= threshold:
                        result[1] = 0.0
                    if contribution[2] <= threshold:
                        result[2] = 0.0
                else:
                    result = support_func(geom, direction, data_provider)
            else:
                result = support_func(geom, direction, data_provider)
                if geom.shape_type == GeoType.BOX:
                    contribution = wp.cw_mul(wp.abs(direction), geom.scale)
                    threshold = _CENTERED_BOX_SUPPORT_TIE_EPSILON * (
                        contribution[0] + contribution[1] + contribution[2]
                    )
                    if contribution[0] <= threshold:
                        result[0] = 0.0
                    if contribution[1] <= threshold:
                        result[1] = 0.0
                    if contribution[2] <= threshold:
                        result[2] = 0.0
            return result

    else:

        @wp.func
        def shape_support(geom: Any, direction: wp.vec3, data_provider: Any) -> wp.vec3:
            result = wp.vec3(0.0, 0.0, 0.0)
            if wp.static(fuse_builtin_box_support):
                if geom.shape_type == GeoType.BOX:
                    result = _support_map_box(geom, direction)
                else:
                    result = support_func(geom, direction, data_provider)
            else:
                result = support_func(geom, direction, data_provider)
            return result

    return shape_support


def create_triangle_prism_penetration_refiner(support_func: Any):
    """Create physical-surface refinement for triangle-prism collision proxies.

    MPR operates on closed convex proxies, but a proxy may contain artificial
    faces that are needed only to give it volume. The returned function maps a
    triangle-prism result back to its physical face.

    Args:
        support_func: Support function for individual shapes.

    Returns:
        A function that refines MPR witness points, normal, and penetration.
    """

    shape_support = create_shape_support_function(support_func, center_ties=True)

    @wp.func
    def refine_penetration(
        geom_a: Any,
        geom_b: Any,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: Any,
        point_a: wp.vec3,
        point_b: wp.vec3,
        normal: wp.vec3,
        penetration: float,
    ) -> tuple[wp.vec3, wp.vec3, wp.vec3, float]:
        if geom_a.shape_type == int(GeoTypeEx.TRIANGLE_PRISM):
            surface_normal = wp.cross(geom_a.scale, geom_a.auxiliary)
            normal_length_sq = wp.length_sq(surface_normal)
            if normal_length_sq >= 1.0e-24:
                surface_normal /= wp.sqrt(normal_length_sq)
                if surface_normal[2] < 0.0:
                    surface_normal = -surface_normal

                surface_point_a = shape_support(geom_a, surface_normal, data_provider)
                direction_b = wp.quat_rotate_inv(orientation_b, -surface_normal)
                surface_point_b = shape_support(geom_b, direction_b, data_provider)
                surface_point_b = wp.quat_rotate(orientation_b, surface_point_b) + position_b
                if extend != 0.0:
                    offset = surface_normal * extend * 0.5
                    surface_point_a += offset
                    surface_point_b -= offset
                surface_penetration = wp.dot(surface_point_a - surface_point_b, surface_normal)

                # A finite triangle must not use a support point beyond its
                # footprint. A neighboring heightfield triangle may own that
                # point, while at the outer boundary there may be no surface.
                projected_b = surface_point_b - wp.dot(surface_point_b, surface_normal) * surface_normal
                closest_b = closest_point_on_triangle(
                    projected_b,
                    wp.vec3(0.0),
                    geom_a.scale,
                    geom_a.auxiliary,
                )
                support_on_face = wp.length_sq(projected_b - closest_b) < 1.0e-10

                if not support_on_face:
                    # A neighboring cell owns the deepest point, so measure this cell's overlap
                    # at MPR's own witness instead.  The face normal still applies -- the
                    # surface is what shape B is resting on -- but the depth is only what this
                    # triangle actually carries.
                    surface_point_b = point_b
                    surface_penetration = wp.dot(surface_point_a - point_b, surface_normal)
                normal = surface_normal
                penetration = surface_penetration
                point_b = surface_point_b
                point_a = point_b + penetration * normal

        return point_a, point_b, normal, penetration

    return refine_penetration


@wp.func
def _shape_center(geom: Any) -> wp.vec3:
    """Return a local interior-point approximation for a supported shape."""
    if geom.shape_type == int(GeoType.CONVEX_MESH):
        mesh = wp.mesh_get(unpack_mesh_ptr(geom.auxiliary))
        scale = geom.scale
        first = wp.cw_mul(mesh.points[0], scale)
        lower = first
        upper = first
        for i in range(1, mesh.points.shape[0]):
            point = wp.cw_mul(mesh.points[i], scale)
            lower = wp.min(lower, point)
            upper = wp.max(upper, point)
        return 0.5 * (lower + upper)
    return wp.vec3(0.0)


@wp.func
def _adjust_minkowski_center(geom_a: Any, center_b_world: wp.vec3, center_b_to_a: wp.vec3) -> wp.vec3:
    """Adjust the Minkowski center for shapes that need a local contact seed."""
    if geom_a.shape_type != int(GeoTypeEx.TRIANGLE) and geom_a.shape_type != int(GeoTypeEx.TRIANGLE_PRISM):
        return center_b_to_a

    tri_a = wp.vec3(0.0)
    tri_b = geom_a.scale
    tri_c = geom_a.auxiliary
    face_normal = wp.cross(tri_b - tri_a, tri_c - tri_a)
    face_normal_length_sq = wp.length_sq(face_normal)
    projection = closest_point_on_triangle(center_b_world, tri_a, tri_b, tri_c)
    if face_normal_length_sq < 1.0e-20:
        return projection - center_b_world

    face_normal_unit = face_normal / wp.sqrt(face_normal_length_sq)
    signed_plane_distance = wp.dot(center_b_world - tri_a, face_normal_unit)
    plane_projection = center_b_world - signed_plane_distance * face_normal_unit
    inside_face = (
        wp.dot(wp.cross(tri_b - tri_a, plane_projection - tri_a), face_normal) >= 0.0
        and wp.dot(wp.cross(tri_c - tri_b, plane_projection - tri_b), face_normal) >= 0.0
        and wp.dot(wp.cross(tri_a - tri_c, plane_projection - tri_c), face_normal) >= 0.0
    )
    if inside_face:
        projection = plane_projection
        center_b_to_a = -signed_plane_distance * face_normal_unit
    else:
        center_b_to_a = projection - center_b_world

    to_centroid = (tri_a + tri_b + tri_c) / 3.0 - projection
    to_centroid -= wp.dot(to_centroid, face_normal_unit) * face_normal_unit
    distance_to_centroid = wp.length(to_centroid)
    if distance_to_centroid > 1.0e-12:
        nudge_distance = 0.01 * wp.min(distance_to_centroid, wp.abs(signed_plane_distance))
        center_b_to_a += to_centroid * (nudge_distance / distance_to_centroid)

    if geom_a.shape_type == int(GeoTypeEx.TRIANGLE_PRISM):
        # MPR reports the face its ray from this seed to the origin exits through, so a seed on
        # the boundary of shape A decides nothing.  A triangle has no interior and has to be
        # seeded on its face, but a prism does, and seeding a prism on its top face makes that
        # ray reverse the moment shape B's center crosses the face: from then on the portal
        # settles on the extruded bottom and reports a metre of penetration pointing into the
        # terrain.  Sink the seed to mid-extrusion instead.  The offset is along the extrusion
        # axis and shorter than the extrusion, so the seed stays inside the prism for any
        # triangle, however steep.
        center_b_to_a -= wp.vec3(0.0, 0.0, 0.5 * TRIANGLE_PRISM_EXTRUSION)
    return center_b_to_a


@wp.func
def _minkowski_center_fallback(geom_a: Any, center_b_world: wp.vec3) -> wp.vec3:
    """Return a nonzero triangle seed when the Minkowski centers coincide."""
    if geom_a.shape_type != int(GeoTypeEx.TRIANGLE) and geom_a.shape_type != int(GeoTypeEx.TRIANGLE_PRISM):
        return wp.vec3(0.0)

    tri_a = wp.vec3(0.0)
    tri_b = geom_a.scale
    tri_c = geom_a.auxiliary
    face_normal = wp.cross(tri_b - tri_a, tri_c - tri_a)
    face_normal_length_sq = wp.length_sq(face_normal)
    if face_normal_length_sq < 1.0e-20:
        return wp.vec3(0.0)

    face_normal /= wp.sqrt(face_normal_length_sq)
    projection = closest_point_on_triangle(center_b_world, tri_a, tri_b, tri_c)
    to_centroid = (tri_a + tri_b + tri_c) / 3.0 - projection
    to_centroid -= wp.dot(to_centroid, face_normal) * face_normal
    to_centroid_length_sq = wp.length_sq(to_centroid)

    fallback_direction = -face_normal
    if wp.dot(center_b_world - projection, face_normal) < 0.0:
        fallback_direction = face_normal
    if to_centroid_length_sq > 1.0e-20:
        fallback_direction += 0.01 * to_centroid / wp.sqrt(to_centroid_length_sq)
    return wp.normalize(fallback_direction) * 1.0e-5


@wp.struct
class MinkowskiCenter:
    """Store a Minkowski interior point and optional coincident-center fallback."""

    B: wp.vec3
    BtoA: wp.vec3


def create_shape_center_function(use_precomputed_center: bool = False):
    """Create the common Minkowski-center function used by MPR and GJK.

    The returned function supplies the initial interior point of the
    Minkowski difference. Most primitives use their local origin. Uncached
    convex meshes use the center of their scaled AABB, while cached callers
    use ``geom.center`` and must provide a valid interior-point approximation.

    Triangle-like shape A needs a partner-relative seed. Its center is moved
    to the point on the physical triangle nearest shape B's center and nudged
    toward the triangle centroid. This avoids portals collapsing onto one
    vertex when a large, thin triangle is paired with a much smaller shape.

    Args:
        use_precomputed_center: Use the center stored in each geometry instead
            of computing convex-mesh AABB centers.

    Returns:
        A shape-center function with a ``fallback`` attribute for the
        coincident-center case.
    """

    @wp.func
    def shape_center(
        geom_a: Any,
        geom_b: Any,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        data_provider: Any,
    ) -> MinkowskiCenter:
        """Compute an interior point of the Minkowski difference.

        Args:
            geom_a: Shape A geometry data.
            geom_b: Shape B geometry data.
            orientation_b: Shape B orientation relative to shape A.
            position_b: Shape B position relative to shape A.
            data_provider: Support-map data provider.

        Returns:
            Centers in the relative frame. ``B`` is shape B's center and
            ``BtoA`` points from it to the selected center on shape A.
        """
        center = MinkowskiCenter()
        if wp.static(use_precomputed_center):
            center_a = geom_a.center
            center_b_local = geom_b.center
        else:
            center_a = _shape_center(geom_a)
            center_b_local = _shape_center(geom_b)

        center.B = position_b + wp.quat_rotate(orientation_b, center_b_local)
        center.BtoA = _adjust_minkowski_center(geom_a, center.B, center_a - center.B)
        return center

    shape_center.fallback = _minkowski_center_fallback
    return shape_center


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
    result.center = wp.vec3(0.0, 0.0, 0.0)
    result.shape_index = shape_idx

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
    triangle_normal = wp.cross(ab, ac)
    if wp.dot(triangle_normal, triangle_normal) < EPS2:
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
