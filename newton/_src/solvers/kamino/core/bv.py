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
KAMINO: Bounding Volumes Module
"""

from __future__ import annotations

import warp as wp

from .math import FLOAT32_MAX, FLOAT32_MIN
from .shapes import ShapeType
from .types import float32, int32, mat83f, transformf, vec3f, vec4f, vec8f

###
# Module interface
###

__all__ = [
    "aabb_box",
    "aabb_capsule",
    "aabb_cone",
    "aabb_cylinder",
    "aabb_ellipsoid",
    "aabb_geom",
    "aabb_sphere",
    "bs_box",
    "bs_capsule",
    "bs_cone",
    "bs_cylinder",
    "bs_ellipsoid",
    "bs_geom",
    "bs_sphere",
    "has_aabb_overlap",
    "has_bs_overlap",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Bounding-Spheres (BS) Functions
###


@wp.func
def bs_sphere(pose: transformf, radius: float32) -> float32:
    return radius


@wp.func
def bs_cylinder(pose: transformf, radius: float32, height: float32) -> float32:
    return float32(0.0)


@wp.func
def bs_cone(pose: transformf, radius: float32, height: float32) -> float32:
    return float32(0.0)


@wp.func
def bs_capsule(pose: transformf, radius: float32, height: float32) -> float32:
    return 0.5 * height + radius


@wp.func
def bs_ellipsoid(pose: transformf, abc: vec3f) -> float32:
    return wp.max(abc[0], wp.max(abc[1], abc[2]))


@wp.func
def bs_box(pose: transformf, size: vec3f) -> float32:
    d = size[0]
    w = size[1]
    h = size[2]
    radius = 0.5 * wp.sqrt(d * d + w * w + h * h)
    return radius


@wp.func
def bs_geom(pose: transformf, params: vec4f, sid: int32) -> float32:
    """
    Compute the radius of the Bounding Sphere (BS) of a geometry element.

    Args:
        sid (int32): Shape index of the geometry element.
        params (vec4f): Shape parameters of the geometry element.
        T_g (transformf): Pose of the geometry element in world coordinates.

    Returns:
        float32: The radius of the BS of the geometry element.
    """
    r = float32(0.0)
    if sid == int(ShapeType.SPHERE.value):
        r = bs_sphere(pose, params[0])
    elif sid == int(ShapeType.CYLINDER.value):
        r = bs_cylinder(pose, params[0], params[1])
    elif sid == int(ShapeType.CONE.value):
        r = bs_cone(pose, params[0], params[1])
    elif sid == int(ShapeType.CAPSULE.value):
        r = bs_capsule(pose, params[0], params[1])
    elif sid == int(ShapeType.ELLIPSOID.value):
        r = bs_ellipsoid(pose, vec3f(params[0], params[1], params[2]))
    elif sid == int(ShapeType.BOX.value):
        r = bs_box(pose, vec3f(params[0], params[1], params[2]))
    return r


@wp.func
def has_bs_overlap(pose1: transformf, pose2: transformf, radius1: float32, radius2: float32) -> wp.bool:
    return False


###
# Axis-Aligned Bounding Box (AABB) Functions
###


@wp.func
def aabb_sphere(pose: transformf, radius: float32) -> mat83f:
    r_g = wp.transform_get_translation(pose)
    min_corner = r_g - vec3f(radius, radius, radius)
    max_corner = r_g + vec3f(radius, radius, radius)
    # Generate 8 corners of the AABB
    aabb = mat83f(
        min_corner[0],
        min_corner[1],
        min_corner[2],
        min_corner[0],
        min_corner[1],
        max_corner[2],
        min_corner[0],
        max_corner[1],
        min_corner[2],
        min_corner[0],
        max_corner[1],
        max_corner[2],
        max_corner[0],
        min_corner[1],
        min_corner[2],
        max_corner[0],
        min_corner[1],
        max_corner[2],
        max_corner[0],
        max_corner[1],
        min_corner[2],
        max_corner[0],
        max_corner[1],
        max_corner[2],
    )
    return aabb


@wp.func
def aabb_cylinder(pose: transformf, radius: float32, height: float32) -> mat83f:
    return mat83f()


@wp.func
def aabb_cone(pose: transformf, radius: float32, height: float32) -> mat83f:
    return mat83f()


@wp.func
def aabb_capsule(pose: transformf, radius: float32, height: float32) -> mat83f:
    return mat83f()


@wp.func
def aabb_ellipsoid(pose: transformf, abc: vec3f) -> mat83f:
    return mat83f()


@wp.func
def aabb_box(pose: transformf, size: vec3f) -> mat83f:
    R_b = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    r_g = wp.transform_get_translation(pose)
    dx = 0.5 * size[0]
    dy = 0.5 * size[1]
    dz = 0.5 * size[2]
    b_v_0 = vec3f(-dx, -dy, -dz)
    b_v_1 = vec3f(-dx, -dy, dz)
    b_v_2 = vec3f(-dx, dy, -dz)
    b_v_3 = vec3f(-dx, dy, dz)
    b_v_4 = vec3f(dx, -dy, -dz)
    b_v_5 = vec3f(dx, -dy, dz)
    b_v_6 = vec3f(dx, dy, -dz)
    b_v_7 = vec3f(dx, dy, dz)
    w_v_0 = r_g + (R_b @ b_v_0)
    w_v_1 = r_g + (R_b @ b_v_1)
    w_v_2 = r_g + (R_b @ b_v_2)
    w_v_3 = r_g + (R_b @ b_v_3)
    w_v_4 = r_g + (R_b @ b_v_4)
    w_v_5 = r_g + (R_b @ b_v_5)
    w_v_6 = r_g + (R_b @ b_v_6)
    w_v_7 = r_g + (R_b @ b_v_7)
    min_x = wp.min(vec8f(w_v_0[0], w_v_1[0], w_v_2[0], w_v_3[0], w_v_4[0], w_v_5[0], w_v_6[0], w_v_7[0]))
    max_x = wp.max(vec8f(w_v_0[0], w_v_1[0], w_v_2[0], w_v_3[0], w_v_4[0], w_v_5[0], w_v_6[0], w_v_7[0]))
    min_y = wp.min(vec8f(w_v_0[1], w_v_1[1], w_v_2[1], w_v_3[1], w_v_4[1], w_v_5[1], w_v_6[1], w_v_7[1]))
    max_y = wp.max(vec8f(w_v_0[1], w_v_1[1], w_v_2[1], w_v_3[1], w_v_4[1], w_v_5[1], w_v_6[1], w_v_7[1]))
    min_z = wp.min(vec8f(w_v_0[2], w_v_1[2], w_v_2[2], w_v_3[2], w_v_4[2], w_v_5[2], w_v_6[2], w_v_7[2]))
    max_z = wp.max(vec8f(w_v_0[2], w_v_1[2], w_v_2[2], w_v_3[2], w_v_4[2], w_v_5[2], w_v_6[2], w_v_7[2]))
    aabb = mat83f(
        min_x,
        min_y,
        min_z,
        min_x,
        min_y,
        max_z,
        min_x,
        max_y,
        min_z,
        min_x,
        max_y,
        max_z,
        max_x,
        min_y,
        min_z,
        max_x,
        min_y,
        max_z,
        max_x,
        max_y,
        min_z,
        max_x,
        max_y,
        max_z,
    )
    return aabb


@wp.func
def aabb_plane(pose: transformf, size: vec3f) -> mat83f:
    return mat83f()


@wp.func
def aabb_geom(pose: transformf, params: vec4f, sid: int32) -> mat83f:
    """
    Compute the Axis-Aligned Bounding Box (AABB) vertices of a geometry element.

    Args:
        sid (int32): Shape index of the geometry element.
        params (vec4f): Shape parameters of the geometry element.
        T_g (transformf): Pose of the geometry element in world coordinates.

    Returns:
        vec6f: The vertices of the AABB of the geometry element.
    """
    aabb = mat83f()
    if sid == int(ShapeType.SPHERE.value):
        aabb = aabb_sphere(pose, params[0])
    elif sid == int(ShapeType.CYLINDER.value):
        aabb = aabb_cylinder(pose, params[0], params[1])
    elif sid == int(ShapeType.CONE.value):
        aabb = aabb_cone(pose, params[0], params[1])
    elif sid == int(ShapeType.CAPSULE.value):
        aabb = aabb_capsule(pose, params[0], params[1])
    elif sid == int(ShapeType.ELLIPSOID.value):
        aabb = aabb_ellipsoid(pose, vec3f(params[0], params[1], params[2]))
    elif sid == int(ShapeType.BOX.value):
        aabb = aabb_box(pose, vec3f(params[0], params[1], params[2]))
    return aabb


@wp.func
def has_aabb_overlap(aabb1: mat83f, aabb2: mat83f) -> wp.bool:
    # Initialize min/max for AABB A,B
    a_min_x = FLOAT32_MAX
    a_min_y = FLOAT32_MAX
    a_min_z = FLOAT32_MAX
    a_max_x = FLOAT32_MIN
    a_max_y = FLOAT32_MIN
    a_max_z = FLOAT32_MIN
    b_min_x = FLOAT32_MAX
    b_min_y = FLOAT32_MAX
    b_min_z = FLOAT32_MAX
    b_max_x = FLOAT32_MIN
    b_max_y = FLOAT32_MIN
    b_max_z = FLOAT32_MIN

    # Iterate through the 8 corners of both AABBs
    # and find the min/max coordinates
    for i in range(8):
        xa = aabb1[i, 0]
        ya = aabb1[i, 1]
        za = aabb1[i, 2]
        xb = aabb2[i, 0]
        yb = aabb2[i, 1]
        zb = aabb2[i, 2]
        a_min_x = wp.min(xa, a_min_x)
        a_min_y = wp.min(ya, a_min_y)
        a_min_z = wp.min(za, a_min_z)
        a_max_x = wp.max(xa, a_max_x)
        a_max_y = wp.max(ya, a_max_y)
        a_max_z = wp.max(za, a_max_z)
        b_min_x = wp.min(xb, b_min_x)
        b_min_y = wp.min(yb, b_min_y)
        b_min_z = wp.min(zb, b_min_z)
        b_max_x = wp.max(xb, b_max_x)
        b_max_y = wp.max(yb, b_max_y)
        b_max_z = wp.max(zb, b_max_z)

    # Overlap test: check for intersection on all 3 axes
    overlap_x = (a_min_x <= b_max_x) and (a_max_x >= b_min_x)
    overlap_y = (a_min_y <= b_max_y) and (a_max_y >= b_min_y)
    overlap_z = (a_min_z <= b_max_z) and (a_max_z >= b_min_z)

    # Return true if there is an overlap on all 3 axes
    return overlap_x and overlap_y and overlap_z
