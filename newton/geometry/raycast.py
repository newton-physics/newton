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

from newton.geometry import (
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CYLINDER,
    GEO_SPHERE,
)

# A small constant to avoid division by zero and other numerical issues
MINVAL = 1e-15


@wp.func
def _spinlock_acquire(lock: wp.array(dtype=wp.int32)):
    # Try to acquire the lock by setting it to 1 if it's 0
    while wp.atomic_cas(lock, 0, 0, 1) == 1:
        pass


@wp.func
def _spinlock_release(lock: wp.array(dtype=wp.int32)):
    # Release the lock by setting it back to 0
    wp.atomic_exch(lock, 0, 0)


@wp.func
def ray_intersect_sphere(geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, r: float):
    """Computes ray-sphere intersection.

    Args:
        geom_to_world: The world transform of the sphere.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        r: The radius of the sphere.

    Returns:
        The distance along the ray to the closest intersection point, or -1.0 if there is no intersection.
    """
    t_hit = -1.0

    # transform ray to local frame
    world_to_geom = wp.transform_inverse(geom_to_world)
    ray_origin_local = wp.transform_point(world_to_geom, ray_origin)
    ray_direction_local = wp.transform_vector(world_to_geom, ray_direction)

    d_len_sq = wp.dot(ray_direction_local, ray_direction_local)
    if d_len_sq < MINVAL:
        return -1.0

    inv_d_len = 1.0 / wp.sqrt(d_len_sq)
    d_local_norm = ray_direction_local * inv_d_len

    oc = ray_origin_local
    b = wp.dot(oc, d_local_norm)
    c = wp.dot(oc, oc) - r * r

    delta = b * b - c
    if delta >= 0.0:
        sqrt_delta = wp.sqrt(delta)
        t1 = -b - sqrt_delta
        if t1 >= 0.0:
            t_hit = t1 * inv_d_len
        else:
            t2 = -b + sqrt_delta
            if t2 >= 0.0:
                t_hit = t2 * inv_d_len
    return t_hit


@wp.func
def ray_intersect_box(geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, size: wp.vec3):
    """Computes ray-box intersection.

    Args:
        geom_to_world: The world transform of the box.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        size: The half-extents of the box.

    Returns:
        The distance along the ray to the closest intersection point, or -1.0 if there is no intersection.
    """
    # transform ray to local frame
    world_to_geom = wp.transform_inverse(geom_to_world)
    ray_origin_local = wp.transform_point(world_to_geom, ray_origin)
    ray_direction_local = wp.transform_vector(world_to_geom, ray_direction)

    t_hit = -1.0
    t_near = -1.0e10
    t_far = 1.0e10
    hit = 1

    for i in range(3):
        if wp.abs(ray_direction_local[i]) < MINVAL:
            if ray_origin_local[i] < -size[i] or ray_origin_local[i] > size[i]:
                hit = 0
        else:
            inv_d_i = 1.0 / ray_direction_local[i]
            t1 = (-size[i] - ray_origin_local[i]) * inv_d_i
            t2 = (size[i] - ray_origin_local[i]) * inv_d_i

            if t1 > t2:
                temp = t1
                t1 = t2
                t2 = temp

            t_near = wp.max(t_near, t1)
            t_far = wp.min(t_far, t2)

    if hit == 1 and t_near <= t_far and t_far >= 0.0:
        if t_near >= 0.0:
            t_hit = t_near
        else:
            t_hit = t_far
    return t_hit


@wp.func
def ray_intersect_capsule(geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, r: float, h: float):
    """Computes ray-capsule intersection.

    Args:
        geom_to_world: The world transform of the capsule.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        r: The radius of the capsule.
        h: The half-height of the capsule's cylindrical part.

    Returns:
        The distance along the ray to the closest intersection point, or -1.0 if there is no intersection.
    """
    t_hit = -1.0

    # transform ray to local frame
    world_to_geom = wp.transform_inverse(geom_to_world)
    ray_origin_local = wp.transform_point(world_to_geom, ray_origin)
    ray_direction_local = wp.transform_vector(world_to_geom, ray_direction)

    d_len_sq = wp.dot(ray_direction_local, ray_direction_local)
    if d_len_sq < MINVAL:
        return -1.0

    inv_d_len = 1.0 / wp.sqrt(d_len_sq)
    d_local_norm = ray_direction_local * inv_d_len

    min_t = 1.0e10

    # Intersection with cylinder body
    a_cyl = d_local_norm[0] * d_local_norm[0] + d_local_norm[1] * d_local_norm[1]
    if a_cyl > MINVAL:
        b_cyl = 2.0 * (ray_origin_local[0] * d_local_norm[0] + ray_origin_local[1] * d_local_norm[1])
        c_cyl = ray_origin_local[0] * ray_origin_local[0] + ray_origin_local[1] * ray_origin_local[1] - r * r
        delta_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl
        if delta_cyl >= 0.0:
            sqrt_delta_cyl = wp.sqrt(delta_cyl)
            t1 = (-b_cyl - sqrt_delta_cyl) / (2.0 * a_cyl)
            if t1 >= 0.0:
                z = ray_origin_local[2] + t1 * d_local_norm[2]
                if wp.abs(z) <= h:
                    min_t = wp.min(min_t, t1)

            t2 = (-b_cyl + sqrt_delta_cyl) / (2.0 * a_cyl)
            if t2 >= 0.0:
                z = ray_origin_local[2] + t2 * d_local_norm[2]
                if wp.abs(z) <= h:
                    min_t = wp.min(min_t, t2)

    # Intersection with sphere caps
    # Top cap
    oc_top = ray_origin_local - wp.vec3(0.0, 0.0, h)
    b_top = wp.dot(oc_top, d_local_norm)
    c_top = wp.dot(oc_top, oc_top) - r * r
    delta_top = b_top * b_top - c_top
    if delta_top >= 0.0:
        sqrt_delta_top = wp.sqrt(delta_top)
        t1_top = -b_top - sqrt_delta_top
        if t1_top >= 0.0:
            if (ray_origin_local[2] + t1_top * d_local_norm[2]) >= h:
                min_t = wp.min(min_t, t1_top)

        t2_top = -b_top + sqrt_delta_top
        if t2_top >= 0.0:
            if (ray_origin_local[2] + t2_top * d_local_norm[2]) >= h:
                min_t = wp.min(min_t, t2_top)

    # Bottom cap
    oc_bot = ray_origin_local - wp.vec3(0.0, 0.0, -h)
    b_bot = wp.dot(oc_bot, d_local_norm)
    c_bot = wp.dot(oc_bot, oc_bot) - r * r
    delta_bot = b_bot * b_bot - c_bot
    if delta_bot >= 0.0:
        sqrt_delta_bot = wp.sqrt(delta_bot)
        t1_bot = -b_bot - sqrt_delta_bot
        if t1_bot >= 0.0:
            if (ray_origin_local[2] + t1_bot * d_local_norm[2]) <= -h:
                min_t = wp.min(min_t, t1_bot)

        t2_bot = -b_bot + sqrt_delta_bot
        if t2_bot >= 0.0:
            if (ray_origin_local[2] + t2_bot * d_local_norm[2]) <= -h:
                min_t = wp.min(min_t, t2_bot)

    if min_t < 1.0e9:
        t_hit = min_t * inv_d_len

    return t_hit


@wp.func
def ray_intersect_cylinder(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, r: float, h: float
):
    """Computes ray-cylinder intersection.

    Args:
        geom_to_world: The world transform of the cylinder.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        r: The radius of the cylinder.
        h: The half-height of the cylinder.

    Returns:
        The distance along the ray to the closest intersection point, or -1.0 if there is no intersection.
    """
    # transform ray to local frame
    world_to_geom = wp.transform_inverse(geom_to_world)
    ray_origin_local = wp.transform_point(world_to_geom, ray_origin)
    ray_direction_local = wp.transform_vector(world_to_geom, ray_direction)

    t_hit = -1.0
    min_t = 1.0e10

    # Intersection with cylinder body
    a_cyl = ray_direction_local[0] * ray_direction_local[0] + ray_direction_local[1] * ray_direction_local[1]
    if a_cyl > MINVAL:
        b_cyl = 2.0 * (ray_origin_local[0] * ray_direction_local[0] + ray_origin_local[1] * ray_direction_local[1])
        c_cyl = ray_origin_local[0] * ray_origin_local[0] + ray_origin_local[1] * ray_origin_local[1] - r * r
        delta_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl
        if delta_cyl >= 0.0:
            sqrt_delta_cyl = wp.sqrt(delta_cyl)
            inv_2a = 1.0 / (2.0 * a_cyl)
            t1 = (-b_cyl - sqrt_delta_cyl) * inv_2a
            if t1 >= 0.0:
                z = ray_origin_local[2] + t1 * ray_direction_local[2]
                if wp.abs(z) <= h:
                    min_t = wp.min(min_t, t1)

            t2 = (-b_cyl + sqrt_delta_cyl) * inv_2a
            if t2 >= 0.0:
                z = ray_origin_local[2] + t2 * ray_direction_local[2]
                if wp.abs(z) <= h:
                    min_t = wp.min(min_t, t2)

    # Intersection with caps
    if wp.abs(ray_direction_local[2]) > MINVAL:
        inv_d_z = 1.0 / ray_direction_local[2]
        # Top cap
        t_top = (h - ray_origin_local[2]) * inv_d_z
        if t_top >= 0.0:
            x = ray_origin_local[0] + t_top * ray_direction_local[0]
            y = ray_origin_local[1] + t_top * ray_direction_local[1]
            if x * x + y * y <= r * r:
                min_t = wp.min(min_t, t_top)

        # Bottom cap
        t_bot = (-h - ray_origin_local[2]) * inv_d_z
        if t_bot >= 0.0:
            x = ray_origin_local[0] + t_bot * ray_direction_local[0]
            y = ray_origin_local[1] + t_bot * ray_direction_local[1]
            if x * x + y * y <= r * r:
                min_t = wp.min(min_t, t_bot)

    if min_t < 1.0e9:
        t_hit = min_t

    return t_hit


@wp.func
def ray_intersect_geom(
    geom_to_world: wp.transform, size: wp.vec3, geomtype: int, ray_origin: wp.vec3, ray_direction: wp.vec3
):
    """
    Computes the intersection of a ray with a geometry.

    Args:
        geom_to_world: The world-to-shape transform.
        size: The size of the geometry.
        geomtype: The type of the geometry.
        ray_origin: The origin of the ray.
        ray_direction: The direction of the ray.

    Returns:
        The distance along the ray to the closest intersection point, or -1.0 if there is no intersection.
    """
    t_hit = -1.0

    if geomtype == GEO_SPHERE:
        r = size[0]
        t_hit = ray_intersect_sphere(geom_to_world, ray_origin, ray_direction, r)

    elif geomtype == GEO_BOX:
        t_hit = ray_intersect_box(geom_to_world, ray_origin, ray_direction, size)

    elif geomtype == GEO_CAPSULE:
        r = size[0]
        h = size[1]
        t_hit = ray_intersect_capsule(geom_to_world, ray_origin, ray_direction, r, h)

    elif geomtype == GEO_CYLINDER:
        r = size[0]
        h = size[1]
        t_hit = ray_intersect_cylinder(geom_to_world, ray_origin, ray_direction, r, h)

    return t_hit


@wp.kernel
def raycast_kernel(
    # Model
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    geom_type: wp.array(dtype=int),
    geom_size: wp.array(dtype=wp.vec3),
    # Ray
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    # Lock helper
    lock: wp.array(dtype=wp.int32),
    # Output
    min_dist: wp.array(dtype=float),
    min_index: wp.array(dtype=int),
    min_body_index: wp.array(dtype=int),
):
    """
    Computes the intersection of a ray with all geometries in the scene.

    Args:
        body_q: Array of body transforms.
        shape_body: Maps shape index to body index.
        shape_transform: Array of local shape transforms.
        geom_type: Array of geometry types for each geometry.
        geom_size: Array of sizes for each geometry.
        ray_origin: The origin of the ray.
        ray_direction: The direction of the ray.
        lock: Lock array used for synchronization. Expected to be initialized to 0.
        min_dist: A single-element array to store the minimum intersection distance. Expected to be initialized to a large value like 1e10.
        min_index: A single-element array to store the index of the closest geometry. Expected to be initialized to -1.
        min_body_index: A single-element array to store the body index of the closest geometry. Expected to be initialized to -1.
    """
    tid = wp.tid()

    # compute shape transform
    b = shape_body[tid]

    X_wb = wp.transform_identity()
    if b >= 0:
        X_wb = body_q[b]

    X_bs = shape_transform[tid]

    geom_to_world = wp.mul(X_wb, X_bs)

    geomtype = geom_type[tid]

    t = ray_intersect_geom(geom_to_world, geom_size[tid], geomtype, ray_origin, ray_direction)

    if t >= 0.0 and t < min_dist[0]:
        _spinlock_acquire(lock)
        # Still use an atomic inside the spinlock to get a volatile read
        old_min = wp.atomic_min(min_dist, 0, t)
        if t <= old_min:
            min_index[0] = tid
            min_body_index[0] = b
        _spinlock_release(lock)
