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

from .types import (
    GeoType,
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
def ray_intersect_cone(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, radius: float, half_height: float
):
    """Computes ray-cone intersection.

    The cone is oriented along the Z-axis with the tip at +half_height and base at -half_height.

    Args:
        geom_to_world: The world transform of the cone.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        radius: The radius of the cone's base.
        half_height: Half the height of the cone (distance from center to tip/base).

    Returns:
        The distance along the ray to the closest intersection point, or -1.0 if there is no intersection.
    """
    world_to_geom = wp.transform_inverse(geom_to_world)
    ray_origin_local = wp.transform_point(world_to_geom, ray_origin)
    ray_direction_local = wp.transform_vector(world_to_geom, ray_direction)

    if wp.abs(half_height) < MINVAL:
        return -1.0

    if radius <= 0.0:
        return -1.0

    # pa = tip (cone extremes), pb = base center, ra = 0 (tip radius), rb = radius (base radius)
    ro = ray_origin_local
    rd = ray_direction_local
    # Check conventions.rst, section "Newton Collision Primitives"
    pa = wp.vec3(0.0, 0.0, half_height)  # tip at +half_height
    pb = wp.vec3(0.0, 0.0, -half_height)  # base center at -half_height
    ra = 0.0  # radius at tip
    rb = radius  # radius at base

    ba = pb - pa
    oa = ro - pa
    ob = ro - pb
    m0 = wp.dot(ba, ba)
    m1 = wp.dot(oa, ba)
    m2 = wp.dot(rd, ba)
    m3 = wp.dot(rd, oa)
    m5 = wp.dot(oa, oa)
    m9 = wp.dot(ob, ba)

    # caps
    if m1 < 0.0:
        temp = oa * m2 - rd * m1
        if wp.dot(temp, temp) < (ra * ra * m2 * m2):
            if wp.abs(m2) > MINVAL:
                return -m1 / m2
    elif m9 > 0.0:
        if wp.abs(m2) > MINVAL:
            t = -m9 / m2
            temp_ob = ob + rd * t
            if wp.dot(temp_ob, temp_ob) < (rb * rb):
                return t

    # body
    rr = ra - rb
    hy = m0 + rr * rr
    k2 = m0 * m0 - m2 * m2 * hy
    k1 = m0 * m0 * m3 - m1 * m2 * hy + m0 * ra * (rr * m2 * 1.0)
    k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0 - m0 * ra)
    h = k1 * k1 - k2 * k0

    if h < 0.0:
        return -1.0  # no intersection

    if wp.abs(k2) < MINVAL:
        return -1.0  # degenerate case

    t = (-k1 - wp.sqrt(h)) / k2
    y = m1 + t * m2

    if y < 0.0 or y > m0:
        return -1.0  # no intersection

    return t


@wp.func
def ray_intersect_mesh(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, size: wp.vec3, mesh_id: wp.uint64
):
    """Computes ray-mesh intersection using Warp's built-in mesh query.

    Args:
        geom_to_world: The world transform of the mesh.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        size: The 3D scale of the mesh.
        mesh_id: The Warp mesh ID for raycasting.

    Returns:
        The parameter t relative to ray direction magnitude to the closest intersection point, or -1.0 if there is no intersection.
    """
    t_hit = -1.0

    if mesh_id == wp.uint64(0):
        return t_hit

    # Transform ray to local frame
    world_to_geom = wp.transform_inverse(geom_to_world)
    ray_origin_local = wp.transform_point(world_to_geom, ray_origin)
    ray_direction_local = wp.transform_vector(world_to_geom, ray_direction)

    # Apply scale transformation with per-component clamping to MINVAL
    safe_size = wp.vec3(
        size[0] if wp.abs(size[0]) > MINVAL else wp.sign(size[0]) * MINVAL,
        size[1] if wp.abs(size[1]) > MINVAL else wp.sign(size[1]) * MINVAL,
        size[2] if wp.abs(size[2]) > MINVAL else wp.sign(size[2]) * MINVAL,
    )
    scaled_origin = wp.cw_div(ray_origin_local, safe_size)
    scaled_direction = wp.cw_div(ray_direction_local, safe_size)

    scaled_dir_length = wp.length(scaled_direction)
    if scaled_dir_length < MINVAL:
        return t_hit

    normalized_direction = scaled_direction / scaled_dir_length

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face_index = int(0)

    max_t = 1.0e6
    if wp.mesh_query_ray(mesh_id, scaled_origin, normalized_direction, max_t, t, u, v, sign, normal, face_index):
        if t >= 0.0:
            original_dir_length = wp.length(ray_direction_local)
            if original_dir_length > MINVAL:
                # Convert from distance along normalized scaled direction
                # to parameter t relative to original ray direction magnitude
                t_hit = t / scaled_dir_length

    return t_hit


@wp.func
def ray_intersect_geom(
    geom_to_world: wp.transform,
    size: wp.vec3,
    geomtype: int,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    mesh_id: wp.uint64,
):
    """
    Computes the intersection of a ray with a geometry.

    Args:
        geom_to_world: The world-to-shape transform.
        size: The size of the geometry.
        geomtype: The type of the geometry.
        ray_origin: The origin of the ray.
        ray_direction: The direction of the ray.
        mesh_id: The Warp mesh ID for mesh geometries.

    Returns:
        The distance along the ray to the closest intersection point, or -1.0 if there is no intersection.
    """
    t_hit = -1.0

    if geomtype == GeoType.SPHERE:
        r = size[0]
        t_hit = ray_intersect_sphere(geom_to_world, ray_origin, ray_direction, r)

    elif geomtype == GeoType.BOX:
        t_hit = ray_intersect_box(geom_to_world, ray_origin, ray_direction, size)

    elif geomtype == GeoType.CAPSULE:
        r = size[0]
        h = size[1]
        t_hit = ray_intersect_capsule(geom_to_world, ray_origin, ray_direction, r, h)

    elif geomtype == GeoType.CYLINDER:
        r = size[0]
        h = size[1]
        t_hit = ray_intersect_cylinder(geom_to_world, ray_origin, ray_direction, r, h)

    elif geomtype == GeoType.CONE:
        r = size[0]
        h = size[1]
        t_hit = ray_intersect_cone(geom_to_world, ray_origin, ray_direction, r, h)

    elif geomtype == GeoType.MESH or geomtype == GeoType.CONVEX_MESH:
        t_hit = ray_intersect_mesh(geom_to_world, ray_origin, ray_direction, size, mesh_id)

    return t_hit


@wp.kernel
def raycast_kernel(
    # Model
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    geom_type: wp.array(dtype=int),
    geom_size: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
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
        shape_source_ptr: Array of mesh IDs for mesh geometries (wp.uint64).
        ray_origin: The origin of the ray.
        ray_direction: The direction of the ray.
        lock: Lock array used for synchronization. Expected to be initialized to 0.
        min_dist: A single-element array to store the minimum intersection distance. Expected to be initialized to a large value like 1e10.
        min_index: A single-element array to store the index of the closest geometry. Expected to be initialized to -1.
        min_body_index: A single-element array to store the body index of the closest geometry. Expected to be initialized to -1.
    """
    shape_idx = wp.tid()

    # compute shape transform
    b = shape_body[shape_idx]

    X_wb = wp.transform_identity()
    if b >= 0:
        X_wb = body_q[b]

    X_bs = shape_transform[shape_idx]

    geom_to_world = wp.mul(X_wb, X_bs)

    geomtype = geom_type[shape_idx]

    # Get mesh ID for mesh-like geometries
    if geomtype == GeoType.MESH or geomtype == GeoType.CONVEX_MESH:
        mesh_id = shape_source_ptr[shape_idx]
    else:
        mesh_id = wp.uint64(0)

    t = ray_intersect_geom(geom_to_world, geom_size[shape_idx], geomtype, ray_origin, ray_direction, mesh_id)

    if t >= 0.0 and t < min_dist[0]:
        _spinlock_acquire(lock)
        # Still use an atomic inside the spinlock to get a volatile read
        old_min = wp.atomic_min(min_dist, 0, t)
        if t <= old_min:
            min_index[0] = shape_idx
            min_body_index[0] = b
        _spinlock_release(lock)


@wp.func
def ray_for_pixel(
    camera_position: wp.vec3,
    camera_direction: wp.vec3,
    camera_up: wp.vec3,
    camera_right: wp.vec3,
    fov_scale: float,
    camera_aspect_ratio: float,
    resolution: wp.vec2,
    pixel_x: int,
    pixel_y: int,
):
    """
    Generate a ray for a given pixel in a perspective camera.

    Args:
        camera_position: Camera position in world space
        camera_direction: Camera forward direction (normalized)
        camera_up: Camera up direction (normalized)
        camera_right: Camera right direction (normalized)
        camera_fov: Vertical field of view in radians
        camera_aspect_ratio: Width/height aspect ratio
        camera_near_clip: Near clipping plane distance
        resolution: Image resolution as (width, height)
        pixel_x: Pixel x coordinate (0 to width-1)
        pixel_y: Pixel y coordinate (0 to height-1)

    Returns:
        Tuple of (ray_origin, ray_direction) in world space
    """
    width = resolution[0]
    height = resolution[1]

    # Convert to normalized coordinates [-1, 1] with (0,0) at center
    ndc_x = (2.0 * float(pixel_x) + 1.0) / width - 1.0
    ndc_y = 1.0 - (2.0 * float(pixel_y) + 1.0) / height  # Flip Y axis

    # Apply field of view and aspect ratio
    cam_x = ndc_x * fov_scale * camera_aspect_ratio
    cam_y = ndc_y * fov_scale
    cam_z = -1.0  # Forward is negative Z in camera space

    ray_dir_camera = wp.vec3(cam_x, cam_y, cam_z)

    # Transform ray direction from camera to world space
    ray_direction_world = (
        camera_right * ray_dir_camera[0] + camera_up * ray_dir_camera[1] + camera_direction * ray_dir_camera[2]
    )
    ray_direction_world = wp.normalize(ray_direction_world)

    return camera_position, ray_direction_world


@wp.kernel
def raycast_sensor_kernel(
    # Model
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    geom_type: wp.array(dtype=int),
    geom_size: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    # Camera parameters
    camera_position: wp.vec3,
    camera_direction: wp.vec3,
    camera_up: wp.vec3,
    camera_right: wp.vec3,
    fov_scale: float,
    camera_aspect_ratio: float,
    resolution: wp.vec2,
    # Output (per-pixel results)
    hit_distances: wp.array2d(dtype=float),
):
    """
    Raycast sensor kernel that casts rays for each pixel in an image.

    Each thread processes one pixel, generating a ray and finding the closest intersection.

    Args:
        body_q: Array of body transforms
        shape_body: Maps shape index to body index
        shape_transform: Array of local shape transforms
        geom_type: Array of geometry types for each geometry
        geom_size: Array of sizes for each geometry
        shape_source_ptr: Array of mesh IDs for mesh geometries
        camera_position: Camera position in world space
        camera_direction: Camera forward direction (normalized)
        camera_up: Camera up direction (normalized)
        camera_right: Camera right direction (normalized)
        fov_scale: Scale factor for field of view, computed as tan(fov_radians/2) where fov_radians is the vertical field of view angle in radians
        camera_aspect_ratio: Width/height aspect ratio
        resolution: Image resolution as (width, height)
        hit_distances: Output array of hit distances per pixel
    """
    pixel_x, pixel_y, shape_idx = wp.tid()

    # Skip if out of bounds
    if pixel_x >= resolution[0] or pixel_y >= resolution[1]:
        return

    # Generate ray for this pixel
    ray_origin, ray_direction = ray_for_pixel(
        camera_position,
        camera_direction,
        camera_up,
        camera_right,
        fov_scale,
        camera_aspect_ratio,
        resolution,
        pixel_x,
        pixel_y,
    )

    # compute shape transform
    b = shape_body[shape_idx]

    X_wb = wp.transform_identity()
    if b >= 0:
        X_wb = body_q[b]

    X_bs = shape_transform[shape_idx]

    geom_to_world = wp.mul(X_wb, X_bs)

    geomtype = geom_type[shape_idx]

    # Get mesh ID for mesh-like geometries
    if geomtype == GeoType.MESH or geomtype == GeoType.CONVEX_MESH:
        mesh_id = shape_source_ptr[shape_idx]
    else:
        mesh_id = wp.uint64(0)

    t = ray_intersect_geom(geom_to_world, geom_size[shape_idx], geomtype, ray_origin, ray_direction, mesh_id)

    if t >= 0.0:
        wp.atomic_min(hit_distances, pixel_y, pixel_x, t)
