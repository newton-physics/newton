# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Some ray intersection functions are adapted from https://iquilezles.org/articles/intersectors/

import warp as wp

from .types import (
    GeoType,
)

# A small constant to avoid division by zero and other numerical issues
MINVAL = 1e-15
# Generic small epsilon for numerical comparisons
EPSILON = 1e-6
# Tolerance for near-parallel ray rejection (e.g. ray vs plane)
PARALLEL_TOL = 1e-6

_DEFAULT_MESH_MAX_T = 1.0e6


@wp.func
def _spinlock_acquire(lock: wp.array[wp.int32]):
    # Try to acquire the lock by setting it to 1 if it's 0
    while wp.atomic_cas(lock, 0, 0, 1) == 1:
        pass


@wp.func
def _spinlock_release(lock: wp.array[wp.int32]):
    # Release the lock by setting it back to 0
    wp.atomic_exch(lock, 0, 0)


@wp.func
def safe_div_vec3(x: wp.vec3, y: wp.vec3) -> wp.vec3:
    """Component-wise division that substitutes ``EPSILON`` for zero denominators."""
    return wp.vec3(
        x[0] / wp.where(y[0] != 0.0, y[0], EPSILON),
        x[1] / wp.where(y[1] != 0.0, y[1], EPSILON),
        x[2] / wp.where(y[2] != 0.0, y[2], EPSILON),
    )


@wp.func
def map_ray_to_local(transform: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """Maps a ray from world space into the local shape frame.

    Args:
        transform: World transform of the shape.
        ray_origin: Starting point of the ray in world space.
        ray_direction: Direction of the ray in world space.

    Returns:
        Tuple of (ray_origin_local, ray_direction_local) in the shape's local frame.
    """
    inv_transform = wp.transform_inverse(transform)
    ray_origin_local = wp.transform_point(inv_transform, ray_origin)
    ray_direction_local = wp.transform_vector(inv_transform, ray_direction)
    return ray_origin_local, ray_direction_local


@wp.func
def map_ray_to_local_scaled(
    transform: wp.transform, scale: wp.vec3, ray_origin: wp.vec3, ray_direction: wp.vec3
) -> tuple[wp.vec3, wp.vec3]:
    """Maps a ray into a shape's local frame and divides by per-axis scale."""
    ray_origin_local, ray_direction_local = map_ray_to_local(transform, ray_origin, ray_direction)
    inv_size = safe_div_vec3(wp.vec3(1.0), scale)
    return wp.cw_mul(ray_origin_local, inv_size), wp.cw_mul(ray_direction_local, inv_size)


@wp.func
def ray_intersect_sphere(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, r: float
) -> tuple[float, wp.vec3]:
    """Computes ray-sphere intersection.

    Args:
        geom_to_world: The world transform of the sphere.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        r: The radius of the sphere.

    Returns:
        The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    ray_origin_local, ray_direction_local = map_ray_to_local(geom_to_world, ray_origin, ray_direction)

    d_len_sq = wp.dot(ray_direction_local, ray_direction_local)
    if d_len_sq < MINVAL:
        return t_hit, normal

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

    if t_hit >= 0.0:
        hit_point = ray_origin + t_hit * ray_direction
        normal = wp.normalize(hit_point - wp.transform_get_translation(geom_to_world))

    return t_hit, normal


@wp.func
def ray_intersect_particle_sphere(
    ray_origin: wp.vec3, ray_direction: wp.vec3, center: wp.vec3, radius: float
) -> tuple[float, wp.vec3]:
    """Compute the closest hit along a (unit-length) ray against a sphere defined directly in world space.

    Args:
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space (should be normalized).
        center: The center of the sphere in world space.
        radius: The radius of the sphere.

    Returns:
        The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    oc = ray_origin - center
    proj = wp.dot(ray_direction, oc)
    c = wp.dot(oc, oc) - radius * radius
    disc = proj * proj - c

    if disc < 0.0:
        return t_hit, normal

    sqrt_disc = wp.sqrt(disc)
    t_hit = -proj - sqrt_disc
    if t_hit < 0.0:
        # hit behind ray origin, try other root
        t_hit = -proj + sqrt_disc

    if t_hit < 0.0:
        return -1.0, wp.vec3(0.0)

    normal = wp.normalize(ray_origin + t_hit * ray_direction - center)
    return t_hit, normal


@wp.func
def ray_intersect_ellipsoid(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, semi_axes: wp.vec3
) -> tuple[float, wp.vec3]:
    """Computes ray-ellipsoid intersection.

    The ellipsoid is defined by semi-axes (a, b, c) along the local X, Y, Z axes respectively.
    Based on Inigo Quilez's ellipsoid intersection algorithm.

    Args:
        geom_to_world: The world transform of the ellipsoid.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        semi_axes: The semi-axes (a, b, c) of the ellipsoid.

    Returns:
        The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    ro, rd = map_ray_to_local(geom_to_world, ray_origin, ray_direction)

    # Reject degenerate rays (matching sphere/capsule pattern)
    d_len_sq = wp.dot(rd, rd)
    if d_len_sq < MINVAL:
        return t_hit, normal

    ra = semi_axes

    # Ensure semi-axes are valid
    if ra[0] < MINVAL or ra[1] < MINVAL or ra[2] < MINVAL:
        return t_hit, normal

    # Scale by inverse semi-axes (transforms ellipsoid to unit sphere)
    ocn = wp.cw_div(ro, ra)
    rdn = wp.cw_div(rd, ra)

    a = wp.dot(rdn, rdn)
    b = wp.dot(ocn, rdn)
    c = wp.dot(ocn, ocn)

    h = b * b - a * (c - 1.0)
    if h < 0.0:
        return t_hit, normal  # No intersection

    h = wp.sqrt(h)

    # Two intersection points: (-b - h) / a and (-b + h) / a
    t1 = (-b - h) / a
    t2 = (-b + h) / a

    # Return nearest positive intersection
    if t1 >= 0.0:
        t_hit = t1
    elif t2 >= 0.0:
        t_hit = t2

    if t_hit >= 0.0:
        hit_local = ro + t_hit * rd
        inv_size = safe_div_vec3(wp.vec3(1.0), semi_axes)
        inv_size_sq = wp.cw_mul(inv_size, inv_size)
        normal_local = wp.cw_mul(hit_local, inv_size_sq)
        normal = wp.normalize(wp.transform_vector(geom_to_world, normal_local))

    return t_hit, normal


_IFACE = wp.types.matrix((3, 2), dtype=wp.int32)(1, 2, 0, 2, 0, 1)


@wp.func
def ray_intersect_box(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, size: wp.vec3
) -> tuple[float, wp.vec3]:
    """Computes ray-box intersection.

    Args:
        geom_to_world: The world transform of the box.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        size: The half-extents of the box.

    Returns:
        The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    ray_origin_local, ray_direction_local = map_ray_to_local(geom_to_world, ray_origin, ray_direction)

    t_hit = -1.0
    normal = wp.vec3(0.0)
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

    if t_hit >= 0.0:
        # Identify the hit face by matching the solved face distance to t_hit.
        normal_local = wp.vec3(0.0)
        for i in range(3):
            if wp.abs(ray_direction_local[i]) > EPSILON:
                for side in range(-1, 2, 2):
                    sol = (float(side) * size[i] - ray_origin_local[i]) / ray_direction_local[i]
                    if sol >= 0.0:
                        id0 = _IFACE[i][0]
                        id1 = _IFACE[i][1]
                        p0 = ray_origin_local[id0] + sol * ray_direction_local[id0]
                        p1 = ray_origin_local[id1] + sol * ray_direction_local[id1]
                        if wp.abs(p0) <= size[id0] and wp.abs(p1) <= size[id1]:
                            if wp.abs(sol - t_hit) < EPSILON:
                                normal_local[i] = -1.0 if side < 0 else 1.0
        normal = wp.normalize(wp.transform_vector(geom_to_world, normal_local))

    return t_hit, normal


@wp.func
def ray_intersect_capsule(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, r: float, h: float
) -> tuple[float, wp.vec3]:
    """Computes ray-capsule intersection.

    Args:
        geom_to_world: The world transform of the capsule.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        r: The radius of the capsule.
        h: The half-height of the capsule's cylindrical part.

    Returns:
        The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    ray_origin_local, ray_direction_local = map_ray_to_local(geom_to_world, ray_origin, ray_direction)

    d_len_sq = wp.dot(ray_direction_local, ray_direction_local)
    if d_len_sq < MINVAL:
        return t_hit, normal

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

    if t_hit >= 0.0:
        hit_local = ray_origin_local + t_hit * ray_direction_local
        z_clamped = wp.min(h, wp.max(-h, hit_local[2]))
        axis_point = wp.vec3(0.0, 0.0, z_clamped)
        normal_local = wp.normalize(hit_local - axis_point)
        normal = wp.normalize(wp.transform_vector(geom_to_world, normal_local))

    return t_hit, normal


@wp.func
def ray_intersect_cylinder(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, r: float, h: float
) -> tuple[float, wp.vec3]:
    """Computes ray-cylinder intersection.

    Args:
        geom_to_world: The world transform of the cylinder.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        r: The radius of the cylinder.
        h: The half-height of the cylinder.

    Returns:
        The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    ray_origin_local, ray_direction_local = map_ray_to_local(geom_to_world, ray_origin, ray_direction)

    t_hit = -1.0
    normal = wp.vec3(0.0)
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

    if t_hit >= 0.0:
        hit_local = ray_origin_local + t_hit * ray_direction_local
        z_clamped = wp.min(h, wp.max(-h, hit_local[2]))
        if z_clamped >= (h - EPSILON) or z_clamped <= (-h + EPSILON):
            normal_local = wp.vec3(0.0, 0.0, z_clamped)
        else:
            normal_local = wp.normalize(hit_local - wp.vec3(0.0, 0.0, z_clamped))
        normal = wp.normalize(wp.transform_vector(geom_to_world, normal_local))

    return t_hit, normal


@wp.func
def ray_intersect_cone(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, radius: float, half_height: float
) -> tuple[float, wp.vec3]:
    """Computes ray-cone intersection.

    The cone is oriented along the Z-axis with the tip at +half_height and base at -half_height.

    Args:
        geom_to_world: The world transform of the cone.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        radius: The radius of the cone's base.
        half_height: Half the height of the cone (distance from center to tip/base).

    Returns:
        The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    ray_origin_local, ray_direction_local = map_ray_to_local(geom_to_world, ray_origin, ray_direction)

    if wp.abs(half_height) < MINVAL:
        return t_hit, normal

    if radius <= 0.0:
        return t_hit, normal

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
                t_hit = -m1 / m2
    elif m9 > 0.0:
        if wp.abs(m2) > MINVAL:
            t = -m9 / m2
            temp_ob = ob + rd * t
            if wp.dot(temp_ob, temp_ob) < (rb * rb):
                t_hit = t

    if t_hit < 0.0:
        # body
        rr = ra - rb
        hy = m0 + rr * rr
        k2 = m0 * m0 - m2 * m2 * hy
        k1 = m0 * m0 * m3 - m1 * m2 * hy + m0 * ra * (rr * m2 * 1.0)
        k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0 - m0 * ra)
        h = k1 * k1 - k2 * k0

        if h >= 0.0 and wp.abs(k2) >= MINVAL:
            t = (-k1 - wp.sqrt(h)) / k2
            y = m1 + t * m2
            if y >= 0.0 and y <= m0:
                t_hit = t

    if t_hit >= 0.0:
        hit_local = ray_origin_local + t_hit * ray_direction_local
        if wp.abs(hit_local[2] - half_height) <= EPSILON:
            normal_local = wp.vec3(0.0, 0.0, 1.0)
        elif wp.abs(hit_local[2] + half_height) <= EPSILON:
            normal_local = wp.vec3(0.0, 0.0, -1.0)
        else:
            radial_sq = hit_local[0] * hit_local[0] + hit_local[1] * hit_local[1]
            radial = wp.sqrt(radial_sq)
            if radial <= EPSILON:
                normal_local = wp.vec3(0.0, 0.0, 1.0)
            else:
                denom = wp.max(2.0 * wp.abs(half_height), EPSILON)
                slope = radius / denom
                normal_local = wp.normalize(wp.vec3(hit_local[0], hit_local[1], slope * radial))
        normal = wp.normalize(wp.transform_vector(geom_to_world, normal_local))

    return t_hit, normal


@wp.func
def ray_intersect_plane(
    geom_to_world: wp.transform, ray_origin: wp.vec3, ray_direction: wp.vec3, size: wp.vec3
) -> tuple[float, wp.vec3]:
    """Computes ray-plane intersection.

    The plane lies at z = 0 in local space with normal along +Z.  ``size`` holds ``(width, length, 0)``:
    the full extents along local X and Y.  A value of ``0`` means infinite in that axis.  The plane is
    double-sided: rays approaching from either side register intersections. Callers that need
    back-face culling can check ``wp.dot(ray_direction, normal)`` themselves.

    Args:
        geom_to_world: The world transform of the plane.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        size: ``(width, length, 0)`` -- full extents; ``0`` = infinite.

    Returns:
        The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    ro, rd = map_ray_to_local(geom_to_world, ray_origin, ray_direction)

    # Ray parallel to the plane (or degenerate)
    if wp.abs(rd[2]) < PARALLEL_TOL:
        return t_hit, normal

    t = -ro[2] / rd[2]
    if t < 0.0:
        return t_hit, normal

    hit_x = ro[0] + t * rd[0]
    hit_y = ro[1] + t * rd[1]

    half_w = size[0] * 0.5
    half_l = size[1] * 0.5

    if half_w > 0.0 and wp.abs(hit_x) > half_w:
        return t_hit, normal
    if half_l > 0.0 and wp.abs(hit_y) > half_l:
        return t_hit, normal

    t_hit = t
    normal = wp.normalize(wp.transform_vector(geom_to_world, wp.vec3(0.0, 0.0, 1.0)))

    return t_hit, normal


@wp.func
def ray_intersect_mesh(
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    size: wp.vec3,
    mesh_id: wp.uint64,
    enable_backface_culling: bool,
    max_t: float,
) -> tuple[float, wp.vec3, float, float, int]:
    """Computes ray-mesh intersection using Warp's built-in mesh query.

    Args:
        geom_to_world: The world transform of the mesh.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        size: The 3D scale of the mesh.
        mesh_id: The Warp mesh ID for raycasting.
        enable_backface_culling: When ``True``, reject hits whose triangle normal
            is aligned with the ray direction (back faces).
        max_t: Maximum parameter ``t`` along the (local, scaled) ray to consider.

    Returns:
        Tuple ``(distance, normal, u, v, face_index)``. The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection; on miss, ``u`` and ``v`` are ``0.0`` and ``face_index`` is -1.
    """
    if mesh_id == wp.uint64(0):
        return -1.0, wp.vec3(0.0), 0.0, 0.0, -1

    ray_origin_local, ray_direction_local = map_ray_to_local_scaled(geom_to_world, size, ray_origin, ray_direction)

    query = wp.mesh_query_ray(mesh_id, ray_origin_local, ray_direction_local, max_t)

    if query.result:
        if not enable_backface_culling or wp.dot(ray_direction_local, query.normal) < 0.0:
            normal = wp.normalize(wp.transform_vector(geom_to_world, safe_div_vec3(query.normal, size)))
            return query.t, normal, query.u, query.v, query.face

    return -1.0, wp.vec3(0.0), 0.0, 0.0, -1


@wp.func
def ray_intersect_mesh_no_transform(
    mesh_id: wp.uint64,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    enable_backface_culling: bool,
    max_t: float,
) -> tuple[float, wp.vec3, float, float, int]:
    """Ray-mesh intersection when the mesh is already expressed in world space.

    Requires the Warp ``wp.Mesh`` handle be supplied in ``mesh_id``.

    Args:
        mesh_id: The Warp mesh ID for raycasting.
        ray_origin: The origin of the ray in world space.
        ray_direction: The direction of the ray in world space.
        enable_backface_culling: When ``True``, reject hits whose triangle normal
            is aligned with the ray direction (back faces).
        max_t: Maximum parameter ``t`` along the ray to consider.

    Returns:
        Tuple ``(distance, normal, u, v, face_index)``. The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection; on miss, ``u`` and ``v`` are ``0.0`` and ``face_index`` is -1.
    """
    if mesh_id == wp.uint64(0):
        return -1.0, wp.vec3(0.0), 0.0, 0.0, -1

    query = wp.mesh_query_ray(mesh_id, ray_origin, ray_direction, max_t)
    if query.result:
        if not enable_backface_culling or wp.dot(ray_direction, query.normal) < 0.0:
            return query.t, wp.normalize(query.normal), query.u, query.v, query.face

    return -1.0, wp.vec3(0.0), 0.0, 0.0, -1


@wp.func
def ray_intersect_geom(
    geom_to_world: wp.transform,
    size: wp.vec3,
    geomtype: int,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    mesh_id: wp.uint64,
) -> tuple[float, wp.vec3]:
    """Dispatches to the appropriate ray-shape intersection routine.

    Args:
        geom_to_world: The world transform of the shape.
        size: The size of the geometry.
        geomtype: The type of the geometry.
        ray_origin: The origin of the ray.
        ray_direction: The direction of the ray.
        mesh_id: The Warp mesh ID for mesh geometries.

    Returns:
        The distance and normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    if geomtype == GeoType.PLANE:
        t_hit, normal = ray_intersect_plane(geom_to_world, ray_origin, ray_direction, size)

    elif geomtype == GeoType.SPHERE:
        t_hit, normal = ray_intersect_sphere(geom_to_world, ray_origin, ray_direction, size[0])

    elif geomtype == GeoType.BOX:
        t_hit, normal = ray_intersect_box(geom_to_world, ray_origin, ray_direction, size)

    elif geomtype == GeoType.CAPSULE:
        t_hit, normal = ray_intersect_capsule(geom_to_world, ray_origin, ray_direction, size[0], size[1])

    elif geomtype == GeoType.CYLINDER:
        t_hit, normal = ray_intersect_cylinder(geom_to_world, ray_origin, ray_direction, size[0], size[1])

    elif geomtype == GeoType.CONE:
        t_hit, normal = ray_intersect_cone(geom_to_world, ray_origin, ray_direction, size[0], size[1])

    elif geomtype == GeoType.ELLIPSOID:
        t_hit, normal = ray_intersect_ellipsoid(geom_to_world, ray_origin, ray_direction, size)

    elif geomtype == GeoType.MESH or geomtype == GeoType.CONVEX_MESH:
        t_hit, normal, _u, _v, _face = ray_intersect_mesh(
            geom_to_world, ray_origin, ray_direction, size, mesh_id, False, _DEFAULT_MESH_MAX_T
        )

    return t_hit, normal


@wp.kernel
def raycast_kernel(
    # Model
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    geom_type: wp.array[int],
    geom_size: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    # Ray
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    # Lock helper
    lock: wp.array[wp.int32],
    # Output
    min_dist: wp.array[float],
    min_index: wp.array[int],
    min_body_index: wp.array[int],
    # Optional: world offsets for multi-world picking
    shape_world: wp.array[int],
    world_offsets: wp.array[wp.vec3],
    visible_worlds_mask: wp.array[int],
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
        shape_world: Optional array mapping shape index to world index. Can be empty to disable world offsets.
        world_offsets: Optional array of world offsets. Can be empty to disable world offsets.
        visible_worlds_mask: Optional mask array (1=visible, 0=hidden per world). Can be empty to disable filtering.
    """
    shape_idx = wp.tid()

    # Skip shapes from non-visible worlds
    if visible_worlds_mask and shape_world.shape[0] > 0:
        world_idx = shape_world[shape_idx]
        if world_idx >= 0:
            if visible_worlds_mask[world_idx] == 0:
                return

    # compute shape transform
    b = shape_body[shape_idx]

    X_wb = wp.transform_identity()
    if b >= 0:
        X_wb = body_q[b]

    X_bs = shape_transform[shape_idx]

    geom_to_world = wp.mul(X_wb, X_bs)

    # Apply world offset if available (for multi-world picking)
    if shape_world.shape[0] > 0 and world_offsets.shape[0] > 0:
        world_idx = shape_world[shape_idx]
        if world_idx >= 0 and world_idx < world_offsets.shape[0]:
            offset = world_offsets[world_idx]
            geom_to_world = wp.transform(geom_to_world.p + offset, geom_to_world.q)

    geomtype = geom_type[shape_idx]

    # Get mesh ID for mesh-like geometries
    if geomtype == GeoType.MESH or geomtype == GeoType.CONVEX_MESH:
        mesh_id = shape_source_ptr[shape_idx]
    else:
        mesh_id = wp.uint64(0)

    t, _normal = ray_intersect_geom(geom_to_world, geom_size[shape_idx], geomtype, ray_origin, ray_direction, mesh_id)

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
        Tuple of (ray_origin, ray_direction) in world space. With the direction normalized.
    """
    width = resolution[0]
    height = resolution[1]

    # Convert to normalized coordinates [-1, 1] with (0,0) at center
    ndc_x = (2.0 * float(pixel_x) + 1.0) / width - 1.0
    ndc_y = 1.0 - (2.0 * float(pixel_y) + 1.0) / height  # Flip Y axis

    # Apply field of view and aspect ratio
    cam_x = ndc_x * fov_scale * camera_aspect_ratio
    cam_y = ndc_y * fov_scale
    cam_z = 1.0  # Forward is negative Z in camera space (camera_direction already looks at -Z)

    ray_dir_camera = wp.vec3(cam_x, cam_y, cam_z)

    # Transform ray direction from camera to world space
    ray_direction_world = (
        camera_right * ray_dir_camera[0] + camera_up * ray_dir_camera[1] + camera_direction * ray_dir_camera[2]
    )
    ray_direction_world = wp.normalize(ray_direction_world)

    return camera_position, ray_direction_world


@wp.kernel
def sensor_raycast_kernel(
    # Model
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    geom_type: wp.array[int],
    geom_size: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    # Camera parameters
    camera_position: wp.vec3,
    camera_direction: wp.vec3,
    camera_up: wp.vec3,
    camera_right: wp.vec3,
    fov_scale: float,
    camera_aspect_ratio: float,
    resolution: wp.vec2,
    # Output (per-pixel results)
    hit_distances: wp.array2d[float],
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

    t, _normal = ray_intersect_geom(geom_to_world, geom_size[shape_idx], geomtype, ray_origin, ray_direction, mesh_id)

    if t >= 0.0:
        wp.atomic_min(hit_distances, pixel_y, pixel_x, t)


@wp.kernel
def sensor_raycast_particles_kernel(
    grid: wp.uint64,
    particle_positions: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    search_radius: float,
    march_step: float,
    max_steps: wp.int32,
    camera_position: wp.vec3,
    camera_direction: wp.vec3,
    camera_up: wp.vec3,
    camera_right: wp.vec3,
    fov_scale: float,
    camera_aspect_ratio: float,
    resolution: wp.vec2,
    max_distance: float,
    hit_distances: wp.array2d[float],
):
    """March rays against particles stored in a hash grid and record the nearest hit if found before max_distance.

    Args:
        grid: The hash grid containing the particles.
        particle_positions: Array of particle positions.
        particle_radius: Array of particle radii.
        search_radius: The radius around each sample point to search for nearby particles.
        march_step: The step size for ray marching.
        max_steps: Maximum number of ray-march iterations allowed for a pixel.
        camera_position: Camera position in world space.
        camera_direction: Camera forward direction (normalized); rays travel along this vector.
        camera_up: Camera up direction (normalized).
        camera_right: Camera right direction (normalized).
        fov_scale: Scale factor for field of view, computed as tan(fov_radians/2) where fov_radians is the vertical field of view angle in radians.
        camera_aspect_ratio: Width/height aspect ratio.
        resolution: Image resolution as (width, height).
        max_distance: Maximum distance to march along the ray.
        hit_distances: Output array of hit distances per pixel.
    """
    pixel_x, pixel_y = wp.tid()

    if pixel_x >= resolution[0] or pixel_y >= resolution[1]:
        return

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

    best = hit_distances[pixel_y, pixel_x]
    if best < 0.0:
        best = max_distance

    search_radius_local = search_radius
    step = march_step

    s = wp.int32(0)
    t = float(0.0)

    while s < max_steps and t <= max_distance and t <= best:
        sample_pos = ray_origin + ray_direction * t

        query = wp.hash_grid_query(grid, sample_pos, search_radius_local)
        candidate = int(0)

        while wp.hash_grid_query_next(query, candidate):
            # Intersect ray with particle sphere
            radius = particle_radius[candidate]
            if radius <= 0.0:
                continue

            center = particle_positions[candidate]
            t_hit, _normal = ray_intersect_particle_sphere(ray_origin, ray_direction, center, radius)

            if t_hit < 0.0:
                continue

            if t_hit > max_distance:
                continue

            if t_hit < best:
                hit_distances[pixel_y, pixel_x] = t_hit
                best = t_hit

        s += 1
        t += step
