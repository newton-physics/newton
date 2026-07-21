# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Some ray intersection functions are adapted from https://iquilezles.org/articles/intersectors/

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..core import MAXVAL
from .types import GeoType

if TYPE_CHECKING:
    from ..sim import Model

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
def map_ray_to_local(
    transform: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    scale: wp.vec3 = wp.vec3(1.0),  # noqa: B008  # Warp bakes @wp.func defaults at compile time
) -> tuple[wp.vec3, wp.vec3]:
    """Maps a ray from world space into the local shape frame, dividing by per-axis scale.

    Args:
        transform: World transform of the shape.
        ray_origin: Starting point of the ray in world space.
        ray_direction: Direction of the ray in world space.
        scale: Per-axis scale of the shape; the local ray is divided by this.

    Returns:
        Tuple of (ray_origin_local, ray_direction_local) in the shape's local frame.
    """
    inv_transform = wp.transform_inverse(transform)
    ray_origin_local = wp.transform_point(inv_transform, ray_origin)
    ray_direction_local = wp.transform_vector(inv_transform, ray_direction)
    if scale[0] != 1.0 or scale[1] != 1.0 or scale[2] != 1.0:
        inv_size = safe_div_vec3(wp.vec3(1.0), scale)
        ray_origin_local = wp.cw_mul(ray_origin_local, inv_size)
        ray_direction_local = wp.cw_mul(ray_direction_local, inv_size)
    return ray_origin_local, ray_direction_local


@wp.func
def ray_intersect_sphere(ray_origin: wp.vec3, ray_direction: wp.vec3, r: float) -> tuple[float, wp.vec3]:
    """Computes ray-sphere intersection in the sphere's local frame.

    Args:
        ray_origin: The origin of the ray in the sphere's local frame.
        ray_direction: The direction of the ray in the sphere's local frame.
        r: The radius of the sphere.

    Returns:
        The distance and local-space normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    d_len_sq = wp.dot(ray_direction, ray_direction)
    if d_len_sq < MINVAL:
        return t_hit, normal

    inv_d_len = 1.0 / wp.sqrt(d_len_sq)
    d_local_norm = ray_direction * inv_d_len

    oc = ray_origin
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
        normal = wp.normalize(ray_origin + t_hit * ray_direction)

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
def ray_intersect_ellipsoid(ray_origin: wp.vec3, ray_direction: wp.vec3, semi_axes: wp.vec3) -> tuple[float, wp.vec3]:
    """Computes ray-ellipsoid intersection in the ellipsoid's local frame.

    The ellipsoid is defined by semi-axes (a, b, c) along the local X, Y, Z axes respectively.
    Based on Inigo Quilez's ellipsoid intersection algorithm.

    Args:
        ray_origin: The origin of the ray in the ellipsoid's local frame.
        ray_direction: The direction of the ray in the ellipsoid's local frame.
        semi_axes: The semi-axes (a, b, c) of the ellipsoid.

    Returns:
        The distance and local-space normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    # Reject degenerate rays (matching sphere/capsule pattern)
    d_len_sq = wp.dot(ray_direction, ray_direction)
    if d_len_sq < MINVAL:
        return t_hit, normal

    ra = semi_axes

    # Ensure semi-axes are valid
    if ra[0] < MINVAL or ra[1] < MINVAL or ra[2] < MINVAL:
        return t_hit, normal

    # Scale by inverse semi-axes (transforms ellipsoid to unit sphere)
    ocn = wp.cw_div(ray_origin, ra)
    rdn = wp.cw_div(ray_direction, ra)

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
        hit_local = ray_origin + t_hit * ray_direction
        inv_size = safe_div_vec3(wp.vec3(1.0), semi_axes)
        inv_size_sq = wp.cw_mul(inv_size, inv_size)
        normal = wp.normalize(wp.cw_mul(hit_local, inv_size_sq))

    return t_hit, normal


_IFACE = wp.types.matrix((3, 2), dtype=wp.int32)(1, 2, 0, 2, 0, 1)


@wp.func
def ray_intersect_box(ray_origin: wp.vec3, ray_direction: wp.vec3, size: wp.vec3) -> tuple[float, wp.vec3]:
    """Computes ray-box intersection in the box's local frame.

    Args:
        ray_origin: The origin of the ray in the box's local frame.
        ray_direction: The direction of the ray in the box's local frame.
        size: The half-extents of the box.

    Returns:
        The distance and local-space normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)
    t_near = -1.0e10
    t_far = 1.0e10
    hit = 1

    for i in range(3):
        if wp.abs(ray_direction[i]) < MINVAL:
            if ray_origin[i] < -size[i] or ray_origin[i] > size[i]:
                hit = 0
        else:
            inv_d_i = 1.0 / ray_direction[i]
            t1 = (-size[i] - ray_origin[i]) * inv_d_i
            t2 = (size[i] - ray_origin[i]) * inv_d_i

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
            if wp.abs(ray_direction[i]) > EPSILON:
                for side in range(-1, 2, 2):
                    sol = (float(side) * size[i] - ray_origin[i]) / ray_direction[i]
                    if sol >= 0.0:
                        id0 = _IFACE[i][0]
                        id1 = _IFACE[i][1]
                        p0 = ray_origin[id0] + sol * ray_direction[id0]
                        p1 = ray_origin[id1] + sol * ray_direction[id1]
                        if wp.abs(p0) <= size[id0] and wp.abs(p1) <= size[id1]:
                            if wp.abs(sol - t_hit) < EPSILON:
                                normal_local[i] = -1.0 if side < 0 else 1.0
        normal = wp.normalize(normal_local)

    return t_hit, normal


@wp.func
def ray_intersect_capsule(ray_origin: wp.vec3, ray_direction: wp.vec3, r: float, h: float) -> tuple[float, wp.vec3]:
    """Computes ray-capsule intersection in the capsule's local frame.

    The capsule is centered at the local origin with its cylindrical axis along local Z.

    Args:
        ray_origin: The origin of the ray in the capsule's local frame.
        ray_direction: The direction of the ray in the capsule's local frame.
        r: The radius of the capsule.
        h: The half-height of the capsule's cylindrical part.

    Returns:
        The distance and local-space normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    d_len_sq = wp.dot(ray_direction, ray_direction)
    if d_len_sq < MINVAL:
        return t_hit, normal

    inv_d_len = 1.0 / wp.sqrt(d_len_sq)
    d_local_norm = ray_direction * inv_d_len

    min_t = 1.0e10

    # Intersection with cylinder body
    a_cyl = d_local_norm[0] * d_local_norm[0] + d_local_norm[1] * d_local_norm[1]
    if a_cyl > MINVAL:
        b_cyl = 2.0 * (ray_origin[0] * d_local_norm[0] + ray_origin[1] * d_local_norm[1])
        c_cyl = ray_origin[0] * ray_origin[0] + ray_origin[1] * ray_origin[1] - r * r
        delta_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl
        if delta_cyl >= 0.0:
            sqrt_delta_cyl = wp.sqrt(delta_cyl)
            t1 = (-b_cyl - sqrt_delta_cyl) / (2.0 * a_cyl)
            if t1 >= 0.0:
                z = ray_origin[2] + t1 * d_local_norm[2]
                if wp.abs(z) <= h:
                    min_t = wp.min(min_t, t1)

            t2 = (-b_cyl + sqrt_delta_cyl) / (2.0 * a_cyl)
            if t2 >= 0.0:
                z = ray_origin[2] + t2 * d_local_norm[2]
                if wp.abs(z) <= h:
                    min_t = wp.min(min_t, t2)

    # Intersection with sphere caps
    # Top cap
    oc_top = ray_origin - wp.vec3(0.0, 0.0, h)
    b_top = wp.dot(oc_top, d_local_norm)
    c_top = wp.dot(oc_top, oc_top) - r * r
    delta_top = b_top * b_top - c_top
    if delta_top >= 0.0:
        sqrt_delta_top = wp.sqrt(delta_top)
        t1_top = -b_top - sqrt_delta_top
        if t1_top >= 0.0:
            if (ray_origin[2] + t1_top * d_local_norm[2]) >= h:
                min_t = wp.min(min_t, t1_top)

        t2_top = -b_top + sqrt_delta_top
        if t2_top >= 0.0:
            if (ray_origin[2] + t2_top * d_local_norm[2]) >= h:
                min_t = wp.min(min_t, t2_top)

    # Bottom cap
    oc_bot = ray_origin - wp.vec3(0.0, 0.0, -h)
    b_bot = wp.dot(oc_bot, d_local_norm)
    c_bot = wp.dot(oc_bot, oc_bot) - r * r
    delta_bot = b_bot * b_bot - c_bot
    if delta_bot >= 0.0:
        sqrt_delta_bot = wp.sqrt(delta_bot)
        t1_bot = -b_bot - sqrt_delta_bot
        if t1_bot >= 0.0:
            if (ray_origin[2] + t1_bot * d_local_norm[2]) <= -h:
                min_t = wp.min(min_t, t1_bot)

        t2_bot = -b_bot + sqrt_delta_bot
        if t2_bot >= 0.0:
            if (ray_origin[2] + t2_bot * d_local_norm[2]) <= -h:
                min_t = wp.min(min_t, t2_bot)

    if min_t < 1.0e9:
        t_hit = min_t * inv_d_len

    if t_hit >= 0.0:
        hit_local = ray_origin + t_hit * ray_direction
        z_clamped = wp.min(h, wp.max(-h, hit_local[2]))
        axis_point = wp.vec3(0.0, 0.0, z_clamped)
        normal = wp.normalize(hit_local - axis_point)

    return t_hit, normal


@wp.func
def ray_intersect_cylinder(ray_origin: wp.vec3, ray_direction: wp.vec3, r: float, h: float) -> tuple[float, wp.vec3]:
    """Computes ray-cylinder intersection in the cylinder's local frame.

    The cylinder is centered at the local origin with its axis along local Z.

    Args:
        ray_origin: The origin of the ray in the cylinder's local frame.
        ray_direction: The direction of the ray in the cylinder's local frame.
        r: The radius of the cylinder.
        h: The half-height of the cylinder.

    Returns:
        The distance and local-space normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)
    min_t = 1.0e10

    # Intersection with cylinder body
    a_cyl = ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]
    if a_cyl > MINVAL:
        b_cyl = 2.0 * (ray_origin[0] * ray_direction[0] + ray_origin[1] * ray_direction[1])
        c_cyl = ray_origin[0] * ray_origin[0] + ray_origin[1] * ray_origin[1] - r * r
        delta_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl
        if delta_cyl >= 0.0:
            sqrt_delta_cyl = wp.sqrt(delta_cyl)
            inv_2a = 1.0 / (2.0 * a_cyl)
            t1 = (-b_cyl - sqrt_delta_cyl) * inv_2a
            if t1 >= 0.0:
                z = ray_origin[2] + t1 * ray_direction[2]
                if wp.abs(z) <= h:
                    min_t = wp.min(min_t, t1)

            t2 = (-b_cyl + sqrt_delta_cyl) * inv_2a
            if t2 >= 0.0:
                z = ray_origin[2] + t2 * ray_direction[2]
                if wp.abs(z) <= h:
                    min_t = wp.min(min_t, t2)

    # Intersection with caps
    if wp.abs(ray_direction[2]) > MINVAL:
        inv_d_z = 1.0 / ray_direction[2]
        # Top cap
        t_top = (h - ray_origin[2]) * inv_d_z
        if t_top >= 0.0:
            x = ray_origin[0] + t_top * ray_direction[0]
            y = ray_origin[1] + t_top * ray_direction[1]
            if x * x + y * y <= r * r:
                min_t = wp.min(min_t, t_top)

        # Bottom cap
        t_bot = (-h - ray_origin[2]) * inv_d_z
        if t_bot >= 0.0:
            x = ray_origin[0] + t_bot * ray_direction[0]
            y = ray_origin[1] + t_bot * ray_direction[1]
            if x * x + y * y <= r * r:
                min_t = wp.min(min_t, t_bot)

    if min_t < 1.0e9:
        t_hit = min_t

    if t_hit >= 0.0:
        hit_local = ray_origin + t_hit * ray_direction
        z_clamped = wp.min(h, wp.max(-h, hit_local[2]))
        if z_clamped >= (h - EPSILON) or z_clamped <= (-h + EPSILON):
            normal_local = wp.vec3(0.0, 0.0, z_clamped)
        else:
            normal_local = wp.normalize(hit_local - wp.vec3(0.0, 0.0, z_clamped))
        normal = wp.normalize(normal_local)

    return t_hit, normal


@wp.func
def ray_intersect_cone(
    ray_origin: wp.vec3, ray_direction: wp.vec3, radius: float, half_height: float
) -> tuple[float, wp.vec3]:
    """Computes ray-cone intersection in the cone's local frame.

    The cone is oriented along the local Z-axis with the tip at +half_height and base at -half_height.

    Args:
        ray_origin: The origin of the ray in the cone's local frame.
        ray_direction: The direction of the ray in the cone's local frame.
        radius: The radius of the cone's base.
        half_height: Half the height of the cone (distance from center to tip/base).

    Returns:
        The distance and local-space normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    if wp.abs(half_height) < MINVAL:
        return t_hit, normal

    if radius <= 0.0:
        return t_hit, normal

    # pa = tip (cone extremes), pb = base center, ra = 0 (tip radius), rb = radius (base radius)
    # Check conventions.rst, section "Newton Collision Primitives"
    pa = wp.vec3(0.0, 0.0, half_height)  # tip at +half_height
    pb = wp.vec3(0.0, 0.0, -half_height)  # base center at -half_height
    ra = 0.0  # radius at tip
    rb = radius  # radius at base

    ba = pb - pa
    oa = ray_origin - pa
    ob = ray_origin - pb
    m0 = wp.dot(ba, ba)
    m1 = wp.dot(oa, ba)
    m2 = wp.dot(ray_direction, ba)
    m3 = wp.dot(ray_direction, oa)
    m5 = wp.dot(oa, oa)
    m9 = wp.dot(ob, ba)

    # caps
    if m1 < 0.0:
        temp = oa * m2 - ray_direction * m1
        if wp.dot(temp, temp) < (ra * ra * m2 * m2):
            if wp.abs(m2) > MINVAL:
                t_hit = -m1 / m2
    elif m9 > 0.0:
        if wp.abs(m2) > MINVAL:
            t = -m9 / m2
            temp_ob = ob + ray_direction * t
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
        hit_local = ray_origin + t_hit * ray_direction
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
        normal = wp.normalize(normal_local)

    return t_hit, normal


@wp.func
def ray_intersect_plane(
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    size: wp.vec3,
    enable_backface_culling: bool = False,
) -> tuple[float, wp.vec3]:
    """Computes ray-plane intersection in the plane's local frame.

    The plane lies at z = 0 in local space with normal along +Z.  ``size`` holds ``(width, length, 0)``:
    the full extents along local X and Y.  A value of ``0`` means infinite in that axis.  By default the
    plane is double-sided: rays approaching from either side register intersections. Set
    ``enable_backface_culling`` to reject rays that approach from behind the +Z normal.

    Args:
        ray_origin: The origin of the ray in the plane's local frame.
        ray_direction: The direction of the ray in the plane's local frame.
        size: ``(width, length, 0)`` -- full extents; ``0`` = infinite.
        enable_backface_culling: When ``True``, reject hits approached from behind the +Z normal.

    Returns:
        The distance and local-space normal of the intersection point along the ray, or -1.0 and a zero vector if there is no intersection.
    """
    t_hit = -1.0
    normal = wp.vec3(0.0)

    # Ray parallel to the plane (or degenerate)
    if wp.abs(ray_direction[2]) < PARALLEL_TOL:
        return t_hit, normal

    # Cull rays approaching from behind (aligned with the +Z normal)
    if enable_backface_culling and ray_direction[2] > 0.0:
        return t_hit, normal

    t = -ray_origin[2] / ray_direction[2]
    if t < 0.0:
        return t_hit, normal

    hit_x = ray_origin[0] + t * ray_direction[0]
    hit_y = ray_origin[1] + t * ray_direction[1]

    half_w = size[0] * 0.5
    half_l = size[1] * 0.5

    if half_w > 0.0 and wp.abs(hit_x) > half_w:
        return t_hit, normal
    if half_l > 0.0 and wp.abs(hit_y) > half_l:
        return t_hit, normal

    t_hit = t
    normal = wp.vec3(0.0, 0.0, 1.0)

    return t_hit, normal


def _make_ray_intersect_mesh(compute_normal: bool):
    """Build a ray-mesh intersection ``@wp.func``, specialized at compile time on whether the
    surface normal is computed. ``wp.static(compute_normal)`` folds away the normal work in the
    depth-only / shadow variant, so no code is duplicated (see :data:`ray_intersect_mesh` and
    :data:`ray_intersect_mesh_no_normal`).
    """

    @wp.func
    def ray_intersect_mesh(
        ray_origin: wp.vec3,
        ray_direction: wp.vec3,
        size: wp.vec3,
        mesh_id: wp.uint64,
        enable_backface_culling: bool,
        max_t: float,
        root: int = -1,
    ) -> tuple[float, wp.vec3, float, float, int]:
        """Computes ray-mesh intersection in the mesh's local frame using Warp's built-in mesh query.

        Args:
            ray_origin: The origin of the ray in the mesh's local frame.
            ray_direction: The direction of the ray in the mesh's local frame.
            size: The 3D scale of the mesh, used to scale-correct the returned local normal.
            mesh_id: The Warp mesh ID for raycasting.
            enable_backface_culling: When ``True``, reject hits whose triangle normal
                is aligned with the ray direction (back faces).
            max_t: Maximum parameter ``t`` along the local ray to consider.
            root: Root node index for grouped mesh traversal, or ``-1`` for the global root.

        Returns:
            Tuple ``(distance, normal, u, v, face_index)``. The distance along the ray and the local-space normal of the intersection point (a zero vector in the ``no_normal`` variant), or -1.0 and a zero vector if there is no intersection; on miss, ``u`` and ``v`` are ``0.0`` and ``face_index`` is -1.
        """
        if mesh_id == wp.uint64(0):
            return -1.0, wp.vec3(0.0), 0.0, 0.0, -1

        query = wp.mesh_query_ray(mesh_id, ray_origin, ray_direction, max_t, root)

        if query.result:
            if not enable_backface_culling or wp.dot(ray_direction, query.normal) < 0.0:
                normal = wp.vec3(0.0)
                if wp.static(compute_normal):
                    normal = wp.normalize(safe_div_vec3(query.normal, size))
                return query.t, normal, query.u, query.v, query.face

        return -1.0, wp.vec3(0.0), 0.0, 0.0, -1

    return ray_intersect_mesh


ray_intersect_mesh = _make_ray_intersect_mesh(True)
ray_intersect_mesh_no_normal = _make_ray_intersect_mesh(False)


@wp.func
def ray_intersect_mesh_anyhit(
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    mesh_id: wp.uint64,
    max_t: float,
    root: int = -1,
) -> float:
    """Tests whether a ray hits a mesh within ``max_t``, in the mesh's local frame.

    Uses Warp's any-hit mesh query, which returns on the first intersection without
    searching for the closest one -- cheaper than :func:`ray_intersect_mesh` for
    occlusion/shadow rays that only need existence of a hit.

    Args:
        ray_origin: The origin of the ray in the mesh's local frame.
        ray_direction: The direction of the ray in the mesh's local frame.
        mesh_id: The Warp mesh ID for raycasting.
        max_t: Maximum parameter ``t`` along the local ray to consider.
        root: Root node index for grouped mesh traversal, or ``-1`` for the global root.

    Returns:
        ``0.0`` if the ray hits the mesh within ``max_t``, otherwise ``-1.0``.
    """
    if mesh_id == wp.uint64(0):
        return -1.0

    hit = wp.mesh_query_ray_anyhit(mesh_id, ray_origin, ray_direction, max_t, root)

    if hit:
        return 0.0

    return -1.0


def _make_ray_intersect_shape(compute_normal: bool):
    """Build the primitive-dispatch ``@wp.func``, specialized at compile time on whether the
    surface normal is computed. ``wp.static(compute_normal)`` folds away the normal transform in
    the depth-only / shadow variant (and, being unused, the primitives' local-normal work is then
    dead-code-eliminated), so no code is duplicated (see :data:`ray_intersect_shape` and
    :data:`ray_intersect_shape_no_normal`).
    """

    @wp.func
    def ray_intersect_shape(
        geom_to_world: wp.transform,
        size: wp.vec3,
        geomtype: int,
        ray_origin: wp.vec3,
        ray_direction: wp.vec3,
        enable_backface_culling: bool,
    ) -> tuple[float, wp.vec3]:
        """Maps a world-space ray into the shape's local frame, dispatches to the matching primitive routine, and rotates the normal back to world.

        Handles primitive shapes only. Mesh-backed shapes (MESH, CONVEX_MESH, HFIELD) are
        intersected via :func:`ray_intersect_mesh`; callers dispatch on the geometry type and
        pick the matching routine.

        Args:
            geom_to_world: The world transform of the shape.
            size: The size of the geometry.
            geomtype: The type of the geometry.
            ray_origin: The origin of the ray in world space.
            ray_direction: The direction of the ray in world space.
            enable_backface_culling: When ``True``, reject plane hits approached from
                behind (ray direction aligned with the plane normal).

        Returns:
            The distance and world-space normal of the intersection point along the ray (a zero vector in the ``no_normal`` variant), or -1.0 and a zero vector if there is no intersection.
        """
        ray_origin_local, ray_direction_local = map_ray_to_local(geom_to_world, ray_origin, ray_direction)

        t_hit = -1.0
        normal_local = wp.vec3(0.0)

        if geomtype == GeoType.PLANE:
            t_hit, normal_local = ray_intersect_plane(
                ray_origin_local, ray_direction_local, size, enable_backface_culling
            )
        elif geomtype == GeoType.SPHERE:
            t_hit, normal_local = ray_intersect_sphere(ray_origin_local, ray_direction_local, size[0])
        elif geomtype == GeoType.BOX:
            t_hit, normal_local = ray_intersect_box(ray_origin_local, ray_direction_local, size)
        elif geomtype == GeoType.CAPSULE:
            t_hit, normal_local = ray_intersect_capsule(ray_origin_local, ray_direction_local, size[0], size[1])
        elif geomtype == GeoType.CYLINDER:
            t_hit, normal_local = ray_intersect_cylinder(ray_origin_local, ray_direction_local, size[0], size[1])
        elif geomtype == GeoType.CONE:
            t_hit, normal_local = ray_intersect_cone(ray_origin_local, ray_direction_local, size[0], size[1])
        elif geomtype == GeoType.ELLIPSOID:
            t_hit, normal_local = ray_intersect_ellipsoid(ray_origin_local, ray_direction_local, size)

        normal = wp.vec3(0.0)
        if wp.static(compute_normal):
            if t_hit >= 0.0:
                normal = wp.normalize(wp.transform_vector(geom_to_world, normal_local))

        return t_hit, normal

    return ray_intersect_shape


ray_intersect_shape = _make_ray_intersect_shape(True)
ray_intersect_shape_no_normal = _make_ray_intersect_shape(False)


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
    """Computes the intersection of a ray with all geometries in the scene.

    HFIELD shapes use their ``wp.Mesh`` ID from ``shape_source_ptr``; the mesh is
    built during :meth:`~newton.ModelBuilder.finalize`.

    Args:
        body_q: Array of body transforms.
        shape_body: Maps shape index to body index.
        shape_transform: Array of local shape transforms.
        geom_type: Array of geometry types for each geometry.
        geom_size: Array of sizes for each geometry.
        shape_source_ptr: Array of mesh IDs for MESH, CONVEX_MESH, and HFIELD geometries (wp.uint64).
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

    if geomtype == GeoType.MESH or geomtype == GeoType.CONVEX_MESH or geomtype == GeoType.HFIELD:
        ray_origin_local, ray_direction_local = map_ray_to_local(
            geom_to_world, ray_origin, ray_direction, geom_size[shape_idx]
        )
        t, _normal, _u, _v, _face = ray_intersect_mesh(
            ray_origin_local,
            ray_direction_local,
            geom_size[shape_idx],
            shape_source_ptr[shape_idx],
            False,
            _DEFAULT_MESH_MAX_T,
        )
    else:
        t, _normal = ray_intersect_shape(
            geom_to_world,
            geom_size[shape_idx],
            geomtype,
            ray_origin,
            ray_direction,
            False,
        )

    if t >= 0.0 and t < min_dist[0]:
        _spinlock_acquire(lock)
        # Still use an atomic inside the spinlock to get a volatile read
        old_min = wp.atomic_min(min_dist, 0, t)
        if t <= old_min:
            min_index[0] = shape_idx
            min_body_index[0] = b
        _spinlock_release(lock)


def intersect_ray(
    model: Model,
    *,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    ray_worlds: wp.array[wp.int32],
    enable_global_world: bool = True,
    out_dist: wp.array[float] | None = None,
    out_shape_id: wp.array[wp.int32] | None = None,
    out_normal: wp.array[wp.vec3] | None = None,
    fast_math: bool = True,
):
    """Intersect rays with model shapes.

    :meth:`~newton.ModelBuilder.finalize` builds the model shape BVH for the
    initial model state. If the queried state changes shape transforms, refit
    the BVH with :meth:`~newton.Model.bvh_refit_shapes` before raycasting
    again. Manually populated models must build the BVH with
    :meth:`~newton.Model.bvh_build_shapes` before first use.

    Each ray is cast against the shapes of its own world (given by
    ``ray_worlds``) and against the shapes of the global world (index ``-1``),
    which are accessible from every world. A ray whose world is ``-1`` is cast
    against the global world only.

    ``out_dist``, ``out_shape_id`` and ``out_normal`` are optional outputs.
    Pass ``None`` to skip writing a channel.

    Args:
        model: Model containing the shapes to query.
        ray_origins: Ray origins in world space [m], shape [ray_count, 3].
        ray_directions: Ray directions in world space, shape [ray_count, 3].
            Values must be normalized and nonzero.
        ray_worlds: Per-ray world index, shape [ray_count]. Use ``-1`` for the
            global world.
        enable_global_world: Whether to enable global world raycasting.
        out_dist: Optional output hit distances [m], shape [ray_count]. ``-1`` on miss.
        out_shape_id: Optional output hit shape indices, shape [ray_count]. ``-1`` on miss.
        out_normal: Optional output hit normals, shape [ray_count, 3].
        fast_math: Whether to compile the intersection kernel with fast math enabled.
    """

    if model.bvh_shapes is None:
        raise RuntimeError(
            "BVH raycasting requires a shape BVH built for the queried state. "
            "ModelBuilder.finalize() builds one for the initial state; call "
            "model.bvh_build_shapes(state) for manually populated models and "
            "model.bvh_refit_shapes(state) after state changes."
        )

    write_dist = out_dist is not None
    write_shape_id = out_shape_id is not None
    write_normal = out_normal is not None

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": fast_math})
    def _intersect_ray_kernel(
        bvh_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        bvh_shape_enabled: wp.array[wp.uint32],
        shape_transform_world: wp.array[wp.transform],
        shape_type: wp.array[int],
        shape_scale: wp.array[wp.vec3],
        shape_source_ptr: wp.array[wp.uint64],
        ray_origin: wp.array[wp.vec3],
        ray_direction: wp.array[wp.vec3],
        ray_world: wp.array[wp.int32],
        out_dist: wp.array[float],
        out_shape_id: wp.array[wp.int32],
        out_normal: wp.array[wp.vec3],
    ):
        rayid = wp.tid()

        origin = ray_origin[rayid]
        direction = ray_direction[rayid]

        min_dist = float(MAXVAL)
        min_shape_id = wp.int32(-1)
        min_normal = wp.vec3(0.0)

        # Pass 0 queries the ray's own world; pass 1 the global world shared by all.
        for i in range(wp.static(2 if enable_global_world else 1)):
            groupid = ray_world[rayid] if i == 0 else bvh_shapes_group_roots.shape[0] - 1

            bvh_root = bvh_shapes_group_roots[groupid]
            if bvh_root < 0:
                continue

            query = wp.bvh_query_ray(bvh_id, origin, direction, bvh_root)
            bvh_shape_id = wp.int32(0)

            while wp.bvh_query_next(query, bvh_shape_id, min_dist):
                shape_id = wp.int32(bvh_shape_enabled[bvh_shape_id])
                geom_type = shape_type[shape_id]

                if geom_type == GeoType.MESH or geom_type == GeoType.CONVEX_MESH or geom_type == GeoType.HFIELD:
                    geom_to_world = shape_transform_world[shape_id]
                    ray_origin_local, ray_direction_local = map_ray_to_local(
                        geom_to_world, origin, direction, shape_scale[shape_id]
                    )
                    hit_dist, hit_normal_local, _u, _v, _face = ray_intersect_mesh(
                        ray_origin_local,
                        ray_direction_local,
                        shape_scale[shape_id],
                        shape_source_ptr[shape_id],
                        False,
                        min_dist,
                    )
                    hit_normal = wp.vec3(0.0)
                    if hit_dist >= 0.0:
                        hit_normal = wp.normalize(wp.transform_vector(geom_to_world, hit_normal_local))
                else:
                    hit_dist, hit_normal = ray_intersect_shape(
                        shape_transform_world[shape_id],
                        shape_scale[shape_id],
                        geom_type,
                        origin,
                        direction,
                        False,
                    )
                if hit_dist >= 0.0 and hit_dist < min_dist:
                    min_dist = hit_dist
                    min_shape_id = shape_id
                    min_normal = hit_normal

        if wp.static(write_dist):
            out_dist[rayid] = wp.where(min_shape_id < 0, -1.0, min_dist)
        if wp.static(write_shape_id):
            out_shape_id[rayid] = min_shape_id
        if wp.static(write_normal):
            out_normal[rayid] = min_normal

    wp.launch(
        kernel=_intersect_ray_kernel,
        dim=ray_origins.shape[0],
        inputs=[
            model.bvh_shapes.id,
            model.bvh_shapes_group_roots,
            model.bvh_shape_enabled,
            model.bvh_shape_world_transforms,
            model.shape_type,
            model.shape_scale,
            model.shape_source_ptr,
            ray_origins,
            ray_directions,
            ray_worlds,
        ],
        outputs=[
            out_dist,
            out_shape_id,
            out_normal,
        ],
        device=model.device,
    )
