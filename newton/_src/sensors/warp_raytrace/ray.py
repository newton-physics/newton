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

from typing import Any

import warp as wp


class vec6f(wp.types.vector(length=6, dtype=wp.float32)):
    pass


EPSILON = 1e-6


@wp.func
def safe_div(x: Any, y: Any) -> Any:
    return x / wp.where(y != 0.0, y, EPSILON)


@wp.func
def map_ray_to_local(
    pos: wp.vec3f, mat: wp.mat33f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.vec3f, wp.vec3f]:
    """Maps ray to local geom frame coordinates.

    Args:
            pos: position of geom frame
            mat: orientation of geom frame
            ray_origin_world: starting point of ray in world coordinates
            ray_direction_world: direction of ray in world coordinates

    Returns:
            3D point and 3D direction in local geom frame
    """

    mat_transposed = wp.transpose(mat)
    ray_origin_local = mat_transposed @ (ray_origin_world - pos)
    ray_direction_local = mat_transposed @ ray_direction_world
    return ray_origin_local, ray_direction_local


@wp.func
def ray_compute_quadratic(a: wp.float32, b: wp.float32, c: wp.float32) -> tuple[wp.float32, wp.vec2f]:
    """Compute solutions from quadratic: a*x^2 + 2*b*x + c = 0."""
    det = b * b - a * c
    if det < EPSILON:
        return wp.inf, wp.vec2f(wp.inf, wp.inf)
    det = wp.sqrt(det)

    # compute the two solutions
    den = safe_div(1.0, a)
    x0 = (-b - det) * den
    x1 = (-b + det) * den
    x = wp.vec2f(x0, x1)

    # finalize result
    if x0 >= 0.0:
        return x0, x
    elif x1 >= 0.0:
        return x1, x
    else:
        return wp.inf, x


@wp.func
def ray_plane(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> wp.float32:
    """Returns the distance at which a ray intersects with a plane."""

    # map to local frame
    ray_origin_local, ray_direction_local = map_ray_to_local(pos, mat, ray_origin_world, ray_direction_world)

    # z-vec not pointing towards front face: reject
    if ray_direction_local[2] > -EPSILON:
        return wp.inf

    # intersection with plane
    t_hit = -ray_origin_local[2] / ray_direction_local[2]
    if t_hit < 0.0:
        return wp.inf

    p = wp.vec2f(
        ray_origin_local[0] + t_hit * ray_direction_local[0], ray_origin_local[1] + t_hit * ray_direction_local[1]
    )

    # accept only within rendered rectangle
    if (size[0] <= 0.0 or wp.abs(p[0]) <= size[0]) and (size[1] <= 0.0 or wp.abs(p[1]) <= size[1]):
        return t_hit
    else:
        return wp.inf


@wp.func
def ray_plane_with_normal(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.bool, wp.float32, wp.vec3f]:
    """Returns distance and normal at which a ray intersects with a plane."""
    t_hit = ray_plane(pos, mat, size, ray_origin_world, ray_direction_world)
    if t_hit == wp.inf:
        return False, wp.inf, wp.vec3f(0.0, 0.0, 0.0)
    # Local plane normal is +Z; rotate to world space
    normal_world = mat @ wp.vec3f(0.0, 0.0, 1.0)
    normal_world = wp.normalize(normal_world)
    return True, t_hit, normal_world


@wp.func
def ray_sphere(
    pos: wp.vec3f, dist_sqr: wp.float32, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> wp.float32:
    """Returns the distance at which a ray intersects with a sphere."""
    dif = ray_origin_world - pos

    a = wp.dot(ray_direction_world, ray_direction_world)
    b = wp.dot(ray_direction_world, dif)
    c = wp.dot(dif, dif) - dist_sqr

    t_hit, _ = ray_compute_quadratic(a, b, c)
    return t_hit


@wp.func
def ray_sphere_with_normal(
    pos: wp.vec3f, dist_sqr: wp.float32, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.bool, wp.float32, wp.vec3f]:
    """Returns distance and normal at which a ray intersects with a sphere."""
    t_hit = ray_sphere(pos, dist_sqr, ray_origin_world, ray_direction_world)
    if t_hit == wp.inf:
        return False, wp.inf, wp.vec3f(0.0, 0.0, 0.0)
    normal = wp.normalize(ray_origin_world + t_hit * ray_direction_world - pos)
    return True, t_hit, normal


@wp.func
def ray_capsule(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> wp.float32:
    """Returns the distance at which a ray intersects with a capsule."""

    # bounding sphere test
    ssz = size[0] + size[1]
    if ray_sphere(pos, ssz * ssz, ray_origin_world, ray_direction_world) == wp.inf:
        return wp.inf

    # map to local frame
    ray_origin_local, ray_direction_local = map_ray_to_local(pos, mat, ray_origin_world, ray_direction_world)

    d_len_sq = wp.dot(ray_direction_local, ray_direction_local)
    if d_len_sq < EPSILON:
        return wp.inf

    inv_d_len = 1.0 / wp.sqrt(d_len_sq)
    d_local_norm = ray_direction_local * inv_d_len

    min_t = 1.0e10
    radius = size[0]
    height = size[1]

    # Intersection with cylinder body
    a_cyl = d_local_norm[0] * d_local_norm[0] + d_local_norm[1] * d_local_norm[1]
    if a_cyl > EPSILON:
        b_cyl = 2.0 * (ray_origin_local[0] * d_local_norm[0] + ray_origin_local[1] * d_local_norm[1])
        c_cyl = ray_origin_local[0] * ray_origin_local[0] + ray_origin_local[1] * ray_origin_local[1] - radius * radius
        delta_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl
        if delta_cyl >= 0.0:
            sqrt_delta_cyl = wp.sqrt(delta_cyl)
            t1 = (-b_cyl - sqrt_delta_cyl) / (2.0 * a_cyl)
            if t1 >= 0.0:
                z = ray_origin_local[2] + t1 * d_local_norm[2]
                if wp.abs(z) <= height:
                    min_t = wp.min(min_t, t1)

            t2 = (-b_cyl + sqrt_delta_cyl) / (2.0 * a_cyl)
            if t2 >= 0.0:
                z = ray_origin_local[2] + t2 * d_local_norm[2]
                if wp.abs(z) <= height:
                    min_t = wp.min(min_t, t2)

    # Intersection with sphere caps
    # Top cap
    oc_top = ray_origin_local - wp.vec3f(0.0, 0.0, height)
    b_top = wp.dot(oc_top, d_local_norm)
    c_top = wp.dot(oc_top, oc_top) - radius * radius
    delta_top = b_top * b_top - c_top
    if delta_top >= 0.0:
        sqrt_delta_top = wp.sqrt(delta_top)
        t1_top = -b_top - sqrt_delta_top
        if t1_top >= 0.0:
            if (ray_origin_local[2] + t1_top * d_local_norm[2]) >= height:
                min_t = wp.min(min_t, t1_top)

        t2_top = -b_top + sqrt_delta_top
        if t2_top >= 0.0:
            if (ray_origin_local[2] + t2_top * d_local_norm[2]) >= height:
                min_t = wp.min(min_t, t2_top)

    # Bottom cap
    oc_bot = ray_origin_local - wp.vec3f(0.0, 0.0, -height)
    b_bot = wp.dot(oc_bot, d_local_norm)
    c_bot = wp.dot(oc_bot, oc_bot) - radius * radius
    delta_bot = b_bot * b_bot - c_bot
    if delta_bot >= 0.0:
        sqrt_delta_bot = wp.sqrt(delta_bot)
        t1_bot = -b_bot - sqrt_delta_bot
        if t1_bot >= 0.0:
            if (ray_origin_local[2] + t1_bot * d_local_norm[2]) <= -height:
                min_t = wp.min(min_t, t1_bot)

        t2_bot = -b_bot + sqrt_delta_bot
        if t2_bot >= 0.0:
            if (ray_origin_local[2] + t2_bot * d_local_norm[2]) <= -height:
                min_t = wp.min(min_t, t2_bot)

    if min_t < 1.0e9:
        return min_t * inv_d_len
    return wp.inf


@wp.func
def ray_capsule_with_normal(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.bool, wp.float32, wp.vec3f]:
    """Returns distance and normal at which a ray intersects with a capsule."""
    t_hit = ray_capsule(pos, mat, size, ray_origin_world, ray_direction_world)
    if t_hit == wp.inf:
        return False, wp.inf, wp.vec3f(0.0, 0.0, 0.0)

    # Compute continuous normal: vector from closest point on axis segment to the hit point
    ray_origin_local, ray_direction_local = map_ray_to_local(pos, mat, ray_origin_world, ray_direction_world)
    hit_local = ray_origin_local + t_hit * ray_direction_local
    z_clamped = wp.min(size[1], wp.max(-size[1], hit_local[2]))
    axis_point = wp.vec3f(0.0, 0.0, z_clamped)
    normal_local = wp.normalize(hit_local - axis_point)
    normal_world = mat @ normal_local
    normal_world = wp.normalize(normal_world)
    return True, t_hit, normal_world


@wp.func
def ray_ellipsoid(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> wp.float32:
    """Returns the distance at which a ray intersects with an ellipsoid."""

    # map to local frame
    ray_origin_local, ray_direction_local = map_ray_to_local(pos, mat, ray_origin_world, ray_direction_world)

    # invert size^2
    s = wp.vec3f(safe_div(1.0, size[0] * size[0]), safe_div(1.0, size[1] * size[1]), safe_div(1.0, size[2] * size[2]))

    # (x * ray_direction_local + ray_origin_local)' * diag(1 / size^2) * (x * ray_direction_local + ray_origin_local) = 1
    slvec = wp.cw_mul(s, ray_direction_local)
    a = wp.dot(slvec, ray_direction_local)
    b = wp.dot(slvec, ray_origin_local)
    c = wp.dot(wp.cw_mul(s, ray_origin_local), ray_origin_local) - 1.0

    # solve a * x^2 + 2 * b * x + c = 0
    t_hit, _ = ray_compute_quadratic(a, b, c)
    return t_hit


@wp.func
def ray_cylinder(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.float32, wp.int32]:
    """Returns the distance at which a ray intersects with a cylinder."""
    # bounding sphere test
    ssz = size[0] * size[0] + size[1] * size[1]
    if ray_sphere(pos, ssz, ray_origin_world, ray_direction_world) == wp.inf:
        return wp.inf, 0

    # map to local frame
    ray_origin_local, ray_direction_local = map_ray_to_local(pos, mat, ray_origin_world, ray_direction_world)

    radius = size[0]
    height = size[1]
    t_hit = wp.inf
    min_t = 1.0e10
    side = 0

    # Intersection with cylinder body
    a_cyl = ray_direction_local[0] * ray_direction_local[0] + ray_direction_local[1] * ray_direction_local[1]
    if a_cyl > EPSILON:
        b_cyl = 2.0 * (ray_origin_local[0] * ray_direction_local[0] + ray_origin_local[1] * ray_direction_local[1])
        c_cyl = ray_origin_local[0] * ray_origin_local[0] + ray_origin_local[1] * ray_origin_local[1] - radius * radius
        delta_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl
        if delta_cyl >= 0.0:
            sqrt_delta_cyl = wp.sqrt(delta_cyl)
            inv_2a = 1.0 / (2.0 * a_cyl)
            t1 = (-b_cyl - sqrt_delta_cyl) * inv_2a
            if t1 >= 0.0:
                z = ray_origin_local[2] + t1 * ray_direction_local[2]
                if wp.abs(z) <= height:
                    min_t = wp.min(min_t, t1)

            t2 = (-b_cyl + sqrt_delta_cyl) * inv_2a
            if t2 >= 0.0:
                z = ray_origin_local[2] + t2 * ray_direction_local[2]
                if wp.abs(z) <= height:
                    min_t = wp.min(min_t, t2)

    # Intersection with caps
    if wp.abs(ray_direction_local[2]) > EPSILON:
        inv_d_z = 1.0 / ray_direction_local[2]
        # Top cap
        t_top = (height - ray_origin_local[2]) * inv_d_z
        if t_top >= 0.0:
            x = ray_origin_local[0] + t_top * ray_direction_local[0]
            y = ray_origin_local[1] + t_top * ray_direction_local[1]
            if x * x + y * y <= radius * radius:
                if t_top <= min_t:
                    min_t = t_top
                    side = 1

        # Bottom cap
        t_bot = (-height - ray_origin_local[2]) * inv_d_z
        if t_bot >= 0.0:
            x = ray_origin_local[0] + t_bot * ray_direction_local[0]
            y = ray_origin_local[1] + t_bot * ray_direction_local[1]
            if x * x + y * y <= radius * radius:
                if t_bot <= min_t:
                    min_t = t_bot
                    side = -1

    if min_t < 1.0e9:
        t_hit = min_t

    return t_hit, side


@wp.func
def ray_cylinder_with_normal(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.bool, wp.float32, wp.vec3f]:
    """Returns distance and normal at which a ray intersects with a cylinder."""
    t_hit, hit_side = ray_cylinder(pos, mat, size, ray_origin_world, ray_direction_world)
    if t_hit == wp.inf:
        return False, wp.inf, wp.vec3f(0.0, 0.0, 0.0)
    # Compute continuous normal: vector from closest point on axis segment to the hit point
    ray_origin_local, ray_direction_local = map_ray_to_local(pos, mat, ray_origin_world, ray_direction_world)
    hit_local = ray_origin_local + t_hit * ray_direction_local
    normal_local = wp.vec3f(0.0, 0.0, 0.0)
    if hit_side == 0:
        z_clamped = wp.min(size[1], wp.max(-size[1], hit_local[2]))
        axis_point = wp.vec3f(0.0, 0.0, z_clamped)
        normal_local = wp.normalize(hit_local - axis_point)
    else:
        normal_local = wp.vec3f(0.0, 0.0, wp.float32(hit_side))
    normal_world = mat @ normal_local
    normal_world = wp.normalize(normal_world)
    return True, t_hit, normal_world


@wp.func
def ray_cone(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> wp.float32:
    """Returns the distance at which a ray intersects with a cone."""
    # bounding sphere test
    ssz = size[0] * size[0] + size[1] * size[1]
    if ray_sphere(pos, ssz, ray_origin_world, ray_direction_world) == wp.inf:
        return wp.inf

    # map to local frame
    ray_origin_local, ray_direction_local = map_ray_to_local(pos, mat, ray_origin_world, ray_direction_world)

    half_height = size[1]
    radius = size[0]

    point_a = wp.vec3f(0.0, 0.0, +half_height)  # tip at +half_height
    point_b = wp.vec3f(0.0, 0.0, -half_height)  # base center at -half_height
    radius_a = 0.0
    radius_b = radius

    ba = point_b - point_a
    oa = ray_origin_local - point_a
    ob = ray_origin_local - point_b
    m0 = wp.dot(ba, ba)
    m1 = wp.dot(oa, ba)
    m2 = wp.dot(ray_direction_local, ba)
    m3 = wp.dot(ray_direction_local, oa)
    m5 = wp.dot(oa, oa)
    m9 = wp.dot(ob, ba)

    # caps
    if m1 < 0.0:
        temp = oa * m2 - ray_direction_local * m1
        if wp.dot(temp, temp) < (radius_a * radius_a * m2 * m2):
            if wp.abs(m2) > EPSILON:
                return -m1 / m2
    elif m9 > 0.0:
        if wp.abs(m2) > EPSILON:
            t_hit = -m9 / m2
            temp_ob = ob + ray_direction_local * t_hit
            if wp.dot(temp_ob, temp_ob) < (radius_b * radius_b):
                return t_hit

    # body
    rr = radius_a - radius_b
    hy = m0 + rr * rr
    k2 = m0 * m0 - m2 * m2 * hy
    k1 = m0 * m0 * m3 - m1 * m2 * hy + m0 * radius_a * (rr * m2 * 1.0)
    k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * radius_a * (rr * m1 * 2.0 - m0 * radius_a)
    h = k1 * k1 - k2 * k0

    if h < 0.0:
        return wp.inf  # no intersection

    if wp.abs(k2) < EPSILON:
        return wp.inf  # degenerate case

    t_hit = (-k1 - wp.sqrt(h)) / k2
    y = m1 + t_hit * m2

    if y < 0.0 or y > m0:
        return wp.inf  # no intersection

    return t_hit


@wp.func
def ray_cone_with_normal(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.bool, wp.float32, wp.vec3f]:
    """Returns distance and normal at which a ray intersects with a cone."""
    t_hit = ray_cone(pos, mat, size, ray_origin_world, ray_direction_world)
    if t_hit == wp.inf:
        return False, wp.inf, wp.vec3f(0.0, 0.0, 0.0)

    ray_origin_local, ray_direction_local = map_ray_to_local(pos, mat, ray_origin_world, ray_direction_world)
    hit_local = ray_origin_local + t_hit * ray_direction_local
    half_height = size[1]
    radius = size[0]

    if wp.abs(hit_local[2] - half_height) <= EPSILON:
        normal_local = wp.vec3f(0.0, 0.0, 1.0)
    elif wp.abs(hit_local[2] + half_height) <= EPSILON:
        normal_local = wp.vec3f(0.0, 0.0, -1.0)
    else:
        radial_sq = hit_local[0] * hit_local[0] + hit_local[1] * hit_local[1]
        radial = wp.sqrt(radial_sq)
        if radial <= EPSILON:
            normal_local = wp.vec3f(0.0, 0.0, 1.0)
        else:
            denom = wp.max(2.0 * wp.abs(half_height), EPSILON)
            slope = radius / denom
            normal_local = wp.vec3f(hit_local[0], hit_local[1], slope * radial)
            normal_local = wp.normalize(normal_local)

    normal_world = mat @ normal_local
    normal_world = wp.normalize(normal_world)
    return True, t_hit, normal_world


_IFACE = wp.types.matrix((3, 2), dtype=wp.int32)(1, 2, 0, 2, 0, 1)


@wp.func
def ray_box(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.float32, vec6f]:
    """Returns the distance at which a ray intersects with a box."""
    all = vec6f(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

    # bounding sphere test
    ssz = wp.dot(size, size)
    if ray_sphere(pos, ssz, ray_origin_world, ray_direction_world) == wp.inf:
        return wp.inf, all

    # map to local frame
    ray_origin_local, ray_direction_local = map_ray_to_local(pos, mat, ray_origin_world, ray_direction_world)

    # init solution
    t_hit = wp.inf

    # loop over axes with non-zero vec
    for i in range(3):
        if wp.abs(ray_direction_local[i]) > EPSILON:
            for side in range(-1, 2, 2):
                # solution of: ray_origin_local[i] + t_hit * ray_direction_local[i] = side * size[i]
                sol = (wp.float32(side) * size[i] - ray_origin_local[i]) / ray_direction_local[i]

                # process if non-negative
                if sol >= 0.0:
                    id0 = _IFACE[i][0]
                    id1 = _IFACE[i][1]

                    # intersection with face
                    p0 = ray_origin_local[id0] + sol * ray_direction_local[id0]
                    p1 = ray_origin_local[id1] + sol * ray_direction_local[id1]

                    # accept within rectangle
                    if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):
                        # update
                        if (t_hit < 0.0) or (sol < t_hit):
                            t_hit = sol

                        # save in all
                        all[2 * i + (side + 1) // 2] = sol

    return t_hit, all


@wp.func
def ray_box_with_normal(
    pos: wp.vec3f, mat: wp.mat33f, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.bool, wp.float32, wp.vec3f]:
    """Returns distance and normal at which a ray intersects with a box."""
    t_hit, all = ray_box(pos, mat, size, ray_origin_world, ray_direction_world)
    if t_hit == wp.inf:
        return False, wp.inf, wp.vec3f(0.0, 0.0, 0.0)

    # Select the face by matching the closest intersection time among the 6 faces
    normal_local = wp.vec3f(0.0, 0.0, 0.0)
    found = wp.bool(False)
    for i in range(3):
        for k in range(2):  # k=0 => -side, k=1 => +side
            t = all[2 * i + k]
            if t >= 0.0 and wp.abs(t - t_hit) < EPSILON:
                normal_local[i] = -1.0 if k == 0 else 1.0
                found = True
                break
        if found:
            break

    normal_world = mat @ normal_local
    normal_world = wp.normalize(normal_world)
    return True, t_hit, normal_world


@wp.func
def ray_mesh(
    mesh_id: wp.uint64,
    ray_origin_world: wp.vec3f,
    ray_direction_world: wp.vec3f,
    max_t: wp.float32,
    double_sided: wp.bool,
) -> tuple[wp.bool, wp.float32, wp.vec3f, wp.float32, wp.float32, wp.int32]:
    """Returns intersection information at which a ray intersects with a mesh.

    Requires wp.Mesh be constructed and their ids to be passed"""

    query = wp.mesh_query_ray(mesh_id, ray_origin_world, ray_direction_world, max_t)
    if query.result:
        return True, query.t, wp.normalize(query.normal), query.u, query.v, query.face

    if double_sided:
        if max_t == wp.inf:
            max_t = 1000.0

        query = wp.mesh_query_ray(mesh_id, ray_origin_world + ray_direction_world * max_t, -ray_direction_world, max_t)
        if query.result:
            return True, max_t - query.t, wp.normalize(query.normal), query.u, query.v, query.face

    return False, wp.inf, wp.vec3f(0.0, 0.0, 0.0), 0.0, 0.0, -1


@wp.func
def scale_mat(scale: wp.vec3f) -> wp.mat33f:
    return wp.mat33f(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2])


@wp.func
def inv_cale_mat(scale: wp.vec3f) -> wp.mat33f:
    return wp.mat33f(
        1.0 / scale[0] if scale[0] != 0 else 0.0,
        0.0,
        0.0,
        0.0,
        1.0 / scale[1] if scale[1] != 0 else 0.0,
        0.0,
        0.0,
        0.0,
        1.0 / scale[2] if scale[2] != 0 else 0.0,
    )


@wp.func
def ray_mesh_with_bvh(
    mesh_bvh_ids: wp.array(dtype=wp.uint64),
    mesh_geom_id: wp.int32,
    pos: wp.vec3f,
    mat: wp.mat33f,
    size: wp.vec3f,
    ray_origin_world: wp.vec3f,
    ray_direction_world: wp.vec3f,
    max_t: wp.float32,
) -> tuple[wp.bool, wp.float32, wp.vec3f, wp.float32, wp.float32, wp.int32, wp.int32]:
    """Returns intersection information at which a ray intersects with a mesh.

    Requires wp.Mesh be constructed and their ids to be passed"""

    ray_origin_local, ray_direction_local = map_ray_to_local(
        pos, inv_cale_mat(size) @ mat, ray_origin_world, ray_direction_world
    )
    query = wp.mesh_query_ray(mesh_bvh_ids[mesh_geom_id], ray_origin_local, ray_direction_local, max_t)

    if query.result and wp.dot(ray_direction_local, query.normal) < 0.0:  # Backface culling in local space
        normal = mat @ query.normal
        normal = wp.normalize(normal)
        return True, query.t, normal, query.u, query.v, query.face, mesh_geom_id

    return False, wp.inf, wp.vec3f(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1
