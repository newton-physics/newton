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

from ...geometry import raycast

EPSILON = 1e-6


@wp.struct
class GeomHit:
    hit: wp.bool
    distance: wp.float32
    normal: wp.vec3f


@wp.func
def safe_div_vec3(x: wp.vec3f, y: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(
        x[0] / wp.where(y[0] != 0.0, y[0], EPSILON),
        x[1] / wp.where(y[1] != 0.0, y[1], EPSILON),
        x[2] / wp.where(y[2] != 0.0, y[2], EPSILON),
    )


@wp.func
def map_ray_to_local(
    transform: wp.transformf, ray_origin: wp.vec3f, ray_direction: wp.vec3f
) -> tuple[wp.vec3f, wp.vec3f]:
    """Maps ray to local shape frame coordinates.

    Args:
            transform: transform of shape frame
            ray_origin: starting point of ray in world coordinates
            ray_direction: direction of ray in world coordinates

    Returns:
            3D point and 3D direction in local shape frame
    """

    inv_transform = wp.transform_inverse(transform)
    ray_origin_local = wp.transform_point(inv_transform, ray_origin)
    ray_direction_local = wp.transform_vector(inv_transform, ray_direction)
    return ray_origin_local, ray_direction_local


@wp.func
def ray_intersect_plane(
    transform: wp.transformf,
    size: wp.vec3f,
    enable_backface_culling: wp.bool,
    ray_origin: wp.vec3f,
    ray_direction: wp.vec3f,
) -> wp.float32:
    """Returns the distance at which a ray intersects with a plane."""

    # map to local frame
    ray_origin_local, ray_direction_local = map_ray_to_local(transform, ray_origin, ray_direction)

    # z-vec not pointing towards front face: reject
    if enable_backface_culling and ray_direction_local[2] > -EPSILON:
        return -1.0

    # intersection with plane
    t_hit = -ray_origin_local[2] / ray_direction_local[2]
    if t_hit < 0.0:
        return -1.0

    p = wp.vec2f(
        ray_origin_local[0] + t_hit * ray_direction_local[0], ray_origin_local[1] + t_hit * ray_direction_local[1]
    )

    # accept only within rendered rectangle
    if (size[0] <= 0.0 or wp.abs(p[0]) <= size[0]) and (size[1] <= 0.0 or wp.abs(p[1]) <= size[1]):
        return t_hit
    return -1.0


@wp.func
def ray_intersect_plane_with_normal(
    transform: wp.transformf,
    size: wp.vec3f,
    enable_backface_culling: wp.bool,
    ray_origin: wp.vec3f,
    ray_direction: wp.vec3f,
) -> GeomHit:
    """Returns distance and normal at which a ray intersects with a plane."""
    geom_hit = GeomHit()
    geom_hit.distance = ray_intersect_plane(transform, size, enable_backface_culling, ray_origin, ray_direction)
    geom_hit.hit = geom_hit.distance > -1
    if geom_hit.hit:
        geom_hit.normal = wp.transform_vector(transform, wp.vec3f(0.0, 0.0, 1.0))
        geom_hit.normal = wp.normalize(geom_hit.normal)
    return geom_hit


@wp.func
def ray_intersect_sphere_with_normal(
    transform: wp.transformf, size: wp.vec3f, ray_origin: wp.vec3f, ray_direction: wp.vec3f
) -> GeomHit:
    """Returns distance and normal at which a ray intersects with a sphere."""
    geom_hit = GeomHit()
    geom_hit.distance = raycast.ray_intersect_sphere(transform, ray_origin, ray_direction, size[0])
    geom_hit.hit = geom_hit.distance > -1
    if geom_hit.hit:
        geom_hit.normal = wp.normalize(
            ray_origin + geom_hit.distance * ray_direction - wp.transform_get_translation(transform)
        )
    return geom_hit


@wp.func
def ray_intersect_particle_sphere_with_normal(
    ray_origin: wp.vec3f, ray_direction: wp.vec3f, center: wp.vec3f, radius: wp.float32
) -> GeomHit:
    """Returns distance and normal at which a ray intersects with a particle sphere."""
    geom_hit = GeomHit()
    geom_hit.distance = raycast.ray_intersect_particle_sphere(ray_origin, ray_direction, center, radius)
    geom_hit.hit = geom_hit.distance > -1
    if geom_hit.hit:
        geom_hit.normal = wp.normalize(ray_origin + geom_hit.distance * ray_direction - center)
    return geom_hit


@wp.func
def ray_intersect_capsule_with_normal(
    transform: wp.transformf, size: wp.vec3f, ray_origin: wp.vec3f, ray_direction: wp.vec3f
) -> GeomHit:
    """Returns distance and normal at which a ray intersects with a capsule."""
    geom_hit = GeomHit()
    geom_hit.distance = raycast.ray_intersect_capsule(transform, ray_origin, ray_direction, size[0], size[1])
    geom_hit.hit = geom_hit.distance > -1
    if geom_hit.hit:
        ray_origin_local, ray_direction_local = map_ray_to_local(transform, ray_origin, ray_direction)
        hit_local = ray_origin_local + geom_hit.distance * ray_direction_local
        z_clamped = wp.min(size[1], wp.max(-size[1], hit_local[2]))
        axis_point = wp.vec3f(0.0, 0.0, z_clamped)
        normal_local = wp.normalize(hit_local - axis_point)
        geom_hit.normal = wp.transform_vector(transform, normal_local)
        geom_hit.normal = wp.normalize(geom_hit.normal)
    return geom_hit


@wp.func
def ray_intersect_ellipsoid_with_normal(
    transform: wp.transformf, size: wp.vec3f, ray_origin: wp.vec3f, ray_direction: wp.vec3f
) -> GeomHit:
    """Returns the distance and normal at which a ray intersects with an ellipsoid."""
    geom_hit = GeomHit()
    geom_hit.distance = raycast.ray_intersect_ellipsoid(transform, ray_origin, ray_direction, size)
    geom_hit.hit = geom_hit.distance > -1

    if geom_hit.hit:
        ray_origin_local, ray_direction_local = map_ray_to_local(transform, ray_origin, ray_direction)

        inv_size = safe_div_vec3(wp.vec3f(1.0), size)
        ray_origin_local = wp.cw_mul(ray_origin_local, inv_size)
        ray_direction_local = wp.cw_mul(ray_direction_local, inv_size)

        geom_hit.normal = ray_origin_local + geom_hit.distance * ray_direction_local
        geom_hit.normal = wp.transform_vector(transform, geom_hit.normal)
        geom_hit.normal = wp.normalize(geom_hit.normal)
    return geom_hit


@wp.func
def ray_intersect_cylinder_with_normal(
    transform: wp.transformf, size: wp.vec3f, ray_origin: wp.vec3f, ray_direction: wp.vec3f
) -> GeomHit:
    """Returns distance and normal at which a ray intersects with a cylinder."""
    geom_hit = GeomHit()
    geom_hit.distance = raycast.ray_intersect_cylinder(transform, ray_origin, ray_direction, size[0], size[1])
    geom_hit.hit = geom_hit.distance > -1
    if geom_hit.hit:
        ray_origin_local, ray_direction_local = map_ray_to_local(transform, ray_origin, ray_direction)
        hit_local = ray_origin_local + geom_hit.distance * ray_direction_local
        z_clamped = wp.min(size[1], wp.max(-size[1], hit_local[2]))
        if z_clamped >= (size[1] - EPSILON) or z_clamped <= (-size[1] + EPSILON):
            geom_hit.normal = wp.vec3f(0.0, 0.0, z_clamped)
        else:
            geom_hit.normal = wp.normalize(hit_local - wp.vec3f(0.0, 0.0, z_clamped))
        geom_hit.normal = wp.transform_vector(transform, geom_hit.normal)
        geom_hit.normal = wp.normalize(geom_hit.normal)
    return geom_hit


@wp.func
def ray_intersect_cone_with_normal(
    transform: wp.transformf, size: wp.vec3f, ray_origin: wp.vec3f, ray_direction: wp.vec3f
) -> GeomHit:
    """Returns distance and normal at which a ray intersects with a cone."""
    geom_hit = GeomHit()
    geom_hit.distance = raycast.ray_intersect_cone(transform, ray_origin, ray_direction, size[0], size[1])
    geom_hit.hit = geom_hit.distance > -1
    if geom_hit.hit:
        ray_origin_local, ray_direction_local = map_ray_to_local(transform, ray_origin, ray_direction)
        hit_local = ray_origin_local + geom_hit.distance * ray_direction_local
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

        geom_hit.normal = wp.transform_vector(transform, normal_local)
        geom_hit.normal = wp.normalize(geom_hit.normal)
    return geom_hit


_IFACE = wp.types.matrix((3, 2), dtype=wp.int32)(1, 2, 0, 2, 0, 1)


@wp.func
def ray_intersect_box_with_normal(
    transform: wp.transformf, size: wp.vec3f, ray_origin: wp.vec3f, ray_direction: wp.vec3f
) -> GeomHit:
    """Returns distance and normal at which a ray intersects with a box."""
    geom_hit = GeomHit()
    geom_hit.distance = raycast.ray_intersect_box(transform, ray_origin, ray_direction, size)
    geom_hit.hit = geom_hit.distance > -1
    if geom_hit.hit:
        geom_hit.normal = wp.vec3f(0.0)
        ray_origin_local, ray_direction_local = map_ray_to_local(transform, ray_origin, ray_direction)

        for i in range(3):
            if wp.abs(ray_direction_local[i]) > EPSILON:
                for side in range(-1, 2, 2):
                    sol = (wp.float32(side) * size[i] - ray_origin_local[i]) / ray_direction_local[i]

                    if sol >= 0.0:
                        id0 = _IFACE[i][0]
                        id1 = _IFACE[i][1]

                        p0 = ray_origin_local[id0] + sol * ray_direction_local[id0]
                        p1 = ray_origin_local[id1] + sol * ray_direction_local[id1]

                        if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):
                            if sol >= 0.0 and wp.abs(sol - geom_hit.distance) < EPSILON:
                                geom_hit.normal[i] = -1.0 if side < 0 else 1.0
                                geom_hit.normal = wp.transform_vector(transform, geom_hit.normal)
                                geom_hit.normal = wp.normalize(geom_hit.normal)
                                return geom_hit
    return geom_hit


@wp.func
def ray_intersect_mesh_no_transform(
    mesh_id: wp.uint64,
    ray_origin: wp.vec3f,
    ray_direction: wp.vec3f,
    enable_backface_culling: wp.bool,
    max_t: wp.float32,
) -> tuple[GeomHit, wp.float32, wp.float32, wp.int32]:
    """Returns intersection information at which a ray intersects with a mesh.

    Requires wp.Mesh be constructed and their ids to be passed"""

    geom_hit = GeomHit()
    geom_hit.hit = False

    query = wp.mesh_query_ray(mesh_id, ray_origin, ray_direction, max_t)
    if query.result:
        if not enable_backface_culling or wp.dot(ray_direction, query.normal) < 0.0:
            geom_hit.distance = query.t
            geom_hit.normal = wp.normalize(query.normal)
            return geom_hit, query.u, query.v, query.face

    return geom_hit, 0.0, 0.0, -1


@wp.func
def ray_intersect_mesh(
    transform: wp.transformf,
    size: wp.vec3f,
    ray_origin: wp.vec3f,
    ray_direction: wp.vec3f,
    mesh_id: wp.uint64,
    enable_backface_culling: wp.bool,
    max_t: wp.float32,
) -> tuple[GeomHit, wp.float32, wp.float32, wp.int32]:
    """Returns intersection information at which a ray intersects with a mesh.

    Requires wp.Mesh be constructed and their ids to be passed"""

    geom_hit = GeomHit()
    geom_hit.hit = False

    ray_origin_local, ray_direction_local = map_ray_to_local(transform, ray_origin, ray_direction)

    inv_size = safe_div_vec3(wp.vec3f(1.0), size)
    ray_origin_local = wp.cw_mul(ray_origin_local, inv_size)
    ray_direction_local = wp.cw_mul(ray_direction_local, inv_size)

    query = wp.mesh_query_ray(mesh_id, ray_origin_local, ray_direction_local, max_t)

    if query.result:
        if not enable_backface_culling or wp.dot(ray_direction_local, query.normal) < 0.0:
            geom_hit.hit = True
            geom_hit.distance = query.t
            geom_hit.normal = wp.transform_vector(transform, wp.cw_mul(size, query.normal))
            geom_hit.normal = wp.normalize(geom_hit.normal)
            return geom_hit, query.u, query.v, query.face

    return geom_hit, 0.0, 0.0, -1
