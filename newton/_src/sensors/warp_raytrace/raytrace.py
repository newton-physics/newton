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

from ...geometry import GeoType, raycast
from . import ray_intersect

NO_HIT_SHAPE_ID = wp.uint32(0xFFFFFFFF)
MAX_SHAPE_ID = wp.uint32(0xFFFFFFF0)
TRIANGLE_MESH_SHAPE_ID = wp.uint32(0xFFFFFFFD)
PARTICLES_SHAPE_ID = wp.uint32(0xFFFFFFFE)


@wp.struct
class ClosestHit:
    distance: wp.float32
    normal: wp.vec3f
    shape_index: wp.uint32
    bary_u: wp.float32
    bary_v: wp.float32
    face_idx: wp.int32
    shape_mesh_index: wp.int32


@wp.func
def get_group_roots(
    group_roots: wp.array(dtype=wp.int32), world_index: wp.int32, want_global_world: wp.int32
) -> wp.int32:
    if want_global_world != 0:
        return group_roots[group_roots.shape[0] - 1]
    return group_roots[world_index]


@wp.func
def closest_hit_shape(
    closest_hit: ClosestHit,
    bvh_shapes_size: wp.int32,
    bvh_shapes_id: wp.uint64,
    bvh_shapes_group_roots: wp.array(dtype=wp.int32),
    world_index: wp.int32,
    enable_global_world: wp.bool,
    shape_enabled: wp.array(dtype=wp.uint32),
    shape_types: wp.array(dtype=wp.int32),
    shape_mesh_indices: wp.array(dtype=wp.int32),
    shape_sizes: wp.array(dtype=wp.vec3f),
    shape_transforms: wp.array(dtype=wp.transformf),
    mesh_ids: wp.array(dtype=wp.uint64),
    enable_backface_culling: wp.bool,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
) -> ClosestHit:
    if bvh_shapes_size:
        for i in range(2 if enable_global_world else 1):
            group_root = get_group_roots(bvh_shapes_group_roots, world_index, i)
            if group_root < 0:
                continue

            query = wp.bvh_query_ray(bvh_shapes_id, ray_origin_world, ray_dir_world, group_root)
            shape_index = wp.int32(0)

            while wp.bvh_query_next(query, shape_index, closest_hit.distance):
                si = shape_enabled[shape_index]

                geom_hit = ray_intersect.GeomHit()
                hit_u = wp.float32(0.0)
                hit_v = wp.float32(0.0)
                hit_face_id = wp.int32(-1)
                hit_mesh_id = wp.int32(-1)

                if shape_types[si] == GeoType.MESH:
                    hit_mesh_id = shape_mesh_indices[si]
                    geom_hit, hit_u, hit_v, hit_face_id = ray_intersect.ray_intersect_mesh(
                        shape_transforms[si],
                        shape_sizes[si],
                        ray_origin_world,
                        ray_dir_world,
                        mesh_ids[hit_mesh_id],
                        enable_backface_culling,
                        closest_hit.distance,
                    )
                elif shape_types[si] == GeoType.PLANE:
                    geom_hit = ray_intersect.ray_intersect_plane_with_normal(
                        shape_transforms[si],
                        shape_sizes[si],
                        enable_backface_culling,
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif shape_types[si] == GeoType.SPHERE:
                    geom_hit = ray_intersect.ray_intersect_sphere_with_normal(
                        shape_transforms[si],
                        shape_sizes[si],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif shape_types[si] == GeoType.ELLIPSOID:
                    geom_hit = ray_intersect.ray_intersect_ellipsoid_with_normal(
                        shape_transforms[si],
                        shape_sizes[si],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif shape_types[si] == GeoType.CAPSULE:
                    geom_hit = ray_intersect.ray_intersect_capsule_with_normal(
                        shape_transforms[si],
                        shape_sizes[si],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif shape_types[si] == GeoType.CYLINDER:
                    geom_hit = ray_intersect.ray_intersect_cylinder_with_normal(
                        shape_transforms[si],
                        shape_sizes[si],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif shape_types[si] == GeoType.CONE:
                    geom_hit = ray_intersect.ray_intersect_cone_with_normal(
                        shape_transforms[si],
                        shape_sizes[si],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif shape_types[si] == GeoType.BOX:
                    geom_hit = ray_intersect.ray_intersect_box_with_normal(
                        shape_transforms[si],
                        shape_sizes[si],
                        ray_origin_world,
                        ray_dir_world,
                    )

                if geom_hit.hit and geom_hit.distance < closest_hit.distance:
                    closest_hit.distance = geom_hit.distance
                    closest_hit.normal = geom_hit.normal
                    closest_hit.shape_index = si
                    closest_hit.bary_u = hit_u
                    closest_hit.bary_v = hit_v
                    closest_hit.face_idx = hit_face_id
                    closest_hit.shape_mesh_index = hit_mesh_id

    return closest_hit


@wp.func
def closest_hit_particles(
    closest_hit: ClosestHit,
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    world_index: wp.int32,
    enable_global_world: wp.bool,
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
) -> ClosestHit:
    if bvh_particles_size:
        for i in range(2 if enable_global_world else 1):
            group_root = get_group_roots(bvh_particles_group_roots, world_index, i)
            if group_root < 0:
                continue

            query = wp.bvh_query_ray(bvh_particles_id, ray_origin_world, ray_dir_world, group_root)
            si = wp.int32(0)

            while wp.bvh_query_next(query, si, closest_hit.distance):
                geom_hit = ray_intersect.ray_intersect_particle_sphere_with_normal(
                    ray_origin_world,
                    ray_dir_world,
                    particles_position[si],
                    particles_radius[si],
                )

                if geom_hit.hit and geom_hit.distance < closest_hit.distance:
                    closest_hit.distance = geom_hit.distance
                    closest_hit.normal = geom_hit.normal
                    closest_hit.shape_index = PARTICLES_SHAPE_ID
                    closest_hit.shape_mesh_index = -1

    return closest_hit


@wp.func
def closest_hit_triangle_mesh(
    closest_hit: ClosestHit,
    triangle_mesh_id: wp.uint64,
    enable_backface_culling: wp.bool,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
) -> ClosestHit:
    if triangle_mesh_id:
        geom_hit, bary_u, bary_v, face_idx = ray_intersect.ray_intersect_mesh_no_transform(
            triangle_mesh_id, ray_origin_world, ray_dir_world, enable_backface_culling, closest_hit.distance
        )
        if geom_hit.hit:
            closest_hit.distance = geom_hit.distance
            closest_hit.normal = geom_hit.normal
            closest_hit.shape_index = TRIANGLE_MESH_SHAPE_ID
            closest_hit.bary_u = bary_u
            closest_hit.bary_v = bary_v
            closest_hit.face_idx = face_idx
            closest_hit.shape_mesh_index = -1

    return closest_hit


@wp.func
def closest_hit(
    bvh_shapes_size: wp.int32,
    bvh_shapes_id: wp.uint64,
    bvh_shapes_group_roots: wp.array(dtype=wp.int32),
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    world_index: wp.int32,
    enable_global_world: wp.bool,
    enable_particles: wp.bool,
    enable_backface_culling: wp.bool,
    max_distance: wp.float32,
    shape_enabled: wp.array(dtype=wp.uint32),
    shape_types: wp.array(dtype=wp.int32),
    shape_mesh_indices: wp.array(dtype=wp.int32),
    shape_sizes: wp.array(dtype=wp.vec3f),
    shape_transforms: wp.array(dtype=wp.transformf),
    mesh_ids: wp.array(dtype=wp.uint64),
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    triangle_mesh_id: wp.uint64,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
) -> ClosestHit:
    closest_hit = ClosestHit()
    closest_hit.distance = max_distance
    closest_hit.normal = wp.vec3f(0.0)
    closest_hit.shape_index = NO_HIT_SHAPE_ID
    closest_hit.bary_u = wp.float32(0.0)
    closest_hit.bary_v = wp.float32(0.0)
    closest_hit.face_idx = wp.int32(-1)
    closest_hit.shape_mesh_index = wp.int32(-1)

    closest_hit = closest_hit_triangle_mesh(
        closest_hit, triangle_mesh_id, enable_backface_culling, ray_origin_world, ray_dir_world
    )

    closest_hit = closest_hit_shape(
        closest_hit,
        bvh_shapes_size,
        bvh_shapes_id,
        bvh_shapes_group_roots,
        world_index,
        enable_global_world,
        shape_enabled,
        shape_types,
        shape_mesh_indices,
        shape_sizes,
        shape_transforms,
        mesh_ids,
        enable_backface_culling,
        ray_origin_world,
        ray_dir_world,
    )

    if enable_particles:
        closest_hit = closest_hit_particles(
            closest_hit,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            world_index,
            enable_global_world,
            particles_position,
            particles_radius,
            ray_origin_world,
            ray_dir_world,
        )

    return closest_hit


@wp.func
def first_hit_shape(
    bvh_shapes_size: wp.int32,
    bvh_shapes_id: wp.uint64,
    bvh_shapes_group_roots: wp.array(dtype=wp.int32),
    world_index: wp.int32,
    enable_global_world: wp.bool,
    shape_enabled: wp.array(dtype=wp.uint32),
    shape_types: wp.array(dtype=wp.int32),
    shape_mesh_indices: wp.array(dtype=wp.int32),
    shape_sizes: wp.array(dtype=wp.vec3f),
    shape_transforms: wp.array(dtype=wp.transformf),
    mesh_ids: wp.array(dtype=wp.uint64),
    enable_backface_culling: wp.bool,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
    max_dist: wp.float32,
) -> wp.bool:
    if bvh_shapes_size:
        for i in range(2 if enable_global_world else 1):
            group_root = get_group_roots(bvh_shapes_group_roots, world_index, i)
            if group_root < 0:
                continue

            query = wp.bvh_query_ray(bvh_shapes_id, ray_origin_world, ray_dir_world, group_root)
            shape_index = wp.int32(0)

            while wp.bvh_query_next(query, shape_index, max_dist):
                si = shape_enabled[shape_index]

                dist = wp.float32(-1)

                if shape_types[si] == GeoType.MESH:
                    geom_hit, _u, _v, _f = ray_intersect.ray_intersect_mesh(
                        shape_transforms[si],
                        shape_sizes[si],
                        ray_origin_world,
                        ray_dir_world,
                        mesh_ids[shape_mesh_indices[si]],
                        enable_backface_culling,
                        max_dist,
                    )
                    if geom_hit.hit:
                        dist = geom_hit.distance
                elif shape_types[si] == GeoType.PLANE:
                    dist = ray_intersect.ray_intersect_plane(
                        shape_transforms[si],
                        shape_sizes[si],
                        enable_backface_culling,
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif shape_types[si] == GeoType.SPHERE:
                    dist = raycast.ray_intersect_sphere(
                        shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si][0]
                    )
                elif shape_types[si] == GeoType.ELLIPSOID:
                    dist = raycast.ray_intersect_ellipsoid(
                        shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si]
                    )
                elif shape_types[si] == GeoType.CAPSULE:
                    dist = raycast.ray_intersect_capsule(
                        shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si][0], shape_sizes[si][1]
                    )
                elif shape_types[si] == GeoType.CYLINDER:
                    dist = raycast.ray_intersect_cylinder(
                        shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si][0], shape_sizes[si][1]
                    )
                elif shape_types[si] == GeoType.CONE:
                    dist = raycast.ray_intersect_cone(
                        shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si][0], shape_sizes[si][1]
                    )
                elif shape_types[si] == GeoType.BOX:
                    dist = raycast.ray_intersect_box(
                        shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si]
                    )
                if dist > -1 and dist < max_dist:
                    return True

    return False


@wp.func
def first_hit_particles(
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    world_index: wp.int32,
    enable_global_world: wp.bool,
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
    max_dist: wp.float32,
) -> wp.bool:
    if bvh_particles_size:
        for i in range(2 if enable_global_world else 1):
            group_root = get_group_roots(bvh_particles_group_roots, world_index, i)
            if group_root < 0:
                continue

            query = wp.bvh_query_ray(bvh_particles_id, ray_origin_world, ray_dir_world, group_root)
            si = wp.int32(0)

            while wp.bvh_query_next(query, si, max_dist):
                geom_hit = ray_intersect.ray_intersect_particle_sphere_with_normal(
                    ray_origin_world,
                    ray_dir_world,
                    particles_position[si],
                    particles_radius[si],
                )

                if geom_hit.hit and geom_hit.distance <= max_dist:
                    return True

    return False


@wp.func
def first_hit_triangle_mesh(
    triangle_mesh_id: wp.uint64,
    enable_backface_culling: wp.bool,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
    max_dist: wp.float32,
) -> wp.bool:
    if triangle_mesh_id:
        geom_hit, _bary_u, _bary_v, _face_idx = ray_intersect.ray_intersect_mesh_no_transform(
            triangle_mesh_id, ray_origin_world, ray_dir_world, enable_backface_culling, max_dist
        )
        return geom_hit.hit
    return False


@wp.func
def first_hit(
    bvh_shapes_size: wp.int32,
    bvh_shapes_id: wp.uint64,
    bvh_shapes_group_roots: wp.array(dtype=wp.int32),
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    world_index: wp.int32,
    enable_global_world: wp.bool,
    enable_particles: wp.bool,
    enable_backface_culling: wp.bool,
    shape_enabled: wp.array(dtype=wp.uint32),
    shape_types: wp.array(dtype=wp.int32),
    shape_mesh_indices: wp.array(dtype=wp.int32),
    shape_sizes: wp.array(dtype=wp.vec3f),
    shape_transforms: wp.array(dtype=wp.transformf),
    mesh_ids: wp.array(dtype=wp.uint64),
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    triangle_mesh_id: wp.uint64,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
    max_dist: wp.float32,
) -> wp.bool:
    if first_hit_triangle_mesh(triangle_mesh_id, enable_backface_culling, ray_origin_world, ray_dir_world, max_dist):
        return True

    if first_hit_shape(
        bvh_shapes_size,
        bvh_shapes_id,
        bvh_shapes_group_roots,
        world_index,
        enable_global_world,
        shape_enabled,
        shape_types,
        shape_mesh_indices,
        shape_sizes,
        shape_transforms,
        mesh_ids,
        enable_backface_culling,
        ray_origin_world,
        ray_dir_world,
        max_dist,
    ):
        return True

    if enable_particles:
        if first_hit_particles(
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            world_index,
            enable_global_world,
            particles_position,
            particles_radius,
            ray_origin_world,
            ray_dir_world,
            max_dist,
        ):
            return True

    return False
