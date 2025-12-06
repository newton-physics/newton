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

from . import ray
from .types import GeomType

TRIANGLE_MESH = -1
TRIANGLE_MESH_GEOM_ID = -100
PARTICLES_GEOM_ID = -200


@wp.struct
class ClosestHit:
    distance: wp.float32
    normal: wp.vec3f
    geom_id: wp.int32
    bary_u: wp.float32
    bary_v: wp.float32
    face_idx: wp.int32
    geom_mesh_id: wp.int32


@wp.func
def get_group_roots(
    group_roots: wp.array(dtype=wp.int32), world_id: wp.int32, want_global_world: wp.int32
) -> tuple[wp.int32, wp.int32]:
    if want_global_world != 0:
        return group_roots.shape[0] - 1, group_roots[group_roots.shape[0] - 1]
    return world_id, group_roots[world_id]


@wp.func
def closest_hit_geom(
    closest_hit: ClosestHit,
    bvh_geom_size: wp.int32,
    bvh_geom_id: wp.uint64,
    bvh_geom_group_roots: wp.array(dtype=wp.int32),
    world_id: wp.int32,
    has_global_world: wp.bool,
    geom_enabled: wp.array(dtype=wp.int32),
    geom_types: wp.array(dtype=wp.int32),
    geom_mesh_indices: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3f),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_positions: wp.array(dtype=wp.vec3f),
    geom_orientations: wp.array(dtype=wp.mat33f),
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
) -> ClosestHit:
    if bvh_geom_size:
        for i in range(2 if has_global_world else 1):
            world_id, group_root = get_group_roots(bvh_geom_group_roots, world_id, i)
            if group_root < 0:
                continue

            query = wp.bvh_query_ray(bvh_geom_id, ray_origin_world, ray_dir_world, group_root)
            geom_index = wp.int32(0)

            while wp.bvh_query_next(query, geom_index, closest_hit.distance):
                gi = geom_enabled[geom_index]

                hit = wp.bool(False)
                hit_dist = wp.float32(wp.inf)
                hit_normal = wp.vec3f(0.0)
                hit_u = wp.float32(0.0)
                hit_v = wp.float32(0.0)
                hit_face_id = wp.int32(-1)
                hit_mesh_id = wp.int32(-1)

                if geom_types[gi] == GeomType.MESH:
                    hit, hit_dist, hit_normal, hit_u, hit_v, hit_face_id, hit_mesh_id = ray.ray_mesh_with_bvh(
                        mesh_ids,
                        geom_mesh_indices[gi],
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                        closest_hit.distance,
                    )
                elif geom_types[gi] == GeomType.PLANE:
                    hit, hit_dist, hit_normal = ray.ray_plane_with_normal(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.SPHERE:
                    hit, hit_dist, hit_normal = ray.ray_sphere_with_normal(
                        geom_positions[gi],
                        geom_sizes[gi][0] * geom_sizes[gi][0],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.CAPSULE:
                    hit, hit_dist, hit_normal = ray.ray_capsule_with_normal(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.CYLINDER:
                    hit, hit_dist, hit_normal = ray.ray_cylinder_with_normal(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.CONE:
                    hit, hit_dist, hit_normal = ray.ray_cone_with_normal(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.BOX:
                    hit, hit_dist, hit_normal = ray.ray_box_with_normal(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )

                if hit and hit_dist < closest_hit.distance:
                    closest_hit.distance = hit_dist
                    closest_hit.normal = hit_normal
                    closest_hit.geom_id = gi
                    closest_hit.bary_u = hit_u
                    closest_hit.bary_v = hit_v
                    closest_hit.face_idx = hit_face_id
                    closest_hit.geom_mesh_id = hit_mesh_id

    return closest_hit


@wp.func
def closest_hit_particles(
    closest_hit: ClosestHit,
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    world_id: wp.int32,
    has_global_world: wp.bool,
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
) -> ClosestHit:
    if bvh_particles_size:
        for i in range(2 if has_global_world else 1):
            world_id, group_root = get_group_roots(bvh_particles_group_roots, world_id, i)
            if group_root < 0:
                continue

            query = wp.bvh_query_ray(bvh_particles_id, ray_origin_world, ray_dir_world, group_root)
            bounds_nr = wp.int32(0)

            while wp.bvh_query_next(query, bounds_nr, closest_hit.distance):
                gi_global = bounds_nr
                gi_bvh_local = gi_global - (world_id * bvh_particles_size)
                gi = gi_bvh_local

                hit, hit_dist, hit_normal = ray.ray_sphere_with_normal(
                    particles_position[gi],
                    particles_radius[gi] * particles_radius[gi],
                    ray_origin_world,
                    ray_dir_world,
                )

                if hit and hit_dist < closest_hit.distance:
                    closest_hit.distance = hit_dist
                    closest_hit.normal = hit_normal
                    closest_hit.geom_id = PARTICLES_GEOM_ID
                    closest_hit.geom_mesh_id = -1

    return closest_hit


@wp.func
def closest_hit_triangle_mesh(
    closest_hit: ClosestHit,
    triangle_mesh_id: wp.uint64,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
) -> ClosestHit:
    if triangle_mesh_id:
        hit, max_distance, normal, bary_u, bary_v, face_idx = ray.ray_mesh(
            triangle_mesh_id, ray_origin_world, ray_dir_world, closest_hit.distance, True
        )
        if hit:
            closest_hit.distance = max_distance
            closest_hit.normal = normal
            closest_hit.geom_id = TRIANGLE_MESH_GEOM_ID
            closest_hit.bary_u = bary_u
            closest_hit.bary_v = bary_v
            closest_hit.face_idx = face_idx
            closest_hit.geom_mesh_id = -1

    return closest_hit


@wp.func
def closest_hit(
    bvh_geom_size: wp.int32,
    bvh_geom_id: wp.uint64,
    bvh_geom_group_roots: wp.array(dtype=wp.int32),
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    world_id: wp.int32,
    has_global_world: wp.bool,
    enable_particles: wp.bool,
    max_distance: wp.float32,
    geom_enabled: wp.array(dtype=wp.int32),
    geom_types: wp.array(dtype=wp.int32),
    geom_mesh_indices: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3f),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_positions: wp.array(dtype=wp.vec3f),
    geom_orientations: wp.array(dtype=wp.mat33f),
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    triangle_mesh_id: wp.uint64,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
) -> ClosestHit:
    closest_hit = ClosestHit()
    closest_hit.distance = max_distance
    closest_hit.normal = wp.vec3f(0.0)
    closest_hit.geom_id = wp.int32(-1)
    closest_hit.bary_u = wp.float32(0.0)
    closest_hit.bary_v = wp.float32(0.0)
    closest_hit.face_idx = wp.int32(-1)
    closest_hit.geom_mesh_id = wp.int32(-1)

    closest_hit = closest_hit_triangle_mesh(closest_hit, triangle_mesh_id, ray_origin_world, ray_dir_world)

    closest_hit = closest_hit_geom(
        closest_hit,
        bvh_geom_size,
        bvh_geom_id,
        bvh_geom_group_roots,
        world_id,
        has_global_world,
        geom_enabled,
        geom_types,
        geom_mesh_indices,
        geom_sizes,
        mesh_ids,
        geom_positions,
        geom_orientations,
        ray_origin_world,
        ray_dir_world,
    )

    if enable_particles:
        closest_hit = closest_hit_particles(
            closest_hit,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            world_id,
            has_global_world,
            particles_position,
            particles_radius,
            ray_origin_world,
            ray_dir_world,
        )

    return closest_hit


@wp.func
def first_hit_geom(
    bvh_geom_size: wp.int32,
    bvh_geom_id: wp.uint64,
    bvh_geom_group_roots: wp.array(dtype=wp.int32),
    world_id: wp.int32,
    has_global_world: wp.bool,
    geom_enabled: wp.array(dtype=wp.int32),
    geom_types: wp.array(dtype=wp.int32),
    geom_mesh_indices: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3f),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_positions: wp.array(dtype=wp.vec3f),
    geom_orientations: wp.array(dtype=wp.mat33f),
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
    max_dist: wp.float32,
) -> wp.bool:
    if bvh_geom_size:
        for i in range(2 if has_global_world else 1):
            world_id, group_root = get_group_roots(bvh_geom_group_roots, world_id, i)
            if group_root < 0:
                continue

            query = wp.bvh_query_ray(bvh_geom_id, ray_origin_world, ray_dir_world, group_root)
            geom_index = wp.int32(0)

            while wp.bvh_query_next(query, geom_index, max_dist):
                gi = geom_enabled[geom_index]

                dist = wp.float32(wp.inf)

                if geom_types[gi] == GeomType.MESH:
                    _h, dist, _n, _u, _v, _f, _mesh_id = ray.ray_mesh_with_bvh(
                        mesh_ids,
                        geom_mesh_indices[gi],
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                        max_dist,
                    )
                elif geom_types[gi] == GeomType.PLANE:
                    dist = ray.ray_plane(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.SPHERE:
                    dist = ray.ray_sphere(
                        geom_positions[gi],
                        geom_sizes[gi][0] * geom_sizes[gi][0],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.CAPSULE:
                    dist = ray.ray_capsule(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.CYLINDER:
                    dist, _ = ray.ray_cylinder(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.CONE:
                    dist = ray.ray_cone(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )
                elif geom_types[gi] == GeomType.BOX:
                    dist, _all = ray.ray_box(
                        geom_positions[gi],
                        geom_orientations[gi],
                        geom_sizes[gi],
                        ray_origin_world,
                        ray_dir_world,
                    )

                if dist < max_dist:
                    return True

    return False


@wp.func
def first_hit_particles(
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    world_id: wp.int32,
    has_global_world: wp.bool,
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
    max_dist: wp.float32,
) -> wp.bool:
    if bvh_particles_size:
        for i in range(2 if has_global_world else 1):
            world_id, group_root = get_group_roots(bvh_particles_group_roots, world_id, i)
            if group_root < 0:
                continue

            query = wp.bvh_query_ray(bvh_particles_id, ray_origin_world, ray_dir_world, group_root)
            bounds_nr = wp.int32(0)

            while wp.bvh_query_next(query, bounds_nr, max_dist):
                gi_global = bounds_nr
                gi_bvh_local = gi_global - (world_id * bvh_particles_size)
                gi = gi_bvh_local

                hit, hit_dist, _hit_normal = ray.ray_sphere_with_normal(
                    particles_position[gi],
                    particles_radius[gi] * particles_radius[gi],
                    ray_origin_world,
                    ray_dir_world,
                )

                if hit and hit_dist <= max_dist:
                    return True

    return False


@wp.func
def first_hit_triangle_mesh(
    triangle_mesh_id: wp.uint64,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
    max_dist: wp.float32,
) -> wp.bool:
    if triangle_mesh_id:
        hit, _max_distance, _normal, _bary_u, _bary_v, _face_idx = ray.ray_mesh(
            triangle_mesh_id, ray_origin_world, ray_dir_world, max_dist, True
        )
        return hit
    return False


@wp.func
def first_hit(
    bvh_geom_size: wp.int32,
    bvh_geom_id: wp.uint64,
    bvh_geom_group_roots: wp.array(dtype=wp.int32),
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    world_id: wp.int32,
    has_global_world: wp.bool,
    enable_particles: wp.bool,
    geom_enabled: wp.array(dtype=wp.int32),
    geom_types: wp.array(dtype=wp.int32),
    geom_mesh_indices: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3f),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_positions: wp.array(dtype=wp.vec3f),
    geom_orientations: wp.array(dtype=wp.mat33f),
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    triangle_mesh_id: wp.uint64,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
    max_dist: wp.float32,
) -> wp.bool:
    if first_hit_triangle_mesh(triangle_mesh_id, ray_origin_world, ray_dir_world, max_dist):
        return True

    if first_hit_geom(
        bvh_geom_size,
        bvh_geom_id,
        bvh_geom_group_roots,
        world_id,
        has_global_world,
        geom_enabled,
        geom_types,
        geom_mesh_indices,
        geom_sizes,
        mesh_ids,
        geom_positions,
        geom_orientations,
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
            world_id,
            has_global_world,
            particles_position,
            particles_radius,
            ray_origin_world,
            ray_dir_world,
            max_dist,
        ):
            return True

    return False
