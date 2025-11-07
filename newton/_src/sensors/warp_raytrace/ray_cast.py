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


@wp.func
def get_group_roots(
    group_roots: wp.array(dtype=wp.int32), world_id: wp.int32, want_global_world: wp.int32
) -> tuple[wp.int32, wp.int32]:
    if want_global_world != 0:
        return group_roots.shape[0] - 1, group_roots[group_roots.shape[0] - 1]
    return world_id, group_roots[world_id]


@wp.func
def closest_hit(
    bvh_id: wp.uint64,
    group_roots: wp.array(dtype=wp.int32),
    world_id: int,
    num_geom_in_bvh: int,
    geom_enabled: wp.array(dtype=int),
    geom_types: wp.array(dtype=int),
    geom_mesh_indices: wp.array(dtype=int),
    geom_sizes: wp.array(dtype=wp.vec3),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_positions: wp.array(dtype=wp.vec3),
    geom_orientations: wp.array(dtype=wp.mat33),
    ray_origin_world: wp.vec3,
    ray_dir_world: wp.vec3,
) -> tuple[wp.int32, wp.float32, wp.vec3, wp.float32, wp.float32, wp.int32, wp.int32]:
    max_dist = wp.float32(wp.inf)
    normal = wp.vec3(0.0, 0.0, 0.0)
    geom_id = wp.int32(-1)
    bary_u = wp.float32(0.0)
    bary_v = wp.float32(0.0)
    face_idx = wp.int32(-1)
    geom_mesh_id = wp.int32(-1)

    for i in range(2):
        world_id, group_root = get_group_roots(group_roots, world_id, i)
        query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
        bounds_nr = wp.int32(0)

        while wp.bvh_query_next(query, bounds_nr, max_dist):
            gi_global = bounds_nr
            gi_bvh_local = gi_global - (world_id * num_geom_in_bvh)
            gi = geom_enabled[gi_bvh_local]

            hit = bool(False)
            dist = wp.float32(wp.inf)
            n = wp.vec3(0.0)
            u = wp.float32(0.0)
            v = wp.float32(0.0)
            f = wp.int32(-1)
            mesh_id = wp.int32(-1)

            if geom_types[gi] == GeomType.PLANE:
                hit, dist, n = ray.ray_plane_with_normal(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.SPHERE:
                hit, dist, n = ray.ray_sphere_with_normal(
                    geom_positions[gi],
                    geom_sizes[gi][0] * geom_sizes[gi][0],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.CAPSULE:
                hit, dist, n = ray.ray_capsule_with_normal(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.CYLINDER:
                hit, dist, n = ray.ray_cylinder_with_normal(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.CONE:
                hit, dist, n = ray.ray_cone_with_normal(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.BOX:
                hit, dist, n = ray.ray_box_with_normal(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.MESH:
                hit, dist, n, u, v, f, mesh_id = ray.ray_mesh_with_bvh(
                    mesh_ids,
                    geom_mesh_indices[gi],
                    geom_positions[gi],
                    geom_orientations[gi],
                    ray_origin_world,
                    ray_dir_world,
                    max_dist,
                )

            if hit and dist < max_dist:
                max_dist = dist
                normal = n
                geom_id = gi
                bary_u = u
                bary_v = v
                face_idx = f
                geom_mesh_id = mesh_id

    return geom_id, max_dist, normal, bary_u, bary_v, face_idx, geom_mesh_id


@wp.func
def first_hit(
    bvh_id: wp.uint64,
    group_roots: wp.array(dtype=wp.int32),
    world_id: int,
    num_geom_in_bvh: int,
    geom_enabled: wp.array(dtype=int),
    geom_types: wp.array(dtype=int),
    geom_mesh_indices: wp.array(dtype=int),
    geom_sizes: wp.array(dtype=wp.vec3),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_positions: wp.array(dtype=wp.vec3),
    geom_orientations: wp.array(dtype=wp.mat33),
    ray_origin_world: wp.vec3,
    ray_dir_world: wp.vec3,
    max_dist: wp.float32,
) -> bool:
    """A simpler version of cast_ray_first_hit that only checks for the first hit."""

    for i in range(2):
        world_id, group_root = get_group_roots(group_roots, world_id, i)

        query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
        bounds_nr = wp.int32(0)

        while wp.bvh_query_next(query, bounds_nr, max_dist):
            gi_global = bounds_nr
            gi_bvh_local = gi_global - (world_id * num_geom_in_bvh)
            gi = geom_enabled[gi_bvh_local]

            dist = wp.float32(wp.inf)

            if geom_types[gi] == GeomType.PLANE:
                dist = ray.ray_plane(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.SPHERE:
                dist = ray.ray_sphere(
                    geom_positions[gi],
                    geom_sizes[gi][0] * geom_sizes[gi][0],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.CAPSULE:
                dist = ray.ray_capsule(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.CYLINDER:
                dist, _ = ray.ray_cylinder(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.CONE:
                dist = ray.ray_cone(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.BOX:
                dist, _all = ray.ray_box(
                    geom_positions[gi],
                    geom_orientations[gi],
                    geom_sizes[gi],
                    ray_origin_world,
                    ray_dir_world,
                )
            if geom_types[gi] == GeomType.MESH:
                _h, dist, _n, _u, _v, _f, _mesh_id = ray.ray_mesh_with_bvh(
                    mesh_ids,
                    geom_mesh_indices[gi],
                    geom_positions[gi],
                    geom_orientations[gi],
                    ray_origin_world,
                    ray_dir_world,
                    max_dist,
                )

            if dist < max_dist:
                return True

    return False
