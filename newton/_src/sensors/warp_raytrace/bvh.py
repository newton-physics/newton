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

from .types import GeomType


@wp.func
def compute_mesh_bounds(
    pos: wp.vec3f, rot: wp.mat33f, min_bounds: wp.vec3f, max_bounds: wp.vec3f
) -> tuple[wp.vec3f, wp.vec3f]:
    min_bound = wp.vec3f(wp.inf)
    max_bound = wp.vec3f(-wp.inf)

    corner_1 = pos + rot @ wp.vec3f(min_bounds[0], min_bounds[1], min_bounds[2])
    min_bound = wp.min(min_bound, corner_1)
    max_bound = wp.max(max_bound, corner_1)

    corner_2 = pos + rot @ wp.vec3f(max_bounds[0], min_bounds[1], min_bounds[2])
    min_bound = wp.min(min_bound, corner_2)
    max_bound = wp.max(max_bound, corner_2)

    corner_3 = pos + rot @ wp.vec3f(max_bounds[0], max_bounds[1], min_bounds[2])
    min_bound = wp.min(min_bound, corner_3)
    max_bound = wp.max(max_bound, corner_3)

    corner_4 = pos + rot @ wp.vec3f(min_bounds[0], max_bounds[1], min_bounds[2])
    min_bound = wp.min(min_bound, corner_4)
    max_bound = wp.max(max_bound, corner_4)

    corner_5 = pos + rot @ wp.vec3f(min_bounds[0], min_bounds[1], max_bounds[2])
    min_bound = wp.min(min_bound, corner_5)
    max_bound = wp.max(max_bound, corner_5)

    corner_6 = pos + rot @ wp.vec3f(max_bounds[0], min_bounds[1], max_bounds[2])
    min_bound = wp.min(min_bound, corner_6)
    max_bound = wp.max(max_bound, corner_6)

    corner_7 = pos + rot @ wp.vec3f(min_bounds[0], max_bounds[1], max_bounds[2])
    min_bound = wp.min(min_bound, corner_7)
    max_bound = wp.max(max_bound, corner_7)

    corner_8 = pos + rot @ wp.vec3f(max_bounds[0], max_bounds[1], max_bounds[2])
    min_bound = wp.min(min_bound, corner_8)
    max_bound = wp.max(max_bound, corner_8)

    return min_bound, max_bound


@wp.func
def compute_box_bounds(pos: wp.vec3f, rot: wp.mat33f, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    min_bound = wp.vec3f(wp.inf)
    max_bound = wp.vec3f(-wp.inf)

    for x in range(2):
        for y in range(2):
            for z in range(2):
                local_corner = wp.vec3f(
                    size[0] * (2.0 * wp.float32(x) - 1.0),
                    size[1] * (2.0 * wp.float32(y) - 1.0),
                    size[2] * (2.0 * wp.float32(z) - 1.0),
                )
                world_corner = pos + rot @ local_corner
                min_bound = wp.min(min_bound, world_corner)
                max_bound = wp.max(max_bound, world_corner)

    return min_bound, max_bound


@wp.func
def compute_sphere_bounds(pos: wp.vec3f, radius: wp.float32) -> tuple[wp.vec3f, wp.vec3f]:
    return pos - wp.vec3f(radius), pos + wp.vec3f(radius)


@wp.func
def compute_capsule_bounds(pos: wp.vec3f, rot: wp.mat33f, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    radius = size[0]
    half_length = size[1]
    local_end1 = wp.vec3f(-radius, -radius, -half_length - radius)
    local_end2 = wp.vec3f(radius, radius, half_length + radius)
    world_end1 = pos + rot @ local_end1
    world_end2 = pos + rot @ local_end2
    bounds_min = wp.min(world_end1, world_end2)
    bounds_max = wp.max(world_end1, world_end2)
    return bounds_min, bounds_max


@wp.func
def compute_cylinder_bounds(pos: wp.vec3f, rot: wp.mat33f, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    radius = size[0]
    half_length = size[1]
    local_end1 = wp.vec3f(-radius, -radius, -half_length)
    local_end2 = wp.vec3f(radius, radius, half_length)
    world_end1 = pos + rot @ local_end1
    world_end2 = pos + rot @ local_end2
    bounds_min = wp.min(world_end1, world_end2)
    bounds_max = wp.max(world_end1, world_end2)
    return bounds_min, bounds_max


@wp.func
def compute_cone_bounds(pos: wp.vec3f, rot: wp.mat33f, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    return compute_cylinder_bounds(pos, rot, size)


@wp.func
def compute_plane_bounds(pos: wp.vec3f, rot: wp.mat33f, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    # If plane size is non-positive, treat as infinite plane and use a large default extent
    size_scale = wp.max(size[0], size[1]) * 2.0
    if size[0] <= 0.0 or size[1] <= 0.0:
        size_scale = 1000.0

    min_bound = wp.vec3f(wp.inf)
    max_bound = wp.vec3f(-wp.inf)

    for x in range(2):
        for y in range(2):
            local_corner = wp.vec3f(
                size_scale * (2.0 * wp.float32(x) - 1.0),
                size_scale * (2.0 * wp.float32(y) - 1.0),
                0.0,
            )
            world_corner = pos + rot @ local_corner
            min_bound = wp.min(min_bound, world_corner)
            max_bound = wp.max(max_bound, world_corner)

    extent = wp.vec3f(0.1)
    return min_bound - extent, max_bound + extent


@wp.func
def compute_ellipsoid_bounds(pos: wp.vec3f, rot: wp.mat33f, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    extent = wp.vec3f(wp.abs(size[0]), wp.abs(size[1]), wp.abs(size[2]))
    return compute_box_bounds(pos, rot, extent)


@wp.kernel(enable_backward=False)
def compute_geom_bvh_bounds(
    num_geom_in_bvh: wp.int32,
    num_worlds: wp.int32,
    geom_world_index: wp.array(dtype=wp.int32),
    geom_enabled: wp.array(dtype=wp.int32),
    geom_types: wp.array(dtype=wp.int32),
    geom_mesh_indices: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3f),
    geom_positions: wp.array(dtype=wp.vec3f),
    geom_orientations: wp.array(dtype=wp.mat33f),
    mesh_bounds: wp.array2d(dtype=wp.vec3f),
    out_bvh_lowers: wp.array(dtype=wp.vec3f),
    out_bvh_uppers: wp.array(dtype=wp.vec3f),
    out_bvh_groups: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    bvh_geom_local = tid % num_geom_in_bvh
    if bvh_geom_local >= num_geom_in_bvh:
        return

    geom_id = geom_enabled[bvh_geom_local]

    world_id = geom_world_index[geom_id]
    if world_id < 0:
        world_id = num_worlds + world_id

    if world_id >= num_worlds:
        return

    pos = geom_positions[geom_id]
    rot = geom_orientations[geom_id]
    size = geom_sizes[geom_id]
    type = geom_types[geom_id]

    lower = wp.vec3f()
    upper = wp.vec3f()

    if type == GeomType.SPHERE:
        lower, upper = compute_sphere_bounds(pos, size[0])
    elif type == GeomType.CAPSULE:
        lower, upper = compute_capsule_bounds(pos, rot, size)
    elif type == GeomType.CYLINDER:
        lower, upper = compute_cylinder_bounds(pos, rot, size)
    elif type == GeomType.CONE:
        lower, upper = compute_cone_bounds(pos, rot, size)
    elif type == GeomType.PLANE:
        lower, upper = compute_plane_bounds(pos, rot, size)
    elif type == GeomType.MESH:
        min_bounds = mesh_bounds[geom_mesh_indices[geom_id], 0]
        max_bounds = mesh_bounds[geom_mesh_indices[geom_id], 1]
        lower, upper = compute_mesh_bounds(pos, rot, min_bounds, max_bounds)
    elif type == GeomType.ELLIPSOID:
        lower, upper = compute_ellipsoid_bounds(pos, rot, size)
    elif type == GeomType.BOX:
        lower, upper = compute_box_bounds(pos, rot, size)

    out_bvh_lowers[world_id * num_geom_in_bvh + bvh_geom_local] = lower
    out_bvh_uppers[world_id * num_geom_in_bvh + bvh_geom_local] = upper
    out_bvh_groups[world_id * num_geom_in_bvh + bvh_geom_local] = world_id


@wp.kernel(enable_backward=False)
def compute_particle_bvh_bounds(
    num_particle_in_bvh: wp.int32,
    num_worlds: wp.int32,
    particle_world_index: wp.array(dtype=wp.int32),
    particle_position: wp.array(dtype=wp.vec3f),
    particle_radius: wp.array(dtype=wp.float32),
    out_bvh_lowers: wp.array(dtype=wp.vec3f),
    out_bvh_uppers: wp.array(dtype=wp.vec3f),
    out_bvh_groups: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    bvh_geom_local = tid % num_particle_in_bvh
    if bvh_geom_local >= num_particle_in_bvh:
        return

    geom_id = bvh_geom_local # geom_enabled[bvh_geom_local]

    world_id = particle_world_index[geom_id]
    if world_id < 0:
        world_id = num_worlds + world_id

    if world_id >= num_worlds:
        return

    lower, upper = compute_sphere_bounds(particle_position[geom_id], particle_radius[geom_id])

    out_bvh_lowers[world_id * num_particle_in_bvh + bvh_geom_local] = lower
    out_bvh_uppers[world_id * num_particle_in_bvh + bvh_geom_local] = upper
    out_bvh_groups[world_id * num_particle_in_bvh + bvh_geom_local] = world_id


@wp.kernel(enable_backward=False)
def compute_bvh_group_roots(bvh_id: wp.uint64, out_bvh_group_roots: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    out_bvh_group_roots[tid] = wp.bvh_get_group_root(bvh_id, tid)
