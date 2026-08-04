# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...geometry import Gaussian, GeoType, raycast
from . import gaussians
from .types import MeshData

if TYPE_CHECKING:
    from .render_context import RenderContext


NO_HIT_SHAPE_ID = wp.uint32(0xFFFFFFFF)
MAX_SHAPE_ID = wp.uint32(0xFFFFFFF0)
TRIANGLE_MESH_SHAPE_ID = wp.uint32(0xFFFFFFFD)
PARTICLES_SHAPE_ID = wp.uint32(0xFFFFFFFE)


@wp.struct
class ClosestHit:
    distance: wp.float32
    normal: wp.vec3f
    shape_index: wp.uint32
    world_index: wp.int32
    bary_u: wp.float32
    bary_v: wp.float32
    face_idx: wp.int32
    color: wp.vec3f


@wp.func
def _ray_intersect_mesh_smooth(
    transform: wp.transformf,
    scale: wp.vec3f,
    ray_origin: wp.vec3f,
    ray_direction: wp.vec3f,
    mesh_id: wp.uint64,
    shape_mesh_data_id: wp.int32,
    mesh_data: wp.array[MeshData],
    enable_backface_culling: wp.bool,
    max_t: wp.float32,
) -> tuple[wp.float32, wp.vec3f, wp.float32, wp.float32, wp.int32]:
    """Ray-mesh intersection (world-space normal) with optional per-vertex normal interpolation.

    Wraps :func:`~newton._src.geometry.raycast.ray_intersect_mesh`, then when ``shape_mesh_data_id``
    is non-negative and the referenced ``mesh_data`` entry supplies per-vertex normals, replaces the
    triangle face normal with the barycentric interpolation of those vertex normals (smooth shading).
    """
    ray_origin_local, ray_direction_local = raycast.map_ray_to_local(transform, ray_origin, ray_direction, scale)
    t, normal_local, u, v, face = raycast.ray_intersect_mesh(
        ray_origin_local, ray_direction_local, scale, mesh_id, enable_backface_culling, max_t
    )
    if t < 0.0:
        return t, normal_local, u, v, face

    if shape_mesh_data_id > -1:
        normals = mesh_data[shape_mesh_data_id].normals
        if normals.shape[0] > 0:
            n0 = wp.mesh_get_index(mesh_id, face * 3 + 0)
            n1 = wp.mesh_get_index(mesh_id, face * 3 + 1)
            n2 = wp.mesh_get_index(mesh_id, face * 3 + 2)
            vertex_normal = normals[n0] * u + normals[n1] * v + normals[n2] * (1.0 - u - v)
            normal_local = raycast.safe_div_vec3(vertex_normal, scale)

    return t, wp.normalize(wp.transform_vector(transform, normal_local)), u, v, face


@wp.func
def get_world_offset(world_offsets: wp.array[wp.vec3f], world_index: wp.int32) -> wp.vec3f:
    offset = wp.vec3f(0.0)
    if world_index >= 0:
        if world_offsets.shape[0] > 0:
            if world_index < world_offsets.shape[0]:
                offset = world_offsets[world_index]
    return offset


@wp.struct
class GroupQuery:
    root_index: wp.int32
    world_index: wp.int32
    ray_origin_world: wp.vec3f


def create_group_query_functions(config: RenderContext.Config) -> tuple[wp.Function, wp.Function]:
    @wp.func
    def get_group_query_count(group_roots: wp.array[wp.int32]) -> wp.int32:
        query_count = wp.int32(wp.static(2 if config.enable_global_world else 1))
        if wp.static(config.render_worlds_together):
            query_count = group_roots.shape[0]
            if not wp.static(config.enable_global_world):
                query_count = query_count - 1
        return query_count

    @wp.func
    def get_group_query(
        query_index: wp.int32,
        group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        ray_origin_world: wp.vec3f,
    ) -> GroupQuery:
        query = GroupQuery()
        query.root_index = world_index
        query.world_index = world_index
        query.ray_origin_world = ray_origin_world

        if wp.static(config.render_worlds_together):
            query.root_index = query_index
            query.world_index = query_index
            if query_index == group_roots.shape[0] - 1:
                query.world_index = wp.int32(-1)
            query.ray_origin_world = ray_origin_world - get_world_offset(world_offsets, query.world_index)
        elif query_index != 0:
            query.root_index = group_roots.shape[0] - 1
            query.world_index = wp.int32(-1)

        return query

    return get_group_query_count, get_group_query


def create_closest_hit_function(config: RenderContext.Config, state: RenderContext.State) -> wp.Function:
    shade_gaussians = gaussians.create_shade_function(config, state)
    get_group_query_count, get_group_query = create_group_query_functions(config)

    @wp.func
    def closest_hit_shape(
        closest_hit: ClosestHit,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        shape_mesh_data_ids: wp.array[wp.int32],
        mesh_data: wp.array[MeshData],
        gaussians_data: wp.array[Gaussian.Data],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        camera_forward: wp.vec3f,
    ) -> ClosestHit:
        if bvh_shapes_size:
            query_count = get_group_query_count(bvh_shapes_group_roots)
            for i in range(query_count):
                group_query = get_group_query(i, bvh_shapes_group_roots, world_index, world_offsets, ray_origin_world)
                if bvh_shapes_group_roots[group_query.root_index] < 0:
                    continue

                gaussians_hit = wp.vector(length=wp.static(state.num_gaussians), dtype=wp.uint32)
                num_gaussians_hit = wp.int32(0)

                query = wp.bvh_query_ray(
                    bvh_shapes_id,
                    group_query.ray_origin_world,
                    ray_dir_world,
                    bvh_shapes_group_roots[group_query.root_index],
                )
                shape_index = wp.int32(0)

                while wp.bvh_query_next(query, shape_index, closest_hit.distance):
                    si = shape_enabled[shape_index]

                    hit_distance = wp.float32(-1.0)
                    hit_normal = wp.vec3f(0.0)
                    hit_u = wp.float32(0.0)
                    hit_v = wp.float32(0.0)
                    hit_face_id = wp.int32(-1)
                    hit_color = wp.vec3f(0.0)

                    shape_type = shape_types[si]
                    # Heightfields are triangulated meshes; RenderContext remaps
                    # HFIELD -> MESH, so this branch renders them too.
                    if shape_type == GeoType.MESH:
                        hit_distance, hit_normal, hit_u, hit_v, hit_face_id = _ray_intersect_mesh_smooth(
                            shape_transforms[si],
                            shape_sizes[si],
                            group_query.ray_origin_world,
                            ray_dir_world,
                            shape_source_ptr[si],
                            shape_mesh_data_ids[si],
                            mesh_data,
                            wp.static(config.enable_backface_culling),
                            closest_hit.distance,
                        )
                    elif shape_type == GeoType.GAUSSIAN:
                        if num_gaussians_hit < wp.static(state.num_gaussians):
                            gaussians_hit[num_gaussians_hit] = si
                            num_gaussians_hit += 1
                            # gaussian_id = shape_source_ptr[si]
                            # hit_distance, hit_normal, hit_color = shade_gaussians(
                            #     shape_transforms[si],
                            #     shape_sizes[si],
                            #     ray_origin_world,
                            #     ray_dir_world,
                            #     gaussians_data[gaussian_id],
                            #     closest_hit.distance
                            # )
                    else:
                        hit_distance, hit_normal = raycast.ray_intersect_shape(
                            shape_transforms[si],
                            shape_sizes[si],
                            shape_type,
                            group_query.ray_origin_world,
                            ray_dir_world,
                            wp.static(config.enable_backface_culling),
                        )

                    if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                        closest_hit.distance = hit_distance
                        closest_hit.normal = hit_normal
                        closest_hit.shape_index = si
                        closest_hit.world_index = group_query.world_index
                        closest_hit.bary_u = hit_u
                        closest_hit.bary_v = hit_v
                        closest_hit.face_idx = hit_face_id
                        closest_hit.color = hit_color

                # Temporary workaround. Warp BVH queries share some stack data,
                # which breaks nested wp.bvh_query_ray calls.
                # Once it is fixed in Warp, remove this code block and put
                # the commented out block above back in.
                # Although, this workaround may actually be a performance improvement
                # since it only renders gaussians if they are not blocked by other
                # objects.
                if num_gaussians_hit > 0:
                    for gi in range(num_gaussians_hit):
                        si = gaussians_hit[gi]

                        gaussian_id = shape_source_ptr[si]
                        hit_distance, hit_normal, hit_color = shade_gaussians(
                            shape_transforms[si],
                            shape_sizes[si],
                            group_query.ray_origin_world,
                            ray_dir_world,
                            camera_forward,
                            gaussians_data[gaussian_id],
                            closest_hit.distance,
                        )

                        if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                            closest_hit.distance = hit_distance
                            closest_hit.normal = hit_normal
                            closest_hit.shape_index = si
                            closest_hit.world_index = group_query.world_index
                            closest_hit.color = hit_color

        return closest_hit

    @wp.func
    def closest_hit_particles(
        closest_hit: ClosestHit,
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
    ) -> ClosestHit:
        if bvh_particles_size:
            query_count = get_group_query_count(bvh_particles_group_roots)
            for i in range(query_count):
                group_query = get_group_query(
                    i, bvh_particles_group_roots, world_index, world_offsets, ray_origin_world
                )
                if bvh_particles_group_roots[group_query.root_index] < 0:
                    continue

                query = wp.bvh_query_ray(
                    bvh_particles_id,
                    group_query.ray_origin_world,
                    ray_dir_world,
                    bvh_particles_group_roots[group_query.root_index],
                )
                si = wp.int32(0)

                while wp.bvh_query_next(query, si, closest_hit.distance):
                    if topology_particle_mask[si]:
                        continue

                    hit_distance, hit_normal = raycast.ray_intersect_particle_sphere(
                        group_query.ray_origin_world,
                        ray_dir_world,
                        particles_position[si],
                        particles_radius[si],
                    )

                    if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                        closest_hit.distance = hit_distance
                        closest_hit.normal = hit_normal
                        closest_hit.shape_index = PARTICLES_SHAPE_ID
                        closest_hit.world_index = group_query.world_index

        return closest_hit

    @wp.func
    def closest_hit_triangle_mesh(
        closest_hit: ClosestHit,
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
    ) -> ClosestHit:
        if triangle_mesh_id:
            query_count = get_group_query_count(triangle_mesh_group_roots)
            for i in range(query_count):
                group_query = get_group_query(
                    i, triangle_mesh_group_roots, world_index, world_offsets, ray_origin_world
                )
                if triangle_mesh_group_roots[group_query.root_index] < 0:
                    continue

                hit_distance, hit_normal, bary_u, bary_v, face_idx = raycast.ray_intersect_mesh(
                    group_query.ray_origin_world,
                    ray_dir_world,
                    wp.vec3f(1.0),
                    triangle_mesh_id,
                    wp.static(config.enable_backface_culling),
                    closest_hit.distance,
                    triangle_mesh_group_roots[group_query.root_index],
                )
                if hit_distance >= 0.0:
                    closest_hit.distance = hit_distance
                    closest_hit.normal = hit_normal
                    closest_hit.shape_index = TRIANGLE_MESH_SHAPE_ID
                    closest_hit.world_index = group_query.world_index
                    closest_hit.bary_u = bary_u
                    closest_hit.bary_v = bary_v
                    closest_hit.face_idx = face_idx

        return closest_hit

    @wp.func
    def closest_hit(
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        max_distance: wp.float32,
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        shape_mesh_data_ids: wp.array[wp.int32],
        mesh_data: wp.array[MeshData],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        gaussians_data: wp.array[Gaussian.Data],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        camera_forward: wp.vec3f,
    ) -> ClosestHit:
        closest_hit = ClosestHit()
        closest_hit.distance = max_distance
        closest_hit.shape_index = NO_HIT_SHAPE_ID
        closest_hit.world_index = wp.int32(-1)
        closest_hit.color = wp.vec3f(0.0)

        closest_hit = closest_hit_triangle_mesh(
            closest_hit,
            triangle_mesh_id,
            triangle_mesh_group_roots,
            world_index,
            world_offsets,
            ray_origin_world,
            ray_dir_world,
        )

        closest_hit = closest_hit_shape(
            closest_hit,
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            world_index,
            world_offsets,
            shape_enabled,
            shape_types,
            shape_sizes,
            shape_transforms,
            shape_source_ptr,
            shape_mesh_data_ids,
            mesh_data,
            gaussians_data,
            ray_origin_world,
            ray_dir_world,
            camera_forward,
        )

        if wp.static(config.enable_particles) and wp.static(state.has_particles):
            closest_hit = closest_hit_particles(
                closest_hit,
                bvh_particles_size,
                bvh_particles_id,
                bvh_particles_group_roots,
                world_index,
                world_offsets,
                particles_position,
                particles_radius,
                topology_particle_mask,
                ray_origin_world,
                ray_dir_world,
            )

        return closest_hit

    return closest_hit


def create_closest_hit_depth_only_function(config: RenderContext.Config, state: RenderContext.State) -> wp.Function:
    shade_gaussians = gaussians.create_shade_function(config, state)
    get_group_query_count, get_group_query = create_group_query_functions(config)

    @wp.func
    def closest_hit_shape_depth_only(
        closest_hit: ClosestHit,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        shape_mesh_data_ids: wp.array[wp.int32],
        mesh_data: wp.array[MeshData],
        gaussians_data: wp.array[Gaussian.Data],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        camera_forward: wp.vec3f,
    ) -> ClosestHit:
        if bvh_shapes_size:
            query_count = get_group_query_count(bvh_shapes_group_roots)
            for i in range(query_count):
                group_query = get_group_query(i, bvh_shapes_group_roots, world_index, world_offsets, ray_origin_world)
                if bvh_shapes_group_roots[group_query.root_index] < 0:
                    continue

                gaussians_hit = wp.vector(length=wp.static(state.num_gaussians), dtype=wp.uint32)
                num_gaussians_hit = wp.int32(0)

                query = wp.bvh_query_ray(
                    bvh_shapes_id,
                    group_query.ray_origin_world,
                    ray_dir_world,
                    bvh_shapes_group_roots[group_query.root_index],
                )
                shape_index = wp.int32(0)

                while wp.bvh_query_next(query, shape_index, closest_hit.distance):
                    si = shape_enabled[shape_index]

                    hit_dist = -1.0

                    shape_type = shape_types[si]
                    # Heightfields are triangulated meshes; RenderContext remaps
                    # HFIELD -> MESH, so this branch renders them too.
                    if shape_type == GeoType.MESH:
                        ray_origin_local, ray_direction_local = raycast.map_ray_to_local(
                            shape_transforms[si], group_query.ray_origin_world, ray_dir_world, shape_sizes[si]
                        )
                        hit_dist, _normal, _u, _v, _face = raycast.ray_intersect_mesh_no_normal(
                            ray_origin_local,
                            ray_direction_local,
                            shape_sizes[si],
                            shape_source_ptr[si],
                            wp.static(config.enable_backface_culling),
                            closest_hit.distance,
                        )
                    elif shape_type == GeoType.GAUSSIAN:
                        if num_gaussians_hit < wp.static(state.num_gaussians):
                            gaussians_hit[num_gaussians_hit] = si
                            num_gaussians_hit += 1
                    else:
                        hit_dist, _normal = raycast.ray_intersect_shape_no_normal(
                            shape_transforms[si],
                            shape_sizes[si],
                            shape_type,
                            group_query.ray_origin_world,
                            ray_dir_world,
                            wp.static(config.enable_backface_culling),
                        )

                    if hit_dist > -1.0 and hit_dist < closest_hit.distance:
                        closest_hit.distance = hit_dist
                        closest_hit.shape_index = si
                        closest_hit.world_index = group_query.world_index

                if num_gaussians_hit > 0:
                    for gi in range(num_gaussians_hit):
                        si = gaussians_hit[gi]

                        gaussian_id = shape_source_ptr[si]
                        hit_distance, _hit_normal, _hit_color = shade_gaussians(
                            shape_transforms[si],
                            shape_sizes[si],
                            group_query.ray_origin_world,
                            ray_dir_world,
                            camera_forward,
                            gaussians_data[gaussian_id],
                            closest_hit.distance,
                        )

                        if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                            closest_hit.distance = hit_distance
                            closest_hit.shape_index = si
                            closest_hit.world_index = group_query.world_index

        return closest_hit

    @wp.func
    def closest_hit_particles_depth_only(
        closest_hit: ClosestHit,
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
    ) -> ClosestHit:
        if bvh_particles_size:
            query_count = get_group_query_count(bvh_particles_group_roots)
            for i in range(query_count):
                group_query = get_group_query(
                    i, bvh_particles_group_roots, world_index, world_offsets, ray_origin_world
                )
                if bvh_particles_group_roots[group_query.root_index] < 0:
                    continue

                query = wp.bvh_query_ray(
                    bvh_particles_id,
                    group_query.ray_origin_world,
                    ray_dir_world,
                    bvh_particles_group_roots[group_query.root_index],
                )
                si = wp.int32(0)

                while wp.bvh_query_next(query, si, closest_hit.distance):
                    if topology_particle_mask[si]:
                        continue

                    hit_dist, _normal = raycast.ray_intersect_particle_sphere(
                        group_query.ray_origin_world,
                        ray_dir_world,
                        particles_position[si],
                        particles_radius[si],
                    )

                    if hit_dist > -1.0 and hit_dist < closest_hit.distance:
                        closest_hit.distance = hit_dist
                        closest_hit.shape_index = PARTICLES_SHAPE_ID
                        closest_hit.world_index = group_query.world_index

        return closest_hit

    @wp.func
    def closest_hit_triangle_mesh_depth_only(
        closest_hit: ClosestHit,
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
    ) -> ClosestHit:
        if triangle_mesh_id:
            # Triangle mesh is in world space; its local frame is the world frame (see
            # closest_hit_triangle_mesh).
            query_count = get_group_query_count(triangle_mesh_group_roots)
            for i in range(query_count):
                group_query = get_group_query(
                    i, triangle_mesh_group_roots, world_index, world_offsets, ray_origin_world
                )
                if triangle_mesh_group_roots[group_query.root_index] < 0:
                    continue

                hit_dist, _normal, _bary_u, _bary_v, _face_idx = raycast.ray_intersect_mesh_no_normal(
                    group_query.ray_origin_world,
                    ray_dir_world,
                    wp.vec3f(1.0),
                    triangle_mesh_id,
                    wp.static(config.enable_backface_culling),
                    closest_hit.distance,
                    triangle_mesh_group_roots[group_query.root_index],
                )
                if hit_dist >= 0.0:
                    closest_hit.distance = hit_dist
                    closest_hit.shape_index = TRIANGLE_MESH_SHAPE_ID
                    closest_hit.world_index = group_query.world_index

        return closest_hit

    @wp.func
    def closest_hit_depth_only(
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        max_distance: wp.float32,
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        shape_mesh_data_ids: wp.array[wp.int32],
        mesh_data: wp.array[MeshData],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        gaussians_data: wp.array[Gaussian.Data],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        camera_forward: wp.vec3f,
    ) -> ClosestHit:
        closest_hit = ClosestHit()
        closest_hit.distance = max_distance
        closest_hit.shape_index = NO_HIT_SHAPE_ID
        closest_hit.world_index = wp.int32(-1)

        closest_hit = closest_hit_triangle_mesh_depth_only(
            closest_hit,
            triangle_mesh_id,
            triangle_mesh_group_roots,
            world_index,
            world_offsets,
            ray_origin_world,
            ray_dir_world,
        )

        closest_hit = closest_hit_shape_depth_only(
            closest_hit,
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            world_index,
            world_offsets,
            shape_enabled,
            shape_types,
            shape_sizes,
            shape_transforms,
            shape_source_ptr,
            shape_mesh_data_ids,
            mesh_data,
            gaussians_data,
            ray_origin_world,
            ray_dir_world,
            camera_forward,
        )

        if wp.static(config.enable_particles) and wp.static(state.has_particles):
            closest_hit = closest_hit_particles_depth_only(
                closest_hit,
                bvh_particles_size,
                bvh_particles_id,
                bvh_particles_group_roots,
                world_index,
                world_offsets,
                particles_position,
                particles_radius,
                topology_particle_mask,
                ray_origin_world,
                ray_dir_world,
            )

        return closest_hit

    return closest_hit_depth_only


def create_first_hit_function(config: RenderContext.Config, state: RenderContext.State) -> wp.Function:
    get_group_query_count, get_group_query = create_group_query_functions(config)

    @wp.func
    def first_hit_shape(
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        max_dist: wp.float32,
    ) -> wp.bool:
        if bvh_shapes_size:
            query_count = get_group_query_count(bvh_shapes_group_roots)
            for i in range(query_count):
                group_query = get_group_query(i, bvh_shapes_group_roots, world_index, world_offsets, ray_origin_world)
                if bvh_shapes_group_roots[group_query.root_index] < 0:
                    continue

                query = wp.bvh_query_ray(
                    bvh_shapes_id,
                    group_query.ray_origin_world,
                    ray_dir_world,
                    bvh_shapes_group_roots[group_query.root_index],
                )
                shape_index = wp.int32(0)

                while wp.bvh_query_next(query, shape_index, max_dist):
                    si = shape_enabled[shape_index]

                    hit_dist = wp.float32(-1)

                    shape_type = shape_types[si]
                    # Heightfields are triangulated meshes; RenderContext remaps
                    # HFIELD -> MESH, so this branch renders them too.
                    if shape_type == GeoType.MESH:
                        ray_origin_local, ray_direction_local = raycast.map_ray_to_local(
                            shape_transforms[si], group_query.ray_origin_world, ray_dir_world, shape_sizes[si]
                        )
                        hit_dist = raycast.ray_intersect_mesh_anyhit(
                            ray_origin_local,
                            ray_direction_local,
                            shape_source_ptr[si],
                            max_dist,
                        )
                    else:
                        hit_dist, _normal = raycast.ray_intersect_shape_no_normal(
                            shape_transforms[si],
                            shape_sizes[si],
                            shape_type,
                            group_query.ray_origin_world,
                            ray_dir_world,
                            wp.static(config.enable_backface_culling),
                        )
                    if hit_dist > -1 and hit_dist < max_dist:
                        return True

        return False

    @wp.func
    def first_hit_particles(
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        max_dist: wp.float32,
    ) -> wp.bool:
        if bvh_particles_size:
            query_count = get_group_query_count(bvh_particles_group_roots)
            for i in range(query_count):
                group_query = get_group_query(
                    i, bvh_particles_group_roots, world_index, world_offsets, ray_origin_world
                )
                if bvh_particles_group_roots[group_query.root_index] < 0:
                    continue

                query = wp.bvh_query_ray(
                    bvh_particles_id,
                    group_query.ray_origin_world,
                    ray_dir_world,
                    bvh_particles_group_roots[group_query.root_index],
                )
                si = wp.int32(0)

                while wp.bvh_query_next(query, si, max_dist):
                    if topology_particle_mask[si]:
                        continue

                    hit_dist, _normal = raycast.ray_intersect_particle_sphere(
                        group_query.ray_origin_world,
                        ray_dir_world,
                        particles_position[si],
                        particles_radius[si],
                    )

                    if hit_dist > -1.0 and hit_dist < max_dist:
                        return True

        return False

    @wp.func
    def first_hit_triangle_mesh(
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        max_dist: wp.float32,
    ) -> wp.bool:
        if triangle_mesh_id:
            # Triangle mesh is in world space; its local frame is the world frame (see
            # closest_hit_triangle_mesh). Shadow rays only need any hit within ``max_dist``.
            query_count = get_group_query_count(triangle_mesh_group_roots)
            for i in range(query_count):
                group_query = get_group_query(
                    i, triangle_mesh_group_roots, world_index, world_offsets, ray_origin_world
                )
                if triangle_mesh_group_roots[group_query.root_index] < 0:
                    continue

                hit_dist = raycast.ray_intersect_mesh_anyhit(
                    group_query.ray_origin_world,
                    ray_dir_world,
                    triangle_mesh_id,
                    max_dist,
                    triangle_mesh_group_roots[group_query.root_index],
                )
                if hit_dist >= 0.0:
                    return True
        return False

    @wp.func
    def first_hit(
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        world_offsets: wp.array[wp.vec3f],
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        max_distance: wp.float32,
    ) -> wp.bool:
        if first_hit_triangle_mesh(
            triangle_mesh_id,
            triangle_mesh_group_roots,
            world_index,
            world_offsets,
            ray_origin_world,
            ray_dir_world,
            max_distance,
        ):
            return True

        if first_hit_shape(
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            world_index,
            world_offsets,
            shape_enabled,
            shape_types,
            shape_sizes,
            shape_transforms,
            shape_source_ptr,
            ray_origin_world,
            ray_dir_world,
            max_distance,
        ):
            return True

        if wp.static(config.enable_particles) and wp.static(state.has_particles):
            if first_hit_particles(
                bvh_particles_size,
                bvh_particles_id,
                bvh_particles_group_roots,
                world_index,
                world_offsets,
                particles_position,
                particles_radius,
                topology_particle_mask,
                ray_origin_world,
                ray_dir_world,
                max_distance,
            ):
                return True

        return False

    return first_hit
