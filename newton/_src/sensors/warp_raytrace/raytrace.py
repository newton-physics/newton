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

_BACKFACE_EPS = 1.0e-6


@wp.struct
class ClosestHit:
    distance: wp.float32
    normal: wp.vec3f
    shape_index: wp.uint32
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
    """Ray-mesh intersection with optional per-vertex normal interpolation.

    When ``shape_mesh_data_id`` is non-negative and the referenced ``mesh_data`` entry
    supplies per-vertex normals, the returned normal is the barycentric interpolation
    of those vertex normals (for smooth shading); otherwise the triangle's face normal
    is used.
    """
    ray_origin_local, ray_direction_local = raycast.map_ray_to_local_scaled(transform, scale, ray_origin, ray_direction)

    query = wp.mesh_query_ray(mesh_id, ray_origin_local, ray_direction_local, max_t)

    if query.result:
        if not enable_backface_culling or wp.dot(ray_direction_local, query.normal) < 0.0:
            normal_local = query.normal

            if shape_mesh_data_id > -1:
                normals = mesh_data[shape_mesh_data_id].normals
                if normals.shape[0] > 0:
                    n0 = wp.mesh_get_index(mesh_id, query.face * 3 + 0)
                    n1 = wp.mesh_get_index(mesh_id, query.face * 3 + 1)
                    n2 = wp.mesh_get_index(mesh_id, query.face * 3 + 2)
                    normal_local = (
                        normals[n0] * query.u + normals[n1] * query.v + normals[n2] * (1.0 - query.u - query.v)
                    )

            normal_world = wp.transform_vector(transform, raycast.safe_div_vec3(normal_local, scale))
            normal_world = wp.normalize(normal_world)
            return query.t, normal_world, query.u, query.v, query.face

    return wp.float32(-1.0), wp.vec3f(0.0), wp.float32(0.0), wp.float32(0.0), wp.int32(-1)


@wp.func
def _plane_hit_with_culling(
    transform: wp.transformf,
    size: wp.vec3f,
    ray_origin: wp.vec3f,
    ray_direction: wp.vec3f,
    enable_backface_culling: wp.bool,
) -> tuple[wp.float32, wp.vec3f]:
    """Ray-plane intersection; when ``enable_backface_culling`` is set, rejects rays that
    approach the plane from behind (ray direction aligned with the plane normal)."""
    hit_distance, hit_normal = raycast.ray_intersect_plane(transform, ray_origin, ray_direction, size)
    if enable_backface_culling and hit_distance >= 0.0:
        if wp.dot(ray_direction, hit_normal) > -_BACKFACE_EPS:
            return wp.float32(-1.0), wp.vec3f(0.0)
    return hit_distance, hit_normal


@wp.func
def get_group_roots(group_roots: wp.array[wp.int32], world_index: wp.int32, want_global_world: wp.int32) -> wp.int32:
    if want_global_world != 0:
        return group_roots[group_roots.shape[0] - 1]
    return group_roots[world_index]


def create_closest_hit_function(config: RenderContext.Config, state: RenderContext.State) -> wp.Function:
    shade_gaussians = gaussians.create_shade_function(config, state)

    @wp.func
    def closest_hit_shape(
        closest_hit: ClosestHit,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
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
            for i in range(wp.static(2 if config.enable_global_world else 1)):
                group_root = get_group_roots(bvh_shapes_group_roots, world_index, i)
                if group_root < 0:
                    continue

                gaussians_hit = wp.vector(length=wp.static(state.num_gaussians), dtype=wp.uint32)
                num_gaussians_hit = wp.int32(0)

                query = wp.bvh_query_ray(bvh_shapes_id, ray_origin_world, ray_dir_world, group_root)
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
                    if shape_type == GeoType.MESH:
                        hit_distance, hit_normal, hit_u, hit_v, hit_face_id = _ray_intersect_mesh_smooth(
                            shape_transforms[si],
                            shape_sizes[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_source_ptr[si],
                            shape_mesh_data_ids[si],
                            mesh_data,
                            wp.static(config.enable_backface_culling),
                            closest_hit.distance,
                        )
                    elif shape_type == GeoType.PLANE:
                        hit_distance, hit_normal = _plane_hit_with_culling(
                            shape_transforms[si],
                            shape_sizes[si],
                            ray_origin_world,
                            ray_dir_world,
                            wp.static(config.enable_backface_culling),
                        )
                    elif shape_type == GeoType.SPHERE:
                        hit_distance, hit_normal = raycast.ray_intersect_sphere(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                        )
                    elif shape_type == GeoType.ELLIPSOID:
                        hit_distance, hit_normal = raycast.ray_intersect_ellipsoid(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si],
                        )
                    elif shape_type == GeoType.CAPSULE:
                        hit_distance, hit_normal = raycast.ray_intersect_capsule(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                            shape_sizes[si][1],
                        )
                    elif shape_type == GeoType.CYLINDER:
                        hit_distance, hit_normal = raycast.ray_intersect_cylinder(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                            shape_sizes[si][1],
                        )
                    elif shape_type == GeoType.CONE:
                        hit_distance, hit_normal = raycast.ray_intersect_cone(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                            shape_sizes[si][1],
                        )
                    elif shape_type == GeoType.BOX:
                        hit_distance, hit_normal = raycast.ray_intersect_box(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si],
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

                    if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                        closest_hit.distance = hit_distance
                        closest_hit.normal = hit_normal
                        closest_hit.shape_index = si
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
                            ray_origin_world,
                            ray_dir_world,
                            camera_forward,
                            gaussians_data[gaussian_id],
                            closest_hit.distance,
                        )

                        if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                            closest_hit.distance = hit_distance
                            closest_hit.normal = hit_normal
                            closest_hit.shape_index = si
                            closest_hit.color = hit_color

        return closest_hit

    @wp.func
    def closest_hit_particles(
        closest_hit: ClosestHit,
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
    ) -> ClosestHit:
        if bvh_particles_size:
            for i in range(wp.static(2 if config.enable_global_world else 1)):
                group_root = get_group_roots(bvh_particles_group_roots, world_index, i)
                if group_root < 0:
                    continue

                query = wp.bvh_query_ray(bvh_particles_id, ray_origin_world, ray_dir_world, group_root)
                si = wp.int32(0)

                while wp.bvh_query_next(query, si, closest_hit.distance):
                    hit_distance, hit_normal = raycast.ray_intersect_particle_sphere(
                        ray_origin_world,
                        ray_dir_world,
                        particles_position[si],
                        particles_radius[si],
                    )

                    if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                        closest_hit.distance = hit_distance
                        closest_hit.normal = hit_normal
                        closest_hit.shape_index = PARTICLES_SHAPE_ID

        return closest_hit

    @wp.func
    def closest_hit_triangle_mesh(
        closest_hit: ClosestHit,
        triangle_mesh_id: wp.uint64,
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
    ) -> ClosestHit:
        if triangle_mesh_id:
            hit_distance, hit_normal, bary_u, bary_v, face_idx = raycast.ray_intersect_mesh_no_transform(
                triangle_mesh_id,
                ray_origin_world,
                ray_dir_world,
                wp.static(config.enable_backface_culling),
                closest_hit.distance,
            )
            if hit_distance >= 0.0:
                closest_hit.distance = hit_distance
                closest_hit.normal = hit_normal
                closest_hit.shape_index = TRIANGLE_MESH_SHAPE_ID
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
        triangle_mesh_id: wp.uint64,
        gaussians_data: wp.array[Gaussian.Data],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        camera_forward: wp.vec3f,
    ) -> ClosestHit:
        closest_hit = ClosestHit()
        closest_hit.distance = max_distance
        closest_hit.shape_index = NO_HIT_SHAPE_ID
        closest_hit.color = wp.vec3f(0.0)

        closest_hit = closest_hit_triangle_mesh(closest_hit, triangle_mesh_id, ray_origin_world, ray_dir_world)

        closest_hit = closest_hit_shape(
            closest_hit,
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            world_index,
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

        if wp.static(config.enable_particles):
            closest_hit = closest_hit_particles(
                closest_hit,
                bvh_particles_size,
                bvh_particles_id,
                bvh_particles_group_roots,
                world_index,
                particles_position,
                particles_radius,
                ray_origin_world,
                ray_dir_world,
            )

        return closest_hit

    return closest_hit


def create_closest_hit_depth_only_function(config: RenderContext.Config, state: RenderContext.State) -> wp.Function:
    shade_gaussians = gaussians.create_shade_function(config, state)

    @wp.func
    def closest_hit_shape_depth_only(
        closest_hit: ClosestHit,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
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
            for i in range(wp.static(2 if config.enable_global_world else 1)):
                group_root = get_group_roots(bvh_shapes_group_roots, world_index, i)
                if group_root < 0:
                    continue

                gaussians_hit = wp.vector(length=wp.static(state.num_gaussians), dtype=wp.uint32)
                num_gaussians_hit = wp.int32(0)

                query = wp.bvh_query_ray(bvh_shapes_id, ray_origin_world, ray_dir_world, group_root)
                shape_index = wp.int32(0)

                while wp.bvh_query_next(query, shape_index, closest_hit.distance):
                    si = shape_enabled[shape_index]

                    hit_dist = -1.0

                    shape_type = shape_types[si]
                    if shape_type == GeoType.MESH:
                        hit_dist, _normal, _u, _v, _face = raycast.ray_intersect_mesh(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si],
                            shape_source_ptr[si],
                            wp.static(config.enable_backface_culling),
                            closest_hit.distance,
                        )
                    elif shape_type == GeoType.PLANE:
                        hit_dist, _plane_normal = _plane_hit_with_culling(
                            shape_transforms[si],
                            shape_sizes[si],
                            ray_origin_world,
                            ray_dir_world,
                            wp.static(config.enable_backface_culling),
                        )
                    elif shape_type == GeoType.SPHERE:
                        hit_dist, _normal = raycast.ray_intersect_sphere(
                            shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si][0]
                        )
                    elif shape_type == GeoType.ELLIPSOID:
                        hit_dist, _normal = raycast.ray_intersect_ellipsoid(
                            shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si]
                        )
                    elif shape_type == GeoType.CAPSULE:
                        hit_dist, _normal = raycast.ray_intersect_capsule(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                            shape_sizes[si][1],
                        )
                    elif shape_type == GeoType.CYLINDER:
                        hit_dist, _normal = raycast.ray_intersect_cylinder(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                            shape_sizes[si][1],
                        )
                    elif shape_type == GeoType.CONE:
                        hit_dist, _normal = raycast.ray_intersect_cone(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                            shape_sizes[si][1],
                        )
                    elif shape_type == GeoType.BOX:
                        hit_dist, _normal = raycast.ray_intersect_box(
                            shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si]
                        )
                    elif shape_type == GeoType.GAUSSIAN:
                        if num_gaussians_hit < wp.static(state.num_gaussians):
                            gaussians_hit[num_gaussians_hit] = si
                            num_gaussians_hit += 1

                    if hit_dist > -1.0 and hit_dist < closest_hit.distance:
                        closest_hit.distance = hit_dist
                        closest_hit.shape_index = si

                if num_gaussians_hit > 0:
                    for gi in range(num_gaussians_hit):
                        si = gaussians_hit[gi]

                        gaussian_id = shape_source_ptr[si]
                        hit_distance, _hit_normal, _hit_color = shade_gaussians(
                            shape_transforms[si],
                            shape_sizes[si],
                            ray_origin_world,
                            ray_dir_world,
                            camera_forward,
                            gaussians_data[gaussian_id],
                            closest_hit.distance,
                        )

                        if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                            closest_hit.distance = hit_distance
                            closest_hit.shape_index = si

        return closest_hit

    @wp.func
    def closest_hit_particles_depth_only(
        closest_hit: ClosestHit,
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
    ) -> ClosestHit:
        if bvh_particles_size:
            for i in range(wp.static(2 if config.enable_global_world else 1)):
                group_root = get_group_roots(bvh_particles_group_roots, world_index, i)
                if group_root < 0:
                    continue

                query = wp.bvh_query_ray(bvh_particles_id, ray_origin_world, ray_dir_world, group_root)
                si = wp.int32(0)

                while wp.bvh_query_next(query, si, closest_hit.distance):
                    hit_dist, _normal = raycast.ray_intersect_particle_sphere(
                        ray_origin_world,
                        ray_dir_world,
                        particles_position[si],
                        particles_radius[si],
                    )

                    if hit_dist > -1.0 and hit_dist < closest_hit.distance:
                        closest_hit.distance = hit_dist
                        closest_hit.shape_index = PARTICLES_SHAPE_ID

        return closest_hit

    @wp.func
    def closest_hit_triangle_mesh_depth_only(
        closest_hit: ClosestHit,
        triangle_mesh_id: wp.uint64,
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
    ) -> ClosestHit:
        if triangle_mesh_id:
            hit_dist, _normal, _bary_u, _bary_v, _face_idx = raycast.ray_intersect_mesh_no_transform(
                triangle_mesh_id,
                ray_origin_world,
                ray_dir_world,
                wp.static(config.enable_backface_culling),
                closest_hit.distance,
            )
            if hit_dist >= 0.0:
                closest_hit.distance = hit_dist
                closest_hit.shape_index = TRIANGLE_MESH_SHAPE_ID

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
        triangle_mesh_id: wp.uint64,
        gaussians_data: wp.array[Gaussian.Data],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        camera_forward: wp.vec3f,
    ) -> ClosestHit:
        closest_hit = ClosestHit()
        closest_hit.distance = max_distance
        closest_hit.shape_index = NO_HIT_SHAPE_ID

        closest_hit = closest_hit_triangle_mesh_depth_only(
            closest_hit, triangle_mesh_id, ray_origin_world, ray_dir_world
        )

        closest_hit = closest_hit_shape_depth_only(
            closest_hit,
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            world_index,
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

        if wp.static(config.enable_particles):
            closest_hit = closest_hit_particles_depth_only(
                closest_hit,
                bvh_particles_size,
                bvh_particles_id,
                bvh_particles_group_roots,
                world_index,
                particles_position,
                particles_radius,
                ray_origin_world,
                ray_dir_world,
            )

        return closest_hit

    return closest_hit_depth_only


def create_first_hit_function(config: RenderContext.Config, state: RenderContext.State) -> wp.Function:
    @wp.func
    def first_hit_shape(
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
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
            for i in range(wp.static(2 if config.enable_global_world else 1)):
                group_root = get_group_roots(bvh_shapes_group_roots, world_index, i)
                if group_root < 0:
                    continue

                query = wp.bvh_query_ray(bvh_shapes_id, ray_origin_world, ray_dir_world, group_root)
                shape_index = wp.int32(0)

                while wp.bvh_query_next(query, shape_index, max_dist):
                    si = shape_enabled[shape_index]

                    hit_dist = wp.float32(-1)

                    shape_type = shape_types[si]
                    if shape_type == GeoType.MESH:
                        hit_dist, _normal, _u, _v, _face = raycast.ray_intersect_mesh(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si],
                            shape_source_ptr[si],
                            False,
                            max_dist,
                        )
                    elif shape_type == GeoType.PLANE:
                        hit_dist, _plane_normal = _plane_hit_with_culling(
                            shape_transforms[si],
                            shape_sizes[si],
                            ray_origin_world,
                            ray_dir_world,
                            wp.static(config.enable_backface_culling),
                        )
                    elif shape_type == GeoType.SPHERE:
                        hit_dist, _normal = raycast.ray_intersect_sphere(
                            shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si][0]
                        )
                    elif shape_type == GeoType.ELLIPSOID:
                        hit_dist, _normal = raycast.ray_intersect_ellipsoid(
                            shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si]
                        )
                    elif shape_type == GeoType.CAPSULE:
                        hit_dist, _normal = raycast.ray_intersect_capsule(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                            shape_sizes[si][1],
                        )
                    elif shape_type == GeoType.CYLINDER:
                        hit_dist, _normal = raycast.ray_intersect_cylinder(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                            shape_sizes[si][1],
                        )
                    elif shape_type == GeoType.CONE:
                        hit_dist, _normal = raycast.ray_intersect_cone(
                            shape_transforms[si],
                            ray_origin_world,
                            ray_dir_world,
                            shape_sizes[si][0],
                            shape_sizes[si][1],
                        )
                    elif shape_type == GeoType.BOX:
                        hit_dist, _normal = raycast.ray_intersect_box(
                            shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si]
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
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        max_dist: wp.float32,
    ) -> wp.bool:
        if bvh_particles_size:
            for i in range(wp.static(2 if config.enable_global_world else 1)):
                group_root = get_group_roots(bvh_particles_group_roots, world_index, i)
                if group_root < 0:
                    continue

                query = wp.bvh_query_ray(bvh_particles_id, ray_origin_world, ray_dir_world, group_root)
                si = wp.int32(0)

                while wp.bvh_query_next(query, si, max_dist):
                    hit_dist, _normal = raycast.ray_intersect_particle_sphere(
                        ray_origin_world,
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
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        max_dist: wp.float32,
    ) -> wp.bool:
        if triangle_mesh_id:
            hit_dist, _normal, _bary_u, _bary_v, _face_idx = raycast.ray_intersect_mesh_no_transform(
                triangle_mesh_id, ray_origin_world, ray_dir_world, wp.static(config.enable_backface_culling), max_dist
            )
            return hit_dist >= 0.0
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
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        triangle_mesh_id: wp.uint64,
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        max_distance: wp.float32,
    ) -> wp.bool:
        if first_hit_triangle_mesh(triangle_mesh_id, ray_origin_world, ray_dir_world, max_distance):
            return True

        if first_hit_shape(
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            world_index,
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

        if wp.static(config.enable_particles):
            if first_hit_particles(
                bvh_particles_size,
                bvh_particles_id,
                bvh_particles_group_roots,
                world_index,
                particles_position,
                particles_radius,
                ray_origin_world,
                ray_dir_world,
                max_distance,
            ):
                return True

        return False

    return first_hit
