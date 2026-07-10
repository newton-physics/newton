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
def _plane_hit_with_culling_local(
    ray_origin_local: wp.vec3f,
    ray_direction_local: wp.vec3f,
    size: wp.vec3f,
    enable_backface_culling: wp.bool,
) -> tuple[wp.float32, wp.vec3f]:
    """Local-frame ray-plane intersection; when ``enable_backface_culling`` is set,
    rejects rays that approach the plane from behind. The plane normal is +Z in
    local space, so the world-space test ``dot(ray_dir, normal) > -eps`` reduces
    to the local ray direction's Z component (rigid transforms preserve dot
    products)."""
    hit_distance, hit_normal = raycast.ray_intersect_plane_local(ray_origin_local, ray_direction_local, size)
    if enable_backface_culling and hit_distance >= 0.0:
        if ray_direction_local[2] > -_BACKFACE_EPS:
            return wp.float32(-1.0), wp.vec3f(0.0)
    return hit_distance, hit_normal


@wp.func
def _intersect_primitive_local(
    shape_type: wp.int32,
    ray_origin_local: wp.vec3f,
    ray_direction_local: wp.vec3f,
    size: wp.vec3f,
    enable_backface_culling: wp.bool,
) -> tuple[wp.float32, wp.vec3f]:
    """Dispatch a local-frame ray against one primitive's local-frame intersector.

    The ray is mapped into the shape's frame once by the caller and shared by
    every branch, so the divergent shape-type switch only contains the
    per-primitive math. The returned normal is local-space and unnormalized;
    callers convert the *winning* hit to a world normal once after traversal
    instead of per candidate. Meshes are handled separately by callers
    (they additionally produce barycentrics and a face index, and shadow rays
    use the cheaper any-hit query).
    """
    hit_distance = wp.float32(-1.0)
    hit_normal_local = wp.vec3f(0.0)

    if shape_type == GeoType.PLANE:
        hit_distance, hit_normal_local = _plane_hit_with_culling_local(
            ray_origin_local,
            ray_direction_local,
            size,
            enable_backface_culling,
        )
    elif shape_type == GeoType.SPHERE:
        hit_distance, hit_normal_local = raycast.ray_intersect_sphere_local(
            ray_origin_local, ray_direction_local, size[0]
        )
    elif shape_type == GeoType.ELLIPSOID:
        hit_distance, hit_normal_local = raycast.ray_intersect_ellipsoid_local(
            ray_origin_local, ray_direction_local, size
        )
    elif shape_type == GeoType.CAPSULE:
        hit_distance, hit_normal_local = raycast.ray_intersect_capsule_local(
            ray_origin_local, ray_direction_local, size[0], size[1]
        )
    elif shape_type == GeoType.CYLINDER:
        hit_distance, hit_normal_local = raycast.ray_intersect_cylinder_local(
            ray_origin_local, ray_direction_local, size[0], size[1]
        )
    elif shape_type == GeoType.CONE:
        hit_distance, hit_normal_local = raycast.ray_intersect_cone_local(
            ray_origin_local, ray_direction_local, size[0], size[1]
        )
    elif shape_type == GeoType.BOX:
        hit_distance, hit_normal_local = raycast.ray_intersect_box_local(ray_origin_local, ray_direction_local, size)

    return hit_distance, hit_normal_local


@wp.func
def _finalize_shape_normal(
    closest_hit: ClosestHit,
    shape_types: wp.array[wp.int32],
    shape_sizes: wp.array[wp.vec3f],
    shape_transforms: wp.array[wp.transformf],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_data_ids: wp.array[wp.int32],
    mesh_data: wp.array[MeshData],
) -> ClosestHit:
    """Convert the winning shape hit's local normal to a world-space normal.

    Runs once per ray after traversal, so per-vertex normal interpolation
    (smooth shading) and the local-to-world normal transform are paid only for
    the visible hit instead of for every BVH candidate along the ray.
    """
    if closest_hit.shape_index < MAX_SHAPE_ID:
        si = wp.int32(closest_hit.shape_index)
        shape_type = shape_types[si]

        # Gaussian hits already carry a world-space normal from shading.
        if shape_type != GeoType.GAUSSIAN:
            normal_local = closest_hit.normal

            if shape_type == GeoType.MESH:
                shape_mesh_data_id = shape_mesh_data_ids[si]
                if shape_mesh_data_id > -1:
                    normals = mesh_data[shape_mesh_data_id].normals
                    if normals.shape[0] > 0:
                        mesh_id = shape_source_ptr[si]
                        n0 = wp.mesh_get_index(mesh_id, closest_hit.face_idx * 3 + 0)
                        n1 = wp.mesh_get_index(mesh_id, closest_hit.face_idx * 3 + 1)
                        n2 = wp.mesh_get_index(mesh_id, closest_hit.face_idx * 3 + 2)
                        normal_local = (
                            normals[n0] * closest_hit.bary_u
                            + normals[n1] * closest_hit.bary_v
                            + normals[n2] * (1.0 - closest_hit.bary_u - closest_hit.bary_v)
                        )
                normal_local = raycast.safe_div_vec3(normal_local, shape_sizes[si])

            closest_hit.normal = wp.normalize(wp.transform_vector(shape_transforms[si], normal_local))

    return closest_hit


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
                    shape_type = shape_types[si]

                    if wp.static(state.num_gaussians > 0):
                        if shape_type == GeoType.GAUSSIAN:
                            # Gaussians are shaded after traversal (see below).
                            if num_gaussians_hit < wp.static(state.num_gaussians):
                                gaussians_hit[num_gaussians_hit] = si
                                num_gaussians_hit += 1

                    # Map the ray into the shape's frame once; every shape-type
                    # intersector shares it (see _intersect_primitive_local).
                    ray_origin_local, ray_dir_local = raycast.map_ray_to_local(
                        shape_transforms[si], ray_origin_world, ray_dir_world
                    )
                    shape_size = shape_sizes[si]

                    hit_distance = wp.float32(-1.0)
                    hit_normal_local = wp.vec3f(0.0)
                    hit_u = wp.float32(0.0)
                    hit_v = wp.float32(0.0)
                    hit_face_id = wp.int32(-1)

                    # Heightfields are triangulated meshes; RenderContext remaps
                    # HFIELD -> MESH, so this branch renders them too.
                    if shape_type == GeoType.MESH:
                        hit_distance, hit_normal_local, hit_u, hit_v, hit_face_id = raycast.ray_intersect_mesh_local(
                            ray_origin_local,
                            ray_dir_local,
                            shape_size,
                            shape_source_ptr[si],
                            wp.static(config.enable_backface_culling),
                            closest_hit.distance,
                        )
                    else:
                        hit_distance, hit_normal_local = _intersect_primitive_local(
                            shape_type,
                            ray_origin_local,
                            ray_dir_local,
                            shape_size,
                            wp.static(config.enable_backface_culling),
                        )

                    if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                        # The normal stays in local space until the traversal
                        # ends; _finalize_shape_normal converts the winner.
                        closest_hit.distance = hit_distance
                        closest_hit.normal = hit_normal_local
                        closest_hit.shape_index = si
                        closest_hit.bary_u = hit_u
                        closest_hit.bary_v = hit_v
                        closest_hit.face_idx = hit_face_id
                        closest_hit.color = wp.vec3f(0.0)

                # Temporary workaround. Warp BVH queries share some stack data,
                # which breaks nested wp.bvh_query_ray calls.
                # Once it is fixed in Warp, remove this code block and shade
                # gaussians inside the traversal loop.
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

            closest_hit = _finalize_shape_normal(
                closest_hit,
                shape_types,
                shape_sizes,
                shape_transforms,
                shape_source_ptr,
                shape_mesh_data_ids,
                mesh_data,
            )

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
                    shape_type = shape_types[si]

                    if wp.static(state.num_gaussians > 0):
                        if shape_type == GeoType.GAUSSIAN:
                            # Gaussians are shaded after traversal (see below).
                            if num_gaussians_hit < wp.static(state.num_gaussians):
                                gaussians_hit[num_gaussians_hit] = si
                                num_gaussians_hit += 1

                    # Map the ray into the shape's frame once; every shape-type
                    # intersector shares it (see _intersect_primitive_local). The
                    # normal outputs are unused here, so their computation is
                    # eliminated by the compiler.
                    ray_origin_local, ray_dir_local = raycast.map_ray_to_local(
                        shape_transforms[si], ray_origin_world, ray_dir_world
                    )
                    shape_size = shape_sizes[si]

                    hit_dist = wp.float32(-1.0)
                    # Heightfields are triangulated meshes; RenderContext remaps
                    # HFIELD -> MESH, so this branch renders them too.
                    if shape_type == GeoType.MESH:
                        hit_dist, _normal_local, _u, _v, _face = raycast.ray_intersect_mesh_local(
                            ray_origin_local,
                            ray_dir_local,
                            shape_size,
                            shape_source_ptr[si],
                            wp.static(config.enable_backface_culling),
                            closest_hit.distance,
                        )
                    else:
                        hit_dist, _normal_local = _intersect_primitive_local(
                            shape_type,
                            ray_origin_local,
                            ray_dir_local,
                            shape_size,
                            wp.static(config.enable_backface_culling),
                        )

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
                    shape_type = shape_types[si]

                    # Map the ray into the shape's frame once; every shape-type
                    # intersector shares it (see _intersect_primitive_local). The
                    # normal outputs are unused here, so their computation is
                    # eliminated by the compiler.
                    ray_origin_local, ray_dir_local = raycast.map_ray_to_local(
                        shape_transforms[si], ray_origin_world, ray_dir_world
                    )
                    shape_size = shape_sizes[si]

                    hit_dist = wp.float32(-1)
                    # Heightfields are triangulated meshes; RenderContext remaps
                    # HFIELD -> MESH, so this branch renders them too.
                    if shape_type == GeoType.MESH:
                        # Meshes take the cheaper any-hit query: shadow rays only
                        # need occlusion, not the closest triangle.
                        hit_dist = raycast.ray_intersect_mesh_anyhit_local(
                            ray_origin_local,
                            ray_dir_local,
                            shape_size,
                            shape_source_ptr[si],
                            max_dist,
                        )
                    else:
                        hit_dist, _normal_local = _intersect_primitive_local(
                            shape_type,
                            ray_origin_local,
                            ray_dir_local,
                            shape_size,
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
