# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..geometry import Gaussian, GeoType, raycast
from ..geometry.bvh import BVH_SHAPE_INACTIVE
from . import gaussians
from .types import MeshData, RenderConfig, RenderState, SceneData

if TYPE_CHECKING:
    from .render_optix import OptixScene


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
    # Mesh face index, or particle index for particle hits.
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

    return (
        t,
        _mesh_hit_normal(transform, scale, mesh_id, shape_mesh_data_id, mesh_data, normal_local, face, u, v),
        u,
        v,
        face,
    )


@wp.func
def _mesh_hit_normal(
    transform: wp.transformf,
    scale: wp.vec3f,
    mesh_id: wp.uint64,
    shape_mesh_data_id: wp.int32,
    mesh_data: wp.array[MeshData],
    normal_local: wp.vec3f,
    face: wp.int32,
    u: wp.float32,
    v: wp.float32,
) -> wp.vec3f:
    if shape_mesh_data_id > -1:
        normals = mesh_data[shape_mesh_data_id].normals
        if normals.shape[0] > 0:
            n0 = wp.mesh_get_index(mesh_id, face * 3 + 0)
            n1 = wp.mesh_get_index(mesh_id, face * 3 + 1)
            n2 = wp.mesh_get_index(mesh_id, face * 3 + 2)
            vertex_normal = normals[n0] * u + normals[n1] * v + normals[n2] * (1.0 - u - v)
            normal_local = raycast.safe_div_vec3(vertex_normal, scale)

    return wp.normalize(wp.transform_vector(transform, normal_local))


@wp.func
def ray_intersect_triangle(
    ray_origin: wp.vec3f,
    ray_dir: wp.vec3f,
    p0: wp.vec3f,
    p1: wp.vec3f,
    p2: wp.vec3f,
    enable_backface_culling: wp.bool,
) -> tuple[wp.float32, wp.float32, wp.float32, wp.vec3f]:
    """Ray-triangle intersection (Moller-Trumbore).

    Returns ``(t, u, v, normal)`` with Warp's mesh barycentrics, ``hit = u * p0 + v * p1 + (1 - u - v) * p2``,
    and the face normal oriented toward the ray origin; ``t`` is -1 on a miss. With
    ``enable_backface_culling``, faces whose winding faces away from the ray are missed.
    """
    e1 = p1 - p0
    e2 = p2 - p0
    pvec = wp.cross(ray_dir, e2)
    det = wp.dot(e1, pvec)
    if det == 0.0 or (enable_backface_culling and det < 0.0):
        return -1.0, 0.0, 0.0, wp.vec3f(0.0)
    inv_det = 1.0 / det
    tvec = ray_origin - p0
    b1 = wp.dot(tvec, pvec) * inv_det
    qvec = wp.cross(tvec, e1)
    b2 = wp.dot(ray_dir, qvec) * inv_det
    # A small tolerance keeps rays from slipping through shared edges.
    eps = 1.0e-7
    if b1 < -eps or b2 < -eps or b1 + b2 > 1.0 + eps:
        return -1.0, 0.0, 0.0, wp.vec3f(0.0)
    t = wp.dot(e2, qvec) * inv_det
    if t < 0.0:
        return -1.0, 0.0, 0.0, wp.vec3f(0.0)
    normal = wp.normalize(wp.cross(e1, e2))
    if det < 0.0:
        normal = -normal
    return t, 1.0 - b1 - b2, b1, normal


@wp.func
def triangle_face_normal(scene: SceneData, face: wp.int32, ray_dir: wp.vec3f) -> wp.vec3f:
    """Face normal of a deformable triangle, oriented toward the ray origin."""
    p0 = scene.triangle_points[scene.triangle_indices[face * 3 + 0]]
    p1 = scene.triangle_points[scene.triangle_indices[face * 3 + 1]]
    p2 = scene.triangle_points[scene.triangle_indices[face * 3 + 2]]
    normal = wp.normalize(wp.cross(p1 - p0, p2 - p0))
    if wp.dot(normal, ray_dir) > 0.0:
        normal = -normal
    return normal


@wp.func
def get_group_roots(group_roots: wp.array[wp.int32], world_index: wp.int32, want_global_world: wp.int32) -> wp.int32:
    if want_global_world != 0:
        return group_roots[group_roots.shape[0] - 1]
    return group_roots[world_index]


def create_closest_hit_function(
    config: RenderConfig, state: RenderState, with_normal: bool = True, optix: OptixScene | None = None
) -> wp.Function:
    """Closest hit of a camera ray in the scene; without ``with_normal``, only the distance, item and face are found.

    One traversal visits every scene item: shapes, standalone particles and deformable triangles.
    """
    if optix is not None:
        return optix.create_closest_hit(config, state, with_normal)

    shade_gaussians = gaussians.create_shade_function(config, state)
    has_particles = config.enable_particles and state.has_particles

    @wp.func
    def closest_hit(
        scene: SceneData,
        world_index: wp.int32,
        max_distance: wp.float32,
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
        closest_hit = ClosestHit()
        closest_hit.distance = max_distance
        closest_hit.shape_index = NO_HIT_SHAPE_ID
        closest_hit.color = wp.vec3f(0.0)

        if scene.id:
            # Gaussian shapes crossed in the world and the global world, blended together afterwards.
            gaussians_hit = wp.vector(length=wp.static(max(state.gaussian_shapes_per_ray, 1)), dtype=wp.int32)
            num_gaussians_hit = wp.int32(0)
            # int() makes the bound dynamic in Warp, preventing loop unrolling
            # that duplicates BVH traversal state and increases register pressure.
            for i in range(int(wp.static(2 if config.enable_global_world else 1))):
                group_root = get_group_roots(scene.world_roots, world_index, i)
                if group_root < 0:
                    continue

                query = wp.bvh_query_ray(scene.id, ray_origin_world, ray_dir_world, group_root)
                item = wp.int32(0)

                while wp.bvh_query_next(query, item, closest_hit.distance):
                    hit_distance = wp.float32(-1.0)
                    hit_normal = wp.vec3f(0.0)
                    hit_u = wp.float32(0.0)
                    hit_v = wp.float32(0.0)
                    hit_face_id = wp.int32(-1)
                    hit_id = NO_HIT_SHAPE_ID

                    if item < scene.shape_item_count:
                        si = scene.item_shapes[item]
                        if si != BVH_SHAPE_INACTIVE:
                            hit_id = si
                            shape_type = shape_types[si]
                            # Heightfields are triangulated meshes; RenderContext remaps
                            # HFIELD -> MESH, so this branch renders them too.
                            if shape_type == GeoType.MESH:
                                if wp.static(with_normal):
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
                                else:
                                    ray_origin_local, ray_direction_local = raycast.map_ray_to_local(
                                        shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si]
                                    )
                                    hit_distance, _normal, _u, _v, _face = raycast.ray_intersect_mesh_no_normal(
                                        ray_origin_local,
                                        ray_direction_local,
                                        shape_sizes[si],
                                        shape_source_ptr[si],
                                        wp.static(config.enable_backface_culling),
                                        closest_hit.distance,
                                    )
                            elif wp.static(state.gaussian_shapes_per_ray > 0) and shape_type == GeoType.GAUSSIAN:
                                if num_gaussians_hit < wp.static(state.gaussian_shapes_per_ray):
                                    gaussians_hit[num_gaussians_hit] = wp.int32(si)
                                    num_gaussians_hit += 1
                            elif wp.static(with_normal):
                                hit_distance, hit_normal = raycast.ray_intersect_shape(
                                    shape_transforms[si],
                                    shape_sizes[si],
                                    shape_type,
                                    ray_origin_world,
                                    ray_dir_world,
                                    wp.static(config.enable_backface_culling),
                                )
                            else:
                                hit_distance, _normal = raycast.ray_intersect_shape_no_normal(
                                    shape_transforms[si],
                                    shape_sizes[si],
                                    shape_type,
                                    ray_origin_world,
                                    ray_dir_world,
                                    wp.static(config.enable_backface_culling),
                                )
                    elif wp.static(state.has_particles) and item < scene.shape_item_count + scene.particle_item_count:
                        if wp.static(has_particles):
                            particle = scene.item_particles[item - scene.shape_item_count]
                            hit_distance, hit_normal = raycast.ray_intersect_particle_sphere(
                                ray_origin_world,
                                ray_dir_world,
                                scene.particle_positions[particle],
                                scene.particle_radii[particle],
                            )
                            hit_id = PARTICLES_SHAPE_ID
                            hit_face_id = particle
                    elif wp.static(state.has_triangle_mesh):
                        face = item - scene.shape_item_count - scene.particle_item_count
                        hit_distance, hit_u, hit_v, hit_normal = ray_intersect_triangle(
                            ray_origin_world,
                            ray_dir_world,
                            scene.triangle_points[scene.triangle_indices[face * 3 + 0]],
                            scene.triangle_points[scene.triangle_indices[face * 3 + 1]],
                            scene.triangle_points[scene.triangle_indices[face * 3 + 2]],
                            wp.static(config.enable_backface_culling),
                        )
                        hit_id = TRIANGLE_MESH_SHAPE_ID
                        # Albedo-only renders also need the face when textures are disabled.
                        hit_face_id = face

                    if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                        closest_hit.distance = hit_distance
                        closest_hit.shape_index = hit_id
                        closest_hit.face_idx = hit_face_id
                        if wp.static(with_normal):
                            closest_hit.normal = hit_normal
                            closest_hit.bary_u = hit_u
                            closest_hit.bary_v = hit_v
                            closest_hit.color = wp.vec3f(0.0)

            # Splats are blended after the traversals: nested Warp BVH queries share
            # stack data, and this also skips splats hidden behind other items.
            if wp.static(state.gaussian_shapes_per_ray > 0) and num_gaussians_hit > 0:
                hit_distance, hit_color, gaussian_shape = shade_gaussians(
                    gaussians_hit,
                    num_gaussians_hit,
                    shape_transforms,
                    shape_sizes,
                    shape_source_ptr,
                    gaussians_data,
                    ray_origin_world,
                    ray_dir_world,
                    camera_forward,
                    closest_hit.distance,
                )
                if hit_distance >= 0.0 and hit_distance < closest_hit.distance:
                    closest_hit.distance = hit_distance
                    closest_hit.shape_index = wp.uint32(gaussian_shape)
                    if wp.static(with_normal):
                        closest_hit.normal = wp.vec3f(0.0)
                        closest_hit.color = hit_color

        return closest_hit

    return closest_hit


def create_first_hit_function(config: RenderConfig, state: RenderState, optix: OptixScene | None = None) -> wp.Function:
    """Whether anything in the scene blocks a shadow ray within ``max_distance``."""
    if optix is not None:
        return optix.create_first_hit(config, state)

    has_particles = config.enable_particles and state.has_particles

    @wp.func
    def first_hit(
        scene: SceneData,
        world_index: wp.int32,
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        ray_origin_world: wp.vec3f,
        ray_dir_world: wp.vec3f,
        max_distance: wp.float32,
    ) -> wp.bool:
        if scene.id:
            for i in range(int(wp.static(2 if config.enable_global_world else 1))):
                group_root = get_group_roots(scene.world_roots, world_index, i)
                if group_root < 0:
                    continue

                query = wp.bvh_query_ray(scene.id, ray_origin_world, ray_dir_world, group_root)
                item = wp.int32(0)

                while wp.bvh_query_next(query, item, max_distance):
                    hit_dist = wp.float32(-1.0)
                    if item < scene.shape_item_count:
                        si = scene.item_shapes[item]
                        if si != BVH_SHAPE_INACTIVE:
                            shape_type = shape_types[si]
                            # Heightfields are triangulated meshes; RenderContext remaps
                            # HFIELD -> MESH, so this branch renders them too.
                            if shape_type == GeoType.MESH:
                                ray_origin_local, ray_direction_local = raycast.map_ray_to_local(
                                    shape_transforms[si], ray_origin_world, ray_dir_world, shape_sizes[si]
                                )
                                hit_dist = raycast.ray_intersect_mesh_anyhit(
                                    ray_origin_local,
                                    ray_direction_local,
                                    shape_source_ptr[si],
                                    max_distance,
                                )
                            else:
                                hit_dist, _normal = raycast.ray_intersect_shape_no_normal(
                                    shape_transforms[si],
                                    shape_sizes[si],
                                    shape_type,
                                    ray_origin_world,
                                    ray_dir_world,
                                    wp.static(config.enable_backface_culling),
                                )
                    elif wp.static(state.has_particles) and item < scene.shape_item_count + scene.particle_item_count:
                        if wp.static(has_particles):
                            particle = scene.item_particles[item - scene.shape_item_count]
                            hit_dist, _normal = raycast.ray_intersect_particle_sphere(
                                ray_origin_world,
                                ray_dir_world,
                                scene.particle_positions[particle],
                                scene.particle_radii[particle],
                            )
                    elif wp.static(state.has_triangle_mesh):
                        face = item - scene.shape_item_count - scene.particle_item_count
                        # Shadows hit both sides of deformable surfaces.
                        hit_dist, _u, _v, _normal = ray_intersect_triangle(
                            ray_origin_world,
                            ray_dir_world,
                            scene.triangle_points[scene.triangle_indices[face * 3 + 0]],
                            scene.triangle_points[scene.triangle_indices[face * 3 + 1]],
                            scene.triangle_points[scene.triangle_indices[face * 3 + 2]],
                            False,
                        )
                    if hit_dist >= 0.0 and hit_dist < max_distance:
                        return True

        return False

    return first_hit
