# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Camera traversal of :attr:`Model.optix_shapes`, with the Warp traversal signatures.

``bvh_shapes_id`` is the OptiX instance structure; the Warp BVH arguments are unused.
"""

import warp as wp

from ...geometry import Gaussian, GeoType, raycast
from . import raytrace
from .types import MeshData


def create_closest_hit(config, shapes):
    from ...geometry.optix import ray_interval, wo, world_offset  # noqa: PLC0415

    grid, extent, world_count = shapes.grid, shapes.extent, shapes.world_count

    @wp.func
    def closest_hit_shape(
        hit: raytrace.ClosestHit,
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
        origin: wp.vec3f,
        direction: wp.vec3f,
        camera_forward: wp.vec3f,
    ) -> raytrace.ClosestHit:
        # Global planes first: a nearer hit shortens the traversals.
        if wp.static(config.enable_global_world):
            for p in range(wp.static(len(shapes.planes))):
                si = wp.static(shapes.planes[p])
                distance, normal = raycast.ray_intersect_shape(
                    shape_transforms[si],
                    shape_sizes[si],
                    GeoType.PLANE,
                    origin,
                    direction,
                    wp.static(config.enable_backface_culling),
                )
                if distance >= 0.0 and distance < hit.distance:
                    hit.distance = distance
                    hit.normal = normal
                    hit.shape_index = wp.uint32(si)
                    hit.bary_u = 0.0
                    hit.bary_v = 0.0
                    hit.face_idx = -1
        # Unrolled: dynamic traversal loops miscompile with warp-optix 0.3.0.
        for i in range(wp.static(2 if config.enable_global_world and shapes.has_global_instances else 1)):
            world = wp.where(i == 1, wp.static(world_count), world_index)
            interval = ray_interval(origin, direction, wp.static(extent), hit.distance)
            if wp.static(shapes.has_instances) and interval[1] >= interval[0]:
                packed_origin = origin + world_offset(world, wp.static(grid), wp.static(extent))
                flags = wp.uint32(0)
                if wp.static(config.enable_backface_culling):
                    flags = wp.uint32(wp.static(int(wo.RayFlags.CULL_BACK_FACING_TRIANGLES)))
                query = wo.accel_query_ray(
                    bvh_shapes_id, packed_origin, direction, interval[1], interval[0], 255, flags
                )
                if query.result:
                    if query.t < hit.distance:
                        si = int(query.instance_id)
                        hit.distance = query.t
                        hit.shape_index = wp.uint32(si)
                        if shape_types[si] == GeoType.MESH:
                            mesh = shape_source_ptr[si]
                            face = int(query.primitive_index)
                            u = 1.0 - query.u - query.v
                            v = query.u
                            normal = wp.normalize(
                                raycast.safe_div_vec3(wp.mesh_eval_face_normal(mesh, face), shape_sizes[si])
                            )
                            hit.normal = raytrace._mesh_hit_normal(
                                shape_transforms[si],
                                shape_sizes[si],
                                mesh,
                                shape_mesh_data_ids[si],
                                mesh_data,
                                normal,
                                face,
                                u,
                                v,
                            )
                            hit.bary_u = u
                            hit.bary_v = v
                            hit.face_idx = face
                        else:
                            _distance, normal = raycast.ray_intersect_shape(
                                shape_transforms[si],
                                shape_sizes[si],
                                shape_types[si],
                                origin,
                                direction,
                                wp.static(config.enable_backface_culling),
                            )
                            hit.normal = normal
                            hit.bary_u = 0.0
                            hit.bary_v = 0.0
                            hit.face_idx = -1
        return hit

    return closest_hit_shape


def create_first_hit(config, shapes):
    from ...geometry.optix import ray_interval, wo, world_offset  # noqa: PLC0415

    grid, extent, world_count = shapes.grid, shapes.extent, shapes.world_count

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
        origin: wp.vec3f,
        direction: wp.vec3f,
        max_dist: wp.float32,
    ) -> wp.bool:
        if wp.static(config.enable_global_world):
            for p in range(wp.static(len(shapes.planes))):
                si = wp.static(shapes.planes[p])
                distance, _normal = raycast.ray_intersect_shape_no_normal(
                    shape_transforms[si],
                    shape_sizes[si],
                    GeoType.PLANE,
                    origin,
                    direction,
                    wp.static(config.enable_backface_culling),
                )
                if distance >= 0.0 and distance < max_dist:
                    return True
        # Unrolled: dynamic traversal loops miscompile with warp-optix 0.3.0.
        for i in range(wp.static(2 if config.enable_global_world and shapes.has_global_instances else 1)):
            world = wp.where(i == 1, wp.static(world_count), world_index)
            interval = ray_interval(origin, direction, wp.static(extent), max_dist)
            if wp.static(shapes.has_instances) and interval[1] >= interval[0]:
                packed_origin = origin + world_offset(world, wp.static(grid), wp.static(extent))
                if wo.accel_query_ray_anyhit(bvh_shapes_id, packed_origin, direction, interval[1], interval[0], 255):
                    return True
        return False

    return first_hit_shape
