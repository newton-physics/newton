# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Camera shading adapters for the shared OptiX shape acceleration."""

import warp as wp

from ...geometry import GeoType, raycast
from ...geometry.optix import SceneData, ray_interval, world_offset


def create_closest_hit(config, scene):
    from ...geometry import Gaussian  # noqa: PLC0415
    from .raytrace import ClosestHit, _mesh_hit_normal  # noqa: PLC0415
    from .types import MeshData  # noqa: PLC0415

    wo = scene.wo

    @wp.func
    def closest_hit_shape(
        hit: ClosestHit,
        bvh_shapes_size: wp.int32,
        data: SceneData,
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
    ) -> ClosestHit:
        # Keep the two traversals unrolled: dynamic traversal loops miscompile with OptiX 0.3.0.
        for i in range(wp.static(2 if config.enable_global_world else 1)):
            world = wp.where(i == 1, data.world_count, world_index)
            interval = ray_interval(origin, direction, data.extent[0], hit.distance)
            packed_origin = origin + world_offset(world, data.grid, data.extent[0])
            if data.has_instances[world] and interval[1] >= interval[0]:
                flags = wp.uint32(0)
                if wp.static(config.enable_backface_culling):
                    flags = wp.uint32(wp.static(int(wo.RayFlags.CULL_BACK_FACING_TRIANGLES)))
                query = wo.accel_query_ray(data.handle, packed_origin, direction, interval[1], interval[0], 255, flags)
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
                            hit.normal = _mesh_hit_normal(
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
                            distance, normal = raycast.ray_intersect_shape(
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
            for p in range(data.plane_offsets[world], data.plane_offsets[world + 1]):
                si = data.planes[p]
                distance, normal = raycast.ray_intersect_shape(
                    shape_transforms[si],
                    shape_sizes[si],
                    shape_types[si],
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
        return hit

    return closest_hit_shape


def create_first_hit(config, scene):
    wo = scene.wo

    @wp.func
    def first_hit_shape(
        bvh_shapes_size: wp.int32,
        data: SceneData,
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
        # Keep the two traversals unrolled: dynamic traversal loops miscompile with OptiX 0.3.0.
        for i in range(wp.static(2 if config.enable_global_world else 1)):
            world = wp.where(i == 1, data.world_count, world_index)
            interval = ray_interval(origin, direction, data.extent[0], max_dist)
            packed_origin = origin + world_offset(world, data.grid, data.extent[0])
            if data.has_instances[world] and interval[1] >= interval[0]:
                if wo.accel_query_ray_anyhit(data.handle, packed_origin, direction, interval[1], interval[0], 255):
                    return True
            for p in range(data.plane_offsets[world], data.plane_offsets[world + 1]):
                si = data.planes[p]
                distance, _normal = raycast.ray_intersect_shape_no_normal(
                    shape_transforms[si],
                    shape_sizes[si],
                    shape_types[si],
                    origin,
                    direction,
                    wp.static(config.enable_backface_culling),
                )
                if distance >= 0.0 and distance < max_dist:
                    return True
        return False

    return first_hit_shape
