# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional OptiX shape traversal for the camera renderer."""

import numpy as np
import warp as wp

from ...geometry import GeoType, raycast


@wp.struct
class SceneData:
    handle: wp.uint64
    world_count: wp.int32
    grid: wp.int32
    extent: wp.array[wp.float32]
    shape_world: wp.array[wp.int32]
    has_instances: wp.array[wp.bool]
    planes: wp.array[wp.int32]
    plane_offsets: wp.array[wp.int32]


@wp.func
def _world_offset(world: int, grid: int, extent: float) -> wp.vec3:
    center = float(grid - 1) * 0.5
    cell = wp.vec3(float(world % grid), float((world // grid) % grid), float(world // (grid * grid)))
    return (cell - wp.vec3(center)) * (3.0 * extent)


@wp.func
def _ray_interval(origin: wp.vec3, direction: wp.vec3, extent: float, max_distance: float) -> wp.vec2:
    """Clip traversal to one cell, including rays whose origins lie outside it."""
    near = float(0.0)
    far = max_distance
    for axis in range(3):
        if direction[axis] != 0.0:
            a = (-extent - origin[axis]) / direction[axis]
            b = (extent - origin[axis]) / direction[axis]
            near = wp.max(near, wp.min(a, b))
            far = wp.min(far, wp.max(a, b))
        elif wp.abs(origin[axis]) > extent:
            far = -1.0
    return wp.vec2(near, far)


@wp.kernel(enable_backward=False)
def _grow_extent(
    enabled: wp.array[wp.uint32],
    shape_types: wp.array[wp.int32],
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    extent: wp.array[float],
):
    leaf = wp.tid()
    if shape_types[enabled[leaf]] != GeoType.PLANE:
        bounds = wp.max(wp.abs(lowers[leaf]), wp.abs(uppers[leaf]))
        radius = wp.max(bounds[0], wp.max(bounds[1], bounds[2]))
        # Padding also encloses the analytic primitives' padded unit cube.
        wp.atomic_max(extent, 0, radius * 1.01)


class Scene:
    """Own shared geometry and one spatially packed world instance structure."""

    def __init__(self, context):
        if not context.device.is_cuda:
            raise ValueError("The OptiX camera backend requires a CUDA device")
        try:
            import warp_optix as wo  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("The OptiX camera backend requires the newton[optix] extra") from exc
        self.wo = wo
        self.model = context.model
        self.device = context.device
        self.shape_types = context._get_shape_render_type()
        model = context.model
        if model.shape_count >= 1 << 24:
            raise ValueError("OptiX shape instance IDs must fit in 24 bits")
        types = context._get_shape_render_type().numpy() if model.shape_count else np.empty(0, dtype=np.int32)
        if context._has_particles or context._has_triangle_mesh:
            raise NotImplementedError("The OptiX camera backend does not support particles or deformable surfaces")
        if np.any(types == GeoType.GAUSSIAN):
            raise NotImplementedError("The OptiX camera backend does not support Gaussian shapes")
        if model._variant_shape_source:
            raise NotImplementedError("The OptiX camera backend does not support shape variants")
        enabled = (
            model.bvh_shape_enabled.numpy()[: model.bvh_shape_count_enabled].astype(np.int32)
            if model.shape_count
            else np.empty(0, dtype=np.int32)
        )
        enabled = enabled[enabled >= 0]
        worlds = model.shape_world.numpy() if model.shape_count else np.empty(0, dtype=np.int32)
        sources = model.shape_source_ptr.numpy() if model.shape_count else np.empty(0, dtype=np.uint64)
        self.geometry = []
        mesh_handles = {}
        used_meshes = set(sources[enabled[types[enabled] == GeoType.MESH]].tolist())
        for mesh in [*model._mesh_keep_alive, *model.heightfield_meshes]:
            if mesh.id in used_meshes and mesh.id not in mesh_handles:
                accel = wo.Accel([wo.triangles(mesh.points, mesh.indices)], compact=True)
                self.geometry.append(accel)
                mesh_handles[mesh.id] = accel.id
        bounds = wp.array([[-1.001] * 3 + [1.001] * 3], dtype=float, device=context.device)
        primitive = wo.Accel([wo.aabbs(bounds=bounds)], compact=True)
        self.geometry.append(primitive)
        handles = np.zeros(model.shape_count, dtype=np.uint64)
        for si in enabled:
            if types[si] == GeoType.MESH:
                handles[si] = mesh_handles[int(sources[si])]
            elif types[si] != GeoType.PLANE:
                handles[si] = primitive.id
        self.traversables = wp.array(handles, dtype=wp.uint64, device=context.device)
        indices = []
        planes = []
        offsets = [0]
        has_instances = []
        world_ids = np.where(worlds[enabled] < 0, model.world_count, worlds[enabled])
        order = np.argsort(world_ids, kind="stable")
        grouped = enabled[order]
        boundaries = np.searchsorted(world_ids[order], np.arange(model.world_count + 2))
        for world in range(model.world_count + 1):
            selected = grouped[boundaries[world] : boundaries[world + 1]]
            bounded = selected[types[selected] != GeoType.PLANE]
            has_instances.append(len(bounded) > 0)
            indices.extend(bounded)
            planes.extend(selected[types[selected] == GeoType.PLANE])
            offsets.append(len(planes))
        self.indices = wp.array(indices, dtype=wp.int32, device=context.device)
        self.instances = wp.empty(len(indices), dtype=wo.Instance, device=context.device)
        self.data = SceneData()
        self.data.world_count = model.world_count
        self.data.shape_world = model.shape_world
        self.data.grid = int(np.ceil((model.world_count + 1) ** (1.0 / 3.0)))
        while self.data.grid**3 < model.world_count + 1:
            self.data.grid += 1
        self.data.extent = wp.ones(1, dtype=float, device=self.device)
        self.data.has_instances = wp.array(has_instances, dtype=wp.bool, device=self.device)
        self.data.planes = wp.array(planes, dtype=wp.int32, device=self.device)
        self.data.plane_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)

        @wp.kernel(module="unique", enable_backward=False)
        def write_instances(
            data: SceneData,
            indices: wp.array[wp.int32],
            shape_world: wp.array[wp.int32],
            shape_types: wp.array[wp.int32],
            shape_sizes: wp.array[wp.vec3],
            shape_transforms: wp.array[wp.transform],
            traversables: wp.array[wp.uint64],
            instances: wp.array[wo.Instance],
        ):
            i = wp.tid()
            si = indices[i]
            kind = shape_types[si]
            size = shape_sizes[si]
            half = size
            if kind == GeoType.SPHERE:
                half = wp.vec3(size[0])
            elif kind == GeoType.CAPSULE:
                half = wp.vec3(size[0], size[0], size[0] + size[1])
            elif kind == GeoType.CYLINDER or kind == GeoType.CONE:
                radius = size[0]
                if kind == GeoType.CYLINDER and size[2] > 0.0:
                    radius += size[2] - wp.sqrt(wp.max(0.0, size[2] * size[2] - size[1] * size[1]))
                half = wp.vec3(radius, radius, size[1])
            world = wp.where(shape_world[si] < 0, data.world_count, shape_world[si])
            offset = _world_offset(world, data.grid, data.extent[0])
            transform = shape_transforms[si]
            linear = wp.quat_to_matrix(wp.transform_get_rotation(transform)) @ wp.diag(half)
            sbt = wp.where(kind == GeoType.MESH, 0, 1)
            instances[i] = wo.make_instance(
                linear,
                wp.transform_get_translation(transform) + offset,
                traversables[si],
                wp.uint32(si),
                wp.uint32(sbt),
                wp.uint32(255),
                wp.uint32(wp.static(int(wo.InstanceFlags.DISABLE_ANYHIT))),
            )

        self.write_instances = write_instances
        self.accel = None
        self._write_instances()
        if indices:
            self.accel = wo.Accel(
                [wo.instances(self.instances, self.geometry)],
                flags=wo.BuildFlags.ALLOW_UPDATE | wo.BuildFlags.PREFER_FAST_TRACE,
            )
            self.data.handle = self.accel.id

    def _write_instances(self):
        model = self.model
        if model.bvh_shape_count_enabled:
            wp.launch(
                _grow_extent,
                dim=model.bvh_shape_count_enabled,
                inputs=[
                    model.bvh_shape_enabled,
                    self.shape_types,
                    model.bvh_shapes.lowers,
                    model.bvh_shapes.uppers,
                    self.data.extent,
                ],
                device=self.device,
            )
        wp.launch(
            self.write_instances,
            dim=len(self.indices),
            inputs=[
                self.data,
                self.indices,
                model.shape_world,
                self.shape_types,
                model.shape_scale,
                model.bvh_shape_world_transforms,
                self.traversables,
                self.instances,
            ],
            device=self.device,
        )

    def refit(self):
        if self.device.stream.is_capturing:
            raise RuntimeError("The OptiX camera backend does not support CUDA graph capture")
        self._write_instances()
        if self.accel is not None:
            self.accel.refit()

    def pipeline(self, kernel, config):
        wo = self.wo

        @wp.kernel(module="unique", enable_backward=False)
        def intersect(
            bvh_shapes_id: SceneData,
            shape_transforms: wp.array[wp.transform],
            shape_types: wp.array[wp.int32],
            shape_sizes: wp.array[wp.vec3],
        ):
            si = int(wp.optix_get_instance_id())
            world = wp.where(
                bvh_shapes_id.shape_world[si] < 0, bvh_shapes_id.world_count, bvh_shapes_id.shape_world[si]
            )
            offset = _world_offset(world, bvh_shapes_id.grid, bvh_shapes_id.extent[0])
            distance, _normal = raycast.ray_intersect_shape_no_normal(
                shape_transforms[si],
                shape_sizes[si],
                shape_types[si],
                wp.optix_get_world_ray_origin() - offset,
                wp.optix_get_world_ray_direction(),
                wp.static(config.enable_backface_culling),
            )
            if distance >= wp.optix_get_ray_tmin():
                wp.optix_report_intersection(distance, wp.uint32(0), wp.uint32(0), wp.uint32(0))

        return wo.Pipeline(
            kernel,
            device=self.device,
            sbt=wo.SbtLayout(
                hit_groups={"triangles": wo.HitGroup(), "primitives": wo.HitGroup(intersection=intersect)}
            ),
            module_options={"fast_math": kernel.module.options["fast_math"]},
            graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
        )


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
            interval = _ray_interval(origin, direction, data.extent[0], hit.distance)
            packed_origin = origin + _world_offset(world, data.grid, data.extent[0])
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
            interval = _ray_interval(origin, direction, data.extent[0], max_dist)
            packed_origin = origin + _world_offset(world, data.grid, data.extent[0])
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
