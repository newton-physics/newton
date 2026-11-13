# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional packed OptiX acceleration shared by shape queries and camera rendering."""

import functools
import weakref

import numpy as np
import warp as wp

from . import raycast
from .types import GeoType

# Raygen register cap: the occupancy gain outweighs the extra spills.
_MAX_REGISTERS = 80


class UnsupportedError(RuntimeError):
    """The OptiX backend cannot traverse this model on its device."""


@functools.cache
def import_warp_optix():
    """Return the optional ``warp_optix`` module, or ``None`` if it is not installed.

    Imported on first use because importing it registers OptiX with Warp.
    """
    try:
        import warp_optix  # noqa: PLC0415
    except ImportError:
        return None
    return warp_optix


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
def world_offset(world: int, grid: int, extent: float) -> wp.vec3:
    center = float(grid - 1) * 0.5
    cell = wp.vec3(float(world % grid), float((world // grid) % grid), float(world // (grid * grid)))
    return (cell - wp.vec3(center)) * (3.0 * extent)


@wp.func
def ray_interval(origin: wp.vec3, direction: wp.vec3, extent: float, max_distance: float) -> wp.vec2:
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


@wp.kernel(enable_backward=False)
def _mesh_types(types: wp.array[wp.int32]):
    i = wp.tid()
    if types[i] == GeoType.CONVEX_MESH or types[i] == GeoType.HFIELD:
        types[i] = GeoType.MESH


class Scene:
    """Own shared geometry and one spatially packed world instance structure."""

    def __init__(self, model, shape_types=None):
        wo = import_warp_optix()
        if wo is None:
            raise UnsupportedError("The OptiX backend requires the newton[optix] extra")
        if not model.device.is_cuda:
            raise UnsupportedError("The OptiX backend requires a CUDA device")
        try:
            wo.get_context(model.device)
        except Exception as exc:
            # OptiX reports unsupported GPUs and drivers with device-specific errors.
            raise UnsupportedError(f"OptiX is not supported on {model.device}: {exc}") from exc
        if model.shape_count >= 1 << 24:
            raise UnsupportedError("OptiX shape instance IDs must fit in 24 bits")
        if model._variant_shape_source:
            raise UnsupportedError("The OptiX backend does not support shape variants")
        if model.shape_count and model.bvh_shape_enabled is None:
            raise UnsupportedError("Build the model shape BVH before using the OptiX backend")
        self.wo = wo
        # Ray queries cache this scene on the model; do not create an ownership cycle.
        self.model = weakref.proxy(model)
        self.device = model.device
        self.shape_types = shape_types if shape_types is not None else wp.clone(model.shape_type)
        if shape_types is None:
            wp.launch(_mesh_types, dim=model.shape_count, inputs=[self.shape_types], device=self.device)
        types = self.shape_types.numpy() if model.shape_count else np.empty(0, dtype=np.int32)
        if np.any(types == GeoType.GAUSSIAN):
            raise UnsupportedError("The OptiX backend does not support Gaussian shapes")
        enabled = (
            model.bvh_shape_enabled.numpy()[: model.bvh_shape_count_enabled].astype(np.int32)
            if model.shape_count
            else np.empty(0, dtype=np.int32)
        )
        enabled = enabled[enabled >= 0]
        worlds = model.shape_world.numpy() if model.shape_count else np.empty(0, dtype=np.int32)
        sources = model.shape_source_ptr.numpy() if model.shape_count else np.empty(0, dtype=np.uint64)
        self.ray_pipelines = {}
        self.geometry = []
        mesh_handles = {}
        used_meshes = set(sources[enabled[types[enabled] == GeoType.MESH]].tolist())
        for mesh in [*model._mesh_keep_alive, *model.heightfield_meshes]:
            if mesh.id in used_meshes and mesh.id not in mesh_handles:
                accel = wo.Accel([wo.triangles(mesh.points, mesh.indices)], compact=True)
                self.geometry.append(accel)
                mesh_handles[mesh.id] = accel.id
        bounds = wp.array([[-1.001] * 3 + [1.001] * 3], dtype=float, device=model.device)
        primitive = wo.Accel([wo.aabbs(bounds=bounds)], compact=True)
        self.geometry.append(primitive)
        handles = np.zeros(model.shape_count, dtype=np.uint64)
        for si in enabled:
            if types[si] == GeoType.MESH:
                handles[si] = mesh_handles[int(sources[si])]
            elif types[si] != GeoType.PLANE:
                handles[si] = primitive.id
        self.traversables = wp.array(handles, dtype=wp.uint64, device=model.device)
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
        self.indices = wp.array(indices, dtype=wp.int32, device=model.device)
        self.instances = wp.empty(len(indices), dtype=wo.Instance, device=model.device)
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
            offset = world_offset(world, data.grid, data.extent[0])
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
            raise RuntimeError("The OptiX backend does not support CUDA graph capture")
        self._write_instances()
        if self.accel is not None:
            self.accel.refit()

    def pipeline(self, kernel, *, enable_backface_culling=False):
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
            offset = world_offset(world, bvh_shapes_id.grid, bvh_shapes_id.extent[0])
            distance, _normal = raycast.ray_intersect_shape_no_normal(
                shape_transforms[si],
                shape_sizes[si],
                shape_types[si],
                wp.optix_get_world_ray_origin() - offset,
                wp.optix_get_world_ray_direction(),
                wp.static(enable_backface_culling),
            )
            if distance >= wp.optix_get_ray_tmin():
                wp.optix_report_intersection(distance, wp.uint32(0), wp.uint32(0), wp.uint32(0))

        return wo.Pipeline(
            kernel,
            device=self.device,
            sbt=wo.SbtLayout(
                hit_groups={"triangles": wo.HitGroup(), "primitives": wo.HitGroup(intersection=intersect)}
            ),
            max_registers=_MAX_REGISTERS,
            module_options={"fast_math": kernel.module.options["fast_math"]},
            graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
        )


def intersect_rays(
    model,
    ray_origins,
    ray_directions,
    ray_worlds,
    *,
    enable_global_world,
    out_dist,
    out_shape_id,
    out_normal,
    fast_math,
    required,
) -> bool:
    """Launch batched shape queries using the model-owned packed acceleration.

    Returns ``False`` without launching when OptiX is not ``required`` and cannot be used,
    including during CUDA graph capture, so the caller falls back to Warp.
    """
    if model.device.is_cuda and model.device.stream.is_capturing:
        if required:
            raise RuntimeError("The OptiX backend does not support CUDA graph capture")
        return False
    if model._optix_scene is None and model._optix_unsupported is None:
        try:
            model._optix_scene = Scene(model)
        except UnsupportedError as exc:
            model._optix_unsupported = str(exc)
    if model._optix_unsupported is not None:
        if required:
            raise UnsupportedError(model._optix_unsupported)
        return False
    scene = model._optix_scene
    key = (enable_global_world, out_dist is not None, out_shape_id is not None, out_normal is not None, fast_math)
    if key not in scene.ray_pipelines:
        scene.ray_pipelines[key] = _create_ray_pipeline(scene, *key)
    scene.wo.launch(
        scene.ray_pipelines[key],
        dim=ray_origins.shape[0],
        inputs=[
            scene.data,
            model.bvh_shape_world_transforms,
            scene.shape_types,
            model.shape_scale,
            model.shape_source_ptr,
            ray_origins,
            ray_directions,
            ray_worlds,
        ],
        outputs=[out_dist, out_shape_id, out_normal],
    )
    return True


def _create_ray_pipeline(scene, enable_global_world, write_dist, write_shape_id, write_normal, fast_math):
    wo = scene.wo

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": fast_math})
    def intersect(
        bvh_shapes_id: SceneData,
        shape_transforms: wp.array[wp.transform],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3],
        shape_source_ptr: wp.array[wp.uint64],
        ray_origins: wp.array[wp.vec3],
        ray_directions: wp.array[wp.vec3],
        ray_worlds: wp.array[wp.int32],
        out_dist: wp.array[float],
        out_shape_id: wp.array[wp.int32],
        out_normal: wp.array[wp.vec3],
    ):
        tid = wp.tid()
        data = bvh_shapes_id
        origin = ray_origins[tid]
        direction = ray_directions[tid]
        distance = float(wp.static(raycast.MAXVAL))
        shape = int(-1)
        normal = wp.vec3(0.0)
        # Match the camera's unrolled local/global traversal (OptiX 0.3.0).
        for i in range(wp.static(2 if enable_global_world else 1)):
            world = wp.where(i == 1 or ray_worlds[tid] < 0, data.world_count, ray_worlds[tid])
            interval = ray_interval(origin, direction, data.extent[0], distance)
            packed_origin = origin + world_offset(world, data.grid, data.extent[0])
            if data.has_instances[world] and interval[1] >= interval[0]:
                query = wo.accel_query_ray(data.handle, packed_origin, direction, interval[1], interval[0])
                if query.result and query.t < distance:
                    shape = int(query.instance_id)
                    distance = query.t
                    if wp.static(write_normal):
                        if shape_types[shape] == GeoType.MESH:
                            local_normal = raycast.safe_div_vec3(
                                wp.mesh_eval_face_normal(shape_source_ptr[shape], int(query.primitive_index)),
                                shape_sizes[shape],
                            )
                            normal = wp.normalize(wp.transform_vector(shape_transforms[shape], local_normal))
                        else:
                            _t, normal = raycast.ray_intersect_shape(
                                shape_transforms[shape],
                                shape_sizes[shape],
                                shape_types[shape],
                                origin,
                                direction,
                                False,
                            )
            for p in range(data.plane_offsets[world], data.plane_offsets[world + 1]):
                si = data.planes[p]
                t, n = raycast.ray_intersect_shape(
                    shape_transforms[si],
                    shape_sizes[si],
                    shape_types[si],
                    origin,
                    direction,
                    False,
                )
                if t >= 0.0 and t < distance:
                    distance = t
                    shape = si
                    normal = n
        if wp.static(write_dist):
            out_dist[tid] = wp.where(shape < 0, -1.0, distance)
        if wp.static(write_shape_id):
            out_shape_id[tid] = shape
        if wp.static(write_normal):
            out_normal[tid] = normal

    return scene.pipeline(intersect)
