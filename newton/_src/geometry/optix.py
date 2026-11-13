# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional packed OptiX acceleration shared by shape queries and camera rendering."""

import weakref

import numpy as np
import warp as wp

from . import raycast
from .types import GeoType

try:
    import warp_optix as wo
except ImportError:
    wo = None

# Raygen register cap: the occupancy gain outweighs the extra spills.
_MAX_REGISTERS = 80


class UnsupportedError(RuntimeError):
    """OptiX cannot traverse this model on its device."""


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


@wp.func
def primitive_half_extents(kind: int, size: wp.vec3, extent: float) -> wp.vec3:
    """Half extents that scale the shared unit cube to an analytic shape's bounds."""
    if kind == GeoType.PLANE:
        # Infinite axes (size 0) span the cell, wherever it lies on the plane.
        span = 1.7321 * extent
        return wp.vec3(
            wp.where(size[0] > 0.0, 0.5 * size[0], span), wp.where(size[1] > 0.0, 0.5 * size[1], span), 1.0e-3
        )
    if kind == GeoType.SPHERE:
        return wp.vec3(size[0])
    if kind == GeoType.CAPSULE:
        return wp.vec3(size[0], size[0], size[0] + size[1])
    if kind == GeoType.CYLINDER or kind == GeoType.CONE:
        radius = size[0]
        if kind == GeoType.CYLINDER and size[2] > 0.0:
            radius += size[2] - wp.sqrt(wp.max(0.0, size[2] * size[2] - size[1] * size[1]))
        return wp.vec3(radius, radius, size[1])
    return size


class Scene:
    """Shared triangle geometry and one instance structure with a cell per world.

    The cell grid, world count and cell extent are fixed when the scene is built, so kernels
    compile them in. Shapes must stay within twice their initial distance from the origin, and
    infinite planes are bounded by the cell.
    """

    def __init__(self, model):
        if wo is None:
            raise UnsupportedError("The OptiX backend requires the newton[optix] extra")
        if not model.device.is_cuda:
            raise UnsupportedError("The OptiX backend requires a CUDA device")
        try:
            wo.get_context(model.device)
        except Exception as exc:
            raise UnsupportedError(f"OptiX is not supported on {model.device}: {exc}") from exc
        if model.shape_count >= 1 << 24:
            raise UnsupportedError("OptiX shape instance IDs must fit in 24 bits")
        if model._variant_shape_source:
            raise UnsupportedError("The OptiX backend does not support shape variants")
        if model.shape_count and model.bvh_shape_enabled is None:
            raise UnsupportedError("Build the model shape BVH before using the OptiX backend")
        # The model caches this scene; avoid an ownership cycle.
        self.model = weakref.proxy(model)
        self.device = model.device
        types = model.shape_type.numpy() if model.shape_count else np.empty(0, dtype=np.int32)
        # Convex hulls and heightfields are traced as their triangle meshes.
        types = np.where(np.isin(types, (GeoType.CONVEX_MESH, GeoType.HFIELD)), GeoType.MESH, types).astype(np.int32)
        self.shape_types = wp.array(types, dtype=wp.int32, device=self.device)
        if np.any(types == GeoType.GAUSSIAN):
            raise UnsupportedError("The OptiX backend does not support Gaussian shapes")
        leaves = (
            model.bvh_shape_enabled.numpy()[: model.bvh_shape_count_enabled].astype(np.int32)
            if model.shape_count
            else np.empty(0, dtype=np.int32)
        )
        enabled = leaves[leaves >= 0]
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
            is_mesh = types[si] == GeoType.MESH
            handles[si] = mesh_handles[int(sources[si])] if is_mesh else primitive.id
        self.traversables = wp.array(handles, dtype=wp.uint64, device=model.device)
        indices = enabled
        self.indices = wp.array(indices, dtype=wp.int32, device=model.device)

        self.world_count = model.world_count
        self.grid = int(np.ceil((model.world_count + 1) ** (1.0 / 3.0)))
        while self.grid**3 < model.world_count + 1:
            self.grid += 1
        bounded = (leaves >= 0) & (types[np.maximum(leaves, 0)] != GeoType.PLANE)
        radius = 0.0
        if np.any(bounded):
            lowers = model.bvh_shapes.lowers.numpy()[: len(leaves)][bounded]
            uppers = model.bvh_shapes.uppers.numpy()[: len(leaves)][bounded]
            radius = float(np.max(np.maximum(np.abs(lowers), np.abs(uppers))))
        # Room for shapes to move; the padding also encloses the primitives' padded unit cube.
        self.extent = max(2.0 * 1.01 * radius, 1.0)
        self.has_instances = len(indices) > 0

        grid, extent, world_count = self.grid, self.extent, self.world_count

        @wp.kernel(module="unique", enable_backward=False)
        def write_instances(
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
            world = wp.where(shape_world[si] < 0, wp.static(world_count), shape_world[si])
            transform = shape_transforms[si]
            rotation = wp.quat_to_matrix(wp.transform_get_rotation(transform))
            center = wp.transform_get_translation(transform)
            if kind == GeoType.PLANE:
                # Center infinite axes on the plane point nearest the cell center (the origin).
                nearest = -wp.transpose(rotation) @ center
                center += rotation @ wp.vec3(
                    wp.where(size[0] > 0.0, 0.0, nearest[0]), wp.where(size[1] > 0.0, 0.0, nearest[1]), 0.0
                )
            instances[i] = wo.make_instance(
                rotation @ wp.diag(primitive_half_extents(kind, size, wp.static(extent))),
                center + world_offset(world, wp.static(grid), wp.static(extent)),
                traversables[si],
                wp.uint32(si),
                wp.uint32(wp.where(kind == GeoType.MESH, 0, 1)),
                wp.uint32(255),
                wp.uint32(wp.static(int(wo.InstanceFlags.DISABLE_ANYHIT))),
            )

        self.write_instances = write_instances
        self.instances = wp.empty(len(indices), dtype=wo.Instance, device=model.device)
        self.accel = None
        self.handle = 0
        self._write_instances()
        if self.has_instances:
            self.accel = wo.Accel(
                [wo.instances(self.instances, self.geometry)],
                flags=wo.BuildFlags.ALLOW_UPDATE | wo.BuildFlags.PREFER_FAST_TRACE,
            )
            self.handle = self.accel.id

    def _write_instances(self):
        model = self.model
        wp.launch(
            self.write_instances,
            dim=len(self.indices),
            inputs=[
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
        extent = self.extent

        @wp.kernel(module="unique", enable_backward=False)
        def intersect(
            shape_types: wp.array[wp.int32],
            shape_sizes: wp.array[wp.vec3],
        ):
            # The instance scales the unit cube to the shape, so scaling the object-space ray by
            # the half extents gives the shape-local ray with the same distance.
            si = int(wp.optix_get_instance_id())
            kind = shape_types[si]
            size = shape_sizes[si]
            half = primitive_half_extents(kind, size, wp.static(extent))
            distance, _normal = raycast.ray_intersect_shape_local(
                size,
                kind,
                wp.cw_mul(wp.optix_get_object_ray_origin(), half),
                wp.cw_mul(wp.optix_get_object_ray_direction(), half),
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
    """Launch shape queries on the model's OptiX scene; return ``False`` to fall back to Warp."""
    if model.device.is_cuda and model.device.stream.is_capturing:
        if required:
            raise RuntimeError("The OptiX backend does not support CUDA graph capture")
        return False
    if model._optix_scene is None:
        try:
            model._optix_scene = Scene(model)
        except UnsupportedError:
            if required:
                raise
            return False
    scene = model._optix_scene
    # Refit here rather than in Model.bvh_refit_shapes, which may be graph captured.
    scene.refit()
    key = (enable_global_world, out_dist is not None, out_shape_id is not None, out_normal is not None, fast_math)
    if key not in scene.ray_pipelines:
        scene.ray_pipelines[key] = _create_ray_pipeline(scene, *key)
    wo.launch(
        scene.ray_pipelines[key],
        dim=ray_origins.shape[0],
        inputs=[
            scene.handle,
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
    grid, extent, world_count = scene.grid, scene.extent, scene.world_count

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": fast_math})
    def intersect(
        handle: wp.uint64,
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
        origin = ray_origins[tid]
        direction = ray_directions[tid]
        distance = float(wp.static(raycast.MAXVAL))
        shape = int(-1)
        normal = wp.vec3(0.0)
        # Match the camera's unrolled local/global traversal (OptiX 0.3.0).
        for i in range(wp.static(2 if enable_global_world else 1)):
            world = wp.where(i == 1 or ray_worlds[tid] < 0, wp.static(world_count), ray_worlds[tid])
            interval = ray_interval(origin, direction, wp.static(extent), distance)
            if wp.static(scene.has_instances) and interval[1] >= interval[0]:
                packed_origin = origin + world_offset(world, wp.static(grid), wp.static(extent))
                query = wo.accel_query_ray(handle, packed_origin, direction, interval[1], interval[0])
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
        if wp.static(write_dist):
            out_dist[tid] = wp.where(shape < 0, -1.0, distance)
        if wp.static(write_shape_id):
            out_shape_id[tid] = shape
        if wp.static(write_normal):
            out_normal[tid] = normal

    return scene.pipeline(intersect)
