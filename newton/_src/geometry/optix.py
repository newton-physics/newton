# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OptiX shape acceleration, built alongside the shape BVH.

Used by :meth:`Model.optix_build_shapes` and :meth:`Model.optix_refit_shapes`. Importing this module
requires the ``newton[optix]`` extra, so import it lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
import warp_optix as wo

from . import raycast
from .types import GeoType

if TYPE_CHECKING:
    from ..sim import Model

# Raygen register cap: the occupancy gain outweighs the extra spills.
_MAX_REGISTERS = 80


class UnsupportedError(RuntimeError):
    """OptiX cannot trace this model on its device."""


@dataclass(eq=False)
class OptixShapes:
    """OptiX instance acceleration over a model's shapes, with one cell per world.

    The cell grid, world count and cell extent are fixed when the shapes are built, so kernels
    compile them in. Shapes must stay within twice their initial distance from the origin, and
    per-world infinite planes are bounded by the cell. Global planes (usually the ground) are not
    instanced; kernels intersect them analytically.
    """

    id: int
    """Traversable handle of :attr:`accel`, or 0 without instances."""
    accel: wo.Accel | None
    """Instance acceleration structure over all worlds."""
    geometry: list[wo.Accel]
    """Triangle and primitive acceleration structures shared by the instances."""
    shape_types: wp.array[wp.int32]
    """Shape types, with convex hulls and heightfields traced as meshes, shape ``[shape_count]``."""
    shape_world_transforms: wp.array[wp.transformf]
    """World-space shape transforms, shape ``[shape_count]``."""
    shape_traversables: wp.array[wp.uint64]
    """Acceleration structure instanced by each shape, shape ``[shape_count]``."""
    instance_shapes: wp.array[wp.int32]
    """Shape index of each instance."""
    instances: wp.array[wo.Instance]
    planes: tuple[int, ...]
    """Global planes, intersected analytically."""
    grid: int
    extent: float
    world_count: int
    has_global_instances: bool
    ray_pipelines: dict = field(default_factory=dict)

    @property
    def has_instances(self) -> bool:
        return self.instance_shapes.shape[0] > 0


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


@wp.kernel(enable_backward=False)
def compute_optix_shape_instances(
    grid: wp.int32,
    extent: wp.float32,
    world_count: wp.int32,
    instance_shapes: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_types: wp.array[wp.int32],
    shape_sizes: wp.array[wp.vec3],
    shape_transforms: wp.array[wp.transformf],
    shape_traversables: wp.array[wp.uint64],
    out_instances: wp.array[wo.Instance],
):
    i = wp.tid()
    si = instance_shapes[i]
    kind = shape_types[si]
    size = shape_sizes[si]
    world = wp.where(shape_world[si] < 0, world_count, shape_world[si])
    transform = shape_transforms[si]
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(transform))
    center = wp.transform_get_translation(transform)
    if kind == GeoType.PLANE:
        # Center infinite axes on the plane point nearest the cell center (the origin).
        nearest = -wp.transpose(rotation) @ center
        center += rotation @ wp.vec3(
            wp.where(size[0] > 0.0, 0.0, nearest[0]), wp.where(size[1] > 0.0, 0.0, nearest[1]), 0.0
        )
    out_instances[i] = wo.make_instance(
        rotation @ wp.diag(primitive_half_extents(kind, size, extent)),
        center + world_offset(world, grid, extent),
        shape_traversables[si],
        wp.uint32(si),
        wp.uint32(wp.where(kind == GeoType.MESH, 0, 1)),
        wp.uint32(255),
        wp.uint32(wp.static(int(wo.InstanceFlags.DISABLE_ANYHIT))),
    )


def check_supported(model: Model) -> None:
    """Raise :class:`UnsupportedError` if OptiX cannot trace the model's shapes on its device."""
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
    if model.shape_count and np.any(model.shape_type.numpy() == GeoType.GAUSSIAN):
        raise UnsupportedError("The OptiX backend does not support Gaussian shapes")


def build_shape_geometry(
    model: Model, shapes: np.ndarray, shape_types: np.ndarray
) -> tuple[list[wo.Accel], wp.array[wp.uint64]]:
    """Build the triangle structures of the meshes used by ``shapes`` and the primitives' unit cube."""
    sources = model.shape_source_ptr.numpy()
    geometry = []
    mesh_ids = {}
    used_meshes = set(sources[shapes[shape_types[shapes] == GeoType.MESH]].tolist())
    for mesh in [*model._mesh_keep_alive, *model.heightfield_meshes]:
        if mesh.id in used_meshes and mesh.id not in mesh_ids:
            geometry.append(wo.Accel([wo.triangles(mesh.points, mesh.indices)], compact=True))
            mesh_ids[mesh.id] = geometry[-1].id
    bounds = wp.array([[-1.001] * 3 + [1.001] * 3], dtype=float, device=model.device)
    geometry.append(wo.Accel([wo.aabbs(bounds=bounds)], compact=True))
    traversables = np.zeros(model.shape_count, dtype=np.uint64)
    for si in shapes:
        traversables[si] = mesh_ids[int(sources[si])] if shape_types[si] == GeoType.MESH else geometry[-1].id
    return geometry, wp.array(traversables, dtype=wp.uint64, device=model.device)


def compute_optix_shape_instances_launch(model: Model, shapes: OptixShapes) -> None:
    """Write the instance of each shape in ``shapes`` from its current world transform."""
    wp.launch(
        kernel=compute_optix_shape_instances,
        dim=shapes.instance_shapes.shape[0],
        inputs=[
            shapes.grid,
            shapes.extent,
            shapes.world_count,
            shapes.instance_shapes,
            model.shape_world,
            shapes.shape_types,
            model.shape_scale,
            shapes.shape_world_transforms,
            shapes.shape_traversables,
            shapes.instances,
        ],
        device=model.device,
    )


def create_pipeline(shapes: OptixShapes, kernel: wp.Kernel, *, enable_backface_culling: bool = False) -> wo.Pipeline:
    """Compile ``kernel`` as the ray generation program of a pipeline over ``shapes``."""
    extent = shapes.extent

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
        device=shapes.shape_types.device,
        sbt=wo.SbtLayout(hit_groups={"triangles": wo.HitGroup(), "primitives": wo.HitGroup(intersection=intersect)}),
        max_registers=_MAX_REGISTERS,
        module_options={"fast_math": kernel.module.options["fast_math"]},
        graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
    )


def intersect_rays(
    model: Model,
    ray_origins,
    ray_directions,
    ray_worlds,
    *,
    enable_global_world,
    out_dist,
    out_shape_id,
    out_normal,
    fast_math,
) -> None:
    """Intersect rays with :attr:`Model.optix_shapes`."""
    shapes = model.optix_shapes
    if shapes is None:
        raise RuntimeError(
            "OptiX raycasting requires OptiX shapes built for the queried state. Call "
            "model.optix_build_shapes(state) before first use and model.optix_refit_shapes(state) "
            "after state changes."
        )
    if model.device.stream.is_capturing:
        raise RuntimeError("The OptiX backend does not support CUDA graph capture")
    key = (enable_global_world, out_dist is not None, out_shape_id is not None, out_normal is not None, fast_math)
    if key not in shapes.ray_pipelines:
        shapes.ray_pipelines[key] = _create_ray_pipeline(shapes, *key)
    wo.launch(
        shapes.ray_pipelines[key],
        dim=ray_origins.shape[0],
        inputs=[
            shapes.id,
            shapes.shape_world_transforms,
            shapes.shape_types,
            model.shape_scale,
            model.shape_source_ptr,
            ray_origins,
            ray_directions,
            ray_worlds,
        ],
        outputs=[out_dist, out_shape_id, out_normal],
    )


def _create_ray_pipeline(shapes, enable_global_world, write_dist, write_shape_id, write_normal, fast_math):
    grid, extent, world_count = shapes.grid, shapes.extent, shapes.world_count

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
        if wp.static(enable_global_world) or ray_worlds[tid] < 0:
            for p in range(wp.static(len(shapes.planes))):
                si = wp.static(shapes.planes[p])
                plane_distance, plane_normal = raycast.ray_intersect_shape(
                    shape_transforms[si], shape_sizes[si], GeoType.PLANE, origin, direction, False
                )
                if plane_distance >= 0.0 and plane_distance < distance:
                    shape = si
                    distance = plane_distance
                    normal = plane_normal
        # Match the camera's unrolled local/global traversal (OptiX 0.3.0).
        for i in range(wp.static(2 if enable_global_world and shapes.has_global_instances else 1)):
            world = wp.where(i == 1 or ray_worlds[tid] < 0, wp.static(world_count), ray_worlds[tid])
            interval = ray_interval(origin, direction, wp.static(extent), distance)
            if wp.static(shapes.has_instances) and interval[1] >= interval[0]:
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

    return create_pipeline(shapes, intersect)
