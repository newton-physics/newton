# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OptiX shape acceleration, built alongside the shape BVH.

Used by :meth:`Model.bvh_build_shapes` and :meth:`Model.bvh_refit_shapes` with ``backend="optix"``. Importing this module
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

GAUSSIAN_MASK = 4
"""Visibility mask bit of Gaussian shape instances, which only splat traversals include."""
OPAQUE_RAY_MASK = 0xFF & ~GAUSSIAN_MASK
"""Ray mask of traversals for the nearest surface, which skip Gaussian splats."""
GAUSSIAN_SBT_OFFSET = 2
"""Hit group of Gaussian shape instances."""
GAUSSIAN_ASSET_BITS = 16
"""Bits of a Gaussian instance ID holding its asset; the bits above hold its slot."""


class UnsupportedError(RuntimeError):
    """OptiX cannot trace this model on its device."""


@dataclass
class OptixShapes:
    """OptiX instance acceleration over a model's shapes, with one cell per world.

    Each world's shapes are packed into a cube of half-width :attr:`extent`, spaced ``3.5 * extent``
    apart, and rays are clipped to their world's cube. The cell grid, world count and extent are fixed
    when the shapes are built, so kernels compile them in. Planes are intersected analytically, filtered
    by the ray's world.
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
    shape_radii: wp.array[wp.float32]
    """Bounding sphere radius of each shape about its origin [m], shape ``[shape_count]``."""
    shape_traversables: wp.array[wp.uint64]
    """Acceleration structure instanced by each shape, shape ``[shape_count]``."""
    shape_instance_ids: wp.array[wp.uint32]
    """OptiX instance ID of each shape: the shape index, or for Gaussian shapes, their slot and asset
    (see :func:`gaussian_slots`), shape ``[shape_count]``."""
    slot_gaussians: wp.array2d[wp.int32]
    """Gaussian shape in each slot of each world's rays, padded with -1, shape ``[world_count, slots]``."""
    instance_shapes: wp.array[wp.int32]
    """Shape index of each instance."""
    instances: wp.array[wo.Instance]
    """OptiX instance of each shape in :attr:`instance_shapes`."""
    planes: tuple[int, ...]
    """Global planes."""
    world_planes: wp.array2d[wp.int32]
    """Planes of each world, padded with -1, shape ``[world_count, max planes per world]``."""
    grid: int
    """Cells per axis of the world grid."""
    extent: float
    """Half-width of each world's cell [m]."""
    world_count: int
    """Number of worlds; the global world uses the cell after them."""
    has_global_instances: bool
    """Whether the global world's cell holds any instances."""
    out_of_cell: wp.array[wp.int32]
    """Set to 1 on the device when a refit hides a shape that left its cell, or when a camera's particle or
    deformable point leaves its cell (those stay visible and may show in a neighboring world); cleared by a
    rebuild, shape ``[1]``."""
    pipelines: dict = field(default_factory=dict)
    """Camera and raycast pipelines compiled for these shapes."""

    @property
    def has_instances(self) -> bool:
        return self.instance_shapes.shape[0] > 0


@wp.func
def world_offset(world: int, grid: int, extent: float) -> wp.vec3:
    center = float(grid - 1) * 0.5
    cell = wp.vec3(float(world % grid), float((world // grid) % grid), float(world // (grid * grid)))
    return (cell - wp.vec3(center)) * (3.5 * extent)


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
def primitive_half_extents(kind: int, size: wp.vec3) -> wp.vec3:
    """Half extents that scale the shared unit cube to an analytic shape's bounds."""
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
    shape_radii: wp.array[wp.float32],
    shape_transforms: wp.array[wp.transformf],
    shape_traversables: wp.array[wp.uint64],
    shape_instance_ids: wp.array[wp.uint32],
    out_instances: wp.array[wo.Instance],
    out_of_cell: wp.array[wp.int32],
):
    i = wp.tid()
    si = instance_shapes[i]
    kind = shape_types[si]
    size = shape_sizes[si]
    world = wp.where(shape_world[si] < 0, world_count, shape_world[si])
    transform = shape_transforms[si]
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(transform))
    center = wp.transform_get_translation(transform)
    half = primitive_half_extents(kind, size)
    mask = wp.where(kind == GeoType.GAUSSIAN, wp.static(GAUSSIAN_MASK), wp.static(OPAQUE_RAY_MASK))
    if wp.max(wp.abs(center)) + shape_radii[si] > extent:
        # Hide shapes that left their cell rather than let them reach another world's.
        mask = 0
        out_of_cell[0] = 1
    out_instances[i] = wo.make_instance(
        rotation @ wp.diag(half),
        center + world_offset(world, grid, extent),
        shape_traversables[si],
        shape_instance_ids[si],
        wp.uint32(
            wp.where(
                kind == GeoType.MESH or kind == GeoType.BOX,
                0,
                wp.where(kind == GeoType.GAUSSIAN, wp.static(GAUSSIAN_SBT_OFFSET), 1),
            )
        ),
        wp.uint32(mask),
        # Boxes keep the analytic box's behavior: their inside faces are visible with culling.
        wp.uint32(
            wp.where(
                kind == GeoType.BOX,
                wp.static(int(wo.InstanceFlags.DISABLE_ANYHIT | wo.InstanceFlags.DISABLE_TRIANGLE_FACE_CULLING)),
                wp.static(int(wo.InstanceFlags.DISABLE_ANYHIT)),
            )
        ),
    )


def check_supported(model: Model) -> None:
    """Raise :class:`UnsupportedError` if OptiX cannot trace the model's shapes on its device."""
    if not model.device.is_cuda:
        raise UnsupportedError("The OptiX backend requires a CUDA device")
    try:
        wo.get_context(model.device)
    except Exception as exc:
        raise UnsupportedError(f"OptiX is not supported on {model.device}: {exc}") from exc
    # The two highest instance IDs mark a camera's particles and deformable triangles.
    if model.shape_count >= (1 << 24) - 2:
        raise UnsupportedError("OptiX shape instance IDs must fit in 24 bits")
    if model._variant_shape_source:
        raise UnsupportedError("The OptiX backend does not support shape variants")


def build_shape_geometry(
    model: Model, shapes: np.ndarray, shape_types: np.ndarray
) -> tuple[list[wo.Accel], wp.array[wp.uint64]]:
    """Build the triangle structures of the meshes used by ``shapes``, a triangle cube for boxes, the
    other primitives' unit cube, and a structure over the splats of each Gaussian asset."""
    sources = model.shape_source_ptr.numpy()
    geometry = []
    mesh_ids = {}
    splat_ids = {}
    for si in shapes[shape_types[shapes] == GeoType.GAUSSIAN]:
        asset = int(sources[si])
        if asset not in splat_ids:
            from ..render.gaussians import compute_asset_splat_bounds  # noqa: PLC0415

            # The asset's splat bounds, in the shape's space before its scale.
            count = model.shape_source[si].count
            lowers = wp.empty(count, dtype=wp.vec3, device=model.device)
            uppers = wp.empty(count, dtype=wp.vec3, device=model.device)
            wp.launch(
                compute_asset_splat_bounds,
                dim=count,
                inputs=[model.gaussians_data, asset],
                outputs=[lowers, uppers],
                device=model.device,
            )
            geometry.append(wo.Accel([wo.aabbs(lowers, uppers)], compact=True))
            splat_ids[asset] = geometry[-1].id
    used_meshes = set(sources[shapes[shape_types[shapes] == GeoType.MESH]].tolist())
    for mesh in [*model._mesh_keep_alive, *model.heightfield_meshes]:
        if mesh.id in used_meshes and mesh.id not in mesh_ids:
            geometry.append(wo.Accel([wo.triangles(mesh.points, mesh.indices)], compact=True))
            mesh_ids[mesh.id] = geometry[-1].id
    # Boxes are traced as hardware triangles: the unit cube, wound outward.
    corners = wp.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=wp.vec3, device=model.device
    )
    faces = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    box = wo.Accel([wo.triangles(corners, wp.array(faces, dtype=wp.int32, device=model.device))], compact=True)
    bounds = wp.array([[-1.001] * 3 + [1.001] * 3], dtype=float, device=model.device)
    primitive = wo.Accel([wo.aabbs(bounds=bounds)], compact=True)
    geometry += [box, primitive]
    traversables = np.zeros(model.shape_count, dtype=np.uint64)
    for si in shapes:
        kind = shape_types[si]
        if kind == GeoType.MESH:
            traversables[si] = mesh_ids[int(sources[si])]
        elif kind == GeoType.GAUSSIAN:
            traversables[si] = splat_ids[int(sources[si])]
        else:
            traversables[si] = box.id if kind == GeoType.BOX else primitive.id
    return geometry, wp.array(traversables, dtype=wp.uint64, device=model.device)


def gaussian_slots(model: Model, instance_shapes: np.ndarray, shape_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Number the Gaussian shapes a ray can cross: each world's own, then the global ones in every world.

    Splat hits name their shape by this slot, so a ray needs no list of the shapes it crossed, and
    Gaussian instances report ``slot << GAUSSIAN_ASSET_BITS | asset`` as instance ID, so splat
    intersection reads no table. Other instances report their shape index.

    Returns:
        ``(instance_ids, slot_gaussians)``: the instance ID of each shape, and the Gaussian shape in
        each slot of each world, padded with -1.
    """
    instance_ids = np.arange(model.shape_count, dtype=np.uint32)
    gaussian = instance_shapes[shape_types[instance_shapes] == GeoType.GAUSSIAN]
    worlds = model.shape_world.numpy()[gaussian]
    local = gaussian[worlds >= 0][np.argsort(worlds[worlds >= 0], kind="stable")]
    local_worlds = np.sort(worlds[worlds >= 0], kind="stable")
    local_slots = np.arange(len(local)) - np.searchsorted(local_worlds, local_worlds)
    global_shapes = gaussian[worlds < 0]
    local_slot_count = int(local_slots.max(initial=-1)) + 1
    slots = np.zeros(model.shape_count, dtype=np.int64)
    slots[local] = local_slots
    slots[global_shapes] = local_slot_count + np.arange(len(global_shapes))
    assets = model.shape_source_ptr.numpy()[gaussian].astype(np.int64)
    if len(gaussian) and (slots[gaussian].max() >= 256 or assets.max() >= 1 << GAUSSIAN_ASSET_BITS):
        raise UnsupportedError(
            f"OptiX cameras support 256 Gaussian shapes per world and {1 << GAUSSIAN_ASSET_BITS} assets"
        )
    instance_ids[gaussian] = (slots[gaussian] << GAUSSIAN_ASSET_BITS | assets).astype(np.uint32)
    slot_gaussians = np.full((model.world_count, local_slot_count + len(global_shapes)), -1, dtype=np.int32)
    slot_gaussians[local_worlds, local_slots] = local
    slot_gaussians[:, local_slot_count:] = global_shapes
    return instance_ids, slot_gaussians


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
            shapes.shape_radii,
            shapes.shape_world_transforms,
            shapes.shape_traversables,
            shapes.shape_instance_ids,
            shapes.instances,
            shapes.out_of_cell,
        ],
        device=model.device,
    )


def create_pipeline(
    shapes: OptixShapes,
    kernel: wp.Kernel,
    *,
    enable_backface_culling: bool = False,
    splat_intersection: wp.Kernel | None = None,
    payload_words: int = 0,
    particle_build_flags: int | None = None,
    clustered_geometry: bool = False,
) -> wo.Pipeline:
    """Compile ``kernel`` as the ray generation program of a pipeline over ``shapes``.

    Hit groups: triangles, analytic primitives, Gaussian splats at :data:`GAUSSIAN_SBT_OFFSET` with
    ``splat_intersection`` as their program (pipelines that never trace splats omit it), and with
    ``particle_build_flags`` a camera's particles, as built-in spheres built with those flags.
    ``clustered_geometry`` allows tracing cluster geometry.
    """

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
        half = primitive_half_extents(kind, size)
        distance, _normal = raycast.ray_intersect_shape_local(
            size,
            kind,
            wp.cw_mul(wp.optix_get_object_ray_origin(), half),
            wp.cw_mul(wp.optix_get_object_ray_direction(), half),
            wp.static(enable_backface_culling),
        )
        if distance >= wp.optix_get_ray_tmin():
            wp.optix_report_intersection(distance, wp.uint32(0), wp.uint32(0), wp.uint32(0))

    hit_groups = {
        "triangles": wo.HitGroup(),
        "primitives": wo.HitGroup(intersection=intersect),
        "splats": wo.HitGroup(intersection=splat_intersection),
    }
    if particle_build_flags is not None:
        hit_groups["particles"] = wo.HitGroup(builtin=wo.PrimitiveType.SPHERE, build_flags=particle_build_flags)
    options = {"clustered_geometry": True} if clustered_geometry else {}
    return wo.Pipeline(
        kernel,
        device=shapes.shape_types.device,
        sbt=wo.SbtLayout(hit_groups=hit_groups),
        max_registers=_MAX_REGISTERS,
        payload_words=payload_words,
        module_options={"fast_math": kernel.module.options["fast_math"]},
        graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
        **options,
    )


def intersect_rays(
    model: Model,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    ray_worlds: wp.array[wp.int32],
    *,
    enable_global_world: bool,
    out_dist: wp.array[float] | None,
    out_shape_id: wp.array[wp.int32] | None,
    out_normal: wp.array[wp.vec3] | None,
    fast_math: bool,
) -> None:
    """Intersect rays with :attr:`Model.optix_shapes`."""
    shapes = model.optix_shapes
    if shapes is None:
        raise RuntimeError(
            "OptiX raycasting requires OptiX shapes built for the queried state. Call "
            "model.bvh_build_shapes(state, backend='optix') before first use and "
            "model.bvh_refit_shapes(state, backend='optix') after state changes."
        )
    if model.device.stream.is_capturing:
        raise RuntimeError("The OptiX backend does not support CUDA graph capture")
    key = (enable_global_world, out_dist is not None, out_shape_id is not None, out_normal is not None, fast_math)
    if key not in shapes.pipelines:
        shapes.pipelines[key] = _create_ray_pipeline(shapes, *key)
    wo.launch(
        shapes.pipelines[key],
        dim=ray_origins.shape[0],
        inputs=[
            shapes.id,
            shapes.world_planes,
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


def _create_ray_pipeline(
    shapes: OptixShapes,
    enable_global_world: bool,
    write_dist: bool,
    write_shape_id: bool,
    write_normal: bool,
    fast_math: bool,
) -> wo.Pipeline:
    grid, extent, world_count = shapes.grid, shapes.extent, shapes.world_count

    @wp.func
    def nearest_plane(
        si: int,
        shape_transforms: wp.array[wp.transform],
        shape_sizes: wp.array[wp.vec3],
        origin: wp.vec3,
        direction: wp.vec3,
        distance: float,
        shape: int,
        normal: wp.vec3,
    ) -> tuple[float, int, wp.vec3]:
        plane_distance, plane_normal = raycast.ray_intersect_shape(
            shape_transforms[si], shape_sizes[si], GeoType.PLANE, origin, direction, False
        )
        if plane_distance >= 0.0 and plane_distance < distance:
            return plane_distance, si, plane_normal
        return distance, shape, normal

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": fast_math})
    def intersect(
        handle: wp.uint64,
        world_planes: wp.array2d[wp.int32],
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
        # Planes first: a nearer hit shortens the traversals.
        if wp.static(enable_global_world) or ray_worlds[tid] < 0:
            for p in range(wp.static(len(shapes.planes))):
                distance, shape, normal = nearest_plane(
                    wp.static(shapes.planes[p]),
                    shape_transforms,
                    shape_sizes,
                    origin,
                    direction,
                    distance,
                    shape,
                    normal,
                )
        if ray_worlds[tid] >= 0:
            for p in range(wp.static(shapes.world_planes.shape[1])):
                si = world_planes[ray_worlds[tid], p]
                if si >= 0:
                    distance, shape, normal = nearest_plane(
                        si, shape_transforms, shape_sizes, origin, direction, distance, shape, normal
                    )
        # Match the camera's unrolled local/global traversal (OptiX 0.3.0).
        for i in range(wp.static(2 if enable_global_world and shapes.has_global_instances else 1)):
            world = wp.where(i == 1 or ray_worlds[tid] < 0, wp.static(world_count), ray_worlds[tid])
            interval = ray_interval(origin, direction, wp.static(extent), distance)
            if wp.static(shapes.has_instances) and interval[1] >= interval[0]:
                packed_origin = origin + world_offset(world, wp.static(grid), wp.static(extent))
                query = wo.accel_query_ray(
                    handle, packed_origin, direction, interval[1], interval[0], wp.static(OPAQUE_RAY_MASK)
                )
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
