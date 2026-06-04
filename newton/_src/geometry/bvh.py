# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..core import MAXVAL
from .flags import ShapeFlags
from .types import Gaussian, GeoType

if TYPE_CHECKING:
    from ..sim import Model, State


@wp.func
def compute_shape_bounds(
    transform: wp.transformf, scale: wp.vec3f, shape_min_bounds: wp.vec3f, shape_max_bounds: wp.vec3f
) -> tuple[wp.vec3f, wp.vec3f]:
    shape_min_bounds = wp.cw_mul(shape_min_bounds, scale)
    shape_max_bounds = wp.cw_mul(shape_max_bounds, scale)

    min_bound = wp.vec3f(MAXVAL)
    max_bound = wp.vec3f(-MAXVAL)

    corner_1 = wp.transform_point(transform, wp.vec3f(shape_min_bounds[0], shape_min_bounds[1], shape_min_bounds[2]))
    min_bound = wp.min(min_bound, corner_1)
    max_bound = wp.max(max_bound, corner_1)

    corner_2 = wp.transform_point(transform, wp.vec3f(shape_max_bounds[0], shape_min_bounds[1], shape_min_bounds[2]))
    min_bound = wp.min(min_bound, corner_2)
    max_bound = wp.max(max_bound, corner_2)

    corner_3 = wp.transform_point(transform, wp.vec3f(shape_max_bounds[0], shape_max_bounds[1], shape_min_bounds[2]))
    min_bound = wp.min(min_bound, corner_3)
    max_bound = wp.max(max_bound, corner_3)

    corner_4 = wp.transform_point(transform, wp.vec3f(shape_min_bounds[0], shape_max_bounds[1], shape_min_bounds[2]))
    min_bound = wp.min(min_bound, corner_4)
    max_bound = wp.max(max_bound, corner_4)

    corner_5 = wp.transform_point(transform, wp.vec3f(shape_min_bounds[0], shape_min_bounds[1], shape_max_bounds[2]))
    min_bound = wp.min(min_bound, corner_5)
    max_bound = wp.max(max_bound, corner_5)

    corner_6 = wp.transform_point(transform, wp.vec3f(shape_max_bounds[0], shape_min_bounds[1], shape_max_bounds[2]))
    min_bound = wp.min(min_bound, corner_6)
    max_bound = wp.max(max_bound, corner_6)

    corner_7 = wp.transform_point(transform, wp.vec3f(shape_min_bounds[0], shape_max_bounds[1], shape_max_bounds[2]))
    min_bound = wp.min(min_bound, corner_7)
    max_bound = wp.max(max_bound, corner_7)

    corner_8 = wp.transform_point(transform, wp.vec3f(shape_max_bounds[0], shape_max_bounds[1], shape_max_bounds[2]))
    min_bound = wp.min(min_bound, corner_8)
    max_bound = wp.max(max_bound, corner_8)

    return min_bound, max_bound


@wp.func
def compute_box_bounds(transform: wp.transformf, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    min_bound = wp.vec3f(MAXVAL)
    max_bound = wp.vec3f(-MAXVAL)

    for x in range(2):
        for y in range(2):
            for z in range(2):
                local_corner = wp.vec3f(
                    size[0] * (2.0 * wp.float32(x) - 1.0),
                    size[1] * (2.0 * wp.float32(y) - 1.0),
                    size[2] * (2.0 * wp.float32(z) - 1.0),
                )
                world_corner = wp.transform_point(transform, local_corner)
                min_bound = wp.min(min_bound, world_corner)
                max_bound = wp.max(max_bound, world_corner)

    return min_bound, max_bound


@wp.func
def compute_sphere_bounds(pos: wp.vec3f, radius: wp.float32) -> tuple[wp.vec3f, wp.vec3f]:
    return pos - wp.vec3f(radius), pos + wp.vec3f(radius)


@wp.func
def compute_capsule_bounds(transform: wp.transformf, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    radius = size[0]
    half_length = size[1]
    extent = wp.vec3f(radius, radius, half_length + radius)
    return compute_box_bounds(transform, extent)


@wp.func
def compute_cylinder_bounds(transform: wp.transformf, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    radius = size[0]
    half_length = size[1]
    extent = wp.vec3f(radius, radius, half_length)
    return compute_box_bounds(transform, extent)


@wp.func
def compute_cone_bounds(transform: wp.transformf, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    extent = wp.vec3f(size[0], size[0], size[1])
    return compute_box_bounds(transform, extent)


@wp.func
def compute_plane_bounds(transform: wp.transformf, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    # If plane size is non-positive, treat as infinite plane and use a large default extent
    size_scale = wp.max(size[0], size[1]) * 2.0
    if size[0] <= 0.0 or size[1] <= 0.0:
        size_scale = 1000.0

    min_bound = wp.vec3f(MAXVAL)
    max_bound = wp.vec3f(-MAXVAL)

    for x in range(2):
        for y in range(2):
            local_corner = wp.vec3f(
                size_scale * (2.0 * wp.float32(x) - 1.0),
                size_scale * (2.0 * wp.float32(y) - 1.0),
                0.0,
            )
            world_corner = wp.transform_point(transform, local_corner)
            min_bound = wp.min(min_bound, world_corner)
            max_bound = wp.max(max_bound, world_corner)

    extent = wp.vec3f(0.1)
    return min_bound - extent, max_bound + extent


@wp.func
def compute_ellipsoid_bounds(transform: wp.transformf, size: wp.vec3f) -> tuple[wp.vec3f, wp.vec3f]:
    extent = wp.vec3f(wp.abs(size[0]), wp.abs(size[1]), wp.abs(size[2]))
    return compute_box_bounds(transform, extent)


@wp.func
def is_supported_shape_type(shape_type: wp.int32) -> wp.bool:
    if shape_type == GeoType.BOX:
        return True
    if shape_type == GeoType.CAPSULE:
        return True
    if shape_type == GeoType.CYLINDER:
        return True
    if shape_type == GeoType.ELLIPSOID:
        return True
    if shape_type == GeoType.PLANE:
        return True
    if shape_type == GeoType.SPHERE:
        return True
    if shape_type == GeoType.CONE:
        return True
    if shape_type == GeoType.MESH:
        return True
    if shape_type == GeoType.GAUSSIAN:
        return True
    return False


@wp.kernel(enable_backward=False)
def compute_enabled_shapes(
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    out_shape_enabled: wp.array[wp.uint32],
    out_shape_enabled_count: wp.array[wp.int32],
):
    tid = wp.tid()

    if not bool(shape_flags[tid] & ShapeFlags.VISIBLE):
        return

    if not is_supported_shape_type(shape_type[tid]):
        return

    index = wp.atomic_add(out_shape_enabled_count, 0, 1)
    out_shape_enabled[index] = wp.uint32(tid)


@wp.func
def compute_gaussian_bounds(gaussians_data: Gaussian.Data, tid: wp.int32) -> tuple[wp.vec3f, wp.vec3f]:
    transform = gaussians_data.transforms[tid]
    scale = gaussians_data.scales[tid]

    mod = gaussians_data.min_response / wp.max(gaussians_data.opacities[tid], wp.float32(1e-6))
    min_response = wp.clamp(mod, wp.float32(1e-6), wp.float32(0.97))
    ks = wp.sqrt(wp.log(min_response) / wp.float32(-0.5))
    scale = wp.vec3f(scale[0] * ks, scale[1] * ks, scale[2] * ks)

    return compute_ellipsoid_bounds(transform, scale)


@wp.kernel(enable_backward=False)
def compute_shape_local_bounds(
    in_shape_type: wp.array[wp.int32],
    in_shape_ptr: wp.array[wp.uint64],
    in_gaussians: wp.array[Gaussian.Data],
    out_bounds: wp.array2d[wp.vec3f],
):
    tid = wp.tid()

    min_point = wp.vec3(MAXVAL)
    max_point = wp.vec3(-MAXVAL)

    if in_shape_type[tid] == GeoType.MESH:
        mesh = wp.mesh_get(in_shape_ptr[tid])
        for i in range(mesh.points.shape[0]):
            min_point = wp.min(min_point, mesh.points[i])
            max_point = wp.max(max_point, mesh.points[i])

    elif in_shape_type[tid] == GeoType.GAUSSIAN:
        gaussian_id = in_shape_ptr[tid]
        for i in range(in_gaussians[gaussian_id].num_points):
            lower, upper = compute_gaussian_bounds(in_gaussians[gaussian_id], i)
            min_point = wp.min(min_point, lower)
            max_point = wp.max(max_point, upper)

    out_bounds[tid, 0] = min_point
    out_bounds[tid, 1] = max_point


@wp.kernel(enable_backward=False)
def compute_shape_world_transforms(
    in_body_transforms: wp.array[wp.transform],
    in_shape_body: wp.array[wp.int32],
    in_shape_transform: wp.array[wp.transformf],
    out_transforms: wp.array[wp.transformf],
):
    tid = wp.tid()

    body = in_shape_body[tid]
    body_transform = wp.transform_identity()
    if body >= 0:
        body_transform = in_body_transforms[body]

    out_transforms[tid] = wp.mul(body_transform, in_shape_transform[tid])


@wp.kernel(enable_backward=False)
def compute_shape_bvh_bounds(
    shape_count_enabled: wp.int32,
    world_count: wp.int32,
    shape_world_index: wp.array[wp.int32],
    shape_enabled: wp.array[wp.uint32],
    shape_types: wp.array[wp.int32],
    shape_sizes: wp.array[wp.vec3f],
    shape_transforms: wp.array[wp.transformf],
    shape_bounds: wp.array2d[wp.vec3f],
    out_bvh_lowers: wp.array[wp.vec3f],
    out_bvh_uppers: wp.array[wp.vec3f],
    out_bvh_groups: wp.array[wp.int32],
):
    tid = wp.tid()
    bvh_index_local = tid % shape_count_enabled
    if bvh_index_local >= shape_count_enabled:
        return

    shape_index = shape_enabled[bvh_index_local]

    world_index = shape_world_index[shape_index]
    if world_index < 0:
        world_index = world_count + world_index

    if world_index >= world_count:
        return

    transform = shape_transforms[shape_index]
    size = shape_sizes[shape_index]
    geom_type = shape_types[shape_index]

    lower = wp.vec3f()
    upper = wp.vec3f()

    if geom_type == GeoType.SPHERE:
        lower, upper = compute_sphere_bounds(wp.transform_get_translation(transform), size[0])
    elif geom_type == GeoType.CAPSULE:
        lower, upper = compute_capsule_bounds(transform, size)
    elif geom_type == GeoType.CYLINDER:
        lower, upper = compute_cylinder_bounds(transform, size)
    elif geom_type == GeoType.CONE:
        lower, upper = compute_cone_bounds(transform, size)
    elif geom_type == GeoType.PLANE:
        lower, upper = compute_plane_bounds(transform, size)
    elif geom_type == GeoType.ELLIPSOID:
        lower, upper = compute_ellipsoid_bounds(transform, size)
    elif geom_type == GeoType.BOX:
        lower, upper = compute_box_bounds(transform, size)
    elif geom_type == GeoType.MESH or geom_type == GeoType.GAUSSIAN:
        min_bounds = shape_bounds[shape_index, 0]
        max_bounds = shape_bounds[shape_index, 1]
        lower, upper = compute_shape_bounds(transform, size, min_bounds, max_bounds)

    out_bvh_lowers[bvh_index_local] = lower
    out_bvh_uppers[bvh_index_local] = upper
    out_bvh_groups[bvh_index_local] = world_index


@wp.kernel(enable_backward=False)
def compute_particle_bvh_bounds(
    num_particles: wp.int32,
    world_count: wp.int32,
    particle_world_index: wp.array[wp.int32],
    particle_position: wp.array[wp.vec3f],
    particle_radius: wp.array[wp.float32],
    out_bvh_lowers: wp.array[wp.vec3f],
    out_bvh_uppers: wp.array[wp.vec3f],
    out_bvh_groups: wp.array[wp.int32],
):
    tid = wp.tid()
    bvh_index_local = tid % num_particles
    if bvh_index_local >= num_particles:
        return

    particle_index = bvh_index_local

    world_index = particle_world_index[particle_index]
    if world_index < 0:
        world_index = world_count + world_index

    if world_index >= world_count:
        return

    lower, upper = compute_sphere_bounds(particle_position[particle_index], particle_radius[particle_index])

    out_bvh_lowers[bvh_index_local] = lower
    out_bvh_uppers[bvh_index_local] = upper
    out_bvh_groups[bvh_index_local] = world_index


@wp.kernel(enable_backward=False)
def compute_bvh_group_roots(bvh_id: wp.uint64, out_bvh_group_roots: wp.array[wp.int32]):
    tid = wp.tid()
    out_bvh_group_roots[tid] = wp.bvh_get_group_root(bvh_id, tid)


def _compute_shape_bvh_bounds_launch(
    model: Model,
    lowers: wp.array,
    uppers: wp.array,
    groups: wp.array,
) -> None:
    """Launch the shape BVH bounds kernel into the provided ``lowers``/``uppers``/``groups`` arrays."""
    wp.launch(
        kernel=compute_shape_bvh_bounds,
        dim=model.bvh_shape_count_enabled,
        inputs=[
            model.bvh_shape_count_enabled,
            model.world_count + 1,
            model.shape_world,
            model.bvh_shape_enabled,
            model.shape_type,
            model.shape_scale,
            model.bvh_shape_world_transforms,
            model.bvh_shape_bounds,
            lowers,
            uppers,
            groups,
        ],
        device=model.device,
    )


def _compute_shape_world_transforms_launch(model: Model, state: State) -> None:
    """Populate ``model.bvh_shape_world_transforms`` from body poses in *state*."""
    wp.launch(
        kernel=compute_shape_world_transforms,
        dim=model.shape_count,
        inputs=[
            state.body_q,
            model.shape_body,
            model.shape_transform,
            model.bvh_shape_world_transforms,
        ],
        device=model.device,
    )


def build_bvh_shape(model: Model, state: State) -> None:
    """Build the shape BVH stored on *model*.

    Allocates :attr:`~newton.Model.bvh_shapes` and related fields (shape
    enabled filter, per-shape local AABBs, world-space transforms, group
    roots) and populates them from the current *state*. Must be called
    before :func:`refit_bvh_shape` and before any sensor that reads the
    shape BVH (e.g. :class:`~newton.sensors.SensorTiledCamera`).

    Args:
        model: Simulation model providing shape metadata.
        state: Current simulation state with body transforms.
    """
    if model.shape_count == 0:
        return

    device = model.device
    shape_count = model.shape_count
    world_count_total = model.world_count + 1

    model.bvh_shape_bounds = wp.empty((shape_count, 2), dtype=wp.vec3f, ndim=2, device=device)
    wp.launch(
        kernel=compute_shape_local_bounds,
        dim=shape_count,
        inputs=[
            model.shape_type,
            model.shape_source_ptr,
            model.gaussians_data,
            model.bvh_shape_bounds,
        ],
        device=device,
    )

    model.bvh_shape_enabled = wp.empty(shape_count, dtype=wp.uint32, device=device)
    num_enabled = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel=compute_enabled_shapes,
        dim=shape_count,
        inputs=[
            model.shape_type,
            model.shape_flags,
            model.bvh_shape_enabled,
            num_enabled,
        ],
        device=device,
    )
    model.bvh_shape_count_enabled = int(num_enabled.numpy()[0])
    model.bvh_shape_world_transforms = wp.empty(shape_count, dtype=wp.transformf, device=device)

    if model.bvh_shape_count_enabled == 0:
        return

    _compute_shape_world_transforms_launch(model, state)

    lowers = wp.zeros(model.bvh_shape_count_enabled, dtype=wp.vec3f, device=device)
    uppers = wp.zeros(model.bvh_shape_count_enabled, dtype=wp.vec3f, device=device)
    groups = wp.zeros(model.bvh_shape_count_enabled, dtype=wp.int32, device=device)
    _compute_shape_bvh_bounds_launch(model, lowers, uppers, groups)
    model.bvh_shapes = wp.Bvh(lowers, uppers, groups=groups)

    model.bvh_shapes_group_roots = wp.zeros(world_count_total, dtype=wp.int32, device=device)
    wp.launch(
        kernel=compute_bvh_group_roots,
        dim=world_count_total,
        inputs=[model.bvh_shapes.id, model.bvh_shapes_group_roots],
        device=device,
    )


def refit_bvh_shape(model: Model, state: State) -> None:
    """Refit the shape BVH stored on *model* for the current *state*.

    Requires :func:`build_bvh_shape` to have been called beforehand. Updates
    world-space shape transforms from ``state.body_q`` and refits the BVH
    in place.

    Args:
        model: Simulation model providing shape metadata.
        state: Current simulation state with body transforms.
    """
    if model.shape_count == 0 or model.bvh_shape_count_enabled == 0:
        return
    if model.bvh_shapes is None:
        raise RuntimeError("refit_bvh_shape requires build_bvh_shape to have been called first.")

    _compute_shape_world_transforms_launch(model, state)
    _compute_shape_bvh_bounds_launch(model, model.bvh_shapes.lowers, model.bvh_shapes.uppers, model.bvh_shapes.groups)
    model.bvh_shapes.refit()


def _compute_particle_bvh_bounds_launch(
    model: Model,
    state: State,
    lowers: wp.array,
    uppers: wp.array,
    groups: wp.array,
) -> None:
    """Launch the particle BVH bounds kernel into the provided ``lowers``/``uppers``/``groups`` arrays."""
    wp.launch(
        kernel=compute_particle_bvh_bounds,
        dim=state.particle_count,
        inputs=[
            state.particle_count,
            model.world_count + 1,
            model.particle_world,
            state.particle_q,
            model.particle_radius,
            lowers,
            uppers,
            groups,
        ],
        device=model.device,
    )


def build_bvh_particle(model: Model, state: State) -> None:
    """Build the particle BVH stored on *model*.

    Allocates :attr:`~newton.Model.bvh_particles` and
    :attr:`~newton.Model.bvh_particles_group_roots` and populates them from
    the current *state*. Must be called before :func:`refit_bvh_particle`
    and before any sensor that reads the particle BVH.

    Args:
        model: Simulation model providing particle metadata.
        state: Current simulation state with particle positions.
    """
    if state.particle_q is None or state.particle_count == 0:
        return

    device = model.device
    world_count_total = model.world_count + 1
    num_particles = state.particle_count

    lowers = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    uppers = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    groups = wp.zeros(num_particles, dtype=wp.int32, device=device)
    _compute_particle_bvh_bounds_launch(model, state, lowers, uppers, groups)
    model.bvh_particles = wp.Bvh(lowers, uppers, groups=groups)

    model.bvh_particles_group_roots = wp.zeros(world_count_total, dtype=wp.int32, device=device)
    wp.launch(
        kernel=compute_bvh_group_roots,
        dim=world_count_total,
        inputs=[model.bvh_particles.id, model.bvh_particles_group_roots],
        device=device,
    )


def refit_bvh_particle(model: Model, state: State) -> None:
    """Refit the particle BVH stored on *model* for the current *state*.

    Requires :func:`build_bvh_particle` to have been called beforehand.
    Recomputes particle bounds from ``state.particle_q`` and refits the BVH
    in place.

    Args:
        model: Simulation model providing particle metadata.
        state: Current simulation state with particle positions.
    """
    if state.particle_q is None or state.particle_count == 0:
        return
    if model.bvh_particles is None:
        raise RuntimeError("refit_bvh_particle requires build_bvh_particle to have been called first.")

    _compute_particle_bvh_bounds_launch(
        model, state, model.bvh_particles.lowers, model.bvh_particles.uppers, model.bvh_particles.groups
    )
    model.bvh_particles.refit()
