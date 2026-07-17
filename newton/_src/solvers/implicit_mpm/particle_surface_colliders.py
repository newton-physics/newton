# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MPM collider adapters for particle surface extraction."""

import math

import warp as wp

from ...geometry.particle_surface import ParticleSurface
from .rasterized_collisions import Collider, collision_sdf

wp.set_module_options({"enable_backward": False})

_DEFAULT_COLLIDER_EXTRAPOLATION_DEPTH_SCALE = 4.0


@wp.kernel
def _mirror_sparse_sdf_into_colliders(
    node_grid: wp.uint64,
    node_ijk: wp.array[wp.vec3i],
    node_world: wp.array[wp.int32],
    env_offsets: wp.array[wp.vec3i],
    field: wp.array[float],
    field_orig: wp.array[float],
    grid_counts: wp.array[wp.int32],
    voxel_size: float,
    outside_value: float,
    onset: float,
    max_depth: float,
    collider: Collider,
    collider_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    stride: int,
):
    index = wp.tid()
    node_count = wp.volume_voxel_count(node_grid)
    while index < node_count:
        world = node_world[index]
        if world < 0 or grid_counts[7 * world + 3] != 0:
            index += stride
            continue
        coordinate = node_ijk[index] - env_offsets[world]
        position = voxel_size * wp.vec3(float(coordinate[0]), float(coordinate[1]), float(coordinate[2]))

        distance, normal, _velocity, _collider_id, _material_id = collision_sdf(
            position, collider, body_q, body_qd, body_q_prev, 1.0, collider_world, world
        )
        depth = onset - distance
        if depth >= 0.0 and depth <= max_depth:
            mirror_position = position + 2.0 * depth * normal
            env_offset = env_offsets[world]
            mirror_index = mirror_position / voxel_size + wp.vec3(
                float(env_offset[0]), float(env_offset[1]), float(env_offset[2])
            )
            mirror_value = wp.volume_sample_index(
                node_grid,
                mirror_index,
                wp.Volume.LINEAR,
                field_orig,
                outside_value,
            )
            blend_start = 0.5 * max_depth
            if depth <= blend_start:
                field[index] = mirror_value
            else:
                t = (depth - blend_start) / wp.max(max_depth - blend_start, 1.0e-10)
                blend = t * t * (3.0 - 2.0 * t)
                field[index] = (1.0 - blend) * mirror_value + blend * field_orig[index]
        index += stride


def extrapolate_surface_sdf_into_colliders(
    surface: ParticleSurface,
    collider: Collider,
    collider_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    *,
    max_depth: float | None = None,
    onset: float = 0.0,
    redistance: bool = True,
    compute_normals: bool = True,
) -> ParticleSurface.ExtractionMesh:
    """Extrapolate a particle SDF into colliders and extract its surface.

    Args:
        surface: Particle surface containing the SDF field.
        collider: MPM collider representation to extrapolate into.
        collider_world: World index for each collider, or ``-1`` for global.
        body_q: Current rigid body transforms, shape ``(body_count,)``.
        max_depth: Maximum extrapolation depth inside colliders [m]. Defaults to
            the smaller of ``4 * surface.voxel_size`` and the allocated
            particle-surface topology halo.
        onset: Signed collider distance where extrapolation starts [m]. A value
            of ``0`` starts at the collider surface.
        redistance: Whether to redistance the modified SDF before extracting the mesh.
        compute_normals: Whether to compute per-vertex normals.

    Returns:
        Mesh buffers and device-resident logical counts.

    Raises:
        ValueError: If ``surface`` does not contain an SDF field.
    """
    if surface.field_mode != "sdf":
        raise ValueError("Collider extrapolation requires ParticleSurface(field_mode='sdf')")
    capacity = surface._capacity
    if capacity is None:
        raise ValueError("Particle surface field has not been extracted")
    topology_halo = (
        surface.padding
        + surface.field_smooth_iterations * surface.field_smooth_radius
        + surface.redistance_iterations
        + 1
    ) * surface.voxel_size
    if not math.isfinite(onset):
        raise ValueError("onset must be finite")
    if max_depth is None:
        max_depth = min(_DEFAULT_COLLIDER_EXTRAPOLATION_DEPTH_SCALE * surface.voxel_size, topology_halo)
    elif not math.isfinite(max_depth) or max_depth < 0.0:
        raise ValueError("max_depth must be finite and non-negative")
    required_inward_halo = max_depth + max(0.0, -onset)
    if required_inward_halo > topology_halo:
        raise ValueError(
            f"Collider extrapolation requires an inward halo of {required_inward_halo}, "
            f"which exceeds the allocated particle-surface topology halo ({topology_halo})"
        )
    if capacity.volume is None or capacity.voxel_ijk is None or capacity.node_world is None:
        raise ValueError("Particle surface field has no sparse grid topology")

    wp.copy(capacity.field_orig, capacity.field)
    outside_value = surface.threshold
    if surface.surface_method == "particle_sdf":
        outside_value = surface.kernel_radius * surface.particle_sdf_band
    previous_query_distance = collider.query_max_dist
    collider.query_max_dist = max(previous_query_distance, max_depth + abs(onset) + surface.voxel_size)
    try:
        wp.launch(
            _mirror_sparse_sdf_into_colliders,
            dim=capacity.launch_threads,
            inputs=[
                capacity.volume.id,
                capacity.voxel_ijk,
                capacity.node_world,
                capacity.env_offsets,
                capacity.field,
                capacity.field_orig,
                capacity.grid_counts,
                surface.voxel_size,
                outside_value,
                onset,
                max_depth,
                collider,
                collider_world,
                body_q,
                None,
                None,
                capacity.launch_threads,
            ],
            device=capacity.device,
        )
    finally:
        collider.query_max_dist = previous_query_distance

    if redistance:
        surface.redistance(max(surface.redistance_iterations, 1))
    return surface.resurface(compute_normals=compute_normals)
