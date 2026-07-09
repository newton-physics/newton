# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MPM collider adapters for particle surface extraction."""

import warp as wp

from ...geometry.particle_surface import ParticleSurface
from .rasterized_collisions import Collider, collision_sdf

wp.set_module_options({"enable_backward": False})

_DEFAULT_COLLIDER_EXTRAPOLATION_DEPTH_SCALE = 4.0


@wp.func
def _sample_sparse_field_trilinear(
    node_grid: wp.uint64,
    field: wp.array[float],
    env_offset: wp.vec3i,
    inv_voxel_size: float,
    pos: wp.vec3,
    outside_value: float,
) -> float:
    p = pos * inv_voxel_size
    i0 = int(wp.floor(p[0]))
    j0 = int(wp.floor(p[1]))
    k0 = int(wp.floor(p[2]))
    fx = p[0] - float(i0)
    fy = p[1] - float(j0)
    fz = p[2] - float(k0)

    c000 = outside_value
    c100 = outside_value
    c010 = outside_value
    c110 = outside_value
    c001 = outside_value
    c101 = outside_value
    c011 = outside_value
    c111 = outside_value
    for corner in range(8):
        coordinate = wp.vec3i(i0 + ((corner >> 2) & 1), j0 + ((corner >> 1) & 1), k0 + (corner & 1))
        coordinate += env_offset
        node = wp.volume_lookup_index(node_grid, coordinate[0], coordinate[1], coordinate[2])
        value = outside_value
        if node >= 0:
            value = field[node]
        if corner == 0:
            c000 = value
        elif corner == 1:
            c001 = value
        elif corner == 2:
            c010 = value
        elif corner == 3:
            c011 = value
        elif corner == 4:
            c100 = value
        elif corner == 5:
            c101 = value
        elif corner == 6:
            c110 = value
        else:
            c111 = value

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


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
            mirror_value = _sample_sparse_field_trilinear(
                node_grid,
                field_orig,
                env_offsets[world],
                1.0 / voxel_size,
                mirror_position,
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
            ``4 * surface.voxel_size``.
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
    if max_depth is None:
        max_depth = _DEFAULT_COLLIDER_EXTRAPOLATION_DEPTH_SCALE * surface.voxel_size
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
