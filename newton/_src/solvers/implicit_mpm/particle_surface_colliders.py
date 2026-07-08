# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MPM collider adapters for particle surface extraction."""

import warp as wp

from ...geometry.particle_surface import ParticleSurface
from .rasterized_collisions import Collider, collision_sdf

wp.set_module_options({"enable_backward": False})

_DEFAULT_COLLIDER_EXTRAPOLATION_DEPTH_SCALE = 4.0


@wp.func
def _capacity_node_index(i: int, j: int, k: int, dims: wp.vec3i) -> int:
    return (i * dims[1] + j) * dims[2] + k


@wp.func
def _sample_capacity_field_trilinear(
    field: wp.array[float],
    node_start: int,
    dims: wp.vec3i,
    grid_origin: wp.vec3,
    inv_voxel_size: float,
    pos: wp.vec3,
) -> float:
    p = (pos - grid_origin) * inv_voxel_size
    px = wp.clamp(p[0], 0.0, float(dims[0] - 1))
    py = wp.clamp(p[1], 0.0, float(dims[1] - 1))
    pz = wp.clamp(p[2], 0.0, float(dims[2] - 1))

    i0 = wp.clamp(int(wp.floor(px)), 0, dims[0] - 2)
    j0 = wp.clamp(int(wp.floor(py)), 0, dims[1] - 2)
    k0 = wp.clamp(int(wp.floor(pz)), 0, dims[2] - 2)
    fx = px - float(i0)
    fy = py - float(j0)
    fz = pz - float(k0)

    c000 = field[node_start + _capacity_node_index(i0, j0, k0, dims)]
    c100 = field[node_start + _capacity_node_index(i0 + 1, j0, k0, dims)]
    c010 = field[node_start + _capacity_node_index(i0, j0 + 1, k0, dims)]
    c110 = field[node_start + _capacity_node_index(i0 + 1, j0 + 1, k0, dims)]
    c001 = field[node_start + _capacity_node_index(i0, j0, k0 + 1, dims)]
    c101 = field[node_start + _capacity_node_index(i0 + 1, j0, k0 + 1, dims)]
    c011 = field[node_start + _capacity_node_index(i0, j0 + 1, k0 + 1, dims)]
    c111 = field[node_start + _capacity_node_index(i0 + 1, j0 + 1, k0 + 1, dims)]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.kernel
def _mirror_capacity_sdf_into_colliders(
    field: wp.array[float],
    field_orig: wp.array[float],
    grid_origin: wp.array[wp.vec3],
    grid_dims: wp.array[wp.vec3i],
    grid_counts: wp.array[wp.int32],
    grid_node_world_start: wp.array[wp.int32],
    voxel_size: float,
    onset: float,
    max_depth: float,
    collider: Collider,
    collider_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    stride: int,
):
    world, index = wp.tid()
    counts_base = 7 * world
    node_count = grid_counts[counts_base]
    if grid_counts[counts_base + 3] != 0:
        node_count = 0
    dims = grid_dims[world]
    yz = dims[1] * dims[2]
    origin = grid_origin[world]
    node_start = grid_node_world_start[world]
    while index < node_count:
        i = index // yz
        remainder = index - i * yz
        j = remainder // dims[2]
        k = remainder - j * dims[2]
        position = origin + voxel_size * wp.vec3(float(i), float(j), float(k))

        distance, normal, _velocity, _collider_id, _material_id = collision_sdf(
            position, collider, body_q, body_qd, body_q_prev, 1.0, collider_world, world
        )
        depth = onset - distance
        if depth >= 0.0 and depth <= max_depth:
            mirror_position = position + 2.0 * depth * normal
            mirror_value = _sample_capacity_field_trilinear(
                field_orig,
                node_start,
                dims,
                origin,
                1.0 / voxel_size,
                mirror_position,
            )
            blend_start = 0.5 * max_depth
            if depth <= blend_start:
                field[node_start + index] = mirror_value
            else:
                t = (depth - blend_start) / wp.max(max_depth - blend_start, 1.0e-10)
                blend = t * t * (3.0 - 2.0 * t)
                field[node_start + index] = (1.0 - blend) * mirror_value + blend * field_orig[node_start + index]
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

    wp.copy(capacity.field_orig, capacity.field)
    previous_query_distance = collider.query_max_dist
    collider.query_max_dist = max(previous_query_distance, max_depth + abs(onset) + surface.voxel_size)
    try:
        wp.launch(
            _mirror_capacity_sdf_into_colliders,
            dim=(capacity.world_count, capacity.launch_threads),
            inputs=[
                capacity.field,
                capacity.field_orig,
                capacity.grid_origin,
                capacity.grid_dims,
                capacity.grid_counts,
                capacity.grid_node_world_start,
                surface.voxel_size,
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
