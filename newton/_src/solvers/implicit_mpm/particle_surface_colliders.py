# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MPM collider adapters for particle surface extraction."""

import warp as wp

from ...geometry.particle_surface import ParticleSurface
from .rasterized_collisions import Collider, collision_sdf

wp.set_module_options({"enable_backward": False})

_DEFAULT_COLLIDER_EXTRAPOLATION_DEPTH_SCALE = 4.0


@wp.func
def _sample_field_trilinear(
    field: wp.array3d[wp.float32],
    grid_origin: wp.vec3,
    inv_voxel_size: float,
    pos: wp.vec3,
):
    p = (pos - grid_origin) * inv_voxel_size
    px = wp.clamp(p[0], 0.0, float(field.shape[0] - 1))
    py = wp.clamp(p[1], 0.0, float(field.shape[1] - 1))
    pz = wp.clamp(p[2], 0.0, float(field.shape[2] - 1))

    i0 = wp.clamp(int(wp.floor(px)), 0, field.shape[0] - 2)
    j0 = wp.clamp(int(wp.floor(py)), 0, field.shape[1] - 2)
    k0 = wp.clamp(int(wp.floor(pz)), 0, field.shape[2] - 2)

    fx = px - float(i0)
    fy = py - float(j0)
    fz = pz - float(k0)

    c00 = field[i0, j0, k0] * (1.0 - fx) + field[i0 + 1, j0, k0] * fx
    c10 = field[i0, j0 + 1, k0] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0] * fx
    c01 = field[i0, j0, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0, k0 + 1] * fx
    c11 = field[i0, j0 + 1, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0 + 1] * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz


@wp.kernel
def _mirror_sdf_into_colliders(
    field: wp.array3d[wp.float32],
    field_orig: wp.array3d[wp.float32],
    grid_origin: wp.vec3,
    voxel_size: float,
    onset: float,
    max_depth: float,
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
):
    i, j, k = wp.tid()
    x = grid_origin + voxel_size * wp.vec3(float(i), float(j), float(k))

    d_coll, n_coll, _vel, _coll_id, _material_id = collision_sdf(x, collider, body_q, body_qd, body_q_prev, 0.0)

    depth = onset - d_coll
    if depth < 0.0 or depth > max_depth:
        return

    x_mirror = x + 2.0 * depth * n_coll
    phi_mirror = _sample_field_trilinear(field_orig, grid_origin, 1.0 / voxel_size, x_mirror)

    blend_start = 0.5 * max_depth
    if depth <= blend_start:
        field[i, j, k] = phi_mirror
    else:
        t = (depth - blend_start) / wp.max(max_depth - blend_start, 1.0e-10)
        blend = t * t * (3.0 - 2.0 * t)
        field[i, j, k] = (1.0 - blend) * phi_mirror + blend * field_orig[i, j, k]


def extrapolate_surface_sdf_into_colliders(
    surface: ParticleSurface,
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector] = None,
    body_q_prev: wp.array[wp.transform] = None,
    *,
    max_depth: float | None = None,
    onset: float = 0.0,
    redistance: bool = True,
    compute_normals: bool = True,
) -> tuple[wp.array | None, wp.array | None, wp.array | None]:
    """Mirror-extrapolate a particle SDF into MPM collider interiors.

    This is intentionally an MPM adapter, not part of
    :class:`newton.geometry.ParticleSurface`, because it depends on the
    implicit MPM collider representation.
    """
    if surface.field_mode != "sdf":
        raise ValueError("Collider extrapolation requires ParticleSurface(field_mode='sdf')")
    if surface.field is None or surface.grid_dims is None:
        return None, None, None

    if max_depth is None:
        max_depth = _DEFAULT_COLLIDER_EXTRAPOLATION_DEPTH_SCALE * surface.voxel_size

    field_orig = wp.empty_like(surface.field)
    wp.copy(field_orig, surface.field)

    prev_query_dist = collider.query_max_dist
    collider.query_max_dist = max(prev_query_dist, max_depth + abs(onset) + surface.voxel_size)
    try:
        wp.launch(
            _mirror_sdf_into_colliders,
            dim=surface.grid_dims,
            inputs=[
                surface.field,
                field_orig,
                surface.grid_origin,
                surface.voxel_size,
                onset,
                max_depth,
                collider,
                body_q,
                body_qd,
                body_q_prev,
            ],
            device=surface.field.device,
        )
    finally:
        collider.query_max_dist = prev_query_dist

    if redistance:
        surface.redistance(iterations=max(surface.redistance_iterations, 1))
    return surface.resurface(compute_normals=compute_normals)
