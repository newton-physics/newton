# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle surface extraction using anisotropic kernels and marching cubes.

Implements the method from Yu & Turk, "Reconstructing Surfaces of
Particle-Based Fluids Using Anisotropic Kernels", Eurographics/ACM SIGGRAPH
Symposium on Computer Animation, 2010.

The pipeline computes per-particle anisotropy matrices via Weighted PCA,
then evaluates a smooth scalar field on a regular grid using oriented
ellipsoidal kernels, and extracts the isosurface with
:class:`warp.MarchingCubes`.

Typical usage::

    surface_ctx = ParticleSurface(voxel_size=0.01)
    verts, indices, normals = surface_ctx.extract(
        state.particle_q,
        model.particle_radius,
    )
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import warp as wp
import warp.fem as fem

from ..utils.mesh import compute_vertex_normals
from .flags import ParticleFlags

__all__ = ["ParticleSurface", "extract_particle_surface"]

wp.set_module_options({"enable_backward": False})

_DENSITY_KERNEL_SUPPORT = 2.0
_MESH_SMOOTH_SHRINK_PER_VOXEL = 0.15
_MIN_DENSITY_MARCHING_THRESHOLD = 0.01
_REDUCTION_TILE_SIZE = 128


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_aabb(
    positions: wp.array[wp.vec3],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    i = wp.tid()
    p = positions[i]
    wp.atomic_min(lower, 0, p)
    wp.atomic_max(upper, 0, p)


@wp.kernel
def _compute_aabb_tiled(
    positions: wp.array[wp.vec3],
    n: int,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    tile, lane = wp.tid()
    tile_size = wp.static(_REDUCTION_TILE_SIZE)
    i = tile * tile_size + lane
    valid = i < n

    min_x = wp.tile_full(shape=tile_size, value=1.0e30, dtype=float, storage="shared")
    min_y = wp.tile_full(shape=tile_size, value=1.0e30, dtype=float, storage="shared")
    min_z = wp.tile_full(shape=tile_size, value=1.0e30, dtype=float, storage="shared")
    max_x = wp.tile_full(shape=tile_size, value=-1.0e30, dtype=float, storage="shared")
    max_y = wp.tile_full(shape=tile_size, value=-1.0e30, dtype=float, storage="shared")
    max_z = wp.tile_full(shape=tile_size, value=-1.0e30, dtype=float, storage="shared")

    p = wp.vec3(0.0)
    if valid:
        p = positions[i]

    wp.tile_scatter_masked(min_x, lane, p[0], valid)
    wp.tile_scatter_masked(min_y, lane, p[1], valid)
    wp.tile_scatter_masked(min_z, lane, p[2], valid)
    wp.tile_scatter_masked(max_x, lane, p[0], valid)
    wp.tile_scatter_masked(max_y, lane, p[1], valid)
    wp.tile_scatter_masked(max_z, lane, p[2], valid)

    tile_min_x = wp.tile_min(min_x)
    tile_min_y = wp.tile_min(min_y)
    tile_min_z = wp.tile_min(min_z)
    tile_max_x = wp.tile_max(max_x)
    tile_max_y = wp.tile_max(max_y)
    tile_max_z = wp.tile_max(max_z)

    if lane == 0:
        wp.atomic_min(lower, 0, wp.vec3(tile_min_x[0], tile_min_y[0], tile_min_z[0]))
        wp.atomic_max(upper, 0, wp.vec3(tile_max_x[0], tile_max_y[0], tile_max_z[0]))


@wp.func
def _kernel_reach(
    i: int,
    radii: wp.array[float],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    particle_sdf_radius_scale: float,
    particle_sdf_band: float,
    particle_sdf: int,
    anisotropic: int,
) -> wp.vec3:
    reach = density_reach[i]
    if particle_sdf != 0 and anisotropic == 0:
        r = radii[i] * particle_sdf_radius_scale * particle_sdf_band
        reach = wp.vec3(r)
    elif particle_sdf != 0:
        r = wp.max(radii[i] * particle_sdf_radius_scale, 1.0e-8)
        det_root = wp.pow(wp.max(det_G[i], 1.0e-24), 1.0 / 3.0)
        reach = density_reach[i] * (0.5 * particle_sdf_band * det_root * r)
    return reach


@wp.kernel
def _compute_kernel_aabb(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    particle_sdf_radius_scale: float,
    particle_sdf_band: float,
    particle_sdf: int,
    anisotropic: int,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    """Compute exact world-space bounds of particle kernel supports."""
    i = wp.tid()
    p = positions[i]
    reach = _kernel_reach(
        i,
        radii,
        det_G,
        density_reach,
        particle_sdf_radius_scale,
        particle_sdf_band,
        particle_sdf,
        anisotropic,
    )

    wp.atomic_min(lower, 0, p - reach)
    wp.atomic_max(upper, 0, p + reach)


@wp.kernel
def _compute_kernel_aabb_tiled(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    particle_sdf_radius_scale: float,
    particle_sdf_band: float,
    particle_sdf: int,
    anisotropic: int,
    n: int,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    tile, lane = wp.tid()
    tile_size = wp.static(_REDUCTION_TILE_SIZE)
    i = tile * tile_size + lane
    valid = i < n

    min_x = wp.tile_full(shape=tile_size, value=1.0e30, dtype=float, storage="shared")
    min_y = wp.tile_full(shape=tile_size, value=1.0e30, dtype=float, storage="shared")
    min_z = wp.tile_full(shape=tile_size, value=1.0e30, dtype=float, storage="shared")
    max_x = wp.tile_full(shape=tile_size, value=-1.0e30, dtype=float, storage="shared")
    max_y = wp.tile_full(shape=tile_size, value=-1.0e30, dtype=float, storage="shared")
    max_z = wp.tile_full(shape=tile_size, value=-1.0e30, dtype=float, storage="shared")

    lo = wp.vec3(0.0)
    hi = wp.vec3(0.0)
    if valid:
        p = positions[i]
        reach = _kernel_reach(
            i,
            radii,
            det_G,
            density_reach,
            particle_sdf_radius_scale,
            particle_sdf_band,
            particle_sdf,
            anisotropic,
        )
        lo = p - reach
        hi = p + reach

    wp.tile_scatter_masked(min_x, lane, lo[0], valid)
    wp.tile_scatter_masked(min_y, lane, lo[1], valid)
    wp.tile_scatter_masked(min_z, lane, lo[2], valid)
    wp.tile_scatter_masked(max_x, lane, hi[0], valid)
    wp.tile_scatter_masked(max_y, lane, hi[1], valid)
    wp.tile_scatter_masked(max_z, lane, hi[2], valid)

    tile_min_x = wp.tile_min(min_x)
    tile_min_y = wp.tile_min(min_y)
    tile_min_z = wp.tile_min(min_z)
    tile_max_x = wp.tile_max(max_x)
    tile_max_y = wp.tile_max(max_y)
    tile_max_z = wp.tile_max(max_z)
    if lane == 0:
        wp.atomic_min(lower, 0, wp.vec3(tile_min_x[0], tile_min_y[0], tile_min_z[0]))
        wp.atomic_max(upper, 0, wp.vec3(tile_max_x[0], tile_max_y[0], tile_max_z[0]))


@wp.kernel
def _compute_max_radius(
    radii: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    wp.atomic_max(out, 0, radii[i])


@wp.kernel
def _compute_max_radius_tiled(
    radii: wp.array[float],
    n: int,
    out: wp.array[float],
):
    tile, lane = wp.tid()
    tile_size = wp.static(_REDUCTION_TILE_SIZE)
    i = tile * tile_size + lane

    values = wp.tile_full(shape=tile_size, value=0.0, dtype=float, storage="shared")
    valid = i < n
    value = float(0.0)
    if valid:
        value = radii[i]
    wp.tile_scatter_masked(values, lane, value, valid)

    tile_max_value = wp.tile_max(values)
    if lane == 0:
        wp.atomic_max(out, 0, tile_max_value[0])


@wp.kernel
def _build_active_particle_mask(
    flags: wp.array[wp.int32],
    mask: wp.array[wp.int32],
):
    i = wp.tid()
    if (flags[i] & ParticleFlags.ACTIVE) != wp.int32(0):
        mask[i] = wp.int32(1)
    else:
        mask[i] = wp.int32(0)


@wp.kernel
def _compact_active_particles(
    src: wp.array[Any],
    mask: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    dst: wp.array[Any],
):
    i = wp.tid()
    if mask[i] == wp.int32(1):
        dst[offsets[i]] = src[i]


@wp.kernel
def _blur_axis_x(src: wp.array3d[wp.float32], dst: wp.array3d[wp.float32], weights: wp.array[float], hw: int):
    i, j, k = wp.tid()
    val = src[i, j, k] * weights[0]
    if i >= hw and i + hw < src.shape[0]:
        for di in range(1, hw + 1):
            val += (src[i - di, j, k] + src[i + di, j, k]) * weights[di]
    else:
        for di in range(1, hw + 1):
            i_lo = wp.max(i - di, 0)
            i_hi = wp.min(i + di, src.shape[0] - 1)
            val += (src[i_lo, j, k] + src[i_hi, j, k]) * weights[di]
    dst[i, j, k] = val


@wp.kernel
def _blur_axis_y(src: wp.array3d[wp.float32], dst: wp.array3d[wp.float32], weights: wp.array[float], hw: int):
    i, j, k = wp.tid()
    val = src[i, j, k] * weights[0]
    if j >= hw and j + hw < src.shape[1]:
        for dj in range(1, hw + 1):
            val += (src[i, j - dj, k] + src[i, j + dj, k]) * weights[dj]
    else:
        for dj in range(1, hw + 1):
            j_lo = wp.max(j - dj, 0)
            j_hi = wp.min(j + dj, src.shape[1] - 1)
            val += (src[i, j_lo, k] + src[i, j_hi, k]) * weights[dj]
    dst[i, j, k] = val


@wp.kernel
def _blur_axis_z(src: wp.array3d[wp.float32], dst: wp.array3d[wp.float32], weights: wp.array[float], hw: int):
    i, j, k = wp.tid()
    val = src[i, j, k] * weights[0]
    if k >= hw and k + hw < src.shape[2]:
        for dk in range(1, hw + 1):
            val += (src[i, j, k - dk] + src[i, j, k + dk]) * weights[dk]
    else:
        for dk in range(1, hw + 1):
            k_lo = wp.max(k - dk, 0)
            k_hi = wp.min(k + dk, src.shape[2] - 1)
            val += (src[i, j, k_lo] + src[i, j, k_hi]) * weights[dk]
    dst[i, j, k] = val


@wp.kernel
def _fill_field(field: wp.array3d[wp.float32], value: float):
    i, j, k = wp.tid()
    field[i, j, k] = value


@wp.func
def _is_active_particle(flags: wp.array[wp.int32], use_flags: int, i: int) -> bool:
    if use_flags != 0:
        return (flags[i] & ParticleFlags.ACTIVE) != wp.int32(0)
    return True


@wp.func
def _weight_squared(dist_sq: float, inv_radius_sq: float) -> float:
    """Cubic falloff weight: w = (1 - (d/r)^3) for d < r."""
    q_sq = dist_sq * inv_radius_sq
    if q_sq >= 1.0:
        return 0.0
    return 1.0 - q_sq * wp.sqrt(q_sq)


@wp.func
def _cubic_bspline(q: float) -> float:
    """Cubic B-spline kernel with compact support at q=2."""
    if q < 1.0:
        return 1.0 - 1.5 * q * q + 0.75 * q * q * q
    elif q < 2.0:
        t = 2.0 - q
        return 0.25 * t * t * t
    return 0.0


# ---------------------------------------------------------------------------
# Pass 1: Smooth particle centers (Eq. 6 of Yu & Turk 2010)
# ---------------------------------------------------------------------------


@wp.kernel
def _copy_active_or_sentinel_positions(
    positions: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    use_flags: int,
    inactive_position: wp.vec3,
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    if _is_active_particle(flags, use_flags, i):
        out[i] = positions[i]
    else:
        out[i] = inactive_position


@wp.kernel
def _smooth_positions(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    search_radius: float,
    smooth_lambda: float,
    smoothed: wp.array[wp.vec3],
):
    i = wp.tid()
    xi = positions[i]
    offset_sum = wp.vec3(0.0)
    w_sum = float(0.0)
    radius_sq = search_radius * search_radius
    inv_radius_sq = 1.0 / radius_sq

    query = wp.hash_grid_query(grid, xi, search_radius)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        offset = positions[idx] - xi
        dist_sq = wp.dot(offset, offset)
        if dist_sq < radius_sq:
            w = _weight_squared(dist_sq, inv_radius_sq)
            offset_sum += w * offset
            w_sum += w

    if w_sum > 0.0:
        smoothed[i] = xi + (smooth_lambda / w_sum) * offset_sum
    else:
        smoothed[i] = xi


@wp.kernel
def _smooth_positions_flagged(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    inactive_position: wp.vec3,
    search_radius: float,
    smooth_lambda: float,
    smoothed: wp.array[wp.vec3],
):
    i = wp.tid()
    if not _is_active_particle(flags, 1, i):
        smoothed[i] = inactive_position
        return

    xi = positions[i]
    offset_sum = wp.vec3(0.0)
    w_sum = float(0.0)
    radius_sq = search_radius * search_radius
    inv_radius_sq = 1.0 / radius_sq

    query = wp.hash_grid_query(grid, xi, search_radius)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        if _is_active_particle(flags, 1, idx):
            offset = positions[idx] - xi
            dist_sq = wp.dot(offset, offset)
            if dist_sq < radius_sq:
                w = _weight_squared(dist_sq, inv_radius_sq)
                offset_sum += w * offset
                w_sum += w

    if w_sum > 0.0:
        smoothed[i] = xi + (smooth_lambda / w_sum) * offset_sum
    else:
        smoothed[i] = xi


# ---------------------------------------------------------------------------
# Pass 2: Per-particle anisotropy via Weighted PCA (Eqs. 9-16)
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_anisotropy(
    grid: wp.uint64,
    smoothed: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    use_flags: int,
    search_radius: float,
    anisotropy_ratio: float,
    anisotropy_scale: float,
    kernel_scale: float,
    anisotropy_min_neighbors: int,
    anisotropy_strength: float,
    G_out: wp.array[wp.mat33],
    det_G_out: wp.array[float],
    density_reach_out: wp.array[wp.vec3],
):
    i = wp.tid()
    xi = smoothed[i]
    h = search_radius

    if not _is_active_particle(flags, use_flags, i):
        inv_h = 1.0 / h
        G_out[i] = wp.identity(n=3, dtype=float) * inv_h
        det_G_out[i] = inv_h * inv_h * inv_h
        density_reach_out[i] = wp.vec3(_DENSITY_KERNEL_SUPPORT * h)
        return

    mean_offset = wp.vec3(0.0)
    moment_xx = float(0.0)
    moment_xy = float(0.0)
    moment_xz = float(0.0)
    moment_yy = float(0.0)
    moment_yz = float(0.0)
    moment_zz = float(0.0)
    w_sum = float(0.0)
    count = int(0)
    inv_radius_sq = 1.0 / (search_radius * search_radius)

    query = wp.hash_grid_query(grid, xi, search_radius)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        if _is_active_particle(flags, use_flags, idx):
            offset = smoothed[idx] - xi
            w = _weight_squared(wp.dot(offset, offset), inv_radius_sq)
            mean_offset += w * offset
            moment_xx += w * offset[0] * offset[0]
            moment_xy += w * offset[0] * offset[1]
            moment_xz += w * offset[0] * offset[2]
            moment_yy += w * offset[1] * offset[1]
            moment_yz += w * offset[1] * offset[2]
            moment_zz += w * offset[2] * offset[2]
            w_sum += w
            if w > 0.0:
                count += 1

    inv_h = 1.0 / h
    G = wp.identity(n=3, dtype=float) * inv_h
    det_g = inv_h * inv_h * inv_h

    # ``count`` includes the particle itself, so this requires at least
    # ``anisotropy_min_neighbors`` other particles.
    if count > anisotropy_min_neighbors and w_sum > 0.0:
        mean_offset = mean_offset / w_sum
        inv_w_sum = 1.0 / w_sum
        C = wp.mat33(
            moment_xx * inv_w_sum - mean_offset[0] * mean_offset[0],
            moment_xy * inv_w_sum - mean_offset[0] * mean_offset[1],
            moment_xz * inv_w_sum - mean_offset[0] * mean_offset[2],
            moment_xy * inv_w_sum - mean_offset[0] * mean_offset[1],
            moment_yy * inv_w_sum - mean_offset[1] * mean_offset[1],
            moment_yz * inv_w_sum - mean_offset[1] * mean_offset[2],
            moment_xz * inv_w_sum - mean_offset[0] * mean_offset[2],
            moment_yz * inv_w_sum - mean_offset[1] * mean_offset[2],
            moment_zz * inv_w_sum - mean_offset[2] * mean_offset[2],
        )

        U, sigma, _V = wp.svd3(C)

        # Erratum fix: covariance eigenvalues are variances; sqrt for axis lengths
        s1 = wp.sqrt(wp.max(sigma[0], 1.0e-10))
        s2 = wp.sqrt(wp.max(sigma[1], 1.0e-10))
        s3 = wp.sqrt(wp.max(sigma[2], 1.0e-10))

        # Clamp minimum eigenvalue ratio (Eq. 14)
        s2 = wp.max(s2, s1 / anisotropy_ratio)
        s3 = wp.max(s3, s1 / anisotropy_ratio)

        # Match the geometric-mean radius of anisotropic kernels to the
        # isotropic kernel scale, then apply an optional relative multiplier.
        s_geo = wp.pow(s1 * s2 * s3, 1.0 / 3.0)
        anisotropic_axis_scale = kernel_scale * anisotropy_scale / wp.max(s_geo, 1.0e-10)

        inv_s1 = 1.0 / (anisotropic_axis_scale * s1)
        inv_s2 = 1.0 / (anisotropic_axis_scale * s2)
        inv_s3 = 1.0 / (anisotropic_axis_scale * s3)

        # Blend inverse kernel radii in the PCA frame.  Full strength
        # preserves the WPCA ellipsoid for every particle with enough
        # neighbors; sparse particles still use the isotropic fallback below.
        blend = anisotropy_strength

        iso_scale = 1.0 / (kernel_scale * h)
        if blend <= 0.0:
            G = wp.identity(n=3, dtype=float) * iso_scale
            det_g = iso_scale * iso_scale * iso_scale
        else:
            g1 = (1.0 - blend) * iso_scale + blend * inv_h * inv_s1
            g2 = (1.0 - blend) * iso_scale + blend * inv_h * inv_s2
            g3 = (1.0 - blend) * iso_scale + blend * inv_h * inv_s3
            G = U @ wp.diag(wp.vec3(g1, g2, g3)) @ wp.transpose(U)
            det_g = g1 * g2 * g3
    elif count <= anisotropy_min_neighbors and count > 0:
        scale = 1.0 / (kernel_scale * h)
        G = wp.identity(n=3, dtype=float) * scale
        det_g = scale * scale * scale

    G_out[i] = G
    det_G_out[i] = det_g
    G_inv = wp.inverse(G)
    density_reach_out[i] = _DENSITY_KERNEL_SUPPORT * wp.vec3(
        wp.length(wp.vec3(G_inv[0, 0], G_inv[1, 0], G_inv[2, 0])),
        wp.length(wp.vec3(G_inv[0, 1], G_inv[1, 1], G_inv[2, 1])),
        wp.length(wp.vec3(G_inv[0, 2], G_inv[1, 2], G_inv[2, 2])),
    )


@wp.kernel
def _fill_isotropic_G(
    kernel_radius: float,
    kernel_scale: float,
    G_out: wp.array[wp.mat33],
    det_G_out: wp.array[float],
    density_reach_out: wp.array[wp.vec3],
):
    """Fill all particles with the same isotropic G = (1/(kernel_scale*h)) * I."""
    i = wp.tid()
    scale = 1.0 / (kernel_scale * kernel_radius)
    G_out[i] = wp.identity(n=3, dtype=float) * scale
    det_G_out[i] = scale * scale * scale
    density_reach_out[i] = wp.vec3(_DENSITY_KERNEL_SUPPORT / scale)


# ---------------------------------------------------------------------------
# Pass 3: Scalar field evaluation (Eq. 8)
# ---------------------------------------------------------------------------


@wp.kernel
def _eval_scalar_field(
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    G_matrices: wp.array[wp.mat33],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    grid_origin: wp.vec3,
    inv_voxel_size: float,
    nx: int,
    ny: int,
    nz: int,
    field: wp.array3d[wp.float32],
):
    """Particle-centric scalar field evaluation.

    Each particle splatts its contribution onto nearby grid nodes using
    atomic adds, avoiding hash-grid queries from grid nodes entirely.
    """
    pid = wp.tid()
    if not _is_active_particle(flags, use_flags, pid):
        return

    x_p = smoothed[pid]
    r_p = radii[pid]
    volume = 8.0 * r_p * r_p * r_p
    # SPH normalization: integral of P(||u||) over 3D = pi, so sigma = 1/pi
    sigma = wp.static(1.0 / math.pi)
    G = G_matrices[pid]
    dG = det_G[pid]
    weight = volume * sigma * dG

    # Axis-aligned bounding box of the kernel support (||G*r|| < 2).
    reach = density_reach[pid]
    reach_x = reach[0]
    reach_y = reach[1]
    reach_z = reach[2]

    lo_x = wp.max(int(wp.ceil((x_p[0] - reach_x - grid_origin[0]) * inv_voxel_size)), 0)
    lo_y = wp.max(int(wp.ceil((x_p[1] - reach_y - grid_origin[1]) * inv_voxel_size)), 0)
    lo_z = wp.max(int(wp.ceil((x_p[2] - reach_z - grid_origin[2]) * inv_voxel_size)), 0)
    hi_x = wp.min(int(wp.floor((x_p[0] + reach_x - grid_origin[0]) * inv_voxel_size)), nx - 1)
    hi_y = wp.min(int(wp.floor((x_p[1] + reach_y - grid_origin[1]) * inv_voxel_size)), ny - 1)
    hi_z = wp.min(int(wp.floor((x_p[2] + reach_z - grid_origin[2]) * inv_voxel_size)), nz - 1)

    voxel_size = 1.0 / inv_voxel_size
    d_lo = grid_origin + voxel_size * wp.vec3(float(lo_x), float(lo_y), float(lo_z)) - x_p
    Gd_i = G * d_lo
    step_x = voxel_size * wp.vec3(G[0, 0], G[1, 0], G[2, 0])
    step_y = voxel_size * wp.vec3(G[0, 1], G[1, 1], G[2, 1])
    step_z = voxel_size * wp.vec3(G[0, 2], G[1, 2], G[2, 2])
    for i in range(lo_x, hi_x + 1):
        Gd_j = Gd_i
        for j in range(lo_y, hi_y + 1):
            Gd = Gd_j
            for k in range(lo_z, hi_z + 1):
                q_sq = wp.dot(Gd, Gd)
                if q_sq < 4.0:
                    wp.atomic_add(field, i, j, k, weight * _cubic_bspline(wp.sqrt(q_sq)))
                Gd += step_z
            Gd_j += step_y
        Gd_i += step_x


@wp.kernel
def _eval_particle_sdf_union_isotropic(
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    sdf_radius_scale: float,
    sdf_band: float,
    grid_origin: wp.vec3,
    inv_voxel_size: float,
    nx: int,
    ny: int,
    nz: int,
    field: wp.array3d[wp.float32],
):
    """Particle-centric spherical SDF union."""
    pid = wp.tid()
    if not _is_active_particle(flags, use_flags, pid):
        return

    x_p = smoothed[pid]
    r_p = wp.max(radii[pid] * sdf_radius_scale, 1.0e-8)
    reach = sdf_band * r_p

    lo_x = wp.max(int(wp.ceil((x_p[0] - reach - grid_origin[0]) * inv_voxel_size)), 0)
    lo_y = wp.max(int(wp.ceil((x_p[1] - reach - grid_origin[1]) * inv_voxel_size)), 0)
    lo_z = wp.max(int(wp.ceil((x_p[2] - reach - grid_origin[2]) * inv_voxel_size)), 0)
    hi_x = wp.min(int(wp.floor((x_p[0] + reach - grid_origin[0]) * inv_voxel_size)), nx - 1)
    hi_y = wp.min(int(wp.floor((x_p[1] + reach - grid_origin[1]) * inv_voxel_size)), ny - 1)
    hi_z = wp.min(int(wp.floor((x_p[2] + reach - grid_origin[2]) * inv_voxel_size)), nz - 1)

    voxel_size = 1.0 / inv_voxel_size
    reach_sq = reach * reach
    dx = grid_origin[0] + voxel_size * float(lo_x) - x_p[0]
    for i in range(lo_x, hi_x + 1):
        dy = grid_origin[1] + voxel_size * float(lo_y) - x_p[1]
        dx_sq = dx * dx
        for j in range(lo_y, hi_y + 1):
            dz = grid_origin[2] + voxel_size * float(lo_z) - x_p[2]
            dxy_sq = dx_sq + dy * dy
            for k in range(lo_z, hi_z + 1):
                dist_sq = dxy_sq + dz * dz
                if dist_sq <= reach_sq:
                    wp.atomic_min(field, i, j, k, wp.sqrt(dist_sq) - r_p)
                dz += voxel_size
            dy += voxel_size
        dx += voxel_size


@wp.kernel
def _eval_particle_sdf_union_anisotropic(
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    G_matrices: wp.array[wp.mat33],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    sdf_radius_scale: float,
    sdf_band: float,
    grid_origin: wp.vec3,
    inv_voxel_size: float,
    nx: int,
    ny: int,
    nz: int,
    field: wp.array3d[wp.float32],
):
    """Particle-centric anisotropic ellipsoid SDF union."""
    pid = wp.tid()
    if not _is_active_particle(flags, use_flags, pid):
        return

    x_p = smoothed[pid]
    r_p = wp.max(radii[pid] * sdf_radius_scale, 1.0e-8)

    # Normalize the existing anisotropy matrix to an ellipsoid with the
    # particle radius as its geometric-mean radius. This preserves the WPCA
    # axis ratios and orientation without inheriting density-kernel volume.
    G = G_matrices[pid]
    det_root = wp.pow(wp.max(det_G[pid], 1.0e-24), 1.0 / 3.0)
    radius_normalization = det_root * r_p
    H = G * (1.0 / radius_normalization)

    reach = density_reach[pid] * (0.5 * sdf_band * radius_normalization)
    reach_x = reach[0]
    reach_y = reach[1]
    reach_z = reach[2]

    lo_x = wp.max(int(wp.ceil((x_p[0] - reach_x - grid_origin[0]) * inv_voxel_size)), 0)
    lo_y = wp.max(int(wp.ceil((x_p[1] - reach_y - grid_origin[1]) * inv_voxel_size)), 0)
    lo_z = wp.max(int(wp.ceil((x_p[2] - reach_z - grid_origin[2]) * inv_voxel_size)), 0)
    hi_x = wp.min(int(wp.floor((x_p[0] + reach_x - grid_origin[0]) * inv_voxel_size)), nx - 1)
    hi_y = wp.min(int(wp.floor((x_p[1] + reach_y - grid_origin[1]) * inv_voxel_size)), ny - 1)
    hi_z = wp.min(int(wp.floor((x_p[2] + reach_z - grid_origin[2]) * inv_voxel_size)), nz - 1)

    voxel_size = 1.0 / inv_voxel_size
    d_lo = grid_origin + voxel_size * wp.vec3(float(lo_x), float(lo_y), float(lo_z)) - x_p
    Hd_i = H * d_lo
    step_x = voxel_size * wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    step_y = voxel_size * wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    step_z = voxel_size * wp.vec3(H[0, 2], H[1, 2], H[2, 2])
    H_t = wp.transpose(H)
    band_sq = sdf_band * sdf_band
    for i in range(lo_x, hi_x + 1):
        Hd_j = Hd_i
        for j in range(lo_y, hi_y + 1):
            Hd = Hd_j
            for k in range(lo_z, hi_z + 1):
                q_sq = wp.dot(Hd, Hd)
                if q_sq <= band_sq:
                    sdf = -r_p
                    if q_sq > 1.0e-16:
                        q = wp.sqrt(q_sq)
                        grad_norm_times_q = wp.length(H_t * Hd)
                        sdf = (q - 1.0) * q / wp.max(grad_norm_times_q, 1.0e-8 * q)
                    wp.atomic_min(field, i, j, k, sdf)
                Hd += step_z
            Hd_j += step_y
        Hd_i += step_x


@wp.kernel
def _flip_winding(indices: wp.array[wp.int32]):
    """Swap first and second vertex of each triangle to flip face normals."""
    tid = wp.tid()
    base = tid * 3
    tmp = indices[base]
    indices[base] = indices[base + 1]
    indices[base + 1] = tmp


# ---------------------------------------------------------------------------
# Mesh smoothing kernels (Laplacian)
# ---------------------------------------------------------------------------


@wp.kernel
def _laplacian_scatter(
    indices: wp.array[wp.int32],
    verts: wp.array[wp.vec3],
    neighbor_sum: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
):
    tid = wp.tid()
    tri = tid // 3
    local = tid - tri * 3
    base = tri * 3

    i0 = indices[base + local]
    i1 = indices[base + (local + 1) % 3]
    i2 = indices[base + (local + 2) % 3]

    wp.atomic_add(neighbor_sum, i0, verts[i1] + verts[i2])
    wp.atomic_add(valence, i0, 2)


@wp.kernel
def _laplacian_apply(
    verts: wp.array[wp.vec3],
    neighbor_sum: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
    smoothed: wp.array[wp.vec3],
    factor: float,
):
    i = wp.tid()
    v = valence[i]
    if v > 0:
        avg = neighbor_sum[i] / float(v)
        smoothed[i] = verts[i] + factor * (avg - verts[i])
    else:
        smoothed[i] = verts[i]


# ---------------------------------------------------------------------------
# Density to SDF conversion and redistancing
# ---------------------------------------------------------------------------


@wp.kernel
def _density_to_sdf_3d(
    field: wp.array3d[wp.float32],
    threshold: float,
):
    """Convert density field to SDF in-place: sdf = threshold - density."""
    i, j, k = wp.tid()
    field[i, j, k] = threshold - field[i, j, k]


@wp.kernel
def _redistance_step(
    sdf: wp.array3d[wp.float32],
    sdf_out: wp.array3d[wp.float32],
    inv_dx: float,
):
    """One step of Eikonal redistancing with a Godunov upwind scheme."""
    i, j, k = wp.tid()
    nx = sdf.shape[0]
    ny = sdf.shape[1]
    nz = sdf.shape[2]

    d = sdf[i, j, k]
    s = wp.sign(d)

    dx_m = sdf[wp.max(i - 1, 0), j, k]
    dx_p = sdf[wp.min(i + 1, nx - 1), j, k]
    dy_m = sdf[i, wp.max(j - 1, 0), k]
    dy_p = sdf[i, wp.min(j + 1, ny - 1), k]
    dz_m = sdf[i, j, wp.max(k - 1, 0)]
    dz_p = sdf[i, j, wp.min(k + 1, nz - 1)]

    ax = wp.max(wp.max(s * (d - dx_m), 0.0), wp.max(-s * (dx_p - d), 0.0)) * inv_dx
    ay = wp.max(wp.max(s * (d - dy_m), 0.0), wp.max(-s * (dy_p - d), 0.0)) * inv_dx
    az = wp.max(wp.max(s * (d - dz_m), 0.0), wp.max(-s * (dz_p - d), 0.0)) * inv_dx

    grad_mag = wp.sqrt(ax * ax + ay * ay + az * az)

    dx_sq = 1.0 / (inv_dx * inv_dx)
    s_smooth = d / wp.sqrt(d * d + grad_mag * grad_mag * dx_sq + 1.0e-20)

    dt = 0.5 / inv_dx
    sdf_out[i, j, k] = d - dt * s_smooth * (grad_mag - 1.0)


@wp.kernel
def _validate_positive_finite(values: wp.array[float], invalid: wp.array[wp.int32]):
    i = wp.tid()
    value = values[i]
    if value <= 0.0 or not wp.isfinite(value):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _validate_positive_finite_tiled(values: wp.array[float], n: int, invalid: wp.array[wp.int32]):
    tile, lane = wp.tid()
    tile_size = wp.static(_REDUCTION_TILE_SIZE)
    i = tile * tile_size + lane

    flags = wp.tile_full(shape=tile_size, value=wp.int32(0), dtype=wp.int32, storage="shared")
    valid = i < n
    flag = wp.int32(0)
    if valid:
        value = values[i]
        if value <= 0.0 or not wp.isfinite(value):
            flag = wp.int32(1)
    wp.tile_scatter_masked(flags, lane, flag, valid)

    tile_invalid = wp.tile_max(flags)
    if lane == 0 and tile_invalid[0] != wp.int32(0):
        wp.atomic_max(invalid, 0, 1)


def _reduction_tile_dim(n: int) -> int:
    return max(1, (n + _REDUCTION_TILE_SIZE - 1) // _REDUCTION_TILE_SIZE)


def _use_cuda_tile_kernels(device: wp.DeviceLike) -> bool:
    # Tile kernels are CUDA optimizations; keep scalar kernels on CPU.
    return wp.get_device(device).is_cuda


# ---------------------------------------------------------------------------
# ParticleSurface context
# ---------------------------------------------------------------------------


class ParticleSurface:
    """Reusable context for extracting a triangle mesh from particle data.

    Uses the Yu & Turk (2010) anisotropic kernel method: per-particle
    Weighted PCA determines oriented ellipsoidal kernels that produce a
    smooth scalar field whose isosurface tightly wraps the particles.

    Args:
        voxel_size: Edge length of each grid voxel [m].
        kernel_radius: Search radius for neighbor queries [m].
            Defaults to ``3 * voxel_size``.
        threshold: Isosurface level for marching cubes.  The scalar field
            is approximately 1.0 inside dense particle regions.  Defaults to
            0.25.
        smooth_lambda: Blending factor for position smoothing [0, 1].
            Higher values produce smoother surfaces.  Defaults to 0.5.
        anisotropic: Enable per-particle WPCA anisotropic kernels.
            When ``False`` (default), all particles use isotropic kernels.
        anisotropy_ratio: Maximum anisotropic kernel axis ratio.  Higher values
            allow flatter ellipsoids.
        kernel_scale: Kernel radius multiplier relative to ``kernel_radius``.
            This sets the isotropic kernel radius and the geometric-mean
            radius of anisotropic kernels.
        anisotropy_scale: Relative multiplier for anisotropic kernel radii.
            Values greater than 1 widen anisotropic kernels without changing
            the isotropic fallback scale.  Defaults to 1.
        anisotropy_min_neighbors: Minimum number of other particles required
            for anisotropic kernels. Sparser particles use isotropic kernels.
        anisotropy_strength: Blend from isotropic kernels to anisotropic
            kernels [0, 1].  Lower values preserve more normal support from
            boundary particles back into the interior.
        surface_method: Surface reconstruction method. ``"density"`` uses
            anisotropic density splatting. ``"particle_sdf"`` directly unions
            per-particle anisotropic ellipsoid SDFs and stores an SDF field.
        particle_sdf_radius_scale: Radius multiplier for ``surface_method="particle_sdf"``.
        particle_sdf_band: Narrow-band half-width in normalized ellipsoid
            coordinates for ``surface_method="particle_sdf"``.
        padding: Extra voxels added around the particle bounding box.
        field_smooth_iterations: Number of separable Gaussian blur passes
            applied to the scalar field before marching cubes.  Defaults to
            0.
        field_smooth_radius: Half-width of the Gaussian blur in voxels.
            Defaults to 1.
        field_mode: Field representation retained after extraction.
            ``"density"`` keeps the scalar density field used by marching
            cubes.  ``"sdf"`` converts it to a signed distance approximation
            with negative values inside the particle surface.  Defaults to
            ``"sdf"`` for ``surface_method="particle_sdf"`` and ``"density"``
            otherwise.
        redistance_iterations: Number of Eikonal redistancing iterations
            applied when ``field_mode="sdf"``.  Set to 0 to skip.
        mesh_smooth_iterations: Number of Laplacian smoothing passes
            applied to the extracted mesh.  Set to 0 to disable.
        mesh_smooth_lambda: Laplacian step size [0, 1].
        device: Warp device for computation.
    """

    def __init__(
        self,
        voxel_size: float,
        kernel_radius: float | None = None,
        threshold: float = 0.25,
        smooth_lambda: float = 0.5,
        anisotropic: bool = False,
        anisotropy_ratio: float = 4.0,
        kernel_scale: float = 0.5,
        anisotropy_scale: float = 1.0,
        anisotropy_min_neighbors: int = 25,
        padding: int = 2,
        field_smooth_iterations: int = 0,
        field_smooth_radius: int = 1,
        field_mode: Literal["density", "sdf"] | None = None,
        redistance_iterations: int = 0,
        mesh_smooth_iterations: int = 0,
        mesh_smooth_lambda: float = 1.0,
        device: wp.DeviceLike = None,
        anisotropy_strength: float = 1.0,
        surface_method: Literal["density", "particle_sdf"] = "density",
        particle_sdf_radius_scale: float = 1.0,
        particle_sdf_band: float = 2.0,
    ):
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if kernel_radius is None:
            kernel_radius = 3.0 * voxel_size
        elif not math.isfinite(kernel_radius) or kernel_radius <= 0.0:
            raise ValueError("kernel_radius must be positive")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if not 0.0 <= smooth_lambda <= 1.0:
            raise ValueError("smooth_lambda must be in [0, 1]")
        if not math.isfinite(anisotropy_ratio) or anisotropy_ratio < 1.0:
            raise ValueError("anisotropy_ratio must be at least 1")
        if not math.isfinite(kernel_scale) or kernel_scale <= 0.0:
            raise ValueError("kernel_scale must be positive")
        if not math.isfinite(anisotropy_scale) or anisotropy_scale <= 0.0:
            raise ValueError("anisotropy_scale must be positive")
        if anisotropy_min_neighbors < 0:
            raise ValueError("anisotropy_min_neighbors must be non-negative")
        if not 0.0 <= anisotropy_strength <= 1.0:
            raise ValueError("anisotropy_strength must be in [0, 1]")
        if padding < 0:
            raise ValueError("padding must be non-negative")
        if field_smooth_iterations < 0:
            raise ValueError("field_smooth_iterations must be non-negative")
        if field_smooth_radius < 0:
            raise ValueError("field_smooth_radius must be non-negative")
        if redistance_iterations < 0:
            raise ValueError("redistance_iterations must be non-negative")
        if mesh_smooth_iterations < 0:
            raise ValueError("mesh_smooth_iterations must be non-negative")
        if not 0.0 <= mesh_smooth_lambda <= 1.0:
            raise ValueError("mesh_smooth_lambda must be in [0, 1]")
        if not math.isfinite(particle_sdf_radius_scale) or particle_sdf_radius_scale <= 0.0:
            raise ValueError("particle_sdf_radius_scale must be positive")
        if not math.isfinite(particle_sdf_band) or particle_sdf_band <= 0.0:
            raise ValueError("particle_sdf_band must be positive")
        if surface_method not in ("density", "particle_sdf"):
            raise ValueError(f"Unsupported surface_method {surface_method!r}; expected 'density' or 'particle_sdf'")
        if field_mode is None:
            field_mode = "sdf" if surface_method == "particle_sdf" else "density"
        elif field_mode not in ("density", "sdf"):
            raise ValueError(f"Unsupported field_mode {field_mode!r}; expected 'density' or 'sdf'")
        if surface_method == "particle_sdf" and field_mode != "sdf":
            raise ValueError("surface_method='particle_sdf' requires field_mode='sdf' or field_mode=None")
        if redistance_iterations > 0 and field_mode != "sdf":
            raise ValueError("redistance_iterations requires field_mode='sdf'")

        self.voxel_size = voxel_size
        self.kernel_radius = kernel_radius
        self.anisotropic = anisotropic
        self.threshold = threshold
        self.smooth_lambda = smooth_lambda
        self.anisotropy_ratio = anisotropy_ratio
        self.anisotropy_scale = anisotropy_scale
        self.kernel_scale = kernel_scale
        self.anisotropy_min_neighbors = anisotropy_min_neighbors
        self.anisotropy_strength = anisotropy_strength
        self.surface_method = surface_method
        self.particle_sdf_radius_scale = particle_sdf_radius_scale
        self.particle_sdf_band = particle_sdf_band
        self.padding = padding
        self.field_smooth_iterations = field_smooth_iterations
        self.field_smooth_radius = field_smooth_radius
        self.field_mode = field_mode
        self.redistance_iterations = redistance_iterations
        self.mesh_smooth_iterations = mesh_smooth_iterations
        self.mesh_smooth_lambda = mesh_smooth_lambda

        self._device = wp.get_device() if device is None else wp.get_device(device)

        # Cached objects (allocated lazily)
        self._mc: wp.MarchingCubes | None = None
        self._hash_grid: wp.HashGrid | None = None
        self._blur_temp: wp.array3d[wp.float32] | None = None
        self._blur_weights: wp.array[float] | None = None
        self._field: wp.array3d[wp.float32] | None = None
        self._grid_dims: tuple[int, int, int] | None = None
        self._grid_origin: wp.vec3 | None = None
        self._inactive_position: wp.vec3 = wp.vec3(0.0)
        self._hash_grid_dim: int = 0
        self._resource_device: wp.Device | None = None

        # Per-particle temporaries
        self._smoothed: wp.array[wp.vec3] | None = None
        self._G: wp.array[wp.mat33] | None = None
        self._det_G: wp.array[float] | None = None
        self._density_reach: wp.array[wp.vec3] | None = None
        self._hash_positions: wp.array[wp.vec3] | None = None
        self._all_particle_flags: wp.array[wp.int32] | None = None
        self._n_particles: int = 0
        self._max_particles: int = 0

        # Last extraction results
        self._verts: wp.array[wp.vec3] | None = None
        self._indices: wp.array[wp.int32] | None = None
        self._normals: wp.array[wp.vec3] | None = None

    # -- Public properties --

    @property
    def verts(self) -> wp.array[wp.vec3] | None:
        """Vertex positions from the last extraction [m]."""
        return self._verts

    @property
    def indices(self) -> wp.array[wp.int32] | None:
        """Triangle indices from the last extraction, shape ``(3 * triangle_count,)``."""
        return self._indices

    @property
    def normals(self) -> wp.array[wp.vec3] | None:
        """Unit-length per-vertex normals from the last extraction."""
        return self._normals

    @property
    def field(self) -> wp.array3d[wp.float32] | None:
        """Scalar field, shape ``(nx, ny, nz)``; SDF samples use meters."""
        return self._field

    @property
    def grid_origin(self) -> wp.vec3 | None:
        """World-space position of grid node ``(0, 0, 0)`` [m]."""
        return self._grid_origin

    @property
    def grid_dims(self) -> tuple[int, int, int] | None:
        """Grid node counts ``(nx, ny, nz)``."""
        return self._grid_dims

    @property
    def smoothed_positions(self) -> wp.array[wp.vec3] | None:
        """Smoothed particle positions from the last extraction [m]."""
        return self._smoothed

    @property
    def anisotropy_matrices(self) -> wp.array[wp.mat33] | None:
        """Per-particle inverse kernel transforms ``G`` [1/m]."""
        return self._G

    @property
    def anisotropy_det(self) -> wp.array[float] | None:
        """Per-particle ``det(G)`` values [1/m³]."""
        return self._det_G

    def configure_field_grid(
        self,
        grid_origin: wp.vec3 | tuple[float, float, float],
        grid_dims: tuple[int, int, int],
        max_particles: int,
        device: wp.DeviceLike = None,
        hash_grid_dim: int | None = None,
    ) -> ParticleSurface:
        """Preallocate fixed scalar-field resources for CUDA graph capture.

        The configured grid is reused by :meth:`update_field`, avoiding the
        dynamic AABB, allocation, and host-readback work used by
        :meth:`extract`.  Call this once outside CUDA graph capture, then
        capture calls to :meth:`update_field` with particle arrays whose length
        does not exceed ``max_particles``.

        Args:
            grid_origin: World-space position of grid node ``(0, 0, 0)`` [m].
            grid_dims: Grid node counts ``(nx, ny, nz)``.
            max_particles: Maximum particle array length accepted by
                :meth:`update_field`.
            device: Warp device for preallocated resources.  Defaults to this
                context's current device.
            hash_grid_dim: Optional hash-grid resolution.  Defaults to a value
                derived from the fixed grid extent and ``kernel_radius``.

        Returns:
            This :class:`ParticleSurface` instance.
        """
        if len(grid_dims) != 3:
            raise ValueError(f"grid_dims must contain three entries, got {grid_dims}")
        nx, ny, nz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
        if nx < 2 or ny < 2 or nz < 2:
            raise ValueError(f"grid_dims entries must be at least 2, got {grid_dims}")
        if max_particles < 0:
            raise ValueError("max_particles must be non-negative")

        device_obj = self._device if device is None else wp.get_device(device)
        if self._resource_device != device_obj:
            self._clear_device_resources()
            self._resource_device = device_obj
        self._device = device_obj

        grid_origin = wp.vec3(grid_origin)
        grid_end = wp.vec3(
            grid_origin[0] + (nx - 1) * self.voxel_size,
            grid_origin[1] + (ny - 1) * self.voxel_size,
            grid_origin[2] + (nz - 1) * self.voxel_size,
        )
        self._grid_origin = grid_origin
        self._grid_dims = (nx, ny, nz)

        extent = max(
            float(grid_end[0] - grid_origin[0]),
            float(grid_end[1] - grid_origin[1]),
            float(grid_end[2] - grid_origin[2]),
        )
        if hash_grid_dim is None:
            hash_grid_dim = max(16, int(math.ceil(extent / self.kernel_radius)))
        elif hash_grid_dim <= 0:
            raise ValueError("hash_grid_dim must be positive")

        self._field = wp.empty((nx, ny, nz), dtype=wp.float32, device=device_obj)
        if (self.field_smooth_iterations > 0 and self.field_smooth_radius > 0) or self.redistance_iterations > 0:
            self._blur_temp = wp.empty((nx, ny, nz), dtype=wp.float32, device=device_obj)
        else:
            self._blur_temp = None
        if self.field_smooth_iterations > 0 and self.field_smooth_radius > 0:
            self._ensure_blur_weights(device_obj)

        self._mc = wp.MarchingCubes(nx, ny, nz)
        self._mc.domain_bounds_lower_corner = grid_origin
        self._mc.domain_bounds_upper_corner = grid_end

        if self._hash_grid is None or self._hash_grid_dim != hash_grid_dim:
            self._hash_grid = wp.HashGrid(hash_grid_dim, hash_grid_dim, hash_grid_dim, device=device_obj)
            self._hash_grid_dim = hash_grid_dim

        if self._max_particles != max_particles or self._smoothed is None:
            alloc_particles = max(max_particles, 1)
            self._smoothed = wp.empty(alloc_particles, dtype=wp.vec3, device=device_obj)
            self._G = wp.empty(alloc_particles, dtype=wp.mat33, device=device_obj)
            self._det_G = wp.empty(alloc_particles, dtype=float, device=device_obj)
            self._density_reach = wp.empty(alloc_particles, dtype=wp.vec3, device=device_obj)
            self._hash_positions = wp.empty(alloc_particles, dtype=wp.vec3, device=device_obj)
            self._all_particle_flags = wp.empty(alloc_particles, dtype=wp.int32, device=device_obj)
            self._n_particles = alloc_particles
            self._max_particles = max_particles

        sentinel_pad = max(extent + self.kernel_radius * 8.0, self.voxel_size)
        self._inactive_position = grid_origin - wp.vec3(sentinel_pad, sentinel_pad, sentinel_pad)
        return self

    def update_field(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        particle_flags: wp.array[wp.int32] | None = None,
    ) -> wp.array3d[wp.float32]:
        """Update the fixed scalar field without extracting a mesh.

        This method is intended for CUDA graph capture after
        :meth:`configure_field_grid` has preallocated fixed-capacity buffers.
        It does not resize the grid, compact particles, read counts back to
        the host, or allocate output mesh buffers.

        Args:
            positions: Particle positions [m], shape ``(N,)``, dtype ``wp.vec3``.
            radii: Per-particle radii [m], shape ``(N,)``, dtype ``wp.float32``.
            particle_flags: Optional per-particle flags.  Particles without
                :attr:`~newton.ParticleFlags.ACTIVE` are skipped without
                compacting the particle arrays.

        Returns:
            The updated scalar field, shape ``grid_dims``.
        """
        if self._field is None or self._grid_dims is None or self._grid_origin is None:
            raise RuntimeError("configure_field_grid() must be called before update_field()")

        self._validate_positions_layout(positions)
        n = positions.shape[0]
        device = positions.device
        if wp.get_device(device) != self._resource_device:
            raise ValueError(f"positions device ({device}) must match configured device ({self._resource_device})")
        if n > self._max_particles:
            raise ValueError(f"positions length ({n}) exceeds configured max_particles ({self._max_particles})")
        self._validate_radii_layout(positions, radii, n)
        self._validate_particle_flags_layout(particle_flags, n, device)

        self._clear_results(clear_field=False)
        self._update_field_values(positions, radii, n, particle_flags, device)
        return self._field

    def fem_field(self) -> fem.DiscreteField:
        """Return the scalar field as a :class:`warp.fem.DiscreteField`.

        The field lives on a Q1 (trilinear) function space over a
        :class:`warp.fem.Grid3D` matching the extraction grid.  It can be
        used directly with :func:`warp.fem.interpolate` or
        :func:`warp.fem.integrate` to evaluate smooth values, gradients,
        and curvature at arbitrary positions.  With ``field_mode="density"``,
        the values are the scalar density field.  With ``field_mode="sdf"``
        or ``surface_method="particle_sdf"``, negative values are inside the
        particle surface and positive values are outside.

        Must be called after :meth:`extract` or :meth:`update_field`.

        Returns:
            A :class:`warp.fem.DiscreteField` with scalar ``float`` DOFs.
        """
        if self._field is None or self._grid_dims is None or self._grid_origin is None:
            raise RuntimeError("extract() or update_field() must populate the field before fem_field()")

        nx, ny, nz = self._grid_dims
        grid = fem.Grid3D(
            bounds_lo=self._grid_origin,
            bounds_hi=wp.vec3(
                self._grid_origin[0] + (nx - 1) * self.voxel_size,
                self._grid_origin[1] + (ny - 1) * self.voxel_size,
                self._grid_origin[2] + (nz - 1) * self.voxel_size,
            ),
            res=wp.vec3i(nx - 1, ny - 1, nz - 1),
        )
        space = fem.make_polynomial_space(grid, degree=1, dtype=float)
        discrete_field = fem.make_discrete_field(space)
        discrete_field.dof_values = self._field.flatten()
        return discrete_field

    # -- Core extraction --

    def extract(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        compute_normals: bool = True,
        particle_flags: wp.array[wp.int32] | None = None,
        compute_mesh: bool = True,
    ) -> tuple[wp.array[wp.vec3] | None, wp.array[wp.int32] | None, wp.array[wp.vec3] | None]:
        """Extract a triangle mesh from particle positions.

        Args:
            positions: Particle positions [m], shape ``(N,)``, dtype ``wp.vec3``.
            radii: Per-particle radii [m], shape ``(N,)``, dtype ``wp.float32``.
            compute_normals: Whether to compute per-vertex normals.
            particle_flags: Optional per-particle flags.  Particles without
                :attr:`~newton.ParticleFlags.ACTIVE` are skipped.
            compute_mesh: Whether to run Marching Cubes and mesh post-processing.
                Set to ``False`` to update only :attr:`field`, for example before
                modifying the field and calling :meth:`resurface`.

        Returns:
            Tuple of vertex positions [m], triangle indices, and unit-length
            vertex normals. All entries are ``None`` when no surface can be
            extracted.
        """
        self._validate_positions_layout(positions)
        n = positions.shape[0]
        device = positions.device
        self._device = wp.get_device(device)
        self._validate_radii_layout(positions, radii, n)
        self._validate_particle_flags_layout(particle_flags, n, device)
        if n == 0:
            self._clear_results(clear_field=True)
            return None, None, None

        positions, radii = self._filter_active_particles(positions, radii, particle_flags, device)
        if positions is None:
            self._clear_results(clear_field=True)
            return None, None, None
        n = positions.shape[0]
        self._validate_radii_values(radii, device)

        # Step 1: Compute AABB
        lower = wp.array([wp.vec3(1e30, 1e30, 1e30)], dtype=wp.vec3, device=device)
        upper = wp.array([wp.vec3(-1e30, -1e30, -1e30)], dtype=wp.vec3, device=device)
        if _use_cuda_tile_kernels(device):
            wp.launch_tiled(
                _compute_aabb_tiled,
                dim=_reduction_tile_dim(n),
                inputs=[positions, n, lower, upper],
                block_dim=_REDUCTION_TILE_SIZE,
                device=device,
            )
        else:
            wp.launch(_compute_aabb, dim=n, inputs=[positions, lower, upper], device=device)

        aabb_min = lower.numpy()[0]
        aabb_max = upper.numpy()[0]

        # Prepare anisotropy before sizing the field so the dynamic grid only
        # covers supports that actually occur, rather than reserving the
        # theoretical maximum axis ratio in every direction.
        particle_extent = float(np.max(aabb_max - aabb_min)) + 2.0 * self.kernel_radius
        self._ensure_particle_resources(n, particle_extent, device)
        flags, use_flags = self._prepare_particle_values(positions, n, None, device)

        kernel_lower = wp.array([wp.vec3(1e30, 1e30, 1e30)], dtype=wp.vec3, device=device)
        kernel_upper = wp.array([wp.vec3(-1e30, -1e30, -1e30)], dtype=wp.vec3, device=device)
        isotropic_sdf = not self.anisotropic or self.anisotropy_strength <= 0.0 or self.anisotropy_ratio <= 1.0
        kernel_aabb_inputs = [
            self._smoothed[:n],
            radii,
            self._det_G[:n],
            self._density_reach[:n],
            self.particle_sdf_radius_scale,
            self.particle_sdf_band,
            int(self.surface_method == "particle_sdf"),
            int(not isotropic_sdf),
        ]
        if _use_cuda_tile_kernels(device):
            wp.launch_tiled(
                _compute_kernel_aabb_tiled,
                dim=_reduction_tile_dim(n),
                inputs=[*kernel_aabb_inputs, n, kernel_lower, kernel_upper],
                block_dim=_REDUCTION_TILE_SIZE,
                device=device,
            )
        else:
            wp.launch(
                _compute_kernel_aabb,
                dim=n,
                inputs=[*kernel_aabb_inputs, kernel_lower, kernel_upper],
                device=device,
            )

        pad = self.voxel_size * self.padding
        grid_min = np.floor((kernel_lower.numpy()[0] - pad) / self.voxel_size) * self.voxel_size
        grid_max = np.ceil((kernel_upper.numpy()[0] + pad) / self.voxel_size) * self.voxel_size
        dims = np.round((grid_max - grid_min) / self.voxel_size).astype(int) + 1

        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        grid_origin = wp.vec3(float(grid_min[0]), float(grid_min[1]), float(grid_min[2]))
        grid_end = wp.vec3(float(grid_max[0]), float(grid_max[1]), float(grid_max[2]))

        # Step 2: Allocate / resize cached field objects.
        self._ensure_field_resources(nx, ny, nz, grid_origin, grid_end, device)

        # Step 3: Evaluate the scalar field from the prepared particles.
        self._evaluate_field_values(radii, n, flags, use_flags, device)

        if not compute_mesh:
            self._verts = self._indices = self._normals = None
            return None, None, None

        # Step 4: Marching cubes.
        self._mc.surface(self._field, self._marching_threshold())
        verts = self._mc.verts
        indices = self._mc.indices

        # In density mode, high values are inside, so MC's low-to-high
        # orientation is inward. In SDF mode, low values are inside and the
        # default orientation is already outward.
        if self.field_mode == "density" and indices is not None and indices.shape[0] > 0:
            wp.launch(_flip_winding, dim=indices.shape[0] // 3, inputs=[indices], device=device)

        if verts is None or verts.shape[0] == 0:
            self._verts = self._indices = self._normals = None
            return None, None, None

        # Step 5: Laplacian smoothing
        if self.mesh_smooth_iterations > 0 and indices.shape[0] > 0:
            num_verts = verts.shape[0]
            num_tri_verts = indices.shape[0]
            smoothed = wp.empty(num_verts, dtype=wp.vec3, device=device)
            neighbor_sum = wp.zeros(num_verts, dtype=wp.vec3, device=device)
            valence = wp.zeros(num_verts, dtype=wp.int32, device=device)

            for _ in range(self.mesh_smooth_iterations):
                neighbor_sum.zero_()
                valence.zero_()
                wp.launch(
                    _laplacian_scatter, dim=num_tri_verts, inputs=[indices, verts, neighbor_sum, valence], device=device
                )
                wp.launch(
                    _laplacian_apply,
                    dim=num_verts,
                    inputs=[verts, neighbor_sum, valence, smoothed, self.mesh_smooth_lambda],
                    device=device,
                )
                verts, smoothed = smoothed, verts

        # Step 6: Vertex normals
        normals = None
        if compute_normals:
            normals = compute_vertex_normals(verts, indices)

        self._verts = verts
        self._indices = indices
        self._normals = normals
        return verts, indices, normals

    def redistance(self, iterations: int | None = None) -> None:
        """Apply Eikonal redistancing to the current SDF field.

        Args:
            iterations: Number of redistancing iterations.  Defaults to
                :attr:`redistance_iterations`.
        """
        if self.field_mode != "sdf":
            raise ValueError("redistance() requires field_mode='sdf'")
        if self._field is None or self._grid_dims is None:
            return
        if iterations is None:
            iterations = self.redistance_iterations
        if iterations <= 0:
            return

        nx, ny, nz = self._grid_dims
        if self._blur_temp is None or self._blur_temp.shape != self._field.shape:
            self._blur_temp = wp.empty((nx, ny, nz), dtype=wp.float32, device=self._device)

        inv_dx = 1.0 / self.voxel_size
        src = self._field
        dst = self._blur_temp
        for _ in range(iterations):
            wp.launch(_redistance_step, dim=(nx, ny, nz), inputs=[src, dst, inv_dx], device=self._device)
            src, dst = dst, src
        if src is not self._field:
            self._field, self._blur_temp = src, dst

    def resurface(
        self,
        compute_normals: bool = True,
    ) -> tuple[wp.array[wp.vec3] | None, wp.array[wp.int32] | None, wp.array[wp.vec3] | None]:
        """Re-run marching cubes on the current field.

        Use after externally modifying :attr:`field`, for example to
        extrapolate an SDF into collider regions before extracting the mesh.
        """
        if self._mc is None or self._field is None:
            self._verts = self._indices = self._normals = None
            return None, None, None

        self._mc.surface(self._field, self._marching_threshold())
        verts = self._mc.verts
        indices = self._mc.indices

        if self.field_mode == "density" and indices is not None and indices.shape[0] > 0:
            wp.launch(_flip_winding, dim=indices.shape[0] // 3, inputs=[indices], device=self._device)

        if verts is None or verts.shape[0] == 0:
            self._verts = self._indices = self._normals = None
            return None, None, None

        if self.mesh_smooth_iterations > 0 and indices.shape[0] > 0:
            num_verts = verts.shape[0]
            num_tri_verts = indices.shape[0]
            smoothed = wp.empty(num_verts, dtype=wp.vec3, device=self._device)
            neighbor_sum = wp.zeros(num_verts, dtype=wp.vec3, device=self._device)
            valence = wp.zeros(num_verts, dtype=wp.int32, device=self._device)

            for _ in range(self.mesh_smooth_iterations):
                neighbor_sum.zero_()
                valence.zero_()
                wp.launch(
                    _laplacian_scatter,
                    dim=num_tri_verts,
                    inputs=[indices, verts, neighbor_sum, valence],
                    device=self._device,
                )
                wp.launch(
                    _laplacian_apply,
                    dim=num_verts,
                    inputs=[verts, neighbor_sum, valence, smoothed, self.mesh_smooth_lambda],
                    device=self._device,
                )
                verts, smoothed = smoothed, verts

        normals = None
        if compute_normals:
            normals = compute_vertex_normals(verts, indices)

        self._verts = verts
        self._indices = indices
        self._normals = normals
        return verts, indices, normals

    # -- Internal helpers --

    def _clear_results(self, *, clear_field: bool):
        self._verts = None
        self._indices = None
        self._normals = None
        if clear_field:
            self._field = None
            self._grid_dims = None
            self._grid_origin = None

    def _clear_device_resources(self):
        self._clear_results(clear_field=True)
        self._mc = None
        self._hash_grid = None
        self._blur_temp = None
        self._blur_weights = None
        self._hash_grid_dim = 0
        self._smoothed = None
        self._G = None
        self._det_G = None
        self._density_reach = None
        self._hash_positions = None
        self._all_particle_flags = None
        self._n_particles = 0
        self._max_particles = 0

    def _update_field_values(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        n: int,
        particle_flags: wp.array[wp.int32] | None,
        device: wp.DeviceLike,
    ):
        flags, use_flags = self._prepare_particle_values(positions, n, particle_flags, device)
        self._evaluate_field_values(radii, n, flags, use_flags, device)

    def _prepare_particle_values(
        self,
        positions: wp.array[wp.vec3],
        n: int,
        particle_flags: wp.array[wp.int32] | None,
        device: wp.DeviceLike,
    ) -> tuple[wp.array[wp.int32], int]:
        flags, use_flags = self._field_flag_args(particle_flags, n, device)

        hash_positions = positions
        if use_flags != 0:
            if self._hash_positions is None or self._hash_positions.shape[0] < n:
                raise RuntimeError("configure_field_grid() must allocate enough hash positions for flagged updates")
            hash_positions = self._hash_positions[:n]
            wp.launch(
                _copy_active_or_sentinel_positions,
                dim=n,
                inputs=[positions, flags, use_flags, self._inactive_position, hash_positions],
                device=device,
            )

        smoothed = self._smoothed[:n]
        G = self._G[:n]
        det_G = self._det_G[:n]
        density_reach = self._density_reach[:n]

        # Smooth particle positions.
        if self.smooth_lambda > 1.0e-6 and n > 0:
            self._hash_grid.build(hash_positions, self.kernel_radius)
            if use_flags != 0:
                wp.launch(
                    _smooth_positions_flagged,
                    dim=n,
                    inputs=[
                        self._hash_grid.id,
                        hash_positions,
                        flags,
                        self._inactive_position,
                        self.kernel_radius,
                        self.smooth_lambda,
                        smoothed,
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    _smooth_positions,
                    dim=n,
                    inputs=[
                        self._hash_grid.id,
                        hash_positions,
                        self.kernel_radius,
                        self.smooth_lambda,
                        smoothed,
                    ],
                    device=device,
                )
        elif n > 0:
            if use_flags != 0:
                wp.launch(
                    _copy_active_or_sentinel_positions,
                    dim=n,
                    inputs=[positions, flags, use_flags, self._inactive_position, smoothed],
                    device=device,
                )
            else:
                wp.copy(smoothed, positions)

        # Compute per-particle anisotropy.
        if self.anisotropic and n > 0:
            self._hash_grid.build(smoothed, self.kernel_radius)
            wp.launch(
                _compute_anisotropy,
                dim=n,
                inputs=[
                    self._hash_grid.id,
                    smoothed,
                    flags,
                    use_flags,
                    self.kernel_radius,
                    self.anisotropy_ratio,
                    self.anisotropy_scale,
                    self.kernel_scale,
                    self.anisotropy_min_neighbors,
                    self.anisotropy_strength,
                    G,
                    det_G,
                    density_reach,
                ],
                device=device,
            )
        elif n > 0:
            wp.launch(
                _fill_isotropic_G,
                dim=n,
                inputs=[self.kernel_radius, self.kernel_scale, G, det_G, density_reach],
                device=device,
            )

        return flags, use_flags

    def _evaluate_field_values(
        self,
        radii: wp.array[float],
        n: int,
        flags: wp.array[wp.int32],
        use_flags: int,
        device: wp.DeviceLike,
    ):
        nx, ny, nz = self._grid_dims
        smoothed = self._smoothed[:n]
        G = self._G[:n]
        det_G = self._det_G[:n]
        density_reach = self._density_reach[:n]

        # Evaluate scalar field.
        if self.surface_method == "particle_sdf":
            far_sdf = self.kernel_radius * self.particle_sdf_band
            wp.launch(_fill_field, dim=(nx, ny, nz), inputs=[self._field, far_sdf], device=device)
            if n > 0:
                isotropic_sdf = not self.anisotropic or self.anisotropy_strength <= 0.0 or self.anisotropy_ratio <= 1.0
                if isotropic_sdf:
                    wp.launch(
                        _eval_particle_sdf_union_isotropic,
                        dim=n,
                        inputs=[
                            smoothed,
                            radii,
                            flags,
                            use_flags,
                            self.particle_sdf_radius_scale,
                            self.particle_sdf_band,
                            self._grid_origin,
                            1.0 / self.voxel_size,
                            nx,
                            ny,
                            nz,
                            self._field,
                        ],
                        device=device,
                    )
                else:
                    wp.launch(
                        _eval_particle_sdf_union_anisotropic,
                        dim=n,
                        inputs=[
                            smoothed,
                            radii,
                            flags,
                            use_flags,
                            G,
                            det_G,
                            density_reach,
                            self.particle_sdf_radius_scale,
                            self.particle_sdf_band,
                            self._grid_origin,
                            1.0 / self.voxel_size,
                            nx,
                            ny,
                            nz,
                            self._field,
                        ],
                        device=device,
                    )
        else:
            wp.launch(_fill_field, dim=(nx, ny, nz), inputs=[self._field, 0.0], device=device)
            if n > 0:
                wp.launch(
                    _eval_scalar_field,
                    dim=n,
                    inputs=[
                        smoothed,
                        radii,
                        flags,
                        use_flags,
                        G,
                        det_G,
                        density_reach,
                        self._grid_origin,
                        1.0 / self.voxel_size,
                        nx,
                        ny,
                        nz,
                        self._field,
                    ],
                    device=device,
                )

        if self.field_smooth_iterations > 0 and self.field_smooth_radius > 0:
            self._apply_field_blur(nx, ny, nz, device)

        if self.field_mode == "sdf":
            if self.surface_method == "density":
                wp.launch(_density_to_sdf_3d, dim=(nx, ny, nz), inputs=[self._field, self.threshold], device=device)
            if self.redistance_iterations > 0:
                self.redistance()

    def _field_flag_args(
        self,
        particle_flags: wp.array[wp.int32] | None,
        n: int,
        device: wp.DeviceLike,
    ) -> tuple[wp.array[wp.int32], int]:
        if particle_flags is not None:
            return particle_flags, 1
        return self._ensure_all_particle_flags(n, device), 0

    def _ensure_all_particle_flags(self, n: int, device: wp.DeviceLike) -> wp.array[wp.int32]:
        alloc_particles = max(n, 1)
        if (
            self._all_particle_flags is None
            or self._all_particle_flags.shape[0] < alloc_particles
            or self._all_particle_flags.device != wp.get_device(device)
        ):
            self._all_particle_flags = wp.empty(alloc_particles, dtype=wp.int32, device=device)
        return self._all_particle_flags

    def _validate_particle_flags_layout(
        self,
        particle_flags: wp.array[wp.int32] | None,
        n: int,
        device: wp.DeviceLike,
    ):
        if particle_flags is None:
            return
        if particle_flags.ndim != 1:
            raise ValueError(f"particle_flags must be a 1-D array, got shape {particle_flags.shape}")
        if particle_flags.shape[0] != n:
            raise ValueError(f"particle_flags length ({particle_flags.shape[0]}) must match positions length ({n})")
        if particle_flags.device != wp.get_device(device):
            raise ValueError(f"particle_flags device ({particle_flags.device}) must match positions device ({device})")
        if particle_flags.dtype != wp.int32:
            raise TypeError(f"particle_flags must have dtype wp.int32, got {particle_flags.dtype}")

    def _ensure_blur_weights(self, device: wp.DeviceLike):
        hw = self.field_smooth_radius
        if hw <= 0:
            return
        device = wp.get_device(device)
        if (
            self._blur_weights is not None
            and self._blur_weights.shape[0] == hw + 1
            and self._blur_weights.device == device
        ):
            return
        sigma = max(hw / 2.0, 0.5)
        w = np.array([math.exp(-0.5 * (d / sigma) ** 2) for d in range(hw + 1)], dtype=np.float32)
        w /= w[0] + 2.0 * np.sum(w[1:])
        self._blur_weights = wp.array(w, dtype=float, device=device)

    def _grid_padding(self, radii: wp.array, device: wp.DeviceLike) -> float:
        support = _DENSITY_KERNEL_SUPPORT * self.kernel_scale * self.kernel_radius
        if self.anisotropic:
            # Volume normalization limits the longest ellipsoid axis to R^(2/3)
            # when the shortest-to-longest axis ratio is clamped to 1/R.
            axis_ratio = max(float(self.anisotropy_ratio), 1.0) ** (2.0 / 3.0)
            density_support = (
                _DENSITY_KERNEL_SUPPORT
                * float(self.kernel_scale)
                * float(self.anisotropy_scale)
                * axis_ratio
                * self.kernel_radius
            )
            support = max(support, density_support)

        if self.surface_method == "particle_sdf":
            max_radius = wp.zeros(1, dtype=float, device=device)
            if _use_cuda_tile_kernels(device):
                wp.launch_tiled(
                    _compute_max_radius_tiled,
                    dim=_reduction_tile_dim(radii.shape[0]),
                    inputs=[radii, radii.shape[0], max_radius],
                    block_dim=_REDUCTION_TILE_SIZE,
                    device=device,
                )
            else:
                wp.launch(_compute_max_radius, dim=radii.shape[0], inputs=[radii, max_radius], device=device)
            particle_sdf_support = (
                float(max_radius.numpy()[0]) * self.particle_sdf_radius_scale * self.particle_sdf_band
            )
            if self.anisotropic:
                particle_sdf_support *= max(float(self.anisotropy_ratio), 1.0) ** (2.0 / 3.0)
            support = max(support, particle_sdf_support)

        return support + self.voxel_size * self.padding

    def _marching_threshold(self) -> float:
        if self.field_mode == "sdf":
            effective_threshold = 0.0
            if self.mesh_smooth_iterations > 0:
                shrink = (
                    _MESH_SMOOTH_SHRINK_PER_VOXEL
                    * math.sqrt(float(self.mesh_smooth_iterations))
                    * self.mesh_smooth_lambda
                    * self.voxel_size
                )
                if self.surface_method == "particle_sdf" or self.redistance_iterations > 0:
                    effective_threshold = shrink
                else:
                    effective_threshold = shrink / self.kernel_radius
            return effective_threshold

        effective_threshold = self.threshold
        if self.mesh_smooth_iterations > 0:
            shrink = (
                _MESH_SMOOTH_SHRINK_PER_VOXEL
                * math.sqrt(float(self.mesh_smooth_iterations))
                * self.mesh_smooth_lambda
                * self.voxel_size
            )
            effective_threshold = max(self.threshold - shrink / self.kernel_radius, _MIN_DENSITY_MARCHING_THRESHOLD)
        return effective_threshold

    def _validate_radii_layout(self, positions: wp.array[wp.vec3], radii: wp.array[float], n: int):
        if not isinstance(radii, wp.array):
            raise TypeError(f"radii must be a Warp array, got {type(radii).__name__}")
        if radii.ndim != 1:
            raise ValueError(f"radii must be a 1-D array, got shape {radii.shape}")
        if radii.shape[0] != n:
            raise ValueError(f"radii length ({radii.shape[0]}) must match positions length ({n})")
        if radii.device != positions.device:
            raise ValueError(f"radii device ({radii.device}) must match positions device ({positions.device})")
        if radii.dtype != wp.float32:
            raise TypeError(f"radii must have dtype wp.float32, got {radii.dtype}")

    def _validate_positions_layout(self, positions: wp.array[wp.vec3]):
        if not isinstance(positions, wp.array):
            raise TypeError(f"positions must be a Warp array, got {type(positions).__name__}")
        if positions.ndim != 1:
            raise ValueError(f"positions must be a 1-D array, got shape {positions.shape}")
        if positions.dtype != wp.vec3:
            raise TypeError(f"positions must have dtype wp.vec3, got {positions.dtype}")

    def _validate_radii_values(self, radii: wp.array[float], device: wp.DeviceLike):
        invalid = wp.zeros(1, dtype=wp.int32, device=device)
        if _use_cuda_tile_kernels(device):
            wp.launch_tiled(
                _validate_positive_finite_tiled,
                dim=_reduction_tile_dim(radii.shape[0]),
                inputs=[radii, radii.shape[0], invalid],
                block_dim=_REDUCTION_TILE_SIZE,
                device=device,
            )
        else:
            wp.launch(_validate_positive_finite, dim=radii.shape[0], inputs=[radii, invalid], device=device)
        if int(invalid.numpy()[0]) != 0:
            raise ValueError("radii values must be finite and positive")

    def _filter_active_particles(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        particle_flags: wp.array[wp.int32] | None,
        device: wp.DeviceLike,
    ) -> tuple[wp.array[wp.vec3] | None, wp.array[float] | None]:
        if particle_flags is None:
            return positions, radii

        n = positions.shape[0]
        if particle_flags.shape[0] != n:
            raise ValueError(f"particle_flags length ({particle_flags.shape[0]}) must match positions length ({n})")

        mask = wp.empty(n, dtype=wp.int32, device=device)
        wp.launch(_build_active_particle_mask, dim=n, inputs=[particle_flags, mask], device=device)

        offsets = wp.empty(n, dtype=wp.int32, device=device)
        wp.utils.array_scan(mask, offsets, inclusive=False)

        active_count = int(offsets[-1:].numpy()[0]) + int(mask[-1:].numpy()[0])
        if active_count == 0:
            return None, None
        if active_count == n:
            return positions, radii

        active_positions = wp.empty(active_count, dtype=wp.vec3, device=device)
        active_radii = wp.empty(active_count, dtype=radii.dtype, device=device)
        wp.launch(_compact_active_particles, dim=n, inputs=[positions, mask, offsets, active_positions], device=device)
        wp.launch(_compact_active_particles, dim=n, inputs=[radii, mask, offsets, active_radii], device=device)
        return active_positions, active_radii

    def _ensure_particle_resources(
        self,
        n_particles: int,
        extent: float,
        device: wp.DeviceLike,
    ):
        device_obj = wp.get_device(device)

        if self._resource_device != device_obj:
            self._clear_device_resources()
            self._resource_device = device_obj

        hash_dim = max(16, int(math.ceil(extent / self.kernel_radius)))
        if self._hash_grid is None or self._hash_grid_dim != hash_dim:
            self._hash_grid = wp.HashGrid(hash_dim, hash_dim, hash_dim, device=device)
            self._hash_grid_dim = hash_dim

        if self._n_particles != n_particles:
            self._smoothed = wp.empty(n_particles, dtype=wp.vec3, device=device)
            self._G = wp.empty(n_particles, dtype=wp.mat33, device=device)
            self._det_G = wp.empty(n_particles, dtype=float, device=device)
            self._density_reach = wp.empty(n_particles, dtype=wp.vec3, device=device)
            self._n_particles = n_particles

    def _ensure_field_resources(
        self,
        nx: int,
        ny: int,
        nz: int,
        grid_origin: wp.vec3,
        grid_end: wp.vec3,
        device: wp.DeviceLike,
    ):
        new_dims = (nx, ny, nz)
        device_obj = wp.get_device(device)

        if self._resource_device != device_obj:
            self._clear_device_resources()
            self._resource_device = device_obj

        if self._grid_dims != new_dims:
            self._mc = wp.MarchingCubes(nx, ny, nz)
            self._field = wp.empty((nx, ny, nz), dtype=wp.float32, device=device)
            if self.field_smooth_iterations > 0 and self.field_smooth_radius > 0:
                self._blur_temp = wp.empty((nx, ny, nz), dtype=wp.float32, device=device)
            else:
                self._blur_temp = None
            self._grid_dims = new_dims

        self._mc.domain_bounds_lower_corner = grid_origin
        self._mc.domain_bounds_upper_corner = grid_end
        self._grid_origin = grid_origin

    def _apply_field_blur(self, nx: int, ny: int, nz: int, device: wp.DeviceLike):
        """Separable Gaussian blur on the scalar field."""
        hw = self.field_smooth_radius
        self._ensure_blur_weights(device)
        if self._blur_temp is None or self._blur_temp.shape != self._field.shape:
            self._blur_temp = wp.empty((nx, ny, nz), dtype=wp.float32, device=device)

        src = self._field
        dst = self._blur_temp
        w = self._blur_weights
        for _ in range(self.field_smooth_iterations):
            wp.launch(_blur_axis_x, dim=(nx, ny, nz), inputs=[src, dst, w, hw], device=device)
            wp.launch(_blur_axis_y, dim=(nx, ny, nz), inputs=[dst, src, w, hw], device=device)
            wp.launch(_blur_axis_z, dim=(nx, ny, nz), inputs=[src, dst, w, hw], device=device)
            src, dst = dst, src
        if src is not self._field:
            self._field, self._blur_temp = src, dst


def extract_particle_surface(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    voxel_size: float,
    *,
    kernel_radius: float | None = None,
    threshold: float = 0.25,
    smooth_lambda: float = 0.5,
    mesh_smooth_iterations: int = 0,
    compute_normals: bool = True,
    anisotropic: bool = False,
    field_mode: Literal["density", "sdf"] | None = None,
    redistance_iterations: int = 0,
    particle_flags: wp.array[wp.int32] | None = None,
    anisotropy_ratio: float = 4.0,
    kernel_scale: float = 0.5,
    anisotropy_scale: float = 1.0,
    anisotropy_min_neighbors: int = 25,
    anisotropy_strength: float = 1.0,
    field_smooth_iterations: int = 0,
    field_smooth_radius: int = 1,
    surface_method: Literal["density", "particle_sdf"] = "density",
    particle_sdf_radius_scale: float = 1.0,
    particle_sdf_band: float = 2.0,
) -> tuple[wp.array[wp.vec3] | None, wp.array[wp.int32] | None, wp.array[wp.vec3] | None]:
    """Extract a triangle mesh from particle positions (one-shot convenience).

    Args:
        positions: Particle positions [m], shape ``(N,)``, dtype ``wp.vec3``.
        radii: Per-particle radii [m], shape ``(N,)``, dtype ``wp.float32``.
        voxel_size: Edge length of each grid voxel [m].
        kernel_radius: Search radius [m].  Defaults to ``3 * voxel_size``.
        threshold: Isosurface level.
        smooth_lambda: Position smoothing blend factor [0, 1].
        mesh_smooth_iterations: Laplacian mesh smoothing passes.
        compute_normals: Whether to compute per-vertex normals.
        anisotropic: Enable per-particle WPCA anisotropic kernels.
        field_mode: Field representation retained after extraction.  Defaults
            to ``"sdf"`` for ``surface_method="particle_sdf"`` and ``"density"``
            otherwise.
        redistance_iterations: Number of Eikonal redistancing iterations
            applied when ``field_mode="sdf"``.
        particle_flags: Optional per-particle flags.  Particles without
            :attr:`~newton.ParticleFlags.ACTIVE` are skipped.
        anisotropy_ratio: Maximum anisotropic kernel axis ratio.
        kernel_scale: Kernel radius multiplier relative to ``kernel_radius``.
        anisotropy_scale: Relative multiplier for anisotropic kernel radii.
        anisotropy_min_neighbors: Minimum number of other particles required
            for anisotropic kernels.
        anisotropy_strength: Blend from isotropic kernels to anisotropic
            kernels [0, 1].
        field_smooth_iterations: Number of separable Gaussian blur passes
            applied to the scalar field before marching cubes.
        field_smooth_radius: Half-width of the Gaussian blur in voxels.
        surface_method: Surface reconstruction method. ``"density"`` uses
            anisotropic density splatting. ``"particle_sdf"`` directly unions
            per-particle anisotropic ellipsoid SDFs.
        particle_sdf_radius_scale: Radius multiplier for ``surface_method="particle_sdf"``.
        particle_sdf_band: Narrow-band half-width in normalized ellipsoid
            coordinates for ``surface_method="particle_sdf"``.

    Returns:
        Tuple of vertex positions [m], triangle indices, and unit-length
        vertex normals. All entries are ``None`` when no surface can be
        extracted.
    """
    ctx = ParticleSurface(
        voxel_size=voxel_size,
        kernel_radius=kernel_radius,
        threshold=threshold,
        smooth_lambda=smooth_lambda,
        anisotropic=anisotropic,
        anisotropy_ratio=anisotropy_ratio,
        anisotropy_scale=anisotropy_scale,
        kernel_scale=kernel_scale,
        anisotropy_min_neighbors=anisotropy_min_neighbors,
        anisotropy_strength=anisotropy_strength,
        field_smooth_iterations=field_smooth_iterations,
        field_smooth_radius=field_smooth_radius,
        surface_method=surface_method,
        particle_sdf_radius_scale=particle_sdf_radius_scale,
        particle_sdf_band=particle_sdf_band,
        field_mode=field_mode,
        redistance_iterations=redistance_iterations,
        mesh_smooth_iterations=mesh_smooth_iterations,
        device=positions.device,
    )
    return ctx.extract(positions, radii=radii, compute_normals=compute_normals, particle_flags=particle_flags)
