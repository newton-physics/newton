# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Texture-based (tex3d) sparse SDF construction and sampling.

This module provides a GPU-accelerated sparse SDF implementation using 3D CUDA textures.
Construction mirrors the NanoVDB sparse-volume pattern in ``sdf_utils.py``:

1. Check subgrid occupancy by querying mesh SDF at subgrid centers
2. Build background/coarse SDF by querying mesh at subgrid corner positions
3. Populate only occupied subgrid textures by querying mesh at each texel

The format uses:
- A coarse 3D texture for background/far-field sampling
- A packed subgrid 3D texture for narrow-band high-resolution sampling
- An indirection array mapping coarse cells to subgrid blocks

Sampling uses analytical trilinear gradient computation from 8 corner texel reads,
providing exact accuracy with only 8 texture reads (vs 56 for finite differences).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .sdf_utils import get_distance_to_mesh

# Sentinel values for subgrid indirection slots.
# Plain int so wp.static() works in kernels; numpy casts on assignment.
SLOT_EMPTY = 0xFFFFFFFF  # No subgrid data (empty/far-field cell)
SLOT_LINEAR = 0xFFFFFFFE  # Subgrid demoted to coarse interpolation

# ============================================================================
# Texture SDF Data Structure
# ============================================================================


class QuantizationMode:
    """Quantization modes for subgrid SDF data."""

    FLOAT32 = 4  # No quantization, full precision
    UINT16 = 2  # 16-bit quantization
    UINT8 = 1  # 8-bit quantization


@wp.struct
class TextureSDFData:
    """Sparse SDF stored in 3D CUDA textures with indirection array.

    Uses a two-level structure:
    - A coarse 3D texture for background/far-field sampling
    - A packed subgrid 3D texture for narrow-band high-resolution sampling
    - An indirection array mapping coarse cells to subgrid texture blocks
    """

    # Textures and indirection
    coarse_texture: wp.Texture3D
    subgrid_texture: wp.Texture3D
    subgrid_start_slots: wp.array3d[wp.uint32]

    # Grid parameters
    sdf_box_lower: wp.vec3
    sdf_box_upper: wp.vec3
    inv_sdf_dx: wp.vec3
    subgrid_size: int
    subgrid_size_f: float  # float(subgrid_size) - avoids int->float conversion
    subgrid_samples_f: float  # float(subgrid_size + 1) - samples per subgrid dimension
    fine_to_coarse: float

    # Spatial metadata
    voxel_size: wp.vec3
    voxel_radius: wp.float32

    # Quantization parameters for subgrid values
    subgrids_min_sdf_value: float
    subgrids_sdf_value_range: float  # max - min

    # Whether shape_scale was baked into the SDF
    scale_baked: wp.bool


# ============================================================================
# Sparse SDF Construction Kernels
# ============================================================================


@wp.func
def _idx3d(x: int, y: int, z: int, size_x: int, size_y: int) -> int:
    """Convert 3D coordinates to linear index."""
    return z * size_x * size_y + y * size_x + x


@wp.func
def _id_to_xyz(idx: int, size_x: int, size_y: int) -> wp.vec3i:
    """Convert linear index to 3D coordinates."""
    z = idx // (size_x * size_y)
    rem = idx - z * size_x * size_y
    y = rem // size_x
    x = rem - y * size_x
    return wp.vec3i(x, y, z)


@wp.kernel
def _check_subgrid_occupied_kernel(
    mesh: wp.uint64,
    subgrid_centers: wp.array[wp.vec3],
    threshold: wp.vec2f,
    winding_threshold: float,
    subgrid_required: wp.array[wp.int32],
):
    """Mark subgrids that overlap the narrow band by checking mesh SDF at center."""
    tid = wp.tid()
    sample_pos = subgrid_centers[tid]

    signed_distance = get_distance_to_mesh(mesh, sample_pos, 10000.0, winding_threshold)
    is_occupied = wp.bool(False)
    if wp.sign(signed_distance) > 0.0:
        is_occupied = signed_distance < threshold[1]
    else:
        is_occupied = signed_distance > threshold[0]

    if is_occupied:
        subgrid_required[tid] = 1
    else:
        subgrid_required[tid] = 0


@wp.kernel
def _check_subgrid_linearity_kernel(
    mesh: wp.uint64,
    background_sdf: wp.array[float],
    subgrid_required: wp.array[wp.int32],
    subgrid_is_linear: wp.array[wp.int32],
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    bg_size_x: int,
    bg_size_y: int,
    bg_size_z: int,
    error_threshold: float,
):
    """Demote occupied subgrids whose SDF is well-approximated by the coarse grid.

    For each occupied subgrid, queries the mesh SDF at every fine-grid sample
    and compares against the trilinearly interpolated coarse/background SDF.
    Subgrids where the maximum absolute error is below *error_threshold* are
    marked as linear (``subgrid_is_linear[tid] = 1``) and removed from
    ``subgrid_required`` so no subgrid texture data is allocated for them.
    """
    tid = wp.tid()
    if subgrid_required[tid] == 0:
        return

    coords = _id_to_xyz(tid, num_subgrids_x, num_subgrids_y)
    block_x = coords[0]
    block_y = coords[1]
    block_z = coords[2]

    s = 1.0 / float(cells_per_subgrid)
    samples_per_dim = cells_per_subgrid + 1
    max_abs_error = float(0.0)

    for lz in range(samples_per_dim):
        for ly in range(samples_per_dim):
            for lx in range(samples_per_dim):
                gx = block_x * cells_per_subgrid + lx
                gy = block_y * cells_per_subgrid + ly
                gz = block_z * cells_per_subgrid + lz

                pos = min_corner + wp.vec3(
                    float(gx) * cell_size[0],
                    float(gy) * cell_size[1],
                    float(gz) * cell_size[2],
                )
                mesh_val = get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

                coarse_fx = float(block_x) + float(lx) * s
                coarse_fy = float(block_y) + float(ly) * s
                coarse_fz = float(block_z) + float(lz) * s

                x0 = int(wp.floor(coarse_fx))
                y0 = int(wp.floor(coarse_fy))
                z0 = int(wp.floor(coarse_fz))
                x0 = wp.clamp(x0, 0, bg_size_x - 2)
                y0 = wp.clamp(y0, 0, bg_size_y - 2)
                z0 = wp.clamp(z0, 0, bg_size_z - 2)

                tx = wp.clamp(coarse_fx - float(x0), 0.0, 1.0)
                ty = wp.clamp(coarse_fy - float(y0), 0.0, 1.0)
                tz = wp.clamp(coarse_fz - float(z0), 0.0, 1.0)

                v000 = background_sdf[_idx3d(x0, y0, z0, bg_size_x, bg_size_y)]
                v100 = background_sdf[_idx3d(x0 + 1, y0, z0, bg_size_x, bg_size_y)]
                v010 = background_sdf[_idx3d(x0, y0 + 1, z0, bg_size_x, bg_size_y)]
                v110 = background_sdf[_idx3d(x0 + 1, y0 + 1, z0, bg_size_x, bg_size_y)]
                v001 = background_sdf[_idx3d(x0, y0, z0 + 1, bg_size_x, bg_size_y)]
                v101 = background_sdf[_idx3d(x0 + 1, y0, z0 + 1, bg_size_x, bg_size_y)]
                v011 = background_sdf[_idx3d(x0, y0 + 1, z0 + 1, bg_size_x, bg_size_y)]
                v111 = background_sdf[_idx3d(x0 + 1, y0 + 1, z0 + 1, bg_size_x, bg_size_y)]

                c00 = v000 * (1.0 - tx) + v100 * tx
                c10 = v010 * (1.0 - tx) + v110 * tx
                c01 = v001 * (1.0 - tx) + v101 * tx
                c11 = v011 * (1.0 - tx) + v111 * tx
                c0 = c00 * (1.0 - ty) + c10 * ty
                c1 = c01 * (1.0 - ty) + c11 * ty
                coarse_val = c0 * (1.0 - tz) + c1 * tz

                max_abs_error = wp.max(max_abs_error, wp.abs(mesh_val - coarse_val))

    if max_abs_error < error_threshold:
        subgrid_is_linear[tid] = 1
        subgrid_required[tid] = 0


@wp.kernel
def _build_coarse_sdf_from_mesh_kernel(
    mesh: wp.uint64,
    background_sdf: wp.array[float],
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    cells_per_subgrid: int,
    bg_size_x: int,
    bg_size_y: int,
    bg_size_z: int,
    winding_threshold: float,
):
    """Populate background SDF by querying mesh at subgrid corner positions."""
    tid = wp.tid()

    total_bg = bg_size_x * bg_size_y * bg_size_z
    if tid >= total_bg:
        return

    coords = _id_to_xyz(tid, bg_size_x, bg_size_y)
    x_block = coords[0]
    y_block = coords[1]
    z_block = coords[2]

    pos = min_corner + wp.vec3(
        float(x_block * cells_per_subgrid) * cell_size[0],
        float(y_block * cells_per_subgrid) * cell_size[1],
        float(z_block * cells_per_subgrid) * cell_size[2],
    )

    background_sdf[tid] = get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)


@wp.kernel
def _populate_subgrid_texture_float32_kernel(
    mesh: wp.uint64,
    subgrid_required: wp.array[wp.int32],
    subgrid_addresses: wp.array[wp.int32],
    subgrid_start_slots: wp.array3d[wp.uint32],
    subgrid_texture: wp.array[float],
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
):
    """Populate subgrid texture by querying mesh SDF (float32 version)."""
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return
    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = _id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = _id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = block_x * cells_per_subgrid + lx
    gy = block_y * cells_per_subgrid + ly
    gz = block_z * cells_per_subgrid + lz

    pos = min_corner + wp.vec3(
        float(gx) * cell_size[0],
        float(gy) * cell_size[1],
        float(gz) * cell_size[2],
    )
    sdf_val = get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = _id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[block_x, block_y, block_z] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    tex_idx = _idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = sdf_val


@wp.kernel
def _populate_subgrid_texture_uint16_kernel(
    mesh: wp.uint64,
    subgrid_required: wp.array[wp.int32],
    subgrid_addresses: wp.array[wp.int32],
    subgrid_start_slots: wp.array3d[wp.uint32],
    subgrid_texture: wp.array[wp.uint16],
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
    sdf_min: float,
    sdf_range_inv: float,
):
    """Populate subgrid texture by querying mesh SDF (uint16 quantized version)."""
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return
    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = _id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = _id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = block_x * cells_per_subgrid + lx
    gy = block_y * cells_per_subgrid + ly
    gz = block_z * cells_per_subgrid + lz

    pos = min_corner + wp.vec3(
        float(gx) * cell_size[0],
        float(gy) * cell_size[1],
        float(gz) * cell_size[2],
    )
    sdf_val = get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = _id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[block_x, block_y, block_z] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    v_normalized = wp.clamp((sdf_val - sdf_min) * sdf_range_inv, 0.0, 1.0)
    quantized = wp.uint16(v_normalized * 65535.0)

    tex_idx = _idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = quantized


@wp.kernel
def _populate_subgrid_texture_uint8_kernel(
    mesh: wp.uint64,
    subgrid_required: wp.array[wp.int32],
    subgrid_addresses: wp.array[wp.int32],
    subgrid_start_slots: wp.array3d[wp.uint32],
    subgrid_texture: wp.array[wp.uint8],
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
    sdf_min: float,
    sdf_range_inv: float,
):
    """Populate subgrid texture by querying mesh SDF (uint8 quantized version)."""
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return
    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = _id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = _id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = block_x * cells_per_subgrid + lx
    gy = block_y * cells_per_subgrid + ly
    gz = block_z * cells_per_subgrid + lz

    pos = min_corner + wp.vec3(
        float(gx) * cell_size[0],
        float(gy) * cell_size[1],
        float(gz) * cell_size[2],
    )
    sdf_val = get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = _id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[block_x, block_y, block_z] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    v_normalized = wp.clamp((sdf_val - sdf_min) * sdf_range_inv, 0.0, 1.0)
    quantized = wp.uint8(v_normalized * 255.0)

    tex_idx = _idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = quantized


# ============================================================================
# Volume Sampling Kernel (for NanoVDB → texture conversion)
# ============================================================================


@wp.kernel
def _sample_volume_at_positions_kernel(
    volume: wp.uint64,
    positions: wp.array[wp.vec3],
    out_values: wp.array[float],
):
    """Sample NanoVDB volume at world-space positions."""
    tid = wp.tid()
    pos = positions[tid]
    idx = wp.volume_world_to_index(volume, pos)
    out_values[tid] = wp.volume_sample_f(volume, idx, wp.Volume.LINEAR)


# ============================================================================
# Texture Sampling Functions (wp.func, used by collision kernels)
# ============================================================================


@wp.func
def apply_subgrid_start(start_slot: wp.uint32, local_f: wp.vec3, subgrid_samples_f: float) -> wp.vec3:
    """Apply subgrid block offset to local coordinates."""
    block_x = float(start_slot & wp.uint32(0x3FF))
    block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
    block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))

    return wp.vec3(
        local_f[0] + block_x * subgrid_samples_f,
        local_f[1] + block_y * subgrid_samples_f,
        local_f[2] + block_z * subgrid_samples_f,
    )


@wp.func
def apply_subgrid_sdf_scale(raw_value: float, min_value: float, value_range: float) -> float:
    """Apply quantization scale to convert normalized [0,1] value back to SDF distance."""
    return raw_value * value_range + min_value


vec8f = wp.types.vector(length=8, dtype=wp.float32)


@wp.func
def _read_cell_corners(
    sdf: TextureSDFData,
    f: wp.vec3,
) -> tuple[vec8f, float, float, float]:
    """Locate the fine-grid cell containing *f* and read 8 corner texel values.

    Point-samples each corner at integer+0.5 coordinates (exact texel centres)
    so the caller can perform full float32 trilinear interpolation, avoiding
    the 8-bit fixed-point weight precision of CUDA hardware texture filtering.

    Args:
        sdf: texture SDF data.
        f: query position in fine-grid coordinates
            (``cw_mul(clamped - sdf_box_lower, inv_sdf_dx)``).

    Returns:
        ``(corners, tx, ty, tz)`` where *corners* packs the 8 SDF values as
        ``[v000, v100, v010, v110, v001, v101, v011, v111]`` and
        ``(tx, ty, tz)`` are the fractional interpolation weights in [0, 1].
    """
    coarse_x = sdf.coarse_texture.width - 1
    coarse_y = sdf.coarse_texture.height - 1
    coarse_z = sdf.coarse_texture.depth - 1

    fine_verts_x = float(coarse_x) * sdf.subgrid_size_f
    fine_verts_y = float(coarse_y) * sdf.subgrid_size_f
    fine_verts_z = float(coarse_z) * sdf.subgrid_size_f

    fx = wp.clamp(f[0], 0.0, fine_verts_x)
    fy = wp.clamp(f[1], 0.0, fine_verts_y)
    fz = wp.clamp(f[2], 0.0, fine_verts_z)

    num_fine_cells_x = int(fine_verts_x)
    num_fine_cells_y = int(fine_verts_y)
    num_fine_cells_z = int(fine_verts_z)
    ix = wp.clamp(int(wp.floor(fx)), 0, num_fine_cells_x - 1)
    iy = wp.clamp(int(wp.floor(fy)), 0, num_fine_cells_y - 1)
    iz = wp.clamp(int(wp.floor(fz)), 0, num_fine_cells_z - 1)
    tx = fx - float(ix)
    ty = fy - float(iy)
    tz = fz - float(iz)

    x_base = wp.clamp(int(float(ix) * sdf.fine_to_coarse), 0, coarse_x - 1)
    y_base = wp.clamp(int(float(iy) * sdf.fine_to_coarse), 0, coarse_y - 1)
    z_base = wp.clamp(int(float(iz) * sdf.fine_to_coarse), 0, coarse_z - 1)

    start_slot = sdf.subgrid_start_slots[x_base, y_base, z_base]

    v000 = float(0.0)
    v100 = float(0.0)
    v010 = float(0.0)
    v110 = float(0.0)
    v001 = float(0.0)
    v101 = float(0.0)
    v011 = float(0.0)
    v111 = float(0.0)

    if start_slot >= wp.static(SLOT_LINEAR):
        cx = float(x_base)
        cy = float(y_base)
        cz = float(z_base)
        coarse_f = wp.vec3(fx, fy, fz) * sdf.fine_to_coarse
        tx = coarse_f[0] - cx
        ty = coarse_f[1] - cy
        tz = coarse_f[2] - cz
        v000 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 0.5), dtype=float)
        v100 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 0.5), dtype=float)
        v010 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 0.5), dtype=float)
        v110 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 0.5), dtype=float)
        v001 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 1.5), dtype=float)
        v101 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 1.5), dtype=float)
        v011 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 1.5), dtype=float)
        v111 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 1.5), dtype=float)
    else:
        block_x = float(start_slot & wp.uint32(0x3FF))
        block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
        block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))
        tex_ox = block_x * sdf.subgrid_samples_f
        tex_oy = block_y * sdf.subgrid_samples_f
        tex_oz = block_z * sdf.subgrid_samples_f
        lx = float(ix) - float(x_base) * sdf.subgrid_size_f
        ly = float(iy) - float(y_base) * sdf.subgrid_size_f
        lz = float(iz) - float(z_base) * sdf.subgrid_size_f
        ox = tex_ox + lx + 0.5
        oy = tex_oy + ly + 0.5
        oz = tex_oz + lz + 0.5
        v000 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz), dtype=float)
        v100 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz), dtype=float)
        v010 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz), dtype=float)
        v110 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz), dtype=float)
        v001 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz + 1.0), dtype=float)
        v101 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz + 1.0), dtype=float)
        v011 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz + 1.0), dtype=float)
        v111 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz + 1.0), dtype=float)
        v000 = apply_subgrid_sdf_scale(v000, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v100 = apply_subgrid_sdf_scale(v100, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v010 = apply_subgrid_sdf_scale(v010, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v110 = apply_subgrid_sdf_scale(v110, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v001 = apply_subgrid_sdf_scale(v001, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v101 = apply_subgrid_sdf_scale(v101, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v011 = apply_subgrid_sdf_scale(v011, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v111 = apply_subgrid_sdf_scale(v111, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)

    corners = vec8f(v000, v100, v010, v110, v001, v101, v011, v111)
    return corners, tx, ty, tz


@wp.func
def _trilinear(corners: vec8f, tx: float, ty: float, tz: float) -> float:
    """Trilinear interpolation from 8 corner values and fractional weights."""
    c00 = corners[0] + (corners[1] - corners[0]) * tx
    c10 = corners[2] + (corners[3] - corners[2]) * tx
    c01 = corners[4] + (corners[5] - corners[4]) * tx
    c11 = corners[6] + (corners[7] - corners[6]) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    return c0 + (c1 - c0) * tz


@wp.func
def texture_sample_sdf_at_voxel(
    sdf: TextureSDFData,
    ix: int,
    iy: int,
    iz: int,
) -> float:
    """Sample SDF at an exact integer fine-grid vertex with a single texel read.

    At integer grid coordinates the trilinear fractional weights are zero, so
    only the corner-0 texel contributes.  This replaces 8 texture reads with 1
    for the common subgrid case, which is the dominant path in hydroelastic
    marching-cubes corner evaluation.

    For coarse (``SLOT_LINEAR``) cells the value must still be interpolated
    from the coarse grid, so this falls back to :func:`texture_sample_sdf`.

    Args:
        sdf: texture SDF data
        ix: fine-grid x index
        iy: fine-grid y index
        iz: fine-grid z index

    Returns:
        Signed distance value [m].
    """
    coarse_x = sdf.coarse_texture.width - 1
    coarse_y = sdf.coarse_texture.height - 1
    coarse_z = sdf.coarse_texture.depth - 1

    x_base = wp.clamp(int(float(ix) * sdf.fine_to_coarse), 0, coarse_x - 1)
    y_base = wp.clamp(int(float(iy) * sdf.fine_to_coarse), 0, coarse_y - 1)
    z_base = wp.clamp(int(float(iz) * sdf.fine_to_coarse), 0, coarse_z - 1)

    start_slot = sdf.subgrid_start_slots[x_base, y_base, z_base]

    if start_slot < wp.static(SLOT_LINEAR):
        block_x = float(start_slot & wp.uint32(0x3FF))
        block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
        block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))

        lx = float(ix) - float(x_base) * sdf.subgrid_size_f
        ly = float(iy) - float(y_base) * sdf.subgrid_size_f
        lz = float(iz) - float(z_base) * sdf.subgrid_size_f

        ox = block_x * sdf.subgrid_samples_f + lx + 0.5
        oy = block_y * sdf.subgrid_samples_f + ly + 0.5
        oz = block_z * sdf.subgrid_samples_f + lz + 0.5

        raw = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz), dtype=float)
        return raw * sdf.subgrids_sdf_value_range + sdf.subgrids_min_sdf_value

    local_pos = sdf.sdf_box_lower + wp.cw_mul(
        wp.vec3(float(ix), float(iy), float(iz)),
        sdf.voxel_size,
    )
    return texture_sample_sdf(sdf, local_pos)


@wp.func
def texture_sample_sdf(
    sdf: TextureSDFData,
    local_pos: wp.vec3,
) -> float:
    """Sample SDF value from texture with extrapolation for out-of-bounds points.

    Uses manual float32 trilinear interpolation from 8 corner texel reads
    to avoid CUDA hardware texture filtering precision issues (8-bit
    fixed-point interpolation weights that cause jitter in contact forces).

    Fuses cell lookup, texel reads, trilinear blend, and quantization
    de-scale into a single pass for the value-only path.

    Args:
        sdf: texture SDF data
        local_pos: query position in local SDF space [m]

    Returns:
        Signed distance value [m].
    """
    clamped = wp.vec3(
        wp.clamp(local_pos[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(local_pos[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(local_pos[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    diff_mag = wp.length(local_pos - clamped)

    f = wp.cw_mul(clamped - sdf.sdf_box_lower, sdf.inv_sdf_dx)

    coarse_x = sdf.coarse_texture.width - 1
    coarse_y = sdf.coarse_texture.height - 1
    coarse_z = sdf.coarse_texture.depth - 1

    fine_verts_x = float(coarse_x) * sdf.subgrid_size_f
    fine_verts_y = float(coarse_y) * sdf.subgrid_size_f
    fine_verts_z = float(coarse_z) * sdf.subgrid_size_f

    fx = wp.clamp(f[0], 0.0, fine_verts_x)
    fy = wp.clamp(f[1], 0.0, fine_verts_y)
    fz = wp.clamp(f[2], 0.0, fine_verts_z)

    num_fine_cells_x = int(fine_verts_x)
    num_fine_cells_y = int(fine_verts_y)
    num_fine_cells_z = int(fine_verts_z)
    ix = wp.clamp(int(wp.floor(fx)), 0, num_fine_cells_x - 1)
    iy = wp.clamp(int(wp.floor(fy)), 0, num_fine_cells_y - 1)
    iz = wp.clamp(int(wp.floor(fz)), 0, num_fine_cells_z - 1)
    tx = fx - float(ix)
    ty = fy - float(iy)
    tz = fz - float(iz)

    x_base = wp.clamp(int(float(ix) * sdf.fine_to_coarse), 0, coarse_x - 1)
    y_base = wp.clamp(int(float(iy) * sdf.fine_to_coarse), 0, coarse_y - 1)
    z_base = wp.clamp(int(float(iz) * sdf.fine_to_coarse), 0, coarse_z - 1)

    start_slot = sdf.subgrid_start_slots[x_base, y_base, z_base]

    v000 = float(0.0)
    v100 = float(0.0)
    v010 = float(0.0)
    v110 = float(0.0)
    v001 = float(0.0)
    v101 = float(0.0)
    v011 = float(0.0)
    v111 = float(0.0)

    needs_scale = False

    if start_slot >= wp.static(SLOT_LINEAR):
        cx = float(x_base)
        cy = float(y_base)
        cz = float(z_base)
        coarse_f = wp.vec3(fx, fy, fz) * sdf.fine_to_coarse
        tx = coarse_f[0] - cx
        ty = coarse_f[1] - cy
        tz = coarse_f[2] - cz
        v000 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 0.5), dtype=float)
        v100 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 0.5), dtype=float)
        v010 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 0.5), dtype=float)
        v110 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 0.5), dtype=float)
        v001 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 1.5), dtype=float)
        v101 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 1.5), dtype=float)
        v011 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 1.5), dtype=float)
        v111 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 1.5), dtype=float)
    else:
        needs_scale = True
        block_x = float(start_slot & wp.uint32(0x3FF))
        block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
        block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))
        tex_ox = block_x * sdf.subgrid_samples_f
        tex_oy = block_y * sdf.subgrid_samples_f
        tex_oz = block_z * sdf.subgrid_samples_f
        lx = float(ix) - float(x_base) * sdf.subgrid_size_f
        ly = float(iy) - float(y_base) * sdf.subgrid_size_f
        lz = float(iz) - float(z_base) * sdf.subgrid_size_f
        ox = tex_ox + lx + 0.5
        oy = tex_oy + ly + 0.5
        oz = tex_oz + lz + 0.5
        v000 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz), dtype=float)
        v100 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz), dtype=float)
        v010 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz), dtype=float)
        v110 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz), dtype=float)
        v001 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz + 1.0), dtype=float)
        v101 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz + 1.0), dtype=float)
        v011 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz + 1.0), dtype=float)
        v111 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz + 1.0), dtype=float)

    c00 = v000 + (v100 - v000) * tx
    c10 = v010 + (v110 - v010) * tx
    c01 = v001 + (v101 - v001) * tx
    c11 = v011 + (v111 - v011) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    sdf_val = c0 + (c1 - c0) * tz

    if needs_scale:
        sdf_val = sdf_val * sdf.subgrids_sdf_value_range + sdf.subgrids_min_sdf_value

    return sdf_val + diff_mag


@wp.func
def texture_sample_sdf_grad(
    sdf: TextureSDFData,
    local_pos: wp.vec3,
) -> tuple[float, wp.vec3]:
    """Sample SDF value and gradient using analytical trilinear from 8 corner texels.

    Uses :func:`_read_cell_corners` for point-sampled texel reads and performs
    both trilinear interpolation and analytical gradient computation in float32.

    Args:
        sdf: texture SDF data
        local_pos: query position in local SDF space [m]

    Returns:
        Tuple of (distance [m], gradient [unitless]).
    """
    clamped = wp.vec3(
        wp.clamp(local_pos[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(local_pos[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(local_pos[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    diff = local_pos - clamped
    diff_mag = wp.length(diff)

    f = wp.cw_mul(clamped - sdf.sdf_box_lower, sdf.inv_sdf_dx)
    corners, tx, ty, tz = _read_cell_corners(sdf, f)

    sdf_val = _trilinear(corners, tx, ty, tz)

    # Analytical gradient (partial derivatives of trilinear)
    omtx = 1.0 - tx
    omty = 1.0 - ty
    omtz = 1.0 - tz

    v000 = corners[0]
    v100 = corners[1]
    v010 = corners[2]
    v110 = corners[3]
    v001 = corners[4]
    v101 = corners[5]
    v011 = corners[6]
    v111 = corners[7]

    gx = omty * omtz * (v100 - v000) + ty * omtz * (v110 - v010) + omty * tz * (v101 - v001) + ty * tz * (v111 - v011)
    gy = omtx * omtz * (v010 - v000) + tx * omtz * (v110 - v100) + omtx * tz * (v011 - v001) + tx * tz * (v111 - v101)
    gz = omtx * omty * (v001 - v000) + tx * omty * (v101 - v100) + omtx * ty * (v011 - v010) + tx * ty * (v111 - v110)

    grad = wp.cw_mul(wp.vec3(gx, gy, gz), sdf.inv_sdf_dx)

    if diff_mag > 0.0:
        sdf_val = sdf_val + diff_mag
        grad = diff / diff_mag

    return sdf_val, grad


# ============================================================================
# Host-side Construction Functions
# ============================================================================


def build_sparse_sdf_from_mesh(
    mesh: wp.Mesh,
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
    cell_size: np.ndarray,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    subgrid_size: int = 8,
    narrow_band_thickness: float = 0.1,
    quantization_mode: int = QuantizationMode.UINT16,
    winding_threshold: float = 0.5,
    linearization_error_threshold: float | None = None,
    device: str = "cuda",
) -> dict:
    """Build sparse SDF texture representation by querying mesh directly.

    Mirrors the NanoVDB sparse-volume construction pattern: check subgrid
    occupancy at centers, then populate only occupied subgrids.

    Args:
        mesh: Warp mesh with ``support_winding_number=True``.
        grid_size_x: fine grid X dimension [sample].
        grid_size_y: fine grid Y dimension [sample].
        grid_size_z: fine grid Z dimension [sample].
        cell_size: fine grid cell size per axis [m].
        min_corner: lower corner of domain [m].
        max_corner: upper corner of domain [m].
        subgrid_size: cells per subgrid.
        narrow_band_thickness: distance threshold for subgrids [m].
        quantization_mode: :class:`QuantizationMode` value.
        winding_threshold: winding number threshold for inside/outside.
        linearization_error_threshold: maximum absolute SDF error [m] below
            which an occupied subgrid is considered linear and its high-res
            data is omitted.  ``None`` auto-computes from domain extents,
            ``0.0`` disables the optimization.
        device: Warp device string.

    Returns:
        Dictionary with all sparse SDF data.
    """
    # Ceiling division ensures the subgrid grid fully covers the fine grid.
    # Floor division can truncate the domain when the number of fine cells
    # is not a multiple of subgrid_size, leaving narrow-band regions without
    # subgrid coverage.
    num_cells_x = grid_size_x - 1
    num_cells_y = grid_size_y - 1
    num_cells_z = grid_size_z - 1
    w = (num_cells_x + subgrid_size - 1) // subgrid_size
    h = (num_cells_y + subgrid_size - 1) // subgrid_size
    d = (num_cells_z + subgrid_size - 1) // subgrid_size
    total_subgrids = w * h * d

    min_corner_wp = wp.vec3(float(min_corner[0]), float(min_corner[1]), float(min_corner[2]))
    cell_size_wp = wp.vec3(float(cell_size[0]), float(cell_size[1]), float(cell_size[2]))

    # Build background SDF (coarse grid) by querying mesh at subgrid corners
    bg_size_x = w + 1
    bg_size_y = h + 1
    bg_size_z = d + 1
    total_bg = bg_size_x * bg_size_y * bg_size_z

    background_sdf = wp.zeros(total_bg, dtype=float, device=device)

    wp.launch(
        _build_coarse_sdf_from_mesh_kernel,
        dim=total_bg,
        inputs=[
            mesh.id,
            background_sdf,
            min_corner_wp,
            cell_size_wp,
            subgrid_size,
            bg_size_x,
            bg_size_y,
            bg_size_z,
            winding_threshold,
        ],
        device=device,
    )

    # Check subgrid occupancy by querying mesh SDF at subgrid centers
    subgrid_centers = np.empty((total_subgrids, 3), dtype=np.float32)
    for idx in range(total_subgrids):
        bz = idx // (w * h)
        rem = idx - bz * w * h
        by = rem // w
        bx = rem - by * w
        subgrid_centers[idx, 0] = (bx * subgrid_size + subgrid_size * 0.5) * cell_size[0] + min_corner[0]
        subgrid_centers[idx, 1] = (by * subgrid_size + subgrid_size * 0.5) * cell_size[1] + min_corner[1]
        subgrid_centers[idx, 2] = (bz * subgrid_size + subgrid_size * 0.5) * cell_size[2] + min_corner[2]

    subgrid_centers_gpu = wp.array(subgrid_centers, dtype=wp.vec3, device=device)

    # Expand threshold by subgrid half-diagonal (same pattern as NanoVDB tiles)
    half_subgrid = subgrid_size * 0.5 * cell_size
    subgrid_radius = float(np.linalg.norm(half_subgrid))
    threshold = wp.vec2f(-narrow_band_thickness - subgrid_radius, narrow_band_thickness + subgrid_radius)

    subgrid_required = wp.zeros(total_subgrids, dtype=wp.int32, device=device)

    wp.launch(
        _check_subgrid_occupied_kernel,
        dim=total_subgrids,
        inputs=[mesh.id, subgrid_centers_gpu, threshold, winding_threshold, subgrid_required],
        device=device,
    )
    wp.synchronize()

    # Snapshot the full narrow-band occupancy before linearization demotes
    # some subgrids.  The hydroelastic broadphase needs block coordinates
    # for ALL occupied subgrids (including linear ones) so the octree
    # explores the full contact surface.
    subgrid_occupied = subgrid_required.numpy().copy()

    # Demote occupied subgrids whose SDF is well-approximated by the coarse
    # grid (linear field).  Demoted subgrids get 0xFFFFFFFE instead of a
    # subgrid address, saving texture memory while preserving narrow-band
    # membership information.
    if linearization_error_threshold is None:
        extents = max_corner - min_corner
        linearization_error_threshold = float(1e-6 * np.linalg.norm(extents))
    subgrid_is_linear = wp.zeros(total_subgrids, dtype=wp.int32, device=device)
    if linearization_error_threshold > 0.0:
        wp.launch(
            _check_subgrid_linearity_kernel,
            dim=total_subgrids,
            inputs=[
                mesh.id,
                background_sdf,
                subgrid_required,
                subgrid_is_linear,
                subgrid_size,
                min_corner_wp,
                cell_size_wp,
                winding_threshold,
                w,
                h,
                bg_size_x,
                bg_size_y,
                bg_size_z,
                linearization_error_threshold,
            ],
            device=device,
        )
        wp.synchronize()

    # Exclusive scan to assign sequential addresses to required subgrids
    subgrid_addresses = wp.zeros(total_subgrids, dtype=wp.int32, device=device)
    wp._src.utils.array_scan(subgrid_required, subgrid_addresses, inclusive=False)
    wp.synchronize()

    required_np = subgrid_required.numpy()
    num_required = int(np.sum(required_np))

    # Conservative quantization bounds from narrow band range
    global_sdf_min = -narrow_band_thickness - subgrid_radius
    global_sdf_max = narrow_band_thickness + subgrid_radius
    sdf_range = global_sdf_max - global_sdf_min
    if sdf_range < 1e-10:
        sdf_range = 1.0

    if num_required == 0:
        subgrid_start_slots = np.full((w, h, d), SLOT_EMPTY, dtype=np.uint32)
        subgrid_texture_data = np.zeros((1, 1, 1), dtype=np.float32)
        tex_size = 1
        final_sdf_min = 0.0
        final_sdf_range = 1.0
    else:
        cubic_root = num_required ** (1.0 / 3.0)
        tex_blocks_per_dim = max(1, int(np.ceil(cubic_root)))
        while tex_blocks_per_dim**3 < num_required:
            tex_blocks_per_dim += 1

        samples_per_dim = subgrid_size + 1
        tex_size = tex_blocks_per_dim * samples_per_dim

        subgrid_start_slots = np.full((w, h, d), SLOT_EMPTY, dtype=np.uint32)
        subgrid_start_slots_gpu = wp.array(subgrid_start_slots, dtype=wp.uint32, device=device)

        total_tex_samples = tex_size * tex_size * tex_size
        samples_per_subgrid = samples_per_dim**3
        total_work = total_subgrids * samples_per_subgrid

        sdf_range_inv = 1.0 / sdf_range

        if quantization_mode == QuantizationMode.FLOAT32:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=float, device=device)
            wp.launch(
                _populate_subgrid_texture_float32_kernel,
                dim=total_work,
                inputs=[
                    mesh.id,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    min_corner_wp,
                    cell_size_wp,
                    winding_threshold,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                ],
                device=device,
            )
            final_sdf_min = 0.0
            final_sdf_range = 1.0
            subgrid_texture_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))

        elif quantization_mode == QuantizationMode.UINT16:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=wp.uint16, device=device)
            wp.launch(
                _populate_subgrid_texture_uint16_kernel,
                dim=total_work,
                inputs=[
                    mesh.id,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    min_corner_wp,
                    cell_size_wp,
                    winding_threshold,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                    global_sdf_min,
                    sdf_range_inv,
                ],
                device=device,
            )
            final_sdf_min = global_sdf_min
            final_sdf_range = sdf_range
            subgrid_texture_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))

        elif quantization_mode == QuantizationMode.UINT8:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=wp.uint8, device=device)
            wp.launch(
                _populate_subgrid_texture_uint8_kernel,
                dim=total_work,
                inputs=[
                    mesh.id,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    min_corner_wp,
                    cell_size_wp,
                    winding_threshold,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                    global_sdf_min,
                    sdf_range_inv,
                ],
                device=device,
            )
            final_sdf_min = global_sdf_min
            final_sdf_range = sdf_range
            subgrid_texture_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))

        else:
            raise ValueError(f"Unknown quantization mode: {quantization_mode}")

        wp.synchronize()
        subgrid_start_slots = subgrid_start_slots_gpu.numpy()

    # Write SLOT_LINEAR for subgrids that overlap the narrow band but were
    # demoted because their SDF is well-approximated by the coarse grid.
    is_linear_np = subgrid_is_linear.numpy()
    if np.any(is_linear_np):
        for idx in range(total_subgrids):
            if is_linear_np[idx]:
                bz = idx // (w * h)
                rem = idx - bz * w * h
                by = rem // w
                bx = rem - by * w
                subgrid_start_slots[bx, by, bz] = SLOT_LINEAR

    background_sdf_np = background_sdf.numpy().reshape((bg_size_z, bg_size_y, bg_size_x))

    # Padded max covers the full ceiling-divided subgrid grid.
    padded_max = min_corner + np.array([w, h, d], dtype=float) * subgrid_size * cell_size

    return {
        "coarse_sdf": background_sdf_np.astype(np.float32),
        "subgrid_data": subgrid_texture_data,
        "subgrid_start_slots": subgrid_start_slots,
        "coarse_dims": (w, h, d),
        "subgrid_tex_size": tex_size,
        "num_subgrids": num_required,
        "min_extents": min_corner,
        "max_extents": padded_max,
        "cell_size": cell_size,
        "subgrid_size": subgrid_size,
        "quantization_mode": quantization_mode,
        "subgrids_min_sdf_value": final_sdf_min,
        "subgrids_sdf_value_range": final_sdf_range,
        "subgrid_required": required_np,
        "subgrid_occupied": subgrid_occupied,
    }


def create_sparse_sdf_textures(
    sparse_data: dict,
    device: str = "cuda",
) -> tuple[TextureSDFData, wp.Texture3D, wp.Texture3D]:
    """Create TextureSDFData struct with GPU textures from sparse data.

    Args:
        sparse_data: dictionary from :func:`build_sparse_sdf_from_mesh`.
        device: Warp device string.

    Returns:
        Tuple of ``(texture_sdf, coarse_texture, subgrid_texture)``.
        Caller must keep texture references alive to prevent GC.
    """
    coarse_tex = wp.Texture3D(
        sparse_data["coarse_sdf"],
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    subgrid_tex = wp.Texture3D(
        sparse_data["subgrid_data"],
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    subgrid_slots = wp.array(sparse_data["subgrid_start_slots"], dtype=wp.uint32, device=device)

    cell_size = sparse_data["cell_size"]

    min_ext = sparse_data["min_extents"]
    max_ext = sparse_data["max_extents"]

    sdf_params = TextureSDFData()
    sdf_params.coarse_texture = coarse_tex
    sdf_params.subgrid_texture = subgrid_tex
    sdf_params.subgrid_start_slots = subgrid_slots
    sdf_params.sdf_box_lower = wp.vec3(float(min_ext[0]), float(min_ext[1]), float(min_ext[2]))
    sdf_params.sdf_box_upper = wp.vec3(float(max_ext[0]), float(max_ext[1]), float(max_ext[2]))
    sdf_params.inv_sdf_dx = wp.vec3(1.0 / float(cell_size[0]), 1.0 / float(cell_size[1]), 1.0 / float(cell_size[2]))
    sdf_params.subgrid_size = sparse_data["subgrid_size"]
    sdf_params.subgrid_size_f = float(sparse_data["subgrid_size"])
    sdf_params.subgrid_samples_f = float(sparse_data["subgrid_size"] + 1)
    sdf_params.fine_to_coarse = 1.0 / sparse_data["subgrid_size"]

    sdf_params.voxel_size = wp.vec3(float(cell_size[0]), float(cell_size[1]), float(cell_size[2]))
    sdf_params.voxel_radius = float(0.5 * np.linalg.norm(cell_size))

    sdf_params.subgrids_min_sdf_value = sparse_data["subgrids_min_sdf_value"]
    sdf_params.subgrids_sdf_value_range = sparse_data["subgrids_sdf_value_range"]
    sdf_params.scale_baked = False

    return sdf_params, coarse_tex, subgrid_tex


def create_texture_sdf_from_mesh(
    mesh: wp.Mesh,
    *,
    margin: float = 0.05,
    narrow_band_range: tuple[float, float] = (-0.1, 0.1),
    max_resolution: int = 64,
    subgrid_size: int = 8,
    quantization_mode: int = QuantizationMode.UINT16,
    winding_threshold: float = 0.5,
    scale_baked: bool = False,
    device: str | None = None,
) -> tuple[TextureSDFData, wp.Texture3D, wp.Texture3D, list]:
    """Create texture SDF from a Warp mesh.

    This is the main entry point for texture SDF construction. It mirrors the
    parameters of :func:`~newton._src.geometry.sdf_utils._compute_sdf_from_shape_impl`.

    Args:
        mesh: Warp mesh with ``support_winding_number=True``.
        margin: extra AABB padding [m].
        narrow_band_range: signed narrow-band distance range [m] as ``(inner, outer)``.
        max_resolution: maximum grid dimension [voxel].
        subgrid_size: cells per subgrid.
        quantization_mode: :class:`QuantizationMode` value.
        winding_threshold: winding number threshold for inside/outside classification.
        scale_baked: whether shape scale was baked into the mesh vertices.
        device: Warp device string. ``None`` uses the mesh's device.

    Returns:
        Tuple of ``(texture_sdf, coarse_texture, subgrid_texture, block_coords)``.
        Caller must keep texture references alive to prevent GC.
        ``block_coords`` is a list of ``wp.vec3us`` block coordinates for
        hydroelastic broadphase.
    """
    if device is None:
        device = str(mesh.device)

    points_np = mesh.points.numpy()
    mesh_min = np.min(points_np, axis=0)
    mesh_max = np.max(points_np, axis=0)

    min_ext = mesh_min - margin
    max_ext = mesh_max + margin

    # Compute grid dimensions (same math as the former build_dense_sdf)
    ext = max_ext - min_ext
    max_ext_scalar = np.max(ext)
    if max_ext_scalar < 1e-10:
        return create_empty_texture_sdf_data(), None, None, []
    cell_size_scalar = max_ext_scalar / max_resolution
    dims = np.ceil(ext / cell_size_scalar).astype(int) + 1
    grid_x, grid_y, grid_z = int(dims[0]), int(dims[1]), int(dims[2])
    cell_size = ext / (dims - 1)

    narrow_band_thickness = max(abs(narrow_band_range[0]), abs(narrow_band_range[1]))

    sparse_data = build_sparse_sdf_from_mesh(
        mesh,
        grid_x,
        grid_y,
        grid_z,
        cell_size,
        min_ext,
        max_ext,
        subgrid_size=subgrid_size,
        narrow_band_thickness=narrow_band_thickness,
        quantization_mode=quantization_mode,
        winding_threshold=winding_threshold,
        device=device,
    )

    sdf_params, coarse_tex, subgrid_tex = create_sparse_sdf_textures(sparse_data, device)
    sdf_params.scale_baked = scale_baked

    # Dilate the non-linear subgrid set by one ring of occupied neighbors so
    # the hydroelastic broadphase explores flat box faces (which are linear
    # but adjacent to non-linear edges/corners).
    block_coords = block_coords_from_subgrid_required(
        sparse_data["subgrid_required"],
        sparse_data["coarse_dims"],
        sparse_data["subgrid_size"],
        subgrid_occupied=sparse_data["subgrid_occupied"],
    )

    return sdf_params, coarse_tex, subgrid_tex, block_coords


def create_texture_sdf_from_volume(
    sparse_volume: wp.Volume,
    coarse_volume: wp.Volume,
    *,
    min_ext: np.ndarray,
    max_ext: np.ndarray,
    voxel_size: np.ndarray,
    narrow_band_range: tuple[float, float] = (-0.1, 0.1),
    subgrid_size: int = 8,
    scale_baked: bool = False,
    linearization_error_threshold: float | None = None,
    device: str = "cuda",
) -> tuple[TextureSDFData, wp.Texture3D, wp.Texture3D]:
    """Create texture SDF from existing NanoVDB sparse and coarse volumes.

    Samples the NanoVDB volumes at each texel position to build the texture SDF.
    This is used during construction for primitive shapes that already have NanoVDB
    volumes but need texture SDFs for the collision pipeline.

    Args:
        sparse_volume: NanoVDB sparse volume with SDF values.
        coarse_volume: NanoVDB coarse (background) volume with SDF values.
        min_ext: lower corner of the SDF domain [m].
        max_ext: upper corner of the SDF domain [m].
        voxel_size: fine grid cell size per axis [m].
        narrow_band_range: signed narrow-band distance range [m] as ``(inner, outer)``.
        subgrid_size: cells per subgrid.
        scale_baked: whether shape scale was baked into the SDF.
        linearization_error_threshold: maximum absolute SDF error [m] below
            which an occupied subgrid is considered linear and its high-res
            data is omitted.  ``None`` auto-computes from domain extents,
            ``0.0`` disables the optimization.
        device: Warp device string.

    Returns:
        Tuple of ``(texture_sdf, coarse_texture, subgrid_texture)``.
        Caller must keep texture references alive to prevent GC.
    """
    ext = max_ext - min_ext
    # Compute fine grid dimensions from extents and voxel size.
    # Use ceiling division so the coarse grid fully covers the NanoVDB domain.
    cells_per_axis = np.round(ext / voxel_size).astype(int)
    w = int((cells_per_axis[0] + subgrid_size - 1) // subgrid_size)
    h = int((cells_per_axis[1] + subgrid_size - 1) // subgrid_size)
    d = int((cells_per_axis[2] + subgrid_size - 1) // subgrid_size)
    total_subgrids = w * h * d

    # Padded grid covers w*subgrid_size cells (+ 1 vertex) per axis.
    # Keep cell_size = voxel_size so voxel indices map 1:1.
    cell_size = voxel_size.copy()
    padded_max = min_ext + np.array([w, h, d], dtype=float) * subgrid_size * cell_size

    # Build background/coarse SDF by sampling coarse volume at subgrid corners
    bg_size_x = w + 1
    bg_size_y = h + 1
    bg_size_z = d + 1
    total_bg = bg_size_x * bg_size_y * bg_size_z

    # Sample coarse grid from the coarse NanoVDB volume using a GPU kernel
    bg_positions = np.zeros((total_bg, 3), dtype=np.float32)
    for idx in range(total_bg):
        z_block = idx // (bg_size_x * bg_size_y)
        rem = idx - z_block * bg_size_x * bg_size_y
        y_block = rem // bg_size_x
        x_block = rem - y_block * bg_size_x
        bg_positions[idx] = min_ext + np.array(
            [
                float(x_block * subgrid_size) * cell_size[0],
                float(y_block * subgrid_size) * cell_size[1],
                float(z_block * subgrid_size) * cell_size[2],
            ]
        )

    bg_positions_gpu = wp.array(bg_positions, dtype=wp.vec3, device=device)
    bg_sdf_gpu = wp.zeros(total_bg, dtype=float, device=device)
    wp.launch(
        _sample_volume_at_positions_kernel,
        dim=total_bg,
        inputs=[coarse_volume.id, bg_positions_gpu, bg_sdf_gpu],
        device=device,
    )

    # Check subgrid occupancy by sampling sparse volume at subgrid centers
    narrow_band_thickness = max(abs(narrow_band_range[0]), abs(narrow_band_range[1]))
    half_subgrid = subgrid_size * 0.5 * cell_size
    subgrid_radius = float(np.linalg.norm(half_subgrid))

    subgrid_centers = np.empty((total_subgrids, 3), dtype=np.float32)
    for idx in range(total_subgrids):
        bz = idx // (w * h)
        rem = idx - bz * w * h
        by = rem // w
        bx = rem - by * w
        subgrid_centers[idx, 0] = (bx * subgrid_size + subgrid_size * 0.5) * cell_size[0] + min_ext[0]
        subgrid_centers[idx, 1] = (by * subgrid_size + subgrid_size * 0.5) * cell_size[1] + min_ext[1]
        subgrid_centers[idx, 2] = (bz * subgrid_size + subgrid_size * 0.5) * cell_size[2] + min_ext[2]

    center_positions_gpu = wp.array(subgrid_centers, dtype=wp.vec3, device=device)
    center_sdf_gpu = wp.zeros(total_subgrids, dtype=float, device=device)
    wp.launch(
        _sample_volume_at_positions_kernel,
        dim=total_subgrids,
        inputs=[sparse_volume.id, center_positions_gpu, center_sdf_gpu],
        device=device,
    )
    wp.synchronize()

    center_sdf_np = center_sdf_gpu.numpy()
    threshold_inner = -narrow_band_thickness - subgrid_radius
    threshold_outer = narrow_band_thickness + subgrid_radius

    subgrid_required = np.zeros(total_subgrids, dtype=np.int32)
    for idx in range(total_subgrids):
        val = center_sdf_np[idx]
        if val > 0:
            subgrid_required[idx] = 1 if val < threshold_outer else 0
        else:
            subgrid_required[idx] = 1 if val > threshold_inner else 0

    # Snapshot occupancy before linearization (see GPU path comment).
    subgrid_occupied = subgrid_required.copy()

    # Demote occupied subgrids whose SDF is well-approximated by the coarse
    # grid (linear field).
    if linearization_error_threshold is None:
        linearization_error_threshold = float(1e-6 * np.linalg.norm(ext))
    subgrid_is_linear = np.zeros(total_subgrids, dtype=np.int32)
    if linearization_error_threshold > 0.0:
        bg_sdf_np = bg_sdf_gpu.numpy()
        samples_per_dim_lin = subgrid_size + 1
        s_inv = 1.0 / float(subgrid_size)

        occupied_indices = np.nonzero(subgrid_required)[0]
        if len(occupied_indices) > 0:
            all_positions = []
            for idx in occupied_indices:
                bz = idx // (w * h)
                rem = idx - bz * w * h
                by = rem // w
                bx = rem - by * w
                for lz in range(samples_per_dim_lin):
                    for ly in range(samples_per_dim_lin):
                        for lx in range(samples_per_dim_lin):
                            gx = bx * subgrid_size + lx
                            gy = by * subgrid_size + ly
                            gz = bz * subgrid_size + lz
                            pos = min_ext + np.array(
                                [
                                    float(gx) * cell_size[0],
                                    float(gy) * cell_size[1],
                                    float(gz) * cell_size[2],
                                ]
                            )
                            all_positions.append(pos)

            all_positions_gpu = wp.array(np.array(all_positions, dtype=np.float32), dtype=wp.vec3, device=device)
            all_sdf_gpu = wp.zeros(len(all_positions), dtype=float, device=device)
            wp.launch(
                _sample_volume_at_positions_kernel,
                dim=len(all_positions),
                inputs=[sparse_volume.id, all_positions_gpu, all_sdf_gpu],
                device=device,
            )
            wp.synchronize()
            all_sdf_np = all_sdf_gpu.numpy()

            samples_per_subgrid = samples_per_dim_lin**3
            for i, idx in enumerate(occupied_indices):
                bz_i = idx // (w * h)
                rem_i = idx - bz_i * w * h
                by_i = rem_i // w
                bx_i = rem_i - by_i * w
                max_err = 0.0
                base = i * samples_per_subgrid
                for lz in range(samples_per_dim_lin):
                    for ly in range(samples_per_dim_lin):
                        for lx in range(samples_per_dim_lin):
                            local_idx = lz * samples_per_dim_lin * samples_per_dim_lin + ly * samples_per_dim_lin + lx
                            vol_val = all_sdf_np[base + local_idx]

                            cfx = float(bx_i) + float(lx) * s_inv
                            cfy = float(by_i) + float(ly) * s_inv
                            cfz = float(bz_i) + float(lz) * s_inv

                            x0 = max(0, min(int(np.floor(cfx)), bg_size_x - 2))
                            y0 = max(0, min(int(np.floor(cfy)), bg_size_y - 2))
                            z0 = max(0, min(int(np.floor(cfz)), bg_size_z - 2))
                            tx = np.clip(cfx - float(x0), 0.0, 1.0)
                            ty = np.clip(cfy - float(y0), 0.0, 1.0)
                            tz = np.clip(cfz - float(z0), 0.0, 1.0)

                            def _bg(xi, yi, zi):
                                return float(bg_sdf_np[zi * bg_size_x * bg_size_y + yi * bg_size_x + xi])

                            c00 = _bg(x0, y0, z0) * (1.0 - tx) + _bg(x0 + 1, y0, z0) * tx
                            c10 = _bg(x0, y0 + 1, z0) * (1.0 - tx) + _bg(x0 + 1, y0 + 1, z0) * tx
                            c01 = _bg(x0, y0, z0 + 1) * (1.0 - tx) + _bg(x0 + 1, y0, z0 + 1) * tx
                            c11 = _bg(x0, y0 + 1, z0 + 1) * (1.0 - tx) + _bg(x0 + 1, y0 + 1, z0 + 1) * tx
                            c0 = c00 * (1.0 - ty) + c10 * ty
                            c1 = c01 * (1.0 - ty) + c11 * ty
                            coarse_val = c0 * (1.0 - tz) + c1 * tz

                            max_err = max(max_err, abs(vol_val - coarse_val))

                if max_err < linearization_error_threshold:
                    subgrid_is_linear[idx] = 1
                    subgrid_required[idx] = 0

    num_required = int(np.sum(subgrid_required))

    # Conservative quantization bounds from narrow band range
    global_sdf_min = threshold_inner
    global_sdf_max = threshold_outer
    sdf_range = global_sdf_max - global_sdf_min
    if sdf_range < 1e-10:
        sdf_range = 1.0

    if num_required == 0:
        subgrid_start_slots = np.full((w, h, d), SLOT_EMPTY, dtype=np.uint32)
        subgrid_texture_data = np.zeros((1, 1, 1), dtype=np.float32)
        tex_size = 1
    else:
        cubic_root = num_required ** (1.0 / 3.0)
        tex_blocks_per_dim = max(1, int(np.ceil(cubic_root)))
        while tex_blocks_per_dim**3 < num_required:
            tex_blocks_per_dim += 1

        samples_per_dim = subgrid_size + 1
        tex_size = tex_blocks_per_dim * samples_per_dim

        # Assign sequential addresses to required subgrids
        subgrid_start_slots = np.full((w, h, d), SLOT_EMPTY, dtype=np.uint32)
        address = 0
        for idx in range(total_subgrids):
            if subgrid_required[idx]:
                addr_z = address // (tex_blocks_per_dim * tex_blocks_per_dim)
                addr_rem = address - addr_z * tex_blocks_per_dim * tex_blocks_per_dim
                addr_y = addr_rem // tex_blocks_per_dim
                addr_x = addr_rem - addr_y * tex_blocks_per_dim
                bz = idx // (w * h)
                rem = idx - bz * w * h
                by = rem // w
                bx = rem - by * w
                subgrid_start_slots[bx, by, bz] = int(addr_x) | (int(addr_y) << 10) | (int(addr_z) << 20)
                address += 1

        # Build positions array for all subgrid texels, then sample volume
        total_texel_work = num_required * samples_per_dim**3
        texel_positions = np.empty((total_texel_work, 3), dtype=np.float32)
        texel_tex_indices = np.empty(total_texel_work, dtype=np.int32)

        work_idx = 0
        subgrid_texture_data = np.zeros((tex_size, tex_size, tex_size), dtype=np.float32)
        for sg_idx in range(total_subgrids):
            if not subgrid_required[sg_idx]:
                continue
            sg_z = sg_idx // (w * h)
            sg_rem = sg_idx - sg_z * w * h
            sg_y = sg_rem // w
            sg_x = sg_rem - sg_y * w

            slot = subgrid_start_slots[sg_x, sg_y, sg_z]
            addr_x = int(slot & 0x3FF)
            addr_y = int((slot >> 10) & 0x3FF)
            addr_z = int((slot >> 20) & 0x3FF)

            for lz in range(samples_per_dim):
                for ly in range(samples_per_dim):
                    for lx in range(samples_per_dim):
                        gx = sg_x * subgrid_size + lx
                        gy = sg_y * subgrid_size + ly
                        gz = sg_z * subgrid_size + lz
                        pos = min_ext + np.array(
                            [
                                float(gx) * cell_size[0],
                                float(gy) * cell_size[1],
                                float(gz) * cell_size[2],
                            ]
                        )
                        tex_x = addr_x * samples_per_dim + lx
                        tex_y = addr_y * samples_per_dim + ly
                        tex_z = addr_z * samples_per_dim + lz
                        texel_positions[work_idx] = pos
                        texel_tex_indices[work_idx] = tex_z * tex_size * tex_size + tex_y * tex_size + tex_x
                        work_idx += 1

        # Sample all texel positions from the sparse volume on GPU
        texel_positions_gpu = wp.array(texel_positions, dtype=wp.vec3, device=device)
        texel_sdf_gpu = wp.zeros(total_texel_work, dtype=float, device=device)
        wp.launch(
            _sample_volume_at_positions_kernel,
            dim=total_texel_work,
            inputs=[sparse_volume.id, texel_positions_gpu, texel_sdf_gpu],
            device=device,
        )
        wp.synchronize()

        texel_sdf_np = texel_sdf_gpu.numpy()

        # Replace background/corrupted values from sparse volume with
        # coarse volume samples.  The NanoVDB sparse volume uses 1e18 as
        # background for unallocated tiles; linear interpolation near tile
        # boundaries blends this background into texels, corrupting them.
        bg_threshold = threshold_outer * 2.0
        outlier_mask = (texel_sdf_np > bg_threshold) | (texel_sdf_np < -bg_threshold)
        if np.any(outlier_mask):
            outlier_positions = texel_positions[outlier_mask]
            outlier_gpu = wp.array(outlier_positions, dtype=wp.vec3, device=device)
            outlier_sdf_gpu = wp.zeros(len(outlier_positions), dtype=float, device=device)
            wp.launch(
                _sample_volume_at_positions_kernel,
                dim=len(outlier_positions),
                inputs=[coarse_volume.id, outlier_gpu, outlier_sdf_gpu],
                device=device,
            )
            wp.synchronize()
            texel_sdf_np[outlier_mask] = outlier_sdf_gpu.numpy()
        flat_texture = subgrid_texture_data.ravel()
        for i in range(total_texel_work):
            flat_texture[texel_tex_indices[i]] = texel_sdf_np[i]
        subgrid_texture_data = flat_texture.reshape((tex_size, tex_size, tex_size))

    wp.synchronize()

    # Write SLOT_LINEAR for subgrids that overlap the narrow band but were
    # demoted because their SDF is well-approximated by the coarse grid.
    if np.any(subgrid_is_linear):
        for idx in range(total_subgrids):
            if subgrid_is_linear[idx]:
                bz = idx // (w * h)
                rem = idx - bz * w * h
                by = rem // w
                bx = rem - by * w
                subgrid_start_slots[bx, by, bz] = SLOT_LINEAR

    background_sdf_np = bg_sdf_gpu.numpy().reshape((bg_size_z, bg_size_y, bg_size_x))

    sparse_data = {
        "coarse_sdf": background_sdf_np.astype(np.float32),
        "subgrid_data": subgrid_texture_data,
        "subgrid_start_slots": subgrid_start_slots,
        "coarse_dims": (w, h, d),
        "subgrid_tex_size": tex_size,
        "num_subgrids": num_required,
        "min_extents": min_ext,
        "max_extents": padded_max,
        "cell_size": cell_size,
        "subgrid_size": subgrid_size,
        "quantization_mode": QuantizationMode.FLOAT32,
        "subgrids_min_sdf_value": 0.0,
        "subgrids_sdf_value_range": 1.0,
        "subgrid_required": subgrid_required,
        "subgrid_occupied": subgrid_occupied,
    }

    sdf_params, coarse_tex, subgrid_tex = create_sparse_sdf_textures(sparse_data, device)
    sdf_params.scale_baked = scale_baked

    return sdf_params, coarse_tex, subgrid_tex


def block_coords_from_subgrid_required(
    subgrid_required: np.ndarray,
    coarse_dims: tuple[int, int, int],
    subgrid_size: int,
    subgrid_occupied: np.ndarray | None = None,
) -> list:
    """Derive SDF block coordinates from texture subgrid occupancy.

    This converts the texture subgrid occupancy array into voxel-space block
    coordinates compatible with the hydroelastic broadphase pipeline.

    When *subgrid_occupied* is supplied (the pre-linearization narrow-band
    mask), all occupied subgrids are included — matching the behavior of
    NanoVDB active tiles.  This ensures full contact surface coverage for
    flat faces that were demoted to linear interpolation.

    Args:
        subgrid_required: 1D array of occupancy flags for non-linear subgrids.
        coarse_dims: tuple (w, h, d) number of subgrids per axis.
        subgrid_size: cells per subgrid.
        subgrid_occupied: optional 1D array of full narrow-band occupancy
            (before linearization).  When provided, all occupied subgrids
            are included in the output.

    Returns:
        List of ``wp.vec3us`` block coordinates for selected subgrids.
    """
    w, h, _d = coarse_dims
    include = subgrid_occupied if subgrid_occupied is not None else subgrid_required

    coords = []
    for idx in range(len(include)):
        if include[idx]:
            bz = idx // (w * h)
            rem = idx - bz * w * h
            by = rem // w
            bx = rem - by * w
            coords.append(wp.vec3us(bx * subgrid_size, by * subgrid_size, bz * subgrid_size))
    return coords


def create_empty_texture_sdf_data() -> TextureSDFData:
    """Return an empty TextureSDFData struct for shapes without texture SDF.

    An empty struct has ``coarse_texture.width == 0``, which collision kernels
    use to detect the absence of a texture SDF and fall back to BVH.

    Returns:
        A zeroed-out :class:`TextureSDFData` struct.
    """
    sdf = TextureSDFData()
    sdf.subgrid_size = 0
    sdf.subgrid_size_f = 0.0
    sdf.subgrid_samples_f = 0.0
    sdf.fine_to_coarse = 0.0
    sdf.inv_sdf_dx = wp.vec3(0.0, 0.0, 0.0)
    sdf.sdf_box_lower = wp.vec3(0.0, 0.0, 0.0)
    sdf.sdf_box_upper = wp.vec3(0.0, 0.0, 0.0)
    sdf.voxel_size = wp.vec3(0.0, 0.0, 0.0)
    sdf.voxel_radius = 0.0
    sdf.subgrids_min_sdf_value = 0.0
    sdf.subgrids_sdf_value_range = 1.0
    sdf.scale_baked = False
    return sdf


# ============================================================================
# Isomesh extraction from texture SDF (marching cubes)
# ============================================================================


@wp.kernel(enable_backward=False)
def _count_isomesh_faces_texture_kernel(
    sdf_array: wp.array[TextureSDFData],
    active_coarse_cells: wp.array[wp.vec3i],
    subgrid_size: int,
    tri_range_table: wp.array[wp.int32],
    corner_offsets_table: wp.array[wp.vec3ub],
    isovalue: wp.float32,
    face_count: wp.array[int],
):
    cell_idx, local_x, local_y, local_z = wp.tid()
    sdf = sdf_array[0]
    coarse = active_coarse_cells[cell_idx]
    x_id = coarse[0] * subgrid_size + local_x
    y_id = coarse[1] * subgrid_size + local_y
    z_id = coarse[2] * subgrid_size + local_z

    cube_idx = wp.int32(0)
    for i in range(8):
        co = wp.vec3i(corner_offsets_table[i])
        v = texture_sample_sdf_at_voxel(sdf, x_id + co.x, y_id + co.y, z_id + co.z)
        if wp.isnan(v):
            return
        if v < isovalue:
            cube_idx |= 1 << i

    tri_start = tri_range_table[cube_idx]
    tri_end = tri_range_table[cube_idx + 1]
    num_faces = (tri_end - tri_start) // 3
    if num_faces > 0:
        wp.atomic_add(face_count, 0, num_faces)


@wp.kernel(enable_backward=False)
def _generate_isomesh_texture_kernel(
    sdf_array: wp.array[TextureSDFData],
    active_coarse_cells: wp.array[wp.vec3i],
    subgrid_size: int,
    tri_range_table: wp.array[wp.int32],
    flat_edge_verts_table: wp.array[wp.vec2ub],
    corner_offsets_table: wp.array[wp.vec3ub],
    isovalue: wp.float32,
    face_count: wp.array[int],
    vertices: wp.array[wp.vec3],
):
    cell_idx, local_x, local_y, local_z = wp.tid()
    sdf = sdf_array[0]
    coarse = active_coarse_cells[cell_idx]
    x_id = coarse[0] * subgrid_size + local_x
    y_id = coarse[1] * subgrid_size + local_y
    z_id = coarse[2] * subgrid_size + local_z

    cube_idx = wp.int32(0)
    corner_vals = vec8f()
    for i in range(8):
        co = wp.vec3i(corner_offsets_table[i])
        v = texture_sample_sdf_at_voxel(sdf, x_id + co.x, y_id + co.y, z_id + co.z)
        if wp.isnan(v):
            return
        corner_vals[i] = v
        if v < isovalue:
            cube_idx |= 1 << i

    tri_start = tri_range_table[cube_idx]
    tri_end = tri_range_table[cube_idx + 1]
    num_verts = tri_end - tri_start
    num_faces = num_verts // 3
    if num_faces == 0:
        return

    out_idx = wp.atomic_add(face_count, 0, num_faces)

    for fi in range(5):
        if fi >= num_faces:
            return
        for vi in range(3):
            edge_verts = wp.vec2i(flat_edge_verts_table[tri_start + 3 * fi + vi])
            v_from = edge_verts[0]
            v_to = edge_verts[1]
            val_0 = wp.float32(corner_vals[v_from])
            val_1 = wp.float32(corner_vals[v_to])
            p_0 = wp.vec3f(corner_offsets_table[v_from])
            p_1 = wp.vec3f(corner_offsets_table[v_to])
            val_diff = val_1 - val_0
            if wp.abs(val_diff) < 1e-8:
                p = 0.5 * (p_0 + p_1)
            else:
                t = (isovalue - val_0) / val_diff
                p = p_0 + t * (p_1 - p_0)
            vol_idx = p + wp.vec3(float(x_id), float(y_id), float(z_id))
            local_pos = sdf.sdf_box_lower + wp.cw_mul(vol_idx, sdf.voxel_size)
            vertices[3 * out_idx + 3 * fi + vi] = local_pos


def compute_isomesh_from_texture_sdf(
    tex_data_array: wp.array,
    sdf_idx: int,
    subgrid_start_slots: wp.array,
    coarse_dims: tuple[int, int, int],
    device=None,
    isovalue: float = 0.0,
) -> Mesh | None:
    """Extract an isosurface mesh from a texture SDF via marching cubes.

    Iterates over coarse cells that have subgrids (the narrow-band region
    where the surface lives) and runs marching cubes on their fine voxels.

    Args:
        tex_data_array: Warp array of :class:`TextureSDFData` structs.
        sdf_idx: Index into *tex_data_array* to extract.
        subgrid_start_slots: The 3D ``subgrid_start_slots`` array for this SDF
            entry (used to determine which coarse cells are active).
        coarse_dims: ``(cx, cy, cz)`` number of coarse cells per axis.
        device: Warp device.
        isovalue: Surface level to extract [m].  ``0.0`` gives the
            zero-isosurface; positive values extract an outward offset surface.

    Returns:
        :class:`~newton.Mesh` with the isosurface, or ``None`` if empty.
    """
    from .sdf_mc import get_mc_tables  # noqa: PLC0415
    from .types import Mesh  # noqa: PLC0415

    if device is None:
        device = wp.get_device()

    if subgrid_start_slots is None:
        return None

    tex_np = tex_data_array.numpy()
    entry = tex_np[sdf_idx]
    subgrid_size = int(entry["subgrid_size"])
    if subgrid_size == 0:
        return None

    cx, cy, cz = coarse_dims

    single = tex_data_array[sdf_idx : sdf_idx + 1]

    slots_np = subgrid_start_slots.numpy()
    active_cells = []
    for ix in range(cx):
        for iy in range(cy):
            for iz in range(cz):
                if slots_np[ix, iy, iz] != SLOT_EMPTY:
                    active_cells.append((ix, iy, iz))

    if not active_cells:
        return None

    active_coarse_cells = wp.array(active_cells, dtype=wp.vec3i, device=device)
    num_active = len(active_cells)

    mc_tables = get_mc_tables(device)
    tri_range_table = mc_tables[0]
    flat_edge_verts_table = mc_tables[4]
    corner_offsets_table = mc_tables[3]

    face_count = wp.zeros((1,), dtype=int, device=device)
    wp.launch(
        _count_isomesh_faces_texture_kernel,
        dim=(num_active, subgrid_size, subgrid_size, subgrid_size),
        inputs=[single, active_coarse_cells, subgrid_size, tri_range_table, corner_offsets_table, float(isovalue)],
        outputs=[face_count],
        device=device,
    )

    num_faces = int(face_count.numpy()[0])
    if num_faces == 0:
        return None

    max_verts = 3 * num_faces
    verts = wp.empty((max_verts,), dtype=wp.vec3, device=device)

    face_count.zero_()
    wp.launch(
        _generate_isomesh_texture_kernel,
        dim=(num_active, subgrid_size, subgrid_size, subgrid_size),
        inputs=[
            single,
            active_coarse_cells,
            subgrid_size,
            tri_range_table,
            flat_edge_verts_table,
            corner_offsets_table,
            float(isovalue),
        ],
        outputs=[face_count, verts],
        device=device,
    )

    verts_np = verts.numpy()
    faces_np = np.arange(3 * num_faces).reshape(-1, 3)
    faces_np = faces_np[:, ::-1]
    return Mesh(verts_np, faces_np)
