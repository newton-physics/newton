# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deprecated helpers for legacy SDF block-coordinate arrays.

.. deprecated:: 1.3
    This entire module is deprecated and will be removed in a future release.

The hydroelastic broadphase used to consume two precomputed arrays on
``Model`` — ``sdf_block_coords`` (flat ``wp.vec3us`` per active block)
and ``sdf_index2blocks`` (``[start, end)`` per SDF). Both were dropped
when the broadphase began deriving block coordinates arithmetically
from each SDF's coarse-texture dimensions.

The helpers here exist solely so the deprecated ``Model.sdf_block_coords``
and ``Model.sdf_index2blocks`` properties can keep returning equivalent
arrays for callers that still read them. They visit every subgrid in the
coarse grid (matching the new broadphase semantics), so the returned
coords are dense rather than narrow-band.
"""

from __future__ import annotations

import numpy as np
import warp as wp


def compute_block_coords_and_index2blocks(
    coarse_textures: list,
    subgrid_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build legacy ``(block_coords, index2blocks)`` arrays from coarse textures.

    For each coarse texture, emit every subgrid's voxel-space corner as a
    ``vec3us``. The per-SDF range is recorded as ``[start, end)`` rows.

    Args:
        coarse_textures: List of ``wp.Texture3D`` (one per SDF). Entries may
            be ``None`` for SDFs that have no texture data.
        subgrid_size: Cells per subgrid side. Matches the value baked into
            ``TextureSDFData.subgrid_size`` at build time (default 8).

    Returns:
        ``(block_coords, index2blocks)`` as NumPy arrays:
          * ``block_coords`` — shape ``(N, 3)``, dtype ``uint16`` —
            concatenated subgrid corner coordinates in voxel space.
          * ``index2blocks`` — shape ``(num_sdfs, 2)``, dtype ``int32`` —
            per-SDF ``[start, end)`` indices into ``block_coords``.
    """
    coords_chunks: list[np.ndarray] = []
    index2blocks = np.zeros((len(coarse_textures), 2), dtype=np.int32)
    cursor = 0

    for sdf_idx, tex in enumerate(coarse_textures):
        index2blocks[sdf_idx, 0] = cursor
        if tex is not None:
            cw = max(int(tex.width) - 1, 0)
            ch = max(int(tex.height) - 1, 0)
            cd = max(int(tex.depth) - 1, 0)
            n = cw * ch * cd
            if n > 0:
                bz, by, bx = np.meshgrid(
                    np.arange(cd, dtype=np.uint16),
                    np.arange(ch, dtype=np.uint16),
                    np.arange(cw, dtype=np.uint16),
                    indexing="ij",
                )
                sgs = np.uint16(subgrid_size)
                chunk = np.stack(
                    [(bx * sgs).ravel(), (by * sgs).ravel(), (bz * sgs).ravel()],
                    axis=-1,
                ).astype(np.uint16)
                coords_chunks.append(chunk)
                cursor += n
        index2blocks[sdf_idx, 1] = cursor

    if coords_chunks:
        block_coords = np.concatenate(coords_chunks, axis=0)
    else:
        block_coords = np.zeros((0, 3), dtype=np.uint16)

    return block_coords, index2blocks


def build_legacy_sdf_block_arrays(
    coarse_textures: list,
    subgrid_size: int = 8,
    device: str | None = None,
) -> tuple[wp.array, wp.array]:
    """Return the legacy ``(sdf_block_coords, sdf_index2blocks)`` Warp arrays.

    Thin wrapper around :func:`compute_block_coords_and_index2blocks` that
    materializes the numpy results as Warp arrays with the historical
    dtypes (``wp.vec3us`` and ``wp.vec2i``).
    """
    block_coords_np, index2blocks_np = compute_block_coords_and_index2blocks(coarse_textures, subgrid_size=subgrid_size)
    sdf_block_coords = wp.array(block_coords_np, dtype=wp.vec3us, device=device)
    sdf_index2blocks = wp.array(index2blocks_np, dtype=wp.vec2i, device=device)
    return sdf_block_coords, sdf_index2blocks
