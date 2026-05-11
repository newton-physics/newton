# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Utilities: tile and block sizing helpers."""

from __future__ import annotations

import math

__all__ = [
    "get_block_dim",
    "get_num_tiles",
    "get_tile_size",
]


def get_tile_size(size: int, max_size: int = 2048) -> int:
    """Return a power-of-two tile size that covers ``size``.

    The tile size is the smallest power-of-two value greater than or equal to
    ``size``, clamped to ``[1, max_size]``.

    Args:
        size: Number of elements to cover.
        max_size: Maximum allowed tile size.

    Returns:
        A power-of-two tile size in ``[1, max_size]``.
    """
    if size < 1:
        return 1
    return min(max_size, 2 ** math.ceil(math.log(size, 2)))


def get_num_tiles(size: int, tile_size: int) -> int:
    """Return the number of tiles required to cover ``size`` elements.

    Args:
        size: Number of elements to cover.
        tile_size: Number of elements per tile.

    Returns:
        Number of tiles needed so that ``num_tiles * tile_size >= size``.
    """
    return (size + tile_size - 1) // tile_size


def get_block_dim(tile_size: int, ratio: int = 8, min_size: int = 32, max_size: int = 256) -> int:
    """Return a launch block dimension derived from a tile size.

    Computes ``tile_size // ratio`` and clamps the result to
    ``[min_size, max_size]``.

    Args:
        tile_size: Tile size the kernel operates on.
        ratio: Ratio between tile size and block dimension.
        min_size: Minimum block size.
        max_size: Maximum block size.

    Returns:
        A block dimension suitable for passing as ``block_dim`` to tiled launches.
    """
    return max(min_size, min(max_size, tile_size // ratio))
