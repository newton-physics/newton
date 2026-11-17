# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse LLT of the LOX body systems.

:class:`LLTBlockedRCMTraversalSolver` extends Kamino's RCM-ordered, semi-sparse blocked LLT
(:class:`LLTBlockedRCMSolver`) with triangular solves over precomputed lists of nonzero tiles.
Each numeric factorization ends with one launch that lists, for every tile row of a block, its
nonzero tiles left of the diagonal, and for every tile column its nonzero tiles below the diagonal.
The forward and backward substitutions visit exactly these tiles, so a solve scales with the
nonzero tiles of the factor.

Layout of ``tile_traversal`` (``2 * n_tiles_i * n_tiles_i`` entries for block ``i``, from ``2 * tpo[i]``,
where ``tpo`` holds the offsets of Kamino's tile pattern): the forward lists of the tile rows, then the
backward lists of the tile columns. Each list takes ``n_tiles_i`` slots and ends with ``-1``.
"""

from functools import cache
from typing import Any

import warp as wp

from ......core.types import override
from ...linalg.factorize._array_ptr import get_float32_array_offset_ptr, get_int32_array_offset_ptr
from ...linalg.factorize._tile_builtins import (
    HAS_NATIVE_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE,
    HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE,
    make_tile_matmul_left_transpose_update_func,
)
from ...linalg.factorize.llt_blocked_rcm import _sync_threads
from ...linalg.factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver

###
# Module interface
###

__all__ = [
    "LLTBlockedRCMTraversalSolver",
    "make_llt_blocked_rcm_traversal_solve_kernel",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _build_tile_traversal(
    dim: wp.array[wp.int32],
    tpo: wp.array[wp.int32],
    block_size: wp.int32,
    tile_pattern: wp.array[wp.int32],
    tile_traversal: wp.array[wp.int32],
):
    """List the nonzero tiles of one tile row and one tile column of a block's factor."""
    block, tile = wp.tid()
    tiles = (dim[block] + block_size - 1) // block_size
    if tile >= tiles:
        return
    pattern_offset = tpo[block]

    # Forward substitution: nonzero tiles of the tile row, left of the diagonal.
    forward = 2 * pattern_offset + tile * tiles
    count = wp.int32(0)
    for column in range(tile):
        if tile_pattern[pattern_offset + tile * tiles + column] != 0:
            tile_traversal[forward + count] = column
            count += 1
    tile_traversal[forward + count] = -1

    # Backward substitution: nonzero tiles of the tile column, below the diagonal.
    backward = 2 * pattern_offset + tiles * tiles + tile * tiles
    count = wp.int32(0)
    for row in range(tile + 1, tiles):
        if tile_pattern[pattern_offset + row * tiles + tile] != 0:
            tile_traversal[backward + count] = row
            count += 1
    tile_traversal[backward + count] = -1


@cache
def make_llt_blocked_rcm_traversal_solve_kernel(block_size: int):
    """Build the RCM solve over the nonzero tile lists, with fused gather and scatter.

    The solve gathers the right-hand side into RCM order, writes ``x_hat`` in RCM order for the
    backward-substitution dependencies, and scatters each solved tile to the output ``x`` in the
    original order. The tile operations match Kamino's ``make_llt_blocked_rcm_solve_kernel``.
    """

    @wp.kernel(module="unique", module_options={"enable_backward": False, "default_grid_stride": False})
    def llt_blocked_rcm_traversal_solve_kernel(
        # Inputs:
        dim: wp.array[wp.int32],
        ld: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        tpo: wp.array[wp.int32],
        P: wp.array[wp.int32],
        L: wp.array[wp.float32],
        tile_traversal: wp.array[wp.int32],
        b: wp.array[wp.float32],
        # Outputs:
        y: wp.array[wp.float32],
        x_hat: wp.array[wp.float32],
        x: wp.array[wp.float32],
    ):
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        n_i = dim[tid]
        L_i_start = mio[tid]
        v_i_start = vio[tid]
        traversal_start = 2 * tpo[tid]

        L_i_ptr = get_float32_array_offset_ptr(L, L_i_start)
        b_i_ptr = get_float32_array_offset_ptr(b, v_i_start)
        y_i_ptr = get_float32_array_offset_ptr(y, v_i_start)
        x_hat_i_ptr = get_float32_array_offset_ptr(x_hat, v_i_start)
        x_i_ptr = get_float32_array_offset_ptr(x, v_i_start)
        P_i_ptr = get_int32_array_offset_ptr(P, v_i_start)

        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size
        n_tiles = n_i_padded // block_size

        # Rows are stored with stride ld >= n_i; padding keeps tiles aligned and in bounds.
        L_i = wp.array(ptr=L_i_ptr, shape=(ld[tid], ld[tid]), dtype=wp.float32)
        b_i = wp.array(ptr=b_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        y_i = wp.array(ptr=y_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_hat_i = wp.array(ptr=x_hat_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_i = wp.array(ptr=x_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        P_i = wp.array(ptr=P_i_ptr, shape=(n_i,), dtype=wp.int32)

        # Gather the permuted right-hand side once, then solve L y = P b in place.
        for row in range(tid_block, n_i, num_threads_per_block):
            y_i[row, 0] = b_i[P_i[row], 0]
        _sync_threads()
        for i in range(0, n_i_padded, block_size):
            tile_i = i // block_size
            rhs_tile = wp.tile_load(y_i, shape=(block_size, 1), offset=(i, 0), storage="shared")
            L_diag = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            forward = traversal_start + tile_i * n_tiles
            for slot in range(tile_i):
                tile_j = tile_traversal[forward + slot]
                if tile_j < 0:
                    break
                j = tile_j * block_size
                L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                y_block = wp.tile_load(y_i, shape=(block_size, 1), offset=(j, 0))
                wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)
            wp.tile_lower_solve_inplace(L_diag, rhs_tile)
            wp.tile_store(y_i, rhs_tile, offset=(i, 0))

        # Backward substitution: solve L^T x_hat = y and scatter x_hat -> x.
        for i in range(n_i_padded - block_size, -1, -block_size):
            tile_i = i // block_size
            rhs_tile = wp.tile_load(y_i, shape=(block_size, 1), offset=(i, 0))
            L_diag = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))

            if i + block_size > n_i:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    linear_index = tid_block + ii * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = L_diag[row, col]
                    if i + row >= n_i:
                        value = wp.where(i + row == i + col, wp.float32(1), wp.float32(0))
                    L_diag[row, col] = value

            backward = traversal_start + n_tiles * n_tiles + tile_i * n_tiles
            for slot in range(n_tiles - 1 - tile_i):
                tile_j = tile_traversal[backward + slot]
                if tile_j < 0:
                    break
                j = tile_j * block_size
                L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(j, i))
                x_tile = wp.tile_load(x_hat_i, shape=(block_size, 1), offset=(j, 0))
                if wp.static(HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE):
                    wp.tile_matmul_left_transpose_update(rhs_tile, L_tile, x_tile, alpha=-1.0)
                elif wp.static(HAS_NATIVE_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE):
                    wp.static(make_tile_matmul_left_transpose_update_func(block_size))(rhs_tile, L_tile, x_tile, -1.0)
                else:
                    L_T_tile = wp.tile_transpose(L_tile)
                    wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)

            wp.tile_upper_solve_inplace(wp.tile_transpose(L_diag), rhs_tile)
            wp.tile_store(x_hat_i, rhs_tile, offset=(i, 0))

            num_row_iterations = (block_size + num_threads_per_block - 1) // num_threads_per_block
            for ii in range(num_row_iterations):
                row = tid_block + ii * num_threads_per_block
                if row < block_size and i + row < n_i:
                    p_r = P_i[i + row]
                    x_i[p_r, 0] = rhs_tile[row, 0]

    return llt_blocked_rcm_traversal_solve_kernel


###
# Interfaces
###


class LLTBlockedRCMTraversalSolver(LLTBlockedRCMSolver):
    """Kamino's RCM-ordered semi-sparse blocked LLT, whose solves walk the nonzero tiles of the factor.

    Takes the arguments of :class:`LLTBlockedRCMSolver`. Every numeric factorization, eager or
    captured, rebuilds the tile lists from the final tile pattern of the factor, and :meth:`solve`
    visits the listed tiles. :meth:`solve_inplace` runs Kamino's solve over the tile pattern.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._tile_traversal: wp.array[wp.int32] | None = None
        """Nonzero tile lists of the triangular solves, see the module layout."""
        self._traversal_solve_kernel = None
        super().__init__(*args, **kwargs)

    @override
    def _allocate_impl(self, A: Any, **kwargs: Any) -> None:
        super()._allocate_impl(A, **kwargs)
        self._traversal_solve_kernel = make_llt_blocked_rcm_traversal_solve_kernel(self._block_size)
        # Each block holds two lists of n_tiles slots per tile row: twice its n_tiles x n_tiles tile pattern.
        self._tile_traversal = wp.zeros(2 * self._tile_pattern.size, dtype=wp.int32, device=self._device)

    @override
    def _factorize_numeric(self, clear_skipped: bool = True) -> None:
        super()._factorize_numeric(clear_skipped)
        # Every factorization path ends here with the final tile pattern, including the refactored blocks.
        info = self._operator.info
        wp.launch(
            _build_tile_traversal,
            dim=(info.num_blocks, (self._max_dim + self._block_size - 1) // self._block_size),
            inputs=[info.dim, self._tpo, self._block_size, self._tile_pattern],
            outputs=[self._tile_traversal],
            device=self._device,
        )

    @override
    def _solve_impl(self, b: wp.array[Any], x: wp.array[Any]) -> None:
        info = self._operator.info
        # Solve L L^T x_hat = P b over the nonzero tiles and scatter x_hat -> x.
        wp.launch_tiled(
            self._traversal_solve_kernel,
            dim=info.num_blocks,
            inputs=[
                info.dim,
                self._stride(),
                info.mio,
                info.vio,
                self._tpo,
                self._P,
                self._L,
                self._tile_traversal,
                b,
                self._y,
                self._x_hat,
                x,
            ],
            block_dim=self._solve_block_dim,
            device=self._device,
        )
