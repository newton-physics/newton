# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-ordering kernels of the LOX RCM blocked LLT solver.

Kamino's RCM blocked LLT (:mod:`...linalg.factorize.llt_blocked_rcm`) orders each
block and builds its tile pattern from the numeric matrix at every factorization.
LOX computes the ordering, the filled tile pattern, and the tiles visited by each
row of the triangular solves once from the structure of its systems (see
:mod:`.llt_blocked_rcm_solver`). These kernels apply that fixed permutation to the
matrix and walk the precomputed tile lists in the triangular solves; the numeric
factorization uses Kamino's kernels.

Layout conventions (same as Kamino's ``llt_blocked_rcm``):
- ``dim``, ``mio``, ``vio``: active size, matrix offset, and vector offset of each block.
- ``tpo``: tile-pattern offset of each block, with ``n_tiles_i * n_tiles_i`` entries.
- ``tile_traversal``: for each block, ``2 * n_tiles_i * n_tiles_i`` entries starting at
  ``2 * tpo[i]``: the ``-1``-terminated nonzero tile columns of each tile row for the
  forward substitution, then the nonzero tile rows below each tile column for the
  backward substitution.
"""

from functools import cache

import warp as wp

from ....linalg.factorize._tile_builtins import (
    HAS_NATIVE_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE,
    HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE,
    make_tile_matmul_left_transpose_update_func,
)
from ....linalg.factorize.llt_blocked_rcm import get_float32_array_offset_ptr, get_int32_array_offset_ptr

###
# Module interface
###

__all__ = [
    "llt_blocked_rcm_fixed_solve",
    "llt_blocked_rcm_permute_matrix",
    "make_llt_blocked_rcm_fixed_solve_kernel",
    "make_llt_blocked_rcm_permute_matrix_kernel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Kernels
###


@cache
def make_llt_blocked_rcm_permute_matrix_kernel():
    """Build the lower triangle of ``A_hat = P A P^T`` for a fixed permutation."""

    @wp.kernel
    def permute_matrix_kernel(
        dim: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        P: wp.array[wp.int32],
        A: wp.array[wp.float32],
        A_hat: wp.array[wp.float32],
    ):
        b, triangular_index = wp.tid()
        n_i = dim[b]
        triangular_size = n_i * (n_i + 1) // 2
        if triangular_index >= triangular_size:
            return

        r = int((wp.sqrt(float(8 * triangular_index + 1)) - float(1)) * float(0.5))
        row_start = r * (r + 1) // 2
        if row_start > triangular_index:
            r -= 1
            row_start = r * (r + 1) // 2
        elif (r + 1) * (r + 2) // 2 <= triangular_index:
            r += 1
            row_start = r * (r + 1) // 2
        c = triangular_index - row_start
        mat_off = mio[b]
        vec_off = vio[b]
        p_r = P[vec_off + r]
        p_c = P[vec_off + c]
        A_hat[mat_off + r * n_i + c] = A[mat_off + p_r * n_i + p_c]

    return permute_matrix_kernel


@cache
def make_llt_blocked_rcm_fixed_solve_kernel(block_size: int):
    """RCM solve over the precomputed tile traversal, with fused output un-permutation.

    Only the nonzero tiles listed in ``tile_traversal`` are visited. The solve gathers the RHS into permuted coordinates, writes ``x_hat`` in
    permuted coordinates for backward-substitution dependencies, and scatters
    each solved tile directly to the original-coordinate output ``x``.
    """

    @wp.kernel
    def llt_blocked_rcm_solve_kernel(
        # Inputs:
        dim: wp.array[wp.int32],
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
        tp_i_start = tpo[tid]

        L_i_ptr = get_float32_array_offset_ptr(L, L_i_start)
        b_i_ptr = get_float32_array_offset_ptr(b, v_i_start)
        y_i_ptr = get_float32_array_offset_ptr(y, v_i_start)
        x_hat_i_ptr = get_float32_array_offset_ptr(x_hat, v_i_start)
        x_i_ptr = get_float32_array_offset_ptr(x, v_i_start)
        P_i_ptr = get_int32_array_offset_ptr(P, v_i_start)

        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size
        n_tiles = n_i_padded // block_size

        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        b_i = wp.array(ptr=b_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        y_i = wp.array(ptr=y_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_hat_i = wp.array(ptr=x_hat_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_i = wp.array(ptr=x_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        P_i = wp.array(ptr=P_i_ptr, shape=(n_i,), dtype=wp.int32)

        # Forward substitution: solve L y = b.
        for i in range(0, n_i_padded, block_size):
            tile_i = i // block_size
            rhs_tile = wp.tile_zeros(shape=(block_size, 1), dtype=wp.float32, storage="shared")
            num_row_iterations = (block_size + num_threads_per_block - 1) // num_threads_per_block
            for ii in range(num_row_iterations):
                row = tid_block + ii * num_threads_per_block
                active = row < block_size and i + row < n_i
                value = wp.float32(0.0)
                if active:
                    value = b_i[P_i[i + row], 0]
                wp.tile_scatter_masked(rhs_tile, row, 0, value, active)
            L_diag = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            if i > 0:
                traversal_offset = 2 * tp_i_start + tile_i * n_tiles
                for slot in range(n_tiles):
                    tile_j = tile_traversal[traversal_offset + slot]
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
            i_end = i + block_size
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

            if i_end < n_i_padded:
                traversal_offset = 2 * tp_i_start + n_tiles * n_tiles + tile_i * n_tiles
                for slot in range(n_tiles):
                    tile_j = tile_traversal[traversal_offset + slot]
                    if tile_j < 0:
                        break
                    j = tile_j * block_size
                    L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(j, i))
                    x_tile = wp.tile_load(x_hat_i, shape=(block_size, 1), offset=(j, 0))
                    if wp.static(HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE):
                        wp.tile_matmul_left_transpose_update(rhs_tile, L_tile, x_tile, alpha=-1.0)
                    elif wp.static(HAS_NATIVE_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE):
                        wp.static(make_tile_matmul_left_transpose_update_func(block_size))(
                            rhs_tile, L_tile, x_tile, -1.0
                        )
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

    return llt_blocked_rcm_solve_kernel


###
# Launchers
###


def llt_blocked_rcm_permute_matrix(
    kernel,
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    P: wp.array[wp.int32],
    A: wp.array[wp.float32],
    A_hat: wp.array[wp.float32],
    num_blocks: int,
    max_dim: int,
    device: wp.DeviceLike = None,
):
    """Permute a batched matrix using an already initialized permutation."""
    wp.launch(
        kernel=kernel,
        dim=(num_blocks, max_dim * (max_dim + 1) // 2),
        inputs=[dim, mio, vio, P, A, A_hat],
        device=device,
    )


def llt_blocked_rcm_fixed_solve(
    kernel,
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    tpo: wp.array[wp.int32],
    P: wp.array[wp.int32],
    L: wp.array[wp.float32],
    tile_traversal: wp.array[wp.int32],
    b: wp.array[wp.float32],
    y: wp.array[wp.float32],
    x_hat: wp.array[wp.float32],
    x: wp.array[wp.float32],
    num_blocks: int = 1,
    block_dim: int = 128,
    device: wp.DeviceLike = None,
):
    """Launch the fixed-traversal RCM solve kernel."""
    wp.launch_tiled(
        kernel=kernel,
        dim=num_blocks,
        inputs=[dim, mio, vio, tpo, P, L, tile_traversal, b, y, x_hat, x],
        block_dim=block_dim,
        device=device,
    )
