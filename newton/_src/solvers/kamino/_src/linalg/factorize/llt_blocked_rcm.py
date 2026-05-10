# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Linear Algebra: RCM-reordered, semi-sparse Blocked LLT (Cholesky).

This module mirrors :mod:`llt_blocked` (flat-batched layout, Tile API kernels)
but adds:

- Transparent per-block fill-reducing reordering using the GPU-native
  batched Reverse Cuthill-McKee launches from :mod:`.rcm_batch`.
- A tile-granularity *zero-block mask* (the "semi-sparse" part): every inner
  tile_load / tile_matmul in the numeric factorize + solve kernels is gated on
  a per-block ``tile_pattern`` so that tiles which are guaranteed to be zero
  are skipped. The mask itself is built on the GPU from the permuted ``A`` and
  then inflated by a classical block symbolic Cholesky fill-in step - both
  steps are CUDA-graph capturable (fixed launch dimensions).

The caller-facing wrapper (:class:`llt_blocked_rcm_solver.LLTBlockedRCMSolver`)
hides all of this behind the same public API as :class:`LLTBlockedSolver`.

Layout conventions (same as llt_blocked):
- ``dim`` (int32[num_blocks]):          active size ``n_i`` of each block
- ``mio`` (int32[num_blocks]):          matrix-index offset into flat A/L (n_i*n_i per block)
- ``vio`` (int32[num_blocks]):          vector-index offset into flat vectors (n_i per block)
- ``tpo`` (int32[num_blocks]):          tile-pattern-index offset into flat tile_pattern
                                        (``n_tiles_i * n_tiles_i`` entries per block,
                                        where ``n_tiles_i = ceil(n_i / block_size)``)
- ``P``, ``inv_P`` (int32[total_vec_size]): concatenated per-block permutations
                                            indexed by ``vio[i]`` with length ``dim[i]``
"""

from ctypes import sizeof
from functools import cache

import warp as wp

from ...core.types import float32, int32

###
# Module interface
###

__all__ = [
    "llt_blocked_rcm_factorize",
    "llt_blocked_rcm_fused_permute_and_tp",
    "llt_blocked_rcm_permute_vector",
    "llt_blocked_rcm_solve",
    "llt_blocked_rcm_solve_inplace",
    "llt_blocked_rcm_symbolic_fill_in",
    "make_llt_blocked_rcm_factorize_kernel",
    "make_llt_blocked_rcm_fused_permute_and_tp_kernel",
    "make_llt_blocked_rcm_permute_vector_kernel",
    "make_llt_blocked_rcm_solve_inplace_kernel",
    "make_llt_blocked_rcm_solve_kernel",
    "make_llt_blocked_rcm_symbolic_fill_in_kernel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Raw-pointer helpers (mirrors llt_blocked.py)
###

get_array_ptr_cpp = """return (uint64_t)arr.data;"""
"""A native C++ function to get the raw pointer of a warp array."""


def make_get_array_offset_ptr_func(dtype):
    """Creates a function to get the offset pointer of a warp array."""

    @wp.func_native(get_array_ptr_cpp)
    def get_dtype_array_ptr(arr: wp.array[dtype]) -> wp.uint64: ...

    @wp.func
    def get_dtype_array_offset_ptr(arr: wp.array[dtype], start_index: int) -> wp.uint64:
        return get_dtype_array_ptr(arr) + wp.uint64(start_index * wp.static(sizeof(dtype._type_)))

    return get_dtype_array_offset_ptr


get_int32_array_offset_ptr = make_get_array_offset_ptr_func(wp.int32)
get_float32_array_offset_ptr = make_get_array_offset_ptr_func(wp.float32)


###
# Auxiliary kernels: permutations, tile-pattern build, symbolic fill-in, inv_P
###


@cache
def make_llt_blocked_rcm_permute_vector_kernel(max_dim: int):
    """Per-(block, row) kernel: ``b_hat[r] = b[P[r]]`` (or inverse).

    Launched over ``(num_blocks, max_dim)``. Uses ``vio[b]`` for offsets.
    """
    del max_dim

    @wp.kernel(enable_backward=False)
    def permute_vector_kernel(
        dim: wp.array[int32],
        vio: wp.array[int32],
        P: wp.array[int32],
        src: wp.array[float32],
        dst: wp.array[float32],
    ):
        b, r = wp.tid()
        n_i = dim[b]
        if r >= n_i:
            return
        vec_off = vio[b]
        p_r = P[vec_off + r]
        dst[vec_off + r] = src[vec_off + p_r]

    return permute_vector_kernel


@cache
def make_llt_blocked_rcm_fused_permute_and_tp_kernel(block_size: int, max_dim: int):
    """Fused kernel: builds ``inv_P``, permutes ``A -> A_hat``, and reduces
    ``|A_hat|`` into the tile pattern in a single launch.

    Launch dims: ``(num_blocks, max_dim, max_dim)``. Each thread ``(b, r, c)``:

    1. If ``c == 0``: writes ``inv_P[P[r]] = r`` for block ``b``.
    2. Computes ``v = A[P[r], P[c]]`` and writes it into ``A_hat[r, c]``.
    3. If ``|v| > tol``: atomically ORs ``1`` into the tile-pattern slot
       ``(r // block_size, c // block_size)`` for this block.

    Folds what used to be three separate launches into one. The data
    dependency is safe because each thread writes and reads only *its own*
    ``A_hat[r, c]`` slot.

    Callers must zero ``tile_pattern`` before invoking this (the tile-pattern
    contribution is an atomic OR, not an overwrite). ``max_dim`` is baked in
    for the cache key.
    """
    del max_dim

    @wp.kernel(enable_backward=False)
    def fused_permute_and_tp_kernel(
        dim: wp.array[int32],
        mio: wp.array[int32],
        vio: wp.array[int32],
        tpo: wp.array[int32],
        tol: float,
        P: wp.array[int32],
        A: wp.array[float32],
        A_hat: wp.array[float32],
        inv_P: wp.array[int32],
        tile_pattern: wp.array[int32],
    ):
        b, r, c = wp.tid()
        n_i = dim[b]
        if r >= n_i or c >= n_i:
            return
        mat_off = mio[b]
        vec_off = vio[b]
        tp_off = tpo[b]
        n_tiles = (n_i + block_size - 1) // block_size

        p_r = P[vec_off + r]
        p_c = P[vec_off + c]

        # 1. inv_P: a single thread column does this per row.
        if c == int(0):
            inv_P[vec_off + p_r] = r

        # 2. Permuted value.
        v = A[mat_off + p_r * n_i + p_c]
        A_hat[mat_off + r * n_i + c] = v

        # 3. Tile-pattern OR (via atomic_max on 0/1).
        av = v
        if av < float(0):
            av = -av
        if av > tol:
            tr = r // block_size
            tc = c // block_size
            wp.atomic_max(tile_pattern, tp_off + tr * n_tiles + tc, int(1))

    return fused_permute_and_tp_kernel


@cache
def make_llt_blocked_rcm_symbolic_fill_in_kernel(max_n_tiles: int):
    """Per-block kernel (launch dim = num_blocks) that performs block symbolic
    Cholesky fill-in on ``tile_pattern`` in place.

    Rule (from ``llt_blocked_semi_sparse.symbolic_cholesky_dense``): for every
    ``(i, j)`` with ``j < i``, if there exists ``k < j`` such that both
    ``L[i, k]`` and ``L[j, k]`` are nonzero, then ``L[i, j]`` is also nonzero.

    The inflated pattern is the final tile sparsity pattern of ``L`` (lower
    triangle including diagonal); the upper triangle is left unchanged.
    ``max_n_tiles`` is a compile-time upper bound on ``n_tiles_i`` for any
    block in the batch - baked in so Warp can statically unroll-bound the
    inner loops.
    """
    del max_n_tiles  # kept for cache key; kernel itself uses dynamic n_tiles from dim

    @wp.kernel(enable_backward=False)
    def symbolic_fill_in_kernel(
        dim: wp.array[int32],
        tpo: wp.array[int32],
        block_size: int,
        tile_pattern: wp.array[int32],
    ):
        b = wp.tid()
        n_i = dim[b]
        n_tiles = (n_i + block_size - 1) // block_size
        tp_off = tpo[b]

        # Force diagonal to be nonzero (SPD diagonal is always present; also
        # guarantees the forward/backward solve has a diagonal pivot tile).
        for d in range(n_tiles):
            tile_pattern[tp_off + d * n_tiles + d] = int(1)

        # Classical block symbolic Cholesky, lower-triangle only.
        # For each j, then each i > j: L[i,j] |= OR_k<j ( L[i,k] & L[j,k] ).
        for j in range(n_tiles):
            for i in range(j + 1, n_tiles):
                if tile_pattern[tp_off + i * n_tiles + j] == int(0):
                    # Scan k = 0..j-1 for a filled (i,k) and (j,k) pair.
                    filled = int(0)
                    for k in range(j):
                        if tile_pattern[tp_off + i * n_tiles + k] != int(0) and tile_pattern[
                            tp_off + j * n_tiles + k
                        ] != int(0):
                            filled = int(1)
                    if filled == int(1):
                        tile_pattern[tp_off + i * n_tiles + j] = int(1)

    return symbolic_fill_in_kernel


###
# Numeric kernels: factorize / solve / solve_inplace with tile-pattern skips
###


@cache
def make_llt_blocked_rcm_factorize_kernel(block_size: int):
    """Clone of :func:`llt_blocked.make_llt_blocked_factorize_kernel` with tile
    skipping. Reads tile pattern from ``tile_pattern`` indexed by ``tpo[tid]``.

    Tile-skip logic follows :mod:`llt_blocked_semi_sparse`: both halves of an
    update (``L[i,k]`` and ``L[j,k]``, or a single ``L[k,j]``) must be nonzero
    to contribute; the destination tile is also skipped if its pattern slot
    is zero (no need to write it).
    """

    @wp.kernel(enable_backward=False)
    def llt_blocked_rcm_factorize_kernel(
        # Inputs:
        dim: wp.array[int32],
        mio: wp.array[int32],
        tpo: wp.array[int32],
        A: wp.array[float32],
        tile_pattern: wp.array[int32],
        # Outputs:
        L: wp.array[float32],
    ):
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        n_i = dim[tid]
        A_i_start = mio[tid]
        tp_i_start = tpo[tid]

        A_i_ptr = get_float32_array_offset_ptr(A, A_i_start)
        L_i_ptr = get_float32_array_offset_ptr(L, A_i_start)
        tp_i_ptr = get_int32_array_offset_ptr(tile_pattern, tp_i_start)

        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size
        n_tiles = n_i_padded // block_size

        A_i = wp.array(ptr=A_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        TP_i = wp.array(ptr=tp_i_ptr, shape=(n_tiles, n_tiles), dtype=wp.int32)

        # Process the matrix in blocks along its leading dimension.
        for k in range(0, n_i_padded, block_size):
            end = k + block_size
            tile_k = k // block_size

            A_kk_tile = wp.tile_load(A_i, shape=(block_size, block_size), offset=(k, k), storage="shared")

            if k + block_size > n_i:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for i in range(num_iterations):
                    linear_index = tid_block + i * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = A_kk_tile[row, col]
                    if k + row >= n_i or k + col >= n_i:
                        value = wp.where(row == col, float32(1), float32(0))
                    A_kk_tile[row, col] = value

            if k > 0:
                for j in range(0, k, block_size):
                    tile_j = j // block_size
                    # Skip if L[tile_k, tile_j] is known-zero.
                    if TP_i[tile_k, tile_j] == int(0):
                        continue
                    L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(k, j))
                    L_block_T = wp.tile_transpose(L_block)
                    wp.tile_matmul(L_block, L_block_T, A_kk_tile, alpha=-1.0)

            wp.tile_cholesky_inplace(A_kk_tile)
            wp.tile_store(L_i, A_kk_tile, offset=(k, k))

            for i in range(end, n_i_padded, block_size):
                tile_i = i // block_size

                # Skip the whole off-diagonal block panel if L[tile_i, tile_k] is zero.
                if TP_i[tile_i, tile_k] == int(0):
                    continue

                A_ik_tile = wp.tile_load(A_i, shape=(block_size, block_size), offset=(i, k), storage="shared")

                if i + block_size > n_i or k + block_size > n_i:
                    num_tile_elements = block_size * block_size
                    num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                    for ii in range(num_iterations):
                        linear_index = tid_block + ii * num_threads_per_block
                        linear_index = linear_index % num_tile_elements
                        row = linear_index // block_size
                        col = linear_index % block_size
                        value = A_ik_tile[row, col]
                        if i + row >= n_i or k + col >= n_i:
                            value = wp.where(i + row == k + col, float32(1), float32(0))
                        A_ik_tile[row, col] = value

                if k > 0:
                    for j in range(0, k, block_size):
                        tile_j = j // block_size
                        # Need both L[tile_i, tile_j] and L[tile_k, tile_j] nonzero.
                        if TP_i[tile_i, tile_j] == int(0):
                            continue
                        if TP_i[tile_k, tile_j] == int(0):
                            continue
                        L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                        L_2_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(k, j))
                        L_T_tile = wp.tile_transpose(L_2_tile)
                        wp.tile_matmul(L_tile, L_T_tile, A_ik_tile, alpha=-1.0)

                t = wp.tile_transpose(A_ik_tile)
                wp.tile_lower_solve_inplace(A_kk_tile, t)
                sol_tile = wp.tile_transpose(t)
                wp.tile_store(L_i, sol_tile, offset=(i, k))

    return llt_blocked_rcm_factorize_kernel


@cache
def make_llt_blocked_rcm_solve_kernel(block_size: int):
    """Clone of :func:`llt_blocked.make_llt_blocked_solve_kernel` with tile skipping."""

    @wp.kernel(enable_backward=False)
    def llt_blocked_rcm_solve_kernel(
        # Inputs:
        dim: wp.array[int32],
        mio: wp.array[int32],
        vio: wp.array[int32],
        tpo: wp.array[int32],
        L: wp.array[float32],
        tile_pattern: wp.array[int32],
        b: wp.array[float32],
        # Outputs:
        y: wp.array[float32],
        x: wp.array[float32],
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
        x_i_ptr = get_float32_array_offset_ptr(x, v_i_start)
        tp_i_ptr = get_int32_array_offset_ptr(tile_pattern, tp_i_start)

        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size
        n_tiles = n_i_padded // block_size

        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        b_i = wp.array(ptr=b_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        y_i = wp.array(ptr=y_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_i = wp.array(ptr=x_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        TP_i = wp.array(ptr=tp_i_ptr, shape=(n_tiles, n_tiles), dtype=wp.int32)

        # Forward substitution: solve L y = b
        for i in range(0, n_i_padded, block_size):
            tile_i = i // block_size
            rhs_tile = wp.tile_load(b_i, shape=(block_size, 1), offset=(i, 0))
            L_diag = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            if i > 0:
                for j in range(0, i, block_size):
                    tile_j = j // block_size
                    if TP_i[tile_i, tile_j] == int(0):
                        continue
                    L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y_i, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)
            wp.tile_lower_solve_inplace(L_diag, rhs_tile)
            wp.tile_store(y_i, rhs_tile, offset=(i, 0))

        # Backward substitution: solve L^T x = y
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
                        value = wp.where(i + row == i + col, float32(1), float32(0))
                    L_diag[row, col] = value

            if i_end < n_i_padded:
                for j in range(i_end, n_i_padded, block_size):
                    tile_j = j // block_size
                    # In L^T x = y we read L[j, i] (rows below the pivot).
                    if TP_i[tile_j, tile_i] == int(0):
                        continue
                    L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(j, i))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x_i, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)

            wp.tile_upper_solve_inplace(wp.tile_transpose(L_diag), rhs_tile)
            wp.tile_store(x_i, rhs_tile, offset=(i, 0))

    return llt_blocked_rcm_solve_kernel


@cache
def make_llt_blocked_rcm_solve_inplace_kernel(block_size: int):
    """Clone of :func:`llt_blocked.make_llt_blocked_solve_inplace_kernel` with tile skipping.

    Takes ``x`` as in/out; forward substitution reads from ``x`` (as b) and
    writes ``y``, backward substitution reads ``y`` and writes ``x``.
    """

    @wp.kernel(enable_backward=False)
    def llt_blocked_rcm_solve_inplace_kernel(
        # Inputs:
        dim: wp.array[int32],
        mio: wp.array[int32],
        vio: wp.array[int32],
        tpo: wp.array[int32],
        L: wp.array[float32],
        tile_pattern: wp.array[int32],
        # Outputs:
        y: wp.array[float32],
        x: wp.array[float32],
    ):
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        n_i = dim[tid]
        L_i_start = mio[tid]
        v_i_start = vio[tid]
        tp_i_start = tpo[tid]

        L_i_ptr = get_float32_array_offset_ptr(L, L_i_start)
        y_i_ptr = get_float32_array_offset_ptr(y, v_i_start)
        x_i_ptr = get_float32_array_offset_ptr(x, v_i_start)
        tp_i_ptr = get_int32_array_offset_ptr(tile_pattern, tp_i_start)

        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size
        n_tiles = n_i_padded // block_size

        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        y_i = wp.array(ptr=y_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_i = wp.array(ptr=x_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        TP_i = wp.array(ptr=tp_i_ptr, shape=(n_tiles, n_tiles), dtype=wp.int32)

        # Forward substitution: solve L y = x (x is the RHS here, in-place)
        for i in range(0, n_i_padded, block_size):
            tile_i = i // block_size
            rhs_tile = wp.tile_load(x_i, shape=(block_size, 1), offset=(i, 0))
            L_diag = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            if i > 0:
                for j in range(0, i, block_size):
                    tile_j = j // block_size
                    if TP_i[tile_i, tile_j] == int(0):
                        continue
                    L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y_i, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)
            wp.tile_lower_solve_inplace(L_diag, rhs_tile)
            wp.tile_store(y_i, rhs_tile, offset=(i, 0))

        # Backward substitution: solve L^T x = y
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
                        value = wp.where(i + row == i + col, float32(1), float32(0))
                    L_diag[row, col] = value

            if i_end < n_i_padded:
                for j in range(i_end, n_i_padded, block_size):
                    tile_j = j // block_size
                    if TP_i[tile_j, tile_i] == int(0):
                        continue
                    L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(j, i))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x_i, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)

            wp.tile_upper_solve_inplace(wp.tile_transpose(L_diag), rhs_tile)
            wp.tile_store(x_i, rhs_tile, offset=(i, 0))

    return llt_blocked_rcm_solve_inplace_kernel


###
# Launchers (thin wrappers)
###


def llt_blocked_rcm_permute_vector(
    kernel,
    dim: wp.array[int32],
    vio: wp.array[int32],
    P: wp.array[int32],
    src: wp.array[float32],
    dst: wp.array[float32],
    num_blocks: int,
    max_dim: int,
    device: wp.DeviceLike = None,
):
    """Launches the per-block vector permutation kernel ``dst[r] = src[P[r]]``."""
    wp.launch(
        kernel=kernel,
        dim=(num_blocks, max_dim),
        inputs=[dim, vio, P, src, dst],
        device=device,
    )


def llt_blocked_rcm_fused_permute_and_tp(
    kernel,
    dim: wp.array[int32],
    mio: wp.array[int32],
    vio: wp.array[int32],
    tpo: wp.array[int32],
    tol: float,
    P: wp.array[int32],
    A: wp.array[float32],
    A_hat: wp.array[float32],
    inv_P: wp.array[int32],
    tile_pattern: wp.array[int32],
    num_blocks: int,
    max_dim: int,
    device: wp.DeviceLike = None,
):
    """Launches the fused (inv_P + permute_matrix + build_tile_pattern) kernel.

    Callers must zero ``tile_pattern`` before invoking this (the tile-pattern
    contribution is an atomic OR, not an overwrite).
    """
    wp.launch(
        kernel=kernel,
        dim=(num_blocks, max_dim, max_dim),
        inputs=[dim, mio, vio, tpo, float(tol), P, A, A_hat, inv_P, tile_pattern],
        device=device,
    )


def llt_blocked_rcm_symbolic_fill_in(
    kernel,
    dim: wp.array[int32],
    tpo: wp.array[int32],
    block_size: int,
    tile_pattern: wp.array[int32],
    num_blocks: int,
    device: wp.DeviceLike = None,
):
    """Launches the symbolic Cholesky fill-in kernel (in-place on tile_pattern)."""
    wp.launch(
        kernel=kernel,
        dim=num_blocks,
        inputs=[dim, tpo, int(block_size), tile_pattern],
        device=device,
    )


def llt_blocked_rcm_factorize(
    kernel,
    dim: wp.array[int32],
    mio: wp.array[int32],
    tpo: wp.array[int32],
    A: wp.array[float32],
    tile_pattern: wp.array[int32],
    L: wp.array[float32],
    num_blocks: int = 1,
    block_dim: int = 128,
    device: wp.DeviceLike = None,
):
    """Launches the RCM-reordered semi-sparse blocked Cholesky factorization."""
    wp.launch_tiled(
        kernel=kernel,
        dim=num_blocks,
        inputs=[dim, mio, tpo, A, tile_pattern, L],
        block_dim=block_dim,
        device=device,
    )


def llt_blocked_rcm_solve(
    kernel,
    dim: wp.array[int32],
    mio: wp.array[int32],
    vio: wp.array[int32],
    tpo: wp.array[int32],
    L: wp.array[float32],
    tile_pattern: wp.array[int32],
    b: wp.array[float32],
    y: wp.array[float32],
    x: wp.array[float32],
    num_blocks: int = 1,
    block_dim: int = 128,
    device: wp.DeviceLike = None,
):
    """Launches the RCM-reordered semi-sparse blocked Cholesky solve kernel."""
    wp.launch_tiled(
        kernel=kernel,
        dim=num_blocks,
        inputs=[dim, mio, vio, tpo, L, tile_pattern, b, y, x],
        block_dim=block_dim,
        device=device,
    )


def llt_blocked_rcm_solve_inplace(
    kernel,
    dim: wp.array[int32],
    mio: wp.array[int32],
    vio: wp.array[int32],
    tpo: wp.array[int32],
    L: wp.array[float32],
    tile_pattern: wp.array[int32],
    y: wp.array[float32],
    x: wp.array[float32],
    num_blocks: int = 1,
    block_dim: int = 128,
    device: wp.DeviceLike = None,
):
    """Launches the RCM-reordered semi-sparse in-place solve kernel."""
    wp.launch_tiled(
        kernel=kernel,
        dim=num_blocks,
        inputs=[dim, mio, vio, tpo, L, tile_pattern, y, x],
        block_dim=block_dim,
        device=device,
    )
