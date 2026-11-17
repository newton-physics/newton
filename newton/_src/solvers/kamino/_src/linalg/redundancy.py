# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Detection and removal of the numerically redundant rows of batched symmetric positive semi-definite matrices
"""

from __future__ import annotations

from functools import cache

import warp as wp

from ..core.math import FLOAT32_EPS
from .factorize._array_ptr import get_float32_array_offset_ptr

###
# Module interface
###

__all__ = ["RedundantRowShift"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})

###
# Constants
###

_ROW_REMAINING = wp.constant(wp.int32(0))
_ROW_KEPT = wp.constant(wp.int32(1))
_ROW_SHIFTED = wp.constant(wp.int32(2))

_WARP_SIZE = wp.constant(wp.int32(32))

_SHARED_MEMORY_MARGIN = 2048
"""Bytes of shared memory per block reserved for the tile allocator."""

_CHECK_TILE = 32
"""Tile size of the triangular solves of the check."""

_CHECK_COLUMNS = 16
"""Number of right-hand sides of the triangular solves of the check."""

_CHECK_ITERATIONS = 2
"""Number of inverse iterations estimating the smallest eigenvalue of the shifted matrix."""

_CHECK_BLOCK_DIM = 128
"""Number of threads per world of the check."""

_SHIFTED_ROW_MARGIN = wp.constant(wp.float32(10.0))
"""Factor of the threshold above which the residual of a shifted row fails the check."""

_REFERENCE_MARGIN = wp.constant(wp.float32(0.1))
"""Factor of the reference eigenvalue below which the kept rows fail the check."""

_ITERATE_SEED = wp.constant(wp.int32(5381))
"""Seed of the initial iterate of the inverse iterations."""

###
# Functions
###


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@cache
def _make_shared_buffer(dtype: type, size: int):
    """Create the function returning the address of a static shared buffer of a thread block, or zero on the CPU."""
    ctype = {wp.float32: "float", wp.int32: "int"}[dtype]
    snippet = f"""
#if defined(__CUDA_ARCH__)
__shared__ {ctype} values[{size}];
return reinterpret_cast<uint64_t>(values);
#else
return 0;
#endif
"""

    @wp.func_native(snippet)
    def shared_buffer() -> wp.uint64: ...

    return shared_buffer


@wp.func_native("""
#if defined(__CUDA_ARCH__)
extern __shared__ float dynamic_shared[];
return reinterpret_cast<uint64_t>(dynamic_shared);
#else
return 0;
#endif
""")
def _dynamic_shared_base() -> wp.uint64:
    """Return the address of the dynamic shared memory of a thread block, or zero on the CPU."""
    ...


@wp.func
def _packed(i: wp.int32, j: wp.int32) -> wp.int32:
    """Return the index of the entry ``(i, j)``, with ``j <= i``, of a row-major packed lower triangle."""
    return i * (i + 1) // 2 + j


###
# Kernels
###


@wp.kernel
def _shift_stored_rows(
    # Inputs:
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    row_offset: wp.array[wp.int32],
    num_rows: wp.array[wp.int32],
    shifted: wp.array2d[wp.int32],
    # Outputs:
    D: wp.array[wp.float32],
    diagonal: wp.array2d[wp.float32],
):
    """Store the diagonal of the block rows, and shift the rows shifted by the last elimination."""
    wid, i = wp.tid()
    if i >= num_rows[wid]:
        return
    index = mio[wid] + (row_offset[wid] + i) * (dim[wid] + 1)
    value = D[index]
    diagonal[wid, i] = value
    if shifted[wid, i] != 0:
        D[index] = value + wp.max(value, FLOAT32_EPS)


@wp.kernel
def _reset_shifted_rows(
    # Inputs:
    world_mask: wp.array[wp.bool],
    # Outputs:
    shifted: wp.array2d[wp.int32],
    initialized: wp.array[wp.int32],
    reference: wp.array[wp.float32],
):
    """Clear the shifted rows of the masked worlds."""
    wid, i = wp.tid()
    if not world_mask[wid]:
        return
    shifted[wid, i] = 0
    if i == 0:
        initialized[wid] = 0
        reference[wid] = 0.0


@cache
def _make_shift_redundant_rows_kernel(block_dim: int, max_rows: int, shared: bool, shared_triangle: bool):
    """
    Create the elimination kernel of the blocks of up to ``max_rows`` rows, and the sizes of its float and
    int scratch per world.

    The elimination reads the diagonal of the block from the stored diagonal, since the matrix holds the shifts
    of the last elimination, and updates these shifts in the matrix.

    The float scratch holds the pivot column, the diagonal and the pivot search, and the int scratch holds
    the row status and the pivot search. The scratch is in static shared memory if ``shared``, and in the
    rows of the global scratch arrays otherwise. The packed lower triangle is in dynamic shared memory if
    ``shared_triangle``, and in the rows of the global triangle array otherwise.
    """
    rows = max(max_rows, 1)
    triangle_size = max(max_rows * (max_rows + 1) // 2, 1)
    float_size = 2 * rows + block_dim
    int_size = rows + block_dim
    shared_floats = _make_shared_buffer(wp.float32, float_size)
    shared_ints = _make_shared_buffer(wp.int32, int_size)
    diagonal_offset = rows
    ratio_offset = 2 * rows

    @wp.kernel(module="unique", enable_backward=False)
    def _shift_redundant_rows(
        # Inputs:
        dim: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        threshold: wp.float32,
        diagonal: wp.array2d[wp.float32],
        eigenvalue: wp.array[wp.float32],
        # Scratch:
        global_triangle: wp.array2d[wp.float32],
        global_floats: wp.array2d[wp.float32],
        global_ints: wp.array2d[wp.int32],
        # Outputs:
        D: wp.array[wp.float32],
        shifted: wp.array2d[wp.int32],
        initialized: wp.array[wp.int32],
        reference: wp.array[wp.float32],
    ):
        wid, tid = wp.tid()
        threads = wp.block_dim()
        lanes = wp.min(threads, _WARP_SIZE)
        warps = threads // lanes
        warp = tid // lanes
        lane = tid - warp * lanes
        n = num_rows[wid]
        if n == 0:
            return
        ld = dim[wid]
        start = mio[wid] + row_offset[wid] * (ld + 1)

        # Select the elimination scratch of the world
        if wp.static(shared):
            floats = wp.array(ptr=shared_floats(), shape=(float_size,), dtype=wp.float32)
            ints = wp.array(ptr=shared_ints(), shape=(int_size,), dtype=wp.int32)
        else:
            floats = global_floats[wid]
            ints = global_ints[wid]
        if wp.static(shared_triangle):
            # NOTE: The only tile of the kernel reserves the dynamic shared memory of the triangle
            wp.tile_zeros(shape=triangle_size, dtype=wp.float32, storage="shared")
            triangle = wp.array(ptr=_dynamic_shared_base(), shape=(triangle_size,), dtype=wp.float32)
        else:
            triangle = global_triangle[wid]

        # Copy the strict lower triangle of the block and its stored diagonal
        for i in range(warp, n, warps):
            for j in range(lane, i, lanes):
                triangle[_packed(i, j)] = D[start + i * ld + j]
        for i in range(tid, n, threads):
            value = diagonal[wid, i]
            triangle[_packed(i, i)] = value
            floats[diagonal_offset + i] = wp.max(value, FLOAT32_EPS)
            ints[i] = _ROW_REMAINING
        _sync_threads()

        # Eliminate the rows in the order of the largest relative pivot
        for _step in range(n):
            # Find the remaining row with the largest relative pivot
            ratio = wp.float32(-1.0e30)
            row = wp.int32(-1)
            for i in range(tid, n, threads):
                if ints[i] == _ROW_REMAINING:
                    ratio_i = triangle[_packed(i, i)] / floats[diagonal_offset + i]
                    if ratio_i > ratio:
                        ratio = ratio_i
                        row = i
            floats[ratio_offset + tid] = ratio
            ints[rows + tid] = row
            _sync_threads()
            stride = threads // 2
            while stride > 0:
                if tid < stride and floats[ratio_offset + tid + stride] > floats[ratio_offset + tid]:
                    floats[ratio_offset + tid] = floats[ratio_offset + tid + stride]
                    ints[rows + tid] = ints[rows + tid + stride]
                _sync_threads()
                stride = stride // 2
            p = ints[rows]
            best_ratio = floats[ratio_offset]
            if p < 0:
                break

            # Shift the pivot by its diagonal if it is relatively small
            pivot = triangle[_packed(p, p)]
            shift = best_ratio < threshold
            if shift:
                pivot += floats[diagonal_offset + p]

            # Gather the pivot column over the remaining rows
            for i in range(tid, n, threads):
                value = wp.float32(0.0)
                if i != p and ints[i] == _ROW_REMAINING:
                    value = triangle[_packed(wp.max(i, p), wp.min(i, p))]
                floats[i] = value
            _sync_threads()
            if tid == 0:
                ints[p] = wp.where(shift, _ROW_SHIFTED, _ROW_KEPT)

            # Update the Schur complement of the remaining rows
            for i in range(warp, n, warps):
                factor = floats[i] / pivot
                if factor != 0.0:
                    base = _packed(i, 0)
                    for j in range(lane, i + 1, lanes):
                        triangle[base + j] = triangle[base + j] - factor * floats[j]
            _sync_threads()

        # Update the shift of the rows whose status changed
        _sync_threads()
        changed = wp.int32(0)
        for i in range(tid, n, threads):
            dropped = wp.where(ints[i] == _ROW_SHIFTED, 1, 0)
            change = dropped - shifted[wid, i]
            if change != 0:
                D[start + i * (ld + 1)] += wp.float32(change) * floats[diagonal_offset + i]
                changed = 1
            shifted[wid, i] = dropped
        ints[rows + tid] = changed
        _sync_threads()

        # Keep the eigenvalue checked on an unchanged set as the reference of the next checks
        if tid == 0:
            for k in range(1, threads):
                changed = wp.max(changed, ints[rows + k])
            reference[wid] = wp.where(changed == 0, eigenvalue[wid], wp.float32(0.0))
            initialized[wid] = 1

    return _shift_redundant_rows, float_size, int_size


@cache
def _make_check_kernel(permuted: bool):
    """
    Create the kernel checking the shifted rows of each world from the Cholesky factor ``L`` of the shifted
    and regularized matrix ``K``, stored with the row stride ``ld`` and permuted by ``perm`` if ``permuted``.
    """
    tile = _CHECK_TILE
    columns = _CHECK_COLUMNS

    @wp.func
    def forward(L_i: wp.array2d[wp.float32], X_i: wp.array2d[wp.float32], n: wp.int32, tid: wp.int32):
        """Solve ``L Y = X`` in place."""
        threads = wp.block_dim()
        padded = ((n + tile - 1) // tile) * tile
        for i in range(0, padded, tile):
            rhs = wp.tile_load(X_i, shape=(tile, columns), offset=(i, 0))
            L_diag = wp.tile_load(L_i, shape=(tile, tile), offset=(i, i))

            # Pad the diagonal tile beyond the matrix dimension with the identity
            if i + tile > n:
                for k in range((tile * tile + threads - 1) // threads):
                    index = (tid + k * threads) % (tile * tile)
                    row = index // tile
                    col = index - row * tile
                    value = L_diag[row, col]
                    if i + row >= n or i + col >= n:
                        value = wp.where(row == col, wp.float32(1.0), wp.float32(0.0))
                    L_diag[row, col] = value

            for j in range(0, i, tile):
                L_block = wp.tile_load(L_i, shape=(tile, tile), offset=(i, j))
                X_block = wp.tile_load(X_i, shape=(tile, columns), offset=(j, 0))
                wp.tile_matmul(L_block, X_block, rhs, alpha=-1.0)
            wp.tile_lower_solve_inplace(L_diag, rhs)
            wp.tile_store(X_i, rhs, offset=(i, 0))

    @wp.func
    def backward(L_i: wp.array2d[wp.float32], X_i: wp.array2d[wp.float32], n: wp.int32, tid: wp.int32):
        """Solve ``L^T Y = X`` in place."""
        threads = wp.block_dim()
        padded = ((n + tile - 1) // tile) * tile
        for i in range(padded - tile, -1, -tile):
            rhs = wp.tile_load(X_i, shape=(tile, columns), offset=(i, 0))
            L_diag = wp.tile_load(L_i, shape=(tile, tile), offset=(i, i))

            # Pad the diagonal tile beyond the matrix dimension with the identity
            if i + tile > n:
                for k in range((tile * tile + threads - 1) // threads):
                    index = (tid + k * threads) % (tile * tile)
                    row = index // tile
                    col = index - row * tile
                    value = L_diag[row, col]
                    if i + row >= n or i + col >= n:
                        value = wp.where(row == col, wp.float32(1.0), wp.float32(0.0))
                    L_diag[row, col] = value

            for j in range(i + tile, padded, tile):
                L_block = wp.tile_load(L_i, shape=(tile, tile), offset=(j, i))
                X_block = wp.tile_load(X_i, shape=(tile, columns), offset=(j, 0))
                wp.tile_matmul(wp.tile_transpose(L_block), X_block, rhs, alpha=-1.0)
            wp.tile_upper_solve_inplace(wp.tile_transpose(L_diag), rhs)
            wp.tile_store(X_i, rhs, offset=(i, 0))

    @wp.func
    def original_row(perm: wp.array[wp.int32], vstart: wp.int32, h: wp.int32) -> wp.int32:
        if wp.static(permuted):
            return perm[vstart + h]
        return h

    @wp.func
    def factor_row(inverse_perm: wp.array[wp.int32], vstart: wp.int32, row: wp.int32) -> wp.int32:
        if wp.static(permuted):
            return inverse_perm[vstart + row]
        return row

    @wp.kernel(module="unique", enable_backward=False)
    def _check_shifted_rows(
        # Inputs:
        dim: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        ld: wp.array[wp.int32],
        perm: wp.array[wp.int32],
        inverse_perm: wp.array[wp.int32],
        D: wp.array[wp.float32],
        L: wp.array[wp.float32],
        diagonal: wp.array2d[wp.float32],
        shifted: wp.array2d[wp.int32],
        initialized: wp.array[wp.int32],
        reference: wp.array[wp.float32],
        threshold: wp.float32,
        # Scratch:
        X: wp.array3d[wp.float32],
        V: wp.array2d[wp.float32],
        column_rows: wp.array2d[wp.int32],
        # Outputs:
        pivot_rows: wp.array[wp.int32],
        refactor_dim: wp.array[wp.int32],
        eigenvalue_out: wp.array[wp.float32],
    ):
        wid, tid = wp.tid()
        threads = wp.block_dim()
        n = dim[wid]
        rows = num_rows[wid]
        if rows == 0 or initialized[wid] == 0:
            if tid == 0:
                pivot_rows[wid] = rows
                refactor_dim[wid] = wp.where(rows > 0, n, 0)
                eigenvalue_out[wid] = 0.0
            return
        offset = row_offset[wid]
        start = mio[wid]
        vstart = vio[wid]
        padded = ((n + tile - 1) // tile) * tile
        stride = ld[wid]
        L_i = wp.array(ptr=get_float32_array_offset_ptr(L, start), shape=(stride, stride), dtype=wp.float32)
        X_i = X[wid]
        V_i = V[wid]

        # Initialize the iterate with a fixed pseudo-random vector, and count the shifted rows
        partial = wp.float32(0.0)
        for h in range(tid, n, threads):
            value = wp.randf(wp.rand_init(_ITERATE_SEED, h)) - 0.5
            V_i[h] = value
            partial += value * value
        inv_norm = 1.0 / wp.sqrt(wp.tile_sum(wp.tile(partial))[0])
        count = wp.int32(0)
        for i in range(tid, rows, threads):
            count += shifted[wid, i]
        num_shifted = wp.tile_sum(wp.tile(count))[0]

        # Solve with the iterate of the inverse iterations in the first column, and the unit vectors of the
        # shifted rows by chunks in the other columns
        slots = columns - 1
        passes = wp.max(_CHECK_ITERATIONS, (num_shifted + slots - 1) // slots)
        rayleigh = wp.float32(0.0)
        correction = wp.float32(0.0)
        failed = wp.int32(0)
        for p in range(passes):
            iterate = p < _CHECK_ITERATIONS
            for k in range(tid, padded * columns, threads):
                h = k // columns
                value = wp.float32(0.0)
                if iterate and k - h * columns == 0 and h < n:
                    o = original_row(perm, vstart, h)
                    value = wp.sqrt(wp.max(D[start + o * (n + 1)], FLOAT32_EPS)) * V_i[h] * inv_norm
                X_i[h, k - h * columns] = value
            if tid < columns:
                column_rows[wid, tid] = -1
            _sync_threads()
            if tid == 0:
                k = wp.int32(0)
                for i in range(rows):
                    if shifted[wid, i] != 0:
                        c = 1 + k - p * slots
                        if c >= 1 and c < columns:
                            X_i[factor_row(inverse_perm, vstart, offset + i), c] = 1.0
                            column_rows[wid, c] = i
                        k += 1
            _sync_threads()
            forward(L_i, X_i, n, tid)
            backward(L_i, X_i, n, tid)
            _sync_threads()

            # Accumulate the Rayleigh quotient of the inverse, and the regularization of the block rows
            if iterate:
                quotient = wp.float32(0.0)
                norm = wp.float32(0.0)
                regularization = wp.float32(0.0)
                for h in range(tid, n, threads):
                    o = original_row(perm, vstart, h)
                    K = wp.max(D[start + o * (n + 1)], FLOAT32_EPS)
                    x = V_i[h] * inv_norm
                    z = wp.sqrt(K) * X_i[h, 0]
                    quotient += x * z
                    norm += z * z
                    i = o - offset
                    if i >= 0 and i < rows:
                        d = diagonal[wid, i]
                        shift = wp.where(shifted[wid, i] != 0, wp.max(d, FLOAT32_EPS), wp.float32(0.0))
                        regularization += x * x * (K - d - shift) / K
                    V_i[h] = z
                rayleigh = wp.tile_sum(wp.tile(quotient))[0]
                correction = wp.tile_sum(wp.tile(regularization))[0]
                inv_norm = 1.0 / wp.sqrt(wp.tile_sum(wp.tile(norm))[0])

            # A shifted row is independent if its Schur complement exceeds its shift and regularization
            # NOTE: The Schur complement is the residual of a ridge regression of the row on the other rows,
            # with the coefficients -u_o / u_s of the column u of the inverse. The penalty of the coefficients
            # by the regularization of the other block rows is removed, but not by the shifts.
            independent = wp.int32(0)
            if tid >= 1 and tid < columns:
                i = column_rows[wid, tid]
                if i >= 0:
                    u_s = X_i[factor_row(inverse_perm, vstart, offset + i), tid]
                    penalty = wp.float32(0.0)
                    for h in range(n):
                        j = original_row(perm, vstart, h) - offset
                        if j >= 0 and j < rows and j != i:
                            d = diagonal[wid, j]
                            shift = wp.where(shifted[wid, j] != 0, wp.max(d, FLOAT32_EPS), wp.float32(0.0))
                            u_o = X_i[h, tid]
                            penalty += u_o * u_o * (D[start + (offset + j) * (n + 1)] - d - shift)
                    o = offset + i
                    d = diagonal[wid, i]
                    schur = 1.0 / u_s - penalty / (u_s * u_s)
                    residual = (schur - (D[start + o * (n + 1)] - d)) / wp.max(d, FLOAT32_EPS)
                    if not (residual <= _SHIFTED_ROW_MARGIN * threshold):
                        independent = 1
            failed += wp.tile_sum(wp.tile(independent))[0]

        # The kept rows are dependent if the unregularized eigenvalue is small, or much smaller than its reference
        eigenvalue = 1.0 / rayleigh - correction
        limit = threshold
        if reference[wid] > 0.0:
            limit = wp.min(threshold, _REFERENCE_MARGIN * reference[wid])
        if not (eigenvalue >= limit):
            failed += 1

        # Select the world for a new elimination and refactorization
        if tid == 0:
            pivot_rows[wid] = wp.where(failed > 0, rows, 0)
            refactor_dim[wid] = wp.where(failed > 0, n, 0)
            eigenvalue_out[wid] = eigenvalue

    return _check_shifted_rows


###
# Interfaces
###


class RedundantRowShift:
    """
    Drops the numerically redundant rows of a diagonal block of batched symmetric positive semi-definite matrices.

    For each world, the rows of the block are eliminated as in a Cholesky factorization, picking the
    remaining row with the largest relative pivot at each step. The relative pivot of a row is its
    Schur complement diagonal over its original diagonal, i.e. the squared sine of the angle between
    the row and the span of the rows eliminated before it, so it does not depend on the row scaling.
    The rows whose relative pivot is below ``threshold`` are numerically redundant. Their pivot is shifted
    by their original diagonal before the elimination proceeds, and the same shift is added to the
    diagonal of the matrix.

    The shifted matrix is positive definite on the block and can be factorized in any order. Since each
    shifted row lies in the span of the other rows of the block, the solutions of consistent systems
    with the shifted matrix carry no impulse on the shifted rows and solve the original system.

    The elimination runs in float32 on a packed copy of the lower triangle of each block, with one thread
    block per world, and does not synchronize with the host. On CUDA devices, the copy is kept in shared
    memory if it fits, and in global memory otherwise.

    Instead of repeating the elimination at every call with :meth:`apply`, the shifted rows of the last
    elimination can be reused with :meth:`shift`, and checked with :meth:`check` from the Cholesky factor of
    the shifted matrix. The check fails if the smallest eigenvalue of the Jacobi-scaled matrix, without its
    regularization on the block, is below ``threshold``, i.e. if the kept rows became dependent, or if the
    Schur complement of a shifted row exceeds ten times ``threshold`` relative to its diagonal, i.e. if the
    shifted row became independent. :meth:`eliminate` then repeats the elimination of the failed worlds only.
    The coupling of the block with the other rows can also lower the eigenvalue, e.g. with hyperstatic contacts,
    which the elimination of the block cannot fix. So, when the elimination of a failed world keeps the same rows,
    its eigenvalue becomes the reference of the next checks, which only fail below a tenth of it.

    Args:
        num_worlds: The number of worlds.
        max_rows: The largest number of rows of the block across worlds.
        threshold: The relative pivot below which a row is shifted. Must be in ``[0, 1)``.
        block_dim: The number of threads per world. Must be a power of two of at least 32.
        max_dim: The largest dimension of the matrices, required by :meth:`check`.
        device: The device on which to allocate the scratch memory.
    """

    def __init__(
        self,
        num_worlds: int,
        max_rows: int,
        threshold: float,
        block_dim: int = 512,
        max_dim: int = 0,
        device: wp.DeviceLike = None,
    ):
        if not 0.0 <= threshold < 1.0:
            raise ValueError(f"Invalid threshold: {threshold}. Must be in the range [0, 1).")
        if block_dim < 32 or block_dim & (block_dim - 1):
            raise ValueError(f"Invalid block dimension: {block_dim}. Must be a power of two of at least 32.")
        device = wp.get_device(device)
        self.threshold = threshold
        self.block_dim = block_dim
        self.num_worlds = num_worlds

        # Keep the lower triangles in shared memory if they fit
        rows = max(max_rows, 1)
        triangle_size = max(max_rows * (max_rows + 1) // 2, 1)
        shared_bytes = 4 * (triangle_size + 3 * rows + 2 * block_dim) + _SHARED_MEMORY_MARGIN
        self.shared_triangle = device.is_cuda and shared_bytes <= device.max_shared_memory_per_block
        """Whether the lower triangles are eliminated in shared memory."""
        self._kernel, float_size, int_size = _make_shift_redundant_rows_kernel(
            block_dim, max_rows, device.is_cuda, self.shared_triangle
        )

        # Allocate the global scratch not held in shared memory
        triangle_shape = (1, 1) if self.shared_triangle else (num_worlds, triangle_size)
        scratch_worlds = 1 if device.is_cuda else num_worlds
        self._triangle = wp.zeros(triangle_shape, dtype=wp.float32, device=device)
        self._floats = wp.zeros((scratch_worlds, float_size), dtype=wp.float32, device=device)
        self._ints = wp.zeros((scratch_worlds, int_size), dtype=wp.int32, device=device)

        self.shifted = wp.zeros((num_worlds, rows), dtype=wp.int32, device=device)
        """Flags of the rows of each block shifted by the last elimination, of shape ``(num_worlds, max_rows)``."""
        self._diagonal = wp.zeros((num_worlds, rows), dtype=wp.float32, device=device)
        self._initialized = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        self._eigenvalue = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        self._reference = wp.zeros(num_worlds, dtype=wp.float32, device=device)

        self.pivot_rows = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        """The number of block rows of each world to eliminate again after the last check, zero if it passed."""
        self.refactor_dim = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        """The dimension of each matrix to refactorize after the last check, zero if it passed."""

        # Allocate the scratch of the check
        self._rhs = None
        if max_dim > 0:
            padded = ((max_dim + _CHECK_TILE - 1) // _CHECK_TILE) * _CHECK_TILE
            self._rhs = wp.zeros((num_worlds, padded, _CHECK_COLUMNS), dtype=wp.float32, device=device)
            self._iterate = wp.zeros((num_worlds, padded), dtype=wp.float32, device=device)
            self._column_rows = wp.zeros((num_worlds, _CHECK_COLUMNS), dtype=wp.int32, device=device)

    def reset(self, world_mask: wp.array[wp.bool] | None = None):
        """
        Clears the shifted rows of all worlds, or of the worlds selected by ``world_mask``,
        so that the next :meth:`check` fails for them.

        Args:
            world_mask: The worlds to reset, of shape ``(num_worlds,)``.
        """
        if world_mask is None:
            self.shifted.zero_()
            self._initialized.zero_()
            self._reference.zero_()
        else:
            wp.launch(
                _reset_shifted_rows,
                dim=self.shifted.shape,
                inputs=[world_mask],
                outputs=[self.shifted, self._initialized, self._reference],
                device=self.shifted.device,
            )

    def apply(
        self,
        dim: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        D: wp.array[wp.float32],
    ):
        """
        Shifts the diagonal of the numerically redundant rows of a diagonal block of each matrix in-place.

        Args:
            dim: The dimension of each matrix, of shape ``(num_worlds,)``.
            mio: The index offset of each row-major matrix in ``D``, of shape ``(num_worlds,)``.
            row_offset: The first row of the block in each matrix, of shape ``(num_worlds,)``.
            num_rows: The number of rows of the block in each matrix, of shape ``(num_worlds,)``.
            D: The flat array of the matrices.
        """
        self.shift(dim, mio, row_offset, num_rows, D)
        self.eliminate(dim, mio, row_offset, num_rows, D)

    def shift(
        self,
        dim: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        D: wp.array[wp.float32],
    ):
        """
        Shifts the diagonal of the rows shifted by the last elimination in-place, and stores the diagonal of the
        block of each matrix for :meth:`eliminate` and :meth:`check`.

        Args:
            dim: The dimension of each matrix, of shape ``(num_worlds,)``.
            mio: The index offset of each row-major matrix in ``D``, of shape ``(num_worlds,)``.
            row_offset: The first row of the block in each matrix, of shape ``(num_worlds,)``.
            num_rows: The number of rows of the block in each matrix, of shape ``(num_worlds,)``.
            D: The flat array of the matrices.
        """
        wp.launch(
            _shift_stored_rows,
            dim=self.shifted.shape,
            inputs=[dim, mio, row_offset, num_rows, self.shifted],
            outputs=[D, self._diagonal],
            device=D.device,
        )

    def eliminate(
        self,
        dim: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        D: wp.array[wp.float32],
    ):
        """
        Repeats the elimination of the blocks shifted by :meth:`shift`, and updates their shifts in-place.

        Only the diagonal stored by :meth:`shift` and the strict lower triangle of each block are read, so the
        diagonal of ``D`` may hold additional terms, e.g. a regularization. Worlds with zero rows are skipped.

        Args:
            dim: The dimension of each matrix, of shape ``(num_worlds,)``.
            mio: The index offset of each row-major matrix in ``D``, of shape ``(num_worlds,)``.
            row_offset: The first row of the block in each matrix, of shape ``(num_worlds,)``.
            num_rows: The number of rows of the block to eliminate in each matrix, of shape ``(num_worlds,)``,
                e.g. :attr:`pivot_rows`.
            D: The flat array of the matrices.
        """
        wp.launch_tiled(
            self._kernel,
            dim=self.num_worlds,
            inputs=[
                dim,
                mio,
                row_offset,
                num_rows,
                wp.float32(self.threshold),
                self._diagonal,
                self._eigenvalue,
                self._triangle,
                self._floats,
                self._ints,
            ],
            outputs=[D, self.shifted, self._initialized, self._reference],
            block_dim=self.block_dim,
            device=D.device,
        )

    def check(
        self,
        dim: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        D: wp.array[wp.float32],
        L: wp.array[wp.float32],
        ld: wp.array[wp.int32],
        perm: wp.array[wp.int32] | None = None,
        inverse_perm: wp.array[wp.int32] | None = None,
    ):
        """
        Checks the rows shifted by :meth:`shift` from the Cholesky factors of the shifted matrices, and selects
        the worlds that fail the check, or that were never eliminated, in :attr:`pivot_rows` and
        :attr:`refactor_dim`.

        Args:
            dim: The dimension of each matrix, of shape ``(num_worlds,)``.
            mio: The index offset of each row-major matrix in ``D`` and ``L``, of shape ``(num_worlds,)``.
            vio: The index offset of the rows of each matrix in ``perm``, of shape ``(num_worlds,)``.
            row_offset: The first row of the block in each matrix, of shape ``(num_worlds,)``.
            num_rows: The number of rows of the block in each matrix, of shape ``(num_worlds,)``.
            D: The flat array of the shifted matrices, with any regularization.
            L: The flat array of the lower Cholesky factors of the matrices of ``D``.
            ld: The row stride of each factor in ``L``, of shape ``(num_worlds,)``.
            perm: The original row of each row of the factors, if they factorize ``P D P^T``.
            inverse_perm: The row of the factors of each original row, if ``perm`` is given.
        """
        if self._rhs is None:
            raise ValueError("Checking the shifted rows requires `max_dim`.")
        permuted = perm is not None

        # The permutations are not read by the unpermuted kernel
        wp.launch_tiled(
            _make_check_kernel(permuted),
            dim=self.num_worlds,
            inputs=[
                dim,
                mio,
                vio,
                row_offset,
                num_rows,
                ld,
                perm if permuted else dim,
                inverse_perm if permuted else dim,
                D,
                L,
                self._diagonal,
                self.shifted,
                self._initialized,
                self._reference,
                wp.float32(self.threshold),
                self._rhs,
                self._iterate,
                self._column_rows,
            ],
            outputs=[self.pivot_rows, self.refactor_dim, self._eigenvalue],
            block_dim=_CHECK_BLOCK_DIM,
            device=D.device,
        )
