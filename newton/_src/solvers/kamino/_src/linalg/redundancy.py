# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Detection and removal of the numerically redundant rows of batched symmetric positive semi-definite matrices
"""

from __future__ import annotations

from functools import cache

import warp as wp

from ..core.math import FLOAT32_EPS
from ..core.types import vec6f
from .factorize._array_ptr import get_float32_array_offset_ptr
from .factorize.llt_blocked import llt_blocked_factorize, make_llt_blocked_factorize_kernel
from .sparse_matrix import BlockSparseMatrices

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

_ROW_ENTRIES = wp.constant(wp.int32(2))
"""Largest number of non-zero Jacobian blocks of a row assembled by :meth:`RedundantRowShift.assemble`."""

_FACTOR_BLOCK_SIZE = 32
"""Tile size of the factorization of the check."""

_FACTOR_BLOCK_DIM = 128
"""Number of threads per world of the factorization of the check."""

_CHECK_TILE = 32
"""Tile size of the triangular solves of the check."""

_CHECK_COLUMNS = 16
"""Number of right-hand sides of the triangular solves of the check."""

_CHECK_ITERATIONS = 2
"""Number of inverse iterations estimating the smallest eigenvalue of the shifted block."""

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


@wp.func
def _row_shift(diagonal: wp.float32, shifted: wp.int32) -> wp.float32:
    """Return the shift of a block row from its diagonal."""
    return wp.where(shifted != 0, wp.max(diagonal, FLOAT32_EPS), wp.float32(0.0))


@wp.func
def _mass_weighted_product(J_a: vec6f, J_b: vec6f, inv_m: wp.float32, inv_I: wp.mat33f) -> wp.float32:
    """Return the product ``J_a M^-1 J_b^T`` of two Jacobian blocks of a body."""
    linear = inv_m * wp.dot(wp.vec3f(J_a[0], J_a[1], J_a[2]), wp.vec3f(J_b[0], J_b[1], J_b[2]))
    angular = wp.dot(wp.vec3f(J_a[3], J_a[4], J_a[5]), inv_I @ wp.vec3f(J_b[3], J_b[4], J_b[5]))
    return linear + angular


###
# Kernels
###


@wp.kernel
def _store_block_diagonal(
    # Inputs:
    ld: wp.array[wp.int32],
    row_offset: wp.array[wp.int32],
    num_rows: wp.array[wp.int32],
    A: wp.array2d[wp.float32],
    # Outputs:
    diagonal: wp.array2d[wp.float32],
):
    """Store the diagonal of the block of each matrix."""
    wid, i = wp.tid()
    if i >= num_rows[wid]:
        return
    row = row_offset[wid] + i
    diagonal[wid, i] = A[wid, row * ld[wid] + row]


@wp.kernel
def _collect_row_entries(
    # Inputs:
    row_offset: wp.array[wp.int32],
    num_rows: wp.array[wp.int32],
    nzb_start: wp.array[wp.int32],
    num_nzb: wp.array[wp.int32],
    nzb_coords: wp.array2d[wp.int32],
    # Outputs:
    num_entries: wp.array2d[wp.int32],
    entries: wp.array3d[wp.int32],
):
    """Collect the non-zero Jacobian blocks of the rows of the block of each world."""
    wid, k = wp.tid()
    if k >= num_nzb[wid]:
        return
    index = nzb_start[wid] + k
    i = nzb_coords[index, 0] - row_offset[wid]
    if i < 0 or i >= num_rows[wid]:
        return
    slot = wp.atomic_add(num_entries, wid, i, 1)
    if slot < _ROW_ENTRIES:
        entries[wid, i, slot] = index


@wp.kernel
def _assemble_block(
    # Inputs:
    row_offset: wp.array[wp.int32],
    num_rows: wp.array[wp.int32],
    row_start: wp.array[wp.int32],
    bodies_offset: wp.array[wp.int32],
    inv_m: wp.array[wp.float32],
    inv_I: wp.array[wp.mat33f],
    nzb_coords: wp.array2d[wp.int32],
    nzb_values: wp.array[vec6f],
    preconditioner: wp.array[wp.float32],
    num_entries: wp.array2d[wp.int32],
    entries: wp.array3d[wp.int32],
    block_ld: wp.int32,
    # Outputs:
    block: wp.array2d[wp.float32],
    diagonal: wp.array2d[wp.float32],
):
    """Assemble the lower triangle of the block ``P J M^-1 J^T P`` of the rows of each world."""
    wid, i, j = wp.tid()
    if i >= num_rows[wid] or j > i:
        return

    # Sum the products of the Jacobian blocks of the bodies shared by both rows
    value = wp.float32(0.0)
    for a in range(wp.min(num_entries[wid, i], _ROW_ENTRIES)):
        index_a = entries[wid, i, a]
        body = nzb_coords[index_a, 1] // 6
        for b in range(wp.min(num_entries[wid, j], _ROW_ENTRIES)):
            index_b = entries[wid, j, b]
            if nzb_coords[index_b, 1] // 6 == body:
                g = bodies_offset[wid] + body
                value += _mass_weighted_product(nzb_values[index_a], nzb_values[index_b], inv_m[g], inv_I[g])
    if preconditioner:
        start = row_start[wid] + row_offset[wid]
        value *= preconditioner[start + i] * preconditioner[start + j]

    block[wid, i * block_ld + j] = value
    if i == j:
        diagonal[wid, i] = value


@wp.kernel
def _reset_shifted_rows(
    # Inputs:
    world_mask: wp.array[wp.bool],
    # Outputs:
    shifted: wp.array2d[wp.int32],
    shifts: wp.array2d[wp.float32],
    initialized: wp.array[wp.int32],
    reference: wp.array[wp.float32],
):
    """Clear the shifted rows of the masked worlds."""
    wid, i = wp.tid()
    if not world_mask[wid]:
        return
    shifted[wid, i] = 0
    shifts[wid, i] = 0.0
    if i == 0:
        initialized[wid] = 0
        reference[wid] = 0.0


@wp.kernel
def _update_shifts(
    # Inputs:
    ld: wp.array[wp.int32],
    row_offset: wp.array[wp.int32],
    num_rows: wp.array[wp.int32],
    shifted: wp.array2d[wp.int32],
    diagonal: wp.array2d[wp.float32],
    shift_matrix: wp.bool,
    # Outputs:
    shifts: wp.array2d[wp.float32],
    A: wp.array2d[wp.float32],
):
    """Store the shift of each block row, and add it to the diagonal of the matrices if ``shift_matrix``."""
    wid, i = wp.tid()
    value = wp.float32(0.0)
    if i < num_rows[wid]:
        value = _row_shift(diagonal[wid, i], shifted[wid, i])
    shifts[wid, i] = value
    if shift_matrix and value != 0.0:
        row = row_offset[wid] + i
        A[wid, row * ld[wid] + row] += value


@wp.kernel
def _update_reference(
    # Inputs:
    num_rows: wp.array[wp.int32],
    pivot_rows: wp.array[wp.int32],
    shifted: wp.array2d[wp.int32],
    shifts: wp.array2d[wp.float32],
    eigenvalue: wp.array[wp.float32],
    # Outputs:
    reference: wp.array[wp.float32],
):
    """
    Keep the eigenvalue of the eliminated worlds whose shifted rows did not change, given by their shifts before
    the update, as the reference of their next checks.
    """
    wid = wp.tid()
    if pivot_rows[wid] == 0:
        return
    changed = wp.int32(0)
    for i in range(num_rows[wid]):
        if wp.where(shifts[wid, i] != 0.0, 1, 0) != shifted[wid, i]:
            changed = 1
    reference[wid] = wp.where(changed == 0, eigenvalue[wid], wp.float32(0.0))


@wp.kernel
def _update_changed_shifts(
    # Inputs:
    ld: wp.array[wp.int32],
    row_offset: wp.array[wp.int32],
    num_rows: wp.array[wp.int32],
    pivot_rows: wp.array[wp.int32],
    shifted: wp.array2d[wp.int32],
    diagonal: wp.array2d[wp.float32],
    eigenvalue: wp.array[wp.float32],
    # Outputs:
    shifts: wp.array2d[wp.float32],
    reference: wp.array[wp.float32],
    A: wp.array2d[wp.float32],
):
    """
    Update the shifts of the eliminated worlds by their change in the diagonal of the matrices, and keep the
    eigenvalue of the worlds whose shifted rows did not change as the reference of their next checks.
    """
    wid = wp.tid()
    if pivot_rows[wid] == 0:
        return
    changed = wp.int32(0)
    for i in range(num_rows[wid]):
        value = _row_shift(diagonal[wid, i], shifted[wid, i])
        change = value - shifts[wid, i]
        if change != 0.0:
            row = row_offset[wid] + i
            A[wid, row * ld[wid] + row] += change
            changed = 1
        shifts[wid, i] = value
    reference[wid] = wp.where(changed == 0, eigenvalue[wid], wp.float32(0.0))


@wp.kernel
def _copy_shifted_block(
    # Inputs:
    ld: wp.array[wp.int32],
    row_offset: wp.array[wp.int32],
    num_rows: wp.array[wp.int32],
    A: wp.array2d[wp.float32],
    diagonal: wp.array2d[wp.float32],
    shifted: wp.array2d[wp.int32],
    initialized: wp.array[wp.int32],
    # Outputs:
    factor_dim: wp.array[wp.int32],
    factor: wp.array2d[wp.float32],
):
    """Copy the shifted block of the initialized worlds, with a row stride of its dimension, for its factorization."""
    wid, i, j = wp.tid()
    n = wp.where(initialized[wid] != 0, num_rows[wid], 0)
    if i == 0 and j == 0:
        factor_dim[wid] = n
    if i >= n or j >= n:
        return
    value = diagonal[wid, i] + _row_shift(diagonal[wid, i], shifted[wid, i])
    if i != j:
        start = row_offset[wid] * (ld[wid] + 1)
        value = A[wid, start + wp.max(i, j) * ld[wid] + wp.min(i, j)]
    factor[wid, i * n + j] = value


@cache
def _make_shift_redundant_rows_kernel(block_dim: int, max_rows: int, shared: bool, shared_triangle: bool):
    """
    Create the elimination kernel of the blocks of up to ``max_rows`` rows, and the sizes of its float and
    int scratch per world.

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
        ld: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        threshold: wp.float32,
        diagonal: wp.array2d[wp.float32],
        A: wp.array2d[wp.float32],
        # Scratch:
        global_triangle: wp.array2d[wp.float32],
        global_floats: wp.array2d[wp.float32],
        global_ints: wp.array2d[wp.int32],
        # Outputs:
        shifted: wp.array2d[wp.int32],
        initialized: wp.array[wp.int32],
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
        stride = ld[wid]
        start = row_offset[wid] * (stride + 1)
        A_i = A[wid]

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

        # Copy the lower triangle of the block
        for i in range(warp, n, warps):
            for j in range(lane, i, lanes):
                triangle[_packed(i, j)] = A_i[start + i * stride + j]
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
            half = threads // 2
            while half > 0:
                if tid < half and floats[ratio_offset + tid + half] > floats[ratio_offset + tid]:
                    floats[ratio_offset + tid] = floats[ratio_offset + tid + half]
                    ints[rows + tid] = ints[rows + tid + half]
                _sync_threads()
                half = half // 2
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

        # Store the shifted rows
        _sync_threads()
        for i in range(tid, n, threads):
            shifted[wid, i] = wp.where(ints[i] == _ROW_SHIFTED, 1, 0)
        if tid == 0:
            initialized[wid] = 1

    return _shift_redundant_rows, float_size, int_size


@cache
def _make_triangular_solves():
    """Create the in-place tiled solves with a lower Cholesky factor and its transpose of the checks."""
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

    return forward, backward


@cache
def _make_check_kernel():
    """
    Create the kernel checking the shifted rows of each world from the Cholesky factor of its shifted block,
    stored with a row stride of its dimension.
    """
    tile = _CHECK_TILE
    columns = _CHECK_COLUMNS
    forward, backward = _make_triangular_solves()

    @wp.kernel(module="unique", enable_backward=False)
    def _check_shifted_rows(
        # Inputs:
        num_rows: wp.array[wp.int32],
        factor_dim: wp.array[wp.int32],
        L: wp.array2d[wp.float32],
        diagonal: wp.array2d[wp.float32],
        shifted: wp.array2d[wp.int32],
        reference: wp.array[wp.float32],
        threshold: wp.float32,
        # Scratch:
        X: wp.array3d[wp.float32],
        V: wp.array2d[wp.float32],
        column_rows: wp.array2d[wp.int32],
        # Outputs:
        pivot_rows: wp.array[wp.int32],
        eigenvalue_out: wp.array[wp.float32],
    ):
        wid, tid = wp.tid()
        threads = wp.block_dim()
        rows = num_rows[wid]
        n = factor_dim[wid]
        if n == 0:
            if tid == 0:
                pivot_rows[wid] = rows
                eigenvalue_out[wid] = 0.0
            return
        padded = ((n + tile - 1) // tile) * tile
        L_i = wp.array(ptr=get_float32_array_offset_ptr(L[wid], 0), shape=(n, n), dtype=wp.float32)
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
        for i in range(tid, n, threads):
            count += shifted[wid, i]
        num_shifted = wp.tile_sum(wp.tile(count))[0]

        # Solve with the iterate of the inverse iterations in the first column, and the unit vectors of the
        # shifted rows by chunks in the other columns
        slots = columns - 1
        passes = wp.max(_CHECK_ITERATIONS, (num_shifted + slots - 1) // slots)
        rayleigh = wp.float32(0.0)
        failed = wp.int32(0)
        for p in range(passes):
            iterate = p < _CHECK_ITERATIONS
            for k in range(tid, padded * columns, threads):
                h = k // columns
                value = wp.float32(0.0)
                if iterate and k - h * columns == 0 and h < n:
                    K = diagonal[wid, h] + _row_shift(diagonal[wid, h], shifted[wid, h])
                    value = wp.sqrt(wp.max(K, FLOAT32_EPS)) * V_i[h] * inv_norm
                X_i[h, k - h * columns] = value
            for c in range(tid, columns, threads):
                column_rows[wid, c] = -1
            _sync_threads()
            if tid == 0:
                k = wp.int32(0)
                for i in range(n):
                    if shifted[wid, i] != 0:
                        c = 1 + k - p * slots
                        if c >= 1 and c < columns:
                            X_i[i, c] = 1.0
                            column_rows[wid, c] = i
                        k += 1
            _sync_threads()
            forward(L_i, X_i, n, tid)
            backward(L_i, X_i, n, tid)
            _sync_threads()

            # Accumulate the Rayleigh quotient of the inverse of the Jacobi-scaled block
            if iterate:
                quotient = wp.float32(0.0)
                norm = wp.float32(0.0)
                for h in range(tid, n, threads):
                    K = diagonal[wid, h] + _row_shift(diagonal[wid, h], shifted[wid, h])
                    x = V_i[h] * inv_norm
                    z = wp.sqrt(wp.max(K, FLOAT32_EPS)) * X_i[h, 0]
                    quotient += x * z
                    norm += z * z
                    V_i[h] = z
                rayleigh = wp.tile_sum(wp.tile(quotient))[0]
                inv_norm = 1.0 / wp.sqrt(wp.tile_sum(wp.tile(norm))[0])

            # A shifted row is independent if its Schur complement exceeds its shift
            # NOTE: The Schur complement is the residual of a ridge regression of the row on the other rows, with
            # the coefficients -u_o / u_s of the column u of the inverse, penalized by the shifts of the other rows.
            independent = wp.int32(0)
            for c in range(1 + tid, columns, threads):
                i = column_rows[wid, c]
                if i >= 0:
                    d = wp.max(diagonal[wid, i], FLOAT32_EPS)
                    residual = (1.0 / X_i[i, c] - d) / d
                    if not (residual <= _SHIFTED_ROW_MARGIN * threshold):
                        independent += 1
            failed += wp.tile_sum(wp.tile(independent))[0]

        # The kept rows are dependent if the smallest eigenvalue is small, or much smaller than its reference
        eigenvalue = 1.0 / rayleigh
        limit = threshold
        if reference[wid] > 0.0:
            limit = wp.min(threshold, _REFERENCE_MARGIN * reference[wid])
        if not (eigenvalue >= limit):
            failed += 1

        # Select the world for a new elimination
        if tid == 0:
            pivot_rows[wid] = wp.where(failed > 0, rows, 0)
            eigenvalue_out[wid] = eigenvalue

    return _check_shifted_rows


@cache
def _make_factor_check_kernel(permuted: bool):
    """
    Create the kernel checking the shifted rows of each world from the Cholesky factor ``L`` of the shifted
    and regularized matrix ``K``, stored with the row stride ``ld`` and permuted by ``perm`` if ``permuted``.
    """
    tile = _CHECK_TILE
    columns = _CHECK_COLUMNS
    forward, backward = _make_triangular_solves()

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
    def _check_shifted_rows_on_factor(
        # Inputs:
        dim: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        ld: wp.array[wp.int32],
        perm: wp.array[wp.int32],
        inverse_perm: wp.array[wp.int32],
        D_storage: wp.array2d[wp.float32],
        L: wp.array2d[wp.float32],
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
        vstart = vio[wid]
        padded = ((n + tile - 1) // tile) * tile
        stride = ld[wid]
        D = D_storage[wid]
        # NOTE: The rows beyond the dimension are out of bounds, and loaded as zeros by the triangular solves
        L_i = wp.array(ptr=get_float32_array_offset_ptr(L[wid], 0), shape=(n, stride), dtype=wp.float32)
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
                    value = wp.sqrt(wp.max(D[o * (n + 1)], FLOAT32_EPS)) * V_i[h] * inv_norm
                X_i[h, k - h * columns] = value
            for c in range(tid, columns, threads):
                column_rows[wid, c] = -1
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
                    K = wp.max(D[o * (n + 1)], FLOAT32_EPS)
                    x = V_i[h] * inv_norm
                    z = wp.sqrt(K) * X_i[h, 0]
                    quotient += x * z
                    norm += z * z
                    i = o - offset
                    if i >= 0 and i < rows:
                        d = diagonal[wid, i]
                        regularization += x * x * (K - d - _row_shift(d, shifted[wid, i])) / K
                    V_i[h] = z
                rayleigh = wp.tile_sum(wp.tile(quotient))[0]
                correction = wp.tile_sum(wp.tile(regularization))[0]
                inv_norm = 1.0 / wp.sqrt(wp.tile_sum(wp.tile(norm))[0])

            # A shifted row is independent if its Schur complement exceeds its shift and regularization
            # NOTE: The Schur complement is the residual of a ridge regression of the row on the other rows,
            # with the coefficients -u_o / u_s of the column u of the inverse. The penalty of the coefficients
            # by the regularization of the other block rows is removed, but not by the shifts.
            independent = wp.int32(0)
            for c in range(1 + tid, columns, threads):
                i = column_rows[wid, c]
                if i >= 0:
                    u_s = X_i[factor_row(inverse_perm, vstart, offset + i), c]
                    penalty = wp.float32(0.0)
                    for h in range(n):
                        j = original_row(perm, vstart, h) - offset
                        if j >= 0 and j < rows and j != i:
                            d = diagonal[wid, j]
                            u_o = X_i[h, c]
                            penalty += u_o * u_o * (D[(offset + j) * (n + 1)] - d - _row_shift(d, shifted[wid, j]))
                    o = offset + i
                    d = diagonal[wid, i]
                    schur = 1.0 / u_s - penalty / (u_s * u_s)
                    residual = (schur - (D[o * (n + 1)] - d)) / wp.max(d, FLOAT32_EPS)
                    if not (residual <= _SHIFTED_ROW_MARGIN * threshold):
                        independent += 1
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

    return _check_shifted_rows_on_factor


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
    by their original diagonal before the elimination proceeds, and the same shift is stored in :attr:`shifts`.
    In float32, rounding lifts the relative pivots of exactly redundant rows to about ``1e-6``.

    The matrix shifted by :attr:`shifts` is positive definite on the block and can be factorized in any order.
    Since each shifted row lies in the span of the other rows of the block, the solutions of consistent systems
    with the shifted matrix carry no impulse on the shifted rows and solve the original system.

    The block is read from the matrices loaded with :meth:`load`, which :meth:`detect` shifts in-place, or
    assembled from a block-sparse Jacobian with :meth:`assemble`. The elimination runs in float32 on a packed
    copy of the lower triangle of each block, with one thread block per world, and does not synchronize with
    the host. On CUDA devices, the copy is kept in shared memory if it fits, and in global memory otherwise.

    Instead of repeating the elimination at every call, :meth:`detect` can reuse the shifted rows of the last
    elimination, and check them from the Cholesky factor of the shifted block with :meth:`check`. The check
    fails if the smallest eigenvalue of the Jacobi-scaled shifted block is below ``threshold``, i.e. if the
    kept rows became dependent, or if the Schur complement of a shifted row exceeds its shift by ten times
    ``threshold`` relative to its diagonal, i.e. if the shifted row became independent. The elimination is
    then repeated for the failed worlds only. Since the relative pivots do not bound the smallest eigenvalue,
    valid shifted rows can fail the check. So, when the elimination of a failed world keeps the same rows, its
    eigenvalue becomes the reference of its next checks, which only fail below a tenth of it.

    When the loaded matrices are factorized anyway, the shifted rows can instead be checked from the Cholesky
    factor of the whole shifted and regularized matrix, which saves the factorization of the block: :meth:`shift`
    adds the shifts of the last elimination to the matrices before their factorization, :meth:`check_factor`
    checks the factor without the regularization of the block, and :meth:`eliminate` repeats the elimination of
    the failed worlds, which must then be refactorized. The coupling of the block with the other rows can also
    lower the eigenvalue, e.g. with hyperstatic contacts, which the elimination of the block cannot fix, and the
    reference applies in the same way. Conversely, the other rows can span a shifted row that became independent
    of the block, e.g. when leaving a configuration with more redundant rows, which then remains shifted for a
    few calls, while the constraint it carries is not enforced.

    Args:
        num_worlds: The number of worlds.
        max_rows: The largest number of rows of the block across worlds.
        threshold: The relative pivot below which a row is shifted. Must be in ``[0, 1)``.
        block_dim: The number of threads per world of the elimination. Must be a power of two of at least 32.
        check: Whether to allocate the scratch of :meth:`check`.
        sparse: Whether to allocate the blocks assembled by :meth:`assemble`.
        max_dim: The largest dimension of the loaded matrices, required by :meth:`check_factor`.
        device: The device on which to allocate the scratch memory.
    """

    def __init__(
        self,
        num_worlds: int,
        max_rows: int,
        threshold: float,
        block_dim: int = 512,
        check: bool = False,
        sparse: bool = False,
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
        self.max_rows = max_rows

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
        self.shifts = wp.zeros((num_worlds, rows), dtype=wp.float32, device=device)
        """The diagonal shift of the rows of each block, of shape ``(num_worlds, max_rows)``."""
        self._diagonal = wp.zeros((num_worlds, rows), dtype=wp.float32, device=device)
        self._initialized = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        self._eigenvalue = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        self._reference = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        self.pivot_rows = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        """The number of block rows of each world to eliminate again after the last check, zero if it passed."""
        self.refactor_dim = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        """The dimension of each matrix to refactorize after the last :meth:`check_factor`, zero if it passed."""

        # Allocate the dense blocks of each world, with a row stride of ``max_rows``
        self._block = None
        if sparse:
            self._block = wp.zeros((num_worlds, rows * rows), dtype=wp.float32, device=device)
            self._block_ld = wp.full(num_worlds, rows, dtype=wp.int32, device=device)
            self._block_offset = wp.zeros(num_worlds, dtype=wp.int32, device=device)
            self._num_entries = wp.zeros((num_worlds, rows), dtype=wp.int32, device=device)
            self._entries = wp.zeros((num_worlds, rows, _ROW_ENTRIES), dtype=wp.int32, device=device)

        # Allocate the scratch of the checks
        self._factor = None
        if check:
            self._factor = wp.zeros((num_worlds, rows * rows), dtype=wp.float32, device=device)
            self._factor_dim = wp.zeros(num_worlds, dtype=wp.int32, device=device)
            self._factor_kernel = make_llt_blocked_factorize_kernel(_FACTOR_BLOCK_SIZE)
        self._max_dim = max_dim
        solve_rows = max(rows if check else 0, max_dim)
        if solve_rows > 0:
            padded = ((solve_rows + _CHECK_TILE - 1) // _CHECK_TILE) * _CHECK_TILE
            self._rhs = wp.zeros((num_worlds, padded, _CHECK_COLUMNS), dtype=wp.float32, device=device)
            self._iterate = wp.zeros((num_worlds, padded), dtype=wp.float32, device=device)
            self._column_rows = wp.zeros((num_worlds, _CHECK_COLUMNS), dtype=wp.int32, device=device)

        # The block source of the last call to `load` or `assemble`
        self._source: tuple | None = None
        self._num_rows: wp.array[wp.int32] | None = None
        self._shift_matrix = False

    def reset(self, world_mask: wp.array[wp.bool] | None = None):
        """
        Clears the shifted rows of all worlds, or of the worlds selected by ``world_mask``,
        so that the next :meth:`check` fails for them.

        Args:
            world_mask: The worlds to reset, of shape ``(num_worlds,)``.
        """
        if world_mask is None:
            self.shifted.zero_()
            self.shifts.zero_()
            self._initialized.zero_()
            self._reference.zero_()
        else:
            wp.launch(
                _reset_shifted_rows,
                dim=self.shifted.shape,
                inputs=[world_mask],
                outputs=[self.shifted, self.shifts, self._initialized, self._reference],
                device=self.shifted.device,
            )

    def load(
        self,
        dim: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        D: wp.array2d[wp.float32],
    ):
        """
        Selects a diagonal block of row-major dense matrices, which :meth:`detect` shifts in-place, and stores
        its diagonal.

        Args:
            dim: The dimension of each matrix, of shape ``(num_worlds,)``.
            row_offset: The first row of the block in each matrix, of shape ``(num_worlds,)``.
            num_rows: The number of rows of the block in each matrix, of shape ``(num_worlds,)``.
            D: The matrices, one per row.
        """
        self._source = (dim, row_offset, D)
        self._num_rows = num_rows
        self._shift_matrix = True
        wp.launch(
            _store_block_diagonal,
            dim=self.shifted.shape,
            inputs=[*self._source[:2], num_rows, D],
            outputs=[self._diagonal],
            device=D.device,
        )

    def assemble(
        self,
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        jacobian: BlockSparseMatrices,
        bodies_offset: wp.array[wp.int32],
        inv_m: wp.array[wp.float32],
        inv_I: wp.array[wp.mat33f],
        preconditioner: wp.array[wp.float32] | None = None,
    ):
        """
        Assembles the block ``P J M^-1 J^T P`` of the rows of each world from its block-sparse Jacobian ``J``,
        whose rows have at most two non-zero blocks, and stores its diagonal.

        Args:
            row_offset: The first row of the block in each Jacobian, of shape ``(num_worlds,)``.
            num_rows: The number of rows of the block in each Jacobian, of shape ``(num_worlds,)``.
            jacobian: The block-sparse Jacobians, with ``1 x 6`` blocks per body.
            bodies_offset: The index of the first body of each world, of shape ``(num_worlds,)``.
            inv_m: The inverse mass of each body.
            inv_I: The inverse inertia of each body.
            preconditioner: The diagonal ``P`` of each world, indexed by the row start of the Jacobians,
                or ``None`` for the identity.
        """
        if self._block is None:
            raise ValueError("Assembling the blocks requires `sparse`.")
        rows = max(self.max_rows, 1)
        self._source = (self._block_ld, self._block_offset, self._block)
        self._num_rows = num_rows
        self._shift_matrix = False
        device = self._block.device
        self._num_entries.zero_()
        wp.launch(
            _collect_row_entries,
            dim=(self.num_worlds, jacobian.max_of_num_nzb),
            inputs=[row_offset, num_rows, jacobian.nzb_start, jacobian.num_nzb, jacobian.nzb_coords],
            outputs=[self._num_entries, self._entries],
            device=device,
        )
        wp.launch(
            _assemble_block,
            dim=(self.num_worlds, rows, rows),
            inputs=[
                row_offset,
                num_rows,
                jacobian.row_start,
                bodies_offset,
                inv_m,
                inv_I,
                jacobian.nzb_coords,
                jacobian.nzb_values,
                preconditioner,
                self._num_entries,
                self._entries,
                rows,
            ],
            outputs=[self._block, self._diagonal],
            device=device,
        )

    def detect(self, reuse: bool = False):
        """
        Eliminates the blocks selected by the last :meth:`load` or :meth:`assemble`, or only those failing
        :meth:`check` if ``reuse``, and updates :attr:`shifts`, which are added to the diagonal of the loaded
        matrices.

        Args:
            reuse: Whether to keep the shifted rows of the last elimination where they pass the check.
        """
        if self._source is None:
            raise ValueError("The blocks must be loaded or assembled first.")
        num_rows = self._num_rows
        if reuse:
            self.check()
            num_rows = self.pivot_rows
        wp.launch_tiled(
            self._kernel,
            dim=self.num_worlds,
            inputs=[
                *self._source[:2],
                num_rows,
                wp.float32(self.threshold),
                self._diagonal,
                self._source[2],
                self._triangle,
                self._floats,
                self._ints,
            ],
            outputs=[self.shifted, self._initialized],
            block_dim=self.block_dim,
            device=self.shifted.device,
        )
        if reuse:
            wp.launch(
                _update_reference,
                dim=self.num_worlds,
                inputs=[self._num_rows, self.pivot_rows, self.shifted, self.shifts, self._eigenvalue],
                outputs=[self._reference],
                device=self.shifted.device,
            )
        wp.launch(
            _update_shifts,
            dim=self.shifted.shape,
            inputs=[*self._source[:2], self._num_rows, self.shifted, self._diagonal, self._shift_matrix],
            outputs=[self.shifts, self._source[2]],
            device=self.shifted.device,
        )

    def check(self):
        """
        Checks the rows shifted by the last elimination on the blocks selected by the last :meth:`load` or
        :meth:`assemble`, and selects the worlds that fail the check, or that were never eliminated,
        in :attr:`pivot_rows`.
        """
        if self._factor is None:
            raise ValueError("Checking the shifted rows requires `check`.")
        if self._source is None:
            raise ValueError("The blocks must be loaded or assembled first.")
        rows = max(self.max_rows, 1)
        device = self._factor.device

        # Copy and factorize the shifted blocks
        wp.launch(
            _copy_shifted_block,
            dim=(self.num_worlds, rows, rows),
            inputs=[
                *self._source[:2],
                self._num_rows,
                self._source[2],
                self._diagonal,
                self.shifted,
                self._initialized,
            ],
            outputs=[self._factor_dim, self._factor],
            device=device,
        )
        llt_blocked_factorize(
            kernel=self._factor_kernel,
            dim=self._factor_dim,
            A=self._factor,
            L=self._factor,
            num_blocks=self.num_worlds,
            block_dim=_FACTOR_BLOCK_DIM,
        )

        # Check the shifted rows from the factors
        wp.launch_tiled(
            _make_check_kernel(),
            dim=self.num_worlds,
            inputs=[
                self._num_rows,
                self._factor_dim,
                self._factor,
                self._diagonal,
                self.shifted,
                self._reference,
                wp.float32(self.threshold),
                self._rhs,
                self._iterate,
                self._column_rows,
            ],
            outputs=[self.pivot_rows, self._eigenvalue],
            block_dim=_CHECK_BLOCK_DIM,
            device=device,
        )

    def shift(self):
        """
        Adds the shifts of the rows shifted by the last elimination to the diagonal of the matrices loaded with
        :meth:`load`, before their factorization and :meth:`check_factor`.
        """
        if self._source is None or not self._shift_matrix:
            raise ValueError("The matrices must be loaded first.")
        wp.launch(
            _update_shifts,
            dim=self.shifted.shape,
            inputs=[*self._source[:2], self._num_rows, self.shifted, self._diagonal, True],
            outputs=[self.shifts, self._source[2]],
            device=self.shifted.device,
        )

    def check_factor(
        self,
        vio: wp.array[wp.int32],
        L: wp.array2d[wp.float32],
        ld: wp.array[wp.int32],
        perm: wp.array[wp.int32] | None = None,
        inverse_perm: wp.array[wp.int32] | None = None,
    ):
        """
        Checks the rows shifted by :meth:`shift` from the Cholesky factors of the loaded matrices, which may hold
        a regularization on their diagonal, and selects the worlds that fail the check, or that were never
        eliminated, in :attr:`pivot_rows` and :attr:`refactor_dim`.

        Args:
            vio: The index offset of the rows of each matrix in ``perm``, of shape ``(num_worlds,)``.
            L: The lower Cholesky factors of the matrices, one per row.
            ld: The row stride of each factor in ``L``, of shape ``(num_worlds,)``.
            perm: The original row of each row of the factors, if they factorize ``P D P^T``.
            inverse_perm: The row of the factors of each original row, if ``perm`` is given.
        """
        if self._max_dim == 0:
            raise ValueError("Checking the shifted rows on the factor requires `max_dim`.")
        if self._source is None or not self._shift_matrix:
            raise ValueError("The matrices must be loaded first.")
        dim, row_offset, D = self._source
        permuted = perm is not None

        # The permutations are not read by the unpermuted kernel
        wp.launch_tiled(
            _make_factor_check_kernel(permuted),
            dim=self.num_worlds,
            inputs=[
                dim,
                vio,
                row_offset,
                self._num_rows,
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

    def eliminate(self):
        """
        Repeats the elimination of the worlds selected by :meth:`check_factor`, and updates their shifts in the
        diagonal of the loaded matrices, which then need to be refactorized for the dimensions of
        :attr:`refactor_dim`.
        """
        if self._source is None or not self._shift_matrix:
            raise ValueError("The matrices must be loaded first.")
        device = self.shifted.device
        wp.launch_tiled(
            self._kernel,
            dim=self.num_worlds,
            inputs=[
                *self._source[:2],
                self.pivot_rows,
                wp.float32(self.threshold),
                self._diagonal,
                self._source[2],
                self._triangle,
                self._floats,
                self._ints,
            ],
            outputs=[self.shifted, self._initialized],
            block_dim=self.block_dim,
            device=device,
        )
        wp.launch(
            _update_changed_shifts,
            dim=self.num_worlds,
            inputs=[
                *self._source[:2],
                self._num_rows,
                self.pivot_rows,
                self.shifted,
                self._diagonal,
                self._eigenvalue,
            ],
            outputs=[self.shifts, self._reference, self._source[2]],
            device=device,
        )
