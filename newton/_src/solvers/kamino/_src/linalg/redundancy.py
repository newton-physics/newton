# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Detection and removal of the numerically redundant rows of batched symmetric positive semi-definite matrices
"""

from __future__ import annotations

from functools import cache

import warp as wp

from ..core.math import FLOAT32_EPS

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
        dim: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        num_rows: wp.array[wp.int32],
        threshold: wp.float32,
        # Scratch:
        global_triangle: wp.array2d[wp.float32],
        global_floats: wp.array2d[wp.float32],
        global_ints: wp.array2d[wp.int32],
        # Outputs:
        D: wp.array[wp.float32],
        shifted: wp.array2d[wp.int32],
    ):
        wid, tid = wp.tid()
        threads = wp.block_dim()
        lanes = wp.min(threads, _WARP_SIZE)
        warps = threads // lanes
        warp = tid // lanes
        lane = tid - warp * lanes
        n = num_rows[wid]
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

        # Copy the lower triangle and the diagonal of the block
        for i in range(warp, n, warps):
            for j in range(lane, i + 1, lanes):
                triangle[_packed(i, j)] = D[start + i * ld + j]
        for i in range(tid, n, threads):
            floats[diagonal_offset + i] = wp.max(D[start + i * (ld + 1)], FLOAT32_EPS)
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

        # Shift the diagonal of the dropped rows
        for i in range(tid, n, threads):
            dropped = ints[i] == _ROW_SHIFTED
            if dropped:
                D[start + i * (ld + 1)] += floats[diagonal_offset + i]
            shifted[wid, i] = wp.where(dropped, 1, 0)

    return _shift_redundant_rows, float_size, int_size


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

    Args:
        num_worlds: The number of worlds.
        max_rows: The largest number of rows of the block across worlds.
        threshold: The relative pivot below which a row is shifted. Must be in ``[0, 1)``.
        block_dim: The number of threads per world. Must be a power of two of at least 32.
        device: The device on which to allocate the scratch memory.
    """

    def __init__(
        self,
        num_worlds: int,
        max_rows: int,
        threshold: float,
        block_dim: int = 512,
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
        """Flags of the rows of each block shifted by the last call, of shape ``(num_worlds, max_rows)``."""

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
        wp.launch_tiled(
            self._kernel,
            dim=self.num_worlds,
            inputs=[
                dim,
                mio,
                row_offset,
                num_rows,
                wp.float32(self.threshold),
                self._triangle,
                self._floats,
                self._ints,
            ],
            outputs=[D, self.shifted],
            block_dim=self.block_dim,
            device=D.device,
        )
