# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KAMINO: Linear Algebra: Blocked LLT (i.e. Cholesky) factorization using Warp's Tile API."""

from ctypes import sizeof
from functools import cache

import warp as wp
from warp.context import Devicelike

from ...core.types import float32, int32

###
# Module interface
###

__all__ = [
    "llt_blocked_factorize",
    "llt_blocked_solve",
    "llt_blocked_solve_inplace",
    "make_llt_blocked_factorize_kernel",
    "make_llt_blocked_solve_inplace_kernel",
    "make_llt_blocked_solve_kernel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###

get_array_ptr_cpp = """return (uint64_t)arr.data;"""
"""A native C++ function to get the raw pointer of a warp array."""


def make_get_array_offset_ptr_func(dtype):
    """Creates a function to get the offset pointer of a warp array."""

    # Define a Warp wrapper around a native C++ function to get the raw pointer of a warp array
    @wp.func_native(get_array_ptr_cpp)
    def get_dtype_array_ptr(arr: wp.array(dtype=dtype)) -> wp.uint64: ...

    # Define a Warp function to get the raw pointer of a warp array with an offset
    @wp.func
    def get_dtype_array_offset_ptr(arr: wp.array(dtype=dtype), start_index: int) -> wp.uint64:
        return get_dtype_array_ptr(arr) + wp.uint64(start_index * wp.static(sizeof(dtype._type_)))

    return get_dtype_array_offset_ptr


get_int32_array_offset_ptr = make_get_array_offset_ptr_func(wp.int32)
"""A Warp function to get the offset pointer of a int32 warp array."""

get_float32_array_offset_ptr = make_get_array_offset_ptr_func(wp.float32)
"""A Warp function to get the offset pointer of a float32 warp array."""


# @wp.func
# def tile_sum_func(a: wp.tile(dtype=float, shape=(TILE_M, TILE_N))):
#     return wp.tile_sum(a) * 0.5
# def make_tile_pad(block_size: int, dtype=float32):
#     """Creates a function to pad a tile with identity values where the tile exceeds the matrix dimensions."""
#     @wp.func
#     def tile_pad(
#         T: wp.tile(dtype=dtype, shape=(block_size, block_size)),
#         i: int,
#         j: int,
#         n_i: int,
#         n_j: int,
#         tid_block: int,
#         num_threads_per_block: int,
#     ) -> None:
#         """Pads a tile T with identity values where the tile exceeds the matrix dimensions."""
#         if i + block_size > n_i or j + block_size > n_j:
#             num_tile_elements = block_size * block_size
#             num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
#             for ii in range(num_iterations):
#                 linear_index = tid_block + ii * num_threads_per_block
#                 linear_index = linear_index % num_tile_elements
#                 row = linear_index // block_size
#                 col = linear_index % block_size
#                 value = T[row, col]
#                 if i + row >= n_i or j + col >= n_j:
#                     value = wp.where(i + row == j + col, dtype(1), dtype(0))
#                 T[row, col] = value

#     return tile_pad


###
# Kernels
###


@cache
def make_llt_blocked_factorize_kernel(block_size: int):
    @wp.kernel
    def llt_blocked_factorize_kernel(
        # Inputs:
        dim: wp.array(dtype=int32),
        mio: wp.array(dtype=int32),
        A: wp.array(dtype=float32),
        # Outputs:
        L: wp.array(dtype=float32),
    ):
        # Retrieve the thread index and thread-block configuration
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        # Retrieve the matrix block dimensions and size
        n_i = dim[tid]
        A_i_start = mio[tid]

        # Retrieve a pointer to the start of the i-th matrix in A
        A_i_ptr = get_float32_array_offset_ptr(A, A_i_start)
        L_i_ptr = get_float32_array_offset_ptr(L, A_i_start)

        # Create a temporary warp array pointing to the i-th matrix
        A_i = wp.array(ptr=A_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)

        # Round up active_matrix_size to next multiple of block_size
        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size

        # Process the matrix in blocks along its leading dimension.
        for k in range(0, n_i_padded, block_size):
            end = k + block_size

            # Load current diagonal block A[k:end, k:end]
            # and update with contributions from previously computed blocks.
            A_kk_tile = wp.tile_load(A_i, shape=(block_size, block_size), offset=(k, k), storage="shared")

            # The following if pads the matrix if it is not divisible by block_size
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

            # Update the diagonal block with contributions from previously computed blocks
            if k > 0:
                for j in range(0, k, block_size):
                    L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(k, j))
                    L_block_T = wp.tile_transpose(L_block)
                    L_L_T_block = wp.tile_matmul(L_block, L_block_T)
                    A_kk_tile -= L_L_T_block

            # Compute the Cholesky factorization for the block
            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L_i, L_kk_tile, offset=(k, k))

            # Process the blocks below the current block
            for i in range(end, n_i_padded, block_size):
                # Load the current block A[i:end, k:end]
                A_ik_tile = wp.tile_load(A_i, shape=(block_size, block_size), offset=(i, k), storage="shared")

                # The following if pads the matrix if it is not divisible by block_size
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

                # Update the block with contributions from previously computed blocks
                if k > 0:
                    for j in range(0, k, block_size):
                        L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                        L_2_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(k, j))
                        L_T_tile = wp.tile_transpose(L_2_tile)
                        L_L_T_tile = wp.tile_matmul(L_tile, L_T_tile)
                        A_ik_tile -= L_L_T_tile

                # Solve for the current block
                t = wp.tile_transpose(A_ik_tile)
                tmp = wp.tile_lower_solve(L_kk_tile, t)
                sol_tile = wp.tile_transpose(tmp)
                wp.tile_store(L_i, sol_tile, offset=(i, k))

    # Return the kernel function
    return llt_blocked_factorize_kernel


@cache
def make_llt_blocked_solve_kernel(block_size: int):
    @wp.kernel
    def llt_blocked_solve_kernel(
        # Inputs:
        dim: wp.array(dtype=int32),
        mio: wp.array(dtype=int32),
        vio: wp.array(dtype=int32),
        L: wp.array(dtype=float32),
        b: wp.array(dtype=float32),
        # Outputs:
        y: wp.array(dtype=float32),
        x: wp.array(dtype=float32),
    ):
        # Retrieve the thread index and thread-block configuration
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        # Retrieve the matrix block dimensions and size
        n_i = dim[tid]
        L_i_start = mio[tid]
        v_i_start = vio[tid]

        # Retrieve a pointer to the start of the i-th matrix in A
        L_i_ptr = get_float32_array_offset_ptr(L, L_i_start)
        b_i_ptr = get_float32_array_offset_ptr(b, v_i_start)
        y_i_ptr = get_float32_array_offset_ptr(y, v_i_start)
        x_i_ptr = get_float32_array_offset_ptr(x, v_i_start)

        # Create a temporary warp array pointing to the i-th matrix
        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        b_i = wp.array(ptr=b_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        y_i = wp.array(ptr=y_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_i = wp.array(ptr=x_i_ptr, shape=(n_i, 1), dtype=wp.float32)

        # Round up n_i to next multiple of block_size
        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L y = b
        for i in range(0, n_i_padded, block_size):
            rhs_tile = wp.tile_load(b_i, shape=(block_size, 1), offset=(i, 0))
            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y_i, shape=(block_size, 1), offset=(j, 0))
                    Ly_block = wp.tile_matmul(L_block, y_block)
                    rhs_tile -= Ly_block
            L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            y_tile = wp.tile_lower_solve(L_tile, rhs_tile)
            wp.tile_store(y_i, y_tile, offset=(i, 0))

        # Backward substitution: solve L^T x = y
        for i in range(n_i_padded - block_size, -1, -block_size):
            i_end = i + block_size
            rhs_tile = wp.tile_load(y_i, shape=(block_size, 1), offset=(i, 0))
            if i_end < n_i_padded:
                for j in range(i_end, n_i_padded, block_size):
                    L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(j, i))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x_i, shape=(block_size, 1), offset=(j, 0))
                    L_T_x_tile = wp.tile_matmul(L_T_tile, x_tile)
                    rhs_tile -= L_T_x_tile
            L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))

            # The following if pads the matrix if it is not divisible by block_size
            if i + block_size > n_i:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    linear_index = tid_block + ii * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = L_tile[row, col]
                    if i + row >= n_i:
                        value = wp.where(i + row == i + col, float32(1), float32(0))
                    L_tile[row, col] = value
            # wp.static(make_tile_pad(block_size=block_size, dtype=float32))(L_tile, i, i, n_i, n_i, tid_block, num_threads_per_block)

            L_T_tile = wp.tile_transpose(L_tile)
            x_tile = wp.tile_upper_solve(L_T_tile, rhs_tile)
            wp.tile_store(x_i, x_tile, offset=(i, 0))

    # Return the kernel function
    return llt_blocked_solve_kernel


@cache
def make_llt_blocked_solve_inplace_kernel(block_size: int):
    @wp.kernel
    def llt_blocked_solve_inplace_kernel(
        # Inputs:
        dim: wp.array(dtype=int32),
        mio: wp.array(dtype=int32),
        vio: wp.array(dtype=int32),
        L: wp.array(dtype=float32),
        # Outputs:
        y: wp.array(dtype=float32),
        x: wp.array(dtype=float32),
    ):
        # Retrieve the thread index and thread-block configuration
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        # Retrieve the matrix block dimensions and size
        n_i = dim[tid]
        L_i_start = mio[tid]
        v_i_start = vio[tid]

        # Retrieve a pointer to the start of the i-th matrix in A
        L_i_ptr = get_float32_array_offset_ptr(L, L_i_start)
        y_i_ptr = get_float32_array_offset_ptr(y, v_i_start)
        x_i_ptr = get_float32_array_offset_ptr(x, v_i_start)

        # Create a temporary warp array pointing to the i-th matrix
        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        y_i = wp.array(ptr=y_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_i = wp.array(ptr=x_i_ptr, shape=(n_i, 1), dtype=wp.float32)

        # Round up n_i to next multiple of block_size
        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L y = b
        for i in range(0, n_i_padded, block_size):
            rhs_tile = wp.tile_load(x_i, shape=(block_size, 1), offset=(i, 0))
            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y_i, shape=(block_size, 1), offset=(j, 0))
                    Ly_block = wp.tile_matmul(L_block, y_block)
                    rhs_tile -= Ly_block
            L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            y_tile = wp.tile_lower_solve(L_tile, rhs_tile)
            wp.tile_store(y_i, y_tile, offset=(i, 0))

        # Backward substitution: solve L^T x = y
        for i in range(n_i_padded - block_size, -1, -block_size):
            i_end = i + block_size
            rhs_tile = wp.tile_load(y_i, shape=(block_size, 1), offset=(i, 0))
            if i_end < n_i_padded:
                for j in range(i_end, n_i_padded, block_size):
                    L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(j, i))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x_i, shape=(block_size, 1), offset=(j, 0))
                    L_T_x_tile = wp.tile_matmul(L_T_tile, x_tile)
                    rhs_tile -= L_T_x_tile
            L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))

            # The following if pads the matrix if it is not divisible by block_size
            if i + block_size > n_i:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    linear_index = tid_block + ii * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = L_tile[row, col]
                    if i + row >= n_i:
                        value = wp.where(i + row == i + col, float32(1), float32(0))
                    L_tile[row, col] = value

            x_tile = wp.tile_upper_solve(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x_i, x_tile, offset=(i, 0))

    # Return the kernel function
    return llt_blocked_solve_inplace_kernel


###
# Launchers
###


def llt_blocked_factorize(
    kernel,
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    A: wp.array(dtype=float32),
    L: wp.array(dtype=float32),
    num_blocks: int = 1,
    block_dim: int = 128,  # TODO: Rename this to be clearer that this is the number of threads per TILE block and not matrix block
    device: Devicelike = None,
):
    """
    Launches the blocked Cholesky factorization kernel for a block partitioned matrix.

    Args:
        kernel: The kernel function to use for the blocked factorization.
        num_blocks (int): The number of matrix blocks to process.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
        dim (wp.array): An array of shape `(num_blocks,)` containing the active dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the matrix index offset (mio) of each matrix block.
        A (wp.array): The flat input array containing the input matrix blocks to be factorized.
        L (wp.array): The flat output array containing the factorization of each matrix block.
    """
    wp.launch_tiled(kernel=kernel, dim=num_blocks, inputs=[dim, mio, A, L], block_dim=block_dim, device=device)


def llt_blocked_solve(
    kernel,
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    b: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    num_blocks: int = 1,
    block_dim: int = 64,
    device: Devicelike = None,
):
    """
    Launches the blocked Cholesky solve kernel for a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the matrix index offsets of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the vector index offsets of each vector block.
        L (wp.array2d): The flat input array containing the Cholesky factorization of each matrix block.
        b (wp.array): The flat input array containing the stacked right-hand side vectors.
        y (wp.array): The output array where the intermediate result will be stored.
        x (wp.array): The output array where the solution to the linear system `A @ x = b` will be stored.
        kernel: The kernel function to use for the blocked solve.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
    """
    wp.launch_tiled(
        kernel=kernel, dim=num_blocks, inputs=[dim, mio, vio, L, b, y, x], block_dim=block_dim, device=device
    )


def llt_blocked_solve_inplace(
    kernel,
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    num_blocks: int = 1,
    block_dim: int = 64,
    device: Devicelike = None,
):
    """
    Launches the blocked Cholesky in-place solve kernel for a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the matrix index offsets of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the vector index offsets of each vector block.
        L (wp.array2d): The flat input array containing the Cholesky factorization of each matrix block.
        x (wp.array): The input/output array where the solution to the linear system `A @ x = b` will be stored in-place.
        kernel: The kernel function to use for the blocked in-place solve.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
    """
    wp.launch_tiled(kernel=kernel, dim=num_blocks, inputs=[dim, mio, vio, L, y, x], block_dim=block_dim, device=device)
