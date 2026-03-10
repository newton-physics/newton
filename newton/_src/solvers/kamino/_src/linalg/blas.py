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

"""BLAS-like operations for multi-linear systems"""

import functools
from typing import Any

import warp as wp

from ..core.types import FloatType, float32, int32
from .sparse_matrix import BlockDType, BlockSparseMatrices

###
# Module interface
###

__all__ = [
    "block_sparse_gemv",
    "block_sparse_matvec",
    "block_sparse_transpose_gemv",
    "block_sparse_transpose_matvec",
    "dense_gemv",
    "diag_gemv",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


##
# Kernels
##


@wp.kernel
def _mult_left_right_diag_matrix_with_matrix(
    # Inputs:
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    D: wp.array(dtype=float32),
    X: wp.array(dtype=float32),
    # Outputs:
    Y: wp.array(dtype=float32),
):
    # Retrieve the thread indices
    wid, tid = wp.tid()

    # Retrieve the number of active dimensions in the world
    n = dim[wid]

    # Compute i (row) and j (col) indices from the tid
    i = tid // n
    j = tid % n

    # Skip if indices exceed the problem size
    if i >= n or j >= n:
        return

    # Retrieve the matrix index offset of the world
    m_0 = mio[wid]

    # Retrieve the vector index offset of the world
    v_0 = vio[wid]

    # Compute the global index of the matrix entry
    m_ij = m_0 + n * i + j

    # Retrieve the ij entry of the input matrix
    X_ij = X[m_ij]

    # Retrieve the i,j entries of the diagonal matrix
    D_i = D[v_0 + i]
    D_j = D[v_0 + j]

    # Compute the i,j entry of the output matrix
    Y[m_ij] = D_i * D_j * X_ij


@wp.kernel
def _mult_left_diag_matrix_with_vector(
    # Inputs:
    dim: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    D: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    # Outputs:
    y: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    n = dim[wid]

    # Skip if row index exceed the problem size
    if tid >= n:
        return

    # Retrieve the vector index offset of the world
    v_0 = vio[wid]

    # Compute the global index of the vector entry
    v_i = v_0 + tid

    # Retrieve the i-th entry of the input vector
    x_i = x[v_i]

    # Retrieve the i-th entry of the diagonal matrix
    D_i = D[v_i]

    # Compute the i-th entry of the output vector
    y[v_i] = D_i * x_i


@functools.cache
def _make_block_sparse_matvec_kernel(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_matvec_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Vector:
        x: wp.array(dtype=block_type.dtype),
        y: wp.array(dtype=block_type.dtype),
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: y_block += A_block @ x_block
        if wp.static(n_block_rows == 1):
            x_idx_base = col_start[mat_id] + block_coord[1]
            acc = block_type.dtype(0.0)

            for j in range(n_block_cols):
                acc += block[j] * x[x_idx_base + j]

            wp.atomic_add(y, row_start[mat_id] + block_coord[0], acc)

        else:
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_idx_base = row_start[mat_id] + block_coord[0]

            for i in range(n_block_rows):
                acc = block_type.dtype(0.0)

                for j in range(n_block_cols):
                    acc += block[i, j] * x[x_idx_base + j]

                wp.atomic_add(y, y_idx_base + i, acc)

    return block_sparse_matvec_kernel


@functools.cache
def _make_block_sparse_matvec_kernel_2d(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_matvec_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector:
        x: wp.array2d(dtype=block_type.dtype),
        y: wp.array2d(dtype=block_type.dtype),
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: y_block += A_block @ x_block
        if wp.static(n_block_rows == 1):
            x_idx_base = block_coord[1]
            acc = block_type.dtype(0.0)

            for j in range(n_block_cols):
                acc += block[j] * x[mat_id, x_idx_base + j]

            wp.atomic_add(y, mat_id, block_coord[0], acc)

        else:
            x_idx_base = block_coord[1]
            y_idx_base = block_coord[0]

            for i in range(n_block_rows):
                acc = block_type.dtype(0.0)

                for j in range(n_block_cols):
                    acc += block[i, j] * x[mat_id, x_idx_base + j]

                wp.atomic_add(y, mat_id, y_idx_base + i, acc)

    return block_sparse_matvec_kernel


@functools.cache
def _make_block_sparse_transpose_matvec_kernel(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_transpose_matvec_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Vector:
        y: wp.array(dtype=block_type.dtype),
        x: wp.array(dtype=block_type.dtype),
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: x_block += A_block^T @ y_block
        if wp.static(n_block_rows == 1):
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_val = y[row_start[mat_id] + block_coord[0]]

            for i in range(n_block_cols):
                wp.atomic_add(x, x_idx_base + i, block[i] * y_val)

        else:
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_idx_base = row_start[mat_id] + block_coord[0]

            for i in range(n_block_cols):
                acc = block_type.dtype(0.0)

                for j in range(n_block_rows):
                    acc += block[j, i] * y[y_idx_base + j]

                wp.atomic_add(x, x_idx_base + i, acc)

    return block_sparse_transpose_matvec_kernel


@functools.cache
def _make_block_sparse_transpose_matvec_kernel_2d(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_transpose_matvec_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector:
        y: wp.array2d(dtype=block_type.dtype),
        x: wp.array2d(dtype=block_type.dtype),
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: x_block += A_block^T @ y_block
        if wp.static(n_block_rows == 1):
            x_idx_base = block_coord[1]
            y_val = y[mat_id, block_coord[0]]

            for i in range(n_block_cols):
                wp.atomic_add(x, mat_id, x_idx_base + i, block[i] * y_val)

        else:
            x_idx_base = block_coord[1]
            y_idx_base = block_coord[0]

            for i in range(n_block_cols):
                acc = block_type.dtype(0.0)

                for j in range(n_block_rows):
                    acc += block[j, i] * y[mat_id, y_idx_base + j]

                wp.atomic_add(x, mat_id, x_idx_base + i, acc)

    return block_sparse_transpose_matvec_kernel


@functools.cache
def _make_scale_vector_kernel(space_dim: int):
    """Creates a kernel that scales a vector, taking into account a matrix mask and how the current
    size of a matrix affects the active entries of the vector.

    Parameters
    ----------
    space_dim : int
        Space of the vector in reference to the matrices (0: row space, 1: column space).
    """

    sp_dim = wp.constant(space_dim)

    @wp.kernel
    def scale_vector_kernel(
        # Matrix data:
        matrix_dims: wp.array2d(dtype=int32),
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Inputs:
        x: wp.array(dtype=Any),
        beta: Any,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, entry_id = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0 or entry_id >= matrix_dims[mat_id, sp_dim]:
            return

        if wp.static(space_dim == 0):
            idx = row_start[mat_id] + entry_id
            x[idx] = beta * x[idx]
        else:
            idx = col_start[mat_id] + entry_id
            x[idx] = beta * x[idx]

    return scale_vector_kernel


@functools.cache
def _make_scale_vector_kernel_2d(space_dim: int):
    """Creates a kernel that scales a vector, taking into account a matrix mask and how the current
    size of a matrix affects the active entries of the vector.

    Parameters
    ----------
    space_dim : int
        Space of the vector in reference to the matrices (0: row space, 1: column space).
    """

    sp_dim = wp.constant(space_dim)

    @wp.kernel
    def scale_vector_kernel(
        # Matrix data:
        matrix_dims: wp.array2d(dtype=int32),
        # Inputs:
        x: wp.array2d(dtype=Any),
        beta: Any,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, entry_id = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0 or entry_id >= matrix_dims[mat_id, sp_dim]:
            return

        x[mat_id, entry_id] = beta * x[mat_id, entry_id]

    return scale_vector_kernel


@functools.cache
def _make_block_sparse_gemv_kernel(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_gemv_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Vector:
        x: wp.array(dtype=block_type.dtype),
        y: wp.array(dtype=block_type.dtype),
        # Scaling:
        alpha: block_type.dtype,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: z += alpha * A_block @ x_block
        if wp.static(n_block_rows == 1):
            x_idx_base = col_start[mat_id] + block_coord[1]
            acc = block_type.dtype(0.0)

            for j in range(n_block_cols):
                acc += alpha * block[j] * x[x_idx_base + j]

            wp.atomic_add(y, row_start[mat_id] + block_coord[0], acc)

        else:
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_idx_base = row_start[mat_id] + block_coord[0]

            for i in range(n_block_rows):
                acc = block_type.dtype(0.0)

                for j in range(n_block_cols):
                    acc += alpha * block[i, j] * x[x_idx_base + j]

                wp.atomic_add(y, y_idx_base + i, acc)

    return block_sparse_gemv_kernel


@functools.cache
def _make_block_sparse_gemv_kernel_2d(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_gemv_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector:
        x: wp.array2d(dtype=block_type.dtype),
        y: wp.array2d(dtype=block_type.dtype),
        # Scaling:
        alpha: block_type.dtype,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: z += alpha * A_block @ x_block
        if wp.static(n_block_rows == 1):
            x_idx_base = block_coord[1]
            acc = block_type.dtype(0.0)

            for j in range(n_block_cols):
                acc += alpha * block[j] * x[mat_id, x_idx_base + j]

            wp.atomic_add(y, mat_id, block_coord[0], acc)

        else:
            x_idx_base = block_coord[1]
            y_idx_base = block_coord[0]

            for i in range(n_block_rows):
                acc = block_type.dtype(0.0)

                for j in range(n_block_cols):
                    acc += alpha * block[i, j] * x[mat_id, x_idx_base + j]

                wp.atomic_add(y, mat_id, y_idx_base + i, acc)

    return block_sparse_gemv_kernel


@functools.cache
def _make_block_sparse_transpose_gemv_kernel(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_transpose_gemv_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Vector:
        y: wp.array(dtype=block_type.dtype),
        x: wp.array(dtype=block_type.dtype),
        # Scaling:
        alpha: block_type.dtype,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: z += alpha * A_block^T @ y_block
        if wp.static(n_block_rows == 1):
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_val = y[row_start[mat_id] + block_coord[0]]

            for i in range(n_block_cols):
                wp.atomic_add(x, x_idx_base + i, alpha * block[i] * y_val)

        else:
            x_idx_base = col_start[mat_id] + block_coord[1]
            y_idx_base = row_start[mat_id] + block_coord[0]

            for i in range(n_block_cols):
                acc = block_type.dtype(0.0)

                for j in range(n_block_rows):
                    acc += alpha * block[j, i] * y[y_idx_base + j]

                wp.atomic_add(x, x_idx_base + i, acc)

    return block_sparse_transpose_gemv_kernel


@functools.cache
def _make_block_sparse_transpose_gemv_kernel_2d(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_transpose_gemv_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Vector:
        y: wp.array2d(dtype=block_type.dtype),
        x: wp.array2d(dtype=block_type.dtype),
        # Scaling:
        alpha: block_type.dtype,
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: z += alpha * A_block^T @ y_block
        if wp.static(n_block_rows == 1):
            x_idx_base = block_coord[1]
            y_val = y[mat_id, block_coord[0]]

            for i in range(n_block_cols):
                wp.atomic_add(x, mat_id, x_idx_base + i, alpha * block[i] * y_val)

        else:
            x_idx_base = block_coord[1]
            y_idx_base = block_coord[0]

            for i in range(n_block_cols):
                acc = block_type.dtype(0.0)

                for j in range(n_block_rows):
                    acc += alpha * block[j, i] * y[mat_id, y_idx_base + j]

                wp.atomic_add(x, mat_id, x_idx_base + i, acc)

    return block_sparse_transpose_gemv_kernel


@wp.kernel
def _diag_gemv_kernel(
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    D: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    world_active: wp.array(dtype=wp.int32),
    alpha: Any,
    beta: Any,
):
    """Computes y[w] = alpha * D[w] * x[w] + beta * y[w] for each world w."""
    world, row = wp.tid()
    assert world < len(active_dims)
    if world_active[world] == 0 or row >= active_dims[world]:
        return

    zero = type(alpha)(0)
    s = y.dtype(0)

    if alpha != zero:
        s += alpha * D[world, row] * x[world, row]
    if beta != zero:
        s += beta * y[world, row]
    y[world, row] = s


@wp.kernel
def _dense_gemv_kernel(
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    A: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    world_active: wp.array(dtype=wp.int32),
    alpha: Any,
    beta: Any,
    matrix_stride: int,
    tile_size: int,
):
    """Computes y[w] = alpha * (A[w] @ x[w]) + beta * y[w] in-place for each world w."""
    world, row, lane = wp.tid()
    assert world < len(active_dims)
    dim = active_dims[world]
    if world_active[world] == 0 or row >= dim:
        return

    row_stride = active_dims[world]
    zero = type(alpha)(0)
    s = zero
    if alpha != zero:
        for col in range(lane, dim, tile_size):
            s += A[world, row * row_stride + col] * x[world, col]
    row_tile = wp.tile_sum(wp.tile(s * alpha))
    if beta != zero:
        row_tile += beta * wp.tile_load(y[world], shape=1, offset=row)
    wp.tile_store(y[world], row_tile, offset=row)


@functools.cache
def _make_block_sparse_ATA_diagonal_kernel_2d(block_type: BlockDType):
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def block_sparse_ATA_diagonal_kernel(
        # Matrix data:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=block_type.warp_type),
        # Output:
        diag: wp.array2d(dtype=block_type.dtype),
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        """
        For a block sparse matrix (stack) A, computes the diagonal of A^T * A
        """
        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_col = nzb_coords[global_block_idx][1]
        block = nzb_values[global_block_idx]

        # Accumulate coefficients contributed by non-zero block
        if wp.static(n_block_rows == 1):
            for j in range(n_block_cols):
                val = block[j]
                wp.atomic_add(diag, mat_id, block_col + j, val * val)
        else:
            for j in range(n_block_cols):
                acc = block_type.dtype(0.0)
                for i in range(n_block_rows):
                    val = block[i, j]
                    acc += val * val
                wp.atomic_add(diag, mat_id, block_col + j, acc)

    return block_sparse_ATA_diagonal_kernel


class nzb_type_7(BlockDType(dtype=wp.float32, shape=(7,)).warp_type):
    pass


@wp.kernel
def block_sparse_ATA_diagonal_3_4_blocks_kernel_2d(
    # Matrix data:
    num_nzb: wp.array(dtype=int32),
    nzb_start: wp.array(dtype=int32),
    nzb_coords: wp.array2d(dtype=int32),
    nzb_values: wp.array(dtype=nzb_type_7),
    # Output:
    blocks_3: wp.array2d(dtype=wp.float32),
    blocks_4: wp.array2d(dtype=wp.float32),
    # Mask:
    matrix_mask: wp.array(dtype=int32),
):
    """
    For a block sparse matrix (stack) A with 1x7 blocks, computes the blockwise-diagonal of A^T * A,
    with alternating 3x3 and 4x4 blocks
    3x3 and 4x4 blocks are flattened and concatenated in blocks_3 and blocks_4 (to allow atomic_add)
    """
    mat_id, block_idx = wp.tid()

    # Early exit if the matrix is flagged as inactive.
    if matrix_mask[mat_id] == 0:
        return

    # Check if block index is valid for this matrix.
    if block_idx >= num_nzb[mat_id]:
        return

    global_block_idx = nzb_start[mat_id] + block_idx
    block_col = nzb_coords[global_block_idx][1]
    block = nzb_values[global_block_idx]
    block_col_7 = block_col // 7

    # Accumulate coefficients contributed to 3x3 block
    offset = 9 * block_col_7
    for i in range(3):
        val_i = block[i]
        for j in range(3):
            val_j = block[j]
            wp.atomic_add(blocks_3, mat_id, offset + 3 * i + j, val_i * val_j)

    # Accumulate coefficients contributed to 4x4 block
    offset = 16 * block_col_7
    for i in range(4):
        val_i = block[3 + i]
        for j in range(4):
            val_j = block[3 + j]
            wp.atomic_add(blocks_4, mat_id, offset + 4 * i + j, val_i * val_j)


@functools.cache
def _make_cwise_inverse_kernel_2d(dtype: FloatType):
    @wp.kernel
    def cwise_inverse_kernel(
        # Inputs
        x: wp.array2d(dtype=dtype),
        dim: wp.array(dtype=wp.int32),
        mask: wp.array(dtype=wp.int32),
    ):
        mat_id, coeff_id = wp.tid()

        if mat_id >= mask.shape[0] or mask[mat_id] == 0 or coeff_id >= dim[mat_id]:
            return

        x[mat_id, coeff_id] = 1.0 / x[mat_id, coeff_id]

    return cwise_inverse_kernel


@wp.kernel
def blockwise_inverse_kernel_3_2d(
    # Inputs
    blocks: wp.array2d(dtype=wp.mat33f),
    dim: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.int32),
):
    mat_id, block_id = wp.tid()

    if mat_id >= mask.shape[0] or mask[mat_id] == 0 or 7 * block_id >= dim[mat_id]:
        return

    blocks[mat_id, block_id] = wp.inverse(blocks[mat_id, block_id])


@wp.kernel
def blockwise_inverse_kernel_4_2d(
    # Inputs
    blocks: wp.array2d(dtype=wp.mat44f),
    dim: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.int32),
):
    mat_id, block_id = wp.tid()

    if mat_id >= mask.shape[0] or mask[mat_id] == 0 or 7 * block_id >= dim[mat_id]:
        return

    blocks[mat_id, block_id] = wp.inverse(blocks[mat_id, block_id])


@wp.kernel
def _blockwise_diag_3_4_gemv_kernel_2d(
    x: wp.array2d(dtype=wp.float32),
    y: wp.array2d(dtype=wp.float32),
    blocks_3: wp.array2d(dtype=wp.mat33f),
    blocks_4: wp.array2d(dtype=wp.mat44f),
    active_dims: wp.array(dtype=wp.int32),
    world_active: wp.array(dtype=wp.int32),
    alpha: wp.float32,
    beta: wp.float32,
):
    """Computes y[w] = alpha * D[w] * x[w] + beta * y[w] for each world w.
    where D is blockwise-diagonal, alternating 3x3 and 4x4 blocks"""
    world, row_block_id = wp.tid()
    row_id = 7 * row_block_id
    assert world < len(active_dims)
    if world_active[world] == 0 or row_id >= active_dims[world]:
        return

    zero = type(alpha)(0)
    y_3 = wp.vec3f(0.0, 0.0, 0.0)
    y_4 = wp.vec4f(0.0, 0.0, 0.0, 0.0)

    if alpha != zero:
        x_3 = wp.vec3f(x[world, row_id], x[world, row_id + 1], x[world, row_id + 2])
        y_3 += alpha * (blocks_3[world, row_block_id] * x_3)
        x_4 = wp.vec4f(x[world, row_id + 3], x[world, row_id + 4], x[world, row_id + 5], x[world, row_id + 6])
        y_4 = alpha * (blocks_4[world, row_block_id] * x_4)
    if beta != zero:
        y_3 += beta * wp.vec3f(y[world, row_id], y[world, row_id + 1], y[world, row_id + 2])
        y_4 += beta * wp.vec4f(y[world, row_id + 3], y[world, row_id + 4], y[world, row_id + 5], y[world, row_id + 6])

    y[world, row_id] = y_3[0]
    y[world, row_id + 1] = y_3[1]
    y[world, row_id + 2] = y_3[2]
    y[world, row_id + 3] = y_4[0]
    y[world, row_id + 4] = y_4[1]
    y[world, row_id + 5] = y_4[2]
    y[world, row_id + 6] = y_4[3]


##
# Launchers
##


def diag_gemv(
    D: wp.array2d,
    x: wp.array2d,
    y: wp.array2d,
    active_dims: wp.array,
    world_active: wp.array,
    alpha: float,
    beta: float,
):
    """
    Launch kernel for diagonal matrix gemv: y = alpha * D * x + beta * y

    Args:
        D: Diagonal matrices stored as 2D array (n_worlds, max_dim).
        x: Input vectors (n_worlds, max_dim).
        y: Output vectors (n_worlds, max_dim), modified in-place.
        active_dims: Active dimension per world.
        world_active: Boolean mask for active worlds.
        alpha: Scalar multiplier for D * x.
        beta: Scalar multiplier for y.
    """
    n_worlds, max_dim = x.shape
    dtype = x.dtype
    wp.launch(
        _diag_gemv_kernel,
        dim=(n_worlds, max_dim),
        inputs=[x, y, D, active_dims, world_active, dtype(alpha), dtype(beta)],
        device=x.device,
    )


def dense_gemv(
    A: wp.array2d,
    x: wp.array2d,
    y: wp.array2d,
    active_dims: wp.array,
    world_active: wp.array,
    alpha: float,
    beta: float,
    matrix_stride: int,
    block_dim: int = 64,
):
    """
    Launch kernel for dense matrix gemv: y = alpha * A @ x + beta * y

    Args:
        A: Dense matrices stored as 2D array (n_worlds, max_dim * max_dim).
        x: Input vectors (n_worlds, max_dim).
        y: Output vectors (n_worlds, max_dim), modified in-place.
        active_dims: Active dimension per world.
        world_active: Boolean mask for active worlds.
        alpha: Scalar multiplier for A * x.
        beta: Scalar multiplier for y.
        matrix_stride: Stride for matrix row indexing.
        block_dim: Block dimension for tiled computation.
    """
    n_worlds, max_dim = x.shape
    dtype = x.dtype
    if not x.device.is_cuda:
        block_dim = 1
    wp.launch(
        _dense_gemv_kernel,
        dim=(n_worlds, max_dim, block_dim),
        inputs=[x, y, A, active_dims, world_active, dtype(alpha), dtype(beta), matrix_stride, block_dim],
        device=x.device,
        block_dim=block_dim,
    )


def block_sparse_matvec(
    A: BlockSparseMatrices,
    x: wp.array,
    y: wp.array,
    matrix_mask: wp.array,
):
    """
    Launch kernel for block-sparse matrix-vector product: y = A * x

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        x (wp.array): Stack of input vectors, expects either shape (sum_of_max_cols,) for the 1D flattened
        version; or shape (num_matrices, max_of_max_cols) for the 2D version.
        y (wp.array): Stack of output vectors, expects either shape (sum_of_max_rows,) for the 1D flattened
        version; or shape (num_matrices, max_of_max_rows) for the 2D version.
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    y.zero_()

    if len(x.shape) == 1:
        wp.launch(
            kernel=_make_block_sparse_matvec_kernel(A.nzb_dtype),
            dim=(A.num_matrices, A.max_of_num_nzb),
            inputs=[
                A.num_nzb,
                A.nzb_start,
                A.nzb_coords,
                A.nzb_values,
                A.row_start,
                A.col_start,
                x,
                y,
                matrix_mask,
            ],
            device=A.device,
        )
    else:
        wp.launch(
            kernel=_make_block_sparse_matvec_kernel_2d(A.nzb_dtype),
            dim=(A.num_matrices, A.max_of_num_nzb),
            inputs=[
                A.num_nzb,
                A.nzb_start,
                A.nzb_coords,
                A.nzb_values,
                x,
                y,
                matrix_mask,
            ],
            device=A.device,
        )


def block_sparse_transpose_matvec(
    A: BlockSparseMatrices,
    y: wp.array,
    x: wp.array,
    matrix_mask: wp.array,
):
    """
    Launch kernel for block-sparse transpose matrix-vector product: x = A^T * y

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        y (wp.array): Stack of input vectors, expects either shape (sum_of_max_rows,) for the 1D flattened
        version; or shape (num_matrices, max_of_max_rows) for the 2D version.
        x (wp.array): Stack of output vectors, expects either shape (sum_of_max_cols,) for the 1D flattened
        version; or shape (num_matrices, max_of_max_cols) for the 2D version.
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    x.zero_()

    if len(x.shape) == 1:
        wp.launch(
            kernel=_make_block_sparse_transpose_matvec_kernel(A.nzb_dtype),
            dim=(A.num_matrices, A.max_of_num_nzb),
            inputs=[
                A.num_nzb,
                A.nzb_start,
                A.nzb_coords,
                A.nzb_values,
                A.row_start,
                A.col_start,
                y,
                x,
                matrix_mask,
            ],
            device=A.device,
        )
    else:
        wp.launch(
            kernel=_make_block_sparse_transpose_matvec_kernel_2d(A.nzb_dtype),
            dim=(A.num_matrices, A.max_of_num_nzb),
            inputs=[
                A.num_nzb,
                A.nzb_start,
                A.nzb_coords,
                A.nzb_values,
                y,
                x,
                matrix_mask,
            ],
            device=A.device,
        )


def block_sparse_gemv(
    A: BlockSparseMatrices,
    x: wp.array,
    y: wp.array,
    alpha: Any,
    beta: Any,
    matrix_mask: wp.array,
):
    """
    Launch kernel for generalized block-sparse matrix-vector product: y = alpha * (A * x) + beta * y

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        x (wp.array): Stack of input vectors, expects either shape (sum_of_max_cols,) for the 1D flattened
        version; or shape (num_matrices, max_of_max_cols) for the 2D version.
        y (wp.array): Stack of input-output vectors, expects either shape (sum_of_max_rows,) for the 1D
        flattened version; or shape (num_matrices, max_of_max_rows) for the 2D version.
        alpha (Any): Input scaling for matrix-vector multiplication.
        beta (Any): Input scaling for linear offset.
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    if len(x.shape) == 1:
        # Compute y <= beta * y
        wp.launch(
            kernel=_make_scale_vector_kernel(0),
            dim=(A.num_matrices, A.max_of_max_dims[0]),
            inputs=[A.dims, A.row_start, A.col_start, y, beta, matrix_mask],
            device=A.device,
        )

        # Compute y += alpha * A @ x
        wp.launch(
            kernel=_make_block_sparse_gemv_kernel(A.nzb_dtype),
            dim=(A.num_matrices, A.max_of_num_nzb),
            inputs=[
                A.num_nzb,
                A.nzb_start,
                A.nzb_coords,
                A.nzb_values,
                A.row_start,
                A.col_start,
                x,
                y,
                alpha,
                matrix_mask,
            ],
            device=A.device,
        )
    else:
        # Compute y <= beta * y
        wp.launch(
            kernel=_make_scale_vector_kernel_2d(0),
            dim=(A.num_matrices, A.max_of_max_dims[0]),
            inputs=[A.dims, A.row_start, A.col_start, y, beta, matrix_mask],
            device=A.device,
        )

        # Compute y += alpha * A @ x
        wp.launch(
            kernel=_make_block_sparse_gemv_kernel_2d(A.nzb_dtype),
            dim=(A.num_matrices, A.max_of_num_nzb),
            inputs=[
                A.num_nzb,
                A.nzb_start,
                A.nzb_coords,
                A.nzb_values,
                x,
                y,
                alpha,
                matrix_mask,
            ],
            device=A.device,
        )


def block_sparse_transpose_gemv(
    A: BlockSparseMatrices,
    y: wp.array,
    x: wp.array,
    alpha: Any,
    beta: Any,
    matrix_mask: wp.array,
):
    """
    Launch kernel for generalized block-sparse transpose matrix-vector product: x = alpha * (A^T * y) + beta * x

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        y (wp.array): Stack of input vectors, expects either shape (sum_of_max_rows,) for the 1D flattened
        version; or shape (num_matrices, max_of_max_rows) for the 2D version.
        x (wp.array): Stack of input-output vectors, expects either shape (sum_of_max_cols,) for the 1D
        flattened version; or shape (num_matrices, max_of_max_cols) for the 2D version.
        alpha (Any): Input scaling for matrix-vector multiplication.
        beta (Any): Input scaling for linear offset.
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    if len(x.shape) == 1:
        # Compute x <= beta * x
        wp.launch(
            kernel=_make_scale_vector_kernel(1),
            dim=(A.num_matrices, A.max_of_max_dims[1]),
            inputs=[A.dims, A.row_start, A.col_start, x, beta, matrix_mask],
            device=A.device,
        )

        # Compute y += alpha * A^T @ y
        wp.launch(
            kernel=_make_block_sparse_transpose_gemv_kernel(A.nzb_dtype),
            dim=(A.num_matrices, A.max_of_num_nzb),
            inputs=[
                A.num_nzb,
                A.nzb_start,
                A.nzb_coords,
                A.nzb_values,
                A.row_start,
                A.col_start,
                y,
                x,
                alpha,
                matrix_mask,
            ],
            device=A.device,
        )
    else:
        # Compute x <= beta * x
        wp.launch(
            kernel=_make_scale_vector_kernel(1),
            dim=(A.num_matrices, A.max_of_max_dims[1]),
            inputs=[A.dims, A.row_start, A.col_start, x, beta, matrix_mask],
            device=A.device,
        )

        # Compute y += alpha * A^T @ y
        wp.launch(
            kernel=_make_block_sparse_transpose_gemv_kernel(A.nzb_dtype),
            dim=(A.num_matrices, A.max_of_num_nzb),
            inputs=[
                A.num_nzb,
                A.nzb_start,
                A.nzb_coords,
                A.nzb_values,
                y,
                x,
                alpha,
                matrix_mask,
            ],
            device=A.device,
        )


def block_sparse_ATA_inv_diagonal_2d(A: BlockSparseMatrices, inv_diag: wp.array, matrix_mask: wp.array):
    """
    Function computing the inverse of the diagonal of A^T * A given sparse matrix (stack) A.

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        inv_diag (wp.array): Stack of output vectors, expects shape (num_matrices, max_of_max_cols).
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    inv_diag.zero_()
    wp.launch(
        kernel=_make_block_sparse_ATA_diagonal_kernel_2d(A.nzb_dtype),
        dim=(A.num_matrices, A.max_of_num_nzb),
        inputs=[
            A.num_nzb,
            A.nzb_start,
            A.nzb_coords,
            A.nzb_values,
            inv_diag,
            matrix_mask,
        ],
        device=A.device,
    )
    int_size_bytes = 4  # Size of wp.int32 in bytes
    cols = wp.array(
        dtype=wp.int32,
        shape=(A.num_matrices,),
        ptr=A.dims.ptr + int_size_bytes,
        strides=(2 * int_size_bytes,),
        copy=False,
    )
    wp.launch(
        kernel=_make_cwise_inverse_kernel_2d(A.nzb_dtype.dtype),
        dim=(A.num_matrices, A.max_of_max_dims[1]),
        inputs=[
            inv_diag,
            cols,
            matrix_mask,
        ],
        device=A.device,
    )


def block_sparse_ATA_blockwise_3_4_inv_diagonal_2d(
    A: BlockSparseMatrices, inv_blocks_3: wp.array, inv_blocks_4: wp.array, matrix_mask: wp.array
):
    """
    Function computing the blockwise inverse of the diagonal of A^T * A given sparse matrix (stack) A,
    with alternating 3x3 and 4x4 blocks
    A must have block size 1x7

    Args:
        A (BlockSparseMatrices): Sparse matrices.
        inv_blocks (wp.array): Stack of vectors of 3x3 blocks, expects shape (num_matrices, max_of_max_cols / 7).
        matrix_mask (wp.array): Mask vector to skip matrices set to `0` in the mask.
    """
    inv_blocks_3.zero_()
    inv_blocks_4.zero_()
    inv_blocks_3_flat = wp.array(
        dtype=wp.float32,
        ptr=inv_blocks_3.ptr,
        shape=(A.num_matrices, 9 * inv_blocks_3.shape[1]),
        copy=False,
        device=A.device,
    )
    inv_blocks_4_flat = wp.array(
        dtype=wp.float32,
        ptr=inv_blocks_4.ptr,
        shape=(A.num_matrices, 16 * inv_blocks_3.shape[1]),
        copy=False,
        device=A.device,
    )
    wp.launch(
        kernel=block_sparse_ATA_diagonal_3_4_blocks_kernel_2d,
        dim=(A.num_matrices, A.max_of_num_nzb),
        inputs=[
            A.num_nzb,
            A.nzb_start,
            A.nzb_coords,
            A.nzb_values,
            inv_blocks_3_flat,
            inv_blocks_4_flat,
            matrix_mask,
        ],
        device=A.device,
    )
    int_size_bytes = 4  # Size of wp.int32 in bytes
    cols = wp.array(
        dtype=wp.int32,
        shape=(A.num_matrices,),
        ptr=A.dims.ptr + int_size_bytes,
        strides=(2 * int_size_bytes,),
        copy=False,
    )
    wp.launch(
        kernel=blockwise_inverse_kernel_3_2d,
        dim=inv_blocks_3.shape,
        inputs=[
            inv_blocks_3,
            cols,
            matrix_mask,
        ],
        device=A.device,
    )
    wp.launch(
        kernel=blockwise_inverse_kernel_4_2d,
        dim=inv_blocks_4.shape,
        inputs=[
            inv_blocks_4,
            cols,
            matrix_mask,
        ],
        device=A.device,
    )


def get_blockwise_diag_3_4_gemv_2d(
    blocks_3: wp.array2d(dtype=wp.mat33f),
    blocks_4: wp.array2d(dtype=wp.mat44f),
    active_dims: wp.array(dtype=wp.int32),
):
    def gemv(
        x: wp.array2d(dtype=wp.float32),
        y: wp.array2d(dtype=wp.float32),
        world_active: wp.array(dtype=wp.int32),
        alpha: wp.float32,
        beta: wp.float32,
    ):
        wp.launch(
            _blockwise_diag_3_4_gemv_kernel_2d,
            dim=blocks_3.shape,
            inputs=[
                x,
                y,
                blocks_3,
                blocks_4,
                active_dims,
                world_active,
                alpha,
                beta,
            ],
            device=blocks_3.device,
        )

    return gemv
