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

"""
KAMINO: Linear Algebra: Core types and utilities for sparse multi-world linear systems

This module provides data structures and utilities for managing multiple
independent linear systems, including rectangular and square systems.
"""

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import warp as wp
from warp.types import type_size_in_bytes

from ..core.types import FloatType, IntType, float32, int32
from .core import DenseSquareMultiLinearInfo

if TYPE_CHECKING:
    from .core import DenseLinearOperatorData

###
# Module interface
###

__all__ = [
    "BlockDType",
    "BlockSparseMatrices",
    "allocate_block_sparse_from_dense",
    "dense_to_block_sparse_copy_values",
]


###
# Types
###


class BlockDType:
    """A utility type for bundling meta-data about sparse-block types."""

    def __init__(self, dtype: FloatType | IntType, shape: int | tuple[int] | tuple[int, int] | None = None):
        """
        Constructs a new BlockDType descriptor given scalar Warp data-type and the block shape.

        Args:
            dtype (FloatType | IntType):
                The underlying scalar Warp data-type of each sparse block.
            shape (int | tuple[int] | tuple[int, int], optional):
                The shape of each sparse block as an integer (for vectors) or a tuple of integers (for matrices).\n
                If not provided, defaults to scalar blocks.

        Raises:
            TypeError:
                If the `dtype` field is not a valid FloatType or IntType.
            ValueError:
                If the `shape` field is not a positive integer or a tuple of one or two positive integers.
        """
        # Ensure the underlying scalar dtype is valid
        if (dtype not in get_args(FloatType)) and (dtype not in get_args(IntType)):
            raise TypeError("The `dtype` field must be a valid FloatType or IntType such as float32 or int32.")

        # If no shape is provided, default to scalar blocks
        if shape is None:
            shape = ()
        # Otherwise, ensure the shape is valid
        elif isinstance(shape, int):
            if shape <= 0:
                raise ValueError("The `shape` field must be a positive integer.")
            elif shape == 1:
                shape = ()
            else:
                shape = (shape,)
        elif isinstance(shape, tuple):
            if len(shape) > 2:
                raise ValueError(
                    "The `shape` field must be an int or a tuple of one "
                    "or two ints to indicate a vector or matrix block."
                )
            for dim in shape:
                if dim <= 0:
                    raise ValueError("All dimensions in the `shape` field must be positive integers.")
        else:
            raise TypeError("The `shape` field must be an int or a tuple of ints.")

        self._dtype: FloatType | IntType = dtype
        """The underlying data type of the sparse blocks."""

        self._shape: int | tuple[int] | tuple[int, int] = shape
        """The shape of each sparse block."""

    @property
    def dtype(self) -> FloatType | IntType:
        """Returns the underlying data type of the sparse blocks."""
        return self._dtype

    @property
    def shape(self) -> int | tuple[int] | tuple[int, int]:
        """Returns the shape of each sparse block."""
        return self._shape

    @property
    def size(self) -> int:
        """Returns the number of elements contained in each sparse block."""
        if isinstance(self._shape, int):
            return self._shape
        elif isinstance(self._shape, tuple):
            size = 1
            for dim in self._shape:
                size *= dim
            return size
        raise RuntimeError("Unsupported block shape for size computation.")

    @property
    def warp_type(self) -> Any:
        """Returns the corresponding Warp type for this block-sparse type."""
        if self._dtype is None or self._shape is None:
            raise RuntimeError("Both `dtype` and `shape` fields must be specified to get the Warp type.")
        if not isinstance(self._shape, tuple):
            raise RuntimeError(f"Block shape should be a tuple but got {type(self._shape)}.")

        if len(self._shape) == 0:
            return self._dtype
        elif len(self._shape) == 1:

            class _vec_t(wp.types.vector(length=self._shape[0], dtype=self._dtype)):
                pass

            return _vec_t
        elif len(self._shape) == 2:

            class _mat_t(wp.types.matrix(shape=self._shape, dtype=self._dtype)):
                pass

            return _mat_t
        else:
            raise RuntimeError(f"Cannot convert to Warp type: Block shape is invalid: {self._shape}.")


@dataclass
class BlockSparseMatrices:
    """
    A container to represent multiple block-sparse matrices of fixed non-zero block size.
    """

    ###
    # Host-side Metadata
    ###

    device: wp.DeviceLike | None = None
    """Host-side cache of the device on which all data arrays are allocated."""

    nzb_dtype: BlockDType | None = None
    """Host-side cache of the fixed non-zero block data type contained in all sparse matrices."""

    index_dtype: IntType = int32
    """Host-side cache of the integer type used for indexing the underlying data arrays."""

    num_matrices: int = 0
    """
    Host-side cache of the number of sparse matrices represented by this container.\n
    When constructing the BSM via `finalize()`, this is inferred from the length of the provided capacities list.\n
    Alternatively, it can be set directly if the BSM is constructed explicitly.
    """

    sum_of_num_nzb: int = 0
    """
    Host-side cache of the sum of the number of non-zero blocks over all sparse matrices.\n
    When constructing the BSM via `finalize()`, this is computed from the provided capacities list.
    Alternatively, it can be set directly if the BSM is constructed explicitly.
    """

    max_of_num_nzb: int = 0
    """
    Host-side cache of the maximum number of non-zero blocks over all sparse matrices.\n
    When constructing the BSM via `finalize()`, this is computed from the provided capacities list.
    Alternatively, it can be set directly if the BSM is constructed explicitly.
    """

    max_of_max_dims: tuple[int, int] = (0, 0)
    """
    Host-side cache of the maximum of the maximum matrix dimensions over all sparse matrices.
    """

    ###
    # On-device Data (Constant)
    ###

    # These arrays are expected to stay constant once this object is finalized

    max_dims: wp.array | None = None
    """
    The maximum dimensions of each sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`vec2i`.
    """

    max_nzb: wp.array | None = None
    """
    The maximum number of non-zero blocks per sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`int`.
    """

    nzb_start: wp.array | None = None
    """
    The index of the first non-zero block of each sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`int`.
    """

    row_start: wp.array | None = None
    """
    The start index of each row vector block in a flattened data array of size sum_of_max_rows.\n
    Shape of ``(num_matrices,)`` and type :class:`int`.
    """

    col_start: wp.array | None = None
    """
    The start index of each column vector block in a flattened data array of size sum_of_max_cols.\n
    Shape of ``(num_matrices,)`` and type :class:`int`.
    """

    ###
    # On-device Data (Variable)
    ###

    # These are the arrays to update when assembling the matrices

    dims: wp.array | None = None
    """
    The active dimensions of each sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`vec2i`.
    """

    num_nzb: wp.array | None = None
    """
    The active number of non-zero blocks per sparse matrices.\n
    Shape of ``(num_matrices,)`` and type :class:`int`.
    """

    nzb_coords: wp.array | None = None
    """
    The row-column coordinates of each non-zero block within its corresponding sparse matrix.\n
    Shape of ``(sum_of_num_nzb,)`` and type :class:`vec2i`.
    """

    nzb_values: wp.array | None = None
    """
    The flattened array containing all non-zero blocks over all sparse matrices.\n
    Shape of ``(sum_of_num_nzb,)`` and type :class:`float | vector | matrix`.
    """

    ###
    # Properties
    ###

    @property
    def max_rows(self) -> wp.array:
        assert self.max_dims is not None and self.max_dims.ptr is not None
        index_dtype_size_bytes = type_size_in_bytes(self.index_dtype)
        return wp.array(
            dtype=self.index_dtype,
            shape=(self.num_matrices,),
            ptr=self.max_dims.ptr,
            strides=(2 * index_dtype_size_bytes,),
            copy=False,
        )

    @property
    def max_cols(self) -> wp.array:
        assert self.max_dims is not None and self.max_dims.ptr is not None
        index_dtype_size_bytes = type_size_in_bytes(self.index_dtype)
        return wp.array(
            dtype=self.index_dtype,
            shape=(self.num_matrices,),
            ptr=self.max_dims.ptr + index_dtype_size_bytes,
            strides=(2 * index_dtype_size_bytes,),
            copy=False,
        )

    @property
    def num_rows(self) -> wp.array:
        assert self.dims is not None and self.dims.ptr is not None
        index_dtype_size_bytes = type_size_in_bytes(self.index_dtype)
        return wp.array(
            dtype=self.index_dtype,
            shape=(self.num_matrices,),
            ptr=self.dims.ptr,
            strides=(2 * index_dtype_size_bytes,),
            copy=False,
        )

    @property
    def num_cols(self) -> wp.array:
        assert self.dims is not None and self.dims.ptr is not None
        index_dtype_size_bytes = type_size_in_bytes(self.index_dtype)
        return wp.array(
            dtype=self.index_dtype,
            shape=(self.num_matrices,),
            ptr=self.dims.ptr + index_dtype_size_bytes,
            strides=(2 * index_dtype_size_bytes,),
            copy=False,
        )

    @property
    def nzb_row(self) -> wp.array:
        assert self.nzb_coords is not None and self.nzb_coords.ptr is not None
        index_dtype_size_bytes = type_size_in_bytes(self.index_dtype)
        return wp.array(
            dtype=self.index_dtype,
            shape=(self.sum_of_num_nzb,),
            ptr=self.nzb_coords.ptr,
            strides=(2 * index_dtype_size_bytes,),
            copy=False,
        )

    @property
    def nzb_col(self) -> wp.array:
        assert self.nzb_coords is not None and self.nzb_coords.ptr is not None
        index_dtype_size_bytes = type_size_in_bytes(self.index_dtype)
        return wp.array(
            dtype=self.index_dtype,
            shape=(self.sum_of_num_nzb,),
            ptr=self.nzb_coords.ptr + index_dtype_size_bytes,
            strides=(2 * index_dtype_size_bytes,),
            copy=False,
        )

    ###
    # Operations
    ###

    def finalize(
        self,
        max_dims: list[tuple[int, int]],
        capacities: list[int],
        nzb_dtype: BlockDType | None = None,
        index_dtype: IntType | None = None,
        device: wp.DeviceLike | None = None,
    ):
        """
        Finalizes the block-sparse matrix container by allocating on-device data arrays.

        Args:
            max_dims (list[tuple[int, int]]):
                A list of pairs of integers, specifying the maximum number of rows and columns for each matrix.
            capacities (list[int]):
                A list of integers specifying the maximum number of non-zero blocks for each sparse matrix.
            block_type (BlockDType, optional):
                An optional :class:`BlockDType` specifying the fixed type of each non-zero block.\n
                If not provided, it must be set prior to finalization.
            device (wp.DeviceLike, optional):
                An optional device on which to allocate the data arrays.\n
                If not provided, the existing device of the container will be used.

        Raises:
            RuntimeError:
                If the `nzb_dtype` field has not been specified prior to finalization.
            ValueError:
                If the `capacities` field is not a non-empty list of non-negative integers.
        """
        # Override the device if provided
        if device is not None:
            self.device = device
        # Override the block type if provided
        if nzb_dtype is not None:
            self.nzb_dtype = nzb_dtype
        # Override the index type if provided
        if index_dtype is not None:
            self.index_dtype = index_dtype

        # Ensure that the block and index dtypes have been specified
        if self.nzb_dtype is None:
            raise RuntimeError("The `nzb_dtype` field must be specified before finalizing the data arrays.")
        elif not isinstance(self.nzb_dtype, BlockDType):
            raise TypeError("The `nzb_dtype` field must be a valid BlockDType instance.")
        if self.index_dtype is None:
            raise RuntimeError("The `index_type` field must be specified before finalizing the data arrays.")
        elif self.index_dtype not in get_args(IntType):
            raise TypeError("The `index_type` field must be a valid IntType such as int32 or int64.")

        # Ensure that the max dimensions are valid
        if not isinstance(max_dims, list) or len(max_dims) == 0:
            raise ValueError("The `max_dims` field must be a non-empty list of integers 2-tuples.")
        for dims in max_dims:
            if not isinstance(dims, tuple) or len(dims) != 2:
                raise ValueError("All entries in the `max_dims` field must be 2-tuples of non-negative integers.")
            r, c = dims
            if not isinstance(r, int) or not isinstance(c, int) or r < 0 or c < 0:
                raise ValueError("All entries in the `max_dims` field must be 2-tuples of non-negative integers.")

        # Ensure that the capacities are valid
        if not isinstance(capacities, list) or len(capacities) == 0:
            raise ValueError("The `capacities` field must be a non-empty list of integers.")
        for cap in capacities:
            if not isinstance(cap, int) or cap < 0:
                raise ValueError("All entries in the `capacities` field must be non-negative integers.")

        # Ensure that inputs are consistent
        if len(max_dims) != len(capacities):
            raise ValueError("The `max_dims`, and `capacities` fields must have the same size.")

        # Update memory allocation meta-data caches
        self.num_matrices = len(capacities)
        self.max_of_max_dims = tuple(max(x) for x in zip(*max_dims, strict=True))
        self.sum_of_num_nzb = sum(capacities)
        self.max_of_num_nzb = max(capacities)

        # Compute cumulated sums for rows, cols and nzb
        dim_start_np = np.concatenate(([[0, 0]], np.asarray(max_dims).cumsum(axis=0)))[:-1]
        nzb_start_np = np.concatenate(([0], np.asarray(capacities).cumsum()))[:-1]

        # Initialize on-device warp arrays
        with wp.ScopedDevice(self.device):
            self.max_dims = wp.from_numpy(np.asarray(max_dims), shape=(self.num_matrices, 2), dtype=self.index_dtype)
            self.dims = wp.zeros(shape=(self.num_matrices, 2), dtype=self.index_dtype)
            self.row_start = wp.from_numpy(dim_start_np[:, 0], shape=(self.num_matrices,), dtype=self.index_dtype)
            self.col_start = wp.from_numpy(dim_start_np[:, 1], shape=(self.num_matrices,), dtype=self.index_dtype)
            self.max_nzb = wp.from_numpy(np.asarray(capacities), shape=(self.num_matrices,), dtype=self.index_dtype)
            self.num_nzb = wp.zeros(shape=(self.num_matrices,), dtype=self.index_dtype)
            self.nzb_start = wp.from_numpy(nzb_start_np, shape=(self.num_matrices,), dtype=self.index_dtype)
            self.nzb_coords = wp.zeros(shape=(self.sum_of_num_nzb, 2), dtype=self.index_dtype)
            self.nzb_values = wp.zeros(shape=(self.sum_of_num_nzb,), dtype=self.nzb_dtype.warp_type)

    def clear(self):
        """Clears all variable non-zero blocks."""
        self._assert_is_finalized()
        self.dims.zero_()
        self.num_nzb.zero_()
        self.nzb_coords.zero_()

    def zero(self):
        """Sets all non-zero block data to zero."""
        self._assert_is_finalized()
        self.nzb_values.zero_()

    def assign(self, matrices: list[np.ndarray]):
        """
        Assigns data to all sparse matrices from a list of dense NumPy arrays.

        This operation assumes that:
        - the sparse matrices have been finalized
        - the provided dense arrays match the active dimensions of each sparse matrix specified in `dims`
        - the non-zero blocks are filled in row-major order according to the current values of `nzb_coords`.

        Args:
            data (list[np.ndarray]):
                A list of dense NumPy arrays to assign to each sparse matrix.
        """
        # Ensure that the sparse matrices have been finalized
        self._assert_is_finalized()

        # Retrieve the fixed-size block dimensions
        block_nrows, block_ncols = self._get_block_shape()

        # Populate each sparse matrix from the provided dense arrays
        nzb_values_np = np.zeros_like(self.nzb_values.numpy())
        for m in range(self.num_matrices):
            # Retrieve the active matrix dimensions
            dims = self.dims.numpy()[m]
            nrows, ncols = int(dims[0]), int(dims[1])

            # Validate the provided dense array
            dense_matrix = matrices[m]
            if dense_matrix.shape != (nrows, ncols):
                raise ValueError(
                    f"The provided dense array for matrix {m} has shape {dense_matrix.shape}, "
                    f"but expected shape is ({nrows}, {ncols})."
                )

            # Populate non-zero blocks
            num_nzb = int(self.num_nzb.numpy()[m])
            start_idx = int(self.nzb_start.numpy()[m])
            coords = self.nzb_coords.numpy()[start_idx : start_idx + num_nzb]
            for b in range(num_nzb):
                row_idx, col_idx = int(coords[b][0]), int(coords[b][1])
                block_value = dense_matrix[row_idx : row_idx + block_nrows, col_idx : col_idx + block_ncols]
                nzb_values_np[start_idx + b] = block_value

        # Copy the populated non-zero block values to the device
        self.nzb_values.assign(nzb_values_np)

    def numpy(self) -> list[np.ndarray]:
        """Converts all sparse matrices to a list of dense NumPy arrays."""
        # Ensure that the sparse matrices have been finalized
        self._assert_is_finalized()

        # Retrieve the fixed-size block dimensions
        block_nrows, block_ncols = self._get_block_shape()

        # Retrieve sparse data from the device
        dims_np = self.dims.numpy()
        num_nzb_np = self.num_nzb.numpy()
        nzb_start_np = self.nzb_start.numpy()
        nzb_coords_np = self.nzb_coords.numpy()
        nzb_values_np = self.nzb_values.numpy()

        # Construct a list of dense NumPy matrices from the sparse representation
        matrices: list[np.ndarray] = []
        for m in range(self.num_matrices):
            # Retrieve the active matrix dimensions
            dims = dims_np[m]
            nrows, ncols = int(dims[0]), int(dims[1])

            # Allocate dense matrix initially filled with zeros
            dense_matrix = np.zeros((nrows, ncols), dtype=wp.dtype_to_numpy(self.nzb_dtype.dtype))

            # Populate non-zero blocks
            num_nzb = int(num_nzb_np[m])
            start_idx = int(nzb_start_np[m])
            coords = nzb_coords_np[start_idx : start_idx + num_nzb]
            values = nzb_values_np[start_idx : start_idx + num_nzb]
            for b in range(num_nzb):
                row_idx, col_idx = int(coords[b][0]), int(coords[b][1])
                block_value = values[b].reshape((block_nrows, block_ncols))
                dense_matrix[row_idx : row_idx + block_nrows, col_idx : col_idx + block_ncols] += block_value
            matrices.append(dense_matrix)

        # Return the list of dense matrices
        return matrices

    ###
    # Internals
    ###

    def _has_valid_metadata(self) -> bool:
        return (
            self.num_matrices > 0 and self.sum_of_num_nzb > 0 and self.max_of_num_nzb > 0 and self.nzb_dtype is not None
        )

    def _is_finalized(self) -> bool:
        return (
            self.nzb_values is not None
            and self.max_dims is not None
            and self.dims is not None
            and self.max_nzb is not None
            and self.num_nzb is not None
            and self.nzb_start is not None
            and self.nzb_coords is not None
            and self.nzb_values is not None
        )

    def _assert_is_finalized(self):
        if not self._is_finalized():
            raise RuntimeError("No data has been allocated. Call `finalize()` before use.")

    def _get_block_shape(self) -> tuple[int, int]:
        """Retrieves the fixed-size block shape as number of rows and columns according to row-major ordering."""
        # NOTE: Assumes row-major ordering
        block_shape = self.nzb_dtype.shape
        if isinstance(block_shape, int):
            block_nrows = 1
            block_ncols = block_shape
        elif isinstance(block_shape, tuple):
            if len(block_shape) == 0:
                block_nrows = 1
                block_ncols = 1
            elif len(block_shape) == 1:
                block_nrows = 1
                block_ncols = block_shape[0]
            elif len(block_shape) == 2:
                block_nrows = block_shape[0]
                block_ncols = block_shape[1]
            else:
                raise RuntimeError("Unsupported block shape for NumPy conversion.")
        else:
            raise RuntimeError("Unsupported block shape for NumPy conversion.")
        return block_nrows, block_ncols


###
# Dense to Block-Sparse Conversion
###


@wp.kernel
def _copy_square_dims_kernel(
    src_dim: wp.array(dtype=int32),
    dst_dims: wp.array2d(dtype=int32),
):
    """Copies square dimensions from 1D array to 2D (n, n) format."""
    wid = wp.tid()
    d = src_dim[wid]
    dst_dims[wid, 0] = d
    dst_dims[wid, 1] = d


@functools.cache
def _make_dense_to_bsm_detect_kernel(block_size: int):
    """Creates a kernel that detects non-zero blocks in dense matrices and populates BSM coordinates.

    Note: Dense matrices use canonical compact storage where stride = active dim (not maxdim).
    """

    @wp.kernel
    def kernel(
        # Dense matrix info
        dense_dim: wp.array(dtype=int32),
        dense_mio: wp.array(dtype=int32),
        dense_mat: wp.array(dtype=float32),
        # BSM info
        max_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        # Outputs
        num_nzb: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
    ):
        wid, bi, bj = wp.tid()

        dim = dense_dim[wid]
        bs = wp.static(block_size)
        n_blocks = (dim + bs - 1) // bs

        if bi >= n_blocks or bj >= n_blocks:
            return

        row_start = bi * bs
        col_start = bj * bs
        m_offset = dense_mio[wid]

        # Check if any element in this block is non-zero
        # Dense matrices use compact storage: stride = dim
        nonzero_count = int(0)
        for i in range(bs):
            row = row_start + i
            if row < dim:
                for j in range(bs):
                    col = col_start + j
                    if col < dim:
                        idx = m_offset + row * dim + col
                        if dense_mat[idx] != float32(0.0):
                            nonzero_count = nonzero_count + 1

        if nonzero_count > 0:
            slot = wp.atomic_add(num_nzb, wid, 1)
            cap = max_nzb[wid]
            if slot < cap:
                global_idx = nzb_start[wid] + slot
                nzb_coords[global_idx, 0] = row_start
                nzb_coords[global_idx, 1] = col_start

    return kernel


@functools.cache
def _make_dense_to_bsm_copy_kernel(block_size: int):
    """Creates a kernel that copies block values from dense matrices to BSM storage.

    Note: Dense matrices use canonical compact storage where stride = active dim (not maxdim).
    """

    mat_type = wp.types.matrix(shape=(block_size, block_size), dtype=float32)

    @wp.kernel
    def kernel(
        # Dense matrix info
        dense_dim: wp.array(dtype=int32),
        dense_mio: wp.array(dtype=int32),
        dense_mat: wp.array(dtype=float32),
        # BSM info
        nzb_start: wp.array(dtype=int32),
        num_nzb: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        # Output
        nzb_values: wp.array(dtype=mat_type),
    ):
        wid, block_idx = wp.tid()

        if block_idx >= num_nzb[wid]:
            return

        dim = dense_dim[wid]
        m_offset = dense_mio[wid]
        global_idx = nzb_start[wid] + block_idx
        row_start = nzb_coords[global_idx, 0]
        col_start = nzb_coords[global_idx, 1]

        bs = wp.static(block_size)
        block = mat_type()

        # Dense matrices use compact storage: stride = dim
        for i in range(bs):
            row = row_start + i
            for j in range(bs):
                col = col_start + j
                if row < dim and col < dim:
                    idx = m_offset + row * dim + col
                    block[i, j] = dense_mat[idx]
                else:
                    block[i, j] = float32(0.0)

        nzb_values[global_idx] = block

    return kernel


def allocate_block_sparse_from_dense(
    dense_op: "DenseLinearOperatorData",
    block_size: int,
    sparsity_threshold: float = 1.0,
    device: wp.DeviceLike | None = None,
) -> BlockSparseMatrices:
    """
    Allocates a BlockSparseMatrices container sized for converting from a dense operator.

    Args:
        dense_op: The dense linear operator to convert from.
        block_size: The size of each square block.
        sparsity_threshold: Fraction of maximum possible blocks to allocate for (0.0 to 1.0).
            E.g., 0.5 allocates for up to 50% of blocks being non-zero. Default 1.0 (all blocks).
        device: Device to allocate on. Defaults to the dense operator's device.

    Returns:
        A finalized but empty BlockSparseMatrices ready for use with dense_to_block_sparse_copy_values.
    """
    if dense_op.info is None:
        raise ValueError("Dense operator must have info set.")
    if not isinstance(dense_op.info, DenseSquareMultiLinearInfo):
        raise ValueError("Dense operator must be square (DenseSquareMultiLinearInfo).")

    info = dense_op.info
    if info.dimensions is None:
        raise ValueError("Dense operator info must have dimensions set.")
    if device is None:
        device = info.device

    max_dims_list: list[tuple[int, int]] = []
    capacities: list[int] = []

    for dim in info.dimensions:
        n_blocks_per_dim = (dim + block_size - 1) // block_size
        max_blocks = n_blocks_per_dim * n_blocks_per_dim
        capacity = max(1, int(sparsity_threshold * max_blocks))
        max_dims_list.append((dim, dim))
        capacities.append(capacity)

    nzb_dtype = BlockDType(dtype=info.dtype, shape=(block_size, block_size))

    bsm = BlockSparseMatrices()
    bsm.finalize(
        max_dims=max_dims_list,
        capacities=capacities,
        nzb_dtype=nzb_dtype,
        index_dtype=info.itype,
        device=device,
    )

    return bsm


def dense_to_block_sparse_copy_values(
    dense_op: "DenseLinearOperatorData",
    bsm: BlockSparseMatrices,
    block_size: int,
) -> None:
    """
    Converts dense matrix values to block-sparse format (graph-capturable).

    This function detects non-zero blocks and copies their values from the dense
    operator to the block-sparse matrices container. It is fully GPU-based and
    graph-capturable.

    Args:
        dense_op: The dense linear operator containing the matrix data.
        bsm: A pre-allocated BlockSparseMatrices (from allocate_block_sparse_from_dense).
        block_size: The block size (must match the BSM's block size).
    """
    if dense_op.info is None:
        raise ValueError("Dense operator must have info set.")
    if not isinstance(dense_op.info, DenseSquareMultiLinearInfo):
        raise ValueError("Dense operator must be square.")
    if not bsm._is_finalized():
        raise ValueError("BlockSparseMatrices must be finalized before use.")

    info = dense_op.info
    device = bsm.device

    # Reset num_nzb counter for fresh detection
    bsm.num_nzb.zero_()

    # Copy active dimensions from dense to BSM
    wp.launch(
        _copy_square_dims_kernel,
        dim=(info.num_blocks,),
        inputs=[info.dim],
        outputs=[bsm.dims],
        device=device,
    )

    # Compute launch dimensions
    max_dim = info.max_dimension
    max_blocks_per_dim = (max_dim + block_size - 1) // block_size

    # Get cached kernels
    detect_kernel = _make_dense_to_bsm_detect_kernel(block_size)
    copy_kernel = _make_dense_to_bsm_copy_kernel(block_size)

    # Launch detection kernel
    wp.launch(
        detect_kernel,
        dim=(info.num_blocks, max_blocks_per_dim, max_blocks_per_dim),
        inputs=[
            info.dim,
            info.mio,
            dense_op.mat,
            bsm.max_nzb,
            bsm.nzb_start,
        ],
        outputs=[
            bsm.num_nzb,
            bsm.nzb_coords,
        ],
        device=device,
    )

    # Launch copy kernel
    wp.launch(
        copy_kernel,
        dim=(bsm.num_matrices, bsm.max_of_num_nzb),
        inputs=[
            info.dim,
            info.mio,
            dense_op.mat,
            bsm.nzb_start,
            bsm.num_nzb,
            bsm.nzb_coords,
        ],
        outputs=[
            bsm.nzb_values,
        ],
        device=device,
    )
