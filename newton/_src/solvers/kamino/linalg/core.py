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
KAMINO: Linear Algebra: Core types and utilities for multi-linear systems

This module provides data structures and utilities for managing multiple
independent linear systems, including rectangular and square systems.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp
from warp.context import Devicelike

from ..core.types import FloatType, IntType, VecIntType, float32, int32
from ..utils import logger as msg

###
# Module interface
###

__all__ = [
    "DenseLinearOperatorData",
    "DenseRectangularMultiLinearInfo",
    "DenseSquareMultiLinearInfo",
    "make_dtype_tolerance",
]


###
# Types
###


@dataclass
class DenseRectangularMultiLinearInfo:
    """
    A data structure for managing multiple rectangular linear systems of inhomogeneous and mutable shapes, i.e.:

    `A_i @ x_i = b_i`, for `i = 1, ..., num_blocks`,

    where:
    - each `A_i` is a rectangular matrix of active shape `(dim[i][0], dim[i][1])` and
      maximum shape `(maxdim[i][0], maxdim[i][1])` starting at offset `mio[i]`
    - each `b_i` is a right-hand-side (rhs) vector of active shape `(dim[i][0],)`
      and maximum shape `(maxdim[i][1],)` starting at offset `rvio[i]`
    - each `x_i` is a input vector of active shape `(dim[i][1],)` and
      maximum shape `(maxdim[i][0],)` starting at offset `ivio[i]`
    - `num_blocks` is the number of linear systems managed by this data structure

    The underlying data allocation is determined by the sum of `maxdim[i]` values, while the "active"
    shapes are determined by the `dim[i]` values. Thus the allocated memory corresponds to:
    - `sum(maxdim[i][0]*maxdim[i][1] for i in range(num_blocks))` for matrices
    - `sum(maxdim[i][1] for i in range(num_blocks))` for rhs vectors
    - `sum(maxdim[i][0] for i in range(num_blocks))` for input vectors

    Kernels operating on data described by this structure can then use the max over `maxdims` to set
    the multi-dimensional thread block size, while using the `dim` values at execution time to determine
    the actual active shape of each block and access the correct data offsets using the `mio` and `vio` arrays.
    """

    num_blocks: int = 0
    """Host-side cache of the number of data blocks represented in each flat data array."""

    dimensions: list[tuple[int, int]] | None = None
    """Host-side cache of the dimensions of each rectangular linear system."""

    max_dimensions: tuple[int, int] = (0, 0)
    """Host-side cache of the maximum dimension over all matrix blocks."""

    total_mat_size: int = 0
    """
    Host-side cache of the total size of the flat matrix data array.
    This is equal to `sum(maxdim[i][0]*maxdim[i][1] for i in range(num_blocks))`.
    """

    total_rhs_size: int = 0
    """
    Host-side cache of the total size of the flat data array of rhs vectors.
    This is equal to `sum(maxdim[i][1] for i in range(num_blocks))`.
    """

    total_inp_size: int = 0
    """
    Host-side cache of the total size of the flat data array of input vectors.
    This is equal to `sum(maxdim[i][0] for i in range(num_blocks))`.
    """

    dtype: FloatType = float32
    """The data type of the underlying matrix and vector data arrays."""

    itype: IntType = int32
    """The integer type used for indexing the underlying data arrays."""

    device: Devicelike | None = None
    """The device on which the data arrays are allocated."""

    maxdim: wp.array | None = None
    """
    The maximum dimensions of each rectangular matrix block.
    Shape of ``(num_blocks,)`` and type :class:`vec2i`.
    Each entry corresponds to the shape `(max_rows, max_cols)`.
    """

    dim: wp.array | None = None
    """
    The active dimensions of each rectangular matrix block.
    Shape of ``(num_blocks,)`` and type :class:`vec2i`.
    Each entry corresponds to the shape `(rows, cols)`.
    """

    mio: wp.array | None = None
    """
    The matrix index offset (mio) of each block in the flat data array.
    Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
    """

    rvio: wp.array | None = None
    """
    The rhs vector index offset (vio) of each block in the flat data array.
    Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
    """

    ivio: wp.array | None = None
    """
    The input vector index offset (vio) of each block in the flat data array.
    Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
    """

    @staticmethod
    def _check_dimensions(dims: list[tuple[int, int]] | tuple[int, int]) -> list[tuple[int, int]]:
        if isinstance(dims, tuple):
            if len(dims) != 2:
                raise ValueError("Dimension tuple must have exactly two entries.")
            if dims[0] <= 0 or dims[1] <= 0:
                raise ValueError("Dimensions must be positive integers.")
            dims = [dims]
        elif isinstance(dims, list):
            if len(dims) > 0 and not all(
                isinstance(d, tuple) and len(d) == 2 and all(isinstance(i, int) and i > 0 for i in d) for d in dims
            ):
                raise ValueError("All dimensions must be tuples of two positive integers.")
        else:
            raise TypeError("Dimensions must be a pair of positive integers or a list of positive integer pairs.")
        return dims

    def allocate(
        self,
        dimensions: list[tuple[int, int]],
        dtype: FloatType = float32,
        itype: IntType = int32,
        device: Devicelike = None,
    ) -> None:
        """
        Constructs and allocates the data of the rectangular multi-linear system info on the specified device.
        """
        # Ensure the problem dimensions are valid and cache them
        self.dimensions = self._check_dimensions(dimensions)

        # Ensure the dtype and itype are valid
        if not issubclass(dtype, FloatType):
            raise TypeError("Invalid dtype. Expected FloatType type, e.g. `wp.float32` or `wp.float64`.")
        if not issubclass(itype, IntType):
            raise TypeError("Invalid itype. Expected IntType type, e.g. `wp.int32` or `wp.int64`.")
        self.dtype = dtype
        self.itype = itype

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self.device = device

        # Compute the allocation sizes and offsets for the flat data arrays
        mat_sizes = [m * n for m, n in self.dimensions]
        mat_offsets = [0] + [sum(mat_sizes[:i]) for i in range(1, len(mat_sizes) + 1)]
        mat_flat_size = sum(mat_sizes)
        max_mat_rows = max(m for m, _ in self.dimensions)
        max_mat_cols = max(n for _, n in self.dimensions)
        rhs_sizes = [m for m, _ in self.dimensions]
        rhs_offsets = [0] + [sum(rhs_sizes[:i]) for i in range(1, len(rhs_sizes) + 1)]
        rhs_flat_size = sum(rhs_sizes)
        inp_sizes = [n for _, n in self.dimensions]
        inp_offsets = [0] + [sum(inp_sizes[:i]) for i in range(1, len(inp_sizes) + 1)]
        inp_flat_size = sum(inp_sizes)

        # Update the allocation meta-data the specified system dimensions
        self.num_blocks = len(self.dimensions)
        self.max_dimensions = (max_mat_rows, max_mat_cols)
        self.total_mat_size = mat_flat_size
        self.total_rhs_size = rhs_flat_size
        self.total_inp_size = inp_flat_size

        # Declare local 2D dimension type
        class _vec2i(wp.types.vector(length=2, dtype=self.itype)):
            pass

        # Allocate the multi-linear square system info data on the specified device
        with wp.ScopedDevice(self.device):
            self.maxdim = wp.array(self.dimensions, dtype=_vec2i)
            self.dim = wp.array(self.dimensions, dtype=_vec2i)
            self.mio = wp.array(mat_offsets[: self.num_blocks], dtype=self.itype)
            self.rvio = wp.array(rhs_offsets[: self.num_blocks], dtype=self.itype)
            self.ivio = wp.array(inp_offsets[: self.num_blocks], dtype=self.itype)

    def assign(
        self,
        maxdim: wp.array,
        dim: wp.array,
        mio: wp.array,
        rvio: wp.array,
        ivio: wp.array,
        dtype: FloatType = float32,
        device: Devicelike = None,
    ) -> None:
        """
        Assigns the data of the square multi-linear system info from externally allocated arrays.
        """
        # Ensure the problem dimensions are valid and cache them
        self.dimensions = self._check_dimensions(maxdim.list())

        # Ensure the dtype and itype are valid
        if not issubclass(dtype, FloatType):
            raise TypeError("Invalid dtype. Expected FloatType type, e.g. `wp.float32` or `wp.float64`.")
        if not issubclass(maxdim.dtype, VecIntType):
            raise TypeError(
                "Invalid dtype of `maxdim` argument. Expected integer vector type, e.g. `wp.vec2i` or `wp.vec2l`."
            )
        if not issubclass(dim.dtype, VecIntType):
            raise TypeError(
                "Invalid dtype of `dim` argument. Expected integer vector type, e.g. `wp.vec2i` or `wp.vec2l`."
            )
        if not issubclass(mio.dtype, IntType):
            raise TypeError("Invalid dtype of `mio` argument. Expected IntType type, e.g. `wp.int32` or `wp.int64`.")
        if not issubclass(rvio.dtype, IntType):
            raise TypeError("Invalid dtype of `rvio` argument. Expected IntType type, e.g. `wp.int32` or `wp.int64`.")
        if not issubclass(ivio.dtype, IntType):
            raise TypeError("Invalid dtype of `ivio` argument. Expected IntType type, e.g. `wp.int32` or `wp.int64`.")

        # Cache the data type information
        self.dtype = dtype
        self.itype = maxdim.dtype

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self.device = device

        # Compute the allocation sizes and offsets for the flat data arrays
        mat_sizes = [m * n for m, n in self.dimensions]
        mat_flat_size = sum(mat_sizes)
        max_mat_rows = max(m for m, _ in self.dimensions)
        max_mat_cols = max(n for _, n in self.dimensions)
        rhs_sizes = [m for m, _ in self.dimensions]
        rhs_flat_size = sum(rhs_sizes)
        inp_sizes = [n for _, n in self.dimensions]
        inp_flat_size = sum(inp_sizes)

        # Update the allocation meta-data the specified system dimensions
        self.num_blocks = len(self.dimensions)
        self.max_dimensions = (max_mat_rows, max_mat_cols)
        self.total_mat_size = mat_flat_size
        self.total_rhs_size = rhs_flat_size
        self.total_inp_size = inp_flat_size

        # Capture references the rectangular multi-linear system info data on the specified device
        self.maxdim = maxdim
        self.dim = dim
        self.mio = mio
        self.rvio = rvio
        self.ivio = ivio

    def is_matrix_compatible(self, A: wp.array) -> bool:
        """Checks if the provided matrix data array is compatible with the specified info structure."""
        return A.dtype == self.dtype and A.size >= self.total_mat_size

    def is_rhs_compatible(self, b: wp.array) -> bool:
        """Checks if the provided rhs vector data array is compatible with the specified info structure."""
        return b.dtype == self.dtype and b.size >= self.total_rhs_size

    def is_input_compatible(self, x: wp.array) -> bool:
        """Checks if the provided input vector data array is compatible with the specified info structure."""
        return x.dtype == self.dtype and x.size >= self.total_inp_size

    def __str__(self) -> str:
        return (
            f"DenseRectangularMultiLinearInfo(\n"
            f"  num_blocks={self.num_blocks},\n"
            f"  dimensions={self.dimensions},\n"
            f"  max_dimensions={self.max_dimensions},\n"
            f"  total_mat_size={self.total_mat_size},\n"
            f"  total_rhs_size={self.total_rhs_size},\n"
            f"  total_inp_size={self.total_inp_size},\n"
            f"  dtype={self.dtype},\n"
            f"  itype={self.itype},\n"
            f"  device={self.device}\n"
            f")"
        )


@dataclass
class DenseSquareMultiLinearInfo:
    """
    A data structure for managing multiple square linear systems of inhomogeneous and mutable shapes, i.e.:

    `A_i @ x_i = b_i`, for `i = 1, ..., num_blocks`,

    where:
    - each `A_i` is a square matrix of active shape `(dim[i], dim[i])` and
      maximum shape `(maxdim[i], maxdim[i])` starting at offset `mio[i]`
    - each `b_i` is a right-hand-side (rhs) vector of active shape `(dim[i],)`
      and maximum shape `(maxdim[i],)` starting at offset `vio[i]`
    - each `x_i` is a input vector of active shape `(dim[i],)` and
      maximum shape `(maxdim[i],)` starting at offset `vio[i]`
    - `num_blocks` is the number of linear systems managed by this data structure

    The underlying data allocation is determined by the sum of `maxdim[i]` values, while the "active"
    shapes are determined by the `dim[i]` values. Thus the allocated memory corresponds to:
    - `sum(maxdim[i]*maxdim[i] for i in range(num_blocks))` for matrices
    - `sum(maxdim[i] for i in range(num_blocks))` for rhs vectors
    - `sum(maxdim[i] for i in range(num_blocks))` for input vectors

    Kernels operating on data described by this structure can then use the max over `maxdims` to set
    the multi-dimensional thread block size, while using the `dim` values at execution time to determine
    the actual active shape of each block and access the correct data offsets using the `mio` and `vio` arrays.
    """

    num_blocks: int = 0
    """Host-side cache of the number of data blocks represented in each flat data array."""

    dimensions: list[int] | None = None
    """Host-side cache of the dimensions of each square linear system."""

    max_dimension: int = 0
    """Host-side cache of the maximum dimension over all matrix blocks."""

    total_mat_size: int = 0
    """
    Host-side cache of the total size of the flat data array of matrix blocks.
    This is equal to `sum(maxdim[i][0]*maxdim[i][1] for i in range(num_blocks))`.
    """

    total_vec_size: int = 0
    """
    Host-side cache of the total size of the flat data array of vector blocks.
    This is equal to `sum(maxdim[i][1] for i in range(num_blocks))`.
    """

    dtype: FloatType = float32
    """The data type of the underlying matrix and vector data arrays."""

    itype: IntType = int32
    """The integer type used for indexing the underlying data arrays."""

    device: Devicelike | None = None
    """The device on which the data arrays are allocated."""

    maxdim: wp.array | None = None
    """
    The maximum dimensions of each square matrix block.
    Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
    """

    dim: wp.array | None = None
    """
    The active dimensions of each square matrix block.
    Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
    """

    mio: wp.array | None = None
    """
    The matrix index offset (mio) of each matrix block in the flat data array.
    Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
    """

    vio: wp.array | None = None
    """
    The vector index offset (vio) of each vector block in the flat data array.
    Shape of ``(num_blocks,)`` and type :class:`int | int32 | int64`.
    """

    @staticmethod
    def _check_dimensions(dims: list[int] | int) -> list[int]:
        if isinstance(dims, int):
            if dims <= 0:
                raise ValueError("Dimension must be a positive integer.")
            dims = [dims]
        elif isinstance(dims, list):
            if len(dims) > 0 and not all(isinstance(d, int) and d > 0 for d in dims):
                raise ValueError("All dimensions must be positive integers.")
        else:
            raise TypeError("Dimensions must be an positive integer or a list of positive integers.")
        return dims

    def allocate(
        self, dimensions: list[int], dtype: FloatType = float32, itype: IntType = int32, device: Devicelike = None
    ) -> None:
        """
        Constructs and allocates the data of the square multi-linear system info on the specified device.
        """
        # Ensure the problem dimensions are valid and cache them
        self.dimensions = self._check_dimensions(dimensions)

        # Ensure the dtype and itype are valid
        if not issubclass(dtype, FloatType):
            raise TypeError("Invalid dtype. Expected FloatType type, e.g. `wp.float32` or `wp.float64`.")
        if not issubclass(itype, IntType):
            raise TypeError("Invalid itype. Expected IntType type, e.g. `wp.int32` or `wp.int64`.")
        self.dtype = dtype
        self.itype = itype

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self.device = device

        # Compute the allocation sizes and offsets for the flat data arrays
        mat_sizes = [n * n for n in self.dimensions]
        mat_offsets = [0] + [sum(mat_sizes[:i]) for i in range(1, len(mat_sizes) + 1)]
        mat_flat_size = sum(mat_sizes)
        vec_sizes = self.dimensions
        vec_offsets = [0] + [sum(vec_sizes[:i]) for i in range(1, len(vec_sizes) + 1)]
        vec_flat_size = sum(vec_sizes)

        # Update the allocation meta-data the specified system dimensions
        self.num_blocks = len(self.dimensions)
        self.max_dimension = max(self.dimensions)
        self.total_mat_size = mat_flat_size
        self.total_vec_size = vec_flat_size

        # Allocate the multi-linear square system info data on the specified device
        with wp.ScopedDevice(self.device):
            self.maxdim = wp.array(self.dimensions, dtype=self.itype)
            self.dim = wp.array(self.dimensions, dtype=self.itype)
            self.mio = wp.array(mat_offsets[: self.num_blocks], dtype=self.itype)
            self.vio = wp.array(vec_offsets[: self.num_blocks], dtype=self.itype)

    def assign(
        self,
        maxdim: wp.array,
        dim: wp.array,
        mio: wp.array,
        vio: wp.array,
        dtype: FloatType = float32,
        device: Devicelike = None,
    ) -> None:
        """
        Assigns the data of the square multi-linear system info from externally allocated arrays.
        """
        # Ensure the problem dimensions are valid and cache them
        self.dimensions = self._check_dimensions(maxdim.numpy().astype(int).tolist())

        # Ensure the dtype and itype are valid
        if not issubclass(dtype, FloatType):
            raise TypeError("Invalid dtype. Expected FloatType type, e.g. `wp.float32` or `wp.float64`.")
        if not issubclass(maxdim.dtype, IntType):
            raise TypeError("Invalid dtype of `maxdim` argument. Expected IntType type, e.g. `wp.int32` or `wp.int64`.")
        if not issubclass(dim.dtype, IntType):
            raise TypeError("Invalid dtype of `dim` argument. Expected IntType type, e.g. `wp.int32` or `wp.int64`.")
        if not issubclass(mio.dtype, IntType):
            raise TypeError("Invalid dtype of `mio` argument. Expected IntType type, e.g. `wp.int32` or `wp.int64`.")
        if not issubclass(vio.dtype, IntType):
            raise TypeError("Invalid dtype of `vio` argument. Expected IntType type, e.g. `wp.int32` or `wp.int64`.")

        # Cache the data type information
        self.dtype = dtype
        self.itype = maxdim.dtype

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self.device = device

        # Compute the allocation sizes and offsets for the flat data arrays
        mat_sizes = [n * n for n in self.dimensions]
        mat_flat_size = sum(mat_sizes)
        vec_sizes = self.dimensions
        vec_flat_size = sum(vec_sizes)

        # Update the allocation meta-data the specified system dimensions
        self.num_blocks = len(self.dimensions)
        self.max_dimension = max(self.dimensions)
        self.total_mat_size = mat_flat_size
        self.total_vec_size = vec_flat_size

        # Capture references the multi-linear square system info data on the specified device
        self.maxdim = maxdim
        self.dim = dim
        self.mio = mio
        self.vio = vio

    def is_matrix_compatible(self, A: wp.array) -> bool:
        """Checks if the provided matrix data array is compatible with the specified info structure."""
        return A.dtype == self.dtype and A.size >= self.total_mat_size

    def is_rhs_compatible(self, b: wp.array) -> bool:
        """Checks if the provided rhs vector data array is compatible with the specified info structure."""
        return b.dtype == self.dtype and b.size >= self.total_vec_size

    def is_input_compatible(self, x: wp.array) -> bool:
        """Checks if the provided input vector data array is compatible with the specified info structure."""
        return x.dtype == self.dtype and x.size >= self.total_vec_size

    def __str__(self) -> str:
        return (
            f"DenseSquareMultiLinearInfo(\n"
            f"  num_blocks={self.num_blocks},\n"
            f"  dimensions={self.dimensions},\n"
            f"  max_dimension={self.max_dimension},\n"
            f"  total_mat_size={self.total_mat_size},\n"
            f"  total_vec_size={self.total_vec_size},\n"
            f"  dtype={self.dtype},\n"
            f"  itype={self.itype},\n"
            f"  device={self.device}\n"
            f")"
        )


@dataclass
class DenseLinearOperatorData:
    """
    A data structure for encapsulating a multi-linear matrix operator.

    This object essentially wraps a flattened memory allocation of multiple
    matrix blocks along with a data structure describing the layout of the
    blocks, and provides a unified interface for linear solvers to operate
    on the encapsulated operator.

    The `info` member can be owned by this object or captured by reference
    from an external source to avoid unnecessary memory reallocations.
    """

    info: DenseRectangularMultiLinearInfo | DenseSquareMultiLinearInfo | None = None
    """The multi-linear data structure describing the operator."""

    mat: wp.array | None = None
    """The flat data array containing the matrix blocks."""

    def zero(self) -> None:
        self.mat.zero_()


###
# Utilities
###


def make_dtype_tolerance(tol: FloatType | float | None = None, dtype: FloatType = float32) -> FloatType:
    # First ensure the specified dtype is a valid warp type
    if not issubclass(dtype, FloatType):
        raise ValueError("data type 'dtype' must be a FloatType, e.g. a `wp.float32` or `wp.float64` value etc.")

    # Extract machine epsilon for the specified dtype
    eps = float(np.finfo(wp.dtype_to_numpy(dtype)).eps)

    # Default tolerance to machine epsilon if not provided
    if tol is None:
        return dtype(eps)

    # Otherwise ensure the provided tolerance is valid and converted to the requested dtype
    else:
        # Ensure the provided tolerance is a valid FloatType value
        if not isinstance(tol, FloatType | float):
            raise ValueError(
                "tolerance 'tol' must be a FloatType, i.e. a `float`, `wp.float32`, or `wp.float64` value."
            )

        # Ensure the provided tolerance is positive and non-zero
        if float(tol) <= 0:
            raise ValueError("tolerance 'tol' must be a positive value.")

        # Issue warning if the provided tolerance is smaller than machine epsilon
        if float(tol) < eps:
            msg.warning(
                f"tolerance 'tol' = {tol} is smaller than machine epsilon "
                f"for the specified dtype '{dtype}' (eps = {eps}). Clamping to eps."
            )

        # Return the tolerance clamped to machine epsilon for the specified dtype
        return dtype(max(float(tol), eps))
