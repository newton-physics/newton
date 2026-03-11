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

from collections.abc import Callable

import warp as wp

from ..core.types import FloatType
from .blas import (
    block_sparse_gemv,
    block_sparse_matvec,
    block_sparse_transpose_gemv,
    block_sparse_transpose_matvec,
)
from .sparse_matrix import BlockSparseMatrices

###
# Module interface
###

__all__ = [
    "BlockSparseLinearOperators",
]


class BlockSparseLinearOperators:
    """
    A Block-Sparse Linear Operator container for representing
    and operating on multiple independent sparse linear systems.
    """

    def __init__(self, bsm: BlockSparseMatrices | None = None):
        self.bsm = bsm
        self.initialize_default_operators()

        self._active_rows: wp.array | None = None
        self._active_cols: wp.array | None = None

        if self.bsm is not None:
            self._active_rows = self.bsm.num_rows
            self._active_cols = self.bsm.num_cols

    ###
    # On-device Data
    ###

    bsm: BlockSparseMatrices | None = None
    """
    The underlying block-sparse matrix used by this operator.
    """

    ###
    # Operators
    ###

    precompute_op: Callable | None = None
    """
    The operator function for precomputing any necessary data for the operators.\n
    Signature: ``precompute_op(A: BlockSparseLinearOperators)``.
    """

    Ax_op: Callable | None = None
    """
    The operator function for performing sparse matrix-vector products `y = A @ x`.\n
    Example signature: ``Ax_op(A: BlockSparseLinearMatrices, x: wp.array, y: wp.array, matrix_mask: wp.array)``.
    """

    ATy_op: Callable | None = None
    """
    The operator function for performing sparse matrix-transpose-vector products `x = A^T @ y`.\n
    Example signature: ``ATy_op(A: BlockSparseLinearMatrices, y: wp.array, x: wp.array, matrix_mask: wp.array)``.
    """

    gemv_op: Callable | None = None
    """
    The operator function for performing generalized sparse matrix-vector products `y = alpha * A @ x + beta * y`.\n
    Example signature: ``gemv_op(A: BlockSparseLinearMatrices, x: wp.array, y: wp.array, alpha: float, beta: float, matrix_mask: wp.array)``.
    """

    gemvt_op: Callable | None = None
    """
    The operator function for performing generalized sparse matrix-transpose-vector products `x = alpha * A^T @ y + beta * x`.\n
    Example signature: ``gemvt_op(A: BlockSparseLinearMatrices, y: wp.array, x: wp.array, alpha: float, beta: float, matrix_mask: wp.array)``.
    """

    ###
    # Properties
    ###

    @property
    def num_matrices(self) -> int:
        return self.bsm.num_matrices

    @property
    def max_of_max_dims(self) -> tuple[int, int]:
        return self.bsm.max_of_max_dims

    @property
    def dtype(self) -> FloatType:
        return self.bsm.nzb_dtype.dtype

    @property
    def device(self) -> wp.DeviceLike:
        return self.bsm.device

    @property
    def active_rows(self) -> wp.array:
        return self._active_rows

    @property
    def active_cols(self) -> wp.array:
        return self._active_cols

    ###
    # Operations
    ###

    def clear(self):
        """Clears all variable sub-blocks."""
        self.bsm.clear()

    def zero(self):
        """Sets all sub-block data to zero."""
        self.bsm.zero()

    def precompute(self):
        """Precomputes any necessary data for the operators."""
        if self.precompute_op:
            self.precompute_op(self)

    def initialize_default_operators(self):
        """Sets all operator functions to their default implementations."""
        self.Ax_op = block_sparse_matvec
        self.ATy_op = block_sparse_transpose_matvec
        self.gemv_op = block_sparse_gemv
        self.gemvt_op = block_sparse_transpose_gemv

    def matvec(self, x: wp.array, y: wp.array, matrix_mask: wp.array):
        """Performs the sparse matrix-vector product `y = A @ x`."""
        if self.Ax_op is None:
            raise RuntimeError("No `A@x` operator has been assigned.")
        self.Ax_op(self.bsm, x, y, matrix_mask)

    def matvec_transpose(self, y: wp.array, x: wp.array, matrix_mask: wp.array):
        """Performs the sparse matrix-transpose-vector product `x = A^T @ y`."""
        if self.ATy_op is None:
            raise RuntimeError("No `A^T@y` operator has been assigned.")
        self.ATy_op(self.bsm, y, x, matrix_mask)

    def gemv(self, x: wp.array, y: wp.array, matrix_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """Performs a BLAS-like generalized sparse matrix-vector product `y = alpha * A @ x + beta * y`."""
        if self.gemv_op is None:
            raise RuntimeError("No BLAS-like `GEMV` operator has been assigned.")
        self.gemv_op(self.bsm, x, y, alpha, beta, matrix_mask)

    def gemv_transpose(self, y: wp.array, x: wp.array, matrix_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """Performs a BLAS-like generalized sparse matrix-transpose-vector product `x = alpha * A^T @ y + beta * x`."""
        if self.gemvt_op is None:
            raise RuntimeError("No BLAS-like transposed `GEMV` operator has been assigned.")
        self.gemvt_op(self.bsm, y, x, alpha, beta, matrix_mask)
