# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Linear system solvers for multiple independent linear systems.

This module provides interfaces for and implementations of linear
system solvers, that can solve multiple independent linear systems
in parallel, with support for both rectangular and square systems.
Depending on the particular solver implementation, both inter- and
intra-system parallelism may be exploited.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic

import numpy as np
import warp as wp

from .....core.types import override
from . import conjugate, factorize
from .conjugate_fused import MAX_BLOCKS_PER_ROW, build_row_index, build_transpose_index, make_fused_cr_kernel
from .core import DenseLinearOperatorData, DenseSquareMultiLinearInfo, make_dtype_tolerance
from .sparse_matrix import (
    BlockSparseMatrices,
    allocate_block_sparse_from_dense,
    dense_to_block_sparse_copy_values,
)
from .sparse_operator import BlockSparseLinearOperators
from .types import IndexType, ScalarType

###
# Module interface
###

__all__ = [
    "ConjugateGradientSolver",
    "ConjugateResidualSolver",
    "ConjugateResidualSolverFused",
    "DirectSolver",
    "LLTBlockedSolver",
    "LLTSequentialSolver",
    "LinearSolver",
    "LinearSolverType",
]


###
# Interfaces
###


class LinearSolver(ABC, Generic[ScalarType, IndexType]):
    """
    An abstract base class for linear system solvers.
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData[ScalarType, IndexType] | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        dtype: type[ScalarType] = wp.float32,  # type: ignore[assignment]
        device: wp.DeviceLike | None = None,
        **kwargs: dict[str, Any],
    ):
        # Declare and initialize the internal reference to the matrix/operator data
        self._operator: DenseLinearOperatorData[ScalarType, IndexType] | None = operator

        # Override dtype if linear operator is provided
        if operator is not None:
            dtype = operator.info.dtype

        # Declare and initialize internal meta-data
        self._dtype: type[ScalarType] = dtype
        self._atol: float = atol
        self._rtol: float = rtol

        # Declare and initialize the device identifier
        self._device: wp.DeviceLike = device

        # If an operator is provided, proceed with any necessary memory allocations
        if operator is not None:
            self.finalize(operator, **kwargs)

    ###
    # Properties
    ###

    @property
    def operator(self) -> DenseLinearOperatorData[ScalarType, IndexType]:
        if self._operator is None:
            raise ValueError("No linear operator has been allocated!")
        return self._operator

    @property
    def dtype(self) -> type[ScalarType]:
        return self._dtype

    @property
    def device(self) -> wp.DeviceLike:
        return self._device

    ###
    # Internals
    ###

    def _set_tolerance_dtype(self):
        self._atol = make_dtype_tolerance(self._atol, dtype=self._dtype)
        self._rtol = make_dtype_tolerance(self._rtol, dtype=self._dtype)

    ###
    # Implementation API
    ###

    @abstractmethod
    def _allocate_impl(
        self, operator: DenseLinearOperatorData[ScalarType, IndexType], **kwargs: dict[str, Any]
    ) -> None:
        raise NotImplementedError("An allocation operation is not implemented.")

    @abstractmethod
    def _reset_impl(self, A: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A reset operation is not implemented.")

    @abstractmethod
    def _compute_impl(self, A: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A compute operation is not implemented.")

    @abstractmethod
    def _solve_impl(self, b: wp.array[ScalarType], x: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A solve operation is not implemented.")

    @abstractmethod
    def _solve_inplace_impl(self, x: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A solve-in-place operation is not implemented.")

    ###
    # Public API
    ###

    def finalize(self, operator: DenseLinearOperatorData[ScalarType, IndexType], **kwargs: dict[str, Any]) -> None:
        """
        Ingest a linear operator and allocate any necessary internal memory
        based on the multi-linear layout specified by the operator's info.
        """
        # Check the operator is valid
        if operator is None:
            raise ValueError("A valid linear operator must be provided!")
        if not isinstance(operator, DenseLinearOperatorData):
            raise ValueError("The provided operator is not a DenseLinearOperatorData instance!")
        if operator.info is None:
            raise ValueError("The provided operator does not have any associated info!")
        self._operator = operator
        self._dtype = operator.info.dtype
        self._set_tolerance_dtype()
        self._allocate_impl(operator, **kwargs)

    def reset(self) -> None:
        """Resets the internal solver data (e.g. possibly to zeros)."""
        self._reset_impl()

    def compute(self, A: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        """Ingest matrix data and pre-compute any rhs-independent intermediate data."""
        if not self._operator.info.is_matrix_compatible(A):
            raise ValueError("The provided flat matrix data array does not have enough memory!")
        self._compute_impl(A=A, **kwargs)

    def solve(self, b: wp.array[ScalarType], x: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        """Solves the multi-linear systems `A @ x = b`."""
        if not self._operator.info.is_rhs_compatible(b):
            raise ValueError("The provided flat rhs vector data array does not have enough memory!")
        if not self._operator.info.is_input_compatible(x):
            raise ValueError("The provided flat input vector data array does not have enough memory!")
        self._solve_impl(b=b, x=x, **kwargs)

    def solve_inplace(self, x: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        """Solves the multi-linear systems `A @ x = b` in-place, where `x` is initialized with rhs data."""
        if not self._operator.info.is_input_compatible(x):
            raise ValueError("The provided flat input vector data array does not have enough memory!")
        self._solve_inplace_impl(x=x, **kwargs)


class DirectSolver(LinearSolver[ScalarType, IndexType]):
    """
    An abstract base class for direct linear system solvers based on matrix factorization.
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData[ScalarType, IndexType] | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: type[ScalarType] = wp.float32,  # type: ignore[assignment]
        device: wp.DeviceLike | None = None,
        **kwargs: dict[str, Any],
    ):
        # Default factorization tolerance to machine epsilon if not provided
        ftol = make_dtype_tolerance(ftol, dtype=dtype)

        # Initialize internal meta-data
        self._ftol: float | None = ftol
        self._has_factors: bool = False

        # Initialize base class members
        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    ###
    # Internals
    ###

    def _check_has_factorization(self):
        """Checks if the factorization has been computed, otherwise raises error."""
        if not self._has_factors:
            raise ValueError("A factorization has not been computed!")

    ###
    # Implementation API
    ###

    @abstractmethod
    def _factorize_impl(self, A: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A matrix factorization implementation is not provided.")

    @abstractmethod
    def _reconstruct_impl(self, A: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        raise NotImplementedError("A matrix reconstruction implementation is not provided.")

    ###
    # Internals
    ###

    @override
    def _compute_impl(self, A: wp.array[ScalarType], **kwargs: dict[str, Any]):
        self._factorize(A, **kwargs)

    def _factorize(self, A: wp.array[ScalarType], ftol: float | None = None, **kwargs: dict[str, Any]) -> None:
        # Override the current tolerance if provided otherwise ensure
        # it does not exceed machine precision for the current dtype
        if ftol is not None:
            self._ftol = make_dtype_tolerance(ftol, dtype=self._dtype)
        else:
            self._ftol = make_dtype_tolerance(self._ftol, dtype=self._dtype)

        # Factorize the specified matrix data and store any intermediate data
        self._factorize_impl(A, **kwargs)
        self._has_factors = True

    ###
    # Public API
    ###

    def reconstruct(self, A: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        """Reconstructs the original matrix from the current factorization."""
        self._check_has_factorization()
        self._reconstruct_impl(A, **kwargs)


class IterativeSolver(LinearSolver[ScalarType, IndexType]):
    """
    An abstract base class for iterative linear system solvers.
    """

    def __init__(
        self,
        operator: (
            conjugate.BatchedLinearOperator[ScalarType, IndexType]
            | DenseLinearOperatorData[ScalarType, IndexType]
            | BlockSparseLinearOperators[ScalarType, IndexType]
            | None
        ) = None,
        atol: float | wp.array[ScalarType] | None = None,
        rtol: float | wp.array[ScalarType] | None = None,
        dtype: type[ScalarType] = wp.float32,  # type: ignore[assignment]
        device: wp.DeviceLike | None = None,
        maxiter: int | wp.array[wp.int32] | None = None,
        world_active: wp.array[wp.bool] | None = None,
        preconditioner: Any = None,
        loop_granularity: int = 1,
        **kwargs: dict[str, Any],
    ):
        self._maxiter: int | wp.array[wp.int32] | None = maxiter
        self._preconditioner: Any = preconditioner
        self._world_active: wp.array[wp.bool] | None = world_active
        self.atol: float | wp.array[ScalarType] | None = atol
        self.rtol: float | wp.array[ScalarType] | None = rtol

        self._num_worlds: int | None = None
        self._max_dim: int | None = None
        self._batched_operator: conjugate.BatchedLinearOperator[ScalarType, IndexType] | None = None
        self._use_graph_conditionals: bool = kwargs.pop("use_graph_conditionals", True)
        self.loop_granularity = loop_granularity

        # Sparse discovery settings (via kwargs)
        self._discover_sparse: bool = kwargs.pop("discover_sparse", False)
        self._sparse_block_size: int = kwargs.pop("sparse_block_size", 4)
        self._sparse_threshold: float = kwargs.pop("sparse_threshold", 0.5)
        self._sparse_bsm: BlockSparseMatrices[ScalarType, IndexType, Any] | None = None
        self._sparse_operator: conjugate.BatchedLinearOperator[ScalarType, IndexType] | None = None
        self._sparse_solver: Any = None  # Set by concrete classes

        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    def _to_batched_operator(
        self,
        operator: (
            conjugate.BatchedLinearOperator[ScalarType, IndexType]
            | DenseLinearOperatorData[ScalarType, IndexType]
            | BlockSparseLinearOperators[ScalarType, IndexType]
        ),
    ) -> conjugate.BatchedLinearOperator[ScalarType, IndexType]:
        """Convert various operator types to BatchedLinearOperator."""
        if isinstance(operator, conjugate.BatchedLinearOperator):
            return operator
        elif isinstance(operator, DenseLinearOperatorData):
            return conjugate.BatchedLinearOperator.from_dense(operator)
        elif isinstance(operator, BlockSparseLinearOperators):
            # For sparse, need uniform dimensions
            return conjugate.BatchedLinearOperator.from_block_sparse_operator(operator)
        else:
            raise ValueError(f"Unsupported operator type: {type(operator)}")

    @override
    def finalize(
        self,
        operator: (
            conjugate.BatchedLinearOperator[ScalarType, IndexType]
            | DenseLinearOperatorData[ScalarType, IndexType]
            | BlockSparseLinearOperators[ScalarType, IndexType]
        ),
        maxiter: int | wp.array[wp.int32] | None = None,
        world_active: wp.array[wp.bool] | None = None,
        preconditioner: Any = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Ingest a linear operator and allocate any necessary internal memory.

        Accepts BatchedLinearOperator, DenseLinearOperatorData, or BlockSparseMatrices.
        """
        if operator is None:
            raise ValueError("A valid linear operator must be provided!")

        # Update sparse settings from kwargs if provided
        if "discover_sparse" in kwargs:
            self._discover_sparse = kwargs.pop("discover_sparse")
        if "sparse_block_size" in kwargs:
            self._sparse_block_size = kwargs.pop("sparse_block_size")
        if "sparse_threshold" in kwargs:
            self._sparse_threshold = kwargs.pop("sparse_threshold")

        self._batched_operator = self._to_batched_operator(operator)

        if isinstance(operator, DenseLinearOperatorData):
            self._operator = operator
            self._dtype = operator.info.dtype
        else:
            self._operator = None
            self._dtype = self._batched_operator.dtype

        if maxiter is not None:
            self._maxiter = maxiter
        if world_active is not None:
            self._world_active = world_active
        if preconditioner is not None:
            self._preconditioner = preconditioner

        self._num_worlds = self._batched_operator.n_worlds
        self._max_dim = self._batched_operator.max_dim
        self._total_vec_size = self._batched_operator.total_vec_size
        self._solve_iterations: wp.array[wp.int32] | None = None
        self._solve_residual_norm: wp.array[ScalarType] | None = None

        with wp.ScopedDevice(self._device):
            if self._world_active is None:
                self._world_active = wp.full(self._num_worlds, True, dtype=wp.bool)
            elif not isinstance(self._world_active, wp.array):
                raise ValueError("The provided world_active is not a valid wp.array!")
            if self._maxiter is None:
                self._maxiter = wp.full(self._num_worlds, self._max_dim, dtype=wp.int32)
            elif isinstance(self._maxiter, int):
                self._maxiter = wp.full(self._num_worlds, self._maxiter, dtype=wp.int32)
            elif not isinstance(self._maxiter, wp.array):
                raise ValueError("The provided maxiter is not a valid wp.array or int!")

        # Allocate block-sparse matrix if sparse discovery is enabled
        if self._discover_sparse:
            if not isinstance(operator, DenseLinearOperatorData):
                raise ValueError("discover_sparse requires a DenseLinearOperatorData operator.")
            self._sparse_bsm = allocate_block_sparse_from_dense(
                dense_op=operator,
                block_size=self._sparse_block_size,
                sparsity_threshold=self._sparse_threshold,
                device=self._device,
            )
            # Create sparse operator (will be populated at compute time)
            self._sparse_operator = conjugate.BatchedLinearOperator.from_block_sparse(
                self._sparse_bsm, self._batched_operator.active_dims
            )

        self._allocate_impl(operator, **kwargs)

    @override
    def solve(
        self, b: wp.array[ScalarType], x: wp.array[ScalarType], zero_x: bool = False, **kwargs: dict[str, Any]
    ) -> None:
        """Solves the multi-linear systems `A @ x = b`."""
        if self._operator is not None:
            if not self._operator.info.is_rhs_compatible(b):
                raise ValueError("The provided flat rhs vector data array does not have enough memory!")
            if not self._operator.info.is_input_compatible(x):
                raise ValueError("The provided flat input vector data array does not have enough memory!")
        if zero_x:
            x.zero_()
        self._solve_impl(b=b, x=x, **kwargs)

    def get_solve_metadata(self) -> dict[str, Any]:
        return {"iterations": self._solve_iterations, "residual_norm": self._solve_residual_norm}

    def prepare_step(self) -> None:
        """Hook run once per sim step before the (possibly graph-captured) solve loop.

        Raw-Jacobian solvers override this to rebuild their per-step index structures outside the
        captured loop. The default is a no-op so callers can invoke it unconditionally.
        """

    def _update_sparse_bsm(self) -> None:
        """Updates the block-sparse matrix from the dense operator. Called during compute()."""
        if self._discover_sparse and self._sparse_bsm is not None and self._operator is not None:
            dense_to_block_sparse_copy_values(
                dense_op=self._operator,
                bsm=self._sparse_bsm,
                block_size=self._sparse_block_size,
            )


###
# Direct solvers
###


class LLTSequentialSolver(DirectSolver[ScalarType, IndexType]):
    """
    A LLT (i.e. Cholesky) factorization class computing each matrix block sequentially.
    This parallelizes the factorization and solve operations over each block
    and supports heterogeneous matrix block sizes.
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData[ScalarType, IndexType] | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: type[ScalarType] = wp.float32,  # type: ignore[assignment]
        device: wp.DeviceLike | None = None,
        **kwargs: dict[str, Any],
    ):
        # Declare LLT-specific internal data
        self._L: wp.array[ScalarType] | None = None
        """A flat array containing the Cholesky factorization of each matrix block."""
        self._y: wp.array[ScalarType] | None = None
        """A flat array containing the intermediate results for the solve operation."""

        # Initialize base class members
        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    ###
    # Properties
    ###

    @property
    def L(self) -> wp.array[ScalarType]:
        if self._L is None:
            raise ValueError("The factorization array has not been allocated!")
        return self._L

    @property
    def y(self) -> wp.array[ScalarType]:
        if self._y is None:
            raise ValueError("The intermediate result array has not been allocated!")
        return self._y

    ###
    # Implementation
    ###

    @override
    def _allocate_impl(self, A: DenseLinearOperatorData[ScalarType, IndexType], **kwargs: dict[str, Any]) -> None:
        # Check the operator has info
        if A.info is None:
            raise ValueError("The provided operator does not have any associated info!")

        # Ensure that the underlying operator is compatible with LLT
        if not isinstance(A.info, DenseSquareMultiLinearInfo):
            raise ValueError("LLT factorization requires a square matrix.")

        # Allocate the Cholesky factorization matrix and the
        # intermediate result buffer on the specified device
        with wp.ScopedDevice(self._device):
            self._L = wp.zeros(shape=(self._operator.info.total_mat_size,), dtype=self._dtype)
            self._y = wp.zeros(shape=(self._operator.info.total_vec_size,), dtype=self._dtype)

    @override
    def _reset_impl(self) -> None:
        self._L.zero_()
        self._y.zero_()
        self._has_factors = False

    @override
    def _factorize_impl(self, A: wp.array[ScalarType]) -> None:
        factorize.llt_sequential_factorize(
            num_blocks=self._operator.info.num_blocks,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            A=A,
            L=self._L,
        )

    @override
    def _reconstruct_impl(self, A: wp.array[ScalarType]) -> None:
        raise NotImplementedError("LLT matrix reconstruction is not yet implemented.")

    @override
    def _solve_impl(self, b: wp.array[ScalarType], x: wp.array[ScalarType]) -> None:
        # Solve the system L * y = b and L^T * x = y
        factorize.llt_sequential_solve(
            num_blocks=self._operator.info.num_blocks,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            vio=self._operator.info.vio,
            L=self._L,
            b=b,
            y=self._y,
            x=x,
        )

    @override
    def _solve_inplace_impl(self, x: wp.array[ScalarType]) -> None:
        # Solve the system L * y = x and L^T * x = y
        factorize.llt_sequential_solve_inplace(
            num_blocks=self._operator.info.num_blocks,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            vio=self._operator.info.vio,
            L=self._L,
            x=x,
        )


class LLTBlockedSolver(DirectSolver[ScalarType, IndexType]):
    """
    A Blocked LLT (i.e. Cholesky) factorization class computing each matrix block with Tile-based parallelism.
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData[ScalarType, IndexType] | None = None,
        block_size: int = 32,
        solve_block_dim: int = 128,
        factortize_block_dim: int = 128,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: type[ScalarType] = wp.float32,  # type: ignore[assignment]
        device: wp.DeviceLike | None = None,
        **kwargs: dict[str, Any],
    ):
        # Declare LLT-specific internal data
        self._L: wp.array[ScalarType] | None = None
        """A flat array containing the Cholesky factorization of each matrix block."""
        self._y: wp.array[ScalarType] | None = None
        """A flat array containing the intermediate results for the solve operation."""

        # Cache the fixed block size
        self._block_size: int = block_size
        self._solve_block_dim: int = solve_block_dim
        self._factortize_block_dim: int = factortize_block_dim

        # Create the factorization and solve kernels
        self._factorize_kernel = factorize.make_llt_blocked_factorize_kernel(block_size)
        self._solve_kernel = factorize.make_llt_blocked_solve_kernel(block_size)
        self._solve_inplace_kernel = factorize.make_llt_blocked_solve_inplace_kernel(block_size)

        # Initialize base class members
        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    ###
    # Properties
    ###

    @property
    def L(self) -> wp.array[ScalarType]:
        if self._L is None:
            raise ValueError("The factorization array has not been allocated!")
        return self._L

    @property
    def y(self) -> wp.array[ScalarType]:
        if self._y is None:
            raise ValueError("The intermediate result array has not been allocated!")
        return self._y

    ###
    # Implementation
    ###

    @override
    def _allocate_impl(self, A: DenseLinearOperatorData[ScalarType, IndexType], **kwargs: dict[str, Any]) -> None:
        # Check the operator has info
        if A.info is None:
            raise ValueError("The provided operator does not have any associated info!")

        # Ensure that the underlying operator is compatible with LLT
        if not isinstance(A.info, DenseSquareMultiLinearInfo):
            raise ValueError("LLT factorization requires a square matrix.")

        # Allocate the Cholesky factorization matrix and the
        # intermediate result buffer on the specified device
        with wp.ScopedDevice(self._device):
            self._L = wp.zeros(shape=(self._operator.info.total_mat_size,), dtype=self._dtype)
            self._y = wp.zeros(shape=(self._operator.info.total_vec_size,), dtype=self._dtype)

    @override
    def _reset_impl(self) -> None:
        self._L.zero_()
        self._y.zero_()
        self._has_factors = False

    @override
    def _factorize_impl(self, A: wp.array[ScalarType]) -> None:
        factorize.llt_blocked_factorize(
            kernel=self._factorize_kernel,
            num_blocks=self._operator.info.num_blocks,
            block_dim=self._factortize_block_dim,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            A=A,
            L=self._L,
        )

    @override
    def _reconstruct_impl(self, A: wp.array[ScalarType]) -> None:
        raise NotImplementedError("LLT matrix reconstruction is not yet implemented.")

    @override
    def _solve_impl(self, b: wp.array[ScalarType], x: wp.array[ScalarType]) -> None:
        # Solve the system L * y = b and L^T * x = y
        factorize.llt_blocked_solve(
            kernel=self._solve_kernel,
            num_blocks=self._operator.info.num_blocks,
            block_dim=self._solve_block_dim,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            vio=self._operator.info.vio,
            L=self._L,
            b=b,
            y=self._y,
            x=x,
        )

    @override
    def _solve_inplace_impl(self, x: wp.array[ScalarType]) -> None:
        # Solve the system L * y = b and L^T * x = y
        factorize.llt_blocked_solve_inplace(
            kernel=self._solve_inplace_kernel,
            num_blocks=self._operator.info.num_blocks,
            block_dim=self._solve_block_dim,
            dim=self._operator.info.dim,
            mio=self._operator.info.mio,
            vio=self._operator.info.vio,
            L=self._L,
            y=self._y,
            x=x,
        )


###
# Iterative solvers
###


class ConjugateGradientSolver(IterativeSolver[ScalarType, IndexType]):
    """
    A wrapper around the batched Conjugate Gradient implementation in `conjugate.cg`.

    This solves multiple independent SPD systems using a batched operator.
    """

    def __init__(
        self,
        **kwargs: dict[str, Any],
    ):
        self._Mi: conjugate.BatchedLinearOperator[ScalarType, IndexType] | None = None
        self._jacobi_preconditioner: wp.array[ScalarType] | None = None
        self.solver: conjugate.CGSolver[ScalarType, IndexType] | None = None
        super().__init__(**kwargs)

    @override
    def _allocate_impl(self, operator, **kwargs: dict[str, Any]) -> None:
        # Validate square operator for dense case
        if isinstance(operator, DenseLinearOperatorData):
            if not isinstance(operator.info, DenseSquareMultiLinearInfo):
                raise ValueError("ConjugateGradientSolver requires a square matrix operator.")

        if self._preconditioner == "jacobi":
            self._jacobi_preconditioner = wp.zeros(
                shape=(self._total_vec_size,), dtype=self._dtype, device=self._device
            )
            self._Mi = conjugate.BatchedLinearOperator.from_diagonal(
                self._jacobi_preconditioner,
                self._batched_operator.active_dims,
                self._batched_operator.vio,
                self._max_dim,
            )
        elif self._preconditioner is not None:
            raise ValueError(f"Unsupported preconditioner: {self._preconditioner}.")
        else:
            self._Mi = None

        self.solver = conjugate.CGSolver(
            A=self._batched_operator,
            world_active=self._world_active,
            atol=self.atol,
            rtol=self.rtol,
            maxiter=self._maxiter,
            Mi=self._Mi,
            callback=None,
            use_cuda_graph=True,
            use_graph_conditionals=self._use_graph_conditionals,
            loop_granularity=self.loop_granularity,
        )

        if self._discover_sparse and self._sparse_operator is not None:
            self._sparse_solver = conjugate.CGSolver(
                A=self._sparse_operator,
                world_active=self._world_active,
                atol=self.atol,
                rtol=self.rtol,
                maxiter=self._maxiter,
                Mi=self._Mi,
                callback=None,
                use_cuda_graph=True,
                use_graph_conditionals=self._use_graph_conditionals,
                loop_granularity=self.loop_granularity,
            )

    @override
    def _reset_impl(self, A: wp.array[ScalarType] | None = None, **kwargs: dict[str, Any]) -> None:
        if self._jacobi_preconditioner is not None:
            self._jacobi_preconditioner.zero_()
        self._solve_iterations: wp.array[wp.int32] | None = None
        self._solve_residual_norm: wp.array[ScalarType] | None = None

    @override
    def _compute_impl(self, A: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        if self._operator is not None and A.ptr != self._operator.mat.ptr:
            raise ValueError(f"{self.__class__.__name__} cannot be re-used with a different matrix.")
        if self._Mi is not None:
            self._update_preconditioner()
        self._update_sparse_bsm()

    @override
    def _solve_inplace_impl(self, x: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        self._solve_impl(x, x, **kwargs)

    @override
    def _solve_impl(self, b: wp.array[ScalarType], x: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        solver = self._sparse_solver or self.solver
        if solver is None:
            raise ValueError("ConjugateGradientSolver.allocate() must be called before solve().")

        self._solve_iterations, self._solve_residual_norm, _ = solver.solve(
            b=b,
            x=x,
        )

    def _update_preconditioner(self):
        if self._operator is None:
            raise ValueError("Jacobi preconditioner requires a DenseLinearOperatorData operator.")
        wp.launch(
            conjugate.make_jacobi_preconditioner,
            dim=(self._num_worlds, self._max_dim),
            inputs=[
                self._operator.mat,
                self._batched_operator.active_dims,
                self._operator.info.maxdim,
                self._operator.info.mio,
                self._operator.info.vio,
            ],
            outputs=[self._jacobi_preconditioner],
            device=self._device,
        )


class ConjugateResidualSolver(IterativeSolver[ScalarType, IndexType]):
    """
    A wrapper around the batched Conjugate Residual implementation in `conjugate.cr`.

    This solves multiple independent SPD systems using a batched operator.
    """

    def __init__(
        self,
        **kwargs: dict[str, Any],
    ):
        self._Mi: conjugate.BatchedLinearOperator[ScalarType, IndexType] | None = None
        self._jacobi_preconditioner: wp.array[ScalarType] | None = None
        self.solver: conjugate.CRSolver[ScalarType, IndexType] | None = None
        super().__init__(**kwargs)

    @override
    def _allocate_impl(self, operator, **kwargs: dict[str, Any]) -> None:
        if isinstance(operator, DenseLinearOperatorData):
            if not isinstance(operator.info, DenseSquareMultiLinearInfo):
                raise ValueError("ConjugateResidualSolver requires a square matrix operator.")

        if self._preconditioner == "jacobi":
            self._jacobi_preconditioner = wp.zeros(
                shape=(self._total_vec_size,), dtype=self._dtype, device=self._device
            )
            self._Mi = conjugate.BatchedLinearOperator.from_diagonal(
                self._jacobi_preconditioner,
                self._batched_operator.active_dims,
                self._batched_operator.vio,
                self._max_dim,
            )
        elif self._preconditioner is not None:
            raise ValueError(f"Unsupported preconditioner: {self._preconditioner}.")
        else:
            self._Mi = None

        self.solver = conjugate.CRSolver(
            A=self._batched_operator,
            world_active=self._world_active,
            atol=self.atol,
            rtol=self.rtol,
            maxiter=self._maxiter,
            Mi=self._Mi,
            callback=None,
            use_cuda_graph=True,
            use_graph_conditionals=self._use_graph_conditionals,
            loop_granularity=self.loop_granularity,
        )

        if self._discover_sparse and self._sparse_operator is not None:
            self._sparse_solver = conjugate.CRSolver(
                A=self._sparse_operator,
                world_active=self._world_active,
                atol=self.atol,
                rtol=self.rtol,
                maxiter=self._maxiter,
                Mi=self._Mi,
                callback=None,
                use_cuda_graph=True,
                use_graph_conditionals=self._use_graph_conditionals,
                loop_granularity=self.loop_granularity,
            )

    @override
    def _reset_impl(self, A: wp.array[ScalarType] | None = None, **kwargs: dict[str, Any]) -> None:
        if self._jacobi_preconditioner is not None:
            self._jacobi_preconditioner.zero_()
        self._solve_iterations: wp.array[wp.int32] | None = None
        self._solve_residual_norm: wp.array[ScalarType] | None = None

    @override
    def _compute_impl(self, A: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        if self._operator is not None and A.ptr != self._operator.mat.ptr:
            raise ValueError(f"{self.__class__.__name__} cannot be re-used with a different matrix.")
        if self._Mi is not None:
            self._update_preconditioner()
        self._update_sparse_bsm()

    @override
    def _solve_inplace_impl(self, x: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        self._solve_impl(x, x)

    @override
    def _solve_impl(self, b: wp.array[ScalarType], x: wp.array[ScalarType], **kwargs: dict[str, Any]) -> None:
        solver = self._sparse_solver or self.solver
        if solver is None:
            raise ValueError("ConjugateResidualSolver.allocate() must be called before solve().")

        self._solve_iterations, self._solve_residual_norm, _ = solver.solve(
            b=b,
            x=x,
        )

    def _update_preconditioner(self):
        if self._operator is None:
            raise ValueError("Jacobi preconditioner requires a DenseLinearOperatorData operator.")
        wp.launch(
            conjugate.make_jacobi_preconditioner,
            dim=(self._num_worlds, self._max_dim),
            inputs=[
                self._operator.mat,
                self._batched_operator.active_dims,
                self._operator.info.maxdim,
                self._operator.info.mio,
                self._operator.info.vio,
            ],
            outputs=[self._jacobi_preconditioner],
            device=self._device,
        )


class ConjugateResidualSolverFused(IterativeSolver):
    """Single-kernel sparse Conjugate Residual solver for the matrix-free Delassus operator.

    Unlike :class:`ConjugateResidualSolver` (a multi-launch CR loop over the operator's ``matvec``),
    this runs the whole CR iteration -- the ``A = P J M⁻¹ Jᵀ P + diag(eta)`` products and all vector
    updates -- in one Warp tile-block per world, reading the raw constraint Jacobian directly and
    applying ``P`` and ``M⁻¹`` on the fly (no ``P J M⁻¹`` or ``(P J)ᵀ`` value copy is materialized;
    the only auxiliary data are ``int32`` index arrays).

    Requires a :class:`.delassus.BlockSparseMatrixFreeDelassusOperator`.
    """

    # Signals the matrix-free operator to skip assembling its P·J·M⁻¹ / column-major copies, since
    # this solver consumes the raw Jacobian directly (see BlockSparseMatrixFreeDelassusOperator).
    uses_raw_jacobian: bool = True

    def __init__(self, block_dim: int = 128, **kwargs: dict[str, Any]):
        self._delassus_op = None
        # Threads per block (one block solves one world). 128 (four warps) is the measured sweet
        # spot for the dr_legs workload: faster than 64 (more warps hide matvec memory latency) and
        # than 256 (which over-subscribes and loses occupancy).
        self._block_dim = int(block_dim)
        super().__init__(**kwargs)

    @override
    def finalize(self, operator, maxiter=None, world_active=None, preconditioner=None, **kwargs) -> None:
        # Skip IterativeSolver.finalize's batched-operator construction (which would require the
        # operator's assembled block-sparse copy): this solver needs only the raw operator.
        from ..dynamics.delassus import BlockSparseMatrixFreeDelassusOperator  # noqa: PLC0415

        if not isinstance(operator, BlockSparseMatrixFreeDelassusOperator):
            raise ValueError("ConjugateResidualSolverFused requires a BlockSparseMatrixFreeDelassusOperator.")

        self._delassus_op = operator
        self._operator = None
        self._dtype = wp.float32
        if maxiter is not None:
            self._maxiter = maxiter
        if world_active is not None:
            self._world_active = world_active
        if preconditioner is not None:
            self._preconditioner = preconditioner

        n_worlds = operator._model.size.num_worlds
        with wp.ScopedDevice(operator.device):
            if self._world_active is None:
                self._world_active = wp.full(n_worlds, True, dtype=wp.bool)
            if self._maxiter is None:
                self._maxiter = wp.full(n_worlds, int(operator._model.size.max_of_max_total_cts), dtype=wp.int32)
            elif isinstance(self._maxiter, int):
                self._maxiter = wp.full(n_worlds, self._maxiter, dtype=wp.int32)

        self._solve_iterations = None
        self._solve_residual_norm = None
        self._allocate_impl(operator, **kwargs)

    @override
    def _allocate_impl(self, operator, **kwargs: dict[str, Any]) -> None:
        op = operator
        model = op._model
        device = op.device
        self._device = device

        n_worlds = model.size.num_worlds
        max_rows = int(model.size.max_of_max_total_cts)
        # Pad the logical row count to a multiple of block_dim so the scatter tile divides evenly.
        self._max_rows = ((max_rows + self._block_dim - 1) // self._block_dim) * self._block_dim
        self._max_cols = 6 * int(model.size.max_of_num_bodies)
        self._max_major_cols = max(1, int(model.size.max_of_num_bodies))
        self._total_rows = int(model.size.sum_of_max_total_cts)

        cj = op.constraint_jacobian
        self._total_nnz = int(cj.nzb_values.shape[0])
        self._max_of_num_nzb = int(cj.max_of_num_nzb)

        self._kernel = make_fused_cr_kernel(self._max_rows, self._max_cols, MAX_BLOCKS_PER_ROW, self._block_dim)

        with wp.ScopedDevice(device):
            self._row_blk = wp.empty((self._total_rows, MAX_BLOCKS_PER_ROW), dtype=wp.int32)
            self._slot_count = wp.empty((self._total_rows,), dtype=wp.int32)
            self._sort_key = wp.empty((max(2 * self._total_nnz, 2),), dtype=wp.int32)
            self._sort_val = wp.empty((max(2 * self._total_nnz, 2),), dtype=wp.int32)
            # Column-sorted block rows for the transpose gather, refilled each step (see
            # build_transpose_index): lets the transpose hot loop read row_idx_sorted[idx]
            # sequentially instead of chasing the scattered block id nzb_coords[sort_val[idx], 0].
            self._row_idx_sorted = wp.empty((max(2 * self._total_nnz, 2),), dtype=wp.int32)
            self._seg_end = wp.empty((n_worlds,), dtype=wp.int32)
            self._cursor = wp.empty((n_worlds, self._max_major_cols), dtype=wp.int32)
            self._iters = wp.zeros((n_worlds,), dtype=wp.int32)
            self._resid = wp.zeros((n_worlds,), dtype=wp.float32)
            bodies_offset = model.info.bodies_offset.numpy()
            self._nbd = wp.array((np.diff(bodies_offset) * 6).astype(np.int32), dtype=wp.int32)
            self._precond_dummy = wp.zeros((1,), dtype=wp.float32)
        # Combined regularization, refreshed together with the index structures (see _solve_impl).
        self._cached_eta = self._precond_dummy

        # Per-world stopping controls are read in the kernel as device arrays so per-world or
        # PADMM-adaptive (``linear_solver_atol``) tolerances are honored rather than scalarized.
        # ``self.atol``/``self.rtol`` may be a wp.array (used directly, contents read live each solve),
        # a scalar, or None; scalars/None materialize into these cached fallback arrays (see _solve_impl).
        self._atol_arr = wp.full(n_worlds, 1e-8, dtype=wp.float32, device=device)
        self._rtol_arr = wp.full(n_worlds, 1e-8, dtype=wp.float32, device=device)

    @override
    def _reset_impl(self, A: wp.array | None = None, **kwargs: dict[str, Any]) -> None:
        self._solve_iterations = None
        self._solve_residual_norm = None

    @override
    def _compute_impl(self, A: wp.array | None = None, **kwargs: dict[str, Any]) -> None:
        # Matrix-free: nothing to precompute; the index structures are rebuilt per solve.
        pass

    @override
    def _solve_inplace_impl(self, x: wp.array, **kwargs: dict[str, Any]) -> None:
        self._solve_impl(x, x, **kwargs)

    def _refresh_combined_regularization(self, op) -> wp.array:
        # Mirror BlockSparseMatrixFreeDelassusOperator.update()'s regularization step (eta plus
        # armature) without assembling any Jacobian copy. Uses the raw Jacobian's row_start.
        if op._combined_regularization is None:
            return op._eta

        from ..dynamics.delassus import (  # noqa: PLC0415
            _add_armature_regularization_preconditioned_sparse,
            _add_armature_regularization_sparse,
        )

        model = op._model
        data = op._data
        device = op.device
        row_start = op.constraint_jacobian.row_start
        if op._eta is not None:
            wp.copy(op._combined_regularization, op._eta)
        else:
            op._combined_regularization.zero_()

        if op._preconditioner is None:
            wp.launch(
                _add_armature_regularization_sparse,
                dim=(op.num_matrices, model.size.max_of_num_dynamic_joint_cts),
                inputs=[
                    model.info.num_joint_dynamic_cts,
                    model.info.joint_dynamic_cts_offset,
                    row_start,
                    data.joints.inv_m_j,
                ],
                outputs=[op._combined_regularization],
                device=device,
            )
        else:
            wp.launch(
                _add_armature_regularization_preconditioned_sparse,
                dim=(op.num_matrices, model.size.max_of_num_dynamic_joint_cts),
                inputs=[
                    model.info.num_joint_dynamic_cts,
                    model.info.joint_dynamic_cts_offset,
                    data.joints.inv_m_j,
                    row_start,
                    op._preconditioner,
                ],
                outputs=[op._combined_regularization],
                device=device,
            )
        return op._combined_regularization

    def prepare_step(self) -> None:
        """Rebuild the per-step index structures and combined regularization, if the operator was
        marked changed since the last build.

        The index structures and combined regularization depend only on the Jacobian sparsity and
        the regularization -- both fixed across a sim step's PADMM iterations -- so this only does
        work once per step (when ``_raw_jacobian_dirty`` is set). PADMM calls this *before* the
        (possibly CUDA-graph-captured) iteration loop, keeping the build (which includes a segmented
        sort) out of the replayed graph; ``_solve_impl`` also calls it as a lazy fallback.
        """
        op = self._delassus_op
        if not (op._raw_jacobian_dirty or op._regularization_dirty):
            return
        device = op.device
        cj = op.constraint_jacobian  # raw constraint Jacobian J (coordinate block-sparse)

        # Regularization (eta) refresh is needed on either flag; it depends only on eta / armature /
        # preconditioner, not on the Jacobian sparsity, so it never needs the index structures.
        eta = self._refresh_combined_regularization(op)
        self._cached_eta = eta if eta is not None else self._precond_dummy
        op._regularization_dirty = False

        # The index structures depend only on the Jacobian sparsity, so rebuild them (the segmented
        # transpose sort is the expensive part) only when the structure actually changed.
        if not op._raw_jacobian_dirty:
            return
        build_row_index(
            num_nzb=cj.num_nzb,
            nzb_start=cj.nzb_start,
            nzb_coords=cj.nzb_coords,
            row_offset=op._info.vio,
            total_rows=self._total_rows,
            max_of_num_nzb=self._max_of_num_nzb,
            out_row_blk=self._row_blk,
            out_slot_count=self._slot_count,
            device=device,
        )
        build_transpose_index(
            num_nzb=cj.num_nzb,
            nzb_start=cj.nzb_start,
            nzb_coords=cj.nzb_coords,
            total_nnz=self._total_nnz,
            max_of_num_nzb=self._max_of_num_nzb,
            max_major_cols=self._max_major_cols,
            out_sort_key=self._sort_key,
            out_sort_val=self._sort_val,
            out_seg_end=self._seg_end,
            out_cursor=self._cursor,
            out_row_idx_sorted=self._row_idx_sorted,
            device=device,
        )
        op._raw_jacobian_dirty = False

    @override
    def _solve_impl(self, b: wp.array, x: wp.array, **kwargs: dict[str, Any]) -> None:
        op = self._delassus_op
        model = op._model
        data = op._data
        device = op.device
        cj = op.constraint_jacobian  # raw constraint Jacobian J (coordinate block-sparse)

        # Lazy fallback: build once per step if PADMM hasn't already done so before the loop.
        self.prepare_step()

        eta = self._cached_eta
        use_precond = 1 if op._preconditioner is not None else 0
        precond = op._preconditioner if op._preconditioner is not None else self._precond_dummy

        # Resolve per-world stopping controls to device arrays. A wp.array (e.g. PADMM's adaptive
        # ``linear_solver_atol``) is used directly so the kernel reads its live contents; a scalar
        # is written into the cached fallback array; None keeps the 1e-8 default.
        maxiter = self._maxiter
        atol = self.atol if isinstance(self.atol, wp.array) else self._atol_arr
        rtol = self.rtol if isinstance(self.rtol, wp.array) else self._rtol_arr
        if isinstance(self.atol, (int, float)):
            self._atol_arr.fill_(float(self.atol))
        if isinstance(self.rtol, (int, float)):
            self._rtol_arr.fill_(float(self.rtol))

        wp.launch_tiled(
            self._kernel,
            dim=op.num_matrices,
            inputs=[
                op._info.dim,
                self._nbd,
                op._info.vio,
                op._info.vio,
                self._world_active,
                cj.num_nzb,
                cj.nzb_start,
                cj.nzb_coords,
                cj.nzb_values,
                self._row_blk,
                self._sort_key,
                self._sort_val,
                self._row_idx_sorted,
                self._cursor,
                model.bodies.inv_m_i,
                data.bodies.inv_I_i,
                model.info.bodies_offset,
                precond,
                use_precond,
                eta,
                b,
                x,
                maxiter,
                atol,
                rtol,
            ],
            outputs=[self._iters, self._resid],
            block_dim=self._block_dim,
            device=device,
        )
        self._solve_iterations = self._iters
        self._solve_residual_norm = self._resid


###
# Summary
###

LinearSolverType = (
    LLTSequentialSolver
    | LLTBlockedSolver
    | ConjugateGradientSolver
    | ConjugateResidualSolver
    | ConjugateResidualSolverFused
)
"""Type alias over all linear solvers."""

LinearSolverTypeToName = {
    LLTSequentialSolver: "LLTS",
    LLTBlockedSolver: "LLTB",
    ConjugateGradientSolver: "CG",
    ConjugateResidualSolver: "CR",
    ConjugateResidualSolverFused: "CRF",
}

LinearSolverNameToType = {
    "LLTS": LLTSequentialSolver,
    "LLTB": LLTBlockedSolver,
    "CG": ConjugateGradientSolver,
    "CR": ConjugateResidualSolver,
    "CRF": ConjugateResidualSolverFused,
}
