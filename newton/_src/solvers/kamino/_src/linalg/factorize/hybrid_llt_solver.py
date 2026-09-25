# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-block dispatcher for dense and semi-sparse Cholesky kernels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ......core.types import override
from .. import factorize
from ..core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from ..linear import LLTBlockedSolver
from .llt_blocked_rcm_solver import LLTBlockedRCMSolver, _fixed_symbolic_data

__all__ = ["HybridLLTBlockedSolver"]


@wp.kernel(enable_backward=False)
def _gather_block_dimensions(
    block_indices: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    gathered_dimensions: wp.array[wp.int32],
):
    block = wp.tid()
    gathered_dimensions[block] = dimensions[block_indices[block]]


@wp.kernel(enable_backward=False)
def _gather_block_info(
    block_indices: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    gathered_dimensions: wp.array[wp.int32],
    gathered_matrix_offsets: wp.array[wp.int32],
    gathered_vector_offsets: wp.array[wp.int32],
):
    block = wp.tid()
    source = block_indices[block]
    gathered_dimensions[block] = dimensions[source]
    gathered_matrix_offsets[block] = matrix_offsets[source]
    gathered_vector_offsets[block] = vector_offsets[source]


@wp.kernel(enable_backward=False)
def _initialize_identity_permutations(
    dimensions: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    inverse_permutation: wp.array[wp.int32],
):
    block, index = wp.tid()
    if index < dimensions[block]:
        offset = vector_offsets[block] + index
        permutation[offset] = index
        inverse_permutation[offset] = index


@wp.kernel(enable_backward=False)
def _scatter_subset_permutations(
    block_indices: wp.array[wp.int32],
    source_max_dimensions: wp.array[wp.int32],
    source_vector_offsets: wp.array[wp.int32],
    compact_vector_offsets: wp.array[wp.int32],
    compact_permutation: wp.array[wp.int32],
    compact_inverse_permutation: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    inverse_permutation: wp.array[wp.int32],
):
    block, index = wp.tid()
    source = block_indices[block]
    if index < source_max_dimensions[source]:
        compact_offset = compact_vector_offsets[block] + index
        source_offset = source_vector_offsets[source] + index
        permutation[source_offset] = compact_permutation[compact_offset]
        inverse_permutation[source_offset] = compact_inverse_permutation[compact_offset]


class _SubsetLLTBlockedRCMSolver(LLTBlockedRCMSolver):
    """Apply the existing RCM implementation to selected packed blocks."""

    def __init__(
        self,
        operator: DenseLinearOperatorData,
        block_indices: Sequence[int],
        symbolic_adjacency: Sequence[Sequence[Sequence[int]]],
        shared_factor: wp.array,
        shared_intermediate: wp.array,
        **kwargs: Any,
    ):
        info = operator.info
        if not isinstance(info, DenseSquareMultiLinearInfo):
            raise ValueError("Hybrid RCM factorization requires a square matrix.")
        selected_indices = tuple(int(index) for index in block_indices)
        if not selected_indices:
            raise ValueError("Hybrid RCM block selection must not be empty.")
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("Hybrid RCM block selection must be unique.")
        if any(index < 0 or index >= info.num_blocks for index in selected_indices):
            raise ValueError("Hybrid RCM block selection is out of range.")
        if len(symbolic_adjacency) != info.num_blocks:
            raise ValueError("symbolic_adjacency must contain one graph per matrix block.")

        dimensions_np = np.asarray(info.dimensions, dtype=np.int32)
        selected_dimensions = dimensions_np[np.asarray(selected_indices, dtype=np.int32)].tolist()
        selected_adjacency = [symbolic_adjacency[index] for index in selected_indices]

        subset_info = DenseSquareMultiLinearInfo()
        subset_info.finalize(
            selected_dimensions,
            dtype=info.dtype,
            itype=info.itype,
            device=info.device,
        )
        compact_vector_offsets = subset_info.vio
        subset_info.mio = wp.empty(len(selected_indices), dtype=info.itype, device=info.device)
        subset_info.vio = wp.empty(len(selected_indices), dtype=info.itype, device=info.device)
        subset_operator = DenseLinearOperatorData(info=subset_info, mat=operator.mat)

        self._source_dimensions = info.dim
        self._source_block_indices = wp.array(selected_indices, dtype=wp.int32, device=info.device)
        wp.launch(
            _gather_block_info,
            dim=len(selected_indices),
            inputs=[self._source_block_indices, info.dim, info.mio, info.vio],
            outputs=[subset_info.dim, subset_info.mio, subset_info.vio],
            device=info.device,
        )
        super().__init__(operator=subset_operator, symbolic_adjacency=selected_adjacency, **kwargs)

        with wp.ScopedDevice(info.device):
            self._A_hat = wp.zeros(info.total_mat_size, dtype=info.dtype)
            self._x_hat = wp.zeros(info.total_vec_size, dtype=info.dtype)
            full_permutation = wp.empty(info.total_vec_size, dtype=wp.int32)
            full_inverse = wp.empty(info.total_vec_size, dtype=wp.int32)
        wp.launch(
            _initialize_identity_permutations,
            dim=(info.num_blocks, info.max_dimension),
            inputs=[info.maxdim, info.vio],
            outputs=[full_permutation, full_inverse],
            device=info.device,
        )
        wp.launch(
            _scatter_subset_permutations,
            dim=(len(selected_indices), max(selected_dimensions)),
            inputs=[
                self._source_block_indices,
                info.maxdim,
                info.vio,
                compact_vector_offsets,
                self._P,
                self._inv_P,
            ],
            outputs=[full_permutation, full_inverse],
            device=info.device,
        )
        self._P = full_permutation
        self._inv_P = full_inverse
        self._L = shared_factor
        self._y = shared_intermediate

    @override
    def _factorize_impl(self, A: wp.array) -> None:
        wp.launch(
            _gather_block_dimensions,
            dim=self._source_block_indices.shape[0],
            inputs=[self._source_block_indices, self._source_dimensions],
            outputs=[self._operator.info.dim],
            device=self._device,
        )
        super()._factorize_impl(A)


class HybridLLTBlockedSolver(LLTBlockedSolver):
    """Dispatch independent matrix blocks to an appropriate LLT kernel.

    Small blocks use the sequential kernels, intermediate and structurally
    dense blocks use ordinary tiled LLT, and sufficiently large sparse blocks
    with fixed symbolic adjacency use RCM-reordered semi-sparse tiled LLT.
    All paths retain the original packed offsets and share one factor buffer.
    """

    def __init__(
        self,
        *args: Any,
        sequential_block_size: int = 6,
        rcm_min_dimension: int = 256,
        rcm_max_fill_ratio: float = 0.75,
        symbolic_adjacency: Sequence[Sequence[Sequence[int]]] | None = None,
        rcm_block_size: int = 32,
        rcm_parallel_factorization: bool = True,
        **kwargs: Any,
    ):
        if sequential_block_size < 0:
            raise ValueError("sequential_block_size must be nonnegative.")
        if rcm_min_dimension <= sequential_block_size:
            raise ValueError("rcm_min_dimension must exceed sequential_block_size.")
        if not 0.0 <= rcm_max_fill_ratio <= 1.0:
            raise ValueError("rcm_max_fill_ratio must be between zero and one.")

        self._sequential_block_size = sequential_block_size
        self._rcm_min_dimension = rcm_min_dimension
        self._rcm_max_fill_ratio = rcm_max_fill_ratio
        self._symbolic_adjacency = symbolic_adjacency
        self._rcm_block_size = rcm_block_size
        self._rcm_parallel_factorization = rcm_parallel_factorization
        self._rcm_block_decisions: dict[tuple[int, int], bool] = {}
        self._sequential_block_indices_host: tuple[int, ...] = ()
        self._tiled_block_indices_host: tuple[int, ...] = ()
        self._rcm_block_indices_host: tuple[int, ...] = ()
        self._sequential_dim: wp.array[wp.int32] | None = None
        self._sequential_mio: wp.array[wp.int32] | None = None
        self._sequential_vio: wp.array[wp.int32] | None = None
        self._sequential_block_indices: wp.array[wp.int32] | None = None
        self._tiled_dim: wp.array[wp.int32] | None = None
        self._tiled_mio: wp.array[wp.int32] | None = None
        self._tiled_vio: wp.array[wp.int32] | None = None
        self._tiled_block_indices: wp.array[wp.int32] | None = None
        self._rcm_solver: _SubsetLLTBlockedRCMSolver | None = None
        super().__init__(*args, **kwargs)

    @property
    def block_size(self) -> int:
        """Return the tile size used by the ordinary solve kernels."""
        return self._solve_block_size

    @property
    def sequential_block_indices(self) -> tuple[int, ...]:
        """Return blocks assigned to sequential LLT kernels."""
        return self._sequential_block_indices_host

    @property
    def tiled_block_indices(self) -> tuple[int, ...]:
        """Return blocks assigned to dense tiled LLT kernels."""
        return self._tiled_block_indices_host

    @property
    def rcm_block_indices(self) -> tuple[int, ...]:
        """Return blocks assigned to RCM semi-sparse LLT kernels."""
        return self._rcm_block_indices_host

    @property
    def uses_reordering(self) -> bool:
        """Return whether any block uses an RCM permutation."""
        return self._rcm_solver is not None

    @property
    def P(self) -> wp.array[wp.int32]:
        """Return packed permutations, including identity on non-RCM blocks."""
        if self._rcm_solver is None:
            raise ValueError("No blocks use an RCM permutation.")
        return self._rcm_solver.P

    @property
    def inv_P(self) -> wp.array[wp.int32]:
        """Return packed inverse permutations, including identity elsewhere."""
        if self._rcm_solver is None:
            raise ValueError("No blocks use an RCM permutation.")
        return self._rcm_solver.inv_P

    @override
    def _allocate_impl(self, A: DenseLinearOperatorData, **kwargs: Any) -> None:
        super()._allocate_impl(A, **kwargs)
        info = self._operator.info
        dimensions = info.dimensions
        if dimensions is None:
            raise ValueError("Hybrid LLT requires host-side block dimensions.")
        if self._symbolic_adjacency is not None and len(self._symbolic_adjacency) != info.num_blocks:
            raise ValueError("symbolic_adjacency must contain one graph per matrix block.")

        dimensions_np = np.asarray(dimensions, dtype=np.int32)
        sequential_mask = dimensions_np <= self._sequential_block_size
        sequential_indices = np.flatnonzero(sequential_mask).astype(np.int32).tolist()
        tiled_indices = np.flatnonzero(~sequential_mask).astype(np.int32).tolist()
        rcm_indices = []
        if self._symbolic_adjacency is not None:
            rcm_indices = []
            retained_tiled_indices = []
            for block in tiled_indices:
                if self._is_rcm_block(block, dimensions[block]):
                    rcm_indices.append(block)
                else:
                    retained_tiled_indices.append(block)
            tiled_indices = retained_tiled_indices

        self._sequential_block_indices_host = tuple(sequential_indices)
        self._tiled_block_indices_host = tuple(tiled_indices)
        self._rcm_block_indices_host = tuple(rcm_indices)
        with wp.ScopedDevice(self._device):
            if sequential_indices:
                self._sequential_block_indices = wp.array(sequential_indices, dtype=wp.int32)
                self._sequential_dim = wp.empty(len(sequential_indices), dtype=wp.int32)
                self._sequential_mio = wp.empty(len(sequential_indices), dtype=wp.int32)
                self._sequential_vio = wp.empty(len(sequential_indices), dtype=wp.int32)
            if tiled_indices:
                self._tiled_block_indices = wp.array(tiled_indices, dtype=wp.int32)
                self._tiled_dim = wp.empty(len(tiled_indices), dtype=wp.int32)
                self._tiled_mio = wp.empty(len(tiled_indices), dtype=wp.int32)
                self._tiled_vio = wp.empty(len(tiled_indices), dtype=wp.int32)

        if sequential_indices:
            wp.launch(
                _gather_block_info,
                dim=len(sequential_indices),
                inputs=[self._sequential_block_indices, info.dim, info.mio, info.vio],
                outputs=[self._sequential_dim, self._sequential_mio, self._sequential_vio],
                device=self._device,
            )
        if tiled_indices:
            wp.launch(
                _gather_block_info,
                dim=len(tiled_indices),
                inputs=[self._tiled_block_indices, info.dim, info.mio, info.vio],
                outputs=[self._tiled_dim, self._tiled_mio, self._tiled_vio],
                device=self._device,
            )

        if rcm_indices:
            self._rcm_solver = _SubsetLLTBlockedRCMSolver(
                operator=A,
                block_indices=rcm_indices,
                shared_factor=self._L,
                shared_intermediate=self._y,
                block_size=self._rcm_block_size,
                solve_block_dim=self._solve_block_dim,
                factorize_block_dim=self._factorize_block_dim,
                symbolic_adjacency=self._symbolic_adjacency,
                parallel_factorization=self._rcm_parallel_factorization,
                dtype=self._dtype,
                device=self._device,
            )

    def _is_rcm_block(self, block: int, dimension: int) -> bool:
        if self._symbolic_adjacency is None or dimension < self._rcm_min_dimension:
            return False
        adjacency = self._symbolic_adjacency[block]
        cache_key = (dimension, id(adjacency))
        decision = self._rcm_block_decisions.get(cache_key)
        if decision is not None:
            return decision
        _, _, tile_pattern, _ = _fixed_symbolic_data([dimension], [adjacency], self._rcm_block_size)
        tile_count = (dimension + self._rcm_block_size - 1) // self._rcm_block_size
        lower_tile_count = tile_count * (tile_count + 1) // 2
        fill_ratio = sum(tile_pattern) / lower_tile_count
        decision = fill_ratio <= self._rcm_max_fill_ratio
        self._rcm_block_decisions[cache_key] = decision
        return decision

    @override
    def _reset_impl(self) -> None:
        super()._reset_impl()
        if self._rcm_solver is not None:
            self._rcm_solver._reset_impl()

    def _gather_dimensions(self) -> None:
        info = self._operator.info
        if self._sequential_block_indices_host:
            wp.launch(
                _gather_block_dimensions,
                dim=len(self._sequential_block_indices_host),
                inputs=[self._sequential_block_indices, info.dim],
                outputs=[self._sequential_dim],
                device=self._device,
            )
        if self._tiled_block_indices_host:
            wp.launch(
                _gather_block_dimensions,
                dim=len(self._tiled_block_indices_host),
                inputs=[self._tiled_block_indices, info.dim],
                outputs=[self._tiled_dim],
                device=self._device,
            )

    @override
    def _factorize_impl(self, A: wp.array) -> None:
        self._gather_dimensions()
        if self._sequential_block_indices_host:
            factorize.llt_sequential_factorize(
                num_blocks=len(self._sequential_block_indices_host),
                dim=self._sequential_dim,
                mio=self._sequential_mio,
                A=A,
                L=self._L,
            )
        if self._tiled_block_indices_host:
            factorize.llt_blocked_factorize(
                kernel=self._factorize_kernel,
                num_blocks=len(self._tiled_block_indices_host),
                block_dim=self._factorize_block_dim,
                dim=self._tiled_dim,
                mio=self._tiled_mio,
                A=A,
                L=self._L,
            )
        if self._rcm_solver is not None:
            self._rcm_solver._factorize_impl(A)

    @override
    def _solve_impl(self, b: wp.array, x: wp.array) -> None:
        if self._sequential_block_indices_host:
            factorize.llt_sequential_solve(
                num_blocks=len(self._sequential_block_indices_host),
                dim=self._sequential_dim,
                mio=self._sequential_mio,
                vio=self._sequential_vio,
                L=self._L,
                b=b,
                y=self._y,
                x=x,
            )
        if self._tiled_block_indices_host:
            factorize.llt_blocked_solve(
                kernel=self._solve_kernel,
                num_blocks=len(self._tiled_block_indices_host),
                block_dim=self._solve_block_dim,
                dim=self._tiled_dim,
                mio=self._tiled_mio,
                vio=self._tiled_vio,
                L=self._L,
                b=b,
                y=self._y,
                x=x,
            )
        if self._rcm_solver is not None:
            self._rcm_solver._solve_impl(b, x)

    @override
    def _solve_inplace_impl(self, x: wp.array) -> None:
        if self._sequential_block_indices_host:
            factorize.llt_sequential_solve_inplace(
                num_blocks=len(self._sequential_block_indices_host),
                dim=self._sequential_dim,
                mio=self._sequential_mio,
                vio=self._sequential_vio,
                L=self._L,
                x=x,
            )
        if self._tiled_block_indices_host:
            factorize.llt_blocked_solve_inplace(
                kernel=self._solve_inplace_kernel,
                num_blocks=len(self._tiled_block_indices_host),
                block_dim=self._solve_block_dim,
                dim=self._tiled_dim,
                mio=self._tiled_mio,
                vio=self._tiled_vio,
                L=self._L,
                y=self._y,
                x=x,
            )
        if self._rcm_solver is not None:
            self._rcm_solver._solve_inplace_impl(x)
