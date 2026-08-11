# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-CSR storage for LIMX affine linear systems."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .affine_types import mat1212, vec12


@wp.kernel
def _block_csr_multiply_12(
    row_offsets: wp.array[int],
    column_indices: wp.array[int],
    values: wp.array[mat1212],
    x: wp.array[vec12],
    output: wp.array[vec12],
):
    row = wp.tid()
    value = vec12(0.0)
    for block in range(row_offsets[row], row_offsets[row + 1]):
        value += values[block] * x[column_indices[block]]
    output[row] = value


@wp.kernel
def _update_diagonal_12(
    values: wp.array[mat1212],
    diagonal_indices: wp.array[int],
    diagonal: wp.array[mat1212],
):
    row = wp.tid()
    block = diagonal_indices[row]
    if block >= 0:
        diagonal[row] = values[block]
    else:
        diagonal[row] = mat1212(0.0)


class BlockCsrMatrix12:
    """A square sparse matrix containing 12-by-12 blocks."""

    def __init__(
        self,
        row_offsets: wp.array[int],
        column_indices: wp.array[int],
        values: wp.array[mat1212],
        diagonal: wp.array[mat1212],
        diagonal_indices: wp.array[int],
        block_keys: np.ndarray,
    ):
        self.row_offsets = row_offsets
        self.column_indices = column_indices
        self.values = values
        self.diagonal = diagonal
        self.diagonal_indices = diagonal_indices
        self._block_keys = block_keys
        self.row_count = len(diagonal)
        self.device = diagonal.device

    def block_index(self, row: int, column: int) -> int:
        """Return the value-array index for a block coordinate."""
        if row < 0 or row >= self.row_count or column < 0 or column >= self.row_count:
            raise ValueError(f"Block coordinate ({row}, {column}) is not present")
        key = row * self.row_count + column
        index = int(np.searchsorted(self._block_keys, key))
        if index >= len(self._block_keys) or self._block_keys[index] != key:
            raise ValueError(f"Block coordinate ({row}, {column}) is not present")
        return index

    def stencil_block_indices(self, stencils: np.ndarray) -> np.ndarray:
        """Return flattened ordered-pair block indices for affine stencils."""
        stencils = np.asarray(stencils, dtype=np.int64)
        if stencils.ndim != 2 or stencils.shape[1] == 0:
            raise ValueError("stencils must have shape [stencil_count, arity]")
        if np.any(stencils < 0) or np.any(stencils >= self.row_count):
            raise ValueError(f"stencils contain an index outside a {self.row_count}-row matrix")
        arity = stencils.shape[1]
        rows = np.repeat(stencils, arity, axis=1)
        columns = np.tile(stencils, (1, arity))
        keys = rows * self.row_count + columns
        indices = np.searchsorted(self._block_keys, keys)
        clipped = np.minimum(indices, max(len(self._block_keys) - 1, 0))
        if (
            len(self._block_keys) == 0
            or np.any(indices >= len(self._block_keys))
            or np.any(self._block_keys[clipped] != keys)
        ):
            raise ValueError("stencil references a block coordinate that is not present")
        return indices.astype(np.int32)

    def clear_values(self) -> None:
        """Zero all numerical blocks and the cached diagonal."""
        self.values.zero_()
        self.diagonal.zero_()

    def update_diagonal(self) -> None:
        """Refresh cached diagonal blocks from the current numerical values."""
        wp.launch(
            _update_diagonal_12,
            dim=self.row_count,
            inputs=[self.values, self.diagonal_indices],
            outputs=[self.diagonal],
            device=self.device,
        )

    def multiply(self, x: wp.array[vec12], output: wp.array[vec12]) -> None:
        """Compute ``output = self * x``.

        Args:
            x: Input vectors, one per block column.
            output: Output vectors, one per block row.
        """
        if len(x) != self.row_count or len(output) != self.row_count:
            raise ValueError(f"Expected {self.row_count} input and output rows")
        if x.device != self.device or output.device != self.device:
            raise ValueError("Matrix, input, and output must use the same device")

        wp.launch(
            _block_csr_multiply_12,
            dim=self.row_count,
            inputs=[self.row_offsets, self.column_indices, self.values, x],
            outputs=[output],
            device=self.device,
        )


class BlockCsrBuilder12:
    """Accumulate 12-by-12 block triplets and finalize them as block-CSR."""

    def __init__(self, row_count: int):
        if row_count <= 0:
            raise ValueError("row_count must be positive")
        self.row_count = row_count
        self._pattern_key_batches: list[np.ndarray] = []
        self._value_keys: list[int] = []
        self._value_blocks: list[np.ndarray] = []

    def add_block(self, row: int, column: int, value: mat1212) -> None:
        """Add a 12-by-12 block, accumulating duplicate coordinates."""
        self._validate_index(row, column)
        block = np.asarray(value, dtype=np.float32)
        if block.shape != (12, 12):
            raise ValueError("Block values must have shape (12, 12)")
        if not np.isfinite(block).all():
            raise ValueError("Block values must be finite")

        self._value_keys.append(row * self.row_count + column)
        self._value_blocks.append(block.copy())

    def ensure_block(self, row: int, column: int) -> None:
        """Ensure a block coordinate exists without adding a numerical value."""
        self._validate_index(row, column)
        self._pattern_key_batches.append(np.asarray([row * self.row_count + column], dtype=np.int64))

    def ensure_stencil_blocks(self, stencils: np.ndarray) -> None:
        """Ensure every ordered affine-body pair for each stencil in one batch."""
        stencils = np.asarray(stencils, dtype=np.int64)
        if stencils.ndim != 2 or stencils.shape[1] == 0:
            raise ValueError("stencils must have shape [stencil_count, arity]")
        if np.any(stencils < 0) or np.any(stencils >= self.row_count):
            raise ValueError(f"stencils contain an index outside a {self.row_count}-row matrix")
        arity = stencils.shape[1]
        rows = np.repeat(stencils, arity, axis=1)
        columns = np.tile(stencils, (1, arity))
        self._pattern_key_batches.append((rows * self.row_count + columns).reshape(-1))

    def finalize(self, device: Any) -> BlockCsrMatrix12:
        """Build sorted CSR arrays on ``device``."""
        key_batches = list(self._pattern_key_batches)
        if self._value_keys:
            key_batches.append(np.asarray(self._value_keys, dtype=np.int64))
        if key_batches:
            block_keys = np.unique(np.concatenate(key_batches))
        else:
            block_keys = np.empty(0, dtype=np.int64)
        rows = block_keys // self.row_count
        columns = block_keys % self.row_count
        row_offsets = np.zeros(self.row_count + 1, dtype=np.int32)
        row_offsets[1:] = np.bincount(rows, minlength=self.row_count).astype(np.int32)
        np.cumsum(row_offsets, out=row_offsets)
        column_indices = columns.astype(np.int32)
        values = np.zeros((len(block_keys), 12, 12), dtype=np.float32)
        if self._value_keys:
            value_indices = np.searchsorted(block_keys, np.asarray(self._value_keys, dtype=np.int64))
            np.add.at(values, value_indices, np.asarray(self._value_blocks, dtype=np.float32))
        diagonal = np.zeros((self.row_count, 12, 12), dtype=np.float32)
        diagonal_indices = np.full(self.row_count, -1, dtype=np.int32)
        diagonal_keys = np.arange(self.row_count, dtype=np.int64) * (self.row_count + 1)
        candidate_diagonal_indices = np.searchsorted(block_keys, diagonal_keys)
        present = candidate_diagonal_indices < len(block_keys)
        present_rows = np.flatnonzero(present)
        present[present_rows] = block_keys[candidate_diagonal_indices[present_rows]] == diagonal_keys[present_rows]
        diagonal_indices[present] = candidate_diagonal_indices[present].astype(np.int32)
        diagonal[present] = values[candidate_diagonal_indices[present]]
        return BlockCsrMatrix12(
            row_offsets=wp.array(row_offsets, dtype=int, device=device),
            column_indices=wp.array(column_indices, dtype=int, device=device),
            values=wp.array(values, dtype=mat1212, device=device),
            diagonal=wp.array(diagonal, dtype=mat1212, device=device),
            diagonal_indices=wp.array(diagonal_indices, dtype=int, device=device),
            block_keys=block_keys,
        )

    def _validate_index(self, row: int, column: int) -> None:
        if row < 0 or row >= self.row_count or column < 0 or column >= self.row_count:
            raise ValueError(f"Block index ({row}, {column}) is outside a {self.row_count}-row matrix")
