# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-CSR storage for LIMX particle linear systems."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


@wp.kernel
def _block_csr_multiply(
    row_offsets: wp.array[int],
    column_indices: wp.array[int],
    values: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    row = wp.tid()
    value = wp.vec3(0.0)
    for block in range(row_offsets[row], row_offsets[row + 1]):
        value += values[block] * x[column_indices[block]]
    output[row] = value


@wp.kernel
def _update_diagonal(
    values: wp.array[wp.mat33],
    diagonal_indices: wp.array[int],
    diagonal: wp.array[wp.mat33],
):
    row = wp.tid()
    block = diagonal_indices[row]
    if block >= 0:
        diagonal[row] = values[block]
    else:
        diagonal[row] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class BlockCsrMatrix:
    """A square sparse matrix containing 3-by-3 blocks."""

    def __init__(
        self,
        row_offsets: wp.array[int],
        column_indices: wp.array[int],
        values: wp.array[wp.mat33],
        diagonal: wp.array[wp.mat33],
        diagonal_indices: wp.array[int],
        block_indices: dict[tuple[int, int], int],
    ):
        self.row_offsets = row_offsets
        self.column_indices = column_indices
        self.values = values
        self.diagonal = diagonal
        self.diagonal_indices = diagonal_indices
        self._block_indices = block_indices
        self.row_count = len(diagonal)
        self.device = diagonal.device

    def block_index(self, row: int, column: int) -> int:
        """Return the value-array index for a block coordinate."""
        try:
            return self._block_indices[(row, column)]
        except KeyError as error:
            raise ValueError(f"Block coordinate ({row}, {column}) is not present") from error

    def clear_values(self) -> None:
        """Zero all numerical blocks and the cached diagonal."""
        self.values.zero_()
        self.diagonal.zero_()

    def update_diagonal(self) -> None:
        """Refresh cached diagonal blocks from the current numerical values."""
        wp.launch(
            _update_diagonal,
            dim=self.row_count,
            inputs=[self.values, self.diagonal_indices],
            outputs=[self.diagonal],
            device=self.device,
        )

    def multiply(self, x: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
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
            _block_csr_multiply,
            dim=self.row_count,
            inputs=[self.row_offsets, self.column_indices, self.values, x],
            outputs=[output],
            device=self.device,
        )


class BlockCsrBuilder:
    """Accumulate block triplets and finalize them as block-CSR."""

    def __init__(self, row_count: int):
        if row_count <= 0:
            raise ValueError("row_count must be positive")
        self.row_count = row_count
        self._blocks: dict[tuple[int, int], np.ndarray] = {}

    def add_block(self, row: int, column: int, value: wp.mat33) -> None:
        """Add a 3-by-3 block, accumulating duplicate coordinates."""
        self._validate_index(row, column)
        block = np.asarray(value, dtype=np.float32).reshape(3, 3)
        if not np.isfinite(block).all():
            raise ValueError("Block values must be finite")

        key = (row, column)
        if key in self._blocks:
            self._blocks[key] += block
        else:
            self._blocks[key] = block.copy()

    def add_scaled_identity(self, row: int, column: int, scale: float) -> None:
        """Add ``scale * I3`` at a block coordinate."""
        if not np.isfinite(scale):
            raise ValueError("Block scale must be finite")
        self.add_block(row, column, wp.mat33(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale))

    def ensure_block(self, row: int, column: int) -> None:
        """Ensure a block coordinate exists without adding a numerical value."""
        self._validate_index(row, column)
        self._blocks.setdefault((row, column), np.zeros((3, 3), dtype=np.float32))

    def finalize(self, device: Any) -> BlockCsrMatrix:
        """Build sorted CSR arrays on ``device``."""
        sorted_blocks = sorted(self._blocks.items())
        row_offsets = np.zeros(self.row_count + 1, dtype=np.int32)
        column_indices = np.empty(len(sorted_blocks), dtype=np.int32)
        values = np.empty((len(sorted_blocks), 3, 3), dtype=np.float32)
        diagonal = np.zeros((self.row_count, 3, 3), dtype=np.float32)
        diagonal_indices = np.full(self.row_count, -1, dtype=np.int32)
        block_indices: dict[tuple[int, int], int] = {}

        for block_index, ((row, column), value) in enumerate(sorted_blocks):
            row_offsets[row + 1] += 1
            column_indices[block_index] = column
            values[block_index] = value
            block_indices[(row, column)] = block_index
            if row == column:
                diagonal[row] = value
                diagonal_indices[row] = block_index

        np.cumsum(row_offsets, out=row_offsets)
        return BlockCsrMatrix(
            row_offsets=wp.array(row_offsets, dtype=int, device=device),
            column_indices=wp.array(column_indices, dtype=int, device=device),
            values=wp.array(values, dtype=wp.mat33, device=device),
            diagonal=wp.array(diagonal, dtype=wp.mat33, device=device),
            diagonal_indices=wp.array(diagonal_indices, dtype=int, device=device),
            block_indices=block_indices,
        )

    def _validate_index(self, row: int, column: int) -> None:
        if row < 0 or row >= self.row_count or column < 0 or column >= self.row_count:
            raise ValueError(f"Block index ({row}, {column}) is outside a {self.row_count}-row matrix")
