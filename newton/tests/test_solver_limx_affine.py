# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.limx.affine_types import mat1212, vec12
from newton._src.solvers.limx.block_csr_12 import BlockCsrBuilder12


class TestAffineBlockCsr(unittest.TestCase):
    def test_sorts_columns_and_accumulates_duplicate_blocks(self):
        """Store one sorted block per coordinate after duplicate additions."""
        builder = BlockCsrBuilder12(2)
        builder.add_block(0, 1, np.eye(12, dtype=np.float32) * 2.0)
        builder.add_block(0, 0, np.eye(12, dtype=np.float32) * 3.0)
        builder.add_block(0, 1, np.eye(12, dtype=np.float32) * -0.5)

        matrix = builder.finalize("cpu")

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 2, 2])
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 1])
        np.testing.assert_allclose(matrix.values.numpy()[0], np.eye(12) * 3.0)
        np.testing.assert_allclose(matrix.values.numpy()[1], np.eye(12) * 1.5)

    def test_resolves_block_indices_after_finalization(self):
        """Resolve finalized coordinates and reject a coordinate absent from the pattern."""
        builder = BlockCsrBuilder12(2)
        builder.ensure_block(0, 1)
        builder.ensure_block(1, 0)

        matrix = builder.finalize("cpu")

        self.assertEqual(matrix.block_index(0, 1), 0)
        self.assertEqual(matrix.block_index(1, 0), 1)
        with self.assertRaises(ValueError):
            matrix.block_index(0, 0)

    def test_extracts_diagonal_and_clears_numerical_blocks(self):
        """Refresh cached diagonal blocks and clear all numerical values."""
        builder = BlockCsrBuilder12(2)
        builder.ensure_block(0, 0)
        builder.ensure_block(0, 1)
        matrix = builder.finalize("cpu")
        values = wp.array(
            [np.eye(12, dtype=np.float32) * 4.0, np.eye(12, dtype=np.float32) * -2.0],
            dtype=mat1212,
            device="cpu",
        )
        wp.copy(matrix.values, values)

        matrix.update_diagonal()

        np.testing.assert_allclose(matrix.diagonal.numpy()[0], np.eye(12) * 4.0)
        np.testing.assert_array_equal(matrix.diagonal.numpy()[1], np.zeros((12, 12)))

        matrix.clear_values()

        np.testing.assert_array_equal(matrix.values.numpy(), np.zeros((2, 12, 12)))
        np.testing.assert_array_equal(matrix.diagonal.numpy(), np.zeros((2, 12, 12)))

    def test_multiplies_native_twelve_dof_blocks(self):
        """Multiply native affine blocks without particle-row expansion."""
        builder = BlockCsrBuilder12(2)
        builder.add_block(0, 0, np.eye(12, dtype=np.float32) * 2.0)
        builder.add_block(0, 1, np.eye(12, dtype=np.float32) * -1.0)
        builder.add_block(1, 0, np.eye(12, dtype=np.float32) * -1.0)
        builder.add_block(1, 1, np.eye(12, dtype=np.float32) * 3.0)
        matrix = builder.finalize("cpu")
        x = wp.array([vec12(*range(12)), vec12(*range(12, 24))], dtype=vec12, device="cpu")
        output = wp.empty_like(x)

        matrix.multiply(x, output)

        dense = np.zeros((24, 24), dtype=np.float32)
        dense[:12, :12] = np.eye(12, dtype=np.float32) * 2.0
        dense[:12, 12:] = np.eye(12, dtype=np.float32) * -1.0
        dense[12:, :12] = np.eye(12, dtype=np.float32) * -1.0
        dense[12:, 12:] = np.eye(12, dtype=np.float32) * 3.0
        np.testing.assert_allclose(output.numpy().reshape(-1), dense @ np.arange(24, dtype=np.float32))

    def test_ensures_all_ordered_stencil_blocks(self):
        """Create and resolve every ordered block pair in a stencil."""
        builder = BlockCsrBuilder12(2)
        stencils = np.asarray([[1, 0]], dtype=np.int32)
        builder.ensure_stencil_blocks(stencils)

        matrix = builder.finalize("cpu")

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 2, 4])
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 1, 0, 1])
        np.testing.assert_array_equal(matrix.stencil_block_indices(stencils), [[3, 2, 1, 0]])

    def test_rejects_invalid_builder_inputs(self):
        """Reject invalid dimensions, coordinates, shapes, and coefficients."""
        with self.assertRaises(ValueError):
            BlockCsrBuilder12(0)

        builder = BlockCsrBuilder12(2)
        for row, column, value in [
            (0, 2, np.eye(12, dtype=np.float32)),
            (0, 0, np.zeros((11, 12), dtype=np.float32)),
            (0, 0, np.full((12, 12), np.nan, dtype=np.float32)),
        ]:
            with self.subTest(row=row, column=column, shape=value.shape), self.assertRaises(ValueError):
                builder.add_block(row, column, value)
