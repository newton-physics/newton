# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from limx.linalg.block_csr import BlockCsrBuilder


class TestBlockCsr(unittest.TestCase):
    def test_duplicate_blocks_accumulate_and_columns_sort(self):
        builder = BlockCsrBuilder(3)
        builder.add_scaled_identity(0, 2, 1.0)
        builder.add_scaled_identity(0, 1, 2.0)
        builder.add_scaled_identity(0, 2, 3.0)

        matrix = builder.finalize("cpu")

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 2, 2, 2])
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [1, 2])
        blocks = matrix.values.numpy()
        np.testing.assert_allclose(blocks[0], np.eye(3) * 2.0)
        np.testing.assert_allclose(blocks[1], np.eye(3) * 4.0)
        np.testing.assert_allclose(matrix.diagonal.numpy(), np.zeros((3, 3, 3)))

    def test_multiply_matches_dense_block_matrix_with_empty_row(self):
        builder = BlockCsrBuilder(3)
        builder.add_scaled_identity(0, 0, 2.0)
        builder.add_scaled_identity(0, 1, -1.0)
        builder.add_scaled_identity(1, 0, -1.0)
        builder.add_scaled_identity(1, 1, 3.0)
        matrix = builder.finalize("cpu")
        x = wp.array(
            [wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0), wp.vec3(7.0)],
            dtype=wp.vec3,
            device="cpu",
        )
        output = wp.empty_like(x)

        matrix.multiply(x, output)

        np.testing.assert_allclose(
            output.numpy(),
            [[-2.0, -1.0, 0.0], [11.0, 13.0, 15.0], [0.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(matrix.diagonal.numpy()[0], np.eye(3) * 2.0)
        np.testing.assert_allclose(matrix.diagonal.numpy()[1], np.eye(3) * 3.0)

    def test_rejects_out_of_range_block_index(self):
        builder = BlockCsrBuilder(2)

        with self.assertRaises(ValueError):
            builder.add_scaled_identity(0, 2, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
