# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the batched semi-sparse blocked Cholesky solver."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_semi_sparse import (
    SemiSparseBlockCholeskySolverBatched,
)
from newton.tests.kamino import setup_tests, test_context


class TestSemiSparseBlockCholeskySolverBatched(unittest.TestCase):
    """Verify semi-sparse blocked Cholesky solves."""

    def setUp(self):
        """Initialize the shared Kamino test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def tearDown(self):
        """Release the test device reference."""
        self.device = None

    def test_solve_handles_multiple_rhs(self):
        """Solve every requested RHS using one shared factorization per batch."""
        rng = np.random.default_rng(1234)
        batch_size = 2
        rhs_size = 5
        max_equations = 9
        active_equations = np.array([7, 9], dtype=np.int32)

        matrices = np.zeros((batch_size, max_equations, max_equations), dtype=np.float32)
        right_hand_sides = np.zeros((batch_size, max_equations, rhs_size), dtype=np.float32)
        expected = np.zeros_like(right_hand_sides)
        for batch_index, size in enumerate(active_equations):
            dense = rng.normal(size=(size, size)).astype(np.float32)
            matrix = dense @ dense.T + np.eye(size, dtype=np.float32)
            rhs = rng.normal(size=(rhs_size, size)).astype(np.float32)
            matrices[batch_index, :size, :size] = matrix
            right_hand_sides[batch_index, :size, :] = rhs.T
            expected[batch_index, :size, :] = np.linalg.solve(matrix, rhs.T)

        solver = SemiSparseBlockCholeskySolverBatched(
            num_batches=batch_size,
            max_num_equations=max_equations,
            block_size=4,
            device=self.device,
            enable_reordering=True,
        )
        solver.capture_sparsity_pattern(matrices, active_equations)

        matrix_wp = wp.array(matrices, dtype=wp.float32, device=self.device)
        active_wp = wp.array(active_equations, dtype=wp.int32, device=self.device)
        mask_wp = wp.ones(batch_size, dtype=wp.bool, device=self.device)
        rhs_wp = wp.array(right_hand_sides, dtype=wp.float32, device=self.device)
        result_wp = wp.zeros_like(rhs_wp)

        solver.factorize(matrix_wp, active_wp, mask_wp)
        with self.assertRaisesRegex(ValueError, "request_rhs_size"):
            solver.solve(rhs_wp, result_wp, mask_wp)

        solver.request_rhs_size(rhs_size)
        solver.solve(rhs_wp, result_wp, mask_wp)

        np.testing.assert_allclose(result_wp.numpy(), expected, rtol=2.0e-4, atol=2.0e-4)

    def test_solve_rejects_insufficient_rows(self):
        """Reject right-hand sides that cannot hold every matrix row."""
        solver = SemiSparseBlockCholeskySolverBatched(
            num_batches=2,
            max_num_equations=9,
            block_size=4,
            device=self.device,
            enable_reordering=True,
        )
        rhs = wp.zeros((2, 8, 3), dtype=wp.float32, device=self.device)
        result = wp.zeros_like(rhs)
        mask = wp.ones(2, dtype=wp.bool, device=self.device)
        solver.request_rhs_size(3)

        with self.assertRaisesRegex(ValueError, "at least 9 rows"):
            solver.solve(rhs, result, mask)


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
