# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RedundantRowShift class from linalg/redundancy.py"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg import RedundantRowShift
from newton.tests.kamino import setup_tests, test_context


def reference_shifted_rows(D: np.ndarray, threshold: float) -> np.ndarray:
    """Returns the rows shifted by a float64 Cholesky elimination pivoted on the largest relative pivot."""
    A = D.astype(np.float64)
    diagonal = np.diag(A).copy()
    remaining = list(range(len(D)))
    shifted = np.zeros(len(D), bool)
    while remaining:
        p = remaining.pop(int(np.argmax(np.diag(A)[remaining] / diagonal[remaining])))
        if A[p, p] < threshold * diagonal[p]:
            A[p, p] += diagonal[p]
            shifted[p] = True
        A[np.ix_(remaining, remaining)] -= np.outer(A[remaining, p], A[p, remaining]) / A[p, p]
    return shifted


class RedundantProblem:
    """
    A batch of PSD matrices ``D = G @ G^T`` whose diagonal block of rows ``[offset, offset + num_rows)`` has
    ``num_redundant`` rows in the span of the other rows of the block, with random row scales.
    """

    def __init__(self, dims, offsets, num_rows, num_redundant, device, seed=7):
        rng = np.random.default_rng(seed)
        self.dims, self.offsets, self.num_rows = dims, offsets, num_rows
        self.matrices = []
        maxdim = max(dims)
        flat = np.zeros(len(dims) * maxdim * maxdim, dtype=np.float32)
        for w, (n, offset, rows, redundant) in enumerate(zip(dims, offsets, num_rows, num_redundant, strict=True)):
            G = rng.standard_normal((n, n + 4))
            block = G[offset : offset + rows]
            independent = rows - redundant
            block[independent:] = rng.standard_normal((redundant, independent)) @ block[:independent]
            G[offset : offset + rows] = block[rng.permutation(rows)]
            G *= np.exp(rng.uniform(-2.0, 2.0, size=(n, 1)))
            D = (G @ G.T).astype(np.float32)
            self.matrices.append(D)
            flat[w * maxdim * maxdim : w * maxdim * maxdim + n * n] = D.ravel()
        self.max_rows = max(num_rows)
        self.dim = wp.array(dims, dtype=wp.int32, device=device)
        self.mio = wp.array([w * maxdim * maxdim for w in range(len(dims))], dtype=wp.int32, device=device)
        self.row_offset = wp.array(offsets, dtype=wp.int32, device=device)
        self.rows = wp.array(num_rows, dtype=wp.int32, device=device)
        self.D = wp.array(flat, dtype=wp.float32, device=device)
        self.maxdim = maxdim

    def shifted_matrix(self, w: int) -> np.ndarray:
        n, start = self.dims[w], w * self.maxdim * self.maxdim
        return self.D.numpy()[start : start + n * n].reshape(n, n)


class TestRedundantRowShift(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = test_context.device

    def test_00_shifts_redundant_rows(self):
        """Exactly the redundant rows of the block are shifted by their diagonal, as in the float64 reference."""
        problem = RedundantProblem(
            dims=[12, 20, 30], offsets=[0, 3, 5], num_rows=[8, 12, 20], num_redundant=[2, 0, 5], device=self.device
        )
        shift = RedundantRowShift(len(problem.dims), problem.max_rows, threshold=1e-5, device=self.device)
        shift.apply(problem.dim, problem.mio, problem.row_offset, problem.rows, problem.D)
        shifted = shift.shifted.numpy()
        for w, D in enumerate(problem.matrices):
            offset, rows = problem.offsets[w], problem.num_rows[w]
            block = slice(offset, offset + rows)
            expected = reference_shifted_rows(D[block, block], 1e-5)
            np.testing.assert_array_equal(shifted[w, :rows].astype(bool), expected)

            # Only the diagonal of the shifted rows changes, by its original value
            difference = problem.shifted_matrix(w).astype(np.float64) - D
            change = np.zeros(len(D))
            change[offset : offset + rows] = np.where(expected, np.diag(D)[block], 0.0)
            np.testing.assert_allclose(difference, np.diag(change), rtol=1e-6, atol=0.0)

    def test_01_shifted_system_solves_original(self):
        """Consistent systems with the shifted matrix carry no impulse on the shifted rows and solve the original."""
        problem = RedundantProblem(
            dims=[16, 24], offsets=[2, 0], num_rows=[10, 18], num_redundant=[3, 4], device=self.device
        )
        shift = RedundantRowShift(len(problem.dims), problem.max_rows, threshold=1e-5, device=self.device)
        shift.apply(problem.dim, problem.mio, problem.row_offset, problem.rows, problem.D)
        shifted = shift.shifted.numpy()
        rng = np.random.default_rng(3)
        for w, matrix in enumerate(problem.matrices):
            D = matrix.astype(np.float64)
            D_shifted = problem.shifted_matrix(w).astype(np.float64)
            b = D @ rng.standard_normal(len(D))
            x = np.linalg.solve(D_shifted, b)
            rows = problem.offsets[w] + np.flatnonzero(shifted[w, : problem.num_rows[w]])
            self.assertEqual(len(rows), [3, 4][w])
            np.testing.assert_allclose(D @ x, b, rtol=0.0, atol=1e-3 * np.linalg.norm(b))
            self.assertLess(np.abs(x[rows]).max(), 1e-3 * np.abs(x).max())

    def test_02_full_rank_block_is_unchanged(self):
        """A full-rank block, or an empty one, is left unchanged."""
        problem = RedundantProblem(
            dims=[10, 8], offsets=[1, 0], num_rows=[6, 0], num_redundant=[0, 0], device=self.device
        )
        D_in = problem.D.numpy().copy()
        shift = RedundantRowShift(len(problem.dims), problem.max_rows, threshold=1e-5, device=self.device)
        shift.apply(problem.dim, problem.mio, problem.row_offset, problem.rows, problem.D)
        np.testing.assert_array_equal(problem.D.numpy(), D_in)
        self.assertEqual(int(shift.shifted.numpy().sum()), 0)

    def test_03_shared_and_global_triangles(self):
        """Small and large blocks, eliminated in shared and global memory on CUDA, match the float64 reference."""
        for rows, shared in ((40, True), (260, False)):
            with self.subTest(rows=rows):
                problem = RedundantProblem(
                    dims=[rows + 20], offsets=[10], num_rows=[rows], num_redundant=[6], device=self.device
                )
                shift = RedundantRowShift(1, rows, threshold=1e-5, device=self.device)
                self.assertEqual(shift.shared_triangle, shared and wp.get_device(self.device).is_cuda)
                shift.apply(problem.dim, problem.mio, problem.row_offset, problem.rows, problem.D)
                block = slice(10, 10 + rows)
                expected = reference_shifted_rows(problem.matrices[0][block, block], 1e-5)
                np.testing.assert_array_equal(shift.shifted.numpy()[0].astype(bool), expected)

    def test_04_graph_capture(self):
        """The shifts are recomputed from the current matrices when replaying a captured graph."""
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("Requires a CUDA device")
        problem = RedundantProblem(dims=[12], offsets=[0], num_rows=[8], num_redundant=[2], device=self.device)
        D_in = problem.D.numpy().copy()
        shift = RedundantRowShift(1, problem.max_rows, threshold=1e-5, device=self.device)
        shift.apply(problem.dim, problem.mio, problem.row_offset, problem.rows, problem.D)
        expected = problem.D.numpy().copy()
        problem.D.assign(D_in)
        with wp.ScopedCapture(device=self.device) as capture:
            shift.apply(problem.dim, problem.mio, problem.row_offset, problem.rows, problem.D)
        problem.D.assign(D_in)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(problem.D.numpy(), expected)


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
