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
    If ``coupling`` is positive, the first row after the block is a combination of the block rows,
    up to a relative perturbation of that size, which makes the matrix ill-conditioned but not the block.
    """

    def __init__(self, dims, offsets, num_rows, num_redundant, device, seed=7, coupling=0.0):
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
            if coupling > 0.0:
                row = rng.standard_normal(rows) @ block
                G[offset + rows] = row + coupling * np.linalg.norm(row) * rng.standard_normal(n + 4) / np.sqrt(n + 4)
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
        self.D_in = flat.copy()
        self.maxdim = maxdim

    def shifted_matrix(self, w: int) -> np.ndarray:
        n, start = self.dims[w], w * self.maxdim * self.maxdim
        return self.D.numpy()[start : start + n * n].reshape(n, n)

    def detect(self, shift: RedundantRowShift, reuse: bool = False):
        """Restores the original matrices, and shifts their redundant rows."""
        self.D.assign(self.D_in)
        shift.load(self.dim, self.mio, self.row_offset, self.rows, self.D)
        shift.detect(reuse)


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
        problem.detect(shift)
        shifted = shift.shifted.numpy()
        shifts = shift.shifts.numpy()
        for w, D in enumerate(problem.matrices):
            offset, rows = problem.offsets[w], problem.num_rows[w]
            block = slice(offset, offset + rows)
            expected = reference_shifted_rows(D[block, block], 1e-5)
            np.testing.assert_array_equal(shifted[w, :rows].astype(bool), expected)
            np.testing.assert_array_equal(shifts[w, :rows], np.where(expected, np.diag(D)[block], 0.0))

            # Only the diagonal of the shifted rows changes, by its original value
            difference = problem.shifted_matrix(w).astype(np.float64) - D
            change = np.zeros(len(D))
            change[block] = shifts[w, :rows]
            np.testing.assert_allclose(difference, np.diag(change), rtol=1e-6, atol=0.0)

    def test_01_shifted_system_solves_original(self):
        """Consistent systems with the shifted matrix carry no impulse on the shifted rows and solve the original."""
        problem = RedundantProblem(
            dims=[16, 24], offsets=[2, 0], num_rows=[10, 18], num_redundant=[3, 4], device=self.device
        )
        shift = RedundantRowShift(len(problem.dims), problem.max_rows, threshold=1e-5, device=self.device)
        problem.detect(shift)
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
        shift = RedundantRowShift(len(problem.dims), problem.max_rows, threshold=1e-5, device=self.device)
        problem.detect(shift)
        np.testing.assert_array_equal(problem.D.numpy(), problem.D_in)
        self.assertEqual(int(shift.shifted.numpy().sum()), 0)

    def test_03_shared_and_global_triangles(self):
        """Small and large blocks, eliminated in shared and global memory on CUDA, match the float64 reference."""
        for rows, shared in ((40, True), (260, False)):
            with self.subTest(rows=rows):
                problem = RedundantProblem(
                    dims=[rows + 20], offsets=[10], num_rows=[rows], num_redundant=[6], device=self.device
                )
                shift = RedundantRowShift(1, rows, threshold=1e-5, check=True, device=self.device)
                self.assertEqual(shift.shared_triangle, shared and wp.get_device(self.device).is_cuda)
                problem.detect(shift)
                block = slice(10, 10 + rows)
                expected = reference_shifted_rows(problem.matrices[0][block, block], 1e-5)
                np.testing.assert_array_equal(shift.shifted.numpy()[0].astype(bool), expected)

                # Check that the shifted rows pass the check
                problem.detect(shift, reuse=True)
                self.assertEqual(int(shift.pivot_rows.numpy()[0]), 0)
                np.testing.assert_array_equal(shift.shifted.numpy()[0].astype(bool), expected)

    def test_04_graph_capture(self):
        """The shifts are recomputed from the current matrices when replaying a captured graph."""
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("Requires a CUDA device")
        problem = RedundantProblem(dims=[12], offsets=[0], num_rows=[8], num_redundant=[2], device=self.device)
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                shift = RedundantRowShift(1, problem.max_rows, threshold=1e-5, check=reuse, device=self.device)
                problem.detect(shift, reuse)
                expected = problem.D.numpy().copy()
                shift.reset()
                problem.D.assign(problem.D_in)
                with wp.ScopedCapture(device=self.device) as capture:
                    shift.load(problem.dim, problem.mio, problem.row_offset, problem.rows, problem.D)
                    shift.detect(reuse)
                problem.D.assign(problem.D_in)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(problem.D.numpy(), expected)


class TestRedundantRowCheck(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = test_context.device
        self.problem = RedundantProblem(
            dims=[30, 40, 50], offsets=[4, 0, 10], num_rows=[20, 30, 36], num_redundant=[3, 5, 6], device=self.device
        )
        self.shift = RedundantRowShift(3, self.problem.max_rows, threshold=1e-5, check=True, device=self.device)
        self.problem.detect(self.shift)
        self.reference = self.shift.shifted.numpy().copy()

    def check(self, flags: np.ndarray) -> np.ndarray:
        """Checks the given shifted rows of the original matrices, and returns the worlds failing the check."""
        problem = self.problem
        problem.D.assign(problem.D_in)
        self.shift.shifted.assign(flags)
        self.shift.load(problem.dim, problem.mio, problem.row_offset, problem.rows, problem.D)
        self.shift.check()
        pivot_rows = self.shift.pivot_rows.numpy()
        np.testing.assert_array_equal(pivot_rows, np.where(pivot_rows > 0, problem.num_rows, 0))
        return pivot_rows > 0

    def test_00_valid_rows_pass(self):
        """The rows shifted by the elimination pass the check."""
        np.testing.assert_array_equal(self.check(self.reference), [False] * 3)

    def test_01_dependent_kept_rows_fail(self):
        """Keeping a redundant row makes the kept rows dependent and fails the check."""
        rng = np.random.default_rng(1)
        for trial in range(5):
            with self.subTest(trial=trial):
                flags = self.reference.copy()
                for w in range(3):
                    flags[w, rng.choice(np.flatnonzero(flags[w]))] = 0
                np.testing.assert_array_equal(self.check(flags), [True] * 3)

    def test_02_independent_shifted_rows_fail(self):
        """Shifting a kept row, which is independent of the other kept rows, fails the check."""
        rng = np.random.default_rng(2)
        for trial in range(5):
            with self.subTest(trial=trial):
                flags = self.reference.copy()
                for w in range(3):
                    kept = np.flatnonzero(flags[w, : self.problem.num_rows[w]] == 0)
                    flags[w, rng.choice(kept)] = 1
                np.testing.assert_array_equal(self.check(flags), [True] * 3)

    def test_03_eliminate_failed_worlds(self):
        """Only the worlds failing the check are eliminated again, which restores their shifts."""
        problem = self.problem
        flags = self.reference.copy()
        flags[1, np.flatnonzero(flags[1])[0]] = 0
        self.shift.shifted.assign(flags)
        problem.detect(self.shift, reuse=True)
        np.testing.assert_array_equal(self.shift.pivot_rows.numpy() > 0, [False, True, False])
        np.testing.assert_array_equal(self.shift.shifted.numpy(), self.reference)

        # Only the diagonal of the shifted rows changes, by its original value
        for w, D in enumerate(problem.matrices):
            offset, rows = problem.offsets[w], problem.num_rows[w]
            change = np.zeros(len(D))
            change[offset : offset + rows] = np.where(self.reference[w, :rows], np.diag(D)[offset : offset + rows], 0)
            difference = problem.shifted_matrix(w).astype(np.float64) - D
            np.testing.assert_allclose(difference, np.diag(change), rtol=1e-6, atol=0.0)

    def test_04_never_eliminated_worlds_fail(self):
        """The worlds never eliminated, or reset, fail the check."""
        self.shift.reset()
        np.testing.assert_array_equal(self.check(self.reference), [True] * 3)
        self.problem.detect(self.shift)
        mask = wp.array([False, True, False], dtype=wp.bool, device=self.device)
        self.shift.reset(mask)
        np.testing.assert_array_equal(self.check(self.reference), [False, True, False])

    def test_05_coupling_with_other_rows(self):
        """The check only depends on the block, not on its coupling with the other rows."""
        self.problem = RedundantProblem(
            dims=[40], offsets=[4], num_rows=[24], num_redundant=[4], device=self.device, coupling=1e-3
        )
        self.shift = RedundantRowShift(1, self.problem.max_rows, threshold=1e-5, check=True, device=self.device)
        self.problem.detect(self.shift)
        reference = self.shift.shifted.numpy().copy()
        self.assertFalse(self.check(reference)[0])
        flags = reference.copy()
        flags[0, np.flatnonzero(flags[0])[0]] = 0
        self.assertTrue(self.check(flags)[0])


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
