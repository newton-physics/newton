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
        flat = np.zeros((len(dims), maxdim * maxdim), dtype=np.float32)
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
            flat[w, : n * n] = D.ravel()
        self.max_rows = max(num_rows)
        self.dim = wp.array(dims, dtype=wp.int32, device=device)
        self.row_offset = wp.array(offsets, dtype=wp.int32, device=device)
        self.rows = wp.array(num_rows, dtype=wp.int32, device=device)
        self.D = wp.array(flat, dtype=wp.float32, device=device)
        self.D_in = flat.copy()
        self.maxdim = maxdim

    def shifted_matrix(self, w: int) -> np.ndarray:
        n = self.dims[w]
        return self.D.numpy()[w, : n * n].reshape(n, n)

    def detect(self, shift: RedundantRowShift, reuse: bool = False):
        """Restores the original matrices, and shifts their redundant rows."""
        self.D.assign(self.D_in)
        shift.load(self.dim, self.row_offset, self.rows, self.D)
        shift.detect(reuse)

    def regularize_and_factorize(self, eta: float, perms: list[np.ndarray] | None = None):
        """
        Adds ``eta`` times the diagonal to the matrices in-place, and returns the flat lower Cholesky factors
        of the matrices, or of the matrices permuted by ``perms``, with the flat permutations and their inverses.
        """
        D = self.D.numpy()
        L = np.zeros_like(D)
        perm = np.zeros(len(self.dims) * self.maxdim, dtype=np.int32)
        inverse = np.zeros_like(perm)
        for w, n in enumerate(self.dims):
            vstart = w * self.maxdim
            K = D[w, : n * n].reshape(n, n).astype(np.float64)
            K += eta * np.diag(np.diag(K))
            D[w, : n * n] = K.ravel()
            p = np.arange(n) if perms is None else perms[w]
            L[w, : n * n] = np.linalg.cholesky(K[np.ix_(p, p)]).ravel()
            perm[vstart : vstart + n] = p
            inverse[vstart + p] = np.arange(n)
        self.D.assign(D)
        device = self.D.device
        return (
            wp.array(L, dtype=wp.float32, device=device),
            wp.array(perm, dtype=wp.int32, device=device),
            wp.array(inverse, dtype=wp.int32, device=device),
        )


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
                    shift.load(problem.dim, problem.row_offset, problem.rows, problem.D)
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
        self.shift.load(problem.dim, problem.row_offset, problem.rows, problem.D)
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

    def test_06_reference_of_unchanged_rows(self):
        """Valid rows failing the check pass the next checks, until they become dependent."""
        rows = 24
        problem = RedundantProblem(dims=[rows], offsets=[0], num_rows=[rows], num_redundant=[0], device=self.device)
        shift = RedundantRowShift(1, rows, threshold=1e-5, check=True, device=self.device)

        def set_eigenvalue(eigenvalue: float):
            """Set a block whose relative pivots exceed its smallest eigenvalue by about the number of rows."""
            v = np.full(rows, 1.0 / np.sqrt(rows))
            D = np.eye(rows) - (1.0 - eigenvalue) * np.outer(v, v)
            problem.D_in[0, : rows * rows] = D.astype(np.float32).ravel()

        # Check that the valid rows fail, and that the elimination keeps them as the reference
        set_eigenvalue(8e-6)
        problem.detect(shift)
        self.assertEqual(int(shift.shifted.numpy().sum()), 0)
        problem.detect(shift, reuse=True)
        self.assertGreater(int(shift.pivot_rows.numpy()[0]), 0)
        eigenvalue = float(shift._eigenvalue.numpy()[0])
        self.assertGreater(eigenvalue, 0.0)
        self.assertLess(eigenvalue, 1e-5)
        self.assertEqual(float(shift._reference.numpy()[0]), eigenvalue)

        # Check that the same rows now pass, also with a smaller eigenvalue, but not once they become dependent
        for value, passes in ((8e-6, True), (4e-6, True), (0.0, False)):
            with self.subTest(eigenvalue=value):
                set_eigenvalue(value)
                problem.detect(shift, reuse=True)
                self.assertEqual(int(shift.pivot_rows.numpy()[0]) == 0, passes)
        self.assertEqual(int(shift.shifted.numpy().sum()), 1)
        self.assertEqual(float(shift._reference.numpy()[0]), 0.0)


class TestRedundantRowFactorCheck(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = test_context.device
        self.problem = RedundantProblem(
            dims=[30, 40, 50], offsets=[4, 0, 10], num_rows=[20, 30, 36], num_redundant=[3, 5, 6], device=self.device
        )
        self.setup_shift(seed=5)

    def setup_shift(self, seed: int):
        """Detects the shifted rows of the problem, and draws the permutations of the factors."""
        problem = self.problem
        worlds = len(problem.dims)
        self.vio = wp.array([w * problem.maxdim for w in range(worlds)], dtype=wp.int32, device=self.device)
        self.shift = RedundantRowShift(
            worlds, problem.max_rows, threshold=1e-5, max_dim=problem.maxdim, device=self.device
        )
        problem.detect(self.shift)
        self.reference = self.shift.shifted.numpy().copy()
        rng = np.random.default_rng(seed)
        self.perms = [rng.permutation(n) for n in problem.dims]

    def check(self, flags: np.ndarray, permuted: bool = False, eta: float = 1e-5) -> np.ndarray:
        """Shifts the given rows of the original matrices, factorizes them, and returns the worlds failing the check."""
        problem = self.problem
        problem.D.assign(problem.D_in)
        self.shift.shifted.assign(flags)
        self.shift.load(problem.dim, problem.row_offset, problem.rows, problem.D)
        self.shift.shift()
        L, perm, inverse = problem.regularize_and_factorize(eta, self.perms if permuted else None)
        self.shift.check_factor(self.vio, L, problem.dim, perm if permuted else None, inverse if permuted else None)
        pivot_rows = self.shift.pivot_rows.numpy()
        np.testing.assert_array_equal(pivot_rows, np.where(pivot_rows > 0, problem.num_rows, 0))
        np.testing.assert_array_equal(self.shift.refactor_dim.numpy(), np.where(pivot_rows > 0, problem.dims, 0))
        return pivot_rows > 0

    def test_00_valid_rows_pass(self):
        """The rows shifted by the elimination pass the check, with and without a permutation of the factors."""
        for permuted in (False, True):
            with self.subTest(permuted=permuted):
                np.testing.assert_array_equal(self.check(self.reference, permuted), [False] * 3)

    def test_01_dependent_kept_rows_fail(self):
        """Keeping a redundant row makes the kept rows dependent and fails the check."""
        rng = np.random.default_rng(1)
        for permuted in (False, True):
            with self.subTest(permuted=permuted):
                flags = self.reference.copy()
                for w in range(3):
                    flags[w, rng.choice(np.flatnonzero(flags[w]))] = 0
                np.testing.assert_array_equal(self.check(flags, permuted), [True] * 3)

    def test_02_independent_shifted_rows_fail(self):
        """Shifting a kept row, which is independent of the other kept rows, fails the check."""
        rng = np.random.default_rng(2)
        for permuted in (False, True):
            with self.subTest(permuted=permuted):
                flags = self.reference.copy()
                for w in range(3):
                    kept = np.flatnonzero(flags[w, : self.problem.num_rows[w]] == 0)
                    flags[w, rng.choice(kept)] = 1
                np.testing.assert_array_equal(self.check(flags, permuted), [True] * 3)

    def test_03_eliminate_failed_worlds(self):
        """Only the worlds failing the check are eliminated again, which restores their shifts."""
        problem = self.problem
        flags = self.reference.copy()
        flags[1, np.flatnonzero(flags[1])[0]] = 0
        np.testing.assert_array_equal(self.check(flags), [False, True, False])
        D_checked = problem.D.numpy().copy()
        self.shift.eliminate()
        np.testing.assert_array_equal(self.shift.shifted.numpy(), self.reference)

        # Only the diagonal of the restored row changes, by its original value
        difference = problem.D.numpy().astype(np.float64) - D_checked
        row = problem.offsets[1] + np.flatnonzero(flags[1] != self.reference[1])[0]
        expected = np.zeros_like(difference)
        expected[1, row * (problem.dims[1] + 1)] = problem.D_in[1, row * (problem.dims[1] + 1)]
        np.testing.assert_allclose(difference, expected, rtol=1e-6, atol=0.0)

    def test_04_reference_of_unchanged_rows(self):
        """Valid rows failing the check because of the other rows pass the next checks, until they become dependent."""
        self.problem = RedundantProblem(
            dims=[40], offsets=[4], num_rows=[24], num_redundant=[4], device=self.device, coupling=1e-2
        )
        self.setup_shift(seed=4)
        reference = self.reference
        for permuted in (False, True):
            with self.subTest(permuted=permuted):
                self.shift.reset()
                self.shift._initialized.fill_(1)

                # Check that the valid rows fail, and that the elimination keeps them as the reference
                self.assertTrue(self.check(reference, permuted, eta=1e-6)[0])
                eigenvalue = float(self.shift._eigenvalue.numpy()[0])
                self.assertGreater(eigenvalue, 0.0)
                self.assertLess(eigenvalue, 1e-5)
                self.shift.eliminate()
                np.testing.assert_array_equal(self.shift.shifted.numpy(), reference)
                self.assertEqual(float(self.shift._reference.numpy()[0]), eigenvalue)

                # Check that the same rows now pass, but not with a kept redundant row
                self.assertFalse(self.check(reference, permuted, eta=1e-6)[0])
                flags = reference.copy()
                flags[0, np.flatnonzero(flags[0])[0]] = 0
                self.assertTrue(self.check(flags, permuted, eta=1e-6)[0])

                # Check that a reset clears the reference
                self.shift.reset()
                self.shift._initialized.fill_(1)
                self.assertTrue(self.check(reference, permuted, eta=1e-6)[0])


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
