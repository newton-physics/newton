# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for per-block hybrid LLT dispatch."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg import (
    DenseLinearOperatorData,
    DenseSquareMultiLinearInfo,
    HybridLLTBlockedSolver,
)
from newton.tests.kamino import setup_tests, test_context


def _complete_adjacency(dimension: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(column for column in range(dimension) if column != row) for row in range(dimension))


def _path_adjacency(dimension: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(column for column in (row - 1, row + 1) if 0 <= column < dimension) for row in range(dimension))


def _make_spd(adjacency: tuple[tuple[int, ...], ...], rng: np.random.Generator) -> np.ndarray:
    dimension = len(adjacency)
    matrix = np.zeros((dimension, dimension), dtype=np.float32)
    for row, neighbors in enumerate(adjacency):
        for column in neighbors:
            if column > row:
                matrix[row, column] = matrix[column, row] = rng.uniform(-0.1, 0.1)
    matrix[np.diag_indices(dimension)] = np.sum(np.abs(matrix), axis=1) + 1.0
    return matrix


class TestLinAlgHybridLLTBlockedSolver(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _make_problem(self, adjacency):
        rng = np.random.default_rng(17)
        matrices = [_make_spd(block, rng) for block in adjacency]
        right_hand_sides = [rng.standard_normal(matrix.shape[0]).astype(np.float32) for matrix in matrices]
        info = DenseSquareMultiLinearInfo()
        info.finalize(
            dimensions=[matrix.shape[0] for matrix in matrices],
            dtype=wp.float32,
            device=self.device,
        )
        matrix = wp.array(np.concatenate([value.reshape(-1) for value in matrices]), device=self.device)
        right_hand_side = wp.array(np.concatenate(right_hand_sides), device=self.device)
        solution = wp.zeros(info.total_vec_size, dtype=wp.float32, device=self.device)
        return info, matrices, right_hand_sides, matrix, right_hand_side, solution

    def test_mixed_batch_preserves_small_block_specialization(self):
        """Dispatch isolated 6-DoF blocks separately from a large sparse block."""
        adjacency = (
            _complete_adjacency(6),
            _complete_adjacency(6),
            _complete_adjacency(20),
            _path_adjacency(128),
        )
        info, matrices, right_hand_sides, matrix, right_hand_side, solution = self._make_problem(adjacency)
        solver = HybridLLTBlockedSolver(
            operator=DenseLinearOperatorData(info=info, mat=matrix),
            factorize_block_size=64,
            solve_block_dim=256,
            rcm_min_dimension=64,
            symbolic_adjacency=adjacency,
            device=self.device,
        )

        self.assertEqual(solver.sequential_block_indices, (0, 1))
        self.assertEqual(solver.tiled_block_indices, (2,))
        self.assertEqual(solver.rcm_block_indices, (3,))
        self.assertIs(solver.L, solver._rcm_solver.L)

        solver.compute(matrix)
        solver.solve(right_hand_side, solution)

        offsets = np.cumsum([0, *(value.shape[0] for value in matrices)])
        solution_np = solution.numpy()
        for block, (block_matrix, block_rhs) in enumerate(zip(matrices, right_hand_sides, strict=True)):
            expected = np.linalg.solve(block_matrix, block_rhs)
            np.testing.assert_allclose(
                solution_np[offsets[block] : offsets[block + 1]], expected, rtol=2.0e-4, atol=2.0e-5
            )
        np.testing.assert_array_equal(solver.P.numpy()[:12], np.tile(np.arange(6, dtype=np.int32), 2))

    def test_dense_intermediate_block_uses_tiled_llt(self):
        """Keep an intermediate dense block on the ordinary tiled path."""
        adjacency = (_complete_adjacency(48),)
        info, matrices, right_hand_sides, matrix, right_hand_side, solution = self._make_problem(adjacency)
        solver = HybridLLTBlockedSolver(
            operator=DenseLinearOperatorData(info=info, mat=matrix),
            rcm_min_dimension=32,
            symbolic_adjacency=adjacency,
            device=self.device,
        )

        self.assertEqual(solver.sequential_block_indices, ())
        self.assertEqual(solver.tiled_block_indices, (0,))
        self.assertEqual(solver.rcm_block_indices, ())

        solver.compute(matrix)
        solver.solve(right_hand_side, solution)

        expected = np.linalg.solve(matrices[0], right_hand_sides[0])
        np.testing.assert_allclose(solution.numpy(), expected, rtol=2.0e-4, atol=2.0e-5)

    def test_permutation_initializes_inactive_capacity(self):
        """Initialize packed permutations through every block's maximum dimension."""
        adjacency = (_complete_adjacency(6), _path_adjacency(128))
        info, _matrices, _right_hand_sides, matrix, _right_hand_side, _solution = self._make_problem(adjacency)
        info.dim.assign([3, 64])
        solver = HybridLLTBlockedSolver(
            operator=DenseLinearOperatorData(info=info, mat=matrix),
            rcm_min_dimension=64,
            symbolic_adjacency=adjacency,
            device=self.device,
        )

        permutation = solver.P.numpy()
        np.testing.assert_array_equal(permutation[:6], np.arange(6, dtype=np.int32))
        np.testing.assert_array_equal(np.sort(permutation[6:]), np.arange(128, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
