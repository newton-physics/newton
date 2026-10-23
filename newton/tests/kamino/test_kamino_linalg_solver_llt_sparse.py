# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical contracts for the block-sparse Cholesky backend."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize.llt_sparse import LLTBlockSparseSolver


class TestLLTBlockSparseSolver(unittest.TestCase):
    def test_dense_capacity_refresh(self):
        """Refresh changing active dimensions and matrix sparsity under a fixed capacity."""
        rng = np.random.default_rng(312)
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device):
                info = DenseSquareMultiLinearInfo()
                info.finalize([13], dtype=wp.float32, device=device)
                matrix = wp.zeros(13 * 13, device=device)
                rhs = wp.array(rng.normal(size=13).astype(np.float32), device=device)
                x = wp.full(13, -123.0, device=device)
                solver = LLTBlockSparseSolver(operator=DenseLinearOperatorData(info=info, mat=matrix))
                solver.compute(matrix)
                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device) as capture:
                        solver.compute(matrix)
                        solver.solve(rhs, x)
                    graph = capture.graph
                for n in (7, 13, 0, 11):
                    info.dim.assign([n])
                    storage = np.zeros(169, dtype=np.float32)
                    basis = rng.normal(size=(n, n))
                    a = (basis @ basis.T + np.eye(n)).astype(np.float32)
                    storage[: n * n] = a.ravel()
                    matrix.assign(storage)
                    x.fill_(-123.0)
                    if graph is None:
                        solver.compute(matrix)
                        solver.solve(rhs, x)
                    else:
                        wp.capture_launch(graph)
                    if n:
                        np.testing.assert_allclose(
                            x.numpy()[:n], np.linalg.solve(a.astype(np.float64), rhs.numpy()[:n]), atol=3e-6, rtol=3e-6
                        )
                    np.testing.assert_array_equal(x.numpy()[n:], -123.0)
                solver.finalize(DenseLinearOperatorData(info=info, mat=matrix))
                with self.assertRaisesRegex(ValueError, "factorization"):
                    solver.solve(rhs, x)
                solver.compute(matrix)
                solver.solve(rhs, x)
                np.testing.assert_allclose(
                    x.numpy()[:11], np.linalg.solve(a.astype(np.float64), rhs.numpy()[:11]), atol=3e-6, rtol=3e-6
                )

    def test_ragged_graphs_and_repeated_solves(self):
        """Solve heterogeneous filled graphs with noncontiguous rows and inactive worlds."""
        rng = np.random.default_rng(619)
        groups = [[[0, 3], [1, 4, 6], [2, 5], [7]], [[0], [1, 3], [2]], [[0]]]
        edges = [[(0, 1), (1, 2), (2, 3), (3, 0)], [(0, 2)], []]
        matrices = []
        for rows, pairs in zip(groups, edges, strict=True):
            n = sum(map(len, rows))
            a = np.zeros((n, n), dtype=np.float32)
            for i, j in [*pairs, *((k, k) for k in range(len(rows)))]:
                block = rng.normal(0.0, 0.1, (len(rows[i]), len(rows[j])))
                a[np.ix_(rows[i], rows[j])] += block
                a[np.ix_(rows[j], rows[i])] += block.T
            a += np.diag(np.sum(np.abs(a), axis=1) + 1.0)
            matrices.append(a)
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device):
                info = DenseSquareMultiLinearInfo()
                info.finalize([8, 4, 1], dtype=wp.float32, device=device)
                source = np.concatenate([a.ravel() for a in matrices])
                a = wp.array(source, device=device)
                solver = LLTBlockSparseSolver(block_rows=groups, block_edges=edges, device=device)
                solver.finalize(DenseLinearOperatorData(info=info, mat=a))
                # The cycle adds one fill block; the disconnected second graph adds none.
                self.assertEqual(solver.values.shape[0], 9 + 4 + 1)
                rhs = rng.normal(size=13).astype(np.float32)
                b = wp.array(rhs, device=device)
                x = wp.full(13, -123.0, device=device)
                solver.compute(a)
                solver.solve(b, x)
                for i, offset in enumerate((0, 8, 12)):
                    n = len(matrices[i])
                    np.testing.assert_allclose(
                        x.numpy()[offset : offset + n],
                        np.linalg.solve(matrices[i].astype(np.float64), rhs[offset : offset + n]),
                        atol=2e-6,
                        rtol=2e-6,
                    )
                reconstructed = wp.empty_like(a)
                solver.reconstruct(reconstructed)
                np.testing.assert_allclose(reconstructed.numpy(), source, atol=2e-6, rtol=2e-6)
                # Reuse the same factors for strided multiple RHS and whitened-response Grams.
                rhs_matrix = rng.normal(size=(13, 5)).astype(np.float32)
                rhs_matrix_wp = wp.array(rhs_matrix.ravel(), device=device)
                forward = wp.full(65, -123.0, device=device)
                result = wp.full(65, -123.0, device=device)
                layout = {
                    "offsets": wp.array([0, 40, 60], dtype=wp.int32, device=device),
                    "strides": wp.array([5, 5, 5], dtype=wp.int32, device=device),
                    "columns": wp.array([3, 2, 1], dtype=wp.int32, device=device),
                    "max_columns": 3,
                }
                solver.solve_triangular(rhs_matrix_wp, forward, forward=True, **layout)
                solver.solve_triangular(forward, result, forward=False, **layout)
                for i, (offset, width) in enumerate(((0, 3), (8, 2), (12, 1))):
                    n = len(matrices[i])
                    b_matrix = rhs_matrix[offset : offset + n, :width].astype(np.float64)
                    expected = np.linalg.solve(matrices[i].astype(np.float64), b_matrix)
                    actual = result.numpy().reshape(13, 5)[offset : offset + n]
                    y = forward.numpy().reshape(13, 5)[offset : offset + n, :width]
                    np.testing.assert_allclose(actual[:, :width], expected, rtol=2e-6, atol=2e-6)
                    np.testing.assert_allclose(y.T @ y, b_matrix.T @ expected, rtol=3e-6, atol=3e-6)
                    np.testing.assert_array_equal(actual[:, width:], -123.0)
                active = wp.array([8, 0, 1], dtype=wp.int32, device=device)
                info.dim = active
                x.fill_(-123.0)
                solver.solve(b, x)
                np.testing.assert_array_equal(x.numpy()[8:12], -123.0)
                info.dim.assign([8, 4, 1])
                wp.copy(x, b)
                solver.solve_inplace(x)
                reference = x.numpy()
                # Every capture replay must refresh the factors and preserve RHS aliasing.
                if device.is_cuda:
                    with wp.ScopedCapture(device) as capture:
                        solver.compute(a)
                        solver.solve(b, x)
                    for multiplier in (2.0, 0.5):
                        a.assign(source * multiplier)
                        wp.capture_launch(capture.graph)
                        np.testing.assert_allclose(x.numpy(), reference / multiplier, atol=3e-6, rtol=3e-6)

    def test_sparse_storage_and_pivot_retry(self):
        """Keep chain storage sparse and retry only the failing world's factors."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device):
                groups = [[list(range(i, i + 6)) for i in range(0, 96, 6)]] * 2
                edges = [[(i, i + 1) for i in range(15)]] * 2
                info = DenseSquareMultiLinearInfo()
                info.finalize([96, 96], dtype=wp.float32, device=device)
                matrices = np.stack([np.eye(96, dtype=np.float32)] * 2)
                matrices[1, 0, 0] = -1e-6
                a = wp.array(matrices.ravel(), device=device)
                solver = LLTBlockSparseSolver(
                    operator=DenseLinearOperatorData(info=info, mat=a),
                    block_rows=groups,
                    block_edges=edges,
                    failed_pivot_shift=1e-5,
                )
                self.assertLess(solver.values.size, a.size // 4)
                solver.compute(a)
                x = wp.ones(192, device=device)
                solver.solve_inplace(x)
                np.testing.assert_array_equal(x.numpy()[:96], 1.0)
                np.testing.assert_allclose(x.numpy()[96], 1.0 / 9e-6, rtol=2e-6)
                np.testing.assert_array_equal(a.numpy(), matrices.ravel())


if __name__ == "__main__":
    unittest.main()
