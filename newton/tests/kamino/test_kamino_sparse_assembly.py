# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for captured sparse assembly and numerical ordering."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver
from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    _build_sparse_bilateral_block,
    _set_sparse_bilateral_diagonal,
)


@wp.kernel
def _assemble(
    source: wp.array[wp.float32],
    inverse: wp.array[wp.int32],
    permuted: bool,
    n: int,
    target: wp.array[wp.float32],
):
    row, col = wp.tid()
    target_row = row
    target_col = col
    if permuted:
        target_row = inverse[row]
        target_col = inverse[col]
    target[target_row * n + target_col] = source[row * n + col]


class TestKaminoSparseAssemblyLifecycle(unittest.TestCase):
    def test_capture_reset_and_changing_values(self):
        """Preserve numeric ordering through first use, resets, and changing values."""
        if not wp.is_cuda_available() or not wp.is_conditional_graph_supported():
            self.skipTest("Conditional CUDA graphs are required")
        device = wp.get_device("cuda:0")
        n = 65
        first = np.eye(n, dtype=np.float32) * 4.0
        for row in range(n - 1):
            first[row, row + 1] = first[row + 1, row] = -0.25
        second = first.copy()
        second[0, -1] = second[-1, 0] = 0.5
        second[1, 40] = second[40, 1] = -0.125
        rows, cols = np.nonzero(np.triu(second, 1))
        rhs_np = np.random.default_rng(78).normal(size=n).astype(np.float32)
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                info = DenseSquareMultiLinearInfo()
                info.finalize(dimensions=[n], dtype=wp.float32, device=device)
                matrix = wp.zeros(n * n, dtype=wp.float32, device=device)
                source = wp.array(first.ravel(), dtype=wp.float32, device=device)
                solver = LLTBlockedRCMSolver(
                    operator=DenseLinearOperatorData(info=info, mat=matrix),
                    parallel_factorization=parallel,
                    device=device,
                )
                solver.configure_sparse_assembly(
                    wp.zeros(len(rows), dtype=wp.int32, device=device),
                    wp.array(rows, dtype=wp.int32, device=device),
                    wp.array(cols, dtype=wp.int32, device=device),
                )
                rhs = wp.array(rhs_np, dtype=wp.float32, device=device)
                solution = wp.zeros_like(rhs)

                def assemble(target, inverse, source=source, info=info):
                    wp.launch(
                        _assemble,
                        dim=(n, n),
                        inputs=[source, info.vio if inverse is None else inverse, inverse is not None, n, target],
                        device=device,
                    )

                def capture(solver=solver, assemble=assemble, rhs=rhs, solution=solution):
                    with wp.ScopedCapture(device=device) as captured:
                        for _ in range(2):
                            solver.compute_sparse(assemble)
                            solver.solve(rhs, solution)
                    return captured.graph

                def check(expected, fresh_order=False, solution=solution, solver=solver, info=info, source=source):
                    actual = solution.numpy()
                    np.testing.assert_allclose(actual, np.linalg.solve(expected.astype(np.float64), rhs_np), atol=1e-6)
                    p = solver.P.numpy()
                    np.testing.assert_array_equal(np.sort(p), np.arange(n))
                    np.testing.assert_array_equal(solver.inv_P.numpy()[p], np.arange(n))
                    if fresh_order:
                        reference = LLTBlockedRCMSolver(
                            operator=DenseLinearOperatorData(info=info, mat=source), device=device
                        )
                        reference.compute(source)
                        np.testing.assert_array_equal(p, reference.P.numpy())

                discarded = capture()
                del discarded
                graph = capture()
                self.assertEqual(int(solver._structural_ready.numpy()[0]), 0)
                # Values may change after capture but before its first replay.
                source.assign(second.ravel())
                wp.capture_launch(graph)
                check(second, fresh_order=True)
                source.assign(first.ravel())
                wp.capture_launch(graph)
                check(first)
                solver.reset()
                self.assertEqual(int(solver._structural_ready.numpy()[0]), 0)
                wp.capture_launch(graph)
                check(first, fresh_order=True)
                source.assign(second.ravel())
                solver.compute_sparse(assemble)
                self.assertEqual(int(solver._structural_ready.numpy()[0]), 0)
                wp.capture_launch(graph)
                check(second)
                # Recapture after initialization, then reset and replay it.
                replacement = capture()
                solver.reset()
                wp.capture_launch(replacement)
                check(second, fresh_order=True)


def _redundant_bridge_jacobian():
    """Construct redundant rows with a numerically cancelling structural bridge."""
    n = 65
    jacobian = np.zeros((n, n + 4, 6), dtype=np.float32)
    rng = np.random.default_rng(781)
    rng.normal(size=(n, 6))
    for row in range(n):
        common = n if row < 32 else n + 1
        jacobian[row, row, 0] = 1.0
        jacobian[row, common] = rng.normal(size=6).astype(np.float32) * 0.1
    for row in (0, n - 1):
        jacobian[row, n + 2, 0] = jacobian[row, n + 3, 0] = 1.0
    jacobian[n - 1, n + 3, 0] = -1.0
    # Retain the failing ordering-change seed: changing numerical RCM to a
    # topology-only ordering amplified the redundant-system solve error.
    rng = np.random.default_rng(17)
    jacobian[:, n, :] *= rng.uniform(0.5, 2.0, size=6).astype(np.float32)
    jacobian[:, n + 1, :] *= rng.uniform(0.5, 2.0, size=6).astype(np.float32)
    for row in range(n):
        jacobian[row, row, 0] *= np.float32(rng.uniform(0.5, 2.0))
    jacobian[1] = jacobian[0]
    jacobian[33] = jacobian[32]
    return jacobian


def _relative_error(actual, expected):
    """Measure relative Euclidean error without dividing by zero."""
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0e-30))


class TestKaminoSparseAssemblyQuality(unittest.TestCase):
    def test_redundant_cancelling_bridge(self):
        """Preserve numerical ordering and accuracy as a structural bridge becomes nonzero."""
        if not wp.is_cuda_available():
            self.skipTest("Blocked sparse assembly requires CUDA")
        device = wp.get_device("cuda:0")
        with wp.ScopedDevice(device):
            if not wp.is_conditional_graph_supported():
                self.skipTest("Sparse assembly requires CUDA conditional graphs")
            for parallel in (False, True):
                with self.subTest(parallel=parallel):
                    self._check_bridge(device, parallel)

    def _check_bridge(self, device, parallel):
        """Compare captured sparse assembly with ordinary numerical RCM factorization."""
        jacobian = _redundant_bridge_jacobian()
        n, body_count, _ = jacobian.shape
        coordinates = [
            (row, int(body)) for row in range(n) for body in np.flatnonzero(np.any(jacobian[row] != 0, axis=1))
        ]
        pairs = np.asarray(
            [
                (0, row, col, body, i, j)
                for i, (row, body) in enumerate(coordinates)
                for j, (col, other) in enumerate(coordinates)
                if row < col and body == other
            ],
            dtype=np.int32,
        ).T
        pair_arrays = [wp.array(values, dtype=wp.int32, device=device) for values in pairs]
        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[n], dtype=wp.float32, device=device)
        control_matrix = wp.zeros(n * n, dtype=wp.float32, device=device)
        candidate_matrix = wp.zeros_like(control_matrix)
        control = LLTBlockedRCMSolver(
            operator=DenseLinearOperatorData(info=info, mat=control_matrix),
            parallel_factorization=parallel,
            device=device,
        )
        candidate = LLTBlockedRCMSolver(
            operator=DenseLinearOperatorData(info=info, mat=candidate_matrix),
            parallel_factorization=parallel,
            device=device,
        )
        candidate.configure_sparse_assembly(*pair_arrays[:3])
        inv_mass = wp.ones(body_count, dtype=wp.float32, device=device)
        inv_inertia = wp.array(
            np.repeat(np.eye(3, dtype=np.float32)[None], body_count, axis=0), dtype=wp.mat33f, device=device
        )
        values = wp.zeros(len(coordinates), dtype=vec6f, device=device)
        diagonal = wp.zeros(n, dtype=wp.float32, device=device)
        scale = wp.zeros_like(diagonal)
        observed = wp.zeros_like(control_matrix)
        rhs_np = np.random.default_rng(654).normal(size=n).astype(np.float32)
        rhs = wp.array(rhs_np, dtype=wp.float32, device=device)
        control_solution = wp.zeros_like(rhs)
        candidate_solution = wp.zeros_like(rhs)

        def assemble(matrix, inverse):
            matrix.zero_()
            wp.launch(
                _set_sparse_bilateral_diagonal,
                dim=(1, n),
                inputs=[
                    info.dim,
                    info.vio,
                    info.mio,
                    info.vio,
                    diagonal,
                    matrix,
                    scale,
                    inverse if inverse is not None else info.vio,
                    inverse is not None,
                ],
                device=device,
            )
            wp.launch(
                _build_sparse_bilateral_block,
                dim=pairs.shape[1],
                inputs=[
                    inv_mass,
                    inv_inertia,
                    *pair_arrays,
                    values,
                    info.dim,
                    info.mio,
                    info.vio,
                    scale,
                    matrix,
                    inverse if inverse is not None else info.vio,
                    inverse is not None,
                ],
                device=device,
            )
            if inverse is not None:
                wp.copy(observed, matrix)

        # Capture before any candidate numerical initialization. Replaying twice
        # exercises both bootstrap and direct permuted assembly.
        with wp.ScopedCapture(device=device) as capture:
            candidate.compute_sparse(assemble)
            candidate.solve(rhs, candidate_solution)
        initial_order = None
        for bridge in (-1.0, -0.875):
            with self.subTest(bridge=bridge):
                jacobian[n - 1, n + 3, 0] = bridge
                values.assign(np.asarray([jacobian[row, body] for row, body in coordinates]))
                dense = jacobian.astype(np.float64).reshape(n, -1)
                unscaled = dense @ dense.T
                diag_np = np.diag(unscaled).astype(np.float32)
                diagonal.assign(diag_np)
                assemble(control_matrix, None)
                control.compute(control_matrix)
                control.solve(rhs, control_solution)
                wp.capture_launch(capture.graph)
                order = candidate.P.numpy()
                np.testing.assert_array_equal(order, control.P.numpy())
                if initial_order is None:
                    initial_order = order.copy()
                np.testing.assert_array_equal(order, initial_order)
                wp.capture_launch(capture.graph)
                matrix_np = control_matrix.numpy().reshape(n, n)
                np.testing.assert_allclose(
                    observed.numpy().reshape(n, n), matrix_np[np.ix_(order, order)], rtol=3.0e-6, atol=3.0e-7
                )
                scaling = np.sqrt(1.0 / (diag_np.astype(np.float64) + np.finfo(np.float32).eps))
                oracle = scaling[:, None] * unscaled * scaling[None, :] + np.eye(n) * 7.0e-7
                expected = np.linalg.solve(oracle, rhs_np.astype(np.float64))
                actual = candidate_solution.numpy().astype(np.float64)
                reference = control_solution.numpy().astype(np.float64)
                self.assertTrue(np.isfinite(actual).all())
                self.assertLessEqual(
                    _relative_error(actual, expected), max(2.0e-5, 1.25 * _relative_error(reference, expected))
                )
                self.assertLessEqual(
                    _relative_error(oracle @ actual, rhs_np),
                    max(2.0e-5, 1.25 * _relative_error(oracle @ reference, rhs_np)),
                )
                factor = np.tril(candidate.L.numpy().reshape(n, n)).astype(np.float64)
                self.assertLess(_relative_error(factor @ factor.T, oracle[np.ix_(order, order)]), 2.0e-6)
