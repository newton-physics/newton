# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense references for cached bilateral right-hand sides."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.solvers.dvi.solver import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi.sparse import _build_sparse_bilateral_row_nzb_topology
from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    _assemble_sparse_bilateral_unilateral_coupling,
    _build_sparse_bilateral_rhs,
    make_sparse_bilateral_inverse_kernel,
)
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIState


class TestKaminoBilateralRHS(unittest.TestCase):
    def test_rhs_row_group_boundaries(self):
        """Match dense products across row-group tails, empty inputs, and padded layouts."""
        if not wp.is_cuda_available():
            self.skipTest("Cooperative RHS row groups require CUDA")
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(917)
        n, pvio, bvio, mio = 84, 3, 2, 5

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=device)

        def floats(values):
            return wp.array(values, dtype=wp.float32, device=device)

        for nu in (0, 15, 16, 17, 43, 65):
            stride = nu + 7
            coupling = rng.normal(0.0, 0.1, (n, nu)).astype(np.float32)
            impulses = rng.normal(size=pvio + n + nu + 3).astype(np.float32)
            free = rng.normal(size=impulses.size).astype(np.float32)
            scale = rng.uniform(0.5, 1.5, bvio + n + 3).astype(np.float32)
            expected = -scale[bvio : bvio + n] * (
                coupling.astype(np.float64) @ impulses[pvio + n : pvio + n + nu] + free[pvio : pvio + n]
            )
            for compact in (False, True):
                storage = np.full(mio + n * stride + 3, -999.0, dtype=np.float32)
                active_stride = nu if compact else stride
                storage[mio : mio + n * active_stride].reshape(n, active_stride)[:, :nu] = coupling
                for workers in (8, 16, 32):
                    with self.subTest(nu=nu, compact=compact, workers=workers):
                        result = wp.full(bvio + n + 3, -123.0, dtype=wp.float32, device=device)
                        wp.launch(
                            _build_sparse_bilateral_rhs,
                            dim=(1, n + 3, workers),
                            inputs=[
                                ints([pvio]),
                                ints([n]),
                                floats(free),
                                ints([n + nu]),
                                ints([mio]),
                                ints([stride]),
                                floats(storage),
                                floats(impulses),
                                compact,
                                workers,
                                ints([bvio]),
                                floats(scale),
                                result,
                            ],
                            device=device,
                            block_dim=128,
                        )
                        actual = result.numpy()
                        np.testing.assert_allclose(actual[bvio : bvio + n], expected, atol=2e-6, rtol=2e-6)
                        np.testing.assert_array_equal(actual[:bvio], -123.0)
                        np.testing.assert_array_equal(actual[bvio + n :], -123.0)

    def test_fused_matrix_free_inverse(self):
        """Match dense RHS and inverse products for both transpose layouts and inactive worlds."""
        if not wp.is_cuda_available():
            self.skipTest("The fused matrix-free inverse requires CUDA")
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(821)
        for n in (1, 7, 33, 128):
            # A seven-row boundary cuts through a column-major six-row block.
            rows = ((n + 5 + 5) // 6) * 6
            body_dofs = 18
            jacobian = rng.normal(size=(rows, body_dofs)).astype(np.float32) * 0.1
            weighted = rng.normal(size=(n, body_dofs)).astype(np.float32) * 0.1
            inverse = np.eye(n, dtype=np.float32) + np.full((n, n), 0.005, dtype=np.float32)
            pvio, bvio, mio = 3, 2, 5
            impulses = rng.normal(size=pvio + rows + 2).astype(np.float32)
            free = rng.normal(size=pvio + rows + 2).astype(np.float32)
            scale = rng.uniform(0.5, 1.5, size=bvio + n + 2).astype(np.float32)
            previous = rng.normal(size=bvio + n + 2).astype(np.float32)
            inverse_storage = np.concatenate((np.zeros(mio, dtype=np.float32), inverse.ravel()))
            body = jacobian[n:].astype(np.float64).T @ impulses[pvio + n : pvio + rows].astype(np.float64)
            expected_rhs = -scale[bvio : bvio + n] * (weighted.astype(np.float64) @ body + free[pvio : pvio + n])
            expected_solution = inverse.astype(np.float64).T @ expected_rhs
            coords = [(row, col) for row in range(n) for col in range(0, body_dofs, 6)]
            blocks = [weighted[row, col : col + 6] for row, col in coords]
            for column_major in (False, True):
                if column_major:
                    transpose_coords = [(row, col) for row in range(0, rows, 6) for col in range(body_dofs)]
                    transpose_values = [jacobian[row : row + 6, col] for row, col in transpose_coords]
                else:
                    transpose_coords = [(row, col) for row in range(rows) for col in range(0, body_dofs, 6)]
                    transpose_values = [jacobian[row, col : col + 6] for row, col in transpose_coords]
                for active in (False, True):
                    with self.subTest(n=n, column_major=column_major, active=active):

                        def ints(values):
                            return wp.array(values, dtype=wp.int32, device=device)

                        def floats(values):
                            return wp.array(values, dtype=wp.float32, device=device)

                        rhs = wp.full(bvio + n + 2, -123.0, dtype=wp.float32, device=device)
                        solution = floats(previous)
                        lambdas = floats(impulses)
                        wp.launch(
                            make_sparse_bilateral_inverse_kernel(32),
                            dim=(1, 128),
                            inputs=[
                                ints([pvio]),
                                ints([n]),
                                floats(free),
                                ints([n if active else 0]),
                                ints([mio]),
                                ints([bvio]),
                                floats(scale),
                                floats(inverse_storage),
                                rhs,
                                solution,
                                lambdas,
                                ints([len(transpose_coords)]),
                                ints([0]),
                                ints(transpose_coords),
                                wp.array(transpose_values, dtype=vec6f, device=device),
                                ints([pvio]),
                                ints([body_dofs]),
                                column_major,
                                ints([0]),
                                ints(np.arange(n + 1) * 3),
                                ints(np.arange(n * 3)),
                                ints(coords),
                                wp.array(blocks, dtype=vec6f, device=device),
                                0,  # world_start
                            ],
                            device=device,
                            block_dim=128,
                        )
                        actual_solution = solution.numpy()
                        expected = previous.copy()
                        if active:
                            expected[bvio : bvio + n] = expected_solution
                            np.testing.assert_allclose(rhs.numpy()[bvio : bvio + n], expected_rhs, rtol=2e-5, atol=2e-5)
                        np.testing.assert_allclose(actual_solution, expected, rtol=2e-5, atol=2e-5)
                        expected_lambdas = impulses.copy()
                        expected_lambdas[pvio : pvio + n] = scale[bvio : bvio + n] * expected[bvio : bvio + n]
                        np.testing.assert_allclose(lambdas.numpy(), expected_lambdas, rtol=2e-5, atol=2e-5)
                        np.testing.assert_array_equal(rhs.numpy()[:bvio], -123.0)
                        np.testing.assert_array_equal(rhs.numpy()[bvio + n :], -123.0)

    def test_large_non_schur_workspace_stays_matrix_free(self):
        """Avoid oversized coupling allocations and int32 errors in non-Schur solves."""
        solver = DVISolver()
        solver._device = wp.get_device("cpu")
        solver._size = SimpleNamespace(sum_of_max_inequalities=1, num_worlds=1, sum_of_max_total_cts=1)
        solver._joint_rows_host = [46341]
        solver._data = SimpleNamespace(
            state=DVIState(),
            bilateral_operator=SimpleNamespace(info=SimpleNamespace(total_vec_size=1)),
        )
        zeros = wp.zeros

        def bounded_zeros(shape, *args, **kwargs):
            self.assertLessEqual(shape, 2, "Non-Schur setup attempted a large coupling allocation")
            return zeros(shape, *args, **kwargs)

        for stride in (1000, 46341):
            with self.subTest(stride=stride), mock.patch.object(wp, "zeros", side_effect=bounded_zeros):
                solver._unilateral_strides_host = [stride]
                solver._allocate_projection_workspace(SimpleNamespace(sparse=True))
                self.assertEqual(solver.data.state.bilateral_coupling.size, 1)
                self.assertFalse(solver.data.state._sparse_coupling_allocated)

    def test_coupling_workspace_without_responses(self):
        """Cache only the coupling when alternating solves do not use Schur responses."""
        state = DVIState()
        size = SimpleNamespace(sum_of_max_inequalities=2, num_worlds=1, sum_of_max_total_cts=6)
        state.allocate_sparse_projection(size, [4], [2], 4, False, cache_bilateral_coupling=True)
        self.assertEqual(state.bilateral_coupling.size, 8)
        self.assertEqual(state.bilateral_response.size, 1)
        self.assertEqual(state.bilateral_response_factor.size, 1)
        cached = state.bilateral_coupling.ptr
        state.allocate_sparse_projection(size, [4], [2], 4, True)
        self.assertEqual(state.bilateral_coupling.ptr, cached)
        self.assertEqual(state.bilateral_response.size, 8)

    def test_mixed_bounded_row_order(self):
        """Resolve friction and effort rows independently of joint block order."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device):

                def ints(values, device=device):
                    return wp.array(values, dtype=wp.int32, device=device)

                def floats(values, device=device):
                    return wp.array(values, dtype=wp.float32, device=device)

                n, nu, stride = 2, 4, 10
                # Each joint stores its bilateral, friction and effort blocks
                # together. Constraint rows group friction before effort.
                rows = np.array([0, 2, 4, 1, 3, 5])
                coords = np.column_stack((rows, np.array([0, 0, 0, 6, 6, 6])))
                raw = np.zeros((6, 6), dtype=np.float32)
                raw[:, 0] = rows + 1
                scale = np.linspace(0.5, 1.5, n + nu).astype(np.float32)
                weighted = raw * scale[rows, None]
                jacobian = SimpleNamespace(nzb_start=ints([0]), nzb_coords=ints(coords))
                problem = SimpleNamespace(
                    delassus=SimpleNamespace(constraint_jacobian=jacobian),
                    data=SimpleNamespace(njc=ints([n]), nbc=ints([nu])),
                )
                path = SimpleNamespace(jacobians=SimpleNamespace(joint_constraint_nzb_count=ints([6])), device=device)
                _build_sparse_bilateral_row_nzb_topology(path, problem)
                dense_j = np.zeros((n + nu, 12))
                for block, (row, col) in enumerate(coords):
                    dense_j[row, col : col + 6] = raw[block] * scale[row]
                expected = (dense_j @ dense_j.T)[:n, n:]
                for compact in (False, True):
                    with self.subTest(compact=compact):
                        coupling = wp.full(n * stride + 7, -123.0, dtype=wp.float32, device=device)
                        wp.launch(
                            _assemble_sparse_bilateral_unilateral_coupling,
                            dim=(1, nu + 3, 3),
                            inputs=[
                                ints([6]),
                                ints([0]),
                                ints(coords),
                                wp.array(weighted, dtype=vec6f, device=device),
                                wp.array(raw, dtype=vec6f, device=device),
                                ints([n + nu]),
                                ints([n]),
                                ints([nu]),
                                ints([0]),
                                ints([0]),
                                ints([0]),
                                ints([0]),
                                ints([0]),
                                floats(scale),
                                ints([-1]),
                                ints([-1]),
                                ints([-1]),
                                ints([-1]),
                                *path.bilateral_row_nzb_topology,
                                ints([0]),
                                ints([stride]),
                                coupling,
                                3,
                                compact,
                            ],
                            device=device,
                        )
                        active_stride = nu if compact else stride
                        actual = coupling.numpy()
                        np.testing.assert_allclose(
                            actual[: n * active_stride].reshape(n, active_stride)[:, :nu], expected, atol=2e-6
                        )
                        np.testing.assert_array_equal(actual[n * active_stride :], -123.0)
                        lambdas = np.arange(n + nu, dtype=np.float32) * 0.1
                        free = np.arange(n + nu, dtype=np.float32) * 0.2
                        bilateral_scale = np.array([0.75, 1.25], dtype=np.float32)
                        result = wp.full(n + 7, -123.0, dtype=wp.float32, device=device)
                        reference = -bilateral_scale * (free[:n] + expected @ lambdas[n:])
                        for workers in (8, 16, 32) if device.is_cuda else (1,):
                            with self.subTest(workers=workers):
                                wp.launch(
                                    _build_sparse_bilateral_rhs,
                                    dim=(1, n + 3, workers),
                                    inputs=[
                                        ints([0]),
                                        ints([n]),
                                        floats(free),
                                        ints([n + nu]),
                                        ints([0]),
                                        ints([stride]),
                                        coupling,
                                        floats(lambdas),
                                        compact,
                                        workers,
                                        ints([0]),
                                        floats(bilateral_scale),
                                        result,
                                    ],
                                    block_dim=128 if device.is_cuda else 1,
                                    device=device,
                                )
                                np.testing.assert_allclose(result.numpy()[:n], reference, atol=2e-6)
                                np.testing.assert_array_equal(result.numpy()[n:], -123.0)
