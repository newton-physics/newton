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
)
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIState


class TestKaminoBilateralRHS(unittest.TestCase):
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
                # The old per-joint bounded offsets interleave row groups.
                per_joint_offsets = wp.array([[1, -1], [2, -1], [4, -1], [5, -1]], dtype=wp.vec2i, device=device)
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
                                ints([0]),
                                floats(scale),
                                ints([-1]),
                                ints([-1]),
                                per_joint_offsets,
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
                        for workers in (8, 32) if device.is_cuda else (1,):
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
