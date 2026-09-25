# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for blocked and split Kamino DVI responses."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm import make_llt_blocked_rcm_solve_kernel
from newton._src.solvers.kamino._src.solvers.dvi.kernels import (
    _solve_bilateral_unilateral_response_compact,
    _solve_bilateral_unilateral_response_cooperative,
)
from newton._src.solvers.kamino._src.solvers.dvi.response import (
    _add_forward_bilateral_gradient,
    _update_forward_bilateral_rhs,
    make_response_kernel,
)
from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    _assemble_compact_unilateral_schur_tiled,
)


class TestKaminoBlockedResponse(unittest.TestCase):
    def test_dense_reference(self):
        """Verify partial tiles, permuted scaling, zero tiles, and guarded output."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires CUDA tile solves")
        device = wp.get_cuda_devices()[0]
        rng = np.random.default_rng(813)
        for n, nu in ((17, 9), (65, 13), (129, 37)):
            with self.subTest(n=n, nu=nu):
                lower = np.tril(rng.normal(0, 0.02, (n, n))).astype(np.float32)
                np.fill_diagonal(lower, 2.0)
                lower[32:, :32] = 0
                permutation = rng.permutation(n).astype(np.int32)
                scale = rng.uniform(0.5, 2, n).astype(np.float32)
                stride = nu + 5
                coupling = rng.normal(size=(n, stride)).astype(np.float32)
                expected = np.linalg.solve(lower.astype(np.float64), (scale[:, None] * coupling[:, :nu])[permutation])
                # Compact worlds store the coupling densely with row stride nu.
                coupling_input = np.zeros(n * stride, dtype=np.float32)
                coupling_input[: n * nu] = coupling[:, :nu].ravel()
                tiles = (n + 31) // 32
                pattern = np.ones((tiles, tiles), dtype=np.int32)
                pattern[1:, 0] = 0

                def ints(values):
                    return wp.array(values, dtype=wp.int32, device=device)

                def floats(values):
                    return wp.array(np.asarray(values).ravel(), dtype=wp.float32, device=device)

                output = wp.full(n * stride + 7, -123.0, device=device)
                wp.launch(
                    make_response_kernel(),
                    dim=((nu + 3) // 4, 128),
                    inputs=[
                        ints([n + nu]),
                        ints([n]),
                        ints([0]),
                        ints([0]),
                        floats(scale),
                        floats(lower),
                        ints(permutation),
                        ints([0]),
                        ints([stride]),
                        floats(coupling_input),
                        output,
                        ints([0]),
                        ints(pattern.ravel()),
                        nu,
                    ],
                    block_dim=128,
                    device=device,
                )
                result = output.numpy()
                np.testing.assert_allclose(result[: n * nu].reshape(n, nu), expected, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(result[n * nu :], -123.0)

                # Reuse these whitened columns to update a forward-solved RHS,
                # then recover the final original-coordinate solution.
                rhs = rng.normal(size=n).astype(np.float32)
                initial_y = np.linalg.solve(lower.astype(np.float64), rhs[permutation].astype(np.float64))
                y = wp.empty(n, dtype=wp.float32, device=device)
                untouched_x = wp.full(n, -123.0, device=device)
                untouched_x_hat = wp.full(n, -123.0, device=device)
                wp.launch(
                    make_llt_blocked_rcm_solve_kernel(32, True, False),
                    dim=(1, 128),
                    inputs=[
                        ints([n]),
                        ints([0]),
                        ints([0]),
                        ints([0]),
                        ints(permutation),
                        floats(lower),
                        ints(pattern.ravel()),
                        floats(rhs),
                        y,
                        untouched_x_hat,
                        untouched_x,
                    ],
                    block_dim=128,
                    device=device,
                )
                np.testing.assert_allclose(y.numpy(), initial_y, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(untouched_x.numpy(), -123.0)
                np.testing.assert_array_equal(untouched_x_hat.numpy(), -123.0)
                initial_gradient = rng.normal(size=n + nu).astype(np.float32)
                gradient = floats(initial_gradient)
                wp.launch(
                    _add_forward_bilateral_gradient,
                    dim=(1, nu + 3, 32),
                    inputs=[ints([n + nu]), ints([n]), ints([0]), ints([0]), ints([0]), output, y, gradient],
                    block_dim=128,
                    device=device,
                )
                expected_gradient = initial_gradient.astype(np.float64)
                expected_gradient[n:] += expected.T @ initial_y
                np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=2e-6, atol=2e-6)
                delta = rng.normal(0, 0.01, nu).astype(np.float32)
                wp.launch(
                    _update_forward_bilateral_rhs,
                    dim=(1, n, 32),
                    inputs=[
                        ints([n + nu]),
                        ints([n]),
                        ints([0]),
                        ints([0]),
                        ints([0]),
                        output,
                        floats(np.zeros(n + nu)),
                        floats(np.concatenate((np.zeros(n), delta))),
                        y,
                    ],
                    block_dim=128,
                    device=device,
                )
                x = wp.empty(n, dtype=wp.float32, device=device)
                wp.launch(
                    make_llt_blocked_rcm_solve_kernel(32, False),
                    dim=(1, 128),
                    inputs=[
                        ints([n]),
                        ints([0]),
                        ints([0]),
                        ints([0]),
                        ints(permutation),
                        floats(lower),
                        ints(pattern.ravel()),
                        floats(np.full(n, np.nan)),
                        y,
                        wp.empty_like(y),
                        x,
                    ],
                    block_dim=128,
                    device=device,
                )
                expected_y = initial_y.astype(np.float64) - expected @ delta.astype(np.float64)
                expected_x = np.empty(n)
                expected_x[permutation] = np.linalg.solve(lower.astype(np.float64).T, expected_y)
                np.testing.assert_allclose(y.numpy(), expected_y, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(x.numpy(), expected_x, rtol=2e-6, atol=2e-6)


class TestCooperativeResponse(unittest.TestCase):
    def test_unpermuted_compact_response(self):
        """Match dense elimination without permutation and with an inactive column."""
        if not wp.get_cuda_device_count():
            self.skipTest("Cooperative response construction requires CUDA")
        device = wp.get_cuda_devices()[0]
        rng = np.random.default_rng(4282)
        njc, nu = 33, 3
        lower = np.tril(rng.normal(0.0, 0.02, (njc, njc))).astype(np.float32)
        np.fill_diagonal(lower, 2.0)
        scale = rng.uniform(0.5, 1.5, njc).astype(np.float32)
        coupling = rng.normal(size=(njc, nu)).astype(np.float32)
        coupling[:, -1] = 0.0
        expected = np.linalg.solve(lower.astype(np.float64), scale[:, None] * coupling)

        def i32(values):
            return wp.array(values, dtype=wp.int32, device=device)

        def f32(values):
            return wp.array(np.asarray(values).ravel(), dtype=wp.float32, device=device)

        response = wp.full(njc * nu, -123.0, dtype=wp.float32, device=device)
        wp.launch(
            _solve_bilateral_unilateral_response_cooperative,
            dim=2 * 32,
            block_dim=128,
            inputs=[
                i32([njc + nu]),
                i32([njc]),
                i32([0]),
                i32([0]),
                f32(scale),
                f32(lower),
                i32([-1] * njc),
                False,
                i32([0]),
                i32([nu]),
                f32(coupling),
                wp.empty(njc * nu, dtype=wp.float32, device=device),
                response,
                0,
                2,
                True,
                i32([0] * njc),
                False,
            ],
            device=device,
        )
        actual = response.numpy().reshape(njc, nu)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_array_equal(actual[:, -1], 0.0)


def _assert_bits_equal(a, b):
    np.testing.assert_array_equal(a.copy().view(np.uint32), b.copy().view(np.uint32))


class TestSplitResponse(unittest.TestCase):
    def test_captured_capacity_boundaries(self):
        """Preserve response accuracy and fallback bits across changing compact capacity."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Captured split response requires CUDA")
        rng = np.random.default_rng(812)
        ns = np.array([84, 32, 0], dtype=np.int32)
        capacities = np.array([128, 128, 8], dtype=np.int32)
        guard = 11
        sizes = ns * capacities
        offsets = np.concatenate(([guard], guard + np.cumsum(sizes + guard)[:-1])).astype(np.int32)
        mio = np.concatenate(([0], np.cumsum(ns * ns)[:-1])).astype(np.int32)
        vio = np.concatenate(([0], np.cumsum(ns)[:-1])).astype(np.int32)
        qio = np.concatenate(([0], np.cumsum(ns + capacities)[:-1])).astype(np.int32)
        factors, scales, orders, prefixes, jacobians = [], [], [], [], []
        local_factors, local_scales, local_orders = [], [], []
        for n in ns:
            bodies = max(2, int(n) // 4 + 2)
            J = np.zeros((n, bodies * 6), dtype=np.float64)
            for row in range(n):
                body = row % bodies
                J[row, body * 6 : body * 6 + 6] = rng.normal(size=6)
                other = (body + 1) % bodies
                J[row, other * 6 : other * 6 + 6] = rng.normal(size=6)
            D = J @ J.T
            scale = np.sqrt(1.0 / (np.diag(D) + 1.0)).astype(np.float32)
            order = rng.permutation(n).astype(np.int32)
            A = scale[:, None] * D * scale[None, :] + np.eye(n)
            L = np.linalg.cholesky(A[np.ix_(order, order)]).astype(np.float32)
            prefix = [int(np.flatnonzero(L[row])[0]) // 16 * 16 for row in range(n)]
            factors.extend(L.ravel())
            scales.extend(scale)
            orders.extend(order)
            prefixes.extend(prefix)
            jacobians.append(J)
            local_factors.append(L)
            local_scales.append(scale)
            local_orders.append(order)

        def ints(x):
            return wp.array(np.asarray(x, dtype=np.int32), dtype=wp.int32, device=device)

        def floats(x):
            return wp.array(np.asarray(x, dtype=np.float32), dtype=wp.float32, device=device)

        dim = ints(ns)
        njc, bmio, bvio, rio, stride, qoffset = map(ints, (ns, mio, vio, offsets, capacities, qio))
        L, scale, order, prefix = floats(factors), floats(scales), ints(orders), ints(prefixes)
        total = int(np.sum(sizes + guard) + guard)
        coupling = wp.zeros(total, dtype=wp.float32, device=device)
        outputs = [wp.zeros_like(coupling) for _ in range(2)]
        scratch = [wp.zeros_like(coupling) for _ in range(2)]
        q = [wp.zeros(int(np.sum(ns + capacities)), dtype=wp.float32, device=device) for _ in range(2)]

        def launch(index):
            if index:
                wp.launch(
                    _solve_bilateral_unilateral_response_compact,
                    dim=(len(ns), 128),
                    block_dim=128,
                    inputs=[dim, njc, bmio, bvio, scale, L, order, rio, stride, coupling, outputs[index], prefix],
                    device=device,
                )
            wp.launch(
                _solve_bilateral_unilateral_response_cooperative,
                dim=len(ns) * 64 * 32,
                block_dim=256,
                inputs=[
                    dim,
                    njc,
                    bmio,
                    bvio,
                    scale,
                    L,
                    order,
                    True,
                    rio,
                    stride,
                    coupling,
                    scratch[index],
                    outputs[index],
                    0,
                    64,
                    True,
                    prefix,
                    bool(index),
                ],
                device=device,
            )
            wp.launch(
                _assemble_compact_unilateral_schur_tiled,
                dim=(len(ns), 16, 128),
                block_dim=128,
                inputs=[dim, njc, qoffset, rio, stride, outputs[index], scratch[index], q[index], 16, 0],
                device=device,
            )

        graphs = []
        for index in range(2):
            launch(index)
            with wp.ScopedCapture(device=device) as capture:
                launch(index)
            graphs.append(capture.graph)
        for nus in ([103, 64, 0], [104, 65, 1], [0, 0, 0], [125, 64, 0], [103, 65, 0]):
            dim.assign(ns + np.asarray(nus, dtype=np.int32))
            rhs = np.full(total, -999.0, dtype=np.float32)
            references = []
            for world, (n, nu, capacity, off) in enumerate(zip(ns, nus, capacities, offsets, strict=True)):
                J = jacobians[world]
                W = np.zeros((J.shape[1], nu), dtype=np.float64)
                for col in range(max(0, nu - 1)):
                    body = int(rng.integers(J.shape[1] // 6))
                    W[body * 6 : body * 6 + 6, col] = rng.normal(0, 0.1, 6)
                C = (J @ W).astype(np.float32)
                if nu * nu <= n * capacity:
                    # Compact worlds store the coupling densely with row stride nu.
                    rhs[off : off + n * nu] = C.ravel()
                else:
                    view = rhs[off : off + n * capacity].reshape(n, capacity)
                    view[:, :nu] = C
                b = (local_scales[world][:, None] * C)[local_orders[world]]
                white = np.linalg.solve(local_factors[world].astype(np.float64), b.astype(np.float64))
                references.append(white)
            coupling.assign(rhs)
            for index in range(2):
                outputs[index].fill_(-456.0)
                scratch[index].fill_(-123.0)
                q[index].fill_(-789.0)
                wp.capture_launch(graphs[index])
            a, b = [x.numpy() for x in outputs]
            sa, sb = [x.numpy() for x in scratch]
            expected_q = np.full(q[0].size, -789.0, dtype=np.float32)
            gap_mask = np.ones(total, dtype=bool)
            for world, (n, nu, capacity, off) in enumerate(zip(ns, nus, capacities, offsets, strict=True)):
                size = n * capacity
                gap_mask[off : off + size] = False
                compact = nu * nu <= size
                if compact:
                    white = references[world]
                    np.testing.assert_allclose(b[off : off + n * nu].reshape(n, nu), white, atol=2.0e-6, rtol=2.0e-6)
                    np.testing.assert_allclose(
                        sb[off : off + nu * nu].reshape(nu, nu), white.T @ white, atol=5.0e-6, rtol=5.0e-6
                    )
                    np.testing.assert_array_equal(b[off + n * nu : off + size], -456.0)
                    np.testing.assert_array_equal(sb[off + nu * nu : off + size], -123.0)
                    expected_q[qio[world] + n : qio[world] + n + nu] = 0.0
                    if nu:
                        np.testing.assert_array_equal(b[off : off + n * nu].reshape(n, nu)[:, -1], 0.0)
                        np.testing.assert_array_equal(sb[off : off + nu * nu].reshape(nu, nu)[-1], 0.0)
                else:
                    _assert_bits_equal(b[off : off + size], a[off : off + size])
                    _assert_bits_equal(sb[off : off + size], sa[off : off + size])
                    inactive = b[off : off + size].reshape(n, capacity)[:, nu:]
                    np.testing.assert_array_equal(inactive, -456.0)
            np.testing.assert_array_equal(b[gap_mask], -456.0)
            np.testing.assert_array_equal(sb[gap_mask], -123.0)
            np.testing.assert_array_equal(q[1].numpy(), expected_q)
            np.testing.assert_array_equal(q[0].numpy(), expected_q)
