# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for Kamino DVI Schur assembly, sweeps, and residuals."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi.kernels import _compute_dvi_status_residuals
from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    _assemble_compact_unilateral_schur_blocked,
    _solve_dvi_compact_schur_pgs_cooperative,
    _solve_dvi_sparse_inequalities_pgs_cooperative,
)
from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    _prepare_full_sparse_unilateral_schur as kernel,
)
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIConfigStruct, DVIStatus


class TestKaminoCompactSchur(unittest.TestCase):
    def setUp(self):
        if not wp.get_cuda_device_count():
            self.skipTest("Compact tile and warp kernels require CUDA")
        self.device = wp.get_cuda_devices()[0]

    def ints(self, values):
        return wp.array(values, dtype=wp.int32, device=self.device)

    def floats(self, values):
        return wp.array(np.asarray(values, dtype=np.float32).ravel(), dtype=wp.float32, device=self.device)

    def test_blocked_gram_boundaries(self):
        """Match a dense Gram oracle and preserve guards at empty, partial, and fallback sizes."""
        rng = np.random.default_rng(28931)
        cases = ((0, 0), (32, 0), (1, 1), (33, 3), (65, 31), (129, 65), (129, 127), (129, 128), (129, 129))
        for n, nu in cases:
            with self.subTest(n=n, nu=nu):
                stride = max(nu, 1)
                offset = 7
                capacity = n * stride
                white = rng.normal(size=(n, nu)).astype(np.float32)
                response = self.floats(np.pad(white.ravel(), (offset, capacity - n * nu + 7)))
                result = wp.full(offset + capacity + 7, -123.0, device=self.device)
                q = wp.full(n + nu + 7, -77.0, device=self.device)
                wp.launch(
                    _assemble_compact_unilateral_schur_blocked,
                    dim=(1, 256),
                    inputs=[
                        self.ints([n + nu]),
                        self.ints([n]),
                        self.ints([0]),
                        self.ints([offset]),
                        self.ints([stride]),
                        response,
                        result,
                        q,
                    ],
                    device=self.device,
                    block_dim=256,
                )
                expected = np.full(offset + capacity + 7, -123.0, dtype=np.float32)
                expected_q = np.full(n + nu + 7, -77.0, dtype=np.float32)
                if nu <= 128 and nu * nu <= capacity:
                    expected[offset : offset + nu * nu] = (white.astype(np.float64).T @ white).ravel()
                    expected_q[n : n + nu] = 0.0
                np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=5e-5)
                np.testing.assert_array_equal(q.numpy(), expected_q)

    def test_pipelined_sweeps_match_general_kernel(self):
        """Preserve mixed-constraint updates and reversed schedules through 128 compact rows."""
        rng = np.random.default_rng(28932)
        for nb, nl, nc in ((0, 0, 1), (2, 2, 3), (31, 1, 32), (0, 0, 42), (7, 0, 2)):
            with self.subTest(bounded=nb, limits=nl, contacts=nc):
                nu = nb + nl + 3 * nc
                n = max(32, nu)
                slots = nb + nl + nc
                dense = rng.normal(0, 0.1, (nu, nu)).astype(np.float32)
                operator = dense @ dense.T + np.eye(nu, dtype=np.float32)
                q0 = rng.normal(size=nu).astype(np.float32)
                initial = rng.uniform(0.0, 0.1, n + nu).astype(np.float32)
                order = rng.permutation(slots).astype(np.int32)
                # Reverse colors and group-local rows independently, preserving group order.
                group_bounds = np.unique(np.linspace(0, slots, min(4, slots) + 1).astype(np.int32))
                groups = len(group_bounds) - 1
                color_bounds = np.unique(np.array([0, groups // 2, groups], dtype=np.int32))
                config = DVIConfigStruct()
                # A non-fused iteration selects the general implementation without
                # its dispatch to the pipeline. One outer iteration makes the two
                # schedules equivalent, including odd tangent sweeps.
                config.max_alternating_iterations = 1
                config.inequality_sweeps_per_iteration = 8
                config.regularization = 0.001
                config.omega = 0.8
                config.tolerance = 1e-5
                data = {
                    "problem_nbc": self.ints([nb]),
                    "problem_nl": self.ints([nl]),
                    "problem_nc": self.ints([nc]),
                    "problem_njc": self.ints([n]),
                    "problem_bcio": self.ints([0]),
                    "problem_lio": self.ints([0]),
                    "problem_cio": self.ints([0]),
                    "problem_uio": self.ints([0]),
                    "problem_bcgo": self.ints([n]),
                    "problem_lcgo": self.ints([n + nb]),
                    "problem_ccgo": self.ints([n + nb + nl]),
                    "problem_vio": self.ints([0]),
                    "bilateral_vio": self.ints([0]),
                    "response_mio": self.ints([0]),
                    "response_stride": self.ints([nu]),
                    "limit_indices": self.ints(list(range(nl))),
                    "contact_indices": self.ints(list(range(nc))),
                    "problem_mu": self.floats(np.full(nc, 0.6)),
                    "problem_bound_lower": self.floats(np.full(nb, -0.3)),
                    "problem_bound_upper": self.floats(np.full(nb, 0.7)),
                    "problem_P": self.floats(np.ones(n + nu)),
                    "problem_v_b": self.floats(np.zeros(n + nu)),
                    "problem_diag": self.floats(np.concatenate([np.ones(n), operator.diagonal()])),
                    "projected_diag": self.floats(np.concatenate([np.ones(n), operator.diagonal()])),
                    "compact_schur": self.floats(-operator.T),
                    "inequality_num_colors": self.ints([len(color_bounds) - 1]),
                    "inequality_ids_by_color": self.ints(order),
                    "inequality_color_starts": self.ints(color_bounds),
                    "inequality_group_starts": self.ints(group_bounds),
                    "solver_config": wp.array([config], dtype=DVIConfigStruct, device=self.device),
                    "enable_compact_schur": True,
                    "block_iteration": 0,
                }
                results = []
                for kernel in (
                    _solve_dvi_sparse_inequalities_pgs_cooperative,
                    _solve_dvi_compact_schur_pgs_cooperative,
                ):
                    data["compact_q"] = self.floats(np.concatenate([np.zeros(n), q0]))
                    data["solution_lambdas"] = self.floats(initial)
                    data["solver_status"] = wp.zeros(1, dtype=DVIStatus, device=self.device)
                    inputs = []
                    for arg in kernel.adj.args:
                        if arg.label in data:
                            inputs.append(data[arg.label])
                        else:
                            # Full compact sweeps never read sparse body-space inputs.
                            shape = (1, 2) if arg.type.ndim == 2 else 1
                            inputs.append(wp.zeros(shape, dtype=arg.type.dtype, device=self.device))
                    wp.launch(kernel, dim=32, inputs=inputs, device=self.device, block_dim=32)
                    results.append((data["solution_lambdas"].numpy(), data["compact_q"].numpy()))
                np.testing.assert_array_equal(results[1][0][:n], initial[:n])
                np.testing.assert_allclose(results[0][0], results[1][0], atol=3e-6, rtol=3e-6)
                np.testing.assert_allclose(results[0][1], results[1][1], atol=3e-6, rtol=3e-6)


class TestKaminoFullSchurAssembly(unittest.TestCase):
    def test_dense_oracle(self):
        """Cover mixed row types, capacity boundaries, and inactive worlds."""
        rng = np.random.default_rng(123)
        cases = [
            (33, 3, 2, 2, 16),
            (256, 31, 1, 32, 256),
            (256, 32, 1, 32, 256),
            (600, 512, 0, 0, 600),
            (600, 513, 0, 0, 600),
            (2, 1, 1, 2, 2),
            (33, 0, 0, 0, 8),
            (33, 3, 2, 2, 16),
        ]
        data = {
            name: []
            for name in (
                "bsm_num_nzb",
                "bsm_nzb_start",
                "bsm_nzb_coords",
                "bsm_nzb_values",
                "jacobian_nzb_values",
                "bsm_row_start",
                "bsm_col_start",
                "bounded_nzb_offsets",
                "limit_nzb_offsets",
                "contact_nzb_offsets",
                "limit_indices",
                "contact_indices",
                "problem_nbc",
                "problem_nl",
                "problem_nc",
                "problem_bcio",
                "problem_lio",
                "problem_cio",
                "problem_vio",
                "problem_P",
                "problem_v_f",
                "eta",
                "problem_njc",
                "response_mio",
                "response_stride",
                "compact_schur",
                "compact_q",
                "solver_config",
                "body_space",
                "solution_lambdas",
            )
        }
        expected_s, expected_q = [], []

        for wid, (n, nb, nl, nc, stride) in enumerate(cases):
            nu = nb + nl + 3 * nc
            weighted = np.zeros((nu, 12))
            jacobian = np.zeros((nu, 12))
            start = len(data["bsm_nzb_values"])
            data["bsm_nzb_start"].append(start)
            data["bsm_row_start"].append(len(data["eta"]))
            data["bsm_col_start"].append(len(data["body_space"]))
            data["problem_bcio"].append(len(data["bounded_nzb_offsets"]))
            data["problem_lio"].append(len(data["limit_indices"]))
            data["problem_cio"].append(len(data["contact_indices"]))
            data["problem_vio"].append(len(data["problem_P"]))
            data["response_mio"].append(len(data["compact_schur"]))
            for name, value in (
                ("problem_njc", n),
                ("problem_nbc", nb),
                ("problem_nl", nl),
                ("problem_nc", nc),
                ("response_stride", stride),
            ):
                data[name].append(value)

            def block(row, body, active=True, n=n, weighted=weighted, jacobian=jacobian):
                index = len(data["bsm_nzb_values"])
                w, j = rng.normal(size=(2, 6)).astype(np.float32)
                data["bsm_nzb_coords"].append((n + row, body * 6))
                data["bsm_nzb_values"].append(w)
                data["jacobian_nzb_values"].append(j)
                if active:
                    weighted[row, body * 6 : body * 6 + 6] = w
                    jacobian[row, body * 6 : body * 6 + 6] = j
                return index

            for row in range(nb):
                first = block(row, 0)
                second = block(row, 1) if row % 2 else -1
                data["bounded_nzb_offsets"].append((first, second))
            for limit in range(nl):
                active = limit % 2 == 0
                data["limit_indices"].append(len(data["limit_nzb_offsets"]) if active else -1)
                data["limit_nzb_offsets"].append(block(nb + limit, 0, active))
                block(nb + limit, 1, active)
            for contact in range(nc):
                data["contact_indices"].append(len(data["contact_nzb_offsets"]))
                data["contact_nzb_offsets"].append(len(data["bsm_nzb_values"]))
                for body in range(1 + contact % 2):
                    for component in range(3):
                        block(nb + nl + contact * 3 + component, body)
            data["bsm_num_nzb"].append(len(data["bsm_nzb_values"]) - start)
            p, vf, eta, lambdas, q = rng.normal(size=(5, n + nu)).astype(np.float32)
            body = rng.normal(size=12).astype(np.float32)
            schur = rng.normal(size=n * stride).astype(np.float32)
            expected_matrix, expected_velocity = schur.copy(), q.copy()
            if wid != len(cases) - 1 and 0 < nu <= 512 and nu * nu <= n * stride:
                operator = (weighted @ jacobian.T) * p[None, n:] + np.diag(eta[n:])
                expected_matrix[: nu * nu] -= operator.T.ravel()
                expected_velocity[n:] = weighted @ body + eta[n:] * lambdas[n:] + vf[n:]
            expected_s.extend(expected_matrix)
            expected_q.extend(expected_velocity)
            for name, values in (
                ("problem_P", p),
                ("problem_v_f", vf),
                ("eta", eta),
                ("solution_lambdas", lambdas),
                ("compact_q", q),
                ("body_space", body),
                ("compact_schur", schur),
            ):
                data[name].extend(values)
            config = DVIConfigStruct()
            config.max_alternating_iterations = 0 if wid == len(cases) - 1 else 8
            data["solver_config"].append(config)

        for device in wp.get_devices():
            arrays = {}
            for arg in kernel.adj.args:
                name = arg.label
                if name == "enable_compact_schur":
                    arrays[name] = True
                elif name == "workers_per_world":
                    arrays[name] = 128
                elif name == "block_iteration":
                    arrays[name] = 0
                else:
                    arrays[name] = wp.array(data[name], dtype=arg.type.dtype, device=device)
            reference_s = None
            reference_q = None
            for workers in (32, 128, 4096):
                arrays["workers_per_world"] = workers
                arrays["compact_schur"].assign(np.asarray(data["compact_schur"], dtype=np.float32))
                arrays["compact_q"].assign(np.asarray(data["compact_q"], dtype=np.float32))
                wp.launch(
                    kernel,
                    dim=len(cases) * workers,
                    inputs=[arrays[arg.label] for arg in kernel.adj.args],
                    block_dim=128,
                    device=device,
                )
                actual_s = arrays["compact_schur"].numpy()
                actual_q = arrays["compact_q"].numpy()
                np.testing.assert_allclose(actual_s, expected_s, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(actual_q, expected_q, atol=1e-5, rtol=1e-5)
                if reference_s is not None:
                    np.testing.assert_array_equal(actual_s, reference_s)
                    np.testing.assert_array_equal(actual_q, reference_q)
                reference_s, reference_q = actual_s, actual_q


class TestDVIResiduals(unittest.TestCase):
    def test_cooperative_matches_serial(self):
        """Preserve all status fields across ragged worlds and constraint types."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires CUDA warp reductions")
        device = wp.get_cuda_devices()[0]
        rng = np.random.default_rng(916)
        counts = np.array(
            [(137, 53, 9, 7), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 3, 5), (65, 33, 0, 0), (3, 1, 1, 1)], dtype=np.int32
        )
        njc, nbc, nl, nc = counts.T
        dimensions = njc + nbc + nl + 3 * nc
        vector_offsets = np.concatenate(([0], np.cumsum(dimensions)[:-1]))
        bounded_offsets = np.concatenate(([0], np.cumsum(nbc)[:-1]))
        contact_offsets = np.concatenate(([0], np.cumsum(nc)[:-1]))
        size = int(dimensions.sum())

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=device)

        def floats(values):
            return wp.array(values, dtype=wp.float32, device=device)

        configs = []
        statuses = []
        for index in range(len(counts)):
            config = DVIConfigStruct()
            config.tolerance = 1.0e-5
            configs.append(config)
            status = DVIStatus()
            status.iterations = index
            statuses.append(status)
        initial_status = wp.array(statuses, dtype=DVIStatus, device=device)
        inputs = [
            ints(dimensions),
            ints(vector_offsets),
            ints(njc),
            ints(nbc),
            ints(nl),
            ints(nc),
            ints(njc),
            ints(njc + nbc),
            ints(njc + nbc + nl),
            ints(bounded_offsets),
            ints(contact_offsets),
            floats(rng.uniform(0, 1, int(nc.sum()))),
            floats(-rng.uniform(0, 2, int(nbc.sum()))),
            floats(rng.uniform(0, 2, int(nbc.sum()))),
            wp.array(configs, dtype=DVIConfigStruct, device=device),
        ]
        for converged in (False, True):
            velocities = np.zeros(size, dtype=np.float32) if converged else rng.normal(size=size).astype(np.float32)
            impulses = np.zeros(size, dtype=np.float32) if converged else rng.normal(size=size).astype(np.float32)
            outputs = []
            for workers in (1, 32):
                status = wp.clone(initial_status)
                wp.launch(
                    _compute_dvi_status_residuals,
                    dim=len(counts) * workers,
                    inputs=[*inputs, floats(velocities), floats(impulses), status, workers],
                    block_dim=128,
                    device=device,
                )
                outputs.append(status.numpy())
            np.testing.assert_array_equal(outputs[0], outputs[1])
            if converged:
                np.testing.assert_array_equal(outputs[1]["converged"], 1)
                np.testing.assert_array_equal(outputs[1]["r_b"], 0)
                np.testing.assert_array_equal(outputs[1]["r_p"], 0)
                np.testing.assert_array_equal(outputs[1]["r_d"], 0)
                np.testing.assert_array_equal(outputs[1]["r_c"], 0)
            else:
                # The single bilateral row has an analytic residual, including
                # when its world shares a CUDA block with empty/contact worlds.
                residual = abs(velocities[vector_offsets[2]])
                self.assertEqual(outputs[1]["r_b"][2], residual)
                self.assertEqual(outputs[1]["r_d"][2], residual)
