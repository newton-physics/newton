# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense references for the structural bilateral/unilateral coupling cache."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.solvers.dvi import sparse
from newton._src.solvers.kamino._src.solvers.dvi.coupling import PackedCoupling
from newton._src.solvers.kamino._src.solvers.dvi.joint_blocks import JointBlockSolver
from newton.tests.kamino.test_kamino_joint_blocks import _fixture, _packed_matrix


class TestKaminoPackedCoupling(unittest.TestCase):
    def test_fallback_keeps_materialized_rhs(self):
        """Ignore the packed cache when dispatch selects the original bilateral solver."""
        state = SimpleNamespace(
            _sparse_coupling_allocated=True,
            bilateral_response_mio=None,
            bilateral_response_stride=None,
            bilateral_coupling=object(),
            bilateral_preconditioner=None,
            bilateral_rhs=object(),
            bilateral_solution=object(),
        )
        solver, packed, blocks = Mock(), Mock(), Mock()
        path = SimpleNamespace(
            size=SimpleNamespace(num_worlds=2, max_of_num_bilateral_joint_cts=16),
            device=SimpleNamespace(is_cuda=True),
            bilateral_solver=solver,
            bilateral_inverse=None,
            packed_coupling=packed,
            data=SimpleNamespace(
                state=state,
                solution=SimpleNamespace(lambdas=None),
                bilateral_operator=SimpleNamespace(info=SimpleNamespace(dim=None, vio=None)),
            ),
        )
        problem = SimpleNamespace(data=SimpleNamespace(vio=None, njc=None, v_f=None, dim=None))
        for selected in (blocks, None, blocks):
            packed.reset_mock()
            solver.reset_mock()
            with (
                patch.object(sparse, "_can_use_fused_bilateral_inverse", return_value=False),
                patch.object(wp, "launch") as launch,
            ):
                sparse._solve_sparse_bilateral_block(path, problem, joint_blocks=selected)
            rhs_calls = [
                call
                for call in launch.call_args_list
                if call.kwargs.get("kernel") is sparse._build_sparse_bilateral_rhs
            ]
            self.assertEqual(len(rhs_calls), int(selected is None))
            self.assertEqual(packed.build_rhs.call_count, int(selected is not None))
            self.assertEqual(solver.solve.call_count, int(selected is None))
            if rhs_calls:
                self.assertIs(rhs_calls[0].kwargs["inputs"][6], state.bilateral_coupling)

    def test_mixed_coupling_rhs_and_response(self):
        """Refresh mixed row mappings and match dense coupling, RHS, and response Grams."""
        if not wp.get_cuda_device_count():
            self.skipTest("Packed response requires CUDA tile kernels")
        device = wp.get_cuda_devices()[0]
        path, problem, matrices, matrix, mio, bvio, rio, _, scale, storage = _fixture(device)
        n, nu = 16, 6
        rng = np.random.default_rng(61293)

        def ints(values):
            return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

        def floats(values):
            return wp.array(np.asarray(values, dtype=np.float32).ravel(), dtype=wp.float32, device=device)

        base = path.jacobians._J_cts.bsm.nzb_coords.numpy()[:22].tolist()
        joint = base + [[row, body] for row in (16, 17) for body in (0, 6)]
        local = joint + [[18, 0]] + [[row, body] for body in (0, 6) for row in (19, 20, 21)]
        count = len(local)
        coords = np.asarray(local * 2, dtype=np.int32)
        raw = rng.normal(0, 0.2, (len(coords), 6)).astype(np.float32)
        mass = np.zeros((12, 12))
        for body in (0, 6):
            basis = rng.normal(0, 0.2, (6, 6))
            mass[body : body + 6, body : body + 6] = basis @ basis.T + np.eye(6)
        pvio = np.array([3, 29])
        p = rng.uniform(0.5, 1.5, 55).astype(np.float32)
        free = rng.normal(0, 0.1, 55).astype(np.float32)
        impulses = rng.normal(0, 0.1, 55).astype(np.float32)
        jacobian = SimpleNamespace(
            nzb_coords=ints(coords),
            nzb_start=ints([0, count]),
            num_nzb=ints([count, count]),
            nzb_values=wp.array(raw, dtype=vec6f, device=device),
        )
        weighted = wp.zeros(len(raw), dtype=vec6f, device=device)
        bsm = SimpleNamespace(
            nzb_coords=jacobian.nzb_coords, nzb_start=jacobian.nzb_start, num_nzb=jacobian.num_nzb, nzb_values=weighted
        )
        problem.delassus = SimpleNamespace(_needs_update=False, bsm=bsm, constraint_jacobian=jacobian)
        problem.data = SimpleNamespace(
            dim=ints([n + nu] * 2),
            njc=ints([n] * 2),
            nbc=ints([2, 2]),
            nl=ints([1, 1]),
            nc=ints([1, 1]),
            bcio=ints([0, 2]),
            lio=ints([0, 1]),
            cio=ints([0, 1]),
            vio=ints(pvio),
            P=floats(p),
            v_f=floats(free),
        )
        path.jacobians._J_cts.bsm = jacobian
        path.jacobians.joint_constraint_nzb_count = ints([len(joint)] * 2)
        path.jacobians.bounded_constraint_nzb_offsets = wp.array(
            [[w * count + 22 + 2 * row, w * count + 23 + 2 * row] for w in range(2) for row in range(2)],
            dtype=wp.vec2i,
            device=device,
        )
        path.jacobians.limit_constraint_nzb_offsets = ints([len(joint), count + len(joint)])
        path.jacobians.contact_constraint_nzb_offsets = ints(
            [w * count + len(joint) + 1 + 3 * c for w in range(2) for c in range(2)]
        )
        state = path.data.state
        state.limit_indices, state.contact_indices = ints([0, 1]), ints([0, 2])
        state.bilateral_rhs = wp.full(len(scale), -123.0, device=device)
        path.data.solution = SimpleNamespace(lambdas=floats(impulses))
        sparse._build_sparse_bilateral_row_nzb_topology(path, problem)
        blocks = JointBlockSolver.create(path)
        blocks._matrix.assign(_packed_matrix(blocks, matrix, mio, 20))
        blocks.prepare(path, problem)
        packed = PackedCoupling(path, blocks)
        order = blocks._schedule["order"].numpy()
        output = state.bilateral_response
        reference = wp.empty_like(output)
        for iteration, contact in enumerate((0, -1, 1, 0)):
            raw *= np.float32(0.9)
            p *= np.float32(1.01)
            jacobian.nzb_values.assign(raw)
            problem.data.P.assign(p)
            weighted.assign(
                np.asarray(
                    [
                        p[pvio[i // count] + row] * raw[i] @ mass[body : body + 6, body : body + 6]
                        for i, (row, body) in enumerate(coords)
                    ],
                    dtype=np.float32,
                )
            )
            state.contact_indices.assign([contact if contact < 0 else 2 * w + contact for w in range(2)])
            if iteration in (0, 2):
                packed.values.fill_(float("nan"))
                output.fill_(float("nan"))
            packed.assemble(path, problem)
            packed.build_rhs(path, problem)
            values, masks = packed.values.numpy(), packed.masks.numpy()
            expected_storage = storage.copy()
            expected_rhs = np.full(len(scale), -123.0, dtype=np.float32)
            expected_c = []
            for world in range(2):
                dense = np.zeros((n + nu, 12))
                selected = list(range(len(joint) + 1))
                if contact >= 0:
                    # Contact zero shares both bodies; contact one is unary.
                    selected += list(range(len(joint) + 1 + 3 * contact, len(joint) + 7))
                for index in selected:
                    row, body = local[index]
                    dense[row, body : body + 6] += p[pvio[world] + row] * raw[world * count + index]
                expected = (dense @ mass @ dense.T)[:n, n:]
                expected_c.append(expected)
                decoded = np.zeros((n, nu))
                for col in range(nu):
                    mask = int(masks[world * packed.width + col])
                    for padded, row in enumerate(order):
                        group, lane = divmod(padded, 6)
                        if row >= 0 and mask & (1 << group):
                            slot = (mask & ((1 << group) - 1)).bit_count()
                            decoded[row, col] = values[((world * packed.slots + slot) * 6 + lane) * packed.width + col]
                np.testing.assert_allclose(decoded, expected, atol=2e-6, rtol=2e-5)
                expected_storage[rio[world] : rio[world] + n * nu] = expected.ravel()
                expected_rhs[bvio[world] : bvio[world] + n] = -scale[bvio[world] : bvio[world] + n] * (
                    free[pvio[world] : pvio[world] + n] + expected @ impulses[pvio[world] + n : pvio[world] + n + nu]
                )
            np.testing.assert_allclose(state.bilateral_rhs.numpy(), expected_rhs, atol=2e-6, rtol=2e-5)
            state.bilateral_coupling.fill_(float("nan"))
            blocks.response(path, problem, coupling=packed)
            actual = output.numpy()
            state.bilateral_coupling.assign(expected_storage)
            state.bilateral_response = reference
            blocks.response(path, problem)
            state.bilateral_response = output
            expected = reference.numpy()
            active = np.zeros(len(actual), dtype=bool)
            for world in range(2):
                region = slice(rio[world], rio[world] + n * nu)
                active[region] = True
                np.testing.assert_allclose(actual[region], expected[region], atol=2e-6, rtol=2e-5)
                y = actual[region].reshape(n, nu).astype(np.float64)
                b = scale[bvio[world] : bvio[world] + n, None] * expected_c[world]
                np.testing.assert_allclose(y.T @ y, b.T @ np.linalg.solve(matrices[world], b), atol=2e-6, rtol=2e-5)
            self.assertTrue(np.isnan(actual[~active]).all())


if __name__ == "__main__":
    unittest.main()
