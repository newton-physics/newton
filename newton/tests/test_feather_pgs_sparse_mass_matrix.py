# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sparse_mass_matrix import (
    _get_crba_sparse_factor_kernel,
    _SparseMassMatrixPlan,
    solve_sparse_mass_matrix,
)


def _branched_humanoid_topology():
    """Return 43 DOFs with a free base, legs, spine, arms, and branching fingers."""
    parents, counts = [-1], [6]

    def chain(parent, length):
        for _ in range(length):
            parents.append(parent)
            counts.append(1)
            parent = len(parents) - 1
        return parent

    for _ in range(2):
        chain(0, 6)
    spine = chain(0, 3)
    for _ in range(2):
        hand = chain(spine, 7)
        for _ in range(4):
            chain(hand, 1)
    return parents, counts


class TestSparseMassMatrix(unittest.TestCase):
    def test_factor_schedules_follow_dependencies(self):
        """Cover each packed factor entry once and schedule independent branch pivots together."""
        for parents, counts in [([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2]), _branched_humanoid_topology()]:
            plan = _SparseMassMatrixPlan.build(parents, counts)
            levels = np.empty(plan.dof_count, dtype=int)
            for level in range(plan.factor_level_count):
                begin, end = plan.factor_level_offsets[level : level + 2]
                levels[plan.factor_level_columns[begin:end]] = level
            np.testing.assert_array_equal(np.sort(plan.factor_level_columns), np.arange(plan.dof_count))
            np.testing.assert_array_equal(np.sort(plan.column_entries), np.arange(plan.nonzero_count))
            np.testing.assert_array_equal(
                np.sort(plan.factor_level_entries), np.flatnonzero(plan.entry_rows != plan.columns)
            )
            for row in range(plan.dof_count):
                begin, end = plan.row_offsets[row : row + 2]
                first = row - (end - begin) + 1
                np.testing.assert_array_equal(plan.columns[begin:end], np.arange(first, row + 1))
                self.assertTrue(np.all(levels[first:row] < levels[row]))
                begin, end = plan.column_offsets[row : row + 2]
                entries = plan.column_entries[begin:end]
                np.testing.assert_array_equal(plan.columns[entries], row)
                self.assertTrue(np.all(np.diff(plan.entry_rows[entries]) > 0))
            if plan.dof_count == 43:
                self.assertLess(plan.factor_level_count, plan.dof_count // 2)

    def test_forest_fixed_multidof_order(self):
        """Derive fill-free support for unordered forests and fixed joints."""
        plan = _SparseMassMatrixPlan.build([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2])
        np.testing.assert_array_equal(np.sort(plan.permutation), np.arange(9))
        np.testing.assert_array_equal(plan.inverse_permutation[plan.permutation], np.arange(9))
        self.assertEqual(plan.endpoint_support(2).size, 3)
        self.assertEqual(plan.endpoint_support(0, 5).size, 8)
        for joint in range(6):
            support = plan.endpoint_support(joint)
            mask = sum(1 << int(i) for i in support)
            self.assertEqual(int(plan.joint_ancestor_mask[joint]), mask)
        rng = np.random.default_rng(54)
        lower = np.zeros((9, 9))
        lower[plan.entry_rows, plan.columns] = rng.normal(size=plan.nonzero_count)
        lower[np.diag_indices(9)] = 2.0
        inverse = np.linalg.inv(lower)
        np.testing.assert_allclose(inverse[plan.lookup < 0], 0.0, atol=1.0e-14)
        matrix = lower @ lower.T
        structural = (plan.lookup >= 0) | (plan.lookup.T >= 0)
        np.testing.assert_allclose(matrix[~structural], 0.0, atol=1.0e-14)

    def test_invalid_topology(self):
        """Reject cycles, invalid parents, counts, and unsupported mask widths."""
        for parents, counts in [
            ([1, 0], [1, 1]),
            ([2], [1]),
            ([-2], [1]),
            ([-1], [-1]),
            ([-1], [1.5]),
            ([-1], [65]),
            ([-1], []),
            ([], []),
        ]:
            with self.subTest(parents=parents, counts=counts), self.assertRaises(ValueError):
                _SparseMassMatrixPlan.build(parents, counts)

    def _factor(self, parents, counts, *, fused_drive=True, mask=None, armature=0.4, device="cpu"):
        plan = _SparseMassMatrixPlan.build(parents, counts)
        n, j = plan.dof_count, len(parents)
        rng = np.random.default_rng(742)
        motion = rng.normal(size=(2 * n, 6)).astype(np.float32)
        body_inertia = np.empty((2 * j, 6, 6), dtype=np.float32)
        composite = np.zeros_like(body_inertia)
        dense = np.zeros((2, n, n))
        for art in range(2):
            for joint in range(j):
                q = rng.normal(size=(6, 6))
                body_inertia[art * j + joint] = q @ q.T + np.eye(6)
                physical_support = plan.permutation[plan.endpoint_support(joint)]
                jacobian = np.zeros((6, n))
                jacobian[:, physical_support] = motion[art * n + physical_support].T
                dense[art] += jacobian.T @ body_inertia[art * j + joint] @ jacobian
                ancestor = joint
                while ancestor != -1:
                    composite[art * j + ancestor] += body_inertia[art * j + joint]
                    ancestor = parents[ancestor]
        drive_map = np.arange(2 * n, dtype=np.int32)
        drive_map[::3] = -1
        stiffness = rng.uniform(-0.1, 0.5, size=2 * n).astype(np.float32)
        regularizer = np.full((2, n), armature, dtype=np.float32)
        mapping = np.array([1, 0], dtype=np.int32)
        for group, art in enumerate(mapping):
            diagonal = regularizer[group].astype(np.float64)
            if fused_drive:
                diagonal += np.where(
                    drive_map[art * n : (art + 1) * n] >= 0, np.maximum(stiffness[art * n : (art + 1) * n], 0.0), 0.0
                )
            dense[art] += np.diag(diagonal)

        def array(value, dtype):
            return wp.array(value, dtype=dtype, device=device)

        indices = plan.to_device(device)
        inverse = wp.full((2, plan.nonzero_count), -98.0, dtype=float, device=device)
        status = wp.full(2, 17, dtype=int, device=device)
        if mask is None:
            mask = [1, 1]
        joint_child = rng.permutation(2 * j).astype(np.int32)
        reordered_composite = np.empty_like(composite)
        reordered_composite[joint_child] = composite
        inputs = [
            array(mapping, int),
            array(mask, int),
            array([0, j, 2 * j], int),
            array([0, n, 2 * n], int),
            array(joint_child, int),
            array(motion, wp.spatial_vector),
            array(reordered_composite, wp.spatial_matrix),
            array(regularizer, float),
            int(fused_drive),
            array(drive_map, int),
            array(stiffness, float),
            indices,
        ]
        wp.launch(
            _get_crba_sparse_factor_kernel(n, plan.nonzero_count),
            dim=64,
            inputs=inputs,
            outputs=[inverse, status],
            device=device,
            block_dim=128,
        )
        return plan, dense, inverse, status, inputs

    def test_crba_factor_solve_and_endpoint_response(self):
        """Match dense dynamics and endpoint response for branched multi-DOF forests."""
        for parents, counts in [([-1, 0, 0, 1], [6, 1, 3, 0]), ([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2])]:
            for fused_drive in (False, True):
                with self.subTest(parents=parents, fused_drive=fused_drive):
                    plan, dense, inverse, status, inputs = self._factor(parents, counts, fused_drive=fused_drive)
                    np.testing.assert_array_equal(status.numpy(), [0, 0])
                    n = plan.dof_count
                    rng = np.random.default_rng(66)
                    tau = rng.normal(size=2 * n).astype(np.float32)
                    qdd = wp.zeros(2 * n, dtype=float, device="cpu")
                    scratch = wp.zeros((2, n), dtype=float, device="cpu")
                    wp.launch(
                        solve_sparse_mass_matrix,
                        dim=64,
                        inputs=[
                            inputs[0],
                            inputs[3],
                            inputs[-1],
                            inverse,
                            wp.array(tau, device="cpu"),
                            wp.ones(2, dtype=int, device="cpu"),
                            scratch,
                        ],
                        outputs=[qdd],
                        device="cpu",
                        block_dim=128,
                    )
                    for group, art in enumerate([1, 0]):
                        whiten = np.zeros((n, n))
                        whiten[plan.entry_rows, plan.columns] = inverse.numpy()[group]
                        reordered = dense[art][np.ix_(plan.permutation, plan.permutation)]
                        np.testing.assert_allclose(
                            whiten @ np.linalg.cholesky(reordered), np.eye(n), rtol=2.0e-5, atol=2.0e-5
                        )
                        np.testing.assert_allclose(
                            qdd.numpy()[art * n : (art + 1) * n],
                            np.linalg.solve(dense[art], tau[art * n : (art + 1) * n]),
                            rtol=1.0e-4,
                            atol=1.0e-5,
                        )
                        support = plan.endpoint_support(0, len(parents) - 1)
                        row = np.zeros(n)
                        row[plan.permutation[support]] = rng.normal(size=len(support))
                        z = whiten @ row[plan.permutation]
                        np.testing.assert_allclose(np.delete(z, support), 0.0, atol=1.0e-12)
                        np.testing.assert_allclose(
                            z @ z, row @ np.linalg.solve(dense[art], row), rtol=3.0e-5, atol=1.0e-6
                        )

    def test_mask_bit_63_and_fixed_root(self):
        """Preserve the highest mask bit and fixed-root endpoint support."""
        plan = _SparseMassMatrixPlan.build([-1, 0], [0, 64])
        self.assertEqual(plan.endpoint_support(0).size, 0)
        self.assertEqual(int(plan.joint_ancestor_mask[1]), (1 << 64) - 1)
        self.assertEqual(int(plan.ancestor_mask[0]), (1 << 64) - 1)
        with self.assertRaises(ValueError):
            plan.endpoint_support(2)

    def _check_branched_factor(self, device):
        """Compare a broad articulated tree with the independent dense oracle."""
        plan, dense, inverse, status, inputs = self._factor(*_branched_humanoid_topology(), armature=2.0, device=device)
        np.testing.assert_array_equal(status.numpy(), [0, 0])
        n = plan.dof_count
        tau = np.random.default_rng(521).normal(size=2 * n).astype(np.float32)
        qdd = wp.zeros(2 * n, device=device)
        scratch = wp.zeros((2, n), device=device)
        wp.launch(
            solve_sparse_mass_matrix,
            dim=64,
            inputs=[
                inputs[0],
                inputs[3],
                inputs[-1],
                inverse,
                wp.array(tau, device=device),
                wp.ones(2, dtype=int, device=device),
                scratch,
            ],
            outputs=[qdd],
            device=device,
            block_dim=128,
        )
        for group, art in enumerate([1, 0]):
            whiten = np.zeros((n, n))
            whiten[plan.entry_rows, plan.columns] = inverse.numpy()[group]
            expected = dense[art][np.ix_(plan.permutation, plan.permutation)]
            np.testing.assert_allclose(whiten @ np.linalg.cholesky(expected), np.eye(n), atol=1.0e-4, rtol=1.0e-4)
            np.testing.assert_allclose(
                qdd.numpy()[art * n : (art + 1) * n],
                np.linalg.solve(dense[art], tau[art * n : (art + 1) * n]),
                atol=2.0e-5,
                rtol=2.0e-4,
            )

    def test_branched_43_dof_factor_and_predictor(self):
        """Match dense dynamics for a broad 43-DOF articulated tree on CPU."""
        self._check_branched_factor("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse factor CUDA schedule requires CUDA")
    def test_branched_43_dof_factor_and_predictor_cuda(self):
        """Match the dense oracle with parallel factor dependencies and inverse columns."""
        self._check_branched_factor("cuda:0")

    def test_mask_preserves_factor(self):
        """Retain the inverse factor and status when an articulation is not refreshed."""
        _, _, inverse, status, _ = self._factor([-1, 0, 0], [2, 1, 1], mask=[0, 1])
        np.testing.assert_array_equal(inverse.numpy()[1], -98.0)
        np.testing.assert_array_equal(status.numpy(), [0, 17])

    def test_invalid_pivot_is_not_regularized(self):
        """Report an invalid factor without adding a new pivot floor."""
        _, _, inverse, status, _ = self._factor([-1], [1], armature=-1.0e6)
        np.testing.assert_array_equal(status.numpy(), [1, 1])
        self.assertTrue(np.isnan(inverse.numpy()).all())


if __name__ == "__main__":
    unittest.main()
