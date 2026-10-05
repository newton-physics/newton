# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare sparse factor-coordinate sweeps with physical-coordinate PGS."""

import unittest
from itertools import product

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.friction import friction_pair_candidate
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from newton._src.solvers.feather_pgs.sparse_pgs import _get_pgs_solve_sparse_kernel

_SWEEP_CASES = ((0, 0, 1.0), (3, 1, 0.7), (3, 4, 1.2))


def _sweep(
    jacobian,
    response,
    velocity,
    bias,
    diagonal,
    types,
    parents,
    mu,
    impulses,
    *,
    friction_start=0,
    iteration_offset=0,
    omega=1.0,
):
    velocity, impulses = velocity.copy(), impulses.copy()
    for iteration in range(8):
        for row, kind in enumerate(types):
            if kind == PGS_CONSTRAINT_TYPE_FRICTION and iteration_offset + iteration < friction_start:
                impulses[row] = 0.0
                continue
            if kind == PGS_CONSTRAINT_TYPE_FRICTION and row != parents[row] + 1:
                continue
            residual = jacobian[row] @ velocity + bias[row]
            old = impulses[row]
            if kind == PGS_CONSTRAINT_TYPE_FRICTION:
                parent, sibling = parents[row], row + 1
                load = impulses[parent]
                next_row = parents[parent]
                while next_row >= 0 and next_row != parent:
                    load += impulses[next_row]
                    next_row = parents[next_row]
                radius = max(mu[row] * load, 0.0)
                pair = friction_pair_candidate(
                    float(diagonal[row]),
                    float(jacobian[row] @ response[sibling]),
                    float(diagonal[sibling]),
                    wp.vec2(float(residual), float(jacobian[sibling] @ velocity + bias[sibling])),
                    wp.vec2(float(old), float(impulses[sibling])),
                    float(radius),
                    omega,
                )
                pair = np.asarray(pair, dtype=np.float64)
                magnitude = np.linalg.norm(pair)
                if magnitude > radius:
                    pair *= radius / magnitude
                velocity += response[sibling] * (pair[1] - impulses[sibling])
                impulses[sibling] = pair[1]
                value = pair[0]
            else:
                value = old - omega * residual / diagonal[row]
                if kind in (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT):
                    value = max(value, 0.0)
            velocity += response[row] * (value - old)
            impulses[row] = value
    return velocity, impulses


def _problem(dofs=43, support=18, seeded=False):
    rng = np.random.default_rng(471)
    factor = np.tril(rng.normal(scale=0.04, size=(dofs, dofs))) + np.eye(dofs)
    indices = np.sort(rng.choice(dofs, support, replace=False))
    rows = np.zeros((8, dofs))
    rows[:, indices] = rng.normal(scale=0.3, size=(8, support))
    rows[0] = 0.0
    rows[0, indices[0]] = 0.8
    initial = rng.normal(scale=0.2, size=dofs)
    bias = np.array((-0.2, -0.1, 0.04, -0.03, -0.12, -0.15, 0.02, 0.04))
    types = np.array(
        (
            PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
        ),
        dtype=np.int32,
    )
    parents = np.array((-1, 4, 1, 1, 1, -1, 5, 5), dtype=np.int32)
    mu = np.array((0, 0, 0.6, 0.6, 0, 0, 0.4, 0.4), dtype=np.float64)
    impulses = np.array((0, 0.4, 0.02, -0.01, 0.3, 0.2, -0.03, 0.01)) if seeded else np.zeros(8)
    jacobian = rows @ factor.T
    response = np.linalg.solve(factor.T, rows.T).T
    incident = jacobian @ initial
    diagonal = np.sum(rows * rows, axis=1) + 1.0e-6
    return factor, indices, rows, initial, bias, types, parents, mu, impulses, jacobian, response, incident, diagonal


def _mixed_problem():
    """Build independently coupled robot/free and free/free rows in physical coordinates."""
    rng = np.random.default_rng(472)
    factor = np.zeros((21, 21))
    for first, count in ((0, 9), (9, 6), (15, 6)):
        block = slice(first, first + count)
        factor[block, block] = np.tril(rng.normal(scale=0.08, size=(count, count))) + np.eye(count)
    jacobian = np.zeros((14, 21))
    supports = [
        [0],
        [0, 2, 6, 8, *range(9, 15)],
        [0, 2, 6, 8, *range(9, 15)],
        [0, 2, 6, 8, *range(9, 15)],
        [0, 1, 4],
        [1, 3, 5, *range(15, 21)],
        [1, 3, 5, *range(15, 21)],
        [1, 3, 5, *range(15, 21)],
        *[list(range(9, 21))] * 3,
        *[list(range(15, 21))] * 3,
    ]
    for row, support in enumerate(supports):
        jacobian[row, support] = rng.normal(scale=0.3, size=len(support))
    response = np.linalg.solve(factor @ factor.T, jacobian.T).T
    # Preserve a supplied MF response that differs from the cached factor's
    # inverse; substituting z for r must fail this fixture.
    response[8:] *= np.linspace(0.7, 1.3, 21)
    # Only the robot changes coordinates; free bodies retain physical deltas.
    factor[9:, 9:] = np.eye(12)
    z = np.linalg.solve(factor, jacobian.T).T
    r = response @ factor
    initial = rng.normal(scale=0.2, size=21)
    bias = np.array((-0.2, -0.1, 0.04, -0.03, -0.12, -0.15, 0.02, 0.04, -0.13, 0.04, -0.02, -0.18, 0.02, 0.01))
    types = np.array(
        [
            PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
            *[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION],
            PGS_CONSTRAINT_TYPE_CONTACT,
            *[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION] * 3,
        ],
        dtype=np.int32,
    )
    parents = np.array((-1, 4, 1, 1, 1, -1, 5, 5, -1, 8, 8, -1, 11, 11), dtype=np.int32)
    mu = np.where(types == PGS_CONSTRAINT_TYPE_FRICTION, 0.6, 0.0)
    diagonal = np.einsum("ij,ij->i", jacobian, response) + 1.0e-6
    return factor, z, r, jacobian, response, initial, bias, types, parents, mu, diagonal


class TestFeatherPGSSparsePGS(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_free_velocity_limit_is_stateless(self):
        """Apply free-body velocity limits without undoing old impulses or applying relaxation."""
        device = "cuda:0"

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        meta = np.zeros((2, 4), dtype=np.int32)
        meta[:, 0] = 0xFFFF  # Active endpoint at offset zero; static endpoint -1.
        meta[:, 1] = np.float32(0.5).view(np.int32)
        meta[:, 3] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT | (-1 << 16)
        rows = np.zeros((2, 1, 6), dtype=np.float32)
        rows[:, 0, 0] = 1.0
        velocity = np.zeros((2, 6), dtype=np.float32)
        velocity[:, 0] = [-2.0, 2.0]
        impulses = array([[3.0], [3.0]])
        delta = wp.zeros((2, 6), device=device)
        wp.launch_tiled(
            _get_pgs_solve_sparse_kernel(1, 6, 6, mf_max_constraints=1),
            dim=[1],
            inputs=[
                2,
                array([0, 0], int),
                array([[0.0], [0.0]]),
                array([[1.0], [1.0]]),
                array([[0.0], [0.0]]),
                array(np.full((2, 1, 6), -1), int),
                array(np.zeros((2, 1, 6))),
                array(np.zeros((2, 1, 12))),
                array([[0.0], [0.0]]),
                array([[0], [0]], int),
                array([[-1], [-1]], int),
                array([[0.0], [0.0]]),
                array(np.ones((2, 6)), int),
                array(np.arange(12).reshape(2, 6), int),
                array(velocity.reshape(-1)),
                array([1, 1], int),
                array(meta, int),
                impulses,
                array(rows),
                array(np.zeros_like(rows)),
                array(rows),
                array(np.zeros_like(rows)),
                array([[0.0], [0.0]]),
                1,
                0.4,
                0,
                0,
            ],
            outputs=[delta],
            block_dim=64,
            device=device,
        )
        np.testing.assert_array_equal(impulses.numpy(), [[1.0], [0.0]])
        expected = np.zeros((2, 6))
        expected[0, 0] = 1.0
        np.testing.assert_array_equal(delta.numpy(), expected)

    def test_mixed_factor_sweeps_preserve_supplied_response(self):
        """Preserve dense-then-MF row order and distinct free-body responses."""
        factor, z, r, jacobian, response, initial, bias, types, parents, mu, diagonal = _mixed_problem()
        impulses = np.zeros(len(types))
        np.testing.assert_allclose(z @ r.T, jacobian @ response.T, atol=1.0e-12)
        self.assertGreater(np.linalg.norm(z[8:] - r[8:]), 0.1)
        for friction_start, iteration_offset, omega in _SWEEP_CASES:
            with self.subTest(friction_start=friction_start, iteration_offset=iteration_offset, omega=omega):
                settings = {"friction_start": friction_start, "iteration_offset": iteration_offset, "omega": omega}
                expected, expected_impulses = _sweep(
                    jacobian, response, initial, bias, diagonal, types, parents, mu, impulses, **settings
                )
                delta, actual_impulses = _sweep(
                    z,
                    r,
                    np.zeros_like(initial),
                    jacobian @ initial + bias,
                    diagonal,
                    types,
                    parents,
                    mu,
                    impulses,
                    **settings,
                )
                np.testing.assert_allclose(
                    initial + np.linalg.solve(factor.T, delta), expected, atol=3.0e-6, rtol=3.0e-5
                )
                np.testing.assert_allclose(actual_impulses, expected_impulses, atol=3.0e-6, rtol=3.0e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_mixed_sweeps_match_physical_reference(self):
        """Preserve mixed rows, capacities, friction timing and relaxation across empty worlds."""
        for include_mf, (friction_start, iteration_offset, omega) in product((False, True), _SWEEP_CASES):
            with self.subTest(
                include_mf=include_mf, friction_start=friction_start, iteration_offset=iteration_offset, omega=omega
            ):
                self._check_cuda_mixed_sweeps(include_mf, friction_start, iteration_offset, omega)

    def _check_cuda_mixed_sweeps(self, include_mf, friction_start, iteration_offset, omega):
        """Compare physical output while independently toggling the native free/free family."""
        factor, z, r, jacobian, response, initial, bias, types, parents, mu, diagonal = _mixed_problem()
        count = 14 if include_mf else 8
        expected, expected_impulses = _sweep(
            jacobian[:count],
            response[:count],
            initial,
            bias[:count],
            diagonal[:count],
            types[:count],
            parents[:count],
            mu[:count],
            np.zeros(count),
            friction_start=friction_start,
            iteration_offset=iteration_offset,
            omega=omega,
        )
        device, worlds, capacity, mf_capacity, support = "cuda:0", 3, 10, 8, 21

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def padded(value, cap=capacity, dtype=np.float32):
            result = np.zeros((worlds, cap, *value.shape[1:]), dtype=dtype)
            result[:, : len(value)] = value
            return array(result, int if dtype == np.int32 else float)

        row_dof = np.full((worlds, capacity, support), -1, dtype=np.int32)
        row_factor = np.zeros((worlds, capacity, support), dtype=np.float32)
        row_free_response = np.zeros((worlds, capacity, 12), dtype=np.float32)
        for row in range(8):
            indices = np.flatnonzero(
                np.any(z[1:4] != 0, axis=0)
                if 1 <= row < 4
                else np.any(z[5:8] != 0, axis=0)
                if 5 <= row < 8
                else z[row] != 0
            )
            indices = np.concatenate((indices[indices >= 9], indices[indices < 9]))
            row_dof[:, row, : len(indices)] = indices
            row_factor[:, row, : len(indices)] = z[row, indices]
            free_indices = indices[indices >= 9]
            row_free_response[:, row, : len(free_indices)] = r[row, free_indices]
        mf_z, mf_r = np.zeros((6, 12)), np.zeros((6, 12))
        mf_z[:3], mf_r[:3] = z[8:11, 9:21], r[8:11, 9:21]
        mf_z[3:, :6], mf_r[3:, :6] = z[11:, 15:21], r[11:, 15:21]
        meta = np.zeros((worlds, 4 * mf_capacity), dtype=np.int32)
        for row in range(6):
            first, second = (9, 15) if row < 3 else (15, -1)
            meta[:, 4 * row] = (first << 16) | (second & 0xFFFF)
            meta[:, 4 * row + 1] = np.float32(1.0 / diagonal[8 + row]).view(np.int32)
            meta[:, 4 * row + 2] = np.float32(bias[8 + row]).view(np.int32)
            parent = -1 if row % 3 == 0 else row - row % 3
            meta[:, 4 * row + 3] = int(types[8 + row]) | (parent << 16)
        dense_impulses = padded(np.zeros(8))
        mf_impulses = padded(np.zeros(6), mf_capacity)
        delta = wp.full((worlds, 21), float("nan"), device=device)
        wp.launch_tiled(
            _get_pgs_solve_sparse_kernel(
                capacity, 21, support, mf_max_constraints=mf_capacity if include_mf else 0, free_row_dofs=6
            ),
            dim=[2],
            inputs=[
                worlds,
                array([8, 0, 8], int),
                padded(bias[:8]),
                padded(diagonal[:8]),
                dense_impulses,
                array(row_dof, int),
                array(row_factor),
                array(row_free_response),
                padded((jacobian @ initial)[:8]),
                padded(types[:8], dtype=np.int32),
                padded(parents[:8], dtype=np.int32),
                padded(mu[:8]),
                array(np.tile(np.arange(21) >= 9, (worlds, 1)), int),
                array(np.arange(worlds * 21).reshape(worlds, 21), int),
                array(np.tile(initial, worlds)),
                array([6, 0, 6] if include_mf else [0, 0, 0], int),
                array(meta, int),
                mf_impulses,
                padded(mf_z[:, :6], mf_capacity),
                padded(mf_z[:, 6:], mf_capacity),
                padded(mf_r[:, :6], mf_capacity),
                padded(mf_r[:, 6:], mf_capacity),
                padded(mu[8:], mf_capacity),
                8,
                omega,
                friction_start,
                iteration_offset,
            ],
            outputs=[delta],
            block_dim=64,
            device=device,
        )
        actual_delta = delta.numpy()
        np.testing.assert_array_equal(actual_delta[1], 0.0)
        for world in (0, 2):
            actual = initial + np.linalg.solve(factor.T, actual_delta[world])
            np.testing.assert_allclose(actual, expected, atol=3.0e-5, rtol=3.0e-4)
            np.testing.assert_allclose(
                dense_impulses.numpy()[world, :8], expected_impulses[:8], atol=3.0e-5, rtol=3.0e-4
            )
            if include_mf:
                np.testing.assert_allclose(
                    mf_impulses.numpy()[world, :6], expected_impulses[8:], atol=3.0e-5, rtol=3.0e-4
                )

    def test_factor_coordinate_residual_and_cross_term(self):
        """Preserve physical residuals and paired-tangent effective mass."""
        for dofs, support in ((7, 6), (43, 18), (75, 39)):
            with self.subTest(dofs=dofs, support=support):
                factor, _, rows, initial, _, _, _, _, _, jacobian, response, incident, _ = _problem(dofs, support)
                delta = np.linspace(-0.3, 0.2, dofs)
                physical = initial + np.linalg.solve(factor.T, delta)
                np.testing.assert_allclose(jacobian @ physical, incident + rows @ delta, atol=1.0e-12)
                np.testing.assert_allclose(jacobian @ response.T, rows @ rows.T, atol=1.0e-12)

    def test_factor_sweeps_match_physical_sweeps(self):
        """Preserve joint limits, pooled friction loads and seeded-impulse semantics."""
        for seeded, (friction_start, iteration_offset, omega) in product((False, True), _SWEEP_CASES):
            with self.subTest(
                seeded=seeded, friction_start=friction_start, iteration_offset=iteration_offset, omega=omega
            ):
                factor, _, rows, initial, bias, types, parents, mu, impulses, jacobian, response, incident, diag = (
                    _problem(seeded=seeded)
                )
                settings = {"friction_start": friction_start, "iteration_offset": iteration_offset, "omega": omega}
                expected, expected_impulses = _sweep(
                    jacobian, response, initial, bias, diag, types, parents, mu, impulses, **settings
                )
                delta, actual_impulses = _sweep(
                    rows, rows, np.zeros_like(initial), incident + bias, diag, types, parents, mu, impulses, **settings
                )
                actual = initial + np.linalg.solve(factor.T, delta)
                np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=2.0e-5)
                np.testing.assert_allclose(actual_impulses, expected_impulses, atol=2.0e-6, rtol=2.0e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_sparse_sweeps_match_reference(self):
        """Match delayed and relaxed serial sweeps, including padding and zero-row worlds."""
        device = wp.get_device("cuda:0")
        for dofs, support in ((43, 18), (75, 39)):
            for seeded, (friction_start, iteration_offset, omega) in product((False, True), _SWEEP_CASES):
                with self.subTest(
                    dofs=dofs,
                    support=support,
                    seeded=seeded,
                    friction_start=friction_start,
                    iteration_offset=iteration_offset,
                    omega=omega,
                ):
                    _, indices, rows, initial, bias, types, parents, mu, impulses, _, _, incident, diag = _problem(
                        dofs, support, seeded
                    )
                    expected, expected_impulses = _sweep(
                        rows,
                        rows,
                        np.zeros_like(initial),
                        incident + bias,
                        diag,
                        types,
                        parents,
                        mu,
                        impulses,
                        friction_start=friction_start,
                        iteration_offset=iteration_offset,
                        omega=omega,
                    )
                    worlds, capacity = 3, 10
                    row_dof = np.full((worlds, capacity, support), -1, dtype=np.int32)
                    row_dof[:, :8] = indices
                    row_dof[:, 0, 1:] = -1
                    row_factor = np.zeros((worlds, capacity, support), dtype=np.float32)
                    row_factor[:, :8] = rows[:, indices]

                    def padded(array, dtype=np.float32, shape=(worlds, capacity)):
                        result = np.zeros(shape, dtype=dtype)
                        result[:, :8] = array
                        return wp.array(result, device=device)

                    actual_impulses = padded(impulses)
                    delta = wp.full((worlds, dofs), float("nan"), device=device)
                    kernel = _get_pgs_solve_sparse_kernel(capacity, dofs, support)
                    wp.launch_tiled(
                        kernel,
                        dim=[2],
                        inputs=[
                            worlds,
                            wp.array([8, 0, 8], dtype=int, device=device),
                            padded(bias),
                            padded(diag),
                            actual_impulses,
                            wp.array(row_dof, device=device),
                            wp.array(row_factor, device=device),
                            wp.zeros((worlds, capacity, 12), device=device),
                            padded(incident),
                            padded(types, np.int32),
                            padded(parents, np.int32),
                            padded(mu),
                            wp.zeros((worlds, dofs), dtype=int, device=device),
                            wp.array(np.arange(worlds * dofs).reshape(worlds, dofs), dtype=int, device=device),
                            wp.zeros(worlds * dofs, device=device),
                            wp.zeros(worlds, dtype=int, device=device),
                            wp.zeros((worlds, 4), dtype=int, device=device),
                            wp.zeros((worlds, 1), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1), device=device),
                            8,
                            omega,
                            friction_start,
                            iteration_offset,
                        ],
                        outputs=[delta],
                        block_dim=64,
                        device=device,
                    )
                    actual = delta.numpy()
                    result_impulses = actual_impulses.numpy()
                    np.testing.assert_array_equal(actual[1], 0.0)
                    for world in (0, 2):
                        np.testing.assert_allclose(actual[world], expected, atol=2.0e-5, rtol=2.0e-4)
                        np.testing.assert_allclose(
                            result_impulses[world, :8], expected_impulses, atol=2.0e-5, rtol=2.0e-4
                        )


if __name__ == "__main__":
    unittest.main()
