# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LOX consensus body weights and their symmetric eigenvalue solves."""

import unittest

import numpy as np
import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from newton._src.solvers.kamino._src.core.types import mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox.system_kernels import (
    _compute_body_weight_mass_proportional,
    normalize_symmetric_matrix,
)
from newton.tests.kamino import setup_tests, test_context

_NOISY_BODY_BLOCK_DIAGONAL = (5.24412966, 5.24412966, 5.24412966, 2.96570277, 7.72686768, 7.72686768)
"""Diagonal of the inertia-normalized block of a thin rod segment."""

_NOISY_BODY_BLOCK_UPPER = (
    (0, 1, -6.591430523419183e-27),
    (0, 3, 3.5682975193425614e-28),
    (0, 4, 8.463330634011092e-37),
    (0, 5, -7.623092153516266e-21),
    (1, 2, 8.147104961933342e-29),
    (1, 3, -7.225274523646463e-13),
    (1, 4, -1.7136843729722424e-21),
    (1, 5, 1.5435633031302132e-05),
    (2, 3, 1.804798721341742e-20),
    (2, 4, -1.5435633031302132e-05),
    (2, 5, 1.713714463282611e-21),
    (3, 4, -3.106206341156682e-20),
    (3, 5, -3.537425072863698e-07),
    (4, 5, -1.7157175389480783e-15),
)
"""Upper off-diagonal entries of the block, with float32 rounding residue far below its scale."""


def _noisy_body_block() -> np.ndarray:
    block = np.diag(np.asarray(_NOISY_BODY_BLOCK_DIAGONAL, dtype=np.float32))
    for row, col, value in _NOISY_BODY_BLOCK_UPPER:
        block[row, col] = block[col, row] = value
    return block


@wp.kernel
def _compute_unit_mass_weights(
    smooth_diagonal: wp.array[mat66f],
    sigma: wp.float32,
    beta: wp.float32,
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
):
    i = wp.tid()
    body_weight, body_inverse_weight = _compute_body_weight_mass_proportional(
        smooth_diagonal[i], 1.0, wp.identity(n=3, dtype=wp.float32), sigma, beta
    )
    weight[i] = body_weight
    inverse_weight[i] = body_inverse_weight


@wp.kernel
def _compute_normalized_eigenvalues(matrix: wp.array[mat66f], eigenvalues: wp.array[vec6f]):
    i = wp.tid()
    normalized, scale = normalize_symmetric_matrix(matrix[i], 6)
    values, _vectors = symmetric_eigenvalues_qr(normalized, 1.0e-7)
    eigenvalues[i] = scale * values


class TestLOXBodyWeight(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_body_weight_tolerates_float32_residue(self):
        """Weight a body whose normalized block carries rounding residue far below its scale."""
        device = self.default_device
        block = _noisy_body_block()
        weight = wp.empty(1, dtype=mat66f, device=device)
        inverse_weight = wp.empty(1, dtype=mat66f, device=device)
        sigma, beta = 1.0e-3, 4.0
        wp.launch(
            _compute_unit_mass_weights,
            dim=1,
            inputs=[wp.array(block[None], dtype=mat66f, device=device), sigma, beta],
            outputs=[weight, inverse_weight],
            device=device,
        )

        # With unit mass and inertia, the weight is alpha times the identity
        eta = np.linalg.eigvalsh(block.astype(np.float64)).min()
        alpha = max(sigma * eta, min(beta, eta))
        np.testing.assert_allclose(weight.numpy()[0], alpha * np.eye(6), rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(inverse_weight.numpy()[0], np.eye(6) / alpha, rtol=1.0e-5, atol=1.0e-6)

    def test_normalized_eigenvalues_match_reference(self):
        """Match the reference eigenvalues of symmetric blocks seeded with residue across the float32 range."""
        device = self.default_device
        rng = np.random.default_rng(7)
        matrices = []
        for _ in range(512):
            basis, _ = np.linalg.qr(rng.normal(size=(6, 6)))
            matrix = basis @ np.diag(np.exp(rng.uniform(-3.0, 3.0, size=6))) @ basis.T
            residue = np.where(
                rng.random((6, 6)) < 0.5,
                rng.normal(size=(6, 6)) * 10.0 ** rng.uniform(-44.0, -12.0, size=(6, 6)),
                0.0,
            )
            matrices.append(matrix + 0.5 * (residue + residue.T))
        matrices = np.asarray(matrices, dtype=np.float32)
        eigenvalues = wp.empty(len(matrices), dtype=vec6f, device=device)
        wp.launch(
            _compute_normalized_eigenvalues,
            dim=len(matrices),
            inputs=[wp.array(matrices, dtype=mat66f, device=device)],
            outputs=[eigenvalues],
            device=device,
        )

        computed = np.sort(eigenvalues.numpy(), axis=1)
        reference = np.linalg.eigvalsh(matrices.astype(np.float64))
        self.assertTrue(np.all(np.isfinite(computed)))
        error = np.abs(computed - reference) / np.abs(reference).max(axis=1, keepdims=True)
        self.assertLess(float(error.max()), 1.0e-5)


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
