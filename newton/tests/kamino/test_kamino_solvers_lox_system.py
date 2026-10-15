# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LOX symmetric eigenvalue solves."""

import unittest

import numpy as np
import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from newton._src.solvers.kamino._src.core.types import mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox.system_kernels import normalize_symmetric_matrix
from newton.tests.kamino import setup_tests, test_context


@wp.kernel
def _compute_normalized_eigenvalues(matrix: wp.array[mat66f], eigenvalues: wp.array[vec6f]):
    i = wp.tid()
    normalized, scale = normalize_symmetric_matrix(matrix[i], 6)
    values, _vectors = symmetric_eigenvalues_qr(normalized, 1.0e-7)
    eigenvalues[i] = scale * values


class TestLOXSymmetricEigenvalues(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

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
