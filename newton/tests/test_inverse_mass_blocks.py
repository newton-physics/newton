# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check cooperative Cholesky and parallel inverse columns."""

import unittest

import numpy as np
import warp as wp

from newton._src.actuators.joint_space_response import (
    _factor_block_kernel,
    _inverse_block_cooperative_kernel,
    _solve_inverse_columns_kernel,
)


class TestInverseMassBlocks(unittest.TestCase):
    """Cover active factors, padded inverse entries and generic block paths."""

    @unittest.skipUnless(wp.is_cuda_available(), "Cooperative factorization requires CUDA")
    def test_inverse_width_padding_and_pivot_floor(self):
        """Match dense factors and inverses across widths and singular pivots."""
        if tuple(int(part) for part in wp.__version__.split(".")[:2]) < (1, 18):
            self.skipTest("Cooperative kernels require Warp 1.18")
        device = wp.get_cuda_device()
        rng = np.random.default_rng(29)
        for width in (1, 7, 49, 65, 128):
            factors = rng.normal(size=(2, width, width)).astype(np.float32) * 0.1
            H = factors @ factors.transpose(0, 2, 1) + 0.2 * np.eye(width, dtype=np.float32)
            for singular in (False, True):
                with self.subTest(width=width, singular=singular):
                    matrix = wp.array(np.zeros_like(H) if singular else H, device=device)
                    counts = wp.array([width, max(0, width - 2)], dtype=int, device=device)
                    baseline = [np.full(H.shape, 0.25, dtype=np.float32) for _ in range(2)]
                    active_counts = [width, max(0, width - 2)]
                    for art, count in enumerate(active_counts):
                        if count == 0:
                            continue
                        if singular:
                            lower = np.sqrt(np.finfo(np.float32).eps) * np.eye(count, dtype=np.float32)
                            inverse = np.eye(count, dtype=np.float32) / np.finfo(np.float32).eps
                        else:
                            dense = H[art, :count, :count].astype(np.float64)
                            lower = np.linalg.cholesky(dense)
                            inverse = np.linalg.inv(dense)
                        ii, jj = np.tril_indices(count)
                        baseline[0][art, ii, jj] = lower[ii, jj]
                        baseline[1][art, :count, :count] = inverse
                    actual = [wp.full(H.shape, 0.25, device=device) for _ in range(2)]
                    if 1 < width <= 64:
                        wp.launch(
                            _inverse_block_cooperative_kernel,
                            dim=(2, 64),
                            inputs=[matrix, counts],
                            outputs=actual,
                            device=device,
                            block_dim=64,
                        )
                    else:
                        wp.launch(
                            _factor_block_kernel, dim=2, inputs=[matrix, counts], outputs=[actual[0]], device=device
                        )
                        wp.launch(
                            _solve_inverse_columns_kernel, dim=(2, width), inputs=[counts, *actual], device=device
                        )
                    for scalar, tiled in zip(baseline, actual, strict=True):
                        np.testing.assert_allclose(tiled.numpy(), scalar, rtol=0.0002, atol=5e-05)


if __name__ == "__main__":
    unittest.main()
