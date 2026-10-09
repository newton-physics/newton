# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for Kamino RCM factors and tiled pointer offsets."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm import (
    get_float32_array_offset_ptr,
    get_int32_array_offset_ptr,
)
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver
from newton._src.solvers.kamino._src.solvers.dvi.kernels import (
    _find_bilateral_factor_row_start,
    _find_bilateral_factor_row_start_rcm,
)


class TestKaminoFactorRowStart(unittest.TestCase):
    def test_ragged_tile_masks(self):
        """Preserve exact prefixes across ragged tiles, signed zeros, and NaNs."""
        for block_size in (8, 16, 32):
            rng = np.random.default_rng(23)
            dimensions = [0, 1, 15, 16, 17, 31, 32, 33, 64, 65, 97]
            matrices, patterns, mio, vio, tpo, expected = [], [], [], [], [], []
            for n in dimensions:
                tiles = (n + block_size - 1) // block_size
                mask = rng.integers(0, 2, size=(tiles, tiles), dtype=np.int32)
                np.fill_diagonal(mask, 1)
                factor = np.tril(rng.normal(size=(n, n)).astype(np.float32))
                for i in range(tiles):
                    for j in range(i):
                        if mask[i, j] == 0:
                            factor[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 0
                # Conservative symbolic masks may include rows that are numerically zero.
                for row in range(n):
                    factor[row, : min(row, row // 2)] = -0.0
                if n > 32:
                    factor[-1, block_size] = np.nan
                    mask[(n - 1) // block_size, 1] = 1
                    factor[-2, 3] = np.float32(1e-20)
                    mask[(n - 2) // block_size, 0] = 1
                mio.append(len(matrices))
                tpo.append(len(patterns))
                vio.append(len(expected) + 1)
                matrices.extend(factor.ravel())
                patterns.extend(mask.ravel())
                expected.append(-77)
                for row in range(n):
                    nonzero = np.flatnonzero(factor[row, :row] != 0)
                    first = int(nonzero[0]) if len(nonzero) else row
                    expected.append(first // 16 * 16)
                expected.append(-77)
            for device in wp.get_devices():
                dims = wp.array(dimensions, dtype=wp.int32, device=device)
                matrices_wp = wp.array(matrices, dtype=wp.float32, device=device)
                matrix_offsets = wp.array(mio, dtype=wp.int32, device=device)
                vector_offsets = wp.array(vio, dtype=wp.int32, device=device)
                pattern_offsets = wp.array(tpo, dtype=wp.int32, device=device)
                pattern = wp.array(patterns, dtype=wp.int32, device=device)
                reference = wp.full(len(expected), -77, dtype=wp.int32, device=device)
                candidate = wp.full(len(expected), -77, dtype=wp.int32, device=device)
                inputs = [dims, matrix_offsets, vector_offsets, matrices_wp]
                wp.launch(
                    _find_bilateral_factor_row_start,
                    dim=(len(dimensions), max(dimensions)),
                    inputs=[*inputs, reference],
                    device=device,
                )
                wp.launch(
                    _find_bilateral_factor_row_start_rcm,
                    dim=(len(dimensions), max(dimensions)),
                    inputs=[*inputs, candidate, pattern_offsets, pattern, block_size],
                    device=device,
                )
                np.testing.assert_array_equal(reference.numpy(), expected)
                np.testing.assert_array_equal(candidate.numpy(), expected)


class TestKaminoFactorTileReuse(unittest.TestCase):
    def test_disappearing_tiles_are_zero(self):
        """Clear old off-diagonal values when a reused factor's sparsity shrinks."""
        if not wp.is_cuda_available():
            self.skipTest("Blocked factorization requires CUDA")
        device = wp.get_device("cuda:0")
        for parallel in (False, True):
            for n in (64, 65):
                with self.subTest(parallel=parallel, dimension=n):
                    rng = np.random.default_rng(32)
                    dense = rng.normal(size=(n, n)).astype(np.float32)
                    first = dense @ dense.T / n + np.eye(n, dtype=np.float32)
                    second = np.eye(n, dtype=np.float32) * 4.0
                    info = DenseSquareMultiLinearInfo()
                    info.finalize(dimensions=[n], dtype=wp.float32, device=device)
                    matrix = wp.array(first.ravel(), dtype=wp.float32, device=device)
                    operator = DenseLinearOperatorData(info=info, mat=matrix)
                    solver = LLTBlockedRCMSolver(
                        operator=operator,
                        block_size=32,
                        reuse_permutation=True,
                        parallel_factorization=parallel,
                        device=device,
                    )
                    solver.compute(matrix)
                    initial = np.tril(solver.L.numpy().reshape(n, n))
                    self.assertGreater(np.count_nonzero(initial[32:, :32]), 0)
                    matrix.assign(second.ravel())
                    if n == 65:
                        with wp.ScopedCapture(device=device) as capture:
                            solver.compute(matrix)
                        wp.capture_launch(capture.graph)
                    else:
                        solver.compute(matrix)
                    factor = np.tril(solver.L.numpy().reshape(n, n))
                    np.testing.assert_array_equal(factor, np.eye(n, dtype=np.float32) * 2.0)


@wp.kernel
def _compute_pointer_offsets(
    floats: wp.array[wp.float32],
    integers: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    offsets: wp.array2d[wp.uint64],
):
    i = wp.tid()
    offsets[i, 0] = get_float32_array_offset_ptr(floats, indices[i]) - get_float32_array_offset_ptr(floats, 0)
    offsets[i, 1] = get_int32_array_offset_ptr(integers, indices[i]) - get_int32_array_offset_ptr(integers, 0)


class TestKaminoPointerOffsets(unittest.TestCase):
    def test_large_byte_offsets(self):
        """Preserve 64-bit byte offsets across the signed 32-bit boundary."""
        indices = np.array([0, 1, 2**29 - 1, 2**29, 2**29 + 1, 2**30, 2**31 - 1], dtype=np.int32)
        expected = np.repeat((indices.astype(np.uint64) * 4)[:, None], 2, axis=1)
        for device in wp.get_devices():
            with self.subTest(device=device):
                # Only compute addresses: do not allocate or dereference multi-GiB buffers.
                floats = wp.zeros(1, dtype=wp.float32, device=device)
                integers = wp.zeros(1, dtype=wp.int32, device=device)
                indices_device = wp.array(indices, dtype=wp.int32, device=device)
                offsets = wp.empty((len(indices), 2), dtype=wp.uint64, device=device)
                wp.launch(
                    _compute_pointer_offsets,
                    dim=len(indices),
                    inputs=[floats, integers, indices_device],
                    outputs=[offsets],
                    device=device,
                )
                np.testing.assert_array_equal(offsets.numpy(), expected)
