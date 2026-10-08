# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for math utilities in ``newton.math``."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_test_devices


@wp.kernel(enable_backward=False)
def _allclose_kernel(a: wp.array[wp.vec3], b: wp.array[wp.vec3], result: wp.array[bool]):
    """Evaluate vector tolerance comparisons on a Warp device."""
    i = wp.tid()
    result[i] = newton.math.vec_allclose(a[i], b[i], rtol=0.1, atol=0.25)


@wp.kernel(enable_backward=False)
def _inside_limits_kernel(
    a: wp.array[wp.vec3], lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], result: wp.array[bool]
):
    """Evaluate inclusive vector bounds on a Warp device."""
    i = wp.tid()
    result[i] = newton.math.vec_inside_limits(a[i], lower[i], upper[i])


class TestMathVectorPredicates(unittest.TestCase):
    """Check finite boundaries and unordered values in vector predicates."""

    def test_vec_allclose_rejects_nan(self):
        """Reject NaN operands while accepting the tolerance boundary."""
        a = np.array(
            [[11.25, 0.0, 2.0], [11.5, 0.0, 2.0], [np.nan, 0.0, 2.0], [10.0, 0.0, 2.0]],
            dtype=np.float32,
        )
        b = np.array(
            [[10.0, 0.0, 2.0], [10.0, 0.0, 2.0], [10.0, 0.0, 2.0], [np.nan, 0.0, 2.0]],
            dtype=np.float32,
        )
        expected = [True, False, False, False]

        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                result = wp.empty(len(a), dtype=bool, device=device)
                wp.launch(
                    _allclose_kernel,
                    dim=len(a),
                    inputs=[wp.array(a, dtype=wp.vec3, device=device), wp.array(b, dtype=wp.vec3, device=device)],
                    outputs=[result],
                    device=device,
                )
                np.testing.assert_array_equal(result.numpy(), expected)

    def test_vec_inside_limits_rejects_nan(self):
        """Reject NaN values and bounds while accepting inclusive endpoints."""
        a = np.array(
            [[0.0, 1.0, 2.0], [-2.0, 1.0, 2.0], [2.0, 1.0, 2.0], [np.nan, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            dtype=np.float32,
        )
        lower = np.array(
            [
                [-1.0, 0.0, 2.0],
                [-1.0, 0.0, 2.0],
                [-1.0, 0.0, 2.0],
                [-1.0, 0.0, 2.0],
                [np.nan, 0.0, 2.0],
                [-1.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        upper = np.array(
            [[0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [np.nan, 1.0, 3.0]],
            dtype=np.float32,
        )
        expected = [True, False, False, False, False, False]

        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                result = wp.empty(len(a), dtype=bool, device=device)
                wp.launch(
                    _inside_limits_kernel,
                    dim=len(a),
                    inputs=[
                        wp.array(a, dtype=wp.vec3, device=device),
                        wp.array(lower, dtype=wp.vec3, device=device),
                        wp.array(upper, dtype=wp.vec3, device=device),
                    ],
                    outputs=[result],
                    device=device,
                )
                np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
