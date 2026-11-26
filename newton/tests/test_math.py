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


@wp.kernel(enable_backward=False)
def _quat_velocity_kernel(q_now: wp.array[wp.quat], q_prev: wp.array[wp.quat], dt: float, result: wp.array[wp.vec3]):
    """Evaluate finite-difference angular velocity on a Warp device."""
    i = wp.tid()
    result[i] = newton.math.quat_velocity(q_now[i], q_prev[i], dt)


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

    def test_vec_allclose_handles_infinities(self):
        """Match NumPy for infinite operands without accepting other invalid elements."""
        values = (0.0, np.inf, -np.inf, np.nan)
        pairs = [(left, right) for left in values for right in values]
        a = np.array(
            [[left, 0.0, 2.0] for left, _ in pairs] + [[np.inf, 0.0, 3.0], [np.inf, 0.0, np.nan]],
            dtype=np.float32,
        )
        b = np.array(
            [[right, 0.0, 2.0] for _, right in pairs] + [[np.inf, 0.0, 2.0], [np.inf, 0.0, 2.0]],
            dtype=np.float32,
        )
        expected = [np.allclose(left, right, rtol=0.1, atol=0.25) for left, right in zip(a, b, strict=True)]

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


class TestMathQuatVelocity(unittest.TestCase):
    """Check finite-difference angular velocity from two orientations."""

    def test_quat_velocity_identical_and_small_rotations(self):
        """Return exactly zero for identical orientations and the rotation rate otherwise."""
        rng = np.random.default_rng(0)
        q = rng.normal(size=(256, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        dt = 1.0e-3
        angle = 1.0e-3
        axis = np.array([0.0, 0.6, 0.8])
        step = np.array([*(np.sin(0.5 * angle) * axis), np.cos(0.5 * angle)])
        # Warp quaternions are (x, y, z, w); q_now = step * q_prev rotates by `angle` about `axis`.
        x0, y0, z0, w0 = q.T
        x1, y1, z1, w1 = step
        rotated = np.stack(
            (
                w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
                w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
                w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            ),
            axis=1,
        )

        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                q_prev = wp.array(q.astype(np.float32), dtype=wp.quat, device=device)
                for q_now, expected, atol in (
                    (q, np.zeros(3), 0.0),
                    (-q, np.zeros(3), 0.0),
                    (rotated, axis * (angle / dt), 1.0e-3),
                ):
                    result = wp.empty(len(q), dtype=wp.vec3, device=device)
                    wp.launch(
                        _quat_velocity_kernel,
                        dim=len(q),
                        inputs=[wp.array(q_now.astype(np.float32), dtype=wp.quat, device=device), q_prev, dt],
                        outputs=[result],
                        device=device,
                    )
                    np.testing.assert_allclose(
                        result.numpy(), np.broadcast_to(expected, (len(q), 3)), rtol=0.0, atol=atol
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
