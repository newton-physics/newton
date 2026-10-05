# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Kamino core math's gyroscopic angular-velocity update."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.math import compute_body_twist_update_with_eom


@wp.kernel
def _update_free_body(
    inertia: wp.mat33f,
    omega: wp.array[wp.vec3f],
    result: wp.array[wp.vec3f],
):
    tid = wp.tid()
    spin = omega[tid]
    _linear, next_omega = compute_body_twist_update_with_eom(
        0.02,
        wp.vec3f(0.0),
        1.0,
        inertia,
        wp.inverse(inertia),
        wp.spatial_vectorf(0.0, 0.0, 0.0, spin[0], spin[1], spin[2]),
        wp.spatial_vectorf(0.0),
    )
    result[tid] = next_omega


class TestKaminoGyroscopicUpdate(unittest.TestCase):
    def test_torque_free_update_preserves_angular_momentum_magnitude(self):
        """Keep |I ω| constant where explicit Euler would lengthen it."""
        inertia = np.diag([1.0, 2.0, 4.0])
        omega = np.random.default_rng(42).uniform(-20.0, 20.0, size=(64, 3))
        expected = np.linalg.norm(omega @ inertia, axis=1)
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            with self.subTest(device=device):
                result = wp.empty(len(omega), dtype=wp.vec3f, device=device)
                wp.launch(
                    _update_free_body,
                    dim=len(omega),
                    inputs=[wp.mat33f(inertia), wp.array(omega, dtype=wp.vec3f, device=device)],
                    outputs=[result],
                    device=device,
                )
                momentum = np.linalg.norm(result.numpy() @ inertia, axis=1)
                np.testing.assert_allclose(momentum, expected, rtol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
