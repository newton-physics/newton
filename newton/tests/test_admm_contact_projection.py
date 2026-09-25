# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical regressions for the ADMM local contact projection."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.coupled.admm_utils import contact_u_update_kernel


def _project(device, velocity, bounds, normals, *, friction=0.0, weights=1.0, dual=None, count=None):
    velocity = np.asarray(velocity, dtype=np.float32)
    size = len(velocity)
    with wp.ScopedDevice(device):
        result = wp.array(np.full_like(velocity, 42.0), dtype=wp.vec3)
        wp.launch(
            contact_u_update_kernel,
            dim=size,
            inputs=[
                wp.array([size if count is None else count], dtype=wp.int32),
                wp.array(np.broadcast_to(bounds, (size,)).copy(), dtype=wp.float32),
                wp.array(np.broadcast_to(weights, (size,)).copy(), dtype=wp.float32),
                1000.0,
                wp.array(np.broadcast_to(friction, (size,)).copy(), dtype=wp.float32),
                wp.array(normals, dtype=wp.vec3),
                wp.array(np.zeros_like(velocity) if dual is None else dual, dtype=wp.vec3),
                wp.array(velocity, dtype=wp.vec3),
            ],
            outputs=[result],
        )
        return result.numpy()


class TestAdmmContactProjection(unittest.TestCase):
    """Check separating identity and binding Coulomb behavior on available devices."""

    def test_separating_identity(self):
        """Preserve velocities exactly instead of subtracting and re-adding large bounds."""
        velocity = np.array([[0.123456789, 0.1, 0.0314159]], dtype=np.float32)
        for device in wp.get_devices():
            for bound in (-1.0, -1000.0, -1.0e7):
                with self.subTest(device=str(device), bound=bound):
                    actual = _project(device, velocity, bound, [[0, 1, 0]], weights=10.0)
                    np.testing.assert_array_equal(actual, velocity)

    def test_separating_oblique_normals(self):
        """Preserve all separating velocity components with oblique normals."""
        rng = np.random.default_rng(17)
        normals = rng.normal(size=(100, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        velocity = rng.normal(size=(100, 3)).astype(np.float32)
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                actual = _project(device, velocity, -10000.0, normals, friction=0.8)
                np.testing.assert_array_equal(actual, velocity)

    def test_binding_coulomb_cases(self):
        """Retain frictionless, sliding, sticking, and negative-friction clamping."""
        # n = +y, bound = -0.1: shifted normal velocity is -1.9.
        velocity = [[3.0, -2.0, 4.0]]
        for device in wp.get_devices():
            for friction in (-0.5, 0.0, 0.3, 10.0):
                with self.subTest(device=str(device), friction=friction):
                    scale = max(0.0, 1.0 - max(0.0, friction) * 1.9 / 5.0)
                    expected = [[3.0 * scale, -0.1, 4.0 * scale]]
                    actual = _project(device, velocity, -0.1, [[0, 1, 0]], friction=friction)
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

    def test_dual_shift_and_unpopulated_rows(self):
        """Apply current dual shifts while leaving rows outside active_count untouched."""
        velocity = np.array([[0.1, 0.2, 0.3], [1, 2, 3]], dtype=np.float32)
        dual = np.ones_like(velocity)
        for device in wp.get_devices():
            for weight in (0.0, 1.0, 10.0):
                with self.subTest(device=str(device), weight=weight):
                    actual = _project(device, velocity, -1000.0, [[0, 1, 0]] * 2, weights=weight, dual=dual, count=1)
                    expected = velocity[0] - dual[0] / np.float32(1000 * weight) if weight > 0 else velocity[0]
                    np.testing.assert_array_equal(actual[0], expected)
                    np.testing.assert_array_equal(actual[1], [42, 42, 42])

    def test_inactive_sentinel(self):
        """Retain the sentinel fast path for inactive rows."""
        velocity = np.array([[0.1, -2.0, 0.3]], dtype=np.float32)
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                np.testing.assert_array_equal(_project(device, velocity, -1e8, [[0, 1, 0]]), velocity)

    def test_nonfinite_velocity(self):
        """Avoid hiding nonfinite input velocities."""
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                result = _project(device, [[np.nan, 0, 0]], -1000.0, [[0, 1, 0]])
                self.assertFalse(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
