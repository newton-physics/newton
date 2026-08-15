# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the public Kamino body-inertia projection API."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.kamino import setup_tests, test_context


class TestSolverKaminoBodyInertiaProjection(unittest.TestCase):
    """Verify body-inertia projection in user-defined velocity coordinates."""

    def setUp(self):
        """Initialize the shared Kamino test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def tearDown(self):
        """Release the test device reference."""
        self.device = None

    def _make_solver(self, world_count: int = 1):
        source = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(source)
        source.add_body(
            xform=wp.transform(
                wp.vec3(0.2, -0.3, 0.4),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi),
            ),
            mass=5.0,
            inertia=wp.mat33(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0),
            lock_inertia=True,
        )

        if world_count == 1:
            builder = source
        else:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverKamino.register_custom_attributes(builder)
            builder.replicate(source, world_count=world_count)

        model = builder.finalize(device=self.device, skip_validation_joints=True)
        return model, newton.solvers.SolverKamino(model)

    @staticmethod
    def _reference_projection(basis: np.ndarray) -> np.ndarray:
        inertia_world = np.diag([3.0, 2.0, 4.0])
        linear = basis[:, :3]
        angular = basis[:, 3:]
        return 5.0 * linear @ linear.T + angular @ inertia_world @ angular.T

    def test_projection_matches_kinetic_energy(self):
        """Project rotated body inertia through a non-orthogonal twist basis."""
        model, solver = self._make_solver()
        state = model.state()
        basis_np = np.array(
            [
                [[1.0, -0.5, 0.25, 0.2, 0.4, -0.1]],
                [[-0.3, 0.8, 0.1, -0.6, 0.2, 0.5]],
                [[0.7, 0.2, -0.4, 0.3, -0.2, 0.9]],
            ],
            dtype=np.float32,
        )
        basis = wp.array(basis_np, dtype=wp.spatial_vectorf, device=self.device)
        projection = wp.empty((1, 3, 3), dtype=wp.float32, device=self.device)

        state_before = state.body_q.numpy().copy()
        basis_before = basis.numpy().copy()
        solver.eval_body_inertia_projection(state, basis, projection)

        expected = self._reference_projection(basis_np[:, 0])
        np.testing.assert_allclose(projection.numpy()[0], expected, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(projection.numpy()[0], projection.numpy()[0].T, atol=0.0)
        np.testing.assert_array_equal(state.body_q.numpy(), state_before)
        np.testing.assert_array_equal(basis.numpy(), basis_before)

    def test_projection_separates_and_masks_worlds(self):
        """Accumulate each world independently and clear masked output matrices."""
        model, solver = self._make_solver(world_count=2)
        state = model.state()
        basis_np = np.zeros((2, 2, 6), dtype=np.float32)
        basis_np[0, 0, 0] = 1.0
        basis_np[1, 0, 3] = 2.0
        basis_np[0, 1, 1] = 3.0
        basis_np[1, 1, 4] = 4.0
        basis = wp.array(basis_np, dtype=wp.spatial_vectorf, device=self.device)
        projection = wp.empty((2, 2, 2), dtype=wp.float32, device=self.device)
        world_mask = wp.array([True, False], dtype=wp.bool, device=self.device)

        solver.eval_body_inertia_projection(state, basis, projection)

        expected_world_0 = self._reference_projection(basis_np[:, 0])
        expected_world_1 = self._reference_projection(basis_np[:, 1])
        result = projection.numpy()
        np.testing.assert_allclose(result[0], expected_world_0, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(result[1], expected_world_1, rtol=1.0e-5, atol=1.0e-5)

        projection.fill_(17.0)
        solver.eval_body_inertia_projection(state, basis, projection, world_mask)

        result = projection.numpy()
        np.testing.assert_allclose(result[0], expected_world_0, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_array_equal(result[1], np.zeros((2, 2), dtype=np.float32))

    def test_projection_validates_arrays(self):
        """Reject incompatible projection arrays before launching kernels."""
        model, solver = self._make_solver()
        state = model.state()
        basis = wp.zeros((2, 1), dtype=wp.spatial_vectorf, device=self.device)
        projection = wp.zeros((1, 2, 2), dtype=wp.float32, device=self.device)

        with self.assertRaisesRegex(ValueError, "body_velocity_basis must have shape"):
            solver.eval_body_inertia_projection(
                state,
                wp.zeros((2, 2), dtype=wp.spatial_vectorf, device=self.device),
                projection,
            )
        with self.assertRaisesRegex(ValueError, "projection must have shape"):
            solver.eval_body_inertia_projection(
                state,
                basis,
                wp.zeros((1, 2, 3), dtype=wp.float32, device=self.device),
            )
        with self.assertRaisesRegex(TypeError, "projection must have dtype"):
            solver.eval_body_inertia_projection(
                state,
                basis,
                wp.zeros((1, 2, 2), dtype=wp.float64, device=self.device),
            )
        with self.assertRaisesRegex(ValueError, "world_mask must have shape"):
            solver.eval_body_inertia_projection(
                state,
                basis,
                projection,
                wp.zeros(2, dtype=wp.bool, device=self.device),
            )


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
