# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX local Coulomb-contact primitives and normal contact laws."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.lox.contact import (
    compute_contact_normal_compliance,
    compute_contact_penetration_bias,
    compute_contact_recovery_fraction,
    compute_contact_restitution_target,
    compute_contact_scaled_alart_curnier_residual,
    solve_contact_coulomb_newton,
)
from newton.tests.kamino import setup_tests, test_context


@wp.kernel
def _solve_contacts(
    delassus: wp.array[wp.mat33f],
    free_velocity: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
    residual: wp.array[wp.vec3f],
):
    contact = wp.tid()
    reaction_i = solve_contact_coulomb_newton(delassus[contact], free_velocity[contact], friction[contact])
    velocity_i = delassus[contact] @ reaction_i + free_velocity[contact]
    reaction[contact] = reaction_i
    velocity[contact] = velocity_i
    residual[contact] = compute_contact_scaled_alart_curnier_residual(
        delassus[contact], reaction_i, velocity_i, friction[contact]
    )


@wp.kernel
def _evaluate_normal_law(parameters: wp.array2d[wp.float32], result: wp.array2d[wp.float32]):
    i = wp.tid()
    distance = parameters[i, 0]
    previous_velocity = parameters[i, 1]
    free_velocity = parameters[i, 2]
    restitution = parameters[i, 3]
    time_step = parameters[i, 4]
    fraction = parameters[i, 5]
    stiffness = parameters[i, 6]
    mechanical_delassus = parameters[i, 7]
    damping = parameters[i, 8]
    normal_compliance = compute_contact_normal_compliance(stiffness, damping, time_step)
    if stiffness > 0.0:
        fraction = compute_contact_recovery_fraction(stiffness, damping, time_step)
    bias = compute_contact_penetration_bias(distance, time_step, fraction)
    target = compute_contact_restitution_target(distance, previous_velocity, free_velocity, restitution, time_step)
    impulse = wp.max(-(free_velocity + bias - target) / (mechanical_delassus + normal_compliance), 0.0)
    result[i, 0] = normal_compliance
    result[i, 1] = bias
    result[i, 2] = target
    result[i, 3] = impulse
    result[i, 4] = free_velocity + mechanical_delassus * impulse


class TestLOXContact(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _solve(
        self,
        delassus: np.ndarray,
        free_velocity: np.ndarray,
        friction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delassus_wp = wp.array(delassus, dtype=wp.mat33f, device=self.default_device)
        free_velocity_wp = wp.array(free_velocity, dtype=wp.vec3f, device=self.default_device)
        friction_wp = wp.array(friction, dtype=wp.float32, device=self.default_device)
        reaction_wp = wp.empty(len(friction), dtype=wp.vec3f, device=self.default_device)
        velocity_wp = wp.empty(len(friction), dtype=wp.vec3f, device=self.default_device)
        residual_wp = wp.empty(len(friction), dtype=wp.vec3f, device=self.default_device)

        wp.launch(
            _solve_contacts,
            dim=len(friction),
            inputs=[delassus_wp, free_velocity_wp, friction_wp],
            outputs=[reaction_wp, velocity_wp, residual_wp],
            device=self.default_device,
        )
        return reaction_wp.numpy(), velocity_wp.numpy(), residual_wp.numpy()

    def test_separating_contact(self):
        """Leave a separating contact inactive."""
        delassus = np.asarray([np.diag([3.0, 4.0, 2.0])], dtype=np.float32)
        free_velocity = np.asarray([[-4.0, 2.0, 0.25]], dtype=np.float32)
        friction = np.asarray([0.7], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus, free_velocity, friction)

        np.testing.assert_array_equal(reaction[0], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(velocity[0], free_velocity[0], rtol=0.0, atol=1.0e-7)
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)

    def test_frictionless_contact_skips_degenerate_tangent_block(self):
        """Solve frictionless contact with a degenerate tangent block."""
        # Only W_N is used in the frictionless branch, which protects the
        # prototype while preprocessing regularizes active frictional blocks.
        delassus = np.asarray([np.diag([0.0, 0.0, 2.0])], dtype=np.float32)
        free_velocity = np.asarray([[3.0, -2.0, -1.0]], dtype=np.float32)
        friction = np.asarray([0.0], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus, free_velocity, friction)

        np.testing.assert_allclose(reaction[0], [0.0, 0.0, 0.5], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(velocity[0], [3.0, -2.0, 0.0], rtol=0.0, atol=1.0e-7)
        self.assertTrue(np.all(np.isfinite(residual[0])))
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)

    def test_sticking_contact(self):
        """Solve a sticking frictional contact."""
        delassus = np.asarray([np.diag([3.0, 4.0, 2.0])], dtype=np.float32)
        free_velocity = np.asarray([[0.2, -0.1, -1.0]], dtype=np.float32)
        friction = np.asarray([0.8], dtype=np.float32)
        expected_reaction = np.asarray([-0.2 / 3.0, 0.025, 0.5], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus, free_velocity, friction)

        np.testing.assert_allclose(reaction[0], expected_reaction, rtol=2.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(velocity[0], np.zeros(3), rtol=0.0, atol=2.0e-7)
        self.assertLess(np.linalg.norm(reaction[0, :2]), friction[0] * reaction[0, 2])
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)

    def test_sliding_contact(self):
        """Project a sliding contact onto the Coulomb cone."""
        delassus = np.asarray([np.diag([1.0, 1.0, 2.0])], dtype=np.float32)
        free_velocity = np.asarray([[2.0, 0.0, -1.0]], dtype=np.float32)
        friction = np.asarray([0.5], dtype=np.float32)
        expected_reaction = np.asarray([-0.25, 0.0, 0.5], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus, free_velocity, friction)

        np.testing.assert_allclose(reaction[0], expected_reaction, rtol=2.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(velocity[0], [1.75, 0.0, 0.0], rtol=2.0e-6, atol=2.0e-7)
        self.assertAlmostEqual(np.linalg.norm(reaction[0, :2]), friction[0] * reaction[0, 2], delta=2.0e-7)
        self.assertLess(np.dot(reaction[0, :2], velocity[0, :2]), 0.0)
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)

    def test_strongly_coupled_full_block(self):
        """Solve a strongly coupled contact Delassus block."""
        delassus_i = np.asarray(
            [
                [1.7, 0.25, 0.35],
                [0.25, 0.8, -0.15],
                [0.35, -0.15, 2.0],
            ],
            dtype=np.float32,
        )
        friction = np.asarray([0.5], dtype=np.float32)
        expected_reaction = np.asarray([-0.18, -0.135, 0.45], dtype=np.float32)
        expected_velocity = np.asarray([0.28, 0.21, 0.0], dtype=np.float32)
        free_velocity_i = expected_velocity - delassus_i @ expected_reaction

        reaction, velocity, residual = self._solve(delassus_i[None, ...], free_velocity_i[None, ...], friction)

        np.testing.assert_allclose(reaction[0], expected_reaction, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(velocity[0], expected_velocity, rtol=2.0e-5, atol=2.0e-6)
        self.assertLess(np.linalg.norm(residual[0]), 2.0e-6)

    def test_bracketed_bisection_fallback(self):
        """Keep the bounded bisection fallback stable for a difficult contact."""
        # This coupled problem rejects multiple pure Newton steps and does not
        # reach the root tolerance within the bounded local solve.
        delassus_i = np.asarray(
            [
                [7.38549845, -0.55674596, -2.21409240],
                [-0.55674596, 0.53603802, -0.05816527],
                [-2.21409240, -0.05816527, 3.05015024],
            ],
            dtype=np.float32,
        )
        free_velocity_i = np.asarray([-3.11677977, -4.67238632, -0.97200092], dtype=np.float32)
        friction = np.asarray([5.40898023], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus_i[None, ...], free_velocity_i[None, ...], friction)

        self.assertTrue(np.all(np.isfinite(reaction[0])))
        self.assertAlmostEqual(velocity[0, 2], 0.0, delta=1.0e-5)
        self.assertLess(np.dot(reaction[0, :2], velocity[0, :2]), 0.0)
        self.assertLess(np.linalg.norm(residual[0]), 3.0e-2)

    def test_nearly_singular_regularized_block(self):
        """Stabilize contact reactions for a nearly singular Delassus block."""
        delassus_i = np.diag([1.0e-6, 2.0e-6, 1.0]).astype(np.float32)
        expected_reaction = np.asarray([-0.1, 0.1, 1.0], dtype=np.float32)
        free_velocity_i = -(delassus_i @ expected_reaction)
        friction = np.asarray([0.5], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus_i[None, ...], free_velocity_i[None, ...], friction)

        np.testing.assert_allclose(reaction[0], expected_reaction, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(velocity[0], np.zeros(3), rtol=0.0, atol=2.0e-7)
        self.assertTrue(np.all(np.isfinite(reaction[0])))
        self.assertTrue(np.all(np.isfinite(residual[0])))
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)


class TestLOXContactLaw(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _evaluate(self, rows):
        """Evaluate rows ``(distance, v_begin, v_free, e, h, fraction, k, W_nn[, d])`` of the scalar normal law."""
        device = self.default_device
        rows = [list(row) + [0.0] * (9 - len(row)) for row in rows]
        parameters = wp.array(np.asarray(rows, dtype=np.float32), dtype=wp.float32, device=device)
        result = wp.empty((len(rows), 5), dtype=wp.float32, device=device)
        wp.launch(_evaluate_normal_law, dim=len(rows), inputs=[parameters], outputs=[result], device=device)
        return result.numpy()

    def test_speculative_switch_revisits_trial_velocity(self):
        """Switch bounce on when the current trial closes a positive gap."""
        rows = [[0.01, -2.0, velocity, 0.5, 0.1, 1.0, 0.0, 1.0] for velocity in (-0.05, -0.2, 0.0)]
        result = self._evaluate(rows)
        np.testing.assert_allclose(result[:, 2], [-0.1, 1.0, -0.1], atol=1.0e-6)
        np.testing.assert_allclose(result[:, 3], [0.0, 1.2, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(result[:, 4], [-0.05, 1.0, 0.0], atol=1.0e-6)

    def test_trial_boundary_and_small_gap(self):
        """Keep exactly feasible and small positive-gap trials open, and bounce at zero gap."""
        rows = [
            [0.125, -2.0, -0.5, 0.5, 0.25, 1.0, 0.0, 1.0],
            [0.005, -2.0, 0.0, 0.5, 0.25, 1.0, 0.0, 1.0],
            [0.0, -2.0, 0.0, 0.5, 0.25, 1.0, 0.0, 1.0],
        ]
        result = self._evaluate(rows)
        np.testing.assert_allclose(result[:, 2], [-0.5, -0.02, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(result[:, 3], [0.0, 0.0, 1.0], atol=1.0e-6)

    def test_restitution_endpoints_without_threshold(self):
        """Reproduce Newton restitution endpoints for fast, slow, and separating contacts."""
        rows = [[0.0, -2.0, -2.0, e, 0.1, 1.0, 0.0, 0.5] for e in (0.0, 0.5, 1.0)]
        rows.extend([0.0, velocity, -0.01, 1.0, 0.1, 1.0, 0.0, 0.5] for velocity in (-0.005, -0.01, 0.0, 2.0))
        result = self._evaluate(rows)
        np.testing.assert_allclose(result[:, 2], [0.0, 1.0, 2.0, 0.005, 0.01, 0.0, -2.0], atol=1.0e-6)
        np.testing.assert_allclose(result[:, 4], [0.0, 1.0, 2.0, 0.005, 0.01, 0.0, -0.01], atol=1.0e-6)

    def test_compliance_static_load_deflection(self):
        """Balance a static load at the spring deflection for different timesteps and dampings."""
        stiffness, force, inverse_mass = 500.0, 10.0, 0.5
        rows = [
            [-force / stiffness, 0.0, -h * force * inverse_mass, 0.0, h, 1.0, stiffness, inverse_mass, damping]
            for damping in (0.0, 50.0)
            for h in (0.01, 0.02, 0.1)
        ]
        result = self._evaluate(rows)
        np.testing.assert_allclose(result[:, 3], [0.1, 0.2, 1.0] * 2, atol=1.0e-6)
        np.testing.assert_allclose(result[:, 4], 0.0, atol=1.0e-6)

    def test_compliance_hard_limit_and_separation(self):
        """Approach the hard-contact impulse and avoid attractive separated forces."""
        rows = [[0.0, -1.0, -1.0, 0.0, 0.1, 1.0, stiffness, 0.5] for stiffness in (100.0, 1.0e8, 0.0)]
        rows.append([0.01, 0.0, 1.0, 0.0, 0.1, 1.0, 100.0, 0.5])
        result = self._evaluate(rows)
        np.testing.assert_allclose(result[:, 0], [1.0, 1.0e-6, 0.0, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(result[:, 3], [2.0 / 3.0, 2.0, 2.0, 0.0], atol=5.0e-6)

    def test_compliance_damping(self):
        """Stiffen the approach and slow the penetration recovery of a damped spring."""
        rows = [
            [0.0, -1.0, -1.0, 0.0, 0.1, 1.0, 100.0, 0.5, 0.0],
            [0.0, -1.0, -1.0, 0.0, 0.1, 1.0, 100.0, 0.5, 10.0],
            [-0.02, 0.0, 0.0, 0.0, 0.1, 1.0, 100.0, 0.5, 10.0],
        ]
        result = self._evaluate(rows)
        # Compliance 1 / (h (k h + d)) and recovery fraction k h / (k h + d) = 1/2
        np.testing.assert_allclose(result[:, 0], [1.0, 0.5, 0.5], atol=1.0e-6)
        np.testing.assert_allclose(result[:, 1], [0.0, 0.0, -0.1], atol=1.0e-6)
        np.testing.assert_allclose(result[:, 3], [2.0 / 3.0, 1.0, 0.1], atol=1.0e-6)


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
