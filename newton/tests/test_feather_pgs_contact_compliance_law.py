# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gate the experimental unilateral material response and unsupported combinations."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.contact_compliance import material_coefficients, normal_coefficients
from newton.solvers import SolverFeatherPGS
from newton.tests._contact_compliance_reference import solve_reference, support_trajectory


class TestContactComplianceLaw(unittest.TestCase):
    """Test isolated compliant contact equations independently of integration."""

    device = os.environ.get("HYDRO_TEST_DEVICE", "cpu")

    def trajectory(self, *, dt=0.0025, iterations=16, damping=20.0, x=0.0, v=0.0, compliant=True):
        """Return an eight-second support trace for a 0.3 kg, 3000 N/m pad."""
        with wp.ScopedDevice(self.device):
            result = wp.zeros((round(8.0 / dt), 3), dtype=float)
            wp.launch(
                support_trajectory, 1, inputs=[0.3, 9.81, 3000.0, damping, dt, x, v, iterations, int(compliant), result]
            )
            return result.numpy()

    def test_material_fallbacks_and_units(self):
        """Preserve weighted stiffness and apply friction weighting only once."""
        self.assertEqual(material_coefficients(100, 0, 0, shape_friction=2), (100, 0, 2))
        self.assertEqual(material_coefficients(100, 3, 0.25, shape_friction=2), (100, 3, 0.5))
        with self.assertRaises(ValueError):
            material_coefficients(0, 0, 0)

    def test_unsupported_solver_modes_rejected(self):
        """Reject settings whose update law has not been wired or validated."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        for key, value in (
            ("articulated_contact_response", "propagation-colored"),
            ("pgs_velocity_iterations", 1),
            ("pgs_contact_regularization", 0.01),
            ("pgs_warmstart", True),
            ("enable_restitution", True),
            ("friction_mode", "coulomb_newton"),
            ("contact_friction_shared_anchor", True),
        ):
            kwargs = {"pgs_mode": "matrix_free", "enable_restitution": False, key: value}
            with self.assertRaisesRegex(ValueError, key):
                SolverFeatherPGS(model, contact_compliance=True, **kwargs)

    def test_static_indentation_and_baseline_failure(self):
        """Recover mg/k indentation that the hard-normal baseline cannot supply."""
        target = -0.3 * 9.81 / 3000.0
        compliant = self.trajectory()
        hard = self.trajectory(compliant=False)
        self.assertLess(abs(compliant[-1, 0] - target), 2e-7)
        self.assertGreater(abs(hard[-1, 0] - target), 0.0009)
        self.assertLess(abs(compliant[-1, 2] - 0.3 * 9.81), 2e-4)

    def test_timestep_and_iteration_convergence(self):
        """Keep static stiffness invariant across substeps and iteration counts."""
        target = -0.3 * 9.81 / 3000.0
        for dt in (0.005, 0.0025, 0.00125):
            for iterations in (1, 8, 16, 128):
                trace = self.trajectory(dt=dt, iterations=iterations)
                self.assertLess(abs(trace[-1, 0] - target), 2e-7)

    def test_no_attraction(self):
        """Return zero tensile impulse even for fast separation while indented."""
        for phi, velocity in ((0.01, 0.0), (0.0, 1.0), (-0.001, 10.0)):
            impulse, residual = solve_reference(
                [[1 / 0.3]], [velocity], [phi], [3000.0], [20.0], dt=0.0025, device=self.device
            )
            self.assertEqual(float(impulse[0]), 0.0)
            self.assertGreaterEqual(float(residual[0]), 0.0)

    def test_damping_reduces_oscillation(self):
        """Reduce early normal oscillation without changing static indentation."""
        undamped = self.trajectory(damping=0, x=-0.002)
        damped = self.trajectory(damping=20, x=-0.002)
        self.assertLess(np.sqrt(np.mean(damped[:80, 1] ** 2)), np.sqrt(np.mean(undamped[:80, 1] ** 2)))
        self.assertLess(abs(damped[-1, 0] - undamped[-1, 0]), 2e-7)

    def test_no_dashpot_force_across_open_gap(self):
        """Avoid a damping force before the predicted trajectory closes the gap."""
        result, _ = solve_reference([[1.0]], [-0.1], [0.01], [3000.0], [10000.0], dt=0.0025, device=self.device)
        self.assertEqual(float(result[0]), 0.0)

    def test_coupled_rows_match_independent_linear_solve(self):
        """Match an independent exact solve on an all-active coupled fixture."""
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        phi = [-0.002, -0.001]
        stiffness, damping = [3000.0, 5000.0], [20.0, 10.0]
        gamma, bias = np.array(
            [normal_coefficients(k, c, p, dt=0.0025) for k, c, p in zip(stiffness, damping, phi, strict=True)]
        ).T
        velocity = np.array([-0.1, -0.2])
        expected = np.linalg.solve(matrix + np.diag(gamma), -(velocity + bias))
        self.assertTrue((expected > 0).all())
        result, residual = solve_reference(matrix, velocity, phi, stiffness, damping, dt=0.0025, device=self.device)
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(residual, 0, atol=1e-6)

    def test_area_partition_preserves_normal_load(self):
        """Preserve compliance when one patch is split into equal quadrature rows."""
        for count in (1, 4, 16):
            result, residual = solve_reference(
                np.full((count, count), 1 / 0.3),
                np.full(count, -0.1),
                np.full(count, -0.001),
                np.full(count, 3000 / count),
                np.full(count, 20 / count),
                dt=0.0025,
                iterations=128,
                device=self.device,
            )
            gamma, bias = normal_coefficients(3000, 20, -0.001, dt=0.0025)
            expected = (0.1 - bias) / (1 / 0.3 + gamma)
            self.assertAlmostEqual(float(result.sum()), expected, places=6)
            np.testing.assert_allclose(residual, 0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
