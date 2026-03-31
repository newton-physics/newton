# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for inertia validation and correction functionality."""

import unittest
import warnings

import numpy as np
import warp as wp

from newton import ModelBuilder
from newton._src.geometry.inertia import verify_and_correct_inertia


class TestInertiaValidation(unittest.TestCase):
    """Test cases for inertia verification and correction."""

    def test_negative_mass_correction(self):
        """Test that negative mass is corrected to zero."""
        mass = -10.0
        inertia = wp.mat33([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(mass, inertia)

            self.assertTrue(was_corrected)
            self.assertEqual(corrected_mass, 0.0)
            # Zero mass should have zero inertia
            self.assertTrue(np.allclose(np.array(corrected_inertia), 0.0))
            self.assertTrue(len(w) > 0)
            self.assertIn("Negative mass", str(w[0].message))

    def test_mass_bound(self):
        """Test that mass below bound is clamped."""
        mass = 0.5
        bound_mass = 1.0
        inertia = wp.mat33([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, _corrected_inertia, was_corrected = verify_and_correct_inertia(
                mass, inertia, bound_mass=bound_mass
            )

            self.assertTrue(was_corrected)
            self.assertEqual(corrected_mass, bound_mass)
            self.assertTrue(len(w) > 0)
            self.assertIn("below bound", str(w[0].message))

    def test_negative_inertia_diagonal(self):
        """Test that negative inertia diagonal elements are corrected."""
        mass = 1.0
        inertia = wp.mat33([[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -3.0]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(mass, inertia)

            self.assertTrue(was_corrected)
            self.assertEqual(corrected_mass, mass)

            inertia_array = np.array(corrected_inertia).reshape(3, 3)
            self.assertTrue(inertia_array[0, 0] >= 0)
            self.assertTrue(inertia_array[1, 1] >= 0)
            self.assertTrue(inertia_array[2, 2] >= 0)
            self.assertTrue(len(w) > 0)
            self.assertIn("Eigenvalues below threshold detected", str(w[0].message))

    def test_inertia_bound(self):
        """Test that inertia diagonal elements below bound are clamped."""
        mass = 1.0
        bound_inertia = 1.0
        inertia = wp.mat33([[0.1, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(
                mass, inertia, bound_inertia=bound_inertia
            )

            self.assertTrue(was_corrected)
            self.assertEqual(corrected_mass, mass)

            inertia_array = np.array(corrected_inertia).reshape(3, 3)
            self.assertGreaterEqual(inertia_array[0, 0], bound_inertia)
            self.assertGreaterEqual(inertia_array[1, 1], bound_inertia)
            self.assertGreaterEqual(inertia_array[2, 2], bound_inertia)
            self.assertTrue(len(w) > 0)

    def test_triangle_inequality_violation(self):
        """Test correction of inertia that violates triangle inequality."""
        mass = 1.0
        # Violates Ixx + Iyy >= Izz (0.1 + 0.1 < 10.0)
        inertia = wp.mat33([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 10.0]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(
                mass, inertia, balance_inertia=True
            )

            self.assertTrue(was_corrected)
            self.assertEqual(corrected_mass, mass)

            # Check that triangle inequalities are satisfied
            inertia_array = np.array(corrected_inertia).reshape(3, 3)
            Ixx, Iyy, Izz = inertia_array[0, 0], inertia_array[1, 1], inertia_array[2, 2]

            self.assertGreaterEqual(Ixx + Iyy, Izz - 1e-10)
            self.assertGreaterEqual(Iyy + Izz, Ixx - 1e-10)
            self.assertGreaterEqual(Izz + Ixx, Iyy - 1e-10)

            self.assertTrue(len(w) > 0)
            self.assertIn("triangle inequality", str(w[0].message))

    def test_no_balance_inertia(self):
        """Test that triangle inequality violation is reported but not corrected when balance_inertia=False."""
        mass = 1.0
        # Violates Ixx + Iyy >= Izz
        inertia = wp.mat33([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 10.0]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(
                mass, inertia, balance_inertia=False
            )

            self.assertFalse(was_corrected)  # No correction made when balance_inertia=False
            self.assertEqual(corrected_mass, mass)

            # Inertia should not be balanced
            inertia_array = np.array(corrected_inertia).reshape(3, 3)
            self.assertAlmostEqual(inertia_array[0, 0], 0.1)
            self.assertAlmostEqual(inertia_array[1, 1], 0.1)
            self.assertAlmostEqual(inertia_array[2, 2], 10.0)

            self.assertTrue(len(w) > 0)
            self.assertIn("triangle inequality", str(w[0].message))

    def test_valid_inertia_no_correction(self):
        """Test that valid inertia is not corrected."""
        mass = 1.0
        inertia = wp.mat33([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(mass, inertia)

            self.assertFalse(was_corrected)
            self.assertEqual(corrected_mass, mass)
            self.assertTrue(np.allclose(np.array(corrected_inertia).reshape(3, 3), np.array(inertia).reshape(3, 3)))
            self.assertEqual(len(w), 0)

    def test_model_builder_integration_fast(self):
        """Test that fast inertia validation works in ModelBuilder.finalize()."""
        builder = ModelBuilder()
        builder.balance_inertia = True
        builder.bound_mass = 0.1
        builder.bound_inertia = 0.01
        builder.validate_inertia_detailed = False  # Use fast validation (default)

        # Add a body with invalid inertia
        invalid_inertia = wp.mat33([[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 1.0]])
        body_idx = builder.add_body(
            mass=0.05,  # Below bound
            inertia=invalid_inertia,  # Violates triangle inequality
            label="test_body",
        )

        with warnings.catch_warnings(record=True) as w:
            model = builder.finalize()

            # Should get one summary warning
            self.assertEqual(len(w), 1)
            self.assertIn("Inertia validation corrected 1 bodies", str(w[0].message))
            self.assertIn("validate_inertia_detailed=True", str(w[0].message))

            # Check that mass and inertia were corrected
            body_mass = model.body_mass.numpy()[body_idx]
            body_inertia = model.body_inertia.numpy()[body_idx]

            self.assertGreaterEqual(body_mass, builder.bound_mass)

            Ixx, Iyy, Izz = body_inertia[0, 0], body_inertia[1, 1], body_inertia[2, 2]
            self.assertGreaterEqual(Ixx, builder.bound_inertia)
            self.assertGreaterEqual(Iyy, builder.bound_inertia)
            self.assertGreaterEqual(Izz, builder.bound_inertia)

    def test_model_builder_integration_detailed(self):
        """Test that detailed inertia validation works in ModelBuilder.finalize()."""
        builder = ModelBuilder()
        builder.balance_inertia = True
        builder.bound_mass = 0.1
        builder.bound_inertia = 0.01
        builder.validate_inertia_detailed = True  # Use detailed validation

        # Add a body with invalid inertia
        invalid_inertia = wp.mat33([[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 1.0]])
        body_idx = builder.add_body(
            mass=0.05,  # Below bound
            inertia=invalid_inertia,  # Violates triangle inequality
            label="test_body",
        )

        with warnings.catch_warnings(record=True) as w:
            model = builder.finalize()

            # Should get multiple detailed warnings
            self.assertGreater(len(w), 1)
            warning_messages = [str(warning.message) for warning in w]
            self.assertTrue(any("Mass 0.05 is below bound" in msg for msg in warning_messages))

            # Check that mass and inertia were corrected
            body_mass = model.body_mass.numpy()[body_idx]
            body_inertia = model.body_inertia.numpy()[body_idx]

            self.assertGreaterEqual(body_mass, builder.bound_mass)

            Ixx, Iyy, Izz = body_inertia[0, 0], body_inertia[1, 1], body_inertia[2, 2]
            self.assertGreaterEqual(Ixx, builder.bound_inertia)
            self.assertGreaterEqual(Iyy, builder.bound_inertia)
            self.assertGreaterEqual(Izz, builder.bound_inertia)

            # Check triangle inequalities
            self.assertGreaterEqual(Ixx + Iyy, Izz - 1e-10)
            self.assertGreaterEqual(Iyy + Izz, Ixx - 1e-10)
            self.assertGreaterEqual(Izz + Ixx, Iyy - 1e-10)

            self.assertTrue(len(w) > 0)

    def test_default_validation_catches_negative_mass(self):
        """Test that validation runs by default and catches critical issues."""
        builder = ModelBuilder()
        # Don't set any validation options - use defaults

        # Add a body with negative mass
        body_idx = builder.add_body(
            mass=-1.0,  # Negative mass - critical issue
            label="test_body",
        )

        with warnings.catch_warnings(record=True) as w:
            model = builder.finalize()

            # Should get warning about issues found
            self.assertEqual(len(w), 1)
            self.assertIn("Inertia validation corrected 1 bodies", str(w[0].message))

            # Mass should be corrected to 0
            body_mass = model.body_mass.numpy()[body_idx]
            self.assertEqual(body_mass, 0.0)

    def test_nan_mass(self):
        """Test that NaN mass is handled without crashing."""
        mass = float("nan")
        inertia = wp.mat33([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(mass, inertia)

            self.assertTrue(was_corrected)
            self.assertEqual(corrected_mass, 0.0)
            self.assertTrue(np.allclose(np.array(corrected_inertia), 0.0))
            self.assertTrue(len(w) > 0)
            self.assertIn("NaN/Inf", str(w[0].message))

    def test_nan_inertia(self):
        """Test that NaN inertia is handled without crashing."""
        mass = 1.0
        inertia = wp.mat33([[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(mass, inertia)

            self.assertTrue(was_corrected)
            self.assertEqual(corrected_mass, 0.0)
            self.assertTrue(np.allclose(np.array(corrected_inertia), 0.0))
            self.assertTrue(len(w) > 0)
            self.assertIn("NaN/Inf", str(w[0].message))

    def test_inf_inertia(self):
        """Test that Inf inertia is handled without crashing."""
        mass = 1.0
        inertia = wp.mat33([[float("inf"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(mass, inertia)

            self.assertTrue(was_corrected)
            self.assertEqual(corrected_mass, 0.0)
            self.assertTrue(np.allclose(np.array(corrected_inertia), 0.0))
            self.assertTrue(len(w) > 0)
            self.assertIn("NaN/Inf", str(w[0].message))

    def test_zero_mass_not_overridden_by_bound(self):
        """Test that zero mass is not overridden by bound_mass (zero = static body)."""
        mass = 0.0
        bound_mass = 1.0
        inertia = wp.mat33([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with warnings.catch_warnings(record=True):
            corrected_mass, _corrected_inertia, _was_corrected = verify_and_correct_inertia(
                mass, inertia, bound_mass=bound_mass
            )

            self.assertEqual(corrected_mass, 0.0)

    def test_singular_inertia_repaired(self):
        """Test that singular inertia for positive-mass body is made positive-definite."""
        mass = 1.0
        inertia = wp.mat33([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        with warnings.catch_warnings(record=True):
            corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(mass, inertia)

            self.assertTrue(was_corrected)
            self.assertEqual(corrected_mass, mass)
            inertia_array = np.array(corrected_inertia).reshape(3, 3)
            eigenvalues = np.linalg.eigvals(inertia_array)
            self.assertTrue(np.all(eigenvalues > 0))


class TestInertiaValidationParity(unittest.TestCase):
    """Test that CPU (detailed) and GPU (fast) validation paths produce identical results."""

    def _finalize_both_paths(self, mass, inertia, bound_mass=None, bound_inertia=None, balance_inertia=True):
        """Helper to finalize a single-body model with both validation paths and return results."""
        results = {}
        for detailed in [True, False]:
            builder = ModelBuilder()
            builder.balance_inertia = balance_inertia
            builder.bound_mass = bound_mass
            builder.bound_inertia = bound_inertia
            builder.validate_inertia_detailed = detailed

            body_idx = builder.add_body(
                mass=mass,
                inertia=wp.mat33(np.array(inertia, dtype=np.float32)),
                label="test_body",
            )

            with warnings.catch_warnings(record=True):
                model = builder.finalize()

            mode = "detailed" if detailed else "fast"
            results[mode] = {
                "model_mass": float(model.body_mass.numpy()[body_idx]),
                "model_inertia": np.array(model.body_inertia.numpy()[body_idx]),
            }
        return results

    def _assert_parity(self, results, atol=1e-5):
        """Assert that detailed and fast results match."""
        np.testing.assert_allclose(
            results["detailed"]["model_mass"],
            results["fast"]["model_mass"],
            atol=atol,
            err_msg="Model mass mismatch between detailed and fast paths",
        )
        np.testing.assert_allclose(
            results["detailed"]["model_inertia"],
            results["fast"]["model_inertia"],
            atol=atol,
            err_msg="Model inertia mismatch between detailed and fast paths",
        )

    def test_parity_negative_mass(self):
        """Both paths should correct negative mass identically."""
        results = self._finalize_both_paths(mass=-5.0, inertia=np.diag([1.0, 1.0, 1.0]))
        self._assert_parity(results)
        self.assertEqual(results["detailed"]["model_mass"], 0.0)

    def test_parity_zero_mass_with_bound(self):
        """Zero mass must stay zero even with bound_mass set."""
        results = self._finalize_both_paths(mass=0.0, inertia=np.diag([1.0, 1.0, 1.0]), bound_mass=0.1)
        self._assert_parity(results)
        self.assertEqual(results["detailed"]["model_mass"], 0.0)
        self.assertEqual(results["fast"]["model_mass"], 0.0)

    def test_parity_positive_mass_below_bound(self):
        """Both paths should clamp positive mass below bound identically."""
        results = self._finalize_both_paths(mass=0.05, inertia=np.diag([1.0, 1.0, 1.0]), bound_mass=0.1)
        self._assert_parity(results)
        self.assertAlmostEqual(results["detailed"]["model_mass"], 0.1, places=5)

    def test_parity_negative_mass_with_bound(self):
        """Negative mass should become zero, not bound_mass."""
        results = self._finalize_both_paths(mass=-1.0, inertia=np.diag([1.0, 1.0, 1.0]), bound_mass=0.1)
        self._assert_parity(results)
        self.assertEqual(results["detailed"]["model_mass"], 0.0)

    def test_parity_singular_inertia(self):
        """Both paths should repair singular inertia for positive-mass bodies."""
        results = self._finalize_both_paths(mass=1.0, inertia=np.zeros((3, 3)))
        self._assert_parity(results)
        # Inertia should be positive-definite
        eigenvalues = np.linalg.eigvals(results["detailed"]["model_inertia"])
        self.assertTrue(np.all(eigenvalues > 0))

    def test_parity_semidefinite_inertia(self):
        """Both paths should repair semi-definite inertia (one zero eigenvalue)."""
        results = self._finalize_both_paths(mass=1.0, inertia=np.diag([0.0, 1.0, 1.0]))
        self._assert_parity(results)

    def test_parity_nonsymmetric_inertia(self):
        """Both paths should symmetrize non-symmetric inertia."""
        inertia = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        results = self._finalize_both_paths(mass=1.0, inertia=inertia)
        self._assert_parity(results)
        # Result should be symmetric
        inertia = results["detailed"]["model_inertia"]
        np.testing.assert_allclose(inertia, inertia.T, atol=1e-6)

    def test_parity_exact_triangle_boundary(self):
        """Exact triangle equality diag(1,1,2) should be a no-op in both paths."""
        results = self._finalize_both_paths(mass=1.0, inertia=np.diag([1.0, 1.0, 2.0]))
        self._assert_parity(results)

    def test_parity_triangle_inequality_violation(self):
        """Both paths should correct triangle inequality violations identically."""
        results = self._finalize_both_paths(mass=1.0, inertia=np.diag([0.1, 0.1, 10.0]))
        self._assert_parity(results)

    def test_parity_near_boundary_triangle(self):
        """Near-boundary triangle cases should get consistent corrections."""
        results = self._finalize_both_paths(mass=1.0, inertia=np.diag([1.0, 1.0, 2.001]))
        self._assert_parity(results)

    def test_parity_nan_mass(self):
        """Both paths should handle NaN mass identically (zero out)."""
        results = self._finalize_both_paths(mass=float("nan"), inertia=np.diag([1.0, 1.0, 1.0]))
        self._assert_parity(results)
        self.assertEqual(results["detailed"]["model_mass"], 0.0)

    def test_parity_nan_inertia(self):
        """Both paths should handle NaN inertia identically."""
        inertia = np.diag([float("nan"), 1.0, 1.0])
        results = self._finalize_both_paths(mass=1.0, inertia=inertia)
        self._assert_parity(results)
        self.assertEqual(results["detailed"]["model_mass"], 0.0)

    def test_parity_inf_inertia(self):
        """Both paths should handle Inf inertia identically."""
        inertia = np.diag([float("inf"), 1.0, 1.0])
        results = self._finalize_both_paths(mass=1.0, inertia=inertia)
        self._assert_parity(results)
        self.assertEqual(results["detailed"]["model_mass"], 0.0)

    def test_parity_valid_inertia(self):
        """Valid inertia should pass through unchanged in both paths."""
        inertia = np.diag([2.0, 3.0, 4.0])
        results = self._finalize_both_paths(mass=1.0, inertia=inertia)
        self._assert_parity(results)
        np.testing.assert_allclose(results["detailed"]["model_inertia"], np.diag([2.0, 3.0, 4.0]), atol=1e-5)

    def test_lightweight_inertia_preserved(self):
        """Test that small but valid inertia for lightweight components is not inflated."""
        # Franka Panda finger-like inertia (7.5e-7 < old absolute 1e-6 threshold,
        # but valid relative to max eigenvalue)
        diag = [2.375e-6, 2.375e-6, 7.5e-7]
        small_inertia = wp.mat33(np.diag(diag).astype(np.float32))

        for detailed in [True, False]:
            with self.subTest(detailed=detailed):
                builder = ModelBuilder()
                builder.validate_inertia_detailed = detailed
                idx = builder.add_body(mass=0.015, inertia=small_inertia, label="finger")
                with warnings.catch_warnings(record=True) as w:
                    model = builder.finalize()
                self.assertEqual(len(w), 0, f"Unexpected warnings: {[str(x.message) for x in w]}")
                np.testing.assert_allclose(model.body_inertia.numpy()[idx].diagonal(), diag, atol=1e-10)

    def test_lightweight_inertia_parity(self):
        """Test that both paths preserve lightweight inertia identically."""
        # Robotiq 2F85 gripper pad inertia (all eigenvalues below 1e-6)
        small_inertia = np.diag([4.74e-7, 3.65e-7, 1.24e-7])
        results = {}
        for detailed in [True, False]:
            builder = ModelBuilder()
            builder.validate_inertia_detailed = detailed
            idx = builder.add_body(mass=0.0035, inertia=wp.mat33(small_inertia.astype(np.float32)), label="pad")
            with warnings.catch_warnings(record=True) as w:
                model = builder.finalize()
            self.assertEqual(len(w), 0, f"Unexpected warnings: {[str(x.message) for x in w]}")
            mode = "detailed" if detailed else "fast"
            results[mode] = {
                "model_mass": float(model.body_mass.numpy()[idx]),
                "model_inertia": np.array(model.body_inertia.numpy()[idx]),
            }
        self._assert_parity(results)
        np.testing.assert_allclose(
            results["detailed"]["model_inertia"].diagonal(), small_inertia.diagonal(), atol=1e-10
        )

    def test_builder_state_unchanged_after_finalize(self):
        """finalize() should not mutate builder state — corrections live only on the Model."""
        for detailed in (True, False):
            with self.subTest(detailed=detailed):
                builder = ModelBuilder()
                builder.validate_inertia_detailed = detailed

                original_mass = -1.0
                original_inertia = wp.mat33([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                body_idx = builder.add_body(
                    mass=original_mass,
                    inertia=original_inertia,
                    label="test_body",
                )

                with warnings.catch_warnings(record=True):
                    model = builder.finalize()

                # Builder retains original (uncorrected) values
                self.assertEqual(builder.body_mass[body_idx], original_mass)

                # Model has corrected values
                self.assertAlmostEqual(float(model.body_mass.numpy()[body_idx]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
