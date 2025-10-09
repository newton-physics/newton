# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            self.assertIn("Negative eigenvalues detected", str(w[0].message))

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
            I_m=invalid_inertia,  # Violates triangle inequality
            key="test_body",
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
            I_m=invalid_inertia,  # Violates triangle inequality
            key="test_body",
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
            key="test_body",
        )

        with warnings.catch_warnings(record=True) as w:
            model = builder.finalize()

            # Should get warning about issues found
            self.assertEqual(len(w), 1)
            self.assertIn("Inertia validation corrected 1 bodies", str(w[0].message))

            # Mass should be corrected to 0
            body_mass = model.body_mass.numpy()[body_idx]
            self.assertEqual(body_mass, 0.0)


if __name__ == "__main__":
    unittest.main()
