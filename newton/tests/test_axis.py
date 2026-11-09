# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton


class TestAxis(unittest.TestCase):
    def test_integer_equality_and_hashing(self):
        """Match integer keys and set members for every axis."""
        for axis in newton.Axis:
            with self.subTest(axis=axis):
                value = int(axis)
                self.assertEqual(axis, value)
                self.assertEqual(value, axis)
                self.assertFalse(axis != value)
                self.assertFalse(value != axis)
                self.assertEqual(hash(axis), hash(value))
                self.assertEqual({axis: "axis"}[value], "axis")
                self.assertEqual({value: "integer"}[axis], "integer")
                self.assertIn(value, {axis})
                self.assertIn(axis, {value})

    def test_string_comparison_warns_and_preserves_equality(self):
        """Warn while preserving string comparisons during deprecation."""
        axis = newton.Axis.X
        for string in ("x", "X"):
            with self.subTest(string=string):
                with self.assertWarnsRegex(DeprecationWarning, "Axis.from_any"):
                    self.assertTrue(axis == string)
                with self.assertWarnsRegex(DeprecationWarning, "Axis.from_any"):
                    self.assertTrue(string == axis)
                with self.assertWarnsRegex(DeprecationWarning, "Axis.from_any"):
                    self.assertFalse(axis != string)
                with self.assertWarnsRegex(DeprecationWarning, "Axis.from_any"):
                    self.assertFalse(string != axis)

        with self.assertWarnsRegex(DeprecationWarning, "Axis.from_any"):
            self.assertFalse(axis == "y")
        with self.assertWarnsRegex(DeprecationWarning, "Axis.from_any"):
            self.assertTrue(axis != "y")

    def test_normalized_string_keys(self):
        """Use converted strings as consistent axis dictionary keys."""
        axes = {newton.Axis.X: "value"}
        self.assertEqual(axes[newton.Axis.from_any("x")], "value")
        self.assertEqual(axes[newton.Axis.from_any("X")], "value")

    def test_axis_inequality(self):
        """Keep distinct axes unequal and each axis equal to itself."""
        self.assertEqual(newton.Axis.X, newton.Axis.X)
        self.assertFalse(newton.Axis.X != newton.Axis.X)
        self.assertNotEqual(newton.Axis.X, newton.Axis.Y)
        self.assertTrue(newton.Axis.X != newton.Axis.Y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
