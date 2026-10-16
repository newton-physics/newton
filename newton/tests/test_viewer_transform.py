# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.viewer.transform import (
    transform_add_translation,
    transform_assign,
    transform_assign_matrix,
    transform_assign_position_wxyz,
    transform_from_array,
    transform_inverse,
    transform_multiply,
    transform_point,
    transform_to_matrix,
    transform_to_position_wxyz,
    transform_vector,
)


class TestViewerTransform(unittest.TestCase):
    def setUp(self):
        self.transform = wp.transform(
            wp.vec3(1.0, 2.0, 3.0),
            wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7),
        )

    def test_conversion_round_trip(self):
        """Round-trip Viser and matrix representations in place."""
        position, wxyz = transform_to_position_wxyz(self.transform)
        from_viser = wp.transform_identity()
        transform_assign_position_wxyz(from_viser, position, wxyz)
        np.testing.assert_allclose(from_viser, self.transform)

        from_matrix = wp.transform_identity()
        transform_assign_matrix(from_matrix, transform_to_matrix(self.transform))
        np.testing.assert_allclose(from_matrix, self.transform, rtol=1.0e-6, atol=1.0e-6)

    def test_transform_operations_match_warp(self):
        """Match Warp's transform operations without public host calls."""
        other = wp.transform(wp.vec3(-2.0, 0.5, 4.0), wp.quat_rpy(0.2, -0.4, 0.1))
        point = wp.vec3(2.0, -1.0, 0.5)

        np.testing.assert_allclose(transform_inverse(self.transform), wp.transform_inverse(self.transform))
        np.testing.assert_allclose(
            transform_multiply(self.transform, other), wp.transform_multiply(self.transform, other)
        )
        np.testing.assert_allclose(transform_point(self.transform, point), wp.transform_point(self.transform, point))
        np.testing.assert_allclose(transform_vector(self.transform, point), wp.transform_vector(self.transform, point))

    def test_array_copy_and_translation_preserve_inputs(self):
        """Create fast views and copies without replacing caller-owned values."""
        source = np.asarray((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        from_array = transform_from_array(source)
        target = wp.transform_identity()
        transform_assign(target, from_array)
        translated = transform_add_translation(target, (4.0, 5.0, 6.0))

        np.testing.assert_allclose(target, source)
        np.testing.assert_allclose(translated, (5.0, 7.0, 9.0, 0.0, 0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
