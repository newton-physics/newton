# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.utils import compute_inertia_obb
from newton.tests.unittest_utils import assert_np_equal, requires


class TestOBB(unittest.TestCase):
    def test_empty_vertices(self):
        """Test OBB computation with empty vertices."""
        vertices = np.array([], dtype=np.float32).reshape(0, 3)

        tf_inertia, extents_inertia = compute_inertia_obb(vertices)
        assert_np_equal(np.array(tf_inertia[0:3]), np.zeros(3), tol=1e-6)
        assert_np_equal(np.array(extents_inertia), np.zeros(3), tol=1e-6)

    def test_single_vertex(self):
        """Test OBB computation with a single vertex."""
        vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        tf_inertia, extents_inertia = compute_inertia_obb(vertices)
        assert_np_equal(np.array(tf_inertia[0:3]), vertices[0], tol=1e-6)
        assert_np_equal(np.array(extents_inertia), np.zeros(3), tol=1e-6)

    @requires("scipy")
    def test_axis_aligned_box(self):
        """Test OBB computation on an axis-aligned box."""
        # Create vertices of a 2x3x4 box centered at origin
        vertices = np.array(
            [
                [-1, -1.5, -2],
                [1, -1.5, -2],
                [1, 1.5, -2],
                [-1, 1.5, -2],
                [-1, -1.5, 2],
                [1, -1.5, 2],
                [1, 1.5, 2],
                [-1, 1.5, 2],
            ],
            dtype=np.float32,
        )

        tf_inertia, extents_inertia = compute_inertia_obb(vertices)
        assert_np_equal(np.array(tf_inertia[0:3]), np.zeros(3), tol=1e-4)
        # Half-extents; sorted because the axis order may vary.
        expected_extents = np.array([1.0, 1.5, 2.0])
        assert_np_equal(np.sort(np.array(extents_inertia)), np.sort(expected_extents), tol=1e-4)

    @requires("scipy")
    def test_rotated_box(self):
        """Test OBB computation on a rotated box."""
        # Create a box and rotate it 45 degrees around Z axis
        angle = np.pi / 4
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        # Original box vertices (2x3x4)
        box_vertices = np.array(
            [
                [-1, -1.5, -2],
                [1, -1.5, -2],
                [1, 1.5, -2],
                [-1, 1.5, -2],
                [-1, -1.5, 2],
                [1, -1.5, 2],
                [1, 1.5, 2],
                [-1, 1.5, 2],
            ],
            dtype=np.float32,
        )

        # Rotate vertices
        vertices = (rotation @ box_vertices.T).T

        tf_inertia, extents_inertia = compute_inertia_obb(vertices)
        extents_inertia_np = np.array(extents_inertia)
        volume_inertia = 8.0 * extents_inertia_np[0] * extents_inertia_np[1] * extents_inertia_np[2]

        self.assertGreater(volume_inertia, 0)

        # Inertia method might find a different orientation, but volume shouldn't be
        # drastically larger than the original box volume (24.0)
        self.assertLess(volume_inertia, 100.0)  # Reasonable upper bound

        # Center should be at origin
        assert_np_equal(np.array(tf_inertia[0:3]), np.zeros(3), tol=1e-4)

        # Test orientation: verify OBB axes are orthogonal and properly aligned
        quat_inertia = np.array(tf_inertia[3:7])  # [x, y, z, w]
        quat_wp = wp.quat(*quat_inertia)

        # Compute the OBB axes by rotating the standard basis vectors
        x_axis = np.array(wp.quat_rotate(quat_wp, wp.vec3(1.0, 0.0, 0.0)))
        y_axis = np.array(wp.quat_rotate(quat_wp, wp.vec3(0.0, 1.0, 0.0)))
        z_axis = np.array(wp.quat_rotate(quat_wp, wp.vec3(0.0, 0.0, 1.0)))

        # Verify axes are orthonormal
        self.assertAlmostEqual(np.linalg.norm(x_axis), 1.0, delta=1e-6)
        self.assertAlmostEqual(np.linalg.norm(y_axis), 1.0, delta=1e-6)
        self.assertAlmostEqual(np.linalg.norm(z_axis), 1.0, delta=1e-6)
        self.assertAlmostEqual(np.dot(x_axis, y_axis), 0.0, delta=1e-6)
        self.assertAlmostEqual(np.dot(y_axis, z_axis), 0.0, delta=1e-6)
        self.assertAlmostEqual(np.dot(z_axis, x_axis), 0.0, delta=1e-6)

        # For a box rotated 45° around Z:
        # - One axis aligns with Z: z ≈ ±1, x ≈ 0, y ≈ 0
        # - Two axes in XY plane at 45°: z ≈ 0, x,y ≈ ±0.707

        axes = [x_axis, y_axis, z_axis]

        # Count axes aligned with Z
        num_z_aligned = sum(
            1 for axis in axes if abs(abs(axis[2]) - 1.0) < 0.1 and abs(axis[0]) < 0.1 and abs(axis[1]) < 0.1
        )

        # Count axes in XY plane with 45° rotation (x,y components ≈ ±0.707)
        num_xy_at_45deg = sum(
            1
            for axis in axes
            if abs(axis[2]) < 0.1 and abs(abs(axis[0]) - 0.707) < 0.1 and abs(abs(axis[1]) - 0.707) < 0.1
        )

        # Exactly one axis should align with Z, two should be in XY plane at 45°
        self.assertEqual(num_z_aligned, 1, "Exactly one OBB axis should align with global Z")
        self.assertEqual(num_xy_at_45deg, 2, "Exactly two OBB axes should be in XY plane at 45° angle")

    @requires("scipy")
    def test_concave_mesh(self):
        """Test OBB computation on a concave shape (L-shaped)."""
        # Create an L-shaped point cloud
        vertices = []
        # Horizontal part of L
        for x in np.linspace(0, 4, 10):
            for y in np.linspace(0, 1, 5):
                vertices.append([x, y, 0])
                vertices.append([x, y, 1])
        # Vertical part of L
        for x in np.linspace(0, 1, 5):
            for y in np.linspace(1, 4, 10):
                vertices.append([x, y, 0])
                vertices.append([x, y, 1])

        vertices = np.array(vertices, dtype=np.float32)

        _tf_inertia, extents_inertia = compute_inertia_obb(vertices)
        extents_inertia_np = np.array(extents_inertia)

        # Basic sanity check - extents should be positive
        self.assertTrue(np.all(extents_inertia_np > 0))

        # The bounding box should encompass the L shape, whose arms are 4 long
        max_extents_inertia = np.max(extents_inertia_np) * 2  # Convert half-extent to full
        self.assertGreaterEqual(max_extents_inertia, 3.9)

    @requires("scipy")
    def test_symmetric_shape(self):
        """Test OBB computation on a highly symmetric shape with equal variance on all axes."""
        # Create vertices of a regular octahedron (all axes have equal variance)
        vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32)

        # Add some copies to increase point density
        vertices = np.vstack([vertices] * 10)

        tf_inertia, extents_inertia = compute_inertia_obb(vertices)

        # Center should be at origin
        assert_np_equal(np.array(tf_inertia[0:3]), np.zeros(3), tol=1e-6)

        # The octahedron has many valid OBB orientations, so only check that the box is non-degenerate
        extents_inertia_np = np.array(extents_inertia)
        self.assertTrue(np.all(extents_inertia_np > 0))

    @requires("scipy")
    def test_deterministic_results(self):
        """Test that OBB computation gives consistent results across multiple runs."""
        # Create a simple test shape
        vertices = np.array(
            [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [0.5, 0.5, 3], [1.5, 0.5, 3]], dtype=np.float32
        )

        # Run multiple times and check consistency
        for _ in range(5):
            tf1, ext1 = compute_inertia_obb(vertices, num_angle_steps=90)
            tf2, ext2 = compute_inertia_obb(vertices, num_angle_steps=90)

            # Results should be identical
            assert_np_equal(np.array(tf1[0:3]), np.array(tf2[0:3]), tol=1e-6)
            assert_np_equal(np.array(ext1), np.array(ext2), tol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
