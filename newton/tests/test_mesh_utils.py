# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared mesh utilities."""

import unittest

import numpy as np
import warp as wp

from newton._src.utils.mesh import compute_vertex_normals


class TestComputeVertexNormals(unittest.TestCase):
    """Verify normal computation accepts the documented index layouts."""

    def test_supported_index_layouts(self):
        """Verify flat and triangle-shaped Warp and NumPy indices."""
        points_np = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        points_wp = wp.array(
            points_np,
            dtype=wp.vec3,
            device="cpu",
        )
        expected = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
        warp_index_cases = {
            "flat Warp": wp.array([0, 1, 2], dtype=wp.int32, device="cpu"),
            "triangle-shaped Warp": wp.array([[0, 1, 2]], dtype=wp.int32, device="cpu"),
            "flat NumPy": np.array([0, 1, 2], dtype=np.int32),
            "triangle-shaped NumPy": np.array([[0, 1, 2]], dtype=np.int32),
        }

        for name, indices in warp_index_cases.items():
            with self.subTest(points="Warp", indices=name):
                normals = compute_vertex_normals(points_wp, indices)
                np.testing.assert_allclose(normals.numpy(), expected)

        for name, indices in {
            "flat NumPy": np.array([0, 1, 2], dtype=np.int32),
            "triangle-shaped NumPy": np.array([[0, 1, 2]], dtype=np.int32),
        }.items():
            with self.subTest(points="NumPy", indices=name):
                normals = compute_vertex_normals(points_np, indices)
                np.testing.assert_allclose(normals, expected)

    def test_invalid_warp_index_layouts(self):
        """Reject Warp indices that are neither flat nor triangle-shaped."""
        points = wp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=wp.vec3,
            device="cpu",
        )
        invalid_shapes = ((1, 2), (1, 4), (1, 1, 3))

        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                indices = wp.array(np.arange(np.prod(shape), dtype=np.int32).reshape(shape), device="cpu")
                with self.assertRaisesRegex(ValueError, "indices must be flat or \\(N, 3\\) for Warp inputs"):
                    compute_vertex_normals(points, indices)

    def test_invalid_numpy_index_layouts(self):
        """Reject NumPy indices that are neither flat nor triangle-shaped."""
        points_np = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        points_wp = wp.array(points_np, dtype=wp.vec3, device="cpu")
        indices = np.array([[0, 1, 2, 0]], dtype=np.int32)

        for name, points in (("Warp", points_wp), ("NumPy", points_np)):
            with self.subTest(points=name):
                with self.assertRaisesRegex(ValueError, "indices must be flat or \\(N, 3\\) for NumPy inputs"):
                    compute_vertex_normals(points, indices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
