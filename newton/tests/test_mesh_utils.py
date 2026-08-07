# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared mesh utilities."""

import unittest

import numpy as np
import warp as wp

from newton._src.utils.mesh import compute_vertex_normals


class TestComputeVertexNormals(unittest.TestCase):
    """Verify normal computation accepts the documented index layouts."""

    def test_two_dimensional_warp_indices(self):
        """Warp triangle indices may use the same (N, 3) layout as NumPy."""
        points = wp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=wp.vec3,
            device="cpu",
        )
        indices = wp.array([[0, 1, 2]], dtype=wp.int32, device="cpu")

        normals = compute_vertex_normals(points, indices)

        np.testing.assert_allclose(
            normals.numpy(),
            np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
