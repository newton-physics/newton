# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton


def triangle_areas(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute per-triangle areas from flattened triangle indices.

    Args:
        vertices: Vertex positions with shape (N, 3).
        indices: Flattened triangle indices with shape (3 * M,).

    Returns:
        Triangle areas with shape (M,).
    """
    tris = indices.reshape(-1, 3)
    a = vertices[tris[:, 0]]
    b = vertices[tris[:, 1]]
    c = vertices[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


class TestMeshPrimitives(unittest.TestCase):
    def test_cone_has_no_degenerate_triangles(self):
        mesh = newton.Mesh.create_cone(
            1.0,
            0.75,
            segments=32,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )

        areas = triangle_areas(mesh.vertices, mesh.indices)
        self.assertTrue(np.all(areas > 1.0e-10))

    def test_truncated_cylinder_has_no_degenerate_triangles(self):
        mesh = newton.Mesh.create_cylinder(
            1.0,
            0.75,
            top_radius=0.3,
            segments=32,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )

        areas = triangle_areas(mesh.vertices, mesh.indices)
        self.assertTrue(np.all(areas > 1.0e-10))


if __name__ == "__main__":
    unittest.main()
