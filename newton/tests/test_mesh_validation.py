# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh quality validation utilities."""

import unittest
import warnings

import numpy as np

from newton._src.utils.mesh import validate_tet_mesh, validate_triangle_mesh


def _equilateral_triangle(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a single equilateral triangle (vertices, indices)."""
    h = scale * np.sqrt(3) / 2
    verts = np.array([[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [scale / 2, h, 0.0]])
    indices = np.array([[0, 1, 2]])
    return verts, indices


def _regular_tet(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a single regular tetrahedron (vertices, indices)."""
    s = scale
    verts = np.array([
        [1, 1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, 1],
    ], dtype=float) * (s / 2)
    indices = np.array([[0, 1, 2, 3]])
    return verts, indices


class TestValidateTriangleMesh(unittest.TestCase):
    def test_clean_mesh_no_warning(self):
        verts, inds = _equilateral_triangle(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 0)

    def test_empty_mesh_warns(self):
        verts = np.zeros((0, 3))
        inds = np.zeros((0, 3), dtype=int)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("no triangles", str(w[0].message))

    def test_small_area(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1e-4, 0.0, 0.0],
            [0.5e-4, 1e-4, 0.0],
        ])
        inds = np.array([[0, 1, 2]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("area < 1 mm", str(w[0].message))

    def test_sliver_triangle(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1e-3, 0.0],
        ])
        inds = np.array([[0, 1, 2]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("sliver", str(w[0].message))
        self.assertIn("aspect ratio > 10", str(w[0].message))

    def test_extreme_angle(self):
        angle_rad = np.radians(2.0)
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.1, np.tan(angle_rad) * 0.1, 0.0],
        ])
        inds = np.array([[0, 1, 2]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("minimum angle < 5", msgs)

    def test_non_manifold_edge(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 0.0, 1.0],
        ])
        inds = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("non-manifold", msgs)

    def test_borderline_pass(self):
        area_target = 2e-6
        h = 2 * area_target / 0.01
        verts = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.005, h, 0.0],
        ])
        inds = np.array([[0, 1, 2]])
        actual_area = 0.5 * 0.01 * h
        self.assertGreater(actual_area, 1e-6)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        quality_warnings = [wi for wi in w if "area < 1 mm" in str(wi.message)]
        self.assertEqual(len(quality_warnings), 0)

    def test_mixed_issues(self):
        good_verts = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.05, 0.08, 0.0],
        ])
        tiny_verts = np.array([
            [0.0, 0.0, 0.0],
            [1e-4, 0.0, 0.0],
            [0.5e-4, 1e-4, 0.0],
        ])
        sliver_verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1e-3, 0.0],
        ])
        verts = np.vstack([good_verts, tiny_verts + [2, 0, 0], sliver_verts + [4, 0, 0]])
        inds = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        msg = str(w[0].message)
        self.assertIn("area < 1 mm", msg)
        self.assertIn("sliver", msg)

    def test_remesh_suggestion(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1e-4, 0.0, 0.0],
            [0.5e-4, 1e-4, 0.0],
        ])
        inds = np.array([[0, 1, 2]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertIn("Consider remeshing", str(w[0].message))


class TestValidateTetMesh(unittest.TestCase):
    def test_clean_mesh_no_warning(self):
        verts, inds = _regular_tet(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 0)

    def test_empty_mesh_warns(self):
        verts = np.zeros((0, 3))
        inds = np.zeros((0, 4), dtype=int)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("no tetrahedra", str(w[0].message))

    def test_inverted_tet(self):
        verts, inds = _regular_tet(scale=0.1)
        inds[0, 2], inds[0, 3] = inds[0, 3], inds[0, 2]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("inverted", str(w[0].message))

    def test_small_volume(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1e-3, 0.0, 0.0],
            [0.0, 1e-3, 0.0],
            [0.0, 0.0, 1e-3],
        ])
        inds = np.array([[0, 1, 2, 3]])
        vol = abs(np.dot(verts[1] - verts[0], np.cross(verts[2] - verts[0], verts[3] - verts[0]))) / 6
        self.assertLess(vol, 1e-9)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("volume < 1 mm", msgs)

    def test_flat_sliver(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1e-6],
        ])
        inds = np.array([[0, 1, 2, 3]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("sliver", msgs)

    def test_needle_sliver(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.01, 0.01, 0.0],
            [0.005, 0.005, 0.01],
        ])
        inds = np.array([[0, 1, 2, 3]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("sliver", msgs)

    def test_non_manifold_face(self):
        base_verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.3, 1.0],
            [0.5, 0.3, -1.0],
            [0.5, 0.3, 2.0],
        ])
        inds = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 2, 5],
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(base_verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("non-manifold", msgs)

    def test_no_remesh_suggestion_in_tet_warning(self):
        verts, inds = _regular_tet(scale=0.1)
        inds[0, 2], inds[0, 3] = inds[0, 3], inds[0, 2]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertNotIn("remeshing", str(w[0].message))


class TestPublicExport(unittest.TestCase):
    def test_importable_from_newton_utils(self):
        from newton.utils import validate_tet_mesh as vtm
        from newton.utils import validate_triangle_mesh as vttm

        self.assertTrue(callable(vtm))
        self.assertTrue(callable(vttm))


if __name__ == "__main__":
    unittest.main()
