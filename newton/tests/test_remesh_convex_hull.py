# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`newton._src.geometry.utils.remesh_convex_hull`.

Covers both the full-3D happy path (regression guard) and the degeneracy
handling added on top of ``scipy.spatial.ConvexHull``:

- empty and single-point input
- coincident points (rank 0)
- collinear points (rank 1)
- coplanar points (rank 2), including ``maxhullvert`` decimation
- general 3D input (rank 3), including ``maxhullvert`` and winding
- the ``eps`` relative tolerance controlling near-flat classification
- a randomized no-raise property over degenerate configurations

The function is not re-exported from any public module, so tests import it
directly from ``newton._src.geometry.utils`` — consistent with how
``test_remesh.py`` already reaches into ``newton._src.geometry``.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from newton._src.geometry.utils import remesh_convex_hull


def _assert_mesh_shape(test: unittest.TestCase, verts: np.ndarray, faces: np.ndarray) -> None:
    """Assert the basic contract: ``verts`` is (M>=3, 3) float32, ``faces`` is (K>=2, 3) int32."""
    test.assertEqual(verts.dtype, np.float32, "verts dtype must be float32")
    test.assertEqual(faces.dtype, np.int32, "faces dtype must be int32")
    test.assertEqual(verts.ndim, 2)
    test.assertEqual(faces.ndim, 2)
    test.assertEqual(verts.shape[1], 3)
    test.assertEqual(faces.shape[1], 3)
    test.assertGreaterEqual(verts.shape[0], 3, "verts must have at least 3 rows")
    test.assertGreaterEqual(faces.shape[0], 2, "faces must have at least 2 rows")
    test.assertTrue(np.all(faces >= 0))
    test.assertTrue(np.all(faces < verts.shape[0]))


def _has_outward_winding(verts: np.ndarray, faces: np.ndarray) -> bool:
    """Return True if every triangle's normal points away from the mesh centroid.

    Uses the same criterion the implementation uses to flip windings. Triangles
    with zero-area normals are ignored (they're degenerate but not wrongly wound).
    """
    centre = verts.mean(axis=0)
    for tri in faces:
        a, b, c = verts[tri]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-12:
            continue
        if np.dot(normal, a - centre) < 0:
            return False
    return True


class TestRemeshConvexHullFull3D(unittest.TestCase):
    """Regression tests for the untouched full-rank (3D) path."""

    def test_unit_cube(self):
        # The 8 corners of a unit cube centred at the origin.
        verts_in = np.array(
            [
                [-1, -1, -1],
                [-1, -1, +1],
                [-1, +1, -1],
                [-1, +1, +1],
                [+1, -1, -1],
                [+1, -1, +1],
                [+1, +1, -1],
                [+1, +1, +1],
            ],
            dtype=np.float64,
        )
        verts, faces = remesh_convex_hull(verts_in)
        _assert_mesh_shape(self, verts, faces)

        # A cube's convex hull: 8 unique corner vertices, 12 triangles.
        self.assertEqual(verts.shape[0], 8)
        self.assertEqual(faces.shape[0], 12)
        self.assertTrue(_has_outward_winding(verts, faces), "cube must have outward-facing normals")

        # Every returned vertex must be one of the inputs.
        for v in verts:
            match = np.all(np.isclose(verts_in.astype(np.float32), v), axis=1)
            self.assertTrue(np.any(match), f"output vertex {v} not present in input")

    def test_tetrahedron(self):
        # Minimal full-rank case (4 non-coplanar points).
        verts_in = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        verts, faces = remesh_convex_hull(verts_in)
        _assert_mesh_shape(self, verts, faces)
        self.assertEqual(verts.shape[0], 4)
        self.assertEqual(faces.shape[0], 4)
        self.assertTrue(_has_outward_winding(verts, faces))

    def test_interior_points_are_trimmed(self):
        # Cube corners plus an interior point. The interior point must be
        # stripped by the "trim vertices to only those used in faces" pass.
        cube = np.array(
            [
                [-1, -1, -1],
                [-1, -1, +1],
                [-1, +1, -1],
                [-1, +1, +1],
                [+1, -1, -1],
                [+1, -1, +1],
                [+1, +1, -1],
                [+1, +1, +1],
            ],
            dtype=np.float64,
        )
        verts_in = np.vstack([cube, np.array([[0.0, 0.0, 0.0]])])
        verts, faces = remesh_convex_hull(verts_in)
        _assert_mesh_shape(self, verts, faces)
        self.assertEqual(verts.shape[0], 8, "interior point must be trimmed")
        # No returned vertex equals the interior point.
        self.assertFalse(np.any(np.all(np.isclose(verts, 0.0), axis=1)))

    def test_maxhullvert_limits_vertex_count(self):
        # Many points on the surface of a sphere → hull would have many verts.
        # maxhullvert should cap the output vertex count.
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((200, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

        verts_unlimited, _ = remesh_convex_hull(pts, maxhullvert=0)
        verts_limited, faces_limited = remesh_convex_hull(pts, maxhullvert=16)
        _assert_mesh_shape(self, verts_limited, faces_limited)

        self.assertGreater(verts_unlimited.shape[0], 16)
        self.assertLessEqual(verts_limited.shape[0], 16)
        self.assertTrue(_has_outward_winding(verts_limited, faces_limited))


class TestRemeshConvexHullDegenerate(unittest.TestCase):
    """Degeneracy handling added by the diff."""

    def setUp(self):
        # The degenerate branches intentionally emit a UserWarning; the tests
        # in this class assert geometry, not the warning (see
        # TestRemeshConvexHullDegenerateWarnings for that). Silence them here
        # so they don't pollute test output.
        self._warn_ctx = warnings.catch_warnings()
        self._warn_ctx.__enter__()
        warnings.simplefilter("ignore", UserWarning)

    def tearDown(self):
        self._warn_ctx.__exit__(None, None, None)

    def test_empty_input_raises(self):
        # Empty input must raise rather than fabricate a phantom collider
        # at the origin; callers (Mesh.compute_convex_hull,
        # ModelBuilder.approximate_meshes, ...) are responsible for deciding
        # whether to skip or supply a fallback.
        with self.assertRaises(ValueError):
            remesh_convex_hull(np.zeros((0, 3), dtype=np.float64))

    def test_single_point(self):
        p = np.array([[1.5, -2.25, 0.125]], dtype=np.float64)
        verts, faces = remesh_convex_hull(p)
        _assert_mesh_shape(self, verts, faces)
        self.assertEqual(verts.shape, (3, 3))
        self.assertEqual(faces.shape, (2, 3))
        # All three output verts must equal the input point.
        np.testing.assert_allclose(verts, np.tile(p.astype(np.float32), (3, 1)))
        # Two triangles with opposite winding (same indices, flipped).
        self.assertTrue(
            np.array_equal(faces, np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32)),
            "two opposite-winding triangles expected",
        )

    def test_coincident_points(self):
        # Many copies of the same point.
        p = np.array([0.7, -1.3, 4.2], dtype=np.float64)
        verts_in = np.tile(p, (10, 1))
        verts, faces = remesh_convex_hull(verts_in)
        _assert_mesh_shape(self, verts, faces)
        self.assertEqual(verts.shape, (3, 3))
        self.assertEqual(faces.shape, (2, 3))
        # All output verts match the shared input point.
        np.testing.assert_allclose(verts, np.tile(p.astype(np.float32), (3, 1)), atol=1e-6)

    def test_collinear_points(self):
        # Points along the x-axis with varying x, identical y/z.
        xs = np.array([-2.0, -0.5, 0.0, 0.25, 1.5, 3.0], dtype=np.float64)
        verts_in = np.stack([xs, np.full_like(xs, 1.0), np.full_like(xs, -2.0)], axis=1)
        verts, faces = remesh_convex_hull(verts_in)
        _assert_mesh_shape(self, verts, faces)
        self.assertEqual(verts.shape, (3, 3))
        self.assertEqual(faces.shape, (2, 3))

        # The two extrema should appear among the output verts.
        xs_out = sorted(verts[:, 0].tolist())
        self.assertAlmostEqual(xs_out[0], xs.min(), places=5)
        self.assertAlmostEqual(xs_out[-1], xs.max(), places=5)
        # y and z are constant along the input line.
        np.testing.assert_allclose(verts[:, 1], 1.0, atol=1e-5)
        np.testing.assert_allclose(verts[:, 2], -2.0, atol=1e-5)

    def test_coplanar_square(self):
        # A unit square on the z=0 plane, with some interior points that
        # must not appear in the output.
        boundary = np.array(
            [
                [-1.0, -1.0, 0.0],
                [+1.0, -1.0, 0.0],
                [+1.0, +1.0, 0.0],
                [-1.0, +1.0, 0.0],
            ],
            dtype=np.float64,
        )
        interior = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.3, -0.4, 0.0],
                [-0.2, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        verts_in = np.vstack([boundary, interior])
        verts, faces = remesh_convex_hull(verts_in)
        _assert_mesh_shape(self, verts, faces)

        # Flat hull: every output vertex must lie on z=0.
        np.testing.assert_allclose(verts[:, 2], 0.0, atol=1e-5)

        # Interior points should not appear in the output.
        for p in interior:
            match = np.all(np.isclose(verts, p.astype(np.float32), atol=1e-5), axis=1)
            self.assertFalse(np.any(match), f"interior point {p} leaked into output")

        # Four corner vertices (boundary of the 2D hull).
        self.assertEqual(verts.shape[0], 4)

        # Fan triangulation emitted twice (CCW + CW): 2 * (M - 2) = 4 triangles.
        m = verts.shape[0]
        self.assertEqual(faces.shape[0], 2 * (m - 2))

        # Every CCW triangle (first half) must have a CW twin in the second half.
        half = faces.shape[0] // 2
        for i in range(half):
            ccw = faces[i]
            cw = faces[i + half]
            self.assertEqual(ccw[0], cw[0])
            self.assertEqual(ccw[1], cw[2])
            self.assertEqual(ccw[2], cw[1])

    def test_coplanar_circle_maxhullvert(self):
        # 100 points on a unit circle in the xy-plane. Request at most 8 verts.
        n = 100
        t = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        verts_in = np.stack([np.cos(t), np.sin(t), np.zeros(n)], axis=1)

        verts, faces = remesh_convex_hull(verts_in, maxhullvert=8)
        _assert_mesh_shape(self, verts, faces)
        self.assertEqual(verts.shape[0], 8, "maxhullvert must be respected on planar input")
        np.testing.assert_allclose(verts[:, 2], 0.0, atol=1e-5)
        self.assertEqual(faces.shape[0], 2 * (8 - 2))

    def test_near_flat_treated_as_coplanar_by_default(self):
        # A thin slab: large xy spread, z within ±1e-8. With default eps=1e-6
        # this is classified as coplanar (rank 2) and goes through the flat
        # branch, so every output vertex must satisfy |z| <= the input bound.
        rng = np.random.default_rng(0)
        xy = rng.uniform(-1.0, 1.0, size=(50, 2))
        z = rng.uniform(-1e-8, 1e-8, size=(50, 1))
        verts_in = np.hstack([xy, z])

        verts, faces = remesh_convex_hull(verts_in)
        _assert_mesh_shape(self, verts, faces)
        # Flat branch → every vertex is on the original sheet, so its z is
        # bounded by the input z range (not strictly zero, since the flat
        # branch keeps the original 3D coordinates of the boundary points).
        self.assertLess(np.abs(verts[:, 2]).max(), 1e-7)

        # Even though it's technically rank-3, the triangle count must match
        # the flat-branch contract (2 * (M - 2)), not the 3D-hull count.
        m = verts.shape[0]
        self.assertEqual(faces.shape[0], 2 * (m - 2))

    def test_tight_eps_allows_near_flat_3d_path(self):
        # Same slab as above, but with a very tight eps that forces the 3D
        # branch. We don't assert on exact hull counts (Qhull may flatten or
        # joggle), only that the call returns a well-formed mesh without
        # raising.
        rng = np.random.default_rng(0)
        xy = rng.uniform(-1.0, 1.0, size=(50, 2))
        z = rng.uniform(-1e-8, 1e-8, size=(50, 1))
        verts_in = np.hstack([xy, z])

        verts, faces = remesh_convex_hull(verts_in, eps=1e-12)
        _assert_mesh_shape(self, verts, faces)


class TestRemeshConvexHullNeverRaises(unittest.TestCase):
    """Property-style guarantee: degenerate input must never raise."""

    def setUp(self):
        self._warn_ctx = warnings.catch_warnings()
        self._warn_ctx.__enter__()
        warnings.simplefilter("ignore", UserWarning)

    def tearDown(self):
        self._warn_ctx.__exit__(None, None, None)

    def test_randomized_degenerate_inputs_do_not_raise(self):
        rng = np.random.default_rng(12345)
        cases: list[np.ndarray] = []

        # A mix of degenerate configurations.
        for _ in range(5):
            # Coincident cloud of random size.
            p = rng.standard_normal(3)
            k = int(rng.integers(1, 20))
            cases.append(np.tile(p, (k, 1)))

        for _ in range(5):
            # Collinear cloud with random direction.
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            origin = rng.standard_normal(3)
            ts = rng.uniform(-5.0, 5.0, size=int(rng.integers(2, 30)))
            cases.append(origin + np.outer(ts, direction))

        for _ in range(5):
            # Coplanar cloud on a random plane.
            basis = rng.standard_normal((3, 3))
            q, _ = np.linalg.qr(basis)
            u, v = q[:, 0], q[:, 1]
            origin = rng.standard_normal(3)
            n_pts = int(rng.integers(3, 40))
            uv = rng.uniform(-2.0, 2.0, size=(n_pts, 2))
            cases.append(origin + uv[:, 0:1] * u + uv[:, 1:2] * v)

        # Edge cases at the boundary of the classifier. The empty case is
        # excluded here because it deliberately raises; see
        # TestRemeshConvexHullDegenerate.test_empty_input_raises.
        cases.append(np.zeros((1, 3)))
        cases.append(np.zeros((5, 3)))

        for i, pts in enumerate(cases):
            with self.subTest(case=i, shape=pts.shape):
                try:
                    verts, faces = remesh_convex_hull(pts)
                except Exception as exc:
                    self.fail(f"remesh_convex_hull raised on degenerate case {i}: {exc!r}")
                _assert_mesh_shape(self, verts, faces)


class TestRemeshConvexHullDegenerateWarnings(unittest.TestCase):
    """Each rank-0/1/2 branch must emit a UserWarning so that callers don't
    silently end up with a zero-volume, zero-mass collider."""

    def _assert_degenerate_warning(self, pts: np.ndarray) -> None:
        with self.assertWarnsRegex(UserWarning, r"remesh_convex_hull: .* zero-volume"):
            remesh_convex_hull(pts)

    def test_single_point_warns(self):
        self._assert_degenerate_warning(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))

    def test_coincident_warns(self):
        p = np.array([0.5, -0.25, 1.0], dtype=np.float64)
        self._assert_degenerate_warning(np.tile(p, (8, 1)))

    def test_collinear_warns(self):
        xs = np.linspace(-1.0, 1.0, 10)
        pts = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
        self._assert_degenerate_warning(pts)

    def test_coplanar_warns(self):
        t = np.linspace(0, 2 * np.pi, num=12, endpoint=False)
        pts = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        self._assert_degenerate_warning(pts)

    def test_full_rank_does_not_warn(self):
        # A generic 3D input must not trip any of the degeneracy warnings.
        rng = np.random.default_rng(7)
        pts = rng.standard_normal((50, 3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            remesh_convex_hull(pts)
        degeneracy_warnings = [w for w in caught if "remesh_convex_hull" in str(w.message)]
        self.assertEqual(
            degeneracy_warnings,
            [],
            f"full-rank input must not emit a degeneracy warning, got {degeneracy_warnings}",
        )


if __name__ == "__main__":
    unittest.main()
