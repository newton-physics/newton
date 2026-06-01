# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dihedral-angle edge filter used by SDF-mesh contact generation.

`Mesh._filter_edges_by_dihedral_angle` drops internal edges whose two adjacent
triangle face normals are within an angle threshold (near-coplanar). Boundary
edges and non-manifold edges are always kept. The filter is applied from
`Mesh.build_sdf()` and the resulting simplified set is cached on the mesh for
`ModelBuilder.finalize()` to consume.
"""

import math
import unittest

import numpy as np
import warp as wp

import newton

# ``Mesh.build_sdf`` requires CUDA because the SDF cook only runs on GPU.
_cuda_available = wp.is_cuda_available()


def _flat_quad_mesh() -> newton.Mesh:
    """Two coplanar triangles sharing one internal edge (in the XY plane)."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _single_triangle_mesh() -> newton.Mesh:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _near_antiparallel_pair_mesh() -> newton.Mesh:
    """Two adjacent triangles whose face normals nearly cancel."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 1.0e-7],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 1, 3], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _non_manifold_mesh() -> newton.Mesh:
    """Three triangles sharing the edge (v0, v1)."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


class TestMeshEdgeAngleFilter(unittest.TestCase):
    def test_threshold_zero_returns_full_edges(self):
        mesh = _flat_quad_mesh()
        full = mesh.edges
        filtered = mesh._filter_edges_by_dihedral_angle(0.0)
        np.testing.assert_array_equal(filtered, full)

    def test_negative_threshold_returns_full_edges(self):
        mesh = _flat_quad_mesh()
        full = mesh.edges
        filtered = mesh._filter_edges_by_dihedral_angle(-1.0)
        np.testing.assert_array_equal(filtered, full)

    def test_flat_quad_drops_internal_edge(self):
        mesh = _flat_quad_mesh()
        full = mesh.edges
        # 4 boundary edges + 1 internal coplanar edge.
        self.assertEqual(len(full), 5)

        filtered = mesh._filter_edges_by_dihedral_angle(math.radians(1.0))
        self.assertEqual(len(filtered), 4)

        kept = _edge_set(filtered)
        expected_boundary = {(0, 1), (1, 2), (2, 3), (0, 3)}
        self.assertEqual(kept, expected_boundary)

    def test_cube_drops_face_diagonals(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        full = mesh.edges
        # 12 cube edges + 6 face diagonals = 18 unique geometric edges.
        self.assertEqual(len(full), 18)

        filtered = mesh._filter_edges_by_dihedral_angle(math.radians(1.0))
        # 6 face diagonals are coplanar and should be filtered out.
        self.assertEqual(len(filtered), 12)

        # The 12 silhouette edges all have length equal to one box edge (1.0).
        verts = np.asarray(mesh.vertices)
        for a, b in filtered:
            length = float(np.linalg.norm(verts[a] - verts[b]))
            self.assertAlmostEqual(length, 1.0, places=5)

    def test_open_mesh_keeps_all_boundary_edges(self):
        mesh = _single_triangle_mesh()
        for threshold in (0.0, math.radians(1.0), math.radians(179.0)):
            filtered = mesh._filter_edges_by_dihedral_angle(threshold)
            self.assertEqual(len(filtered), 3, msg=f"threshold={threshold}")

    def test_non_manifold_edge_always_kept(self):
        mesh = _non_manifold_mesh()
        filtered = mesh._filter_edges_by_dihedral_angle(math.radians(179.0))
        kept = _edge_set(filtered)
        self.assertIn((0, 1), kept)

    def test_high_threshold_drops_low_angle_edges(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        # 90 degree dihedral on all silhouette edges; threshold above that drops them too.
        filtered = mesh._filter_edges_by_dihedral_angle(math.radians(91.0))
        self.assertEqual(len(filtered), 0)

    def test_diagnostics_shapes_and_subset(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        edges, angles, normals, area_sums = mesh._filter_edges_by_dihedral_angle(
            math.radians(1.0), return_diagnostics=True
        )
        self.assertEqual(angles.shape, (len(edges),))
        self.assertEqual(normals.shape, (len(edges), 3))
        self.assertEqual(area_sums.shape, (len(edges),))
        # Cube silhouette edges are 90 degree dihedrals between two valid triangles.
        np.testing.assert_allclose(angles, math.radians(90.0), atol=1e-5)
        finite = np.isfinite(normals).all(axis=1)
        self.assertTrue(bool(finite.all()))
        # Each silhouette edge is adjacent to two right-isoceles tris of area 0.5 -> sum 1.0.
        np.testing.assert_allclose(area_sums, 1.0, atol=1e-5)

    def test_diagnostics_nan_for_boundary_edges(self):
        mesh = _single_triangle_mesh()
        edges, angles, normals, area_sums = mesh._filter_edges_by_dihedral_angle(-1.0, return_diagnostics=True)
        self.assertEqual(len(edges), 3)
        self.assertTrue(bool(np.all(np.isnan(angles))))
        self.assertTrue(bool(np.all(np.isnan(normals))))
        self.assertTrue(bool(np.all(np.isnan(area_sums))))

    def test_diagnostics_nan_for_non_manifold_edges(self):
        mesh = _non_manifold_mesh()
        edges, angles, normals, area_sums = mesh._filter_edges_by_dihedral_angle(-1.0, return_diagnostics=True)
        # Locate the non-manifold (0, 1) edge in the returned set.
        rows = [tuple(sorted((int(a), int(b)))) for a, b in edges]
        nm = rows.index((0, 1))
        self.assertTrue(math.isnan(float(angles[nm])))
        self.assertTrue(bool(np.all(np.isnan(normals[nm]))))
        self.assertTrue(math.isnan(float(area_sums[nm])))

    def test_diagnostics_flat_quad_zero_angle(self):
        mesh = _flat_quad_mesh()
        edges, angles, normals, area_sums = mesh._filter_edges_by_dihedral_angle(-1.0, return_diagnostics=True)
        # The internal diagonal (0, 2) is shared by exactly two coplanar triangles
        # whose normals are both +Z, so the dihedral angle is 0 and the average
        # normal is +Z. Boundary edges remain NaN.
        rows = [tuple(sorted((int(a), int(b)))) for a, b in edges]
        diag = rows.index((0, 2))
        self.assertAlmostEqual(float(angles[diag]), 0.0, places=5)
        np.testing.assert_allclose(normals[diag], [0.0, 0.0, 1.0], atol=1e-5)
        # Two right tris of area 0.5 -> sum 1.0.
        self.assertAlmostEqual(float(area_sums[diag]), 1.0, places=5)
        boundary_mask = np.array([row != (0, 2) for row in rows])
        self.assertTrue(bool(np.all(np.isnan(angles[boundary_mask]))))
        self.assertTrue(bool(np.all(np.isnan(area_sums[boundary_mask]))))

    def test_diagnostics_zero_avg_normal_for_near_antiparallel_faces(self):
        mesh = _near_antiparallel_pair_mesh()
        edges, _angles, normals, area_sums = mesh._filter_edges_by_dihedral_angle(-1.0, return_diagnostics=True)
        rows = [tuple(sorted((int(a), int(b)))) for a, b in edges]
        shared = rows.index((0, 1))

        np.testing.assert_allclose(normals[shared], [0.0, 0.0, 0.0], atol=0.0)
        self.assertTrue(math.isfinite(float(area_sums[shared])))

    def test_filter_preserves_edges_subset_and_order(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        full_rows = [tuple(row) for row in mesh.edges.tolist()]
        full_index = {row: i for i, row in enumerate(full_rows)}

        filtered_rows = [tuple(row) for row in mesh._filter_edges_by_dihedral_angle(math.radians(1.0)).tolist()]
        # Subset.
        for row in filtered_rows:
            self.assertIn(row, full_index)
        # First-occurrence order preserved.
        positions = [full_index[row] for row in filtered_rows]
        self.assertEqual(positions, sorted(positions))


class TestModelBuilderEdgeAngleThreshold(unittest.TestCase):
    def test_finalize_uses_full_edges_without_build_sdf(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)

        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_mesh(body=body, mesh=mesh)
        model = builder.finalize()
        ranges = model.shape_edge_range.numpy()
        # No build_sdf() -> builder packs all 18 unique cube edges.
        self.assertEqual(int(ranges[0][1]), 18)
        self.assertEqual(int(model.mesh_edge_indices.shape[0]), 18)

    def test_finalize_shares_edges_across_shapes_referencing_same_mesh(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        builder = newton.ModelBuilder()
        body_a = builder.add_body()
        body_b = builder.add_body()
        builder.add_shape_mesh(body=body_a, mesh=mesh)
        builder.add_shape_mesh(body=body_b, mesh=mesh)
        model = builder.finalize()

        ranges = model.shape_edge_range.numpy()
        # Two mesh shapes, both referencing the same Mesh -> identical (start, count) slice.
        mesh_ranges = [tuple(int(x) for x in r) for r in ranges if int(r[1]) > 0]
        self.assertEqual(len(mesh_ranges), 2)
        self.assertEqual(mesh_ranges[0], mesh_ranges[1])
        # Packed array stores only one copy.
        self.assertEqual(int(model.mesh_edge_indices.shape[0]), mesh_ranges[0][1])


def _open_top_box_mesh() -> newton.Mesh:
    """Cube with the top face removed -> 4 boundary edges along the open rim."""
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )
    tris = np.array(
        [
            0, 2, 1, 0, 3, 2,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ],
        dtype=np.int32,
    )  # fmt: skip
    return newton.Mesh(verts, tris, compute_inertia=False)


class TestBuildCollisionEdges(unittest.TestCase):
    """Tests for Mesh._build_collision_edges (the edge-simplification half of
    Mesh.build_sdf), exercised directly so we don't pay for the SDF cook."""

    def _build(self, mesh: newton.Mesh, **kwargs) -> np.ndarray:
        # Mirror the validation/build split that ``Mesh.build_sdf`` now
        # performs: ``_validate_collision_edge_options`` resolves the
        # half-extents (and rejects invalid combinations) before the
        # downstream ``_build_collision_edges`` call would otherwise pay
        # for the dihedral pass.
        validation_keys = (
            "half_normal_abs",
            "half_normal_rel",
            "half_lateral_abs",
            "half_lateral_rel",
        )
        validation_defaults = dict.fromkeys(validation_keys)
        validation_kwargs = {k: kwargs.pop(k, None) for k in validation_keys}
        validation_defaults.update(validation_kwargs)
        build_defaults = {
            "lower_angle_threshold_rad": math.radians(0.1),
            "upper_angle_threshold_rad": math.radians(10.0),
            "enable_box_absorption": False,
        }
        build_defaults.update(kwargs)
        half_normal, half_lateral = mesh._validate_collision_edge_options(
            lower_angle_threshold_rad=build_defaults["lower_angle_threshold_rad"],
            enable_box_absorption=build_defaults["enable_box_absorption"],
            diagonal=mesh._aabb_diagonal(),
            **validation_defaults,
        )
        mesh._build_collision_edges(
            **build_defaults,
            half_normal=half_normal,
            half_lateral=half_lateral,
        )
        return mesh._collision_edges

    def test_abs_and_rel_together_raises(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        with self.assertRaisesRegex(ValueError, "edge_box_half_normal"):
            self._build(mesh, half_normal_abs=1.0, half_normal_rel=1e-3)
        with self.assertRaisesRegex(ValueError, "edge_box_half_lateral"):
            self._build(mesh, half_lateral_abs=1.0, half_lateral_rel=5e-3)

    def test_negative_value_raises(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._build(mesh, half_normal_abs=-1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._build(mesh, half_lateral_rel=-1.0)

    def test_boundary_edges_preserved_without_absorption(self):
        # Open-top box has 4 boundary edges that must survive the build_sdf
        # path; the fallback (no _collision_edges) keeps them too.
        mesh = _open_top_box_mesh()
        kept = self._build(mesh, lower_angle_threshold_rad=math.radians(0.1))
        fallback = mesh._filter_edges_by_dihedral_angle(math.radians(0.1))
        # The two paths must agree row-for-row when absorption is off.
        np.testing.assert_array_equal(kept, fallback)
        # Concretely: 12 edges (4 boundary along the open rim + 8 manifold
        # silhouette/diagonals; coplanar face diagonals get dropped).
        self.assertEqual(len(kept), 12)

    def test_absorption_removes_only_absorbed_manifold_edges(self):
        # Cube has 0-deg face diagonals (manifold, absorbable) and 90-deg
        # silhouette edges. Big extents -> diagonals absorbed; silhouettes
        # protected by the 10 deg upper threshold.
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        kept = self._build(
            mesh,
            lower_angle_threshold_rad=0.0,
            enable_box_absorption=True,
            half_normal_abs=2.0,
            half_lateral_abs=2.0,
        )
        # At most the 18 unique edges, strictly fewer than 18 (some diagonals removed).
        self.assertLess(len(kept), 18)
        self.assertGreaterEqual(len(kept), 12)

    def test_collision_edges_consumed_by_builder(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        # Seed _collision_edges with a hand-picked subset (e.g. 6 edges) to
        # simulate ``Mesh.build_sdf()`` having populated it.
        seeded = mesh.edges[:6].astype(np.int32)
        mesh._collision_edges = np.ascontiguousarray(seeded)

        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_mesh(body=body, mesh=mesh)
        model = builder.finalize()

        ranges = model.shape_edge_range.numpy()
        self.assertEqual(int(ranges[0][1]), len(seeded))
        np.testing.assert_array_equal(model.mesh_edge_indices.numpy(), seeded)

    def test_empty_mesh_produces_empty_collision_edges(self):
        mesh = newton.Mesh(np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32), compute_inertia=False)
        kept = self._build(mesh, enable_box_absorption=True)
        self.assertEqual(kept.shape, (0, 2))


class TestCollisionEdgesLifecycle(unittest.TestCase):
    """Lifecycle invariants for the ``_collision_edges`` cache attached by
    :meth:`Mesh.build_sdf`: clearing the SDF must also drop the cache,
    :meth:`Mesh.copy` must carry it alongside the SDF, and a failed
    ``build_sdf`` retry must not leave a stale SDF behind.
    """

    @staticmethod
    def _seed_collision_edges(mesh: newton.Mesh, count: int = 4) -> np.ndarray:
        """Populate ``_collision_edges`` without paying for an SDF cook."""
        seeded = np.ascontiguousarray(mesh.edges[:count].astype(np.int32))
        mesh._collision_edges = seeded
        return seeded

    def test_clear_sdf_drops_collision_edges_cache(self):
        # Otherwise ``ModelBuilder.finalize()`` would keep using the
        # SDF-tuned subset for a mesh that no longer has an SDF.
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        mesh.sdf = object()  # placeholder, only the lifecycle matters here
        self._seed_collision_edges(mesh)

        mesh.clear_sdf()

        self.assertIsNone(mesh.sdf)
        self.assertIsNone(mesh._collision_edges)

    def test_copy_carries_collision_edges_with_sdf(self):
        # A copy of an SDF-backed mesh must reuse the simplified contact
        # edges; otherwise it silently falls back to the full edge set and
        # produces different contact counts than the original.
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        mesh.sdf = object()
        seeded = self._seed_collision_edges(mesh)

        copy = mesh.copy()

        self.assertIs(copy.sdf, mesh.sdf)
        self.assertIsNotNone(copy._collision_edges)
        np.testing.assert_array_equal(copy._collision_edges, seeded)
        # The cache must be an independent buffer so mutating one mesh's
        # edges does not bleed into the other.
        self.assertIsNot(copy._collision_edges, mesh._collision_edges)

    @unittest.skipUnless(_cuda_available, "Requires CUDA device")
    def test_build_sdf_rolls_back_sdf_on_edge_option_failure(self):
        # Negative ``edge_lower_angle_threshold_rad`` combined with box
        # absorption is rejected by the edge-option validation that runs
        # before the SDF cook. The mesh must remain SDF-free so a
        # corrected retry doesn't trip the "Mesh already has an SDF"
        # guard, and the cache it would have populated must stay empty.
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        with self.assertRaises(ValueError):
            mesh.build_sdf(
                max_resolution=8,
                edge_lower_angle_threshold_rad=-1.0,
                edge_box_absorption=True,
            )

        self.assertIsNone(mesh.sdf)
        self.assertIsNone(mesh._collision_edges)

        # Sanity check: a corrected call now succeeds without first
        # requiring an explicit ``clear_sdf()``.
        mesh.build_sdf(max_resolution=8, edge_lower_angle_threshold_rad=0.0)
        self.assertIsNotNone(mesh.sdf)

    def test_copy_with_topology_override_drops_collision_edges(self):
        # ``_collision_edges`` is indexed against the original vertex
        # array, so a geometry-replacing copy must not carry it forward
        # — otherwise ``ModelBuilder.finalize()`` could feed stale or
        # out-of-range indices into contact generation.
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        mesh.sdf = object()
        self._seed_collision_edges(mesh)

        # New topology (a single triangle) -- old cached edges are bogus.
        new_verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        new_inds = np.array([0, 1, 2], dtype=np.int32)
        copy_verts = mesh.copy(vertices=new_verts)
        self.assertIsNone(copy_verts._collision_edges)
        self.assertIsNone(copy_verts.sdf)

        copy_inds = mesh.copy(indices=new_inds)
        self.assertIsNone(copy_inds._collision_edges)
        self.assertIsNone(copy_inds.sdf)

        copy_both = mesh.copy(vertices=new_verts, indices=new_inds)
        self.assertIsNone(copy_both._collision_edges)
        self.assertIsNone(copy_both.sdf)


if __name__ == "__main__":
    unittest.main()
