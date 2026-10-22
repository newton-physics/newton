# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for deformable visual meshes: binding kinds, ownership, validation, skinning."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import assert_np_equal


def _add_cloth(builder):
    builder.add_cloth_grid(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=3,
        dim_y=3,
        cell_x=0.5,
        cell_y=0.5,
        mass=0.1,
    )


def _add_soft(builder):
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.5,
        cell_y=0.5,
        cell_z=0.5,
        density=100.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=0.0,
    )


_QUAD = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)


class TestDeformableVisualMeshBindings(unittest.TestCase):
    """Core binding correctness for every kind."""

    def test_outside_tet_vertex_clamps_within_owning_range(self):
        """Log fallback binding and keep normalized weights inside the owning range."""
        builder = newton.ModelBuilder()
        _add_soft(builder)
        verts = np.array([[5.0, 5.0, 5.0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.7, 0.7, 0.7]], dtype=np.float32)
        with self.assertLogs("newton", level="WARNING") as logs:
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet", tet_range=(0, builder.tet_count))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("1 of 4 visual vertices fell outside", logs.records[0].getMessage())
        self.assertIn("clamped to the nearest owning tetrahedron", logs.records[0].getMessage())
        model = builder.finalize()
        rm = model.deformable_visual_meshes[0]
        parents = rm.parent.numpy()
        self.assertTrue((parents >= 0).all() and (parents < model.tet_count).all())
        assert_np_equal(rm.weights.numpy().sum(axis=1), np.ones(len(verts), dtype=np.float32), tol=1.0e-5)

    def test_two_soft_bodies_never_cross_bind(self):
        """With two overlapping-range soft bodies, a mesh owned by the second
        never binds into the first body's tets even for outside vertices."""
        builder = newton.ModelBuilder()
        _add_soft(builder)
        first_tet_end = builder.tet_count
        _add_soft(builder)  # identical grid at the same location
        # This vertex lies inside both grids; ownership must pin it to body 2.
        verts = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.7, 0.2, 0.4], [0.3, 0.6, 0.6]], dtype=np.float32)
        builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet", tet_range=(first_tet_end, builder.tet_count))
        model = builder.finalize()
        parents = model.deformable_visual_meshes[0].parent.numpy()
        self.assertTrue((parents >= first_tet_end).all())

    def test_precomputed_embedding_skips_search(self):
        """Precomputed parent/weights are accepted, validated, and used as-is."""
        builder = newton.ModelBuilder()
        _add_soft(builder)
        verts = np.zeros((4, 3), dtype=np.float32)
        parent = np.zeros(4, dtype=np.int32)
        weights = np.full((4, 4), 0.25, dtype=np.float32)
        builder.add_deformable_visual_mesh(
            verts, _QUAD, kind="tet", tet_range=(0, builder.tet_count), parent=parent, weights=weights
        )
        parent[:] = 1
        weights[:] = 0.0
        model = builder.finalize()
        rm = model.deformable_visual_meshes[0]
        assert_np_equal(rm.parent.numpy(), np.zeros(4, dtype=np.int32))
        assert_np_equal(rm.weights.numpy(), np.full((4, 4), 0.25, dtype=np.float32), tol=0.0)


class TestDeformableVisualMeshValidation(unittest.TestCase):
    """Input validation and ownership errors."""

    def _soft_builder(self):
        builder = newton.ModelBuilder()
        _add_soft(builder)
        return builder

    def test_rejects_conflicting_kind_arguments(self):
        """Reject drivers that do not belong to the requested binding kind."""
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "do not apply to this kind"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet", tet_range=(0, 1), particles=np.arange(4))
        with self.assertRaisesRegex(ValueError, "do not apply to this kind"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="particle", particles=np.arange(4), bodies=[0])

    def test_rejects_unknown_kind_and_missing_mode_argument(self):
        """Each binding kind requires its own driving elements."""
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "unknown kind"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="auto")
        with self.assertRaisesRegex(ValueError, "requires tet_range"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet")
        with self.assertRaisesRegex(ValueError, "requires particles"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="particle")
        with self.assertRaisesRegex(ValueError, "requires bodies"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="body")

    def test_rejects_malformed_geometry(self):
        """Reject invalid connectivity, empty geometry, and mismatched UVs."""
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        for indices in ([-0.2, 1.9, 2.0], np.array([2**32, 1, 2], dtype=np.int64)):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "integer indices"):
                builder.add_deformable_visual_mesh(verts, indices, kind="particle", particles=np.arange(4))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            builder.add_deformable_visual_mesh(np.zeros((0, 3), np.float32), _QUAD, kind="tet", tet_range=(0, 1))
        with self.assertRaisesRegex(ValueError, "multiple of 3"):
            builder.add_deformable_visual_mesh(verts, [0, 1], kind="tet", tet_range=(0, 1))
        with self.assertRaisesRegex(ValueError, "outside the vertex array"):
            builder.add_deformable_visual_mesh(verts, [0, 1, 9], kind="tet", tet_range=(0, 1))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            bad = verts.copy()
            bad[0, 0] = np.nan
            builder.add_deformable_visual_mesh(bad, _QUAD, kind="tet", tet_range=(0, 1))
        with self.assertRaisesRegex(ValueError, "uvs length"):
            builder.add_deformable_visual_mesh(
                verts, _QUAD, kind="tet", tet_range=(0, 1), uvs=np.zeros((2, 2), np.float32)
            )
        with self.assertRaisesRegex(ValueError, "texture requires uvs"):
            builder.add_deformable_visual_mesh(
                verts, _QUAD, kind="tet", tet_range=(0, 1), texture=np.zeros((2, 2, 3), np.uint8)
            )

    def test_rejects_out_of_range_drivers(self):
        """Reject driver indices outside the builder's simulation arrays."""
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        builder.add_body(mass=1.0)
        for kind, options in (
            ("particle", {"particles": [-0.2, 1, 2, 3]}),
            ("body", {"bodies": [-0.2]}),
            ("triangle", {"tri_range": (-0.2, 1)}),
            ("tet", {"tet_range": (0, 1.9)}),
            ("tet", {"tet_range": (0, 1), "parent": [-0.2] * 4, "weights": np.full((4, 4), 0.25)}),
        ):
            with self.subTest(kind=kind, options=options), self.assertRaisesRegex(ValueError, "integer indices"):
                builder.add_deformable_visual_mesh(verts, _QUAD, kind=kind, **options)
        with self.assertRaisesRegex(ValueError, "not a valid non-empty range"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet", tet_range=(0, builder.tet_count + 1))
        with self.assertRaisesRegex(ValueError, "not a valid non-empty range"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet", tet_range=(3, 3))
        with self.assertRaisesRegex(ValueError, "outside the current builder"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="particle", particles=[0, 1, 2, 10**6])
        with self.assertRaisesRegex(ValueError, "outside the current builder"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="body", bodies=[7])
        with self.assertRaisesRegex(ValueError, "requires at least one body"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="body", bodies=[])

    def test_rejects_malformed_precomputed_embedding(self):
        """Precomputed bindings need valid parents and one weight row per vertex."""
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        good_w = np.full((4, 4), 0.25, dtype=np.float32)
        for weights in (np.full((4, 3), 0.25), np.full(12, 0.25)):
            with self.subTest(shape=weights.shape), self.assertRaisesRegex(ValueError, "weights.*shape"):
                builder.add_deformable_visual_mesh(
                    verts[:3], [0, 1, 2], kind="tet", tet_range=(0, 1), parent=[0, 0, 0], weights=weights
                )
        with self.assertRaisesRegex(ValueError, "passed together"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet", tet_range=(0, 1), parent=np.zeros(4, np.int32))
        with self.assertRaisesRegex(ValueError, "within the owning range"):
            builder.add_deformable_visual_mesh(
                verts, _QUAD, kind="tet", tet_range=(0, 1), parent=np.full(4, 3, np.int32), weights=good_w
            )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            bad_w = good_w.copy()
            bad_w[0, 0] = -1.0
            builder.add_deformable_visual_mesh(
                verts, _QUAD, kind="tet", tet_range=(0, 1), parent=np.zeros(4, np.int32), weights=bad_w
            )

    def test_rejects_nonunit_precomputed_weights(self):
        """Reject weights that would scale translations or collapse visual points."""
        builder = self._soft_builder()
        for kind, arity in (("triangle", 3), ("tet", 4)):
            for total in (0.0, 0.5, 2.0):
                with self.subTest(kind=kind, total=total):
                    options = {"tri_range" if kind == "triangle" else "tet_range": (0, 1)}
                    with self.assertRaisesRegex(ValueError, "sum to one"):
                        builder.add_deformable_visual_mesh(
                            np.zeros((4, 3), dtype=np.float32),
                            _QUAD,
                            kind=kind,
                            parent=np.zeros(4, dtype=np.int32),
                            weights=np.full((4, arity), total / arity, dtype=np.float32),
                            **options,
                        )
                    self.assertEqual(len(builder._deformable_visual_meshes), 0)

    def test_mixed_world_drivers_rejected_at_finalize(self):
        """Drivers spanning two replicated worlds raise when the model is built."""
        source = newton.ModelBuilder()
        _add_cloth(source)
        _add_soft(source)
        source.add_body()
        vertices = np.array([[0, 0, 0], [10, 0, 0], [10, 1, 0]], dtype=np.float32)
        modes = (
            ("particle", {"particles": [0, source.particle_count, source.particle_count + 1]}),
            (
                "triangle",
                {
                    "tri_range": (0, 2 * source.tri_count),
                    "parent": [0, source.tri_count, source.tri_count],
                    "weights": [[1, 0, 0]] * 3,
                },
            ),
            (
                "tet",
                {
                    "tet_range": (0, 2 * source.tet_count),
                    "parent": [0, source.tet_count, source.tet_count],
                    "weights": [[1, 0, 0, 0]] * 3,
                },
            ),
            ("body", {"bodies": [0, 1]}),
        )
        for kind, options in modes:
            with self.subTest(kind=kind):
                builder = newton.ModelBuilder()
                builder.replicate(
                    source, 2, xforms=[wp.transform_identity(), wp.transform((10, 0, 0), wp.quat_identity())]
                )
                builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind=kind, **options)
                with self.assertRaisesRegex(ValueError, "single world"):
                    builder.finalize(device="cpu")


if __name__ == "__main__":
    unittest.main()
