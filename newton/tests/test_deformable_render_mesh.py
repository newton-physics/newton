# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for deformable render meshes: binding kinds, ownership, validation, skinning."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.sim.deformable_render import compute_render_mesh_normals, skin_render_mesh
from newton.tests._usd_deformable_test_utils import _add_cable_curve, _add_cloth_mesh, _deformable_stage
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal
from newton.viewer import ViewerNull


class _MeshProbe(ViewerNull):
    """Captures every ``log_mesh`` call keyed by object name.

    Deliberately overrides ``log_mesh`` with the legacy signature (no new
    keywords) to pin that render-mesh drawing works with pre-existing viewer
    subclasses.
    """

    def __init__(self):
        super().__init__(num_frames=1)
        self.calls: dict[str, dict] = {}

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        color=None,
        roughness=None,
        metallic=None,
    ):
        self.calls[name] = {
            "points": None if points is None else points.numpy(),
            "normals": None if normals is None else normals.numpy(),
            "uvs": None if uvs is None else uvs.numpy(),
            "texture": texture,
            "hidden": hidden,
        }

    def _frame(self, state, t=0.0):
        self.begin_frame(t)
        self.log_state(state)
        self.end_frame()


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


def _skin(model, mesh, state):
    out = wp.empty(mesh.vertex_count, dtype=wp.vec3, device=model.device)
    skin_render_mesh(mesh, state, model, out)
    return out.numpy()


class TestDeformableRenderMeshBindings(unittest.TestCase):
    """Core binding correctness for every kind."""

    def test_particle_kind_follows_particles(self):
        """A particle-bound render mesh with a 1:1 map equals particle_q, and the
        model output carries the invariant index, kind, world, and UVs."""
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        n = builder.particle_count
        verts = np.array(builder.particle_q, dtype=np.float32)
        uvs = np.linspace(0.0, 1.0, n * 2, dtype=np.float32).reshape(n, 2)
        index = builder.add_deformable_render_mesh(
            verts, _QUAD, kind="particle", particles=np.arange(n, dtype=np.int32), uvs=uvs, label="skin"
        )
        model = builder.finalize()

        self.assertEqual(model.deformable_render_mesh_count, 1)
        rm = model.deformable_render_meshes[index]
        self.assertEqual(rm.kind, newton.DeformableRenderMesh.Kind.PARTICLE)
        self.assertEqual(rm.index, index)
        self.assertEqual(rm.label, "skin")
        self.assertIsNone(rm.body_path)
        assert_np_equal(rm.uvs.numpy(), uvs, tol=1.0e-6)

        state = model.state()
        moved = state.particle_q.numpy()
        moved[:, 2] += 1.5
        state.particle_q = wp.array(moved, dtype=wp.vec3)
        assert_np_equal(_skin(model, rm, state), moved, tol=1.0e-5)

    def test_triangle_embedding_follows_surface(self):
        """An independently discretized vertex projects onto its closest owning
        triangle (normal offset dropped) and follows the surface; the weights
        form a partition of unity."""
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        # A vertex hovering above the cloth plane and one inside a triangle.
        verts = np.array([[0.25, 0.1, 0.3], [0.6, 0.4, 0.0], [1.0, 1.0, 0.0], [0.1, 0.9, 0.0]], dtype=np.float32)
        builder.add_deformable_render_mesh(verts, _QUAD, kind="triangle", tri_range=(0, builder.tri_count))
        model = builder.finalize()
        rm = model.deformable_render_meshes[0]

        self.assertEqual(rm.kind, newton.DeformableRenderMesh.Kind.TRIANGLE)
        weights = rm.weights.numpy()
        assert_np_equal(weights.sum(axis=1), np.ones(len(verts), dtype=np.float32), tol=1.0e-5)
        parents = rm.parent.numpy()
        self.assertTrue((parents >= 0).all() and (parents < model.tri_count).all())

        # At rest the skinned vertex is the projection: z collapses onto the plane.
        state = model.state()
        skinned = _skin(model, rm, state)
        self.assertAlmostEqual(float(skinned[0, 2]), 0.0, places=5)

        # A rigid translation of the cloth carries the projected vertices with it.
        shift = np.array([0.5, -1.0, 2.0], dtype=np.float32)
        state.particle_q = wp.array(state.particle_q.numpy() + shift, dtype=wp.vec3)
        assert_np_equal(_skin(model, rm, state), skinned + shift, tol=1.0e-4)

    def test_tet_embedding_partition_of_unity_and_rigid_motion(self):
        """Tet barycentric weights sum to one; embedded vertices follow a rigid
        translation and rotation of the soft body exactly."""
        builder = newton.ModelBuilder()
        _add_soft(builder)
        verts = np.array(
            [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5], [0.1, 0.9, 0.4]],
            dtype=np.float32,
        )
        builder.add_deformable_render_mesh(verts, _QUAD, kind="tet", tet_range=(0, builder.tet_count))
        model = builder.finalize()
        rm = model.deformable_render_meshes[0]

        self.assertEqual(rm.kind, newton.DeformableRenderMesh.Kind.TET)
        weights = rm.weights.numpy()
        assert_np_equal(weights.sum(axis=1), np.ones(len(verts), dtype=np.float32), tol=1.0e-5)

        state = model.state()
        shift = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        state.particle_q = wp.array(state.particle_q.numpy() + shift, dtype=wp.vec3)
        assert_np_equal(_skin(model, rm, state), verts + shift, tol=1.0e-4)

        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        state2 = model.state()
        state2.particle_q = wp.array((model.state().particle_q.numpy() @ rot.T).astype(np.float32), dtype=wp.vec3)
        assert_np_equal(_skin(model, rm, state2), verts @ rot.T, tol=1.0e-4)

    def test_body_binding_follows_bodies(self):
        """A body-bound render mesh follows its nearest body's current pose."""
        builder = newton.ModelBuilder()
        b0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        b1 = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(b0, radius=0.1)
        builder.add_shape_sphere(b1, radius=0.1)
        verts = np.array(
            [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [1.0, 0.1, 0.0], [0.9, 0.0, 0.0]],
            dtype=np.float32,
        )
        builder.add_deformable_render_mesh(verts, _QUAD, kind="body", bodies=[b0, b1])
        model = builder.finalize()
        rm = model.deformable_render_meshes[0]

        self.assertEqual(rm.kind, newton.DeformableRenderMesh.Kind.BODY)
        parents = rm.parent.numpy()
        self.assertEqual(parents[0], b0)
        self.assertEqual(parents[2], b1)

        # Translate body 1 only; its two vertices move, body 0's stay.
        state = model.state()
        body_q = state.body_q.numpy()
        body_q[b1, :3] += (0.0, 0.0, 2.0)
        state.body_q = wp.array(body_q, dtype=wp.transform)
        skinned = _skin(model, rm, state)
        assert_np_equal(skinned[:2], verts[:2], tol=1.0e-5)
        assert_np_equal(skinned[2:], verts[2:] + np.array([0.0, 0.0, 2.0], dtype=np.float32), tol=1.0e-5)

    def test_outside_tet_vertex_clamps_within_owning_range(self):
        """A vertex outside every owning tet warns, clamps to the nearest owning
        tet, and still produces a valid partition-of-unity weight row."""
        builder = newton.ModelBuilder()
        _add_soft(builder)
        verts = np.array([[5.0, 5.0, 5.0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.7, 0.7, 0.7]], dtype=np.float32)
        with self.assertWarnsRegex(UserWarning, "clamped to the nearest owning tetrahedron"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="tet", tet_range=(0, builder.tet_count))
        model = builder.finalize()
        rm = model.deformable_render_meshes[0]
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
        builder.add_deformable_render_mesh(verts, _QUAD, kind="tet", tet_range=(first_tet_end, builder.tet_count))
        model = builder.finalize()
        parents = model.deformable_render_meshes[0].parent.numpy()
        self.assertTrue((parents >= first_tet_end).all())

    def test_precomputed_embedding_skips_search(self):
        """Precomputed parent/weights are accepted, validated, and used as-is."""
        builder = newton.ModelBuilder()
        _add_soft(builder)
        verts = np.zeros((4, 3), dtype=np.float32)
        parent = np.zeros(4, dtype=np.int32)
        weights = np.full((4, 4), 0.25, dtype=np.float32)
        builder.add_deformable_render_mesh(
            verts, _QUAD, kind="tet", tet_range=(0, builder.tet_count), parent=parent, weights=weights
        )
        model = builder.finalize()
        rm = model.deformable_render_meshes[0]
        assert_np_equal(rm.parent.numpy(), parent)
        assert_np_equal(rm.weights.numpy(), weights, tol=0.0)

    def test_add_builder_rebases_parents_by_kind(self):
        """add_builder shifts particle, triangle, tet, and body parents into the
        merged index space so skinning stays correct."""
        sub = newton.ModelBuilder()
        _add_cloth(sub)
        n = sub.particle_count
        cloth_verts = np.array(sub.particle_q, dtype=np.float32)
        sub.add_deformable_render_mesh(
            cloth_verts, _QUAD, kind="particle", particles=np.arange(n, dtype=np.int32), label="cloth"
        )
        sub.add_deformable_render_mesh(
            cloth_verts[:4], _QUAD, kind="triangle", tri_range=(0, sub.tri_count), label="tri"
        )
        _add_soft(sub)
        tet_verts = np.array([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75], [0.4, 0.6, 0.5]], np.float32)
        sub.add_deformable_render_mesh(tet_verts, _QUAD, kind="tet", tet_range=(0, sub.tet_count), label="tet")
        b = sub.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
        sub.add_shape_sphere(b, radius=0.1)
        body_verts = np.array([[2.0, 0.1, 0.0], [2.1, 0.0, 0.0], [2.0, 0.0, 0.1], [1.9, 0.0, 0.0]], np.float32)
        sub.add_deformable_render_mesh(body_verts, _QUAD, kind="body", bodies=[b], label="body")

        main = newton.ModelBuilder()
        _add_soft(main)  # pre-existing content shifts every index space
        b0 = main.add_body(xform=wp.transform(wp.vec3(-5.0, 0.0, 0.0), wp.quat_identity()))
        main.add_shape_sphere(b0, radius=0.1)
        main.add_builder(sub)
        model = main.finalize()

        self.assertEqual(model.deformable_render_mesh_count, 4)
        by_label = {rm.label: rm for rm in model.deformable_render_meshes}
        state = model.state()

        # Particle and tet meshes reproduce their bind pose at rest.
        assert_np_equal(_skin(model, by_label["cloth"], state), cloth_verts, tol=1.0e-4)
        assert_np_equal(_skin(model, by_label["tet"], state), tet_verts, tol=1.0e-4)
        assert_np_equal(_skin(model, by_label["body"], state), body_verts, tol=1.0e-4)
        # The triangle mesh reproduces its on-surface projection at rest.
        tri_rest = _skin(model, by_label["tri"], state)
        assert_np_equal(tri_rest, cloth_verts[:4], tol=1.0e-4)

    def test_normals_recompute_from_current_positions(self):
        """compute_render_mesh_normals yields unit normals for the deformed quad."""
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        builder.add_deformable_render_mesh(verts, _QUAD, kind="particle", particles=np.arange(4, dtype=np.int32))
        model = builder.finalize()
        rm = model.deformable_render_meshes[0]
        points = wp.array(verts, dtype=wp.vec3, device=model.device)
        normals = wp.empty(4, dtype=wp.vec3, device=model.device)
        compute_render_mesh_normals(points, rm.indices, normals)
        n = normals.numpy()
        assert_np_equal(np.linalg.norm(n, axis=1), np.ones(4, dtype=np.float32), tol=1.0e-5)
        assert_np_equal(np.abs(n[:, 2]), np.ones(4, dtype=np.float32), tol=1.0e-5)


class TestDeformableRenderMeshValidation(unittest.TestCase):
    """Input validation and ownership errors."""

    def _soft_builder(self):
        builder = newton.ModelBuilder()
        _add_soft(builder)
        return builder

    def test_rejects_conflicting_kind_arguments(self):
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "do not apply to this kind"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="tet", tet_range=(0, 1), particles=np.arange(4))
        with self.assertRaisesRegex(ValueError, "do not apply to this kind"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="particle", particles=np.arange(4), bodies=[0])

    def test_rejects_unknown_kind_and_missing_mode_argument(self):
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "unknown kind"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="auto")
        with self.assertRaisesRegex(ValueError, "requires tet_range"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="tet")
        with self.assertRaisesRegex(ValueError, "requires particles"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="particle")
        with self.assertRaisesRegex(ValueError, "requires bodies"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="body")

    def test_rejects_malformed_geometry(self):
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            builder.add_deformable_render_mesh(np.zeros((0, 3), np.float32), _QUAD, kind="tet", tet_range=(0, 1))
        with self.assertRaisesRegex(ValueError, "multiple of 3"):
            builder.add_deformable_render_mesh(verts, [0, 1], kind="tet", tet_range=(0, 1))
        with self.assertRaisesRegex(ValueError, "outside the vertex array"):
            builder.add_deformable_render_mesh(verts, [0, 1, 9], kind="tet", tet_range=(0, 1))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            bad = verts.copy()
            bad[0, 0] = np.nan
            builder.add_deformable_render_mesh(bad, _QUAD, kind="tet", tet_range=(0, 1))
        with self.assertRaisesRegex(ValueError, "uvs length"):
            builder.add_deformable_render_mesh(
                verts, _QUAD, kind="tet", tet_range=(0, 1), uvs=np.zeros((2, 2), np.float32)
            )

    def test_rejects_out_of_range_drivers(self):
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "not a valid non-empty range"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="tet", tet_range=(0, builder.tet_count + 1))
        with self.assertRaisesRegex(ValueError, "not a valid non-empty range"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="tet", tet_range=(3, 3))
        with self.assertRaisesRegex(ValueError, "outside the current builder"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="particle", particles=[0, 1, 2, 10**6])
        with self.assertRaisesRegex(ValueError, "outside the current builder"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="body", bodies=[7])
        with self.assertRaisesRegex(ValueError, "requires at least one body"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="body", bodies=[])

    def test_rejects_malformed_precomputed_embedding(self):
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        good_w = np.full((4, 4), 0.25, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "passed together"):
            builder.add_deformable_render_mesh(verts, _QUAD, kind="tet", tet_range=(0, 1), parent=np.zeros(4, np.int32))
        with self.assertRaisesRegex(ValueError, "within the owning range"):
            builder.add_deformable_render_mesh(
                verts, _QUAD, kind="tet", tet_range=(0, 1), parent=np.full(4, 3, np.int32), weights=good_w
            )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            bad_w = good_w.copy()
            bad_w[0, 0] = -1.0
            builder.add_deformable_render_mesh(
                verts, _QUAD, kind="tet", tet_range=(0, 1), parent=np.zeros(4, np.int32), weights=bad_w
            )

    def test_mixed_world_drivers_rejected_at_finalize(self):
        """Drivers spanning two replicated worlds raise when the model is built."""
        source = newton.ModelBuilder()
        _add_cloth(source)
        n = source.particle_count
        builder = newton.ModelBuilder()
        builder.replicate(source, 2)
        verts = np.zeros((4, 3), dtype=np.float32)
        # One driver from world 0, three from world 1.
        builder.add_deformable_render_mesh(
            verts, _QUAD, kind="particle", particles=[0, n, n + 1, n + 2], label="crossworld"
        )
        with self.assertRaisesRegex(ValueError, "single world"):
            builder.finalize()

    def test_single_world_replicated_mesh_keeps_world(self):
        """A mesh added before replicate() is duplicated per world with correct
        world ownership and shifted drivers."""
        source = newton.ModelBuilder()
        _add_cloth(source)
        n = source.particle_count
        verts = np.array(source.particle_q, dtype=np.float32)
        source.add_deformable_render_mesh(
            verts, _QUAD, kind="particle", particles=np.arange(n, dtype=np.int32), label="skin"
        )
        builder = newton.ModelBuilder()
        builder.replicate(source, 2)
        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 2)
        worlds = sorted(rm.world for rm in model.deformable_render_meshes)
        self.assertEqual(worlds, [0, 1])


class TestDeformableRenderMeshViewer(unittest.TestCase):
    """ViewerBase drawing: naming, visibility, world offsets, legacy overrides."""

    @staticmethod
    def _cloth_skin_builder():
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        n = builder.particle_count
        verts = np.array(builder.particle_q, dtype=np.float32)
        uvs = np.linspace(0.0, 1.0, n * 2, dtype=np.float32).reshape(n, 2)
        # A non-degenerate grid quad (0-1-5-4) so face normals are well defined.
        quad = np.array([0, 1, 5, 0, 5, 4], dtype=np.int32)
        builder.add_deformable_render_mesh(
            verts, quad, kind="particle", particles=np.arange(n, dtype=np.int32), uvs=uvs, label="skin"
        )
        return builder, uvs

    def test_viewer_draws_skinned_mesh_with_invariant_name(self):
        """The mesh is drawn under its invariant index name with skinned points,
        recomputed normals, and preserved UVs."""
        builder, uvs = self._cloth_skin_builder()
        model = builder.finalize()
        viewer = _MeshProbe()
        viewer.set_model(model)

        state = model.state()
        moved = state.particle_q.numpy()
        moved[:, 2] += 1.5
        state.particle_q = wp.array(moved, dtype=wp.vec3)
        viewer._frame(state)

        call = viewer.calls["/model/render_meshes/0_skin"]
        self.assertFalse(call["hidden"])
        assert_np_equal(call["points"], moved, tol=1.0e-5)
        assert_np_equal(call["uvs"], uvs, tol=1.0e-6)
        self.assertIsNotNone(call["normals"])
        # Unit normals on every vertex the triangles reference.
        referenced = np.linalg.norm(call["normals"][[0, 1, 4, 5]], axis=1)
        assert_np_equal(referenced, np.ones(4, dtype=np.float32), tol=1.0e-4)

    def test_sim_triangles_stay_visible_and_mesh_toggles(self):
        """Render meshes draw in addition to the simulation triangles; toggling
        show_render_mesh hides the mesh but keeps valid geometry registered."""
        builder, _uvs = self._cloth_skin_builder()
        model = builder.finalize()
        viewer = _MeshProbe()
        viewer.set_model(model)
        state = model.state()

        viewer._frame(state)
        self.assertFalse(viewer.calls["/model/triangles"]["hidden"])
        self.assertFalse(viewer.calls["/model/render_meshes/0_skin"]["hidden"])

        viewer.show_render_mesh = False
        viewer._frame(state, 1.0)
        call = viewer.calls["/model/render_meshes/0_skin"]
        self.assertTrue(call["hidden"])
        self.assertEqual(len(call["points"]), model.deformable_render_meshes[0].vertex_count)
        self.assertFalse(viewer.calls["/model/triangles"]["hidden"])

    def test_replicated_worlds_use_distinct_names_and_device_offsets(self):
        """Replicated meshes draw under distinct names, each offset by its own
        world offset."""
        source = newton.ModelBuilder()
        _add_cloth(source)
        n = source.particle_count
        verts = np.array(source.particle_q, dtype=np.float32)
        source.add_deformable_render_mesh(
            verts, _QUAD, kind="particle", particles=np.arange(n, dtype=np.int32), label="skin"
        )
        builder = newton.ModelBuilder()
        builder.replicate(source, 2)
        model = builder.finalize()

        viewer = _MeshProbe()
        viewer.set_model(model)
        viewer.set_world_offsets((4.0, 0.0, 0.0))
        viewer._frame(model.state())

        offsets = viewer.world_offsets.numpy()
        p0 = viewer.calls["/model/render_meshes/0_skin"]["points"]
        p1 = viewer.calls["/model/render_meshes/1_skin"]["points"]
        assert_np_equal(p0, verts + offsets[0], tol=1.0e-5)
        assert_np_equal(p1, verts + offsets[1], tol=1.0e-5)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestDeformableRenderMeshUSDImport(unittest.TestCase):
    """AOUSD graphics-geometry discovery: hierarchy, bind poses, ownership."""

    @staticmethod
    def _stage():
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        return stage

    @staticmethod
    def _add_volume_body(stage, path):
        """A deformable body root with a unit-tet simulation mesh child."""
        from pxr import UsdGeom

        body = UsdGeom.Xform.Define(stage, path)
        body.GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
        tet = UsdGeom.TetMesh.Define(stage, f"{path}/Sim")
        tet.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
        tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
        # Author collision like the importer fixtures: no gating warning under strict runs.
        tet.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        return body, tet

    @staticmethod
    def _add_graphics_mesh(stage, path, points=None):
        from pxr import UsdGeom

        gfx = UsdGeom.Mesh.Define(stage, path)
        pts = points if points is not None else [(0.2, 0.2, 0.2), (0.4, 0.2, 0.2), (0.2, 0.4, 0.2)]
        gfx.CreatePointsAttr([tuple(p) for p in pts])
        gfx.CreateFaceVertexCountsAttr([3])
        gfx.CreateFaceVertexIndicesAttr([0, 1, 2])
        return gfx

    def _import(self, stage, **kwargs):
        builder = newton.ModelBuilder()
        builder.add_usd(stage, **kwargs)
        return builder

    def test_untagged_mesh_under_volume_body_imports_and_skins(self):
        """An untagged graphics Mesh under a volume deformable body becomes a tet
        render mesh with USD ownership metadata, without any custom relationship
        and without return_deformable_results."""
        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        self._add_graphics_mesh(stage, "/World/Tire/Skin")

        builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 1)
        rm = model.deformable_render_meshes[0]
        self.assertEqual(rm.kind, newton.DeformableRenderMesh.Kind.TET)
        self.assertEqual(rm.body_path, "/World/Tire")
        self.assertEqual(rm.sim_path, "/World/Tire/Sim")
        self.assertEqual(rm.graphics_path, "/World/Tire/Skin")

        # Rigid translation of the soft body carries the skinned vertices exactly.
        state = model.state()
        rest = _skin(model, rm, state)
        shift = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        state.particle_q = wp.array(state.particle_q.numpy() + shift, dtype=wp.vec3)
        assert_np_equal(_skin(model, rm, state), rest + shift, tol=1.0e-5)

    def test_multiple_graphics_meshes_and_nested_bodies(self):
        """Multiple graphics meshes under one body all import; a nested deformable
        body's graphics are never assigned to the outer body."""
        stage = self._stage()
        self._add_volume_body(stage, "/World/Outer")
        self._add_graphics_mesh(stage, "/World/Outer/SkinA")
        self._add_graphics_mesh(stage, "/World/Outer/SkinB")
        self._add_volume_body(stage, "/World/Outer/Inner")
        self._add_graphics_mesh(stage, "/World/Outer/Inner/Skin")

        builder = self._import(stage)
        model = builder.finalize()
        by_path = {rm.graphics_path: rm for rm in model.deformable_render_meshes}
        self.assertEqual(set(by_path), {"/World/Outer/SkinA", "/World/Outer/SkinB", "/World/Outer/Inner/Skin"})
        self.assertEqual(by_path["/World/Outer/SkinA"].body_path, "/World/Outer")
        self.assertEqual(by_path["/World/Outer/Inner/Skin"].body_path, "/World/Outer/Inner")
        # The nested body's mesh embeds in the nested body's own tets.
        self.assertNotEqual(
            by_path["/World/Outer/SkinA"].parent.numpy()[0], by_path["/World/Outer/Inner/Skin"].parent.numpy()[0]
        )

    def test_nested_rigid_body_mesh_is_not_deformable_visual(self):
        """A mesh owned by a nested rigid body is not claimed by the outer deformable."""
        from pxr import UsdGeom, UsdPhysics

        stage = self._stage()
        self._add_volume_body(stage, "/World/Soft")
        rigid = UsdGeom.Xform.Define(stage, "/World/Soft/Rigid")
        UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
        self._add_graphics_mesh(stage, "/World/Soft/Rigid/Skin")

        builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 0)

    def test_collision_marked_geometry_is_not_visual(self):
        """A Mesh marked with a collision API under the body is not a render mesh."""
        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        collider = self._add_graphics_mesh(stage, "/World/Tire/Collider")
        collider.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")

        with self.assertWarnsRegex(UserWarning, "approximated by the simulation geometry"):
            builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 0)

    def test_bind_pose_is_honored(self):
        """PhysicsDeformablePoseAPI bindPose drives the embedding: a mesh whose
        default points are far away but whose bind pose lies inside the tet
        embeds without an outside-domain clamp warning."""
        from pxr import Sdf

        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        gfx = self._add_graphics_mesh(
            stage, "/World/Tire/Skin", points=[(9.0, 9.0, 9.0), (9.4, 9.0, 9.0), (9.0, 9.4, 9.0)]
        )
        gfx.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        gfx.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        gfx.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.2, 0.2, 0.2), (0.4, 0.2, 0.2), (0.2, 0.4, 0.2)]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder = self._import(stage)
        self.assertFalse(any("clamped" in str(w.message) for w in caught))
        model = builder.finalize()
        rm = model.deformable_render_meshes[0]
        # The bind-pose vertices (inside the tet) are the rest vertices.
        self.assertLess(float(rm.rest_vertices.numpy().max()), 1.0)

    def test_invalid_simulation_bind_pose_skips_visual_binding(self):
        """An authored simulation bind pose with the wrong count is not replaced silently."""
        from pxr import Sdf

        stage = self._stage()
        _, sim = self._add_volume_body(stage, "/World/Tire")
        self._add_graphics_mesh(stage, "/World/Tire/Skin")
        sim.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        )

        with self.assertWarnsRegex(UserWarning, "invalid_bind_pose_count"):
            builder = self._import(stage)

        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 0)

    def test_non_finite_visual_bind_pose_skips_only_that_visual(self):
        """A malformed visual is diagnosed without discarding a valid sibling visual."""
        from pxr import Sdf

        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        bad = self._add_graphics_mesh(stage, "/World/Tire/BadSkin")
        self._add_graphics_mesh(stage, "/World/Tire/GoodSkin")
        bad.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        bad.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        bad.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(float("nan"), 0.2, 0.2), (0.4, 0.2, 0.2), (0.2, 0.4, 0.2)]
        )

        with self.assertWarnsRegex(UserWarning, "non_finite_bind_point"):
            builder = self._import(stage)

        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 1)
        self.assertEqual(model.deformable_render_meshes[0].graphics_path, "/World/Tire/GoodSkin")

    def test_transforms_embed_in_common_frame(self):
        """Sim and graphics prims with different local transforms meet in the
        world frame: a graphics mesh authored in a shifted local frame lands on
        the simulation geometry."""
        from pxr import Gf, UsdGeom

        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        gfx = self._add_graphics_mesh(
            stage, "/World/Tire/Skin", points=[(-0.8, 0.2, 0.2), (-0.6, 0.2, 0.2), (-0.8, 0.4, 0.2)]
        )
        UsdGeom.Xformable(gfx).AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder = self._import(stage)
        self.assertFalse(any("clamped" in str(w.message) for w in caught))
        model = builder.finalize()
        rest = model.deformable_render_meshes[0].rest_vertices.numpy()
        assert_np_equal(rest[0], np.array([0.2, 0.2, 0.2], dtype=np.float32), tol=1.0e-6)

    def test_cloth_body_graphics_embed_in_triangles(self):
        """An untagged Mesh under a surface deformable body embeds into the
        cloth's owning triangle range."""
        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Body/Sim")
        from pxr import UsdGeom

        body = stage.GetPrimAtPath("/World/Body")
        if not body or not body.IsValid():
            body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        self._add_graphics_mesh(stage, "/World/Body/Skin", points=[(0.2, 0.2, 1.0), (0.6, 0.2, 1.0), (0.2, 0.6, 1.0)])
        del cloth

        builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 1)
        rm = model.deformable_render_meshes[0]
        self.assertEqual(rm.kind, newton.DeformableRenderMesh.Kind.TRIANGLE)
        self.assertEqual(rm.weights.numpy().shape[1], 3)

    def test_cable_body_graphics_bind_to_segment_bodies(self):
        """An untagged Mesh under a cable deformable body binds to the curve's
        imported segment bodies and follows them."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        _add_cable_curve(stage, "/World/Body/Sim", [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)])
        self._add_graphics_mesh(
            stage, "/World/Body/Skin", points=[(0.0, 0.02, 1.0), (0.15, 0.02, 1.0), (0.3, 0.02, 1.0)]
        )

        builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 1)
        rm = model.deformable_render_meshes[0]
        self.assertEqual(rm.kind, newton.DeformableRenderMesh.Kind.BODY)
        self.assertEqual(rm.body_path, "/World/Body")
        parents = rm.parent.numpy()
        self.assertTrue((parents >= 0).all() and (parents < model.body_count).all())

    def test_uvs_survive_import(self):
        """Vertex-interpolated primvars:st arrive on the render mesh."""
        from pxr import Sdf, UsdGeom

        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        gfx = self._add_graphics_mesh(stage, "/World/Tire/Skin")
        pv = UsdGeom.PrimvarsAPI(gfx.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        pv.Set([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])

        builder = self._import(stage)
        model = builder.finalize()
        rm = model.deformable_render_meshes[0]
        self.assertIsNotNone(rm.uvs)
        assert_np_equal(rm.uvs.numpy(), np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32), tol=1e-6)

    def test_load_visual_shapes_false_skips_render_meshes(self):
        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        self._add_graphics_mesh(stage, "/World/Tire/Skin")
        builder = self._import(stage, load_visual_shapes=False)
        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 0)


if __name__ == "__main__":
    warnings.simplefilter("default")
    unittest.main(verbosity=2)
