# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for deformable visual meshes: binding kinds, ownership, validation, skinning."""

import math
import os
import tempfile
import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.sim.deformable_visual import compute_deformable_visual_mesh_normals, skin_deformable_visual_mesh
from newton._src.utils.import_usd_deformable_visual import _sim_bind_positions
from newton.examples.sensors.example_deformable_visual_mesh_camera import Example as DeformableVisualMeshCameraExample
from newton.sensors import SensorTiledCamera
from newton.tests._usd_deformable_test_utils import _add_cable_curve, _add_cloth_mesh, _deformable_stage
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal
from newton.viewer import ViewerFile, ViewerNull, ViewerUSD


class _MeshProbe(ViewerNull):
    """Captures every ``log_mesh`` call keyed by object name.

    Deliberately overrides ``log_mesh`` with the legacy signature (no new
    keywords) to pin that visual-mesh drawing works with pre-existing viewer
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
    skin_deformable_visual_mesh(mesh, state, model, out)
    return out.numpy()


class TestDeformableVisualMeshBindings(unittest.TestCase):
    """Core binding correctness for every kind."""

    def test_particle_kind_follows_particles(self):
        """A particle-bound visual mesh with a 1:1 map equals particle_q, and the
        model output carries the invariant index, kind, world, and UVs."""
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        n = builder.particle_count
        verts = np.array(builder.particle_q, dtype=np.float32)
        uvs = np.linspace(0.0, 1.0, n * 2, dtype=np.float32).reshape(n, 2)
        index = builder.add_deformable_visual_mesh(
            verts, _QUAD, kind="particle", particles=np.arange(n, dtype=np.int32), uvs=uvs, label="skin"
        )
        model = builder.finalize()

        self.assertEqual(model.deformable_visual_mesh_count, 1)
        rm = model.deformable_visual_meshes[index]
        self.assertEqual(rm.kind, newton.DeformableVisualMesh.Kind.PARTICLE)
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
        builder.add_deformable_visual_mesh(verts, _QUAD, kind="triangle", tri_range=(0, builder.tri_count))
        model = builder.finalize()
        rm = model.deformable_visual_meshes[0]

        self.assertEqual(rm.kind, newton.DeformableVisualMesh.Kind.TRIANGLE)
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
        builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet", tet_range=(0, builder.tet_count))
        model = builder.finalize()
        rm = model.deformable_visual_meshes[0]

        self.assertEqual(rm.kind, newton.DeformableVisualMesh.Kind.TET)
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
        """A body-bound visual mesh follows its nearest body's current pose."""
        builder = newton.ModelBuilder()
        b0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        b1 = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(b0, radius=0.1)
        builder.add_shape_sphere(b1, radius=0.1)
        verts = np.array(
            [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [1.0, 0.1, 0.0], [0.9, 0.0, 0.0]],
            dtype=np.float32,
        )
        builder.add_deformable_visual_mesh(verts, _QUAD, kind="body", bodies=[b0, b1])
        model = builder.finalize()
        rm = model.deformable_visual_meshes[0]

        self.assertEqual(rm.kind, newton.DeformableVisualMesh.Kind.BODY)
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
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet", tet_range=(0, builder.tet_count))
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
        model = builder.finalize()
        rm = model.deformable_visual_meshes[0]
        assert_np_equal(rm.parent.numpy(), parent)
        assert_np_equal(rm.weights.numpy(), weights, tol=0.0)

    def test_add_builder_rebases_parents_by_kind(self):
        """add_builder shifts particle, triangle, tet, and body parents into the
        merged index space so skinning stays correct."""
        sub = newton.ModelBuilder()
        _add_cloth(sub)
        n = sub.particle_count
        cloth_verts = np.array(sub.particle_q, dtype=np.float32)
        sub.add_deformable_visual_mesh(
            cloth_verts, _QUAD, kind="particle", particles=np.arange(n, dtype=np.int32), label="cloth"
        )
        sub.add_deformable_visual_mesh(
            cloth_verts[:4], _QUAD, kind="triangle", tri_range=(0, sub.tri_count), label="tri"
        )
        _add_soft(sub)
        tet_verts = np.array([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75], [0.4, 0.6, 0.5]], np.float32)
        sub.add_deformable_visual_mesh(tet_verts, _QUAD, kind="tet", tet_range=(0, sub.tet_count), label="tet")
        b = sub.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
        sub.add_shape_sphere(b, radius=0.1)
        body_verts = np.array([[2.0, 0.1, 0.0], [2.1, 0.0, 0.0], [2.0, 0.0, 0.1], [1.9, 0.0, 0.0]], np.float32)
        sub.add_deformable_visual_mesh(body_verts, _QUAD, kind="body", bodies=[b], label="body")

        main = newton.ModelBuilder()
        _add_soft(main)  # pre-existing content shifts every index space
        b0 = main.add_body(xform=wp.transform(wp.vec3(-5.0, 0.0, 0.0), wp.quat_identity()))
        main.add_shape_sphere(b0, radius=0.1)
        main.add_builder(sub)
        model = main.finalize()

        self.assertEqual(model.deformable_visual_mesh_count, 4)
        by_label = {rm.label: rm for rm in model.deformable_visual_meshes}
        state = model.state()

        # Particle and tet meshes reproduce their bind pose at rest.
        assert_np_equal(_skin(model, by_label["cloth"], state), cloth_verts, tol=1.0e-4)
        assert_np_equal(_skin(model, by_label["tet"], state), tet_verts, tol=1.0e-4)
        assert_np_equal(_skin(model, by_label["body"], state), body_verts, tol=1.0e-4)
        # The triangle mesh reproduces its on-surface projection at rest.
        tri_rest = _skin(model, by_label["tri"], state)
        assert_np_equal(tri_rest, cloth_verts[:4], tol=1.0e-4)

    def test_normals_recompute_from_current_positions(self):
        """compute_deformable_visual_mesh_normals yields unit normals for the deformed quad."""
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        builder.add_deformable_visual_mesh(verts, _QUAD, kind="particle", particles=np.arange(4, dtype=np.int32))
        model = builder.finalize()
        rm = model.deformable_visual_meshes[0]
        points = wp.array(verts, dtype=wp.vec3, device=model.device)
        normals = wp.empty(4, dtype=wp.vec3, device=model.device)
        compute_deformable_visual_mesh_normals(points, rm.indices, normals)
        n = normals.numpy()
        assert_np_equal(np.linalg.norm(n, axis=1), np.ones(4, dtype=np.float32), tol=1.0e-5)
        assert_np_equal(np.abs(n[:, 2]), np.ones(4, dtype=np.float32), tol=1.0e-5)


class TestDeformableVisuals(unittest.TestCase):
    """Shared current visual data allocated and updated through Model."""

    @staticmethod
    def _model():
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        particle_indices = np.array([0, 1, 5, 4], dtype=np.int32)
        vertices = np.asarray(builder.particle_q, dtype=np.float32)[particle_indices]
        builder.add_deformable_visual_mesh(
            vertices,
            _QUAD,
            kind="particle",
            particles=particle_indices,
            label="first",
        )
        builder.add_deformable_visual_mesh(
            vertices,
            _QUAD,
            kind="particle",
            particles=particle_indices,
            label="second",
        )
        return builder.finalize(), particle_indices, vertices

    def test_allocate_update_and_access_mesh_ranges(self):
        model, particle_indices, vertices = self._model()

        visuals = model.deformable_visuals()

        self.assertIsInstance(visuals, newton.DeformableVisuals)
        self.assertIs(visuals.model, model)
        self.assertEqual(visuals.device, model.device)
        self.assertEqual(visuals.vertex_count, 2 * len(vertices))
        self.assertEqual(visuals.mesh_ranges, ((0, 4), (4, 8)))
        self.assertEqual(len(visuals.points), 8)
        self.assertEqual(len(visuals.normals), 8)
        self.assertIsNone(visuals.state)

        with self.assertRaisesRegex(RuntimeError, "update_deformable_visuals"):
            visuals.get_points(model.deformable_visual_meshes[0])

        state = model.state()
        moved = state.particle_q.numpy()
        moved[:, 2] += 1.25
        state.particle_q.assign(moved)

        self.assertIs(model.update_deformable_visuals(state, visuals), visuals)
        self.assertIs(visuals.state, state)
        for mesh in model.deformable_visual_meshes:
            assert_np_equal(visuals.get_points(mesh).numpy(), moved[particle_indices], tol=1.0e-6)
            normals = visuals.get_normals(mesh).numpy()
            assert_np_equal(np.linalg.norm(normals, axis=1), np.ones(4, dtype=np.float32), tol=1.0e-5)

    def test_reuse_and_simultaneous_states_keep_fixed_independent_buffers(self):
        model, particle_indices, _vertices = self._model()
        state_a = model.state()
        state_b = model.state()
        moved_b = state_b.particle_q.numpy()
        moved_b[:, 0] += 3.0
        state_b.particle_q.assign(moved_b)

        visuals = model.deformable_visuals()
        points_ptr = visuals.points.ptr
        model.update_deformable_visuals(state_a, visuals)
        points_a = visuals.get_points(0).numpy().copy()
        model.update_deformable_visuals(state_b, visuals)

        self.assertEqual(visuals.points.ptr, points_ptr)
        self.assertIs(visuals.state, state_b)
        assert_np_equal(visuals.get_points(0).numpy(), moved_b[particle_indices], tol=1.0e-6)
        self.assertFalse(np.allclose(points_a, visuals.get_points(0).numpy()))

        visuals_a = model.deformable_visuals()
        visuals_b = model.deformable_visuals()
        model.update_deformable_visuals(state_a, visuals_a)
        independent_a = visuals_a.get_points(0).numpy().copy()
        model.update_deformable_visuals(state_b, visuals_b)

        self.assertNotEqual(visuals_a.points.ptr, visuals_b.points.ptr)
        assert_np_equal(visuals_a.get_points(0).numpy(), independent_a, tol=0.0)
        assert_np_equal(visuals_b.get_points(0).numpy(), moved_b[particle_indices], tol=1.0e-6)

    def test_update_is_cuda_graph_capturable(self):
        model, particle_indices, _vertices = self._model()
        if not model.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")

        state = model.state()
        visuals = model.deformable_visuals()
        model.update_deformable_visuals(state, visuals)

        with wp.ScopedCapture(model.device) as capture:
            model.update_deformable_visuals(state, visuals)

        moved = state.particle_q.numpy()
        moved[:, 1] += 2.0
        state.particle_q.assign(moved)
        wp.capture_launch(capture.graph)
        visuals.wait()

        for mesh in model.deformable_visual_meshes:
            assert_np_equal(visuals.get_points(mesh).numpy(), moved[particle_indices], tol=1.0e-6)

    def test_rejects_results_and_meshes_from_another_model(self):
        model_a, _particle_indices, _vertices = self._model()
        model_b, _particle_indices, _vertices = self._model()
        visuals_a = model_a.deformable_visuals()

        with self.assertRaisesRegex(ValueError, "created for another model"):
            model_b.update_deformable_visuals(model_b.state(), visuals_a)

        model_a.update_deformable_visuals(model_a.state(), visuals_a)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            visuals_a.get_points(model_b.deformable_visual_meshes[0])


class TestDeformableVisualMeshValidation(unittest.TestCase):
    """Input validation and ownership errors."""

    def _soft_builder(self):
        builder = newton.ModelBuilder()
        _add_soft(builder)
        return builder

    def test_rejects_conflicting_kind_arguments(self):
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "do not apply to this kind"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="tet", tet_range=(0, 1), particles=np.arange(4))
        with self.assertRaisesRegex(ValueError, "do not apply to this kind"):
            builder.add_deformable_visual_mesh(verts, _QUAD, kind="particle", particles=np.arange(4), bodies=[0])

    def test_rejects_unknown_kind_and_missing_mode_argument(self):
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
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
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

    def test_rejects_out_of_range_drivers(self):
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
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
        builder = self._soft_builder()
        verts = np.zeros((4, 3), dtype=np.float32)
        good_w = np.full((4, 4), 0.25, dtype=np.float32)
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

    def test_mixed_world_drivers_rejected_at_finalize(self):
        """Drivers spanning two replicated worlds raise when the model is built."""
        source = newton.ModelBuilder()
        _add_cloth(source)
        n = source.particle_count
        builder = newton.ModelBuilder()
        builder.replicate(source, 2)
        verts = np.zeros((4, 3), dtype=np.float32)
        # One driver from world 0, three from world 1.
        builder.add_deformable_visual_mesh(
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
        source.add_deformable_visual_mesh(
            verts, _QUAD, kind="particle", particles=np.arange(n, dtype=np.int32), label="skin"
        )
        builder = newton.ModelBuilder()
        builder.replicate(source, 2)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 2)
        worlds = sorted(rm.world for rm in model.deformable_visual_meshes)
        self.assertEqual(worlds, [0, 1])


class TestDeformableVisualMeshViewer(unittest.TestCase):
    """ViewerBase drawing: naming, visibility, world offsets, legacy overrides."""

    @staticmethod
    def _cloth_skin_builder(label="skin"):
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        n = builder.particle_count
        verts = np.array(builder.particle_q, dtype=np.float32)
        uvs = np.linspace(0.0, 1.0, n * 2, dtype=np.float32).reshape(n, 2)
        # A non-degenerate grid quad (0-1-5-4) so face normals are well defined.
        quad = np.array([0, 1, 5, 0, 5, 4], dtype=np.int32)
        builder.add_deformable_visual_mesh(
            verts, quad, kind="particle", particles=np.arange(n, dtype=np.int32), uvs=uvs, label=label
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

        call = viewer.calls["/model/deformable_visual_meshes/mesh_0_skin"]
        self.assertFalse(call["hidden"])
        assert_np_equal(call["points"], moved, tol=1.0e-5)
        assert_np_equal(call["uvs"], uvs, tol=1.0e-6)
        self.assertIsNotNone(call["normals"])
        # Unit normals on every vertex the triangles reference.
        referenced = np.linalg.norm(call["normals"][[0, 1, 4, 5]], axis=1)
        assert_np_equal(referenced, np.ones(4, dtype=np.float32), tol=1.0e-4)

    def test_viewer_consumes_explicit_deformable_visuals_without_updating_again(self):
        builder, _uvs = self._cloth_skin_builder()
        model = builder.finalize()
        state = model.state()
        moved = state.particle_q.numpy()
        moved[:, 2] += 0.75
        state.particle_q.assign(moved)
        visuals = model.deformable_visuals()
        model.update_deformable_visuals(state, visuals)

        live = moved.copy()
        live[:, 2] += 1.0
        state.particle_q.assign(live)

        viewer = _MeshProbe()
        viewer.set_model(model)
        viewer.set_deformable_visuals(visuals)
        viewer._frame(state)

        call = viewer.calls["/model/deformable_visual_meshes/mesh_0_skin"]
        assert_np_equal(call["points"], moved, tol=1.0e-5)

        with self.assertRaisesRegex(ValueError, "another state"):
            viewer._frame(model.state(), 1.0)

        viewer.set_deformable_visuals(None)
        viewer._frame(state, 2.0)
        call = viewer.calls["/model/deformable_visual_meshes/mesh_0_skin"]
        assert_np_equal(call["points"], live, tol=1.0e-5)

    def test_viewer_name_is_valid_for_usd_paths(self):
        """Visual mesh names keep the invariant index without creating invalid USD prim names."""
        builder, _uvs = self._cloth_skin_builder(label="123 bad/skin!")
        model = builder.finalize()

        viewer = _MeshProbe()
        viewer.set_model(model)
        viewer._frame(model.state())

        self.assertIn("/model/deformable_visual_meshes/mesh_0_123_bad_skin", viewer.calls)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_viewer_usd_accepts_deformable_visual_mesh_name(self):
        """USD-backed viewers can create prims for visual meshes with numeric indices."""
        builder, _uvs = self._cloth_skin_builder(label="123 bad/skin!")
        model = builder.finalize()

        with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as tmp:
            file_path = tmp.name

        try:
            viewer = ViewerUSD(file_path, num_frames=1)
            viewer.set_model(model)
            viewer.begin_frame(0.0)
            viewer.log_state(model.state())
            viewer.end_frame()
            viewer.close()
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_sim_triangles_stay_visible_and_mesh_toggles(self):
        """Visual meshes draw in addition to the simulation triangles; toggling
        show_deformable_visual_meshes hides the mesh but keeps valid geometry registered."""
        builder, _uvs = self._cloth_skin_builder()
        model = builder.finalize()
        viewer = _MeshProbe()
        viewer.set_model(model)
        state = model.state()

        viewer._frame(state)
        self.assertFalse(viewer.calls["/model/triangles"]["hidden"])
        self.assertFalse(viewer.calls["/model/deformable_visual_meshes/mesh_0_skin"]["hidden"])

        viewer.show_deformable_visual_meshes = False
        viewer._frame(state, 1.0)
        call = viewer.calls["/model/deformable_visual_meshes/mesh_0_skin"]
        self.assertTrue(call["hidden"])
        self.assertEqual(len(call["points"]), model.deformable_visual_meshes[0].vertex_count)
        self.assertFalse(viewer.calls["/model/triangles"]["hidden"])

    def test_replicated_worlds_use_distinct_names_and_device_offsets(self):
        """Replicated meshes draw under distinct names, each offset by its own
        world offset."""
        source = newton.ModelBuilder()
        _add_cloth(source)
        n = source.particle_count
        verts = np.array(source.particle_q, dtype=np.float32)
        source.add_deformable_visual_mesh(
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
        p0 = viewer.calls["/model/deformable_visual_meshes/mesh_0_skin"]["points"]
        p1 = viewer.calls["/model/deformable_visual_meshes/mesh_1_skin"]["points"]
        assert_np_equal(p0, verts + offsets[0], tol=1.0e-5)
        assert_np_equal(p1, verts + offsets[1], tol=1.0e-5)

    def test_viewer_file_roundtrips_deformable_visual_mesh(self):
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        n = builder.particle_count
        verts = np.array(builder.particle_q, dtype=np.float32)
        uvs = np.linspace(0.0, 1.0, n * 2, dtype=np.float32).reshape(n, 2)
        builder.add_deformable_visual_mesh(
            verts,
            _QUAD,
            kind="particle",
            particles=np.arange(n, dtype=np.int32),
            uvs=uvs,
            label="recorded_skin",
        )
        model = builder.finalize()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            file_path = tmp.name

        try:
            recorder = ViewerFile(file_path, auto_save=False)
            recorder.set_model(model)
            recorder.log_state(model.state())
            recorder.save_recording()

            playback = ViewerFile(file_path)
            playback.load_recording()
            restored_model = newton.Model()
            playback.load_model(restored_model)

            self.assertEqual(restored_model.deformable_visual_mesh_count, 1)
            restored_mesh = restored_model.deformable_visual_meshes[0]
            self.assertIsInstance(restored_mesh, newton.DeformableVisualMesh)
            self.assertEqual(restored_mesh.kind, newton.DeformableVisualMesh.Kind.PARTICLE)
            self.assertEqual(restored_mesh.label, "recorded_skin")
            assert_np_equal(restored_mesh.rest_vertices.numpy(), verts, tol=1.0e-6)
            assert_np_equal(restored_mesh.uvs.numpy(), uvs, tol=1.0e-6)

            viewer = _MeshProbe()
            viewer.set_model(restored_model)
            restored_state = restored_model.state()
            playback.load_state(restored_state, 0)
            viewer._frame(restored_state)
            self.assertIn("/model/deformable_visual_meshes/mesh_0_recorded_skin", viewer.calls)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


class TestDeformableVisualMeshSensor(unittest.TestCase):
    """Camera sensor visibility for skinned deformable visual meshes."""

    @staticmethod
    def _unpack_rgba(packed: int) -> np.ndarray:
        value = int(packed)
        return np.array(
            [
                value & 0xFF,
                (value >> 8) & 0xFF,
                (value >> 16) & 0xFF,
                (value >> 24) & 0xFF,
            ],
            dtype=np.uint8,
        )

    @staticmethod
    def _triangle_surface_with_visual_mesh_behind(texture=None, visual_uvs=None):
        builder = newton.ModelBuilder()
        verts = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        particles = [
            builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0, radius=0.0) for p in verts
        ]
        builder.add_triangle(particles[0], particles[1], particles[2])
        builder.add_triangle(particles[0], particles[2], particles[3])

        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()))
        visual_verts = verts.copy()
        visual_verts[:, 2] = -0.5
        if visual_uvs is None:
            visual_uvs = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        builder.add_deformable_visual_mesh(
            visual_verts,
            _QUAD,
            kind="body",
            bodies=[body],
            uvs=visual_uvs,
            texture=texture,
            label="sensor_skin",
        )
        return builder.finalize()

    @staticmethod
    def _camera_setup(sensor, model, width=16, height=16):
        camera_rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(45.0))
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0, 0.0, 2.0), wp.quat_identity())]],
            dtype=wp.transformf,
            device=model.device,
        )
        return camera_rays, camera_transforms

    def test_tiled_camera_sees_particle_bound_visual_mesh(self):
        builder = newton.ModelBuilder()
        verts = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        particles = [builder.add_particle(pos=tuple(p), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0) for p in verts]
        builder.add_deformable_visual_mesh(
            verts,
            _QUAD,
            kind="particle",
            particles=np.array(particles, dtype=np.int32),
            label="sensor_quad",
        )
        model = builder.finalize()
        state = model.state()

        sensor = SensorTiledCamera(
            model,
            config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                max_distance=10.0,
            ),
        )
        width = 16
        height = 16
        camera_rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(45.0))
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0, 0.0, 2.0), wp.quat_identity())]],
            dtype=wp.transformf,
            device=model.device,
        )
        depth_image = sensor.utils.create_depth_image_output(width, height, camera_count=1)

        sensor.update(state, camera_transforms, camera_rays, depth_image=depth_image)
        depth = depth_image.numpy()[0, 0]

        self.assertGreater(int(np.count_nonzero(depth > 0.0)), 0)
        self.assertAlmostEqual(float(depth[height // 2, width // 2]), 2.0, delta=0.25)

        moved = state.particle_q.numpy()
        moved[:, 2] += 0.5
        state.particle_q = wp.array(moved, dtype=wp.vec3, device=model.device)

        sensor.update(state, camera_transforms, camera_rays, depth_image=depth_image)
        moved_depth = depth_image.numpy()[0, 0]
        self.assertAlmostEqual(float(moved_depth[height // 2, width // 2]), 1.5, delta=0.25)

    def test_tiled_camera_consumes_explicit_deformable_visuals_without_updating_again(self):
        model = self._triangle_surface_with_visual_mesh_behind()
        state = model.state()
        visuals = model.deformable_visuals()
        model.update_deformable_visuals(state, visuals)

        body_q = state.body_q.numpy()
        body_q[:, 2] += 1.0
        state.body_q.assign(body_q)

        sensor = SensorTiledCamera(
            model,
            config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                max_distance=10.0,
            ),
        )
        camera_rays, camera_transforms = self._camera_setup(sensor, model)
        depth_image = sensor.utils.create_depth_image_output(16, 16, camera_count=1)

        sensor.update(
            state,
            camera_transforms,
            camera_rays,
            depth_image=depth_image,
            deformable_visuals=visuals,
        )

        depth = depth_image.numpy()[0, 0]
        self.assertAlmostEqual(float(depth[8, 8]), 2.5, delta=0.25)

        with self.assertRaisesRegex(ValueError, "another state"):
            sensor.update(
                model.state(),
                camera_transforms,
                camera_rays,
                depth_image=depth_image,
                deformable_visuals=visuals,
            )

    def test_tiled_camera_can_hide_sim_triangles_for_visual_mesh_capture(self):
        model = self._triangle_surface_with_visual_mesh_behind()
        state = model.state()

        default_sensor = SensorTiledCamera(
            model,
            config=SensorTiledCamera.RenderConfig(enable_particles=False, max_distance=10.0),
        )
        camera_rays, camera_transforms = self._camera_setup(default_sensor, model)
        default_depth_image = default_sensor.utils.create_depth_image_output(16, 16, camera_count=1)
        default_sensor.update(state, camera_transforms, camera_rays, depth_image=default_depth_image)
        default_depth = default_depth_image.numpy()[0, 0]
        self.assertAlmostEqual(float(default_depth[8, 8]), 2.0, delta=0.25)

        visual_sensor = SensorTiledCamera(
            model,
            config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                max_distance=10.0,
            ),
        )
        camera_rays, camera_transforms = self._camera_setup(visual_sensor, model)
        visual_depth_image = visual_sensor.utils.create_depth_image_output(16, 16, camera_count=1)
        visual_sensor.update(state, camera_transforms, camera_rays, depth_image=visual_depth_image)
        visual_depth = visual_depth_image.numpy()[0, 0]
        self.assertAlmostEqual(float(visual_depth[8, 8]), 2.5, delta=0.25)

    def test_tiled_camera_colors_dynamic_visual_mesh_rgb_hits(self):
        model = self._triangle_surface_with_visual_mesh_behind()
        state = model.state()

        sensor = SensorTiledCamera(
            model,
            config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                max_distance=10.0,
            ),
        )
        camera_rays, camera_transforms = self._camera_setup(sensor, model)
        color_image = sensor.utils.create_color_image_output(16, 16, camera_count=1)

        sensor.update(state, camera_transforms, camera_rays, color_image=color_image)
        rgba = sensor.utils.to_rgba_from_color(color_image).numpy()[0, 8, 8]

        self.assertGreater(int(rgba[3]), 0)
        self.assertTrue(int(rgba[0]) != int(rgba[1]) or int(rgba[1]) != int(rgba[2]))

    def test_tiled_camera_samples_dynamic_visual_mesh_texture(self):
        texture = np.zeros((4, 4, 4), dtype=np.uint8)
        texture[..., 0] = 255
        texture[..., 3] = 255
        model = self._triangle_surface_with_visual_mesh_behind(texture=texture)
        state = model.state()

        sensor = SensorTiledCamera(
            model,
            config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                enable_textures=True,
                max_distance=10.0,
            ),
        )
        camera_rays, camera_transforms = self._camera_setup(sensor, model)
        albedo_image = sensor.utils.create_albedo_image_output(16, 16, camera_count=1)

        sensor.update(state, camera_transforms, camera_rays, albedo_image=albedo_image)
        rgba = self._unpack_rgba(albedo_image.numpy()[0, 0, 8, 8])

        self.assertGreater(int(rgba[0]), 240)
        self.assertLess(int(rgba[1]), 16)
        self.assertLess(int(rgba[2]), 16)
        self.assertEqual(int(rgba[3]), 255)

    def test_tiled_camera_wraps_dynamic_visual_mesh_texture_uvs(self):
        texture = np.zeros((4, 4, 4), dtype=np.uint8)
        texture[:, :2, 0] = 255
        texture[:, 2:, 1] = 255
        texture[..., 3] = 255
        visual_uvs = np.array(
            [
                [1.0, 0.0],
                [1.5, 0.0],
                [1.5, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        model = self._triangle_surface_with_visual_mesh_behind(texture=texture, visual_uvs=visual_uvs)
        state = model.state()

        sensor = SensorTiledCamera(
            model,
            config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                enable_textures=True,
                max_distance=10.0,
            ),
        )
        camera_rays, camera_transforms = self._camera_setup(sensor, model)
        albedo_image = sensor.utils.create_albedo_image_output(16, 16, camera_count=1)

        sensor.update(state, camera_transforms, camera_rays, albedo_image=albedo_image)
        rgba = self._unpack_rgba(albedo_image.numpy()[0, 0, 8, 8])

        self.assertGreater(int(rgba[0]), 240)
        self.assertLess(int(rgba[1]), 16)
        self.assertLess(int(rgba[2]), 16)
        self.assertEqual(int(rgba[3]), 255)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestDeformableVisualMeshUSDImport(unittest.TestCase):
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
        visual mesh with USD ownership metadata, without any custom relationship
        and without return_deformable_results."""
        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        self._add_graphics_mesh(stage, "/World/Tire/Skin")

        builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 1)
        rm = model.deformable_visual_meshes[0]
        self.assertEqual(rm.kind, newton.DeformableVisualMesh.Kind.TET)
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
        by_path = {rm.graphics_path: rm for rm in model.deformable_visual_meshes}
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
        self.assertEqual(model.deformable_visual_mesh_count, 0)

    def test_collision_marked_geometry_is_not_visual(self):
        """A Mesh marked with a collision API under the body is not a visual mesh."""
        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        collider = self._add_graphics_mesh(stage, "/World/Tire/Collider")
        collider.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")

        with self.assertWarnsRegex(UserWarning, "approximated by the simulation geometry"):
            builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 0)

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
        rm = model.deformable_visual_meshes[0]
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
        self.assertEqual(model.deformable_visual_mesh_count, 0)

    def test_sim_bind_pose_count_must_match_imported_particle_range(self):
        """A valid USD pose cannot silently fall back when lowering realizes fewer particles."""
        from pxr import Sdf

        stage = self._stage()
        _, sim = self._add_volume_body(stage, "/World/Tire")
        sim.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )
        ctx = SimpleNamespace(
            stage=stage,
            builder=newton.ModelBuilder(),
            get_prim_world_mat=lambda *_args: wp.mat44_identity(),
            incoming_world_xform=wp.transform_identity(),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "invalid_bind_pose_count.*4 points.*3 particles"):
                _sim_bind_positions(ctx, "/World/Tire/Sim", (0, 3))

    def test_degenerate_simulation_bind_tet_skips_visual_binding(self):
        """A visual cannot bind to a tetrahedron collapsed in the simulation bind pose."""
        from pxr import Sdf

        stage = self._stage()
        _, sim = self._add_volume_body(stage, "/World/Tire")
        self._add_graphics_mesh(stage, "/World/Tire/Skin")
        sim.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
        )

        with self.assertWarnsRegex(UserWarning, "degenerate_parent"):
            builder = self._import(stage)

        self.assertEqual(builder.finalize().deformable_visual_mesh_count, 0)

    def test_degenerate_simulation_bind_triangle_skips_visual_binding(self):
        """A visual cannot bind to triangles collapsed in the simulation bind pose."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        sim = _add_cloth_mesh(stage, "/World/Body/Sim")
        self._add_graphics_mesh(stage, "/World/Body/Skin")
        sim.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (3.0, 0.0, 1.0)]
        )

        with self.assertWarnsRegex(UserWarning, "degenerate_parent"):
            builder = self._import(stage)

        self.assertEqual(builder.finalize().deformable_visual_mesh_count, 0)

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
        self.assertEqual(model.deformable_visual_mesh_count, 1)
        self.assertEqual(model.deformable_visual_meshes[0].graphics_path, "/World/Tire/GoodSkin")

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
        rest = model.deformable_visual_meshes[0].rest_vertices.numpy()
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
        self.assertEqual(model.deformable_visual_mesh_count, 1)
        rm = model.deformable_visual_meshes[0]
        self.assertEqual(rm.kind, newton.DeformableVisualMesh.Kind.TRIANGLE)
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
        self.assertEqual(model.deformable_visual_mesh_count, 1)
        rm = model.deformable_visual_meshes[0]
        self.assertEqual(rm.kind, newton.DeformableVisualMesh.Kind.BODY)
        self.assertEqual(rm.body_path, "/World/Body")
        parents = rm.parent.numpy()
        self.assertTrue((parents >= 0).all() and (parents < model.body_count).all())

    def test_uvs_survive_import(self):
        """Vertex-interpolated primvars:st arrive on the visual mesh."""
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
        rm = model.deformable_visual_meshes[0]
        self.assertIsNotNone(rm.uvs)
        assert_np_equal(rm.uvs.numpy(), np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32), tol=1e-6)

    def test_load_visual_shapes_false_skips_visual_meshes(self):
        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        self._add_graphics_mesh(stage, "/World/Tire/Skin")
        builder = self._import(stage, load_visual_shapes=False)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 0)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestDeformableVisualMeshCameraParity(unittest.TestCase):
    """The checked-in AOUSD scene matches its procedural reference scene."""

    @staticmethod
    def _build(load_from_usd: bool):
        example = DeformableVisualMeshCameraExample.__new__(DeformableVisualMeshCameraExample)
        return example._build_model_builder(SimpleNamespace(load_from_usd=load_from_usd))

    def test_procedural_and_usd_builders_match(self):
        procedural = self._build(False)
        usd = self._build(True)

        exact_fields = (
            "body_q",
            "body_com",
            "body_inertia",
            "body_flags",
            "joint_type",
            "joint_parent",
            "joint_child",
            "joint_X_p",
            "joint_X_c",
            "joint_axis",
            "joint_q",
            "joint_qd",
            "joint_target_kd",
            "joint_target_mode",
            "tri_indices",
            "edge_indices",
            "edge_rest_angle",
            "edge_bending_properties",
            "tet_indices",
            "tet_materials",
            "shape_type",
            "shape_body",
            "shape_transform",
            "shape_scale",
            "shape_flags",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_mu",
        )
        for field in exact_fields:
            np.testing.assert_array_equal(np.asarray(getattr(procedural, field)), np.asarray(getattr(usd, field)))

        close_fields = (
            "body_mass",
            "body_inv_mass",
            "body_inv_inertia",
            "joint_target_ke",
            "particle_q",
            "particle_qd",
            "particle_mass",
            "particle_radius",
            "particle_flags",
            "tri_areas",
            "tri_materials",
            "edge_rest_length",
        )
        for field in close_fields:
            np.testing.assert_allclose(
                np.asarray(getattr(procedural, field)),
                np.asarray(getattr(usd, field)),
                rtol=1.0e-6,
                atol=1.0e-7,
            )

        self.assertEqual(len(procedural._deformable_visual_meshes), len(usd._deformable_visual_meshes))
        for procedural_visual, usd_visual in zip(
            procedural._deformable_visual_meshes, usd._deformable_visual_meshes, strict=True
        ):
            np.testing.assert_allclose(
                procedural_visual["rest_vertices"], usd_visual["rest_vertices"], rtol=1.0e-6, atol=1.0e-7
            )
            np.testing.assert_array_equal(procedural_visual["indices"], usd_visual["indices"])
            np.testing.assert_allclose(procedural_visual["uvs"], usd_visual["uvs"], rtol=1.0e-6, atol=1.0e-7)
            self.assertEqual(procedural_visual["texture"], usd_visual["texture"])

        procedural_model = procedural.finalize()
        usd_model = usd.finalize()
        final_fields = (
            "body_q",
            "body_mass",
            "body_inv_mass",
            "body_inertia",
            "body_inv_inertia",
            "joint_target_ke",
            "joint_target_kd",
            "particle_q",
            "particle_mass",
            "particle_radius",
            "tri_areas",
            "tri_materials",
            "edge_rest_length",
            "tet_materials",
            "shape_transform",
            "shape_scale",
        )
        for field in final_fields:
            np.testing.assert_array_equal(
                getattr(procedural_model, field).numpy(),
                getattr(usd_model, field).numpy(),
            )

        for procedural_visual, usd_visual in zip(
            procedural_model.deformable_visual_meshes, usd_model.deformable_visual_meshes, strict=True
        ):
            assert_np_equal(
                _skin(procedural_model, procedural_visual, procedural_model.state()),
                _skin(usd_model, usd_visual, usd_model.state()),
                tol=1.0e-6,
            )


if __name__ == "__main__":
    warnings.simplefilter("default")
    unittest.main(verbosity=2)
