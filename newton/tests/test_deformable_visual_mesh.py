# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for deformable visual meshes: binding kinds, ownership, validation, skinning."""

import contextlib
import math
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.sim.deformable_visual import compute_deformable_visual_mesh_normals, skin_deformable_visual_mesh
from newton._src.viewer.viewer_file import HAS_CBOR2
from newton.exceptions import NewtonWarning
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal
from newton.viewer import ViewerFile, ViewerNull, ViewerUSD


class _MeshProbe(ViewerNull):
    """Captures every ``log_mesh`` call keyed by object name."""

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
        dynamic=False,
        opacity=None,
    ):
        self.calls[name] = {
            "points": None if points is None else points.numpy(),
            "normals": None if normals is None else normals.numpy(),
            "uvs": None if uvs is None else uvs.numpy(),
            "texture": texture,
            "hidden": hidden,
            "opacity": opacity,
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
        expected_vertices, expected_uvs = verts.copy(), uvs.copy()
        indices, particles = _QUAD.copy(), np.arange(n, dtype=np.int32)
        index = builder.add_deformable_visual_mesh(
            verts, indices, kind="particle", particles=particles, uvs=uvs, texture=Path("skin.png"), label="skin"
        )
        verts[:] = 100.0
        uvs[:] = 0.0
        indices[:] = 0
        particles[:] = 0
        model = builder.finalize()

        self.assertEqual(model.deformable_visual_mesh_count, 1)
        rm = model.deformable_visual_meshes[index]
        self.assertEqual(rm.kind, newton.DeformableVisualMesh.Kind.PARTICLE)
        self.assertEqual(rm.index, index)
        self.assertEqual(rm.label, "skin")
        self.assertEqual(rm.texture, "skin.png")
        assert_np_equal(rm.rest_vertices.numpy(), expected_vertices)
        assert_np_equal(rm.uvs.numpy(), expected_uvs, tol=1.0e-6)
        assert_np_equal(rm.indices.numpy(), _QUAD)
        assert_np_equal(rm.parent.numpy(), np.arange(n, dtype=np.int32))

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
        body_q[b1, 3:] = (0, 0, math.sqrt(0.5), math.sqrt(0.5))
        state.body_q.assign(body_q)
        rotated = np.column_stack((1.0 - verts[2:, 1], verts[2:, 0] - 1.0, np.full(2, 2.0)))
        skinned = _skin(model, rm, state)
        assert_np_equal(skinned[:2], verts[:2], tol=1.0e-5)
        assert_np_equal(skinned[2:], rotated, tol=1.0e-5)

    def test_body_visual_survives_unrelated_fixed_joint_collapse(self):
        """Keep a surviving visual driver when an earlier body is removed."""
        builder = newton.ModelBuilder()
        root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        fixed = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        driver = builder.add_link(
            xform=wp.transform((3.0, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3))
        )
        other = builder.add_link(
            xform=wp.transform((10.0, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3))
        )
        builder.add_articulation([builder.add_joint_free(root), builder.add_joint_fixed(root, fixed)])
        builder.add_articulation([builder.add_joint_free(driver)])
        builder.add_articulation([builder.add_joint_free(other)])
        vertices = np.array([[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 0.0, 1.0]], dtype=np.float32)
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="body", bodies=[driver])

        builder.collapse_fixed_joints()
        model = builder.finalize()
        visuals = model.deformable_visuals()
        model.update_deformable_visuals(model.state(), visuals)
        assert_np_equal(visuals.points.numpy(), vertices, tol=1.0e-6)
        self.assertEqual(model.body_count, 3)

    def test_body_visual_follows_merged_driver(self):
        """Preserve visual points when their driver merges into a body or the world."""
        vertices = np.array([[2.0, 1.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 1.0]], dtype=np.float32)
        for fixed_to_world in (False, True):
            with self.subTest(fixed_to_world=fixed_to_world):
                builder = newton.ModelBuilder()
                builder.begin_world()
                root = -1 if fixed_to_world else builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                pose = wp.transform((2.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2))
                child = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
                joints = [] if fixed_to_world else [builder.add_joint_free(root)]
                joints.append(builder.add_joint_fixed(root, child, parent_xform=pose))
                builder.add_articulation(joints)
                builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="body", bodies=[child])
                builder.end_world()

                builder.collapse_fixed_joints()
                model = builder.finalize()
                visuals = model.deformable_visuals()
                model.update_deformable_visuals(model.state(), visuals)
                assert_np_equal(visuals.points.numpy(), vertices, tol=1.0e-6)
                self.assertEqual(model.deformable_visual_meshes[0].world, 0)
                self.assertEqual(model.body_count, 0 if fixed_to_world else 1)

                scene = newton.ModelBuilder()
                scene.add_body(xform=wp.transform((100.0, 0.0, 0.0), wp.quat_identity()))
                scene.replicate(
                    builder, 2, xforms=[wp.transform_identity(), wp.transform((0.0, 0.0, 4.0), wp.quat_identity())]
                )
                replicated = scene.finalize()
                output = replicated.deformable_visuals()
                replicated.update_deformable_visuals(replicated.state(), output)
                assert_np_equal(
                    output.points.numpy(), np.vstack((vertices, vertices + np.array([0, 0, 4]))), tol=1.0e-6
                )
                self.assertEqual([mesh.world for mesh in replicated.deformable_visual_meshes], [0, 1])

    def test_body_binding_keeps_candidate_order_for_ties(self):
        """Nearest origins select the first candidate, retaining rotated offsets."""
        builder = newton.ModelBuilder()
        left = builder.add_body(xform=wp.transform((-1, 0, 0), wp.quat_identity()))
        right = builder.add_body(xform=wp.transform((1, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi / 2)))
        points = np.array([[0, 1, 0], [-0.5, 0, 0], [0.5, 0, 1]], dtype=np.float32)
        builder.add_deformable_visual_mesh(points, [0, 1, 2], kind="body", bodies=[right, left])
        model = builder.finalize(device="cpu")
        mesh = model.deformable_visual_meshes[0]
        np.testing.assert_array_equal(mesh.parent.numpy(), [right, left, right])
        np.testing.assert_allclose(mesh.local_offsets.numpy(), [[1, 1, 0], [0.5, 0, 0], [0, 0.5, 1]], atol=1e-6)
        np.testing.assert_allclose(_skin(model, mesh, model.state()), points, atol=1e-6)

    def test_triangle_embedding_uses_closest_edges_and_corners(self):
        """Euclidean projection must not become independent barycentric clamping."""
        builder = newton.ModelBuilder()
        for point in [(0, 0, 0), (2, 0, 0), (1, 1, 0)]:
            builder.add_particle(wp.vec3(*point), wp.vec3(0), mass=1.0)
        builder.add_triangle(0, 1, 2)
        vertices = np.array(
            [[1, -0.5, 0], [-0.5, 1, 0], [2.5, 1, 0], [-1, -1, 0], [3, -1, 0], [1, 2, 0], [1, 0.2, 3]], dtype=np.float32
        )
        expected = np.array([[1, 0, 0], [0.25, 0.25, 0], [1.75, 0.25, 0], [0, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0.2, 0]])
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="triangle", tri_range=(0, 1))
        model = builder.finalize()
        assert_np_equal(_skin(model, model.deformable_visual_meshes[0], model.state()), expected, tol=1.0e-6)

    def test_triangle_embedding_selects_closest_nondegenerate_parent(self):
        """Ignore collapsed triangles when choosing the closest surface."""
        builder = newton.ModelBuilder()
        points = [(0, 0, 0), (2, 0, 0), (1, 1, 0), (0, -1.1, 0), (1, -2, 0), (2, -1.1, 0)]
        for point in points:
            builder.add_particle(wp.vec3(*point), wp.vec3(0), mass=1.0)
        builder.add_triangle(0, 1, 2)
        builder.add_triangle(3, 4, 5)
        # A formerly valid triangle can be degenerate in a separate visual bind pose.
        for point in [(4, 0, 0), (5, 0, 0), (4, 1, 0)]:
            builder.add_particle(wp.vec3(*point), wp.vec3(0), mass=1.0)
        builder.add_triangle(6, 7, 8)
        builder.particle_q[6:9] = [wp.vec3(1, -0.5, 0)] * 3
        vertices = np.array([[1, -0.5, 0], [1.01, -0.5, 0], [1, -0.49, 0]], dtype=np.float32)
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="triangle", tri_range=(0, 3))
        model = builder.finalize()
        mesh = model.deformable_visual_meshes[0]
        np.testing.assert_array_equal(mesh.parent.numpy(), [0, 0, 0])
        assert_np_equal(_skin(model, mesh, model.state()), np.array([[1, 0, 0], [1.01, 0, 0], [1, 0, 0]]), tol=1.0e-6)

        with self.assertRaisesRegex(ValueError, "add_deformable_visual_mesh:.*non-degenerate triangle"):
            builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="triangle", tri_range=(2, 3))

    def test_triangle_embedding_far_from_surface(self):
        """Bind distant vertices to the closest surface without losing their weights."""
        builder = newton.ModelBuilder()
        for x in (2000.0, 1000.1552):
            start = builder.particle_count
            for point in ((x, 0, 0), (x + 1, 0, 0), (x, 1, 0)):
                builder.add_particle(wp.vec3(*point), wp.vec3(0), mass=1.0)
            builder.add_triangle(start, start + 1, start + 2)
        vertices = np.array(
            [
                [-0.009496063, 0.39774823, 0.12428024],
                [0.00046608306, 0.8325083, 0.13732179],
                [0.0004659774, 0.6846791, 0.16780631],
            ],
            dtype=np.float32,
        )
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="triangle", tri_range=(0, 2))
        # Automatic binding runs on the CPU, independently of the simulation device.
        model = builder.finalize(device="cpu")
        mesh = model.deformable_visual_meshes[0]
        np.testing.assert_array_equal(mesh.parent.numpy(), [1, 1, 1])
        np.testing.assert_allclose(mesh.weights.numpy().sum(axis=1), 1.0, atol=1.0e-6)
        visuals = model.deformable_visuals()
        model.update_deformable_visuals(model.state(), visuals)
        expected = np.column_stack((np.full(3, 1000.1552), vertices[:, 1], np.zeros(3)))
        np.testing.assert_allclose(visuals.get_points(mesh).numpy(), expected, rtol=1.0e-7, atol=1.0e-6)

    def test_automatic_binding_preserves_ties_and_thin_elements(self):
        """Equal candidates choose the first parent, including thin or inverted elements."""
        for kind in ("triangle", "tet"):
            with self.subTest(kind=kind):
                builder = newton.ModelBuilder()
                corners = [(0, 0, 0), (1, 0, 0), (0, 1.0e-5, 0), (0, 0, 1)]
                for corner in corners:
                    builder.add_particle(wp.vec3(*corner), wp.vec3(0), mass=1.0)
                if kind == "triangle":
                    builder.add_triangle(0, 1, 2)
                    builder.add_triangle(0, 1, 2)
                    owning_range = {"tri_range": (0, 2)}
                    vertices = np.array([[0.2, 2.0e-6, 0], [0.4, 1.0e-6, 0], [0.1, 5.0e-6, 0]], dtype=np.float32)
                else:
                    builder.add_tetrahedron(0, 1, 2, 3)
                    builder.add_tetrahedron(0, 1, 2, 3)
                    # Binding must handle an inverted current pose, too.
                    builder.particle_q[3] = wp.vec3(0, 0, -1)
                    owning_range = {"tet_range": (0, 2)}
                    vertices = np.array(
                        [[0.1, 2.0e-6, -0.2], [0.2, 1.0e-6, -0.3], [0.1, 3.0e-6, -0.1]], dtype=np.float32
                    )
                builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind=kind, **owning_range)
                model = builder.finalize()
                mesh = model.deformable_visual_meshes[0]
                np.testing.assert_array_equal(mesh.parent.numpy(), [0, 0, 0])
                assert_np_equal(_skin(model, mesh, model.state()), vertices, tol=1.0e-6)

                builder.particle_q[:] = [wp.vec3(0)] * builder.particle_count
                with self.assertRaisesRegex(ValueError, "add_deformable_visual_mesh:.*non-degenerate"):
                    builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind=kind, **owning_range)

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

    def test_precomputed_weights_preserve_translation(self):
        """Normalize accepted rounding errors so bindings preserve translations."""
        builder = newton.ModelBuilder()
        _add_soft(builder)
        for kind, arity, owning_range in (("triangle", 3, {"tri_range": (0, 1)}), ("tet", 4, {"tet_range": (0, 1)})):
            weights = np.full((3, arity), 1.0 / arity, dtype=np.float32)
            weights[0, 0] += 9.0e-5
            weights[1] = 0.0
            weights[1, :2] = [-1.0e-5, 1.0 + 1.0e-5]
            builder.add_deformable_visual_mesh(
                np.zeros((3, 3)), [0, 1, 2], kind=kind, parent=[0, 0, 0], weights=weights, **owning_range
            )
        model = builder.finalize(device="cpu")
        state = model.state()
        before = [_skin(model, mesh, state) for mesh in model.deformable_visual_meshes]
        translation = np.array([1000.0, -500.0, 250.0], dtype=np.float32)
        state.particle_q.assign(state.particle_q.numpy() + translation)
        for mesh, initial in zip(model.deformable_visual_meshes, before, strict=True):
            with self.subTest(kind=mesh.kind):
                np.testing.assert_allclose(_skin(model, mesh, state), initial + translation, rtol=0, atol=1.0e-4)
                self.assertTrue(np.all(mesh.weights.numpy() >= 0))
                np.testing.assert_allclose(mesh.weights.numpy().sum(axis=1), 1.0, rtol=0, atol=1.0e-7)

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
        main.add_builder(sub, label_prefix="copy")
        model = main.finalize()

        self.assertEqual(model.deformable_visual_mesh_count, 4)
        self.assertEqual(
            [rm.label for rm in model.deformable_visual_meshes], ["copy/cloth", "copy/tri", "copy/tet", "copy/body"]
        )
        by_label = {rm.label.removeprefix("copy/"): rm for rm in model.deformable_visual_meshes}
        state = model.state()

        # Particle and tet meshes reproduce their bind pose at rest.
        assert_np_equal(_skin(model, by_label["cloth"], state), cloth_verts, tol=1.0e-4)
        assert_np_equal(_skin(model, by_label["tet"], state), tet_verts, tol=1.0e-4)
        assert_np_equal(_skin(model, by_label["body"], state), body_verts, tol=1.0e-4)
        # The triangle mesh reproduces its on-surface projection at rest.
        tri_rest = _skin(model, by_label["tri"], state)
        assert_np_equal(tri_rest, cloth_verts[:4], tol=1.0e-4)

    def test_visual_copy_labels_and_parents(self):
        """Keep distinct drivers when copies share a world or use separate worlds."""
        for mode, worlds in (
            ("self", [-1, -1]),
            ("separate_worlds", [0, 1]),
            ("shared_world", [0, 0]),
            ("global", [-1, -1]),
        ):
            with self.subTest(mode=mode):
                builder = newton.ModelBuilder()
                vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
                for point in vertices:
                    builder.add_particle(wp.vec3(*point), wp.vec3(0), mass=1.0)
                builder.add_deformable_visual_mesh(
                    vertices, [0, 1, 2], kind="particle", particles=[0, 1, 2], label="skin"
                )
                if mode != "self":
                    scene = newton.ModelBuilder()
                    if mode == "shared_world":
                        scene.begin_world()
                    scene.replicate(
                        builder,
                        copy_count=2,
                        separate_worlds=mode == "separate_worlds",
                        label_prefixes=["left", "right"],
                    )
                    if mode == "shared_world":
                        scene.end_world()
                    builder = scene
                    labels = ["left/skin", "right/skin"]
                else:
                    builder.add_builder(builder, label_prefix="copy")
                    labels = ["skin", "copy/skin"]
                model = builder.finalize()
                self.assertEqual([mesh.label for mesh in model.deformable_visual_meshes], labels)
                self.assertEqual([mesh.world for mesh in model.deformable_visual_meshes], worlds)
                state = model.state()
                positions = state.particle_q.numpy().copy()
                for i in range(len(labels)):
                    positions[3 * i : 3 * i + 3, 2] = i
                state.particle_q.assign(positions)
                for i, mesh in enumerate(model.deformable_visual_meshes):
                    assert_np_equal(mesh.parent.numpy(), np.arange(3 * i, 3 * i + 3, dtype=np.int32))
                    assert_np_equal(_skin(model, mesh, state), positions[3 * i : 3 * i + 3])

    def test_visual_copy_empty_label_prefix(self):
        """Keep empty prefixes and unlabeled visual meshes unchanged when copying."""
        source = newton.ModelBuilder()
        vertices = np.eye(3, dtype=np.float32)
        for point in vertices:
            source.add_particle(wp.vec3(*point), wp.vec3(0), mass=1.0)
        for label in ("skin", None):
            source.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="particle", particles=[0, 1, 2], label=label)
        scene = newton.ModelBuilder()
        for prefix in (None, "", "copy"):
            scene.add_builder(source, label_prefix=prefix)
        scene.replicate(source, copy_count=3, separate_worlds=False, label_prefixes=[None, "", "copy"])
        model = scene.finalize(device="cpu")
        self.assertEqual(
            [mesh.label for mesh in model.deformable_visual_meshes],
            ["skin", "", "skin", "", "copy/skin", ""] * 2,
        )

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
        assert_np_equal(n[:, 2], np.ones(4, dtype=np.float32), tol=1.0e-5)


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
        second_particles = np.array([6, 7, 11], dtype=np.int32)
        builder.add_deformable_visual_mesh(
            np.asarray(builder.particle_q, dtype=np.float32)[second_particles],
            [0, 1, 2],
            kind="particle",
            particles=second_particles,
            label="second",
        )
        return builder.finalize(), particle_indices, vertices

    def test_allocate_update_and_access_mesh_ranges(self):
        """Different mesh sizes share one output without overlapping ranges."""
        model, _particle_indices, _vertices = self._model()

        visuals = model.deformable_visuals()

        self.assertIsInstance(visuals, newton.DeformableVisuals)
        self.assertIs(visuals.model, model)
        self.assertEqual(visuals.device, model.device)
        self.assertEqual(visuals.vertex_count, 7)
        self.assertEqual(visuals.mesh_ranges, ((0, 4), (4, 7)))
        self.assertEqual(len(visuals.points), 7)
        self.assertEqual(len(visuals.normals), 7)
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
            assert_np_equal(visuals.get_points(mesh).numpy(), moved[mesh.parent.numpy()], tol=1.0e-6)
            normals = visuals.get_normals(mesh).numpy()
            assert_np_equal(np.linalg.norm(normals, axis=1), np.ones(mesh.vertex_count, dtype=np.float32), tol=1.0e-5)

    def test_reuse_and_simultaneous_states_keep_fixed_independent_buffers(self):
        """Reuse a buffer across states, or allocate separate outputs to retain both."""
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
        """Replay captured skinning after changing the simulation state."""
        model, _particle_indices, _vertices = self._model()
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
            assert_np_equal(visuals.get_points(mesh).numpy(), moved[mesh.parent.numpy()], tol=1.0e-6)

    def test_rejects_results_and_meshes_from_another_model(self):
        """Output buffers and mesh handles cannot be used with another model."""
        model_a, _particle_indices, _vertices = self._model()
        model_b, _particle_indices, _vertices = self._model()
        visuals_a = model_a.deformable_visuals()

        with self.assertRaisesRegex(ValueError, "created for another model"):
            model_b.update_deformable_visuals(model_b.state(), visuals_a)

        model_a.update_deformable_visuals(model_a.state(), visuals_a)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            visuals_a.get_points(model_b.deformable_visual_meshes[0])

    def test_rejects_short_state_arrays(self):
        """Reject short particle and body arrays before changing visual output."""
        builder = newton.ModelBuilder()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        for point in vertices:
            builder.add_particle(point, wp.vec3(0), mass=1.0)
        body = builder.add_body()
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="particle", particles=[0, 1, 2])
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="body", bodies=[body])
        # Unused final drivers keep the pre-fix regression safe to execute.
        builder.add_particle(wp.vec3(2), wp.vec3(0), mass=1.0)
        builder.add_body()
        model = builder.finalize()
        state = model.state()
        visuals = model.deformable_visuals()
        model.update_deformable_visuals(state, visuals)
        before = visuals.points.numpy().copy()

        for attribute in ("particle_q", "body_q"):
            with self.subTest(attribute=attribute):
                short_state = model.state()
                setattr(short_state, attribute, getattr(short_state, attribute)[:-1])
                with self.assertRaisesRegex(ValueError, f"State.{attribute}.*at least"):
                    model.update_deformable_visuals(short_state, visuals)
                assert_np_equal(visuals.points.numpy(), before, tol=0.0)
                self.assertIs(visuals.state, state)


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

    def test_replicated_programmatic_mesh_keeps_identity_and_world(self):
        """Preserve programmatic identity while assigning worlds and shifting drivers."""
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
        for world, mesh in enumerate(model.deformable_visual_meshes):
            with self.subTest(world=world):
                self.assertEqual(mesh.world, world)
                self.assertEqual(mesh.label, "skin")
                expected = np.arange(n, dtype=np.int32) + world * n
                np.testing.assert_array_equal(mesh.parent.numpy(), expected)


class TestDeformableVisualMeshViewer(unittest.TestCase):
    """ViewerBase drawing: naming, visibility, world offsets, and shared output."""

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
        """A viewer reads shared output without recomputing the skinning."""
        builder, _uvs = self._cloth_skin_builder()
        model = builder.finalize()
        state = model.state()
        moved = state.particle_q.numpy().copy()
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
        """Export deformed points and bounds without hidden geometry updates."""
        from pxr import UsdGeom

        builder, _uvs = self._cloth_skin_builder(label="123 bad/skin!")
        model = builder.finalize()

        with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as tmp:
            file_path = tmp.name

        try:
            viewer = ViewerUSD(file_path, num_frames=4)
            viewer.set_model(model)
            state = model.state()
            mesh = model.deformable_visual_meshes[0]
            rest = state.particle_q.numpy().copy()
            for frame, visible in enumerate((True, False, False, True)):
                state.particle_q.assign(rest + np.array([0.0, 0.0, frame], dtype=np.float32))
                viewer.show_deformable_visual_meshes = visible
                viewer.begin_frame(frame / 60.0)
                viewer.log_state(state)
                viewer.end_frame()
            path = "/root/model/deformable_visual_meshes/mesh_0_123_bad_skin"
            prim = UsdGeom.Mesh(viewer.stage.GetPrimAtPath(path))
            self.assertTrue(prim)
            self.assertEqual(prim.GetPointsAttr().GetTimeSamples(), [0.0, 3.0])
            self.assertEqual(prim.GetExtentAttr().GetTimeSamples(), [0.0, 3.0])
            for frame in (0, 3):
                points = np.asarray(prim.GetPointsAttr().Get(frame))
                assert_np_equal(np.asarray(prim.GetExtentAttr().Get(frame)), np.array([points.min(0), points.max(0)]))
            for frame, visible in enumerate((True, False, False, True)):
                self.assertEqual(prim.GetVisibilityAttr().Get(frame), "inherited" if visible else "invisible")
            assert_np_equal(np.asarray(prim.GetPointsAttr().Get(3)), _skin(model, mesh, state))
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

    def test_skinned_mesh_toggle_is_independent_from_show_visual(self):
        """Skinned meshes remain visible when regular visual shapes are hidden."""
        builder, _uvs = self._cloth_skin_builder()
        model = builder.finalize()
        viewer = _MeshProbe()
        viewer.set_model(model)
        state = model.state()

        viewer.show_visual = False
        viewer._frame(state)
        name = "/model/deformable_visual_meshes/mesh_0_skin"
        self.assertFalse(viewer.calls[name]["hidden"])
        self.assertTrue(viewer.show_deformable_visual_meshes)

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

        state = model.state()
        moved = state.particle_q.numpy().copy()
        moved[n:, 2] += 1.25
        state.particle_q.assign(moved)
        viewer._frame(state, 1.0)
        p0 = viewer.calls["/model/deformable_visual_meshes/mesh_0_skin"]["points"]
        p1 = viewer.calls["/model/deformable_visual_meshes/mesh_1_skin"]["points"]
        assert_np_equal(p0, verts + offsets[0], tol=1.0e-5)
        assert_np_equal(p1, moved[n:] + offsets[1], tol=1.0e-5)

        viewer.activate("rotated")
        viewer.set_model(model)
        viewer.set_world_offsets((4.0, 0.0, 0.0))
        viewer.set_layer_transform(
            "rotated", wp.transform((1.0, 2.0, 3.0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi / 2))
        )
        viewer._frame(state, 2.0)
        for world in range(2):
            name = f"/layers/rotated/model/deformable_visual_meshes/mesh_{world}_skin"
            original = viewer.calls[f"/model/deformable_visual_meshes/mesh_{world}_skin"]
            # A quarter turn around X maps (x, y, z) to (x, -z, y).
            expected = original["points"][:, [0, 2, 1]].copy()
            expected[:, 1] *= -1
            expected += [1, 2, 3]
            assert_np_equal(viewer.calls[name]["points"], expected, tol=1.0e-5)
            normals = original["normals"][:, [0, 2, 1]].copy()
            normals[:, 1] *= -1
            assert_np_equal(viewer.calls[name]["normals"], normals, tol=1.0e-5)
        viewer.set_visible_worlds([1])
        viewer._frame(state, 2.5)
        self.assertTrue(viewer.calls["/layers/rotated/model/deformable_visual_meshes/mesh_0_skin"]["hidden"])
        self.assertFalse(viewer.calls["/layers/rotated/model/deformable_visual_meshes/mesh_1_skin"]["hidden"])
        viewer.show_deformable_visual_meshes = False
        viewer._frame(state, 3.0)
        self.assertTrue(viewer.calls["/layers/rotated/model/deformable_visual_meshes/mesh_0_skin"]["hidden"])
        self.assertFalse(viewer.calls["/model/deformable_visual_meshes/mesh_0_skin"]["hidden"])

    def test_viewer_reuses_null_world_offsets(self):
        """Reuse one fallback offset buffer when no world offsets are configured."""
        builder, _uvs = self._cloth_skin_builder()
        model = builder.finalize()
        viewer = _MeshProbe()
        viewer.set_model(model)
        viewer.world_offsets = None

        viewer._frame(model.state())
        offsets = viewer._deformable_visual_mesh_null_offsets
        self.assertIsNotNone(offsets)
        self.assertIs(viewer.layer._deformable_visual_mesh_null_offsets, offsets)

        viewer._frame(model.state(), 1.0)
        self.assertIs(viewer._deformable_visual_mesh_null_offsets, offsets)

    def test_null_viewer_does_not_evaluate_visuals(self):
        """Leave shared visual output unevaluated when no renderer consumes it."""
        builder, _uvs = self._cloth_skin_builder()
        model = builder.finalize(device="cpu")
        visuals = model.deformable_visuals()
        viewer = ViewerNull()
        viewer.set_model(model)
        viewer.set_deformable_visuals(visuals)
        viewer.log_state(model.state())
        self.assertIsNone(visuals.state)

    def test_viewer_file_roundtrips_deformable_visual_mesh(self):
        """Replay replicated skins with shared RGBA images and path textures."""
        builder = newton.ModelBuilder()
        _add_cloth(builder)
        n = builder.particle_count
        verts = np.array(builder.particle_q, dtype=np.float32)
        uvs = np.linspace(0.0, 1.0, n * 2, dtype=np.float32).reshape(n, 2)
        texture = np.arange(16, dtype=np.uint8).reshape(2, 2, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "texture.ppm"
            path.write_bytes(b"P6\n2 2\n255\n" + texture[:, :, :3].tobytes())
            for label, source in (("image", texture), ("path", path)):
                builder.add_deformable_visual_mesh(
                    verts,
                    _QUAD,
                    kind="particle",
                    particles=np.arange(n, dtype=np.int32),
                    uvs=uvs,
                    texture=source,
                    label=label,
                )
            scene = newton.ModelBuilder()
            scene.replicate(builder, 2)
            model = scene.finalize()
            state = model.state()
            moved = state.particle_q.numpy().copy()
            moved[n:, 2] += 1.0
            state.particle_q.assign(moved)
            for suffix in ("json", "bin"):
                if suffix == "bin" and not HAS_CBOR2:
                    continue
                with self.subTest(format=suffix):
                    file_path = str(Path(directory) / f"recording.{suffix}")
                    recorder = ViewerFile(file_path, auto_save=False)
                    recorder.set_model(model)
                    visuals = model.deformable_visuals()
                    recorder.set_deformable_visuals(visuals)
                    recorder.log_state(state)
                    self.assertIsNone(visuals.state)
                    recorder.save_recording()
                    playback = ViewerFile(file_path, auto_save=False)
                    playback.load_recording()
                    restored_model = newton.Model()
                    playback.load_model(restored_model)
                    restored_state = restored_model.state()
                    playback.load_state(restored_state, 0)
                    self.assertEqual(restored_model.deformable_visual_mesh_count, 4)
                    for mesh in restored_model.deformable_visual_meshes:
                        self.assertIsInstance(mesh, newton.DeformableVisualMesh)
                        self.assertEqual(mesh.kind, newton.DeformableVisualMesh.Kind.PARTICLE)
                        assert_np_equal(mesh.rest_vertices.numpy(), verts)
                        assert_np_equal(mesh.uvs.numpy(), uvs)
                        if mesh.label == "image":
                            np.testing.assert_array_equal(mesh.texture, texture)
                        else:
                            self.assertEqual(mesh.label, "path")
                            self.assertEqual(mesh.texture, str(path))
                        assert_np_equal(_skin(restored_model, mesh, restored_state), moved[mesh.parent.numpy()])

    def test_gl_reuses_skinned_textures_but_updates_regular_meshes(self):
        """Reuse visual textures untinted by default; preserve explicit colors and ordinary uploads."""
        from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415
        from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip  # noqa: PLC0415

        with (
            wp.ScopedDevice("cpu"),
            tempfile.TemporaryDirectory() as directory,
            contextlib.closing(_make_headless_viewer_gl_or_skip(self)) as viewer,
        ):
            texture = np.zeros((2, 2, 3), dtype=np.uint8)
            texture[:, :, 0] = 255
            path = Path(directory) / "texture.ppm"
            path.write_bytes(b"P6\n2 2\n255\n" + texture.tobytes())
            for source in (str(path), texture):
                with self.subTest(source=type(source).__name__):
                    builder = newton.ModelBuilder()
                    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
                    for point in vertices:
                        builder.add_particle(wp.vec3(*point), wp.vec3(0), mass=1.0)
                    builder.add_deformable_visual_mesh(
                        vertices,
                        [0, 1, 2],
                        kind="particle",
                        particles=[0, 1, 2],
                        uvs=[[0, 0], [1, 0], [0, 1]],
                        texture=source,
                    )
                    model = builder.finalize()
                    viewer.set_model(model)
                    viewer.show_particles = False
                    state = model.state()
                    with mock.patch.object(RendererGL.gl, "glTexImage2D", wraps=RendererGL.gl.glTexImage2D) as upload:
                        for visible in (True, True, False, True):
                            viewer.show_deformable_visual_meshes = visible
                            viewer.log_state(state)
                        self.assertEqual(upload.call_count, 1)
                        viewer.show_deformable_visual_meshes = False
                        viewer.log_state(state)
                        with mock.patch.object(
                            RendererGL.gl, "glBufferData", wraps=RendererGL.gl.glBufferData
                        ) as geometry_upload:
                            viewer.log_state(state)
                            viewer.log_state(state)
                            self.assertEqual(geometry_upload.call_count, 0)
                        viewer.show_deformable_visual_meshes = True
                        viewer.log_state(state)

                        mesh = model.deformable_visual_meshes[0]
                        replacement = np.zeros((2, 2, 3), dtype=np.uint8)
                        mesh.texture = replacement
                        viewer.log_state(state)
                        self.assertEqual(upload.call_count, 2)
                        mesh.texture = None
                        viewer.log_state(state)
                        self.assertEqual(upload.call_count, 2)
                        viewer.log_mesh("ordinary", mesh.rest_vertices, mesh.indices, uvs=mesh.uvs, texture=replacement)
                        replacement[:] = 127
                        viewer.log_mesh("ordinary", mesh.rest_vertices, mesh.indices, uvs=mesh.uvs, texture=replacement)
                        self.assertEqual(upload.call_count, 4)
                        mesh.texture = source
                        viewer.set_model(model)
                        viewer.log_state(state)
                        self.assertEqual(upload.call_count, 5)
                        mesh.texture = str(Path(directory) / "missing.png")
                        with self.assertWarnsRegex(NewtonWarning, "Failed to load texture"):
                            viewer.log_state(state)
                        with warnings.catch_warnings():
                            warnings.simplefilter("error")
                            viewer.log_state(state)
                            viewer.log_state(state)
                        mesh.texture = source
                        viewer.set_camera(pos=wp.vec3(0.3, -2.0, 2.0), pitch=-40.0, yaw=90.0)
                        viewer.show_ground = False
                        viewer.begin_frame(0.0)
                        viewer.log_state(state)
                        viewer.end_frame()
                        frame = viewer.get_frame().numpy().astype(np.int32)
                        red = (frame[:, :, 0] > frame[:, :, 1] + 30) & (frame[:, :, 0] > frame[:, :, 2] + 30)
                        self.assertTrue(red.any(), "The restored visual texture must be sampled by the shader")

                    def center_pixel(state):
                        viewer.begin_frame(1.0)
                        viewer.log_state(state)
                        viewer.end_frame()
                        image = viewer.get_frame().numpy().astype(np.int32)
                        return image[image.shape[0] // 2, image.shape[1] // 2]

                    mesh.texture = None
                    untextured = center_pixel(state)
                    self.assertGreater(untextured[0], untextured[2] + 30)
                    for value in (200, 100):
                        with self.subTest(gray=value):
                            mesh.texture = np.full((2, 2, 3), value, dtype=np.uint8)
                            pixel = center_pixel(state)
                            self.assertGreater(pixel.min(), 30, "Sample the mesh, not the background")
                            self.assertLessEqual(np.ptp(pixel), 5, "Do not tint a neutral texture brown")
                    mesh.texture = None
                    np.testing.assert_allclose(center_pixel(state), untextured, atol=2)

                    # Override through log_mesh; log_state must retain the tint after buffer replacement.
                    viewer.log_mesh(
                        "/model/deformable_visual_meshes/mesh_0",
                        mesh.rest_vertices,
                        mesh.indices,
                        uvs=mesh.uvs,
                        color=(0.1, 0.4, 1.0),
                        dynamic=True,
                    )
                    for image in (None, np.full((2, 2, 3), 200, dtype=np.uint8), None):
                        mesh.texture = image
                        pixel = center_pixel(state)
                        self.assertGreater(pixel[2], pixel[0] + 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
