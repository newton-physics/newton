# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test deformable visual mesh data, evaluation, and viewer rendering."""

import contextlib
import importlib.util
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
from newton._src.viewer.viewer_file import HAS_CBOR2, transfer_to_model
from newton.exceptions import NewtonWarning
from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip
from newton.tests.unittest_utils import assert_np_equal, get_test_devices
from newton.viewer import ViewerFile, ViewerNull, ViewerUSD


def _particle_mesh():
    builder = newton.ModelBuilder()
    for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        builder.add_particle(pos=wp.vec3(point), vel=(0, 0, 0), mass=1.0)
    return builder


def _attach(builder, **kwargs):
    args = {
        "vertices": builder.particle_q[:3],
        "indices": [0, 1, 2],
        "kind": "particle",
        "parent": [0, 1, 2],
        "label": "skin",
    }
    args.update(kwargs)
    return builder.add_deformable_visual_mesh(**args)


class TestDeformableVisualMeshData(unittest.TestCase):
    def test_storage_and_composition(self):
        """Preserve rest geometry and offset particle bindings when copying and replicating."""
        source = _particle_mesh()
        _attach(source)
        target = newton.ModelBuilder()
        target.add_particle(pos=(9, 9, 9), vel=(0, 0, 0), mass=1.0)
        target.add_builder(source, xform=wp.transform((3, 0, 0), wp.quat_identity()), label_prefix="copy")
        target.replicate(source, 2)
        model = target.finalize("cpu")
        self.assertEqual(model.deformable_visual_mesh_count, 3)
        for index, mesh in enumerate(model.deformable_visual_meshes):
            with self.subTest(index=index):
                np.testing.assert_array_equal(mesh.parent.numpy(), np.arange(3) + 1 + 3 * index)
                np.testing.assert_array_equal(mesh.rest_vertices.numpy(), source.particle_q)
                np.testing.assert_array_equal(mesh.indices.numpy(), [0, 1, 2])
                self.assertEqual(mesh.index, index)
                self.assertEqual(mesh.vertex_count, 3)
                self.assertEqual(mesh.world, index - 1)
                self.assertEqual(mesh.kind, newton.geometry.DeformableVisualMesh.Kind.PARTICLE)
                for name in ("world", "kind", "index", "rest_vertices", "indices", "parent"):
                    with self.assertRaises(AttributeError):
                        setattr(mesh, name, getattr(mesh, name))
        self.assertEqual(model.deformable_visual_meshes[0].label, "copy/skin")
        np.testing.assert_array_equal(source.finalize("cpu").deformable_visual_meshes[0].parent.numpy(), [0, 1, 2])
        source.add_builder(source)
        self.assertEqual(source.finalize("cpu").deformable_visual_mesh_count, 2)
        self.assertEqual(newton.ModelBuilder().finalize("cpu").deformable_visual_meshes, ())

    def test_equal_rest_geometry_shares_storage(self):
        """Share equal rest vertices and indices across meshes and worlds while bindings stay separate."""
        source = _particle_mesh()
        vertices = np.asarray(source.particle_q)
        for points in (vertices, vertices.copy(), vertices * 2):
            _attach(source, vertices=points)
        builder = newton.ModelBuilder()
        builder.replicate(source, 3)
        meshes = builder.finalize("cpu").deformable_visual_meshes
        first = meshes[0]
        for i in (1, 3, 4, 6, 7):
            for name in ("rest_vertices", "indices"):
                self.assertEqual(getattr(meshes[i], name).ptr, getattr(first, name).ptr)
            self.assertNotEqual(meshes[i].parent.ptr, first.parent.ptr)
        self.assertNotEqual(meshes[2].rest_vertices.ptr, first.rest_vertices.ptr)
        self.assertEqual(meshes[2].indices.ptr, first.indices.ptr)
        np.testing.assert_array_equal(meshes[2].rest_vertices.numpy(), vertices * 2)

    def test_validation_and_world_ownership(self):
        """Reject invalid geometry and bindings without adding partial records."""
        builder = _particle_mesh()
        for kwargs in (
            {"vertices": []},
            {"vertices": [[float("nan"), 0, 0]] * 3},
            {"indices": []},
            {"indices": [0, 1]},
            {"indices": [0, 1, 3]},
            {"indices": [0, 1, -1]},
            {"parent": [0, 1]},
            {"parent": [0, 1, 3]},
            {"parent": [0, 1, -1]},
            {"parent": [0, 1, 1.5]},
            {"kind": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                _attach(builder, **kwargs)
        self.assertEqual(builder.finalize("cpu").deformable_visual_mesh_count, 0)
        builder = _particle_mesh()
        builder.begin_world()
        builder.add_particle(pos=(1, 1, 1), vel=(0, 0, 0), mass=1.0)
        builder.end_world()
        _attach(builder, parent=[0, 1, 3])
        with self.assertRaisesRegex(ValueError, "drivers span worlds"):
            builder.finalize("cpu")


def _barycentric_mesh(kind):
    builder = _particle_mesh()
    builder.add_particle(pos=wp.vec3(0, 0, 1), vel=(0, 0, 0), mass=1.0)
    if kind == "triangle":
        builder.add_triangle(0, 1, 2)
        weights = np.array([[0.5, 0.3, 0.2], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        owning = {"tri_range": (0, 1)}
    else:
        builder.add_tetrahedron(0, 1, 2, 3)
        weights = np.array([[0.1, 0.2, 0.3, 0.4], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        owning = {"tet_range": (0, 1)}
    return builder, weights, owning


class TestDeformableVisualBarycentric(unittest.TestCase):
    def test_bindings_and_replication(self):
        """Preserve weighted deformation and parent offsets through replication on CPU and CUDA."""
        for device in get_test_devices():
            for kind in ("triangle", "tetrahedron"):
                with self.subTest(device=device, kind=kind):
                    source, weights, owning = _barycentric_mesh(kind)
                    _attach(source, kind=kind, parent=[0, 0, 0], weights=weights, **owning)
                    builder = newton.ModelBuilder()
                    builder.replicate(source, 2)
                    model = builder.finalize(device)
                    state = model.state()
                    positions = state.particle_q.numpy().copy()
                    positions[:4] += (2, 3, 4)
                    positions[4:] *= 2
                    state.particle_q.assign(positions)
                    result = newton.geometry.DeformableVisuals(model)
                    result.update(state)
                    if model.device.is_cuda:
                        with wp.ScopedCapture(model.device) as capture:
                            result.update(state)
                        positions += 1
                        state.particle_q.assign(positions)
                        wp.capture_launch(capture.graph)
                    for i, mesh in enumerate(model.deformable_visual_meshes):
                        np.testing.assert_array_equal(mesh.parent.numpy(), [i] * 3)
                        np.testing.assert_allclose(mesh.weights.numpy(), weights)
                        expected = weights @ positions[i * 4 : i * 4 + weights.shape[1]]
                        np.testing.assert_allclose(result.get_points(mesh).numpy(), expected, atol=1e-6)
                        self.assertEqual(mesh.world, i)

    def test_weights_and_ranges(self):
        """Reject invalid bindings and normalize accepted floating-point roundoff."""
        for kind in ("triangle", "tetrahedron"):
            builder, weights, owning = _barycentric_mesh(kind)
            for bad in (weights * 2, -weights, weights * float("nan"), weights[:, :-1]):
                with self.subTest(kind=kind, weights=bad), self.assertRaises(ValueError):
                    _attach(builder, kind=kind, parent=[0] * 3, weights=bad, **owning)
            for parents in ([-1] * 3, [1] * 3, [0, 0], [0, 0, 0.5]):
                with self.subTest(kind=kind, parents=parents), self.assertRaises((ValueError, TypeError)):
                    _attach(builder, kind=kind, parent=parents, weights=weights, **owning)
            for bad_range in ((-1, 1), (0, 0), (0, 2)):
                with self.subTest(kind=kind, owning=bad_range), self.assertRaises(ValueError):
                    _attach(builder, kind=kind, parent=[0] * 3, weights=weights, **dict.fromkeys(owning, bad_range))
            with self.assertRaises(ValueError):
                _attach(builder, kind=kind, parent=[0] * 3, **owning)
            _attach(builder, kind=kind, parent=[0] * 3, weights=weights * 1.00001, **owning)
            model = builder.finalize("cpu")
            np.testing.assert_allclose(model.deformable_visual_meshes[0].weights.numpy().sum(axis=1), 1, atol=1e-7)

    def test_recorded_object_owning_ranges(self):
        """Resolve owning elements from a recorded surface or volume object and reject the rest."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=0.1,
        )
        builder.add_soft_grid(
            pos=wp.vec3(5.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=1.0,
            cell_y=1.0,
            cell_z=1.0,
            density=100.0,
            k_mu=1.0e3,
            k_lambda=1.0e3,
            k_damp=0.0,
        )
        self.assertEqual((len(builder.surface_label), len(builder.volume_label)), (1, 1))
        eye = np.eye(3, dtype=np.float32)
        quarter = np.full((3, 4), 0.25, dtype=np.float32)
        _attach(builder, kind="triangle", surface=0, parent=[0, 1, 1], weights=eye)
        _attach(builder, kind="triangle", tri_range=(0, 2), parent=[0, 1, 1], weights=eye)
        _attach(builder, kind="tetrahedron", volume=0, parent=[0] * 3, weights=quarter)
        for kwargs in (
            {"kind": "triangle", "surface": 0, "parent": [2] * 3, "weights": eye},
            {"kind": "triangle", "surface": 1, "parent": [0] * 3, "weights": eye},
            {"kind": "triangle", "surface": 0, "tri_range": (0, 2), "parent": [0] * 3, "weights": eye},
            {"kind": "triangle", "volume": 0, "parent": [0] * 3, "weights": eye},
            {"kind": "tetrahedron", "surface": 0, "parent": [0] * 3, "weights": quarter},
            {"kind": "particle", "surface": 0, "parent": [0] * 3},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _attach(builder, **kwargs)
        meshes = builder.finalize("cpu").deformable_visual_meshes
        self.assertEqual(len(meshes), 3)
        np.testing.assert_array_equal(meshes[0].parent.numpy(), meshes[1].parent.numpy())
        np.testing.assert_array_equal(meshes[0].weights.numpy(), meshes[1].weights.numpy())
        np.testing.assert_array_equal(meshes[2].parent.numpy(), [0, 0, 0])

    def test_element_world_ownership(self):
        """Check all particles of each driving element, not just the first one."""
        builder = _particle_mesh()
        builder.begin_world()
        builder.add_particle(pos=wp.vec3(0, 0, 1), vel=(0, 0, 0), mass=1)
        builder.end_world()
        builder.add_triangle(0, 1, 3)
        _attach(builder, kind="triangle", parent=[0] * 3, weights=np.eye(3), tri_range=(0, 1))
        with self.assertRaisesRegex(ValueError, "drivers span worlds"):
            builder.finalize("cpu")


def _skin(model, mesh, state):
    output = newton.geometry.DeformableVisuals(model)
    output.update(state)
    return output.get_points(mesh).numpy()


class TestDeformableVisualTriangleBinding(unittest.TestCase):
    def test_triangle_embedding_uses_closest_edges_and_corners(self):
        """Project onto the closest edge or corner instead of clamping barycentric coordinates independently."""
        builder = newton.ModelBuilder()
        for point in [(0, 0, 0), (2, 0, 0), (1, 1, 0)]:
            builder.add_particle(wp.vec3(*point), wp.vec3(0), mass=1.0)
        builder.add_triangle(0, 1, 2)
        vertices = np.array(
            [[1, -0.5, 0], [-0.5, 1, 0], [2.5, 1, 0], [-1, -1, 0], [3, -1, 0], [1, 2, 0], [1, 0.2, 3]], dtype=np.float32
        )
        expected = np.array([[1, 0, 0], [0.25, 0.25, 0], [1.75, 0.25, 0], [0, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0.2, 0]])
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="triangle", tri_range=(0, 1))
        model = builder.finalize("cpu")
        np.testing.assert_allclose(
            _skin(model, model.deformable_visual_meshes[0], model.state()), expected, atol=1.0e-6
        )

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
        model = builder.finalize("cpu")
        mesh = model.deformable_visual_meshes[0]
        np.testing.assert_array_equal(mesh.parent.numpy(), [0, 0, 0])
        np.testing.assert_allclose(
            _skin(model, mesh, model.state()), np.array([[1, 0, 0], [1.01, 0, 0], [1, 0, 0]]), atol=1.0e-6
        )

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
        visuals = newton.geometry.DeformableVisuals(model)
        visuals.update(model.state())
        expected = np.column_stack((np.full(3, 1000.1552), vertices[:, 1], np.zeros(3)))
        np.testing.assert_allclose(visuals.get_points(mesh).numpy(), expected, rtol=1.0e-7, atol=1.0e-6)

    def test_ties_thin_and_invalid_triangles(self):
        """Break exact distance ties toward the first triangle and reject all-degenerate ranges."""
        builder, _, _ = _barycentric_mesh("triangle")
        builder.add_triangle(0, 1, 2)
        builder.particle_q[2] = wp.vec3(0, 1.0e-5, 0)
        vertices = np.array([[0.2, 2e-6, 0], [0.4, 1e-6, 0], [0.1, 5e-6, 0]])
        _attach(builder, vertices=vertices, kind="triangle", parent=None, tri_range=(0, 2))
        model = builder.finalize("cpu")
        mesh = model.deformable_visual_meshes[0]
        np.testing.assert_array_equal(mesh.parent.numpy(), [0, 0, 0])
        np.testing.assert_allclose(_skin(model, mesh, model.state()), vertices, atol=1e-6)
        builder.particle_q[:] = [wp.vec3(0)] * builder.particle_count
        with self.assertRaisesRegex(ValueError, "non-degenerate"):
            _attach(builder, vertices=vertices, kind="triangle", parent=None, tri_range=(0, 2))


class TestDeformableVisualTetBinding(unittest.TestCase):
    def test_containing_tet_and_precomputed_binding(self):
        """Find an owning tet on the host and keep explicit bindings independent of input arrays."""
        vertices = np.array([[0.2, 0.2, 0.2], [0.1, 0.3, 0.2], [0.2, 0.1, 0.4]], dtype=np.float32)
        builder, _, owning = _barycentric_mesh("tetrahedron")
        _attach(builder, vertices=vertices, kind="tetrahedron", parent=None, **owning)
        # Automatic binding runs on the host; device evaluation is covered by the barycentric tests.
        model = builder.finalize("cpu")
        state = model.state()
        positions = state.particle_q.numpy().copy()
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        state.particle_q.assign(positions @ rotation.T + (1, 2, 3))
        mesh = model.deformable_visual_meshes[0]
        np.testing.assert_allclose(_skin(model, mesh, state), vertices @ rotation.T + (1, 2, 3), atol=1e-6)
        parents = np.zeros(3, dtype=np.int32)
        weights = np.full((3, 4), 0.25, dtype=np.float32)
        builder.particle_q[:] = [wp.vec3(0)] * 4
        _attach(builder, kind="tetrahedron", parent=parents, weights=weights, **owning)
        parents[:] = 99
        weights[:] = 0
        mesh = builder.finalize("cpu").deformable_visual_meshes[1]
        np.testing.assert_array_equal(mesh.parent.numpy(), [0, 0, 0])
        np.testing.assert_array_equal(mesh.weights.numpy(), np.full((3, 4), 0.25))

    def test_fallback_cannot_cross_owning_range(self):
        """Keep an outside vertex with its owning object even when another object's tetrahedron contains it."""
        builder, _, owning = _barycentric_mesh("tetrahedron")
        for point in ((5, 0, 0), (6, 0, 0), (5, 1, 0), (5, 0, 1)):
            builder.add_particle(pos=wp.vec3(point), vel=(0, 0, 0), mass=1)
        builder.add_tetrahedron(4, 5, 6, 7)
        vertices = np.array([[5.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.1, 0.2, 0.3]])
        with self.assertLogs("newton", level="WARNING") as logs:
            _attach(builder, vertices=vertices, kind="tetrahedron", parent=None, **owning)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("1 of 3 visual vertices fell outside", logs.records[0].getMessage())
        model = builder.finalize("cpu")
        mesh = model.deformable_visual_meshes[0]
        np.testing.assert_array_equal(mesh.parent.numpy(), [0, 0, 0])
        np.testing.assert_allclose(mesh.weights.numpy().sum(axis=1), 1, atol=1e-6)
        points = _skin(model, mesh, model.state())
        np.testing.assert_allclose(points[0], np.array([5.2, 0.2, 0.2]) / 5.6, atol=1e-6)
        np.testing.assert_allclose(points[1:], vertices[1:], atol=1e-6)

    def test_ties_thin_inverted_and_degenerate_tets(self):
        """Keep deterministic ties and accept a thin or inverted bind pose."""
        builder, _, _ = _barycentric_mesh("tetrahedron")
        builder.add_tetrahedron(0, 1, 2, 3)
        builder.particle_q[2] = wp.vec3(0, 1e-5, 0)
        builder.particle_q[3] = wp.vec3(0, 0, -1)
        vertices = np.array([[0.1, 2e-6, -0.2], [0.2, 1e-6, -0.3], [0.1, 3e-6, -0.1]])
        _attach(builder, vertices=vertices, kind="tetrahedron", parent=None, tet_range=(0, 2))
        model = builder.finalize("cpu")
        mesh = model.deformable_visual_meshes[0]
        np.testing.assert_array_equal(mesh.parent.numpy(), [0, 0, 0])
        np.testing.assert_allclose(_skin(model, mesh, model.state()), vertices, atol=1e-6)
        builder.particle_q[:] = [wp.vec3(0)] * 4
        with self.assertRaisesRegex(ValueError, "non-degenerate"):
            _attach(builder, vertices=vertices, kind="tetrahedron", parent=None, tet_range=(0, 2))


class TestDeformableVisualRodBinding(unittest.TestCase):
    def test_body_validation_and_state_storage(self):
        """Reject invalid body bindings and incompatible body state before changing output."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=1, inertia=wp.mat33(np.eye(3)))
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        for binding in ({"bodies": []}, {"bodies": [1]}, {"parent": [-1] * 3}, {"parent": [0, 0]}):
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="body", **binding)
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="body", bodies=[0])
        model = builder.finalize("cpu")
        state = model.state()
        output = newton.geometry.DeformableVisuals(model)
        output.update(state)
        before = output.points.numpy().copy()
        invalid = [None, wp.empty(0, dtype=wp.transform, device="cpu")]
        if wp.is_cuda_available():
            invalid.append(wp.empty(1, dtype=wp.transform, device="cuda:0"))
        for value in invalid:
            state.body_q = value
            with self.assertRaises(ValueError):
                output.update(state)
            np.testing.assert_array_equal(output.points.numpy(), before)

    def test_body_binding_uses_center_of_mass_or_given_parent(self):
        """Measure nearest-body binding to the center of mass and let ``parent`` name the body directly."""
        builder = newton.ModelBuilder()
        # A segment whose frame origin sits at its start: its center of mass is at x = 1.
        start = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        other = builder.add_body(xform=wp.transform(wp.vec3(1.5, 0.0, 0.0), wp.quat_identity()))
        # The shape's mass places the first body's center of mass at x = 1.
        builder.add_shape_sphere(start, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), radius=0.1)
        builder.add_shape_sphere(other, radius=0.1)
        verts = np.array([[0.9, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.1, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)
        # Nearer to the other body's origin, but inside the first segment.
        builder.add_deformable_visual_mesh(verts, [0, 1, 2, 0, 2, 3], kind="body", bodies=[start, other])
        builder.add_deformable_visual_mesh(verts, [0, 1, 2, 0, 2, 3], kind="body", parent=[other] * 4)
        model = builder.finalize("cpu")
        nearest, given = model.deformable_visual_meshes
        assert_np_equal(nearest.parent.numpy(), np.full(4, start, dtype=np.int32))
        assert_np_equal(given.parent.numpy(), np.full(4, other, dtype=np.int32))
        state = model.state()
        output = newton.geometry.DeformableVisuals(model)
        output.update(state)
        for mesh in (nearest, given):
            assert_np_equal(output.get_points(mesh).numpy(), verts, tol=1.0e-6)
        with self.assertRaisesRegex(ValueError, "exactly one of parent, bodies, or curve"):
            builder.add_deformable_visual_mesh(
                verts, [0, 1, 2, 0, 2, 3], kind="body", bodies=[start], parent=[start] * 4
            )

    def test_curve_object_candidates(self):
        """Take the candidate bodies from a recorded curve object and reject mixed or foreign options."""
        builder = newton.ModelBuilder()
        bodies, _ = builder.add_rod(rod=newton.Rod([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], radius=0.05))
        self.assertEqual(len(builder.curve_label), 1)
        vertices = np.array([[0.4, 0.1, 0.0], [1.6, 0.1, 0.0], [1.0, 0.1, 0.0]], dtype=np.float32)
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="body", curve=0)
        builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="body", bodies=bodies)
        for kwargs in (
            {"kind": "body", "curve": 1},
            {"kind": "body", "curve": 0, "bodies": bodies},
            {"kind": "body", "curve": 0, "parent": [bodies[0]] * 3},
            {"kind": "body", "curve": 0, "weights": np.eye(3)},
            {"kind": "particle", "parent": [0, 1, 2], "curve": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                builder.add_deformable_visual_mesh(vertices, [0, 1, 2], **kwargs)
        from_curve, from_bodies = builder.finalize("cpu").deformable_visual_meshes
        np.testing.assert_array_equal(from_curve.parent.numpy(), from_bodies.parent.numpy())
        np.testing.assert_array_equal(from_curve.local_offsets.numpy(), from_bodies.local_offsets.numpy())
        self.assertTrue(set(from_curve.parent.numpy().tolist()) <= set(bodies))

    def test_body_binding_follows_bodies(self):
        """Follow the nearest body's current pose."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                b0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
                b1 = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
                builder.add_shape_sphere(b0, radius=0.1)
                builder.add_shape_sphere(b1, radius=0.1)
                verts = np.array(
                    [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [1.0, 0.1, 0.0], [0.9, 0.0, 0.0]],
                    dtype=np.float32,
                )
                builder.add_deformable_visual_mesh(verts, [0, 1, 2, 0, 2, 3], kind="body", bodies=[b0, b1])
                model = builder.finalize()
                rm = model.deformable_visual_meshes[0]

                self.assertEqual(rm.kind, newton.geometry.DeformableVisualMesh.Kind.BODY)
                parents = rm.parent.numpy()
                self.assertEqual(parents[0], b0)
                self.assertEqual(parents[2], b1)

                # Translate body 1 only; its two vertices move, body 0's stay.
                state = model.state()
                output = newton.geometry.DeformableVisuals(model)
                body_q = state.body_q.numpy().copy()
                body_q[b1, :3] += (0.0, 0.0, 2.0)
                state.body_q = wp.array(body_q, dtype=wp.transform)
                output.update(state)
                skinned = output.get_points(rm).numpy()
                assert_np_equal(skinned[:2], verts[:2], tol=1.0e-5)
                assert_np_equal(skinned[2:], verts[2:] + np.array([0.0, 0.0, 2.0], dtype=np.float32), tol=1.0e-5)
                body_q[b1, 3:] = (0, 0, math.sqrt(0.5), math.sqrt(0.5))
                state.body_q.assign(body_q)
                rotated = np.column_stack((1.0 - verts[2:, 1], verts[2:, 0] - 1.0, np.full(2, 2.0)))
                output.update(state)
                skinned = output.get_points(rm).numpy()
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
        model = builder.finalize("cpu")
        visuals = newton.geometry.DeformableVisuals(model)
        visuals.update(model.state())
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
                model = builder.finalize("cpu")
                visuals = newton.geometry.DeformableVisuals(model)
                visuals.update(model.state())
                assert_np_equal(visuals.points.numpy(), vertices, tol=1.0e-6)
                self.assertEqual(model.deformable_visual_meshes[0].world, 0)
                self.assertEqual(model.body_count, 0 if fixed_to_world else 1)

                scene = newton.ModelBuilder()
                scene.add_body(xform=wp.transform((100.0, 0.0, 0.0), wp.quat_identity()))
                scene.replicate(
                    builder, 2, xforms=[wp.transform_identity(), wp.transform((0.0, 0.0, 4.0), wp.quat_identity())]
                )
                replicated = scene.finalize("cpu")
                output = newton.geometry.DeformableVisuals(replicated)
                output.update(replicated.state())
                assert_np_equal(
                    output.points.numpy(), np.vstack((vertices, vertices + np.array([0, 0, 4]))), tol=1.0e-6
                )
                self.assertEqual([mesh.world for mesh in replicated.deformable_visual_meshes], [0, 1])

    def test_body_binding_keeps_candidate_order_for_ties(self):
        """Select the first candidate on equal distances and keep rotated offsets."""
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


class TestDeformableVisualEvaluation(unittest.TestCase):
    def test_evaluate_and_reuse(self):
        """Evaluate different mesh sizes and reuse output across states on CPU and CUDA."""
        for device in get_test_devices():
            with self.subTest(device=device):
                builder = _particle_mesh()
                _attach(builder)
                _attach(builder, parent=[2, 1, 0, 2], vertices=[(0, 0, 0)] * 4, indices=[0, 1, 2, 0, 2, 3])
                model = builder.finalize(device)
                state_a, state_b = model.state(), model.state()
                moved = state_b.particle_q.numpy().copy()
                moved[:, 2] += 2
                moved[1, 2] += 1
                state_b.particle_q.assign(moved)
                result = newton.geometry.DeformableVisuals(model)
                self.assertIs(result.model, model)
                self.assertEqual(result.mesh_ranges, ((0, 3), (3, 7)))
                with self.assertRaisesRegex(RuntimeError, "not been updated"):
                    result.get_points(0)
                pointers = (result.points.ptr, result.normals.ptr)
                result.update(state_a)
                retained = newton.geometry.DeformableVisuals(model)
                retained.update(state_a)
                before = retained.points.numpy().copy()
                self.assertIs(result.update(state_b), result)
                result.wait()
                for mesh in model.deformable_visual_meshes:
                    np.testing.assert_allclose(result.get_points(mesh).numpy(), moved[mesh.parent.numpy()])
                np.testing.assert_allclose(
                    result.get_normals(0).numpy(), np.tile(np.array([-1, 0, 1]) / np.sqrt(2), (3, 1)), atol=1e-6
                )
                np.testing.assert_array_equal(retained.points.numpy(), before)
                self.assertEqual((result.points.ptr, result.normals.ptr), pointers)

    def test_capture_and_wait(self):
        """Capture the first update, replay alternating states, and read from another stream."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires CUDA")
        builder = _particle_mesh()
        _attach(builder)
        model = builder.finalize("cuda:0")
        states = [model.state(), model.state()]
        result = newton.geometry.DeformableVisuals(model)
        graphs = []
        for i, state in enumerate(states):
            if i:
                result.update(state)
            with wp.ScopedCapture(model.device) as capture:
                result.update(state)
            graphs.append(capture.graph)
        moved = states[1].particle_q.numpy().copy() + 2
        states[1].particle_q.assign(moved)
        consumer = wp.Stream(model.device)
        pointers = (result.points.ptr, result.normals.ptr)
        for i in (0, 1, 0):
            wp.capture_launch(graphs[i])
            with wp.ScopedStream(consumer, sync_enter=False):
                result.wait(consumer)
                observed = wp.clone(result.get_points(0)).numpy()
            np.testing.assert_allclose(observed, states[i].particle_q.numpy())
            self.assertEqual((result.points.ptr, result.normals.ptr), pointers)

    def test_reject_foreign_meshes_and_invalid_states(self):
        """Reject meshes of another model and invalid state storage before changing output."""
        builder = _particle_mesh()
        _attach(builder)
        model = builder.finalize("cpu")
        other = builder.finalize("cpu")
        result = newton.geometry.DeformableVisuals(model)
        state = model.state()
        result.update(state)
        before = result.points.numpy().copy()
        with self.assertRaisesRegex(ValueError, "does not belong"):
            result.get_points(other.deformable_visual_meshes[0])
        for index in (-1, 1, 0.5):
            with self.subTest(index=index), self.assertRaises((IndexError, TypeError)):
                result.get_points(index)
        invalid = [None, wp.zeros(2, dtype=wp.vec3, device="cpu")]
        if wp.is_cuda_available():
            invalid.append(wp.zeros(3, dtype=wp.vec3, device="cuda:0"))
        for positions in invalid:
            state.particle_q = positions
            with self.assertRaises(ValueError):
                result.update(state)
            np.testing.assert_array_equal(result.points.numpy(), before)
        empty = newton.ModelBuilder().finalize("cpu")
        self.assertEqual(newton.geometry.DeformableVisuals(empty).update(empty.state()).vertex_count, 0)


class TestDeformableVisualRecording(unittest.TestCase):
    def test_all_bindings_roundtrip_without_evaluation(self):
        """Record the fixed assets and state, then rebuild batches on first playback use."""
        builder, weights, owning = _barycentric_mesh("tetrahedron")
        builder.add_triangle(0, 1, 2)
        body = builder.add_body(mass=1, inertia=wp.mat33(np.eye(3)))
        _attach(builder)
        _attach(builder, kind="triangle", parent=[0] * 3, weights=np.eye(3), tri_range=(0, 1))
        _attach(builder, kind="tetrahedron", parent=[0] * 3, weights=weights, **owning)
        _attach(builder, kind="body", parent=[body] * 3)
        image = np.full((2, 2, 3), 200, dtype=np.uint8)
        _attach(builder, uvs=[[0, 0], [1, 0], [0, 1]], texture=image)
        scene = newton.ModelBuilder()
        scene.replicate(builder, 2)
        # Source paths are set by the USD importer; stand in for it on the first spec.
        scene._deformable_visual_meshes[0].update(
            body_path="/World/Cloth", sim_path="/World/Cloth/Sim", graphics_path="/World/Cloth/Skin"
        )
        model = scene.finalize("cpu")
        state = model.state()
        state.particle_q.assign(state.particle_q.numpy().copy() + 2)
        poses = state.body_q.numpy().copy()
        poses[:, 2] += 3
        state.body_q.assign(poses)
        expected = newton.geometry.DeformableVisuals(model)
        expected.update(state)
        mesh = model.deformable_visual_meshes[0]
        self.assertEqual((mesh.body_path, mesh.sim_path), ("/World/Cloth", "/World/Cloth/Sim"))
        with self.assertRaises(AttributeError):
            mesh.graphics_path = "/elsewhere"
        with wp.ScopedDevice("cpu"), tempfile.TemporaryDirectory() as directory:
            for suffix in ("json", "bin") if HAS_CBOR2 else ("json",):
                with self.subTest(format=suffix):
                    path = str(Path(directory) / ("scene." + suffix))
                    recorder = ViewerFile(path, auto_save=False)
                    recorder.set_model(model)
                    recorder.log_state(state)
                    recorder.save_recording()
                    playback = ViewerFile(path, auto_save=False)
                    playback.load_recording()
                    for restored in (newton.Model(), newton.ModelBuilder().finalize("cpu")):
                        playback.load_model(restored)
                        replay = restored.state()
                        playback.load_state(replay, 0)
                        output = newton.geometry.DeformableVisuals(restored)
                        output.update(replay)
                        self.assertEqual(restored.deformable_visual_mesh_count, 10)
                        np.testing.assert_array_equal(
                            restored.deformable_visual_meshes[4].uvs.numpy(), [[0, 0], [1, 0], [0, 1]]
                        )
                        np.testing.assert_array_equal(restored.deformable_visual_meshes[4].texture, image)
                        for original, mesh in zip(
                            model.deformable_visual_meshes, restored.deformable_visual_meshes, strict=True
                        ):
                            self.assertEqual(mesh.kind, original.kind)
                            self.assertEqual(mesh.world, original.world)
                            for path in ("body_path", "sim_path", "graphics_path"):
                                self.assertEqual(getattr(mesh, path), getattr(original, path))
                            np.testing.assert_array_equal(mesh.parent.numpy(), original.parent.numpy())
                            np.testing.assert_allclose(
                                output.get_points(mesh).numpy(), expected.get_points(original).numpy()
                            )
                    kept = scene.finalize("cpu")
                    meshes = kept.deformable_visual_meshes
                    playback.load_model(kept)
                    self.assertIs(kept.deformable_visual_meshes, meshes)
                    mismatch = builder.finalize("cpu")
                    before = mismatch.particle_q.numpy().copy()
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        playback.load_model(mismatch)
                    np.testing.assert_array_equal(mismatch.particle_q.numpy(), before)

    def test_repeated_numpy_arrays_round_trip_by_reference(self):
        """Resolve a numpy array that a recording stores once and references again."""
        from newton._src.viewer.viewer_file import ArrayCache, deserialize, serialize  # noqa: PLC0415

        def identity(value, _path):
            return value

        # Only the binary format stores a repeated array once and references it afterwards.
        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        encoded = serialize({"first": array, "second": array}, identity, format_type="cbor2", cache=ArrayCache())
        self.assertEqual(encoded["items"]["second"]["__type__"], "numpy.ndarray_ref")
        decoded = deserialize(encoded, identity, format_type="cbor2", cache=ArrayCache())
        np.testing.assert_array_equal(decoded["first"], array)
        np.testing.assert_array_equal(decoded["second"], array)

    def test_older_physics_recording_keeps_current_meshes(self):
        """Load a physics-only recording into a finalized model without replacing its meshes."""
        builder = _particle_mesh()
        physics = builder.finalize("cpu")
        _attach(builder)
        target = builder.finalize("cpu")
        meshes = target.deformable_visual_meshes
        with wp.ScopedDevice("cpu"), tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "physics.json")
            recorder = ViewerFile(path, auto_save=False)
            recorder.set_model(physics)
            recorder.save_recording()
            playback = ViewerFile(path, auto_save=False)
            playback.load_recording()
            playback.load_model(target)
        self.assertIs(target.deformable_visual_meshes, meshes)
        old_order = newton.Model()
        transfer_to_model({"deformable_visual_meshes": list(meshes), "deformable_visual_mesh_count": 0}, old_order)
        self.assertEqual(old_order.deformable_visual_mesh_count, 1)


class TestDeformableVisualViewer(unittest.TestCase):
    def test_gl_render_motion_visibility_and_shared_output(self):
        """Render the skin, move it, and toggle it independently of ordinary visuals."""
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self, width=128, height=96)
        self.addCleanup(viewer.close)
        builder = _particle_mesh()
        _attach(builder, label="0 skin/path")
        model = builder.finalize("cpu")
        viewer.set_model(model)
        viewer.set_camera(pos=wp.vec3(0.3, -2, 2), pitch=-40, yaw=90)
        viewer.show_particles = viewer.show_ground = viewer.show_visual = False
        state = model.state()

        def frame():
            viewer.begin_frame(0)
            viewer.log_state(state)
            viewer.end_frame()
            return viewer.get_frame().numpy().copy()

        before = frame()
        name = "/model/deformable_visual_meshes/mesh_0_0_skin_path"
        self.assertIn(name, viewer.objects)
        viewer.show_deformable_visual_meshes = False
        hidden = frame()
        self.assertTrue(viewer.objects[name].hidden)
        self.assertFalse(np.array_equal(before, hidden))
        viewer.show_deformable_visual_meshes = True
        np.testing.assert_array_equal(frame(), before)
        # A later label edit must not strand a stale object under the old name.
        model.deformable_visual_meshes[0].label = "renamed"
        np.testing.assert_array_equal(frame(), before)
        self.assertEqual([n for n in viewer.objects if "deformable_visual_meshes" in n], [name])
        shared = newton.geometry.DeformableVisuals(model)
        shared.update(state)
        viewer.set_deformable_visuals(shared)
        moved = state.particle_q.numpy().copy()
        moved[2, 0] += 0.5
        state.particle_q.assign(moved)
        np.testing.assert_array_equal(frame(), before)
        shared.update(state)
        after = frame()
        self.assertFalse(np.array_equal(before, after))
        viewer.set_deformable_visuals(None)
        np.testing.assert_array_equal(frame(), after)

    def test_gl_world_and_layer_placement(self):
        """Place replicated skins independently using device world and layer offsets."""
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self)
        self.addCleanup(viewer.close)
        source = _particle_mesh()
        _attach(source)
        scene = newton.ModelBuilder()
        scene.replicate(source, 2)
        model = scene.finalize("cpu")
        viewer.set_model(model)
        viewer.show_particles = False
        viewer.set_world_offsets((2, 0, 0))
        viewer.set_layer_transform("__default__", wp.transform((0, 0, 3), wp.quat_identity()))
        viewer.log_state(model.state())
        offsets = viewer.world_offsets.numpy()
        self.assertEqual(model.deformable_visual_mesh_count, 2)
        for i, mesh in enumerate(model.deformable_visual_meshes):
            self.assertEqual(mesh.world, i)
            self.assertEqual(mesh.label, "skin")
            np.testing.assert_array_equal(
                mesh.parent.numpy(), np.arange(source.particle_count) + i * source.particle_count
            )
            name = f"/model/deformable_visual_meshes/mesh_{i}_skin"
            vertices = viewer.objects[name].vertices.numpy()
            expected = np.asarray(source.particle_q) + offsets[i] + (0, 0, 3)
            np.testing.assert_allclose(vertices["pos"], expected)

    def test_non_rendering_backends_ignore_skins(self):
        """Let Null and File ignore the option like their other display flags."""
        builder = _particle_mesh()
        _attach(builder)
        model = builder.finalize("cpu")
        with tempfile.TemporaryDirectory() as directory:
            viewers = [ViewerNull(num_frames=1), ViewerFile(str(Path(directory) / "scene.json"), auto_save=False)]
            for viewer in viewers:
                viewer.set_model(model)
                viewer.show_deformable_visual_meshes = True
                viewer.log_state(model.state())
                self.assertIsNone(viewer._deformable_visuals)
                viewer.close()


class TestDeformableVisualGLTextures(unittest.TestCase):
    def test_uv_validation_and_copy(self):
        """Reject malformed UVs and texture arrays, and copy accepted UVs before replication."""
        builder = _particle_mesh()
        image = np.full((2, 2, 3), 127, dtype=np.uint8)
        uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        for kwargs in (
            {"texture": image},
            {"uvs": [[0, 0]]},
            {"uvs": [[float("nan"), 0]] * 3},
            {"uvs": uvs, "texture": np.zeros((2, 2), dtype=np.uint8)},
            {"uvs": uvs, "texture": np.zeros((2, 2, 1), dtype=np.uint8)},
            {"uvs": uvs, "texture": np.zeros((0, 2, 3), dtype=np.uint8)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _attach(builder, **kwargs)
        _attach(builder, uvs=uvs, texture=image)
        uvs[:] = 0
        scene = newton.ModelBuilder()
        scene.replicate(builder, 2)
        for mesh in scene.finalize("cpu").deformable_visual_meshes:
            np.testing.assert_array_equal(mesh.uvs.numpy(), [[0, 0], [1, 0], [0, 1]])
            np.testing.assert_array_equal(mesh.texture, image)

    def test_gl_updates_skinned_vertices_on_cuda(self):
        """Render changed vertices without host transfers or repeated index uploads."""
        from newton._src.viewer.gl.opengl import RenderVertex  # noqa: PLC0415

        if not wp.is_cuda_available():
            self.skipTest("CUDA is not available")
        with wp.ScopedDevice("cuda:0"), contextlib.closing(_make_headless_viewer_gl_or_skip(self)) as viewer:
            builder = newton.ModelBuilder()
            points = np.array([[-0.5, 0, 0], [0.5, 0, 0], [0, 0, 1]], dtype=np.float32)
            for point in points:
                builder.add_particle(point, wp.vec3(0), mass=1.0)
            builder.add_deformable_visual_mesh(points, [0, 1, 2], kind="particle", parent=[0, 1, 2])
            model = builder.finalize()
            viewer.set_model(model)
            viewer.show_particles = False
            viewer.show_ground = False
            viewer.set_camera(wp.vec3(0, -3, 0.5), pitch=0, yaw=90)
            state = model.state()
            mesh = model.deformable_visual_meshes[0]
            mapped_types = []
            map_buffer = wp.RegisteredGLBuffer.map

            def map_vertices(buffer, dtype, *args, **kwargs):
                mapped_types.append(dtype)
                return map_buffer(buffer, dtype, *args, **kwargs)

            # Probe the existing interop path separately from the skinned-mesh default.
            with (
                warnings.catch_warnings(record=True),
                mock.patch.object(wp.RegisteredGLBuffer, "map", map_vertices),
            ):
                warnings.simplefilter("always", NewtonWarning)
                viewer.log_mesh("probe", mesh.rest_vertices, mesh.indices, dynamic=True, hidden=True)
            if not mapped_types:
                self.skipTest("CUDA-OpenGL interoperability is unavailable")

            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.end_frame()
            before = viewer.get_frame().numpy().copy()
            points[2, 0] += 0.5
            state.particle_q.assign(points)
            mapped_types.clear()
            to_numpy = wp.array.numpy

            def check_readback(array):
                self.assertNotIn(array.dtype, (RenderVertex, wp.uint32), "Unexpected mesh readback")
                return to_numpy(array)

            viewer.begin_frame(1.0)
            with (
                mock.patch.object(wp.array, "numpy", check_readback),
                mock.patch.object(wp.RegisteredGLBuffer, "map", map_vertices),
            ):
                viewer.log_state(state)
                viewer.log_state(state)
            self.assertEqual(mapped_types, [RenderVertex, RenderVertex])
            viewer.end_frame()
            after = viewer.get_frame().numpy()
            self.assertFalse(np.array_equal(before, after), "The rendered mesh must follow its moved particle")

    def test_gl_reuses_skinned_textures_but_updates_regular_meshes(self):
        """Reuse visual textures untinted by default; preserve explicit colors and ordinary uploads."""
        from newton._src.utils.texture import load_texture  # noqa: PLC0415
        from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

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
                        parent=[0, 1, 2],
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
                    max_size = RendererGL.gl.GLint()
                    RendererGL.gl.glGetIntegerv(RendererGL.gl.GL_MAX_TEXTURE_SIZE, max_size)
                    mesh.texture = np.zeros((1, max_size.value + 1, 3), dtype=np.uint8)
                    with mock.patch("newton._src.utils.texture.load_texture", wraps=load_texture) as load:
                        for _ in range(2):
                            np.testing.assert_allclose(center_pixel(state), untextured, atol=2)
                        self.assertEqual(load.call_count, 1, "Do not reload a rejected immutable texture")
                    for value in (200, 100):
                        with self.subTest(gray=value):
                            mesh.texture = np.full((2, 2, 3), value, dtype=np.uint8)
                            pixel = center_pixel(state)
                            self.assertGreater(pixel.min(), 30, "Sample the mesh, not the background")
                            self.assertLessEqual(np.ptp(pixel), 5, "Do not tint a neutral texture brown")

                    mesh.texture = source
                    with (
                        mock.patch.object(
                            RendererGL.gl, "glTexImage2D", side_effect=RuntimeError("injected upload failure")
                        ),
                        self.assertRaisesRegex(RuntimeError, "injected upload failure"),
                    ):
                        viewer.log_state(state)
                    with mock.patch.object(RendererGL.gl, "glTexImage2D", wraps=RendererGL.gl.glTexImage2D) as upload:
                        viewer.log_state(state)
                        self.assertEqual(upload.call_count, 1, "Retry the same texture after a failed upload")
                        viewer.log_state(state)
                        self.assertEqual(upload.call_count, 1, "Reuse the successful retry")
                    restored = center_pixel(state)
                    self.assertGreater(restored[0], max(restored[1:]) + 30, "Render the recovered red texture")

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


@unittest.skipUnless(importlib.util.find_spec("pxr"), "USD is not installed")
class TestDeformableVisualUSD(unittest.TestCase):
    def test_viewer_usd_exports_skinned_mesh_and_textures(self):
        """Export deformed geometry and replacement textures, skipping hidden updates."""
        from PIL import Image
        from pxr import UsdGeom, UsdShade

        builder = _particle_mesh()
        _attach(builder, label="123 bad/skin!", uvs=[[0, 0], [1, 0], [0, 1]])
        model = builder.finalize("cpu")
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.closing(ViewerUSD(os.path.join(directory, "scene.usda"), num_frames=6)) as viewer,
        ):
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
                np.testing.assert_allclose(
                    np.asarray(prim.GetExtentAttr().Get(frame)), np.array([points.min(0), points.max(0)])
                )
            for frame, visible in enumerate((True, False, False, True)):
                self.assertEqual(prim.GetVisibilityAttr().Get(frame), "inherited" if visible else "invisible")
            np.testing.assert_allclose(
                np.asarray(prim.GetPointsAttr().Get(3)), state.particle_q.numpy()[mesh.parent.numpy()]
            )
            for color in ((255, 0, 0), (0, 255, 0)):
                mesh.texture = np.full((2, 2, 3), color, dtype=np.uint8)
                exported = None
                for _ in range(2):
                    viewer.begin_frame(1.0)
                    viewer.log_state(state)
                    viewer.end_frame()
                    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                    shader = UsdShade.Shader(material.GetPrim().GetChild("DiffuseTexture"))
                    texture_path = shader.GetInput("file").Get().path
                    with Image.open(texture_path) as image:
                        np.testing.assert_array_equal(np.asarray(image)[0, 0], color)
                    stat = os.stat(texture_path)
                    if exported is not None:
                        self.assertEqual((stat.st_ino, stat.st_mtime_ns), exported)
                    exported = (stat.st_ino, stat.st_mtime_ns)
            mesh.texture = None
            viewer.log_state(state)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            surface = UsdShade.Shader(material.GetPrim().GetChild("PreviewSurface"))
            self.assertFalse(surface.GetInput("diffuseColor").HasConnectedSource())
            # An ordinary mesh follows the same rule: omitting the texture on a later call removes it,
            # and a new array with the same pixels reuses the exported file.
            image = np.full((2, 2, 3), (0, 0, 255), dtype=np.uint8)
            uvs = wp.array([[0, 0], [1, 0], [0, 1]], dtype=wp.vec2)
            viewer.log_mesh("/plain", mesh.rest_vertices, mesh.indices, uvs=uvs, texture=image)
            plain = viewer.stage.GetPrimAtPath("/root/plain")
            first, _ = UsdShade.MaterialBindingAPI(plain).ComputeBoundMaterial()
            viewer.log_mesh("/plain", mesh.rest_vertices, mesh.indices, uvs=uvs, texture=image.copy())
            second, _ = UsdShade.MaterialBindingAPI(plain).ComputeBoundMaterial()
            self.assertEqual(first.GetPath(), second.GetPath())
            viewer.log_mesh("/plain", mesh.rest_vertices, mesh.indices)
            material, _ = UsdShade.MaterialBindingAPI(plain).ComputeBoundMaterial()
            surface = UsdShade.Shader(material.GetPrim().GetChild("PreviewSurface"))
            self.assertFalse(surface.GetInput("diffuseColor").HasConnectedSource())


if __name__ == "__main__":
    unittest.main()
