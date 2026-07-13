# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
import inspect
import math
import sys
import textwrap
import unittest
import warnings
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import ModelBuilder
from newton._src.geometry.utils import transform_points
from newton._src.solvers.mujoco.equality import _add_equality_constraint
from newton._src.viewer.viewer_file import depointer_as_key, pointer_as_key, transfer_to_model
from newton.tests.unittest_utils import assert_np_equal


def _eq_set_value(builder, name, idx, value):
    """Inject ``value`` at ``idx`` into the equality-constraint custom-attr table (padding with None)."""
    attr = builder.custom_attributes[f"mujoco:{name}"]
    if attr.values is None:
        attr.values = []
    while len(attr.values) <= idx:
        attr.values.append(None)
    attr.values[idx] = value


class TestModelAttributeSpecs(unittest.TestCase):
    def test_attribute_frequencies_have_count_metadata(self):
        model = newton.Model(device="cpu")
        frequency = newton.Model.AttributeFrequency
        expected_count_frequencies = set(frequency).difference({frequency.ONCE})
        actual_count_frequencies = set(model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS)
        self.assertEqual(
            actual_count_frequencies,
            expected_count_frequencies,
            "Keep Model.AttributeFrequency and Model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS in sync. "
            "Add a count-attribute mapping for each new frequency and remove mappings for deleted frequencies.",
        )

        for attribute_frequency, count_attribute in model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS.items():
            with self.subTest(frequency=attribute_frequency):
                self.assertTrue(
                    hasattr(model, count_attribute),
                    f"Model.AttributeFrequency.{attribute_frequency.name} maps to missing attribute "
                    f"Model.{count_attribute}. Add the count attribute or correct "
                    "Model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS.",
                )
                self.assertEqual(
                    model._attribute_frequency_count(attribute_frequency),
                    getattr(model, count_attribute),
                    f"Model._attribute_frequency_count() must resolve Model.AttributeFrequency."
                    f"{attribute_frequency.name} through Model.{count_attribute}.",
                )

    def test_core_attribute_specs_cover_entity_indexed_storage(self):
        model = newton.Model(device="cpu")

        prefixes = tuple(
            count_attribute.removesuffix("count") for count_attribute in model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS.values()
        )
        private_prefixes = tuple(f"_{prefix}" for prefix in prefixes)
        indexed_container_types = (wp.array, np.ndarray, list, dict, set, tuple)
        indexed_attributes = {
            name
            for name, value in model.__dict__.items()
            if name.startswith(prefixes + private_prefixes) and isinstance(value, indexed_container_types)
        }

        # Most Warp arrays are None until finalization, so runtime inspection alone cannot find them.
        init_source = textwrap.dedent(inspect.getsource(newton.Model.__init__))
        init_node = ast.parse(init_source).body[0]
        for node in ast.walk(init_node):
            if not (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"
            ):
                continue
            name = node.target.attr
            annotation = ast.unparse(node.annotation)
            is_indexed_container = any(
                container in annotation for container in ("wp.array", "np.ndarray", "list[", "dict[", "set[", "tuple[")
            )
            if name.startswith(prefixes + private_prefixes) and is_indexed_container:
                indexed_attributes.add(name)

        missing = sorted(indexed_attributes.difference(model.attribute_specs))
        self.assertEqual(
            missing,
            [],
            "Model attributes are missing AttributeSpec metadata. Add each listed attribute to "
            "Model._CORE_ATTRIBUTE_SPECS with the correct frequency, references, row width, and compaction policy.",
        )


class TestModelBuilderDeprecations(unittest.TestCase):
    def test_joint_target_pos_vel_aliases_warn(self):
        """Legacy ``joint_target_pos`` / ``joint_target_vel`` warn under the
        default flag and raise under ``use_coord_layout_targets=True``;
        ``joint_target_q`` / ``joint_target_qd`` are always silent. The Model
        snapshot freezes the flag at construction, so each branch builds its
        own model under the corresponding flag value."""

        def _build_revolute_model():
            builder = ModelBuilder()
            base = builder.add_link(mass=1.0)
            j = builder.add_joint_revolute(parent=-1, child=base, axis=newton.Axis.Z)
            builder.add_articulation([j])
            return builder.finalize()

        prev_flag = newton.use_coord_layout_targets
        try:
            newton.use_coord_layout_targets = False
            model = _build_revolute_model()
            control = newton.Control()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _ = control.joint_target_pos
                _ = control.joint_target_vel
                _ = model.joint_target_pos
                _ = model.joint_target_vel
            deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            self.assertEqual(len(deprecations), 4)
            self.assertTrue(any("Control.joint_target_pos" in str(w.message) for w in deprecations))
            self.assertTrue(any("Control.joint_target_vel" in str(w.message) for w in deprecations))
            self.assertTrue(any("Model.joint_target_pos" in str(w.message) for w in deprecations))
            self.assertTrue(any("Model.joint_target_vel" in str(w.message) for w in deprecations))

            newton.use_coord_layout_targets = True
            model = _build_revolute_model()
            control = newton.Control()
            with self.assertRaises(AttributeError):
                _ = control.joint_target_pos
            with self.assertRaises(AttributeError):
                _ = control.joint_target_vel
            with self.assertRaises(AttributeError):
                _ = model.joint_target_pos
            with self.assertRaises(AttributeError):
                _ = model.joint_target_vel
        finally:
            newton.use_coord_layout_targets = prev_flag

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = control.joint_target_q
            _ = control.joint_target_qd
            _ = model.joint_target_q
            _ = model.joint_target_qd
        self.assertFalse(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_model_builder_joint_target_pos_vel_setters_warn_and_forward(self):
        prev_flag = newton.use_coord_layout_targets
        try:
            newton.use_coord_layout_targets = False

            builder = ModelBuilder()
            inertia = np.eye(3, dtype=np.float32)
            b0 = builder.add_link(mass=1.0, inertia=inertia)
            j_free = builder.add_joint_free(child=b0)
            b1 = builder.add_link(mass=1.0, inertia=inertia)
            j_ball = builder.add_joint_ball(parent=-1, child=b1)
            b2 = builder.add_link(mass=1.0, inertia=inertia)
            j_revolute = builder.add_joint_revolute(parent=b1, child=b2, axis=newton.Axis.Z)
            builder.add_articulation([j_free])
            builder.add_articulation([j_ball, j_revolute])

            target_pos = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7]
            target_vel = [10.0 + i for i in range(builder.joint_dof_count)]

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                builder.joint_target_pos = target_pos
                builder.joint_target_vel = target_vel

            deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            self.assertEqual(len(deprecations), 2)
            self.assertTrue(any("ModelBuilder.joint_target_pos" in str(w.message) for w in deprecations))
            self.assertTrue(any("ModelBuilder.joint_target_vel" in str(w.message) for w in deprecations))

            free_q_start = builder.joint_q_start[j_free]
            ball_q_start = builder.joint_q_start[j_ball]
            self.assertEqual(builder.joint_target_q[free_q_start + 6], 1.0)
            self.assertEqual(builder.joint_target_q[ball_q_start + 3], 1.0)

            model = builder.finalize(skip_all_validations=True)
            np.testing.assert_allclose(model.joint_target_q.numpy(), target_pos, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(model.joint_target_qd.numpy(), target_vel, rtol=0.0, atol=1e-6)

            newton.use_coord_layout_targets = True
            coord_builder = ModelBuilder()
            with self.assertRaises(AttributeError):
                coord_builder.joint_target_pos = []
            with self.assertRaises(AttributeError):
                coord_builder.joint_target_vel = []
        finally:
            newton.use_coord_layout_targets = prev_flag


class TestModelBuilderBvhConstructor(unittest.TestCase):
    def test_model_builder_forwards_bvh_constructors(self):
        builder = ModelBuilder()
        builder.default_bvh_cfg.mesh_constructor = "cubql"
        builder.default_bvh_cfg.gaussian_constructor = "sah"
        builder.default_bvh_cfg.shape_constructor = "lbvh"

        mesh = newton.Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
        )
        gaussian = newton.Gaussian(positions=np.zeros((1, 3), dtype=np.float32))
        builder.add_shape_mesh(body=-1, mesh=mesh)
        builder.add_shape_gaussian(body=-1, gaussian=gaussian)

        with (
            mock.patch("newton._src.geometry.types.wp.Mesh") as wp_mesh,
            mock.patch.object(
                newton.Gaussian, "finalize", autospec=True, return_value=newton.Gaussian.Data()
            ) as finalize,
            mock.patch.object(newton.Model, "bvh_build_shapes", autospec=True) as build_shapes,
            mock.patch.object(newton.Model, "bvh_build_particles", autospec=True),
        ):
            wp_mesh.return_value.id = 123
            model = builder.finalize(device="cpu")

        wp_mesh.assert_called_once()
        self.assertEqual(wp_mesh.call_args.kwargs["bvh_constructor"], "cubql")
        finalize.assert_called_once_with(gaussian, device="cpu", bvh_constructor="sah")
        build_shapes.assert_called_once_with(model, model, bvh_constructor="lbvh")

    def test_gaussian_finalize_forwards_bvh_constructor_to_warp_bvh(self):
        gaussian = newton.Gaussian(
            positions=np.zeros((1, 3), dtype=np.float32),
            rotations=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            scales=np.ones((1, 3), dtype=np.float32),
            opacities=np.ones(1, dtype=np.float32),
            sh_coeffs=np.ones((1, 3), dtype=np.float32),
        )

        with (
            mock.patch("newton._src.geometry.types.wp.launch"),
            mock.patch("newton._src.geometry.types.wp.Bvh") as wp_bvh,
        ):
            wp_bvh.return_value.id = 456
            gaussian.finalize(device="cpu", bvh_constructor="sah")

        self.assertEqual(wp_bvh.call_args.kwargs["constructor"], "sah")


class TestModelMesh(unittest.TestCase):
    def test_empty_numeric_custom_attribute_uses_wp_full_default(self):
        attr = ModelBuilder.CustomAttribute(
            name="default_shape_attr",
            frequency=newton.Model.AttributeFrequency.SHAPE,
            dtype=wp.float32,
            default=3.5,
        )

        with mock.patch.object(wp, "full", wraps=wp.full) as full_mock:
            values = attr.build_array(4, device="cpu")

        full_mock.assert_called_once_with(
            4,
            3.5,
            dtype=wp.float32,
            requires_grad=False,
            device="cpu",
        )
        np.testing.assert_allclose(values.numpy(), np.full(4, 3.5, dtype=np.float32))

    def test_empty_vector_custom_attribute_uses_wp_full_default(self):
        attr = ModelBuilder.CustomAttribute(
            name="default_vector_shape_attr",
            frequency=newton.Model.AttributeFrequency.SHAPE,
            dtype=wp.vec2,
            default=wp.vec2(1.25, -2.5),
        )

        with mock.patch.object(wp, "full", wraps=wp.full) as full_mock:
            values = attr.build_array(3, device="cpu")

        full_mock.assert_called_once_with(
            3,
            wp.vec2(1.25, -2.5),
            dtype=wp.vec2,
            requires_grad=False,
            device="cpu",
        )
        np.testing.assert_allclose(values.numpy(), np.array([[1.25, -2.5]] * 3, dtype=np.float32))

    def test_empty_sequence_custom_attribute_materializes_default_values(self):
        attr = ModelBuilder.CustomAttribute(
            name="default_table_shape_attr",
            frequency=newton.Model.AttributeFrequency.SHAPE,
            dtype=wp.float32,
            default=[0.0, 1.0, 2.0],
        )

        with mock.patch.object(wp, "full", wraps=wp.full) as full_mock:
            values = attr.build_array(2, device="cpu")

        full_mock.assert_not_called()
        np.testing.assert_allclose(values.numpy(), np.array([[0.0, 1.0, 2.0]] * 2, dtype=np.float32))

    def test_mesh_hash_uses_cached_sha_digest(self):
        mesh = newton.Mesh.create_box(
            1.0,
            0.5,
            0.25,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )

        with mock.patch("newton._src.geometry.types.hashlib.sha256", wraps=hashlib.sha256) as sha256_mock:
            first_hash = hash(mesh)
            second_hash = hash(mesh)

            self.assertEqual(first_hash, second_hash)
            sha256_mock.assert_called_once()

            mesh.vertices = mesh.vertices.copy()
            self.assertEqual(hash(mesh), first_hash)
            self.assertEqual(sha256_mock.call_count, 2)

    def test_finalize_deduplicates_equal_mesh_content(self):
        mesh_a = newton.Mesh.create_box(
            1.0,
            0.5,
            0.25,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        mesh_b = newton.Mesh.create_box(
            1.0,
            0.5,
            0.25,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        self.assertIsNot(mesh_a, mesh_b)
        self.assertEqual(hash(mesh_a), hash(mesh_b))

        builder = ModelBuilder()
        builder.add_shape_mesh(body=-1, mesh=mesh_a)
        builder.add_shape_mesh(body=-1, mesh=mesh_b)

        with (
            mock.patch.object(mesh_a, "finalize", wraps=mesh_a.finalize) as finalize_a,
            mock.patch.object(mesh_b, "finalize", wraps=mesh_b.finalize) as finalize_b,
        ):
            model = builder.finalize(device="cpu")

        finalize_a.assert_called_once()
        finalize_b.assert_not_called()
        shape_source_ptr = model.shape_source_ptr.numpy()
        self.assertEqual(shape_source_ptr[0], shape_source_ptr[1])

    def test_finalize_does_not_deduplicate_different_mesh_layouts(self):
        vertices_a = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices_a = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

        # Reinterpret one triangle as an additional vertex. The unframed byte
        # streams are identical even though the resulting meshes are different.
        vertices_b = np.concatenate((vertices_a, indices_a[:3].view(np.float32).reshape(1, 3)))
        indices_b = indices_a[3:]
        self.assertEqual(vertices_a.tobytes() + indices_a.tobytes(), vertices_b.tobytes() + indices_b.tobytes())

        mesh_a = newton.Mesh(vertices_a, indices_a, compute_inertia=False)
        mesh_b = newton.Mesh(vertices_b, indices_b, compute_inertia=False)
        self.assertNotEqual(hash(mesh_a), hash(mesh_b))

        builder = ModelBuilder()
        builder.add_shape_mesh(body=-1, mesh=mesh_a)
        builder.add_shape_mesh(body=-1, mesh=mesh_b)

        with (
            mock.patch.object(mesh_a, "finalize", wraps=mesh_a.finalize) as finalize_a,
            mock.patch.object(mesh_b, "finalize", wraps=mesh_b.finalize) as finalize_b,
        ):
            model = builder.finalize(device="cpu")

        finalize_a.assert_called_once()
        finalize_b.assert_called_once()
        shape_source_ptr = model.shape_source_ptr.numpy()
        self.assertNotEqual(shape_source_ptr[0], shape_source_ptr[1])

    def test_add_triangles(self):
        rng = np.random.default_rng(123)

        pts = np.array(
            [
                [-0.00585869, 0.34189449, -1.17415233],
                [-1.894547, 0.1788074, 0.9251329],
                [-1.26141048, 0.16140787, 0.08823282],
                [-0.08609255, -0.82722546, 0.65995427],
                [0.78827592, -1.77375711, -0.55582718],
            ]
        )
        tris = np.array([[0, 3, 4], [0, 2, 3], [2, 1, 3], [1, 4, 3]])

        builder1 = ModelBuilder()
        builder2 = ModelBuilder()
        for pt in pts:
            builder1.add_particle(wp.vec3(pt), wp.vec3(), 1.0)
            builder2.add_particle(wp.vec3(pt), wp.vec3(), 1.0)

        # test add_triangle(s) with default arguments:
        areas = builder2.add_triangles(tris[:, 0], tris[:, 1], tris[:, 2])
        for i, t in enumerate(tris):
            area = builder1.add_triangle(t[0], t[1], t[2])
            self.assertAlmostEqual(area, areas[i], places=6)

        # test add_triangle(s) with non default arguments:
        tri_ke = rng.standard_normal(size=pts.shape[0])
        tri_ka = rng.standard_normal(size=pts.shape[0])
        tri_kd = rng.standard_normal(size=pts.shape[0])
        tri_drag = rng.standard_normal(size=pts.shape[0])
        tri_lift = rng.standard_normal(size=pts.shape[0])
        for i, t in enumerate(tris):
            builder1.add_triangle(
                t[0],
                t[1],
                t[2],
                tri_ke=tri_ke[i],
                tri_ka=tri_ka[i],
                tri_kd=tri_kd[i],
                tri_drag=tri_drag[i],
                tri_lift=tri_lift[i],
            )
        builder2.add_triangles(
            tris[:, 0],
            tris[:, 1],
            tris[:, 2],
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            tri_drag=tri_drag,
            tri_lift=tri_lift,
        )

        assert_np_equal(np.array(builder1.tri_indices), np.array(builder2.tri_indices))
        assert_np_equal(np.array(builder1.tri_poses), np.array(builder2.tri_poses), tol=1.0e-6)
        assert_np_equal(np.array(builder1.tri_activations), np.array(builder2.tri_activations))
        assert_np_equal(np.array(builder1.tri_materials), np.array(builder2.tri_materials))

    def test_add_edges(self):
        rng = np.random.default_rng(123)

        pts = np.array(
            [
                [-0.00585869, 0.34189449, -1.17415233],
                [-1.894547, 0.1788074, 0.9251329],
                [-1.26141048, 0.16140787, 0.08823282],
                [-0.08609255, -0.82722546, 0.65995427],
                [0.78827592, -1.77375711, -0.55582718],
            ]
        )
        edges = np.array([[0, 4, 3, 1], [3, 2, 4, 1]])

        builder1 = ModelBuilder()
        builder2 = ModelBuilder()
        for pt in pts:
            builder1.add_particle(wp.vec3(pt), wp.vec3(), 1.0)
            builder2.add_particle(wp.vec3(pt), wp.vec3(), 1.0)

        # test defaults:
        for i in range(2):
            builder1.add_edge(edges[i, 0], edges[i, 1], edges[i, 2], edges[i, 3])
        builder2.add_edges(edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3])

        # test non defaults:
        rest = rng.standard_normal(size=2)
        edge_ke = rng.standard_normal(size=2)
        edge_kd = rng.standard_normal(size=2)
        for i in range(2):
            builder1.add_edge(
                edges[i, 0],
                edges[i, 1],
                edges[i, 2],
                edges[i, 3],
                rest=rest[i],
                edge_ke=edge_ke[i],
                edge_kd=edge_kd[i],
            )
        builder2.add_edges(
            edges[:, 0],
            edges[:, 1],
            edges[:, 2],
            edges[:, 3],
            rest=rest,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
        )

        assert_np_equal(np.array(builder1.edge_indices), np.array(builder2.edge_indices))
        assert_np_equal(np.array(builder1.edge_rest_angle), np.array(builder2.edge_rest_angle), tol=1.0e-4)
        assert_np_equal(np.array(builder1.edge_bending_properties), np.array(builder2.edge_bending_properties))

    def test_soft_mesh_adjacency_from_cloth_mesh(self):
        builder = ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(1.0, 1.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
            ],
            indices=[0, 1, 2, 0, 2, 3],
            density=1.0,
        )

        # The adjacency is built in finalize() from the accumulated edges and triangles.
        model = builder.finalize(device="cpu")
        adjacency = model.soft_mesh_adjacency
        self.assertIsNotNone(adjacency)
        np.testing.assert_array_equal(
            adjacency.tri_edge_indices,
            np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            adjacency.edge_tri_indices,
            np.array([[0, -1], [0, -1], [0, 1], [1, -1], [1, -1]], dtype=np.int32),
        )
        # Vertex adjacency is now built eagerly in finalize (init_vertex_adjacency), so the shared
        # device copy is ready for every consumer.
        self.assertIsNotNone(adjacency.v_adj_tris)
        self.assertIsNotNone(adjacency.v_adj_edges_offsets)

    def test_manual_soft_mesh_adjacency_placeholders_finalize(self):
        builder = ModelBuilder()
        builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(), 1.0)
        builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(), 1.0)
        builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(), 1.0)
        builder.add_triangle(0, 1, 2)
        builder.add_edge(-1, -1, 0, 1)

        # A bare triangle and a placeholder edge (o0 == o1 == -1) stay unlinked: the triangle's
        # opposite vertex matches neither stored opposite, so finalize leaves both map rows at -1.
        model = builder.finalize(device="cpu")
        adjacency = model.soft_mesh_adjacency
        self.assertIsNotNone(adjacency)
        np.testing.assert_array_equal(adjacency.tri_edge_indices, np.array([[-1, -1, -1]], dtype=np.int32))
        np.testing.assert_array_equal(adjacency.edge_tri_indices, np.array([[-1, -1]], dtype=np.int32))

    def test_add_builder_offsets_soft_mesh_adjacency(self):
        base = ModelBuilder()
        base.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(1.0, 1.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
            ],
            indices=[0, 1, 2, 0, 2, 3],
            density=1.0,
        )

        combined = ModelBuilder()
        combined.add_builder(base)
        combined.add_builder(base)

        # add_builder concatenates the edge/triangle tables; finalize() rebuilds the maps, so the
        # second copy's rows are the first's with triangle ids +2 and edge ids +5.
        base_adj = base.finalize(device="cpu").soft_mesh_adjacency
        combined_adj = combined.finalize(device="cpu").soft_mesh_adjacency

        np.testing.assert_array_equal(combined_adj.tri_edge_indices[:2], base_adj.tri_edge_indices)
        np.testing.assert_array_equal(combined_adj.edge_tri_indices[:5], base_adj.edge_tri_indices)
        np.testing.assert_array_equal(combined_adj.tri_edge_indices[2:], base_adj.tri_edge_indices + 5)
        np.testing.assert_array_equal(
            combined_adj.edge_tri_indices[5:],
            np.array([[2, -1], [2, -1], [2, 3], [3, -1], [3, -1]], dtype=np.int32),
        )

    def test_soft_mesh_adjacency_mixes_cloth_and_bare_triangles(self):
        # A cloth mesh (with bending edges) plus a bare add_triangle (no edges) in one builder:
        # finalize() sizes tri_edge_indices to every triangle, leaves the bare triangle's row at -1,
        # keeps the cloth rows linked, and never synthesizes edges for the bare triangle.
        builder = ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(1.0, 1.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
            ],
            indices=[0, 1, 2, 0, 2, 3],
            density=1.0,
        )
        p0 = builder.add_particle(wp.vec3(2.0, 0.0, 0.0), wp.vec3(), 1.0)
        p1 = builder.add_particle(wp.vec3(3.0, 0.0, 0.0), wp.vec3(), 1.0)
        p2 = builder.add_particle(wp.vec3(2.0, 1.0, 0.0), wp.vec3(), 1.0)
        builder.add_triangle(p0, p1, p2)

        adjacency = builder.finalize(device="cpu").soft_mesh_adjacency
        np.testing.assert_array_equal(
            adjacency.tri_edge_indices,
            np.array([[0, 1, 2], [2, 3, 4], [-1, -1, -1]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            adjacency.edge_tri_indices,
            np.array([[0, -1], [0, -1], [0, 1], [1, -1], [1, -1]], dtype=np.int32),
        )

    def test_mesh_adjacency_public_deprecated(self):
        tris = [[0, 1, 2], [0, 2, 3]]
        # Construction from triangle indices is supported (no warning) and eager.
        adj = newton.utils.MeshAdjacency(tris)
        self.assertEqual(adj.edge_indices.shape, (5, 4))
        self.assertEqual(adj.edge_tri_indices.shape, (5, 2))
        self.assertEqual(adj.tri_edge_indices.shape, (2, 3))
        # The legacy .edges dict stays available but is deprecated.
        with self.assertWarns(DeprecationWarning):
            edges = adj.edges
        self.assertEqual(len(edges), 5)
        shared = edges[(0, 2)]
        self.assertEqual({shared.f0, shared.f1}, {0, 1})

    def test_mesh_adjacency_indices_deprecated_alias(self):
        tris = [[0, 1, 2], [0, 2, 3]]
        # `indices` is a deprecated alias for `tri_indices` and builds the same tables.
        with self.assertWarns(DeprecationWarning):
            adj = newton.utils.MeshAdjacency(indices=tris)
        np.testing.assert_array_equal(adj.edge_indices, newton.utils.MeshAdjacency(tri_indices=tris).edge_indices)
        # Passing both names with conflicting values is rejected.
        with self.assertRaises(ValueError):
            newton.utils.MeshAdjacency(tri_indices=tris, indices=[[0, 1, 2]])

    def test_mesh_adjacency_add_edge_deprecated(self):
        adj = newton.utils.MeshAdjacency()
        # add_edge is a deprecated incremental shim; it updates edge_indices / edge_tri_indices.
        with self.assertWarns(DeprecationWarning):
            adj.add_edge(0, 1, 2, 0)
        self.assertEqual(adj.edge_indices.shape, (1, 4))
        self.assertEqual(adj.edge_tri_indices.shape, (1, 2))
        with self.assertWarns(DeprecationWarning):
            adj.add_edge(1, 0, 3, 1)  # second adjacent triangle (endpoints reversed)
        np.testing.assert_array_equal(adj.edge_indices[0], [2, 3, 0, 1])
        np.testing.assert_array_equal(adj.edge_tri_indices[0], [0, 1])
        with self.assertWarns(DeprecationWarning):
            edges = adj.edges
        self.assertEqual({edges[(0, 1)].f0, edges[(0, 1)].f1}, {0, 1})

    def test_mesh_adjacency_to_without_vertex_adjacency_warns(self):
        # to() before init_vertex_adjacency: uploads the topology maps, leaves v_adj_* None + warns.
        adj = newton.utils.MeshAdjacency([[0, 1, 2], [0, 2, 3]])
        with self.assertWarns(UserWarning):
            data = adj.to("cpu")
        self.assertIsNotNone(data.edge_tri_indices)
        self.assertIsNotNone(data.tri_edge_indices)
        self.assertIsNone(data.v_adj_tris)
        self.assertIsNone(data.v_adj_edges_offsets)

    def test_mesh_adjacency_owns_index_copies(self):
        # The constructor stores owned int32 copies, detached from the input arrays/lists.
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        adj = newton.utils.MeshAdjacency(tris)
        tris[0, 0] = 99
        self.assertEqual(int(adj.indices[0, 0]), 0)

    def test_expand_edge_parameter(self):
        expand = newton.ModelBuilder._expand_edge_parameter
        # Scalars broadcast to one value per generated edge.
        self.assertEqual(expand(2.0, 3), [2.0, 2.0, 2.0])
        # A 0-D array is treated as a scalar, not iterated.
        self.assertEqual(expand(np.array(2.0), 3), [2.0, 2.0, 2.0])
        # A per-edge sequence of matching length passes through.
        self.assertEqual(expand([1.0, 2.0, 3.0], 3), [1.0, 2.0, 3.0])
        # None is preserved so add_edges() can substitute its default.
        self.assertIsNone(expand(None, 3))
        # A length mismatch is rejected instead of silently desyncing.
        with self.assertRaises(ValueError):
            expand([1.0, 2.0], 3)

    def test_mesh_approximation(self):
        def box_mesh(scale=(1.0, 1.0, 1.0), transform: wp.transform | None = None):
            mesh = newton.Mesh.create_box(
                scale[0],
                scale[1],
                scale[2],
                duplicate_vertices=False,
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
            vertices, indices = mesh.vertices, mesh.indices
            if transform is not None:
                vertices = transform_points(vertices, transform)
            return newton.Mesh(vertices, indices)

        def npsorted(x):
            return np.array(sorted(x))

        builder = ModelBuilder()
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="approx_attr",
                frequency=newton.Model.AttributeFrequency.SHAPE,
                dtype=wp.float32,
            )
        )
        tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
        scale = wp.vec3(1.0, 3.0, 0.2)
        mesh = box_mesh(scale=scale, transform=tf)
        mesh.maxhullvert = 5
        s0 = builder.add_shape_mesh(body=-1, mesh=mesh)
        s1 = builder.add_shape_mesh(body=-1, mesh=mesh)
        s2 = builder.add_shape_mesh(body=-1, mesh=mesh)
        builder.approximate_meshes(method="convex_hull", shape_indices=[s0])
        builder.approximate_meshes(method="bounding_box", shape_indices=[s1])
        builder.approximate_meshes(method="bounding_sphere", shape_indices=[s2])
        # convex hull
        self.assertEqual(len(builder.shape_source[s0].vertices), 5)
        self.assertEqual(builder.shape_type[s0], newton.GeoType.CONVEX_MESH)
        # the convex hull maintains the original transform
        assert_np_equal(np.array(builder.shape_transform[s0]), np.array(wp.transform_identity()), tol=1.0e-4)
        # bounding box
        self.assertIsNone(builder.shape_source[s1])
        self.assertEqual(builder.shape_type[s1], newton.GeoType.BOX)
        assert_np_equal(npsorted(builder.shape_scale[s1]), npsorted(scale), tol=1.0e-5)
        # only compare the position since the rotation is not guaranteed to be the same
        assert_np_equal(np.array(builder.shape_transform[s1].p), np.array(tf.p), tol=1.0e-4)
        # bounding sphere
        self.assertIsNone(builder.shape_source[s2])
        self.assertEqual(builder.shape_type[s2], newton.GeoType.SPHERE)
        self.assertAlmostEqual(builder.shape_scale[s2][0], wp.length(scale))
        assert_np_equal(np.array(builder.shape_transform[s2]), np.array(tf), tol=1.0e-4)

        # test keep_visual_shapes
        keep_visual_color = (0.1, 0.2, 0.3)
        keep_visual_attr = 1.25
        s3 = builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            color=keep_visual_color,
            label="mesh_keep_visual",
            custom_attributes={"approx_attr": keep_visual_attr},
        )
        builder.approximate_meshes(method="convex_hull", shape_indices=[s3], keep_visual_shapes=True)
        # approximation is created, but not visible
        self.assertEqual(len(builder.shape_source[s3].vertices), 5)
        self.assertEqual(builder.shape_type[s3], newton.GeoType.CONVEX_MESH)
        self.assertEqual(builder.shape_flags[s3] & newton.ShapeFlags.VISIBLE, 0)
        # a new visual shape is created
        visual_shape = s3 + 1
        self.assertIs(builder.shape_source[visual_shape], mesh)
        self.assertEqual(builder.shape_flags[visual_shape] & newton.ShapeFlags.VISIBLE, newton.ShapeFlags.VISIBLE)
        self.assertEqual(builder.shape_label[visual_shape], "mesh_keep_visual_visual")
        np.testing.assert_allclose(
            np.asarray(builder.shape_color[visual_shape], dtype=np.float32),
            keep_visual_color,
            atol=1e-6,
            rtol=1e-6,
        )

        # make sure the original mesh is not modified
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.indices), 36)

        model = builder.finalize(device="cpu")
        self.assertAlmostEqual(model.approx_attr.numpy()[visual_shape], keep_visual_attr, places=6)

    def test_mesh_approximation_convex_decomposition_preserves_visual_properties(self):
        builder = ModelBuilder()
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="approx_attr",
                frequency=newton.Model.AttributeFrequency.SHAPE,
                dtype=wp.float32,
            )
        )
        mesh = newton.Mesh.create_box(
            1.0,
            1.0,
            1.0,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        shape_color = (0.7, 0.2, 0.9)
        shape_label = "mesh_decomp"
        shape_attr = 2.5
        shape = builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            color=shape_color,
            label=shape_label,
            custom_attributes={"approx_attr": shape_attr},
        )

        class FakeCoacdMesh:
            def __init__(self, vertices, faces):
                self.vertices = vertices
                self.faces = faces

        fake_coacd = SimpleNamespace(
            Mesh=FakeCoacdMesh,
            run_coacd=lambda _mesh, **_kwargs: [
                (mesh.vertices.copy(), mesh.indices.copy()),
                (mesh.vertices.copy(), mesh.indices.copy()),
            ],
        )

        with mock.patch.dict(sys.modules, {"coacd": fake_coacd}):
            builder.approximate_meshes(method="coacd", shape_indices=[shape], raise_on_failure=True)

        extra_shape = shape + 1
        self.assertEqual(builder.shape_type[extra_shape], newton.GeoType.CONVEX_MESH)
        self.assertEqual(builder.shape_label[extra_shape], f"{shape_label}_convex_1")
        np.testing.assert_allclose(
            np.asarray(builder.shape_color[extra_shape], dtype=np.float32),
            shape_color,
            atol=1e-6,
            rtol=1e-6,
        )

        model = builder.finalize(device="cpu")
        self.assertAlmostEqual(model.approx_attr.numpy()[extra_shape], shape_attr, places=6)

    def test_approximate_meshes_collision_filter_child_bodies(self):
        def normalize_pair(a, b):
            return (min(a, b), max(a, b))

        def get_filter_set(builder):
            return {normalize_pair(a, b) for a, b in builder.shape_collision_filter_pairs}

        builder = ModelBuilder()

        # Create a chain of 3 bodies (like an articulation)
        body0 = builder.add_link()
        body1 = builder.add_link()
        body2 = builder.add_link()

        # Add initial shapes to each body (like mesh shapes before decomposition)
        shape0_initial = builder.add_shape_sphere(body=body0, radius=0.1)
        shape1_initial = builder.add_shape_sphere(body=body1, radius=0.1)
        shape2_initial = builder.add_shape_sphere(body=body2, radius=0.1)

        # Create joints (establishes parent->child relationships)
        # body0 is parent of body1, body1 is parent of body2
        joint_free = builder.add_joint_free(parent=-1, child=body0)
        joint0 = builder.add_joint_revolute(parent=body0, child=body1, axis=(0, 0, 1))
        joint1 = builder.add_joint_revolute(parent=body1, child=body2, axis=(0, 0, 1))
        builder.add_articulation(joints=[joint_free, joint0, joint1])

        # At this point, initial shapes should be filtered between adjacent bodies
        filter_set = get_filter_set(builder)
        self.assertIn(
            normalize_pair(shape0_initial, shape1_initial),
            filter_set,
            "Initial body0-body1 shapes should be filtered",
        )
        self.assertIn(
            normalize_pair(shape1_initial, shape2_initial),
            filter_set,
            "Initial body1-body2 shapes should be filtered",
        )

        # Now simulate what approximate_meshes() does: add additional shapes to bodies
        # after joints are already created (like convex decomposition adding multiple parts)
        shape0_extra1 = builder.add_shape_box(body=body0, hx=0.1, hy=0.1, hz=0.1)
        shape0_extra2 = builder.add_shape_capsule(body=body0, radius=0.05, half_height=0.1)
        shape1_extra1 = builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1)

        filter_set = get_filter_set(builder)

        # Verify: new body0 shapes should filter with ALL body1 shapes (including initial)
        for parent_shape in [shape0_extra1, shape0_extra2]:
            for child_shape in [shape1_initial, shape1_extra1]:
                expected_pair = normalize_pair(parent_shape, child_shape)
                self.assertIn(
                    expected_pair,
                    filter_set,
                    f"New parent body0 shape {parent_shape} should filter with body1 shape {child_shape}",
                )

        # Verify: new body1 shapes should filter with ALL body0 shapes (parent)
        for child_shape in [shape1_extra1]:
            for parent_shape in [shape0_initial, shape0_extra1, shape0_extra2]:
                expected_pair = normalize_pair(parent_shape, child_shape)
                self.assertIn(
                    expected_pair,
                    filter_set,
                    f"New body1 shape {child_shape} should filter with parent body0 shape {parent_shape}",
                )

        # Verify: new body1 shapes should filter with ALL body2 shapes (child)
        for parent_shape in [shape1_extra1]:
            expected_pair = normalize_pair(parent_shape, shape2_initial)
            self.assertIn(
                expected_pair,
                filter_set,
                f"New body1 shape {parent_shape} should filter with child body2 shape {shape2_initial}",
            )

    def test_shape_gap_negative_warning(self):
        """Test that a warning is raised when shape gap < 0."""
        builder = ModelBuilder()
        body = builder.add_body(mass=1.0)

        # Create a shape with negative gap (should trigger warning)
        cfg = ModelBuilder.ShapeConfig()
        cfg.margin = 0.01
        cfg.gap = -0.005  # Negative gap
        builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg, label="bad_sphere")

        # Should warn about gap < 0
        with self.assertWarns(UserWarning) as cm:
            builder.finalize()

        warning_msg = str(cm.warning)
        self.assertIn("gap < 0", warning_msg)
        self.assertIn("bad_sphere", warning_msg)
        self.assertIn("missed collisions", warning_msg)

    def test_shape_gap_non_negative_no_warning(self):
        """Test that no warning is raised when shape gap >= 0."""
        builder = ModelBuilder()
        body = builder.add_body(mass=1.0)

        # Create a shape with non-negative gap (should not trigger warning)
        cfg = ModelBuilder.ShapeConfig()
        cfg.margin = 0.005
        cfg.gap = 0.01
        builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg)

        # Should NOT warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.finalize()
            gap_warnings = [warning for warning in w if "gap < 0" in str(warning.message)]
            self.assertEqual(len(gap_warnings), 0, "Unexpected warning about gap < 0")

    def test_shape_gap_warning_multiple_shapes(self):
        """Test that the warning correctly reports multiple shapes with gap < 0."""
        builder = ModelBuilder()
        body = builder.add_body(mass=1.0)

        # Create multiple shapes with negative gap
        cfg_bad = ModelBuilder.ShapeConfig()
        cfg_bad.margin = 0.02
        cfg_bad.gap = -0.01

        builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg_bad, label="sphere1")
        builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg_bad, label="box1")

        # One good shape that should not be in the warning
        cfg_good = ModelBuilder.ShapeConfig()
        cfg_good.margin = 0.005
        cfg_good.gap = 0.01
        builder.add_shape_capsule(body=body, radius=0.2, half_height=0.5, cfg=cfg_good, label="good_capsule")

        with self.assertWarns(UserWarning) as cm:
            builder.finalize()

        warning_msg = str(cm.warning)
        self.assertIn("2 shape(s)", warning_msg)
        self.assertIn("sphere1", warning_msg)
        self.assertIn("box1", warning_msg)
        self.assertNotIn("good_capsule", warning_msg)

    def test_collision_filter_pairs_canonical_order(self):
        """Test that collision filter pairs are stored in canonical order (s1 < s2)."""
        builder = ModelBuilder()

        # Create a body with multiple shapes
        body = builder.add_body()
        shape0 = builder.add_shape_sphere(body=body, radius=0.5)
        shape1 = builder.add_shape_box(body=body, hx=1.0, hy=1.0, hz=1.0)
        shape2 = builder.add_shape_capsule(body=body, radius=0.3, half_height=1.0)

        # Add collision filter pairs in non-canonical order to test normalization
        builder.shape_collision_filter_pairs.append((shape1, shape0))  # reversed order
        builder.shape_collision_filter_pairs.append((shape0, shape2))  # correct order
        builder.shape_collision_filter_pairs.append((shape2, shape1))  # reversed order

        # Finalize the model
        model = builder.finalize()

        # Verify all collision filter pairs are in canonical order (s1 < s2)
        for s1, s2 in model.shape_collision_filter_pairs:
            self.assertLess(s1, s2, f"Collision filter pair ({s1}, {s2}) is not in canonical order")

        # Verify we have the expected pairs (should be normalized to canonical order)
        self.assertIn((shape0, shape1), model.shape_collision_filter_pairs)
        self.assertIn((shape0, shape2), model.shape_collision_filter_pairs)
        self.assertIn((shape1, shape2), model.shape_collision_filter_pairs)

    def test_large_replicated_collision_filter_pairs_deprecate_mutation_and_preserve_contacts(self):
        """Large replicated filters should stay compact while finalized-model mutation warns."""

        robot = ModelBuilder()
        body0 = robot.add_body()
        shape0 = robot.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = robot.add_body()
        shape1 = robot.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        robot.shape_collision_filter_pairs.append((shape0, shape1))

        builder = ModelBuilder()
        ground = builder.add_ground_plane()
        builder.replicate(robot, 3)

        builder_filters = builder._shape_collision_filter_pairs  # pyright: ignore[reportPrivateUsage]
        self.assertNotIsInstance(builder_filters, list)
        self.assertEqual(list(builder_filters), [(1, 2), (3, 4), (5, 6)])

        model = builder.finalize()

        internal_filters = model._shape_collision_filter_store()  # pyright: ignore[reportPrivateUsage]
        self.assertFalse(internal_filters.is_materialized)
        self.assertTrue(internal_filters.contains_pair(1, 2))
        self.assertFalse(internal_filters.is_materialized)

        filters = model.shape_collision_filter_pairs
        self.assertIsInstance(filters, set)
        self.assertTrue(internal_filters.is_materialized)
        self.assertIn((1, 2), filters)
        self.assertIn((3, 4), filters)
        self.assertIn((5, 6), filters)
        expected_filters = {(1, 2), (3, 4), (5, 6)}
        self.assertEqual(filters, expected_filters)
        self.assertEqual(filters | {(ground, 1)}, expected_filters | {(ground, 1)})
        with self.assertWarns(DeprecationWarning):
            filters.add((ground, 1))
        self.assertIn((ground, 1), model.shape_collision_filter_pairs)
        with self.assertWarns(DeprecationWarning):
            model.shape_collision_filter_pairs = set()
        self.assertEqual(model.shape_collision_filter_pairs, set())

        shape_contact_pairs = model.shape_contact_pairs
        assert shape_contact_pairs is not None
        contact_pairs = {tuple(pair) for pair in shape_contact_pairs.numpy()}
        self.assertEqual(contact_pairs, {(ground, 1), (ground, 2), (ground, 3), (ground, 4), (ground, 5), (ground, 6)})

    def test_collision_filter_in_place_mutation_warns_at_call_site(self):
        def union(model: newton.Model) -> None:
            model.shape_collision_filter_pairs |= {(0, 1)}

        def intersection(model: newton.Model) -> None:
            model.shape_collision_filter_pairs &= {(0, 1)}

        def difference(model: newton.Model) -> None:
            model.shape_collision_filter_pairs -= {(0, 1)}

        def symmetric_difference(model: newton.Model) -> None:
            model.shape_collision_filter_pairs ^= {(0, 1)}

        for mutation in (union, intersection, difference, symmetric_difference):
            with self.subTest(mutation=mutation.__name__):
                model = ModelBuilder().finalize(device="cpu")
                filters = model.shape_collision_filter_pairs
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", DeprecationWarning)
                    mutation(model)

                self.assertEqual(len(caught), 1)
                self.assertEqual(caught[0].filename, __file__)
                self.assertEqual(caught[0].lineno, mutation.__code__.co_firstlineno + 1)
                self.assertIs(model.shape_collision_filter_pairs, filters)

    def test_builder_collision_filter_pairs_preserve_list_api(self):
        robot = ModelBuilder()
        body0 = robot.add_body()
        shape0 = robot.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = robot.add_body()
        shape1 = robot.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        robot.add_shape_collision_filter_pair(shape0, shape1)

        builder = ModelBuilder()
        builder.replicate(robot, 2)
        builder.add_shape_collision_filter_pair(0, 2)

        filters = builder.shape_collision_filter_pairs
        self.assertIsInstance(filters, list)
        self.assertEqual(filters, [(0, 1), (2, 3), (0, 2)])
        self.assertEqual(filters.copy(), filters)
        self.assertEqual(filters + [(1, 3)], [(0, 1), (2, 3), (0, 2), (1, 3)])  # noqa: RUF005

    def test_builder_collision_filter_pairs_accept_reassigned_lists(self):
        source = ModelBuilder()
        body0 = source.add_body()
        shape0 = source.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = source.add_body()
        shape1 = source.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        source_filters = [(shape0, shape1)]
        source.shape_collision_filter_pairs = source_filters
        self.assertIs(source.shape_collision_filter_pairs, source_filters)

        builder = ModelBuilder()
        destination_filters: list[tuple[int, int]] = []
        builder.shape_collision_filter_pairs = destination_filters
        builder.add_builder(source)
        self.assertIs(builder.shape_collision_filter_pairs, destination_filters)
        self.assertEqual(destination_filters, [(shape0, shape1)])

        model = builder.finalize()
        self.assertEqual(model.shape_collision_filter_pairs, {(shape0, shape1)})

    def test_add_builder_collision_filter_template_cache_tracks_mutations(self):
        """Source-builder filter cache should invalidate when pair contents change."""

        source = ModelBuilder()
        body0 = source.add_body()
        shape0 = source.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = source.add_body()
        shape1 = source.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        body2 = source.add_body()
        shape2 = source.add_shape_box(body=body2, hx=0.5, hy=0.5, hz=0.5)
        source.add_shape_collision_filter_pair(shape0, shape1)

        builder = ModelBuilder()
        builder.add_builder(source)
        source.add_shape_collision_filter_pair(shape0, shape2)
        builder.add_builder(source)

        self.assertIn((0, 1), builder.shape_collision_filter_pairs)
        self.assertIn((3, 4), builder.shape_collision_filter_pairs)
        self.assertIn((3, 5), builder.shape_collision_filter_pairs)

    def test_compact_replicated_collision_filters_allow_residual_filters(self):
        """Residual global filters should work with compact contact-pair generation."""

        robot = ModelBuilder()
        body0 = robot.add_body()
        shape0 = robot.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = robot.add_body()
        shape1 = robot.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        robot.shape_collision_filter_pairs.append((shape0, shape1))

        builder = ModelBuilder()
        ground = builder.add_ground_plane()
        builder.replicate(robot, 3)

        # Match robot examples that add one non-block global/local filter per
        # replicated world. The compact block path should handle these residual
        # filters while generating contact pairs.
        builder.add_shape_collision_filter_pair(ground, 1)
        builder.add_shape_collision_filter_pair(ground, 3)
        builder.add_shape_collision_filter_pair(ground, 5)

        with mock.patch.object(
            builder,
            "_build_shape_collision_filter_packed",
            wraps=builder._build_shape_collision_filter_packed,  # pyright: ignore[reportPrivateUsage]
        ) as build_filters:
            model = builder.finalize()
        build_filters.assert_called_once()

        filters = model.shape_collision_filter_pairs
        self.assertIsInstance(filters, set)
        self.assertIn((1, 2), filters)
        self.assertIn((ground, 1), filters)

        shape_contact_pairs = model.shape_contact_pairs
        assert shape_contact_pairs is not None
        contact_pairs = {tuple(pair) for pair in shape_contact_pairs.numpy()}
        self.assertEqual(contact_pairs, {(ground, 2), (ground, 4), (ground, 6)})

        builder.shape_collision_filter_pairs.append((ground, 2))
        self.assertNotIn((ground, 2), filters)

    def test_compact_replicated_collision_filters_roundtrip_viewer_file(self):
        """ViewerFile should restore compact filters through a native public set."""

        robot = ModelBuilder()
        body0 = robot.add_body()
        shape0 = robot.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = robot.add_body()
        shape1 = robot.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        robot.shape_collision_filter_pairs.append((shape0, shape1))

        builder = ModelBuilder()
        ground = builder.add_ground_plane()
        builder.replicate(robot, 2)
        builder.add_shape_collision_filter_pair(ground, 1)

        model = builder.finalize(device="cpu")
        expected_filters = {(1, 2), (3, 4), (ground, 1)}
        internal_filters = model._shape_collision_filter_store()  # pyright: ignore[reportPrivateUsage]
        self.assertFalse(internal_filters.is_materialized)

        serialized = cast(Mapping[str, Any], pointer_as_key({"model": model}, format_type="json"))
        self.assertTrue(internal_filters.is_materialized)
        deserialized = depointer_as_key(serialized, format_type="json")
        deserialized_model = cast(Mapping[str, Any], cast(Mapping[str, Any], deserialized)["model"])
        restored_model = newton.Model(device="cpu")
        transfer_to_model(deserialized_model, restored_model)

        self.assertIsInstance(restored_model.shape_collision_filter_pairs, set)
        self.assertEqual(restored_model.shape_collision_filter_pairs, expected_filters)

    def test_collision_filter_array_queries_match_set(self):
        """Packed-array membership and broad-phase pairs must match the public set."""

        robot = ModelBuilder()
        body0 = robot.add_body()
        shape0 = robot.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = robot.add_body()
        shape1 = robot.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        robot.add_shape_collision_filter_pair(shape0, shape1)

        builder = ModelBuilder()
        ground = builder.add_ground_plane()
        builder.replicate(robot, 4)
        builder.add_shape_collision_filter_pair(ground, 1)

        model = builder.finalize()

        broad_phase_pairs = model.shape_collision_filter_pairs_array()
        self.assertEqual(broad_phase_pairs.shape, (5, 2))
        pair_list = [tuple(pair) for pair in broad_phase_pairs.tolist()]
        self.assertEqual(pair_list, sorted(pair_list))

        internal_filters = model._shape_collision_filter_store()  # pyright: ignore[reportPrivateUsage]
        assert internal_filters is not None
        self.assertFalse(internal_filters.is_materialized)

        self.assertTrue(model.shape_collision_filter_contains(1, 2))
        self.assertTrue(model.shape_collision_filter_contains(2, 1))
        self.assertTrue(model.shape_collision_filter_contains(ground, 1))
        self.assertFalse(model.shape_collision_filter_contains(ground, 2))
        # NumPy integer indices (e.g. from .numpy() arrays) must not overflow
        # the packed pair code.
        self.assertTrue(model.shape_collision_filter_contains(np.int32(1), np.int32(2)))
        self.assertTrue(model.shape_collision_filter_contains(1, np.int32(2)))
        with self.assertRaises(TypeError):
            model.shape_collision_filter_contains("1", 2)  # pyright: ignore[reportArgumentType]
        with self.assertRaises(TypeError):
            model.shape_collision_filter_contains(1.0, 2)  # pyright: ignore[reportArgumentType]
        self.assertFalse(internal_filters.is_materialized)

        # The canonical array aliases internal state and must be read-only.
        with self.assertRaises(ValueError):
            broad_phase_pairs[0, 0] = 5

        candidates = np.array([[1, 2], [2, 1], [ground, 1], [ground, 2], [3, 4]], dtype=np.int32)
        mask = model.shape_collision_filter_mask(candidates)
        self.assertEqual(mask.tolist(), [True, True, True, False, True])
        unsigned_mask = model.shape_collision_filter_mask(np.array([[1, 2], [ground, 2]], dtype=np.uint32))
        self.assertEqual(unsigned_mask.tolist(), [True, False])
        with self.assertRaises(TypeError):
            model.shape_collision_filter_mask(candidates.astype(np.float64))
        with self.assertRaises(TypeError):
            model.shape_collision_filter_mask(candidates.astype(str))

        self.assertEqual(set(pair_list), set(model.shape_collision_filter_pairs))

        # Rebuilding through the public method must use the model as the filter
        # source even if this builder has changed since finalization.
        builder.add_shape_collision_filter_pair(ground, 2)
        with self.assertWarnsRegex(DeprecationWarning, "generated automatically"):
            builder.find_shape_contact_pairs(model)
        shape_contact_pairs = model.shape_contact_pairs
        assert shape_contact_pairs is not None
        contact_pairs = {tuple(pair) for pair in shape_contact_pairs.numpy()}
        self.assertIn((ground, 2), contact_pairs)

        # After deprecated mutation, queries fall back to native set semantics.
        with self.assertWarns(DeprecationWarning):
            model.shape_collision_filter_pairs.add((ground, 2))
        self.assertTrue(model.shape_collision_filter_contains(ground, 2))
        mask = model.shape_collision_filter_mask(candidates)
        self.assertEqual(mask.tolist(), [True, True, True, True, True])
        self.assertEqual(len(model.shape_collision_filter_pairs_array()), 6)

        # Rebuilding contact pairs after a (deprecated) mutation must honor
        # the mutated model store rather than replaying stale builder filters.
        with self.assertWarnsRegex(DeprecationWarning, "generated automatically"):
            builder.find_shape_contact_pairs(model)
        shape_contact_pairs = model.shape_contact_pairs
        assert shape_contact_pairs is not None
        contact_pairs = {tuple(pair) for pair in shape_contact_pairs.numpy()}
        self.assertNotIn((ground, 2), contact_pairs)

    def test_mixed_replicated_and_global_builder_filters_preserve_contacts(self):
        """Blocks without a world (global add_builder) must not disable the fast path."""

        robot = ModelBuilder()
        body0 = robot.add_body()
        shape0 = robot.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = robot.add_body()
        shape1 = robot.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        robot.add_shape_collision_filter_pair(shape0, shape1)

        builder = ModelBuilder()
        builder.add_builder(robot)  # global world: filter block without world assignment
        builder.replicate(robot, 2)

        model = builder.finalize()

        self.assertEqual(set(model.shape_collision_filter_pairs), {(0, 1), (2, 3), (4, 5)})
        contact_pairs = {tuple(pair) for pair in model.shape_contact_pairs.numpy()}
        expected = {(g, s) for g in (0, 1) for s in (2, 3, 4, 5)}
        self.assertEqual(contact_pairs, expected)

    def test_collision_filter_pairs_reject_invalid_shape_indices(self):
        """Invalid filters should fail consistently before contact generation."""

        builder = ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
        builder.shape_collision_filter_pairs.append((shape, builder.shape_count))

        with self.assertRaisesRegex(ValueError, "shape_collision_filter_pairs contains invalid pair"):
            builder.finalize()

    def test_compact_collision_filter_residuals_reject_invalid_shape_indices(self):
        """Compact residual filters should raise ValueError instead of raw IndexError."""

        robot = ModelBuilder()
        body0 = robot.add_body()
        shape0 = robot.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = robot.add_body()
        shape1 = robot.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        robot.shape_collision_filter_pairs.append((shape0, shape1))

        builder = ModelBuilder()
        ground = builder.add_ground_plane()
        builder.replicate(robot, 3)
        builder.add_shape_collision_filter_pair(ground, builder.shape_count)

        with self.assertRaisesRegex(ValueError, "shape_collision_filter_pairs contains invalid pair"):
            builder.finalize()

    def test_compact_collision_filter_blocks_materialize_before_mutation(self):
        """Public list mutation should not retain stale compact block metadata."""

        robot = ModelBuilder()
        body0 = robot.add_body()
        shape0 = robot.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        body1 = robot.add_body()
        shape1 = robot.add_shape_box(body=body1, hx=0.5, hy=0.5, hz=0.5)
        robot.shape_collision_filter_pairs.append((shape0, shape1))

        builder = ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(robot, 3)
        builder.shape_collision_filter_pairs[0] = (1, 3)

        model = builder.finalize()

        self.assertIsInstance(model.shape_collision_filter_pairs, set)

        contact_pairs = {tuple(pair) for pair in model.shape_contact_pairs.numpy()}
        self.assertIn((1, 2), contact_pairs)
        self.assertEqual(len(contact_pairs), 7)

    def test_collision_filter_fixed_to_world(self):
        """Bodies fixed to world via add_joint_fixed(parent=-1) should auto-filter
        their shapes against world-static shapes regardless of construction order
        (issue #2201)."""

        def joint_first():
            b = ModelBuilder()
            body = b.add_link()
            b.add_joint_fixed(parent=-1, child=body)
            mesh_shape = b.add_shape_sphere(body=body, radius=0.5)
            ground_shape = b.add_ground_plane()
            return b, mesh_shape, ground_shape

        def world_shape_first():
            b = ModelBuilder()
            ground_shape = b.add_ground_plane()
            body = b.add_link()
            b.add_joint_fixed(parent=-1, child=body)
            mesh_shape = b.add_shape_sphere(body=body, radius=0.5)
            return b, mesh_shape, ground_shape

        def body_shape_first():
            b = ModelBuilder()
            body = b.add_link()
            mesh_shape = b.add_shape_sphere(body=body, radius=0.5)
            b.add_joint_fixed(parent=-1, child=body)
            ground_shape = b.add_ground_plane()
            return b, mesh_shape, ground_shape

        for case_name, build in (
            ("joint before shapes", joint_first),
            ("world shape before joint", world_shape_first),
            ("body shape before joint", body_shape_first),
        ):
            with self.subTest(case=case_name):
                builder, mesh_shape, ground_shape = build()
                pair = (min(mesh_shape, ground_shape), max(mesh_shape, ground_shape))
                self.assertEqual(builder.shape_collision_filter_pairs.count(pair), 1)

    def test_collision_filter_floating_base_not_filtered(self):
        """Floating-base bodies (FREE joint to world) must NOT be filtered against
        world shapes — they need to be able to land on the ground."""

        builder = ModelBuilder()
        body = builder.add_link()
        builder.add_joint_free(parent=-1, child=body)
        base_shape = builder.add_shape_sphere(body=body, radius=0.5)
        ground_shape = builder.add_ground_plane()
        pair = (min(base_shape, ground_shape), max(base_shape, ground_shape))
        self.assertNotIn(pair, builder.shape_collision_filter_pairs)

    def test_collision_filter_revolute_to_world_default(self):
        """A revolute (non-fixed) joint to world does NOT auto-filter child shapes
        against world shapes — the child needs to be able to collide with world
        geometry (e.g. a pendulum hitting the ground)."""

        builder = ModelBuilder()
        body = builder.add_link()
        builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        body_shape = builder.add_shape_sphere(body=body, radius=0.5)
        ground_shape = builder.add_ground_plane()
        pair = (min(body_shape, ground_shape), max(body_shape, ground_shape))
        self.assertNotIn(pair, builder.shape_collision_filter_pairs)

    def test_collision_filter_revolute_to_world_explicit(self):
        """Explicit collision_filter_parent=True is honored even for a non-fixed
        joint to world (overrides the smart default) when shapes exist at
        joint-creation time."""

        builder = ModelBuilder()
        body = builder.add_link()
        body_shape = builder.add_shape_sphere(body=body, radius=0.5)
        ground_shape = builder.add_ground_plane()
        builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z, collision_filter_parent=True)
        pair = (min(body_shape, ground_shape), max(body_shape, ground_shape))
        self.assertEqual(builder.shape_collision_filter_pairs.count(pair), 1)

    def test_collision_filter_free_with_real_parent_default_filtered(self):
        """A free joint between two real bodies auto-filters parent/child shape pairs by
        default, matching the legacy behavior for joints between real bodies."""

        builder = ModelBuilder()
        parent = builder.add_link()
        child = builder.add_link()
        parent_shape = builder.add_shape_sphere(body=parent, radius=0.5)
        child_shape = builder.add_shape_sphere(body=child, radius=0.5)
        builder.add_joint_free(parent=parent, child=child)
        pair = (min(parent_shape, child_shape), max(parent_shape, child_shape))
        self.assertEqual(builder.shape_collision_filter_pairs.count(pair), 1)

    def test_collision_filter_fixed_to_world_opt_out(self):
        """collision_filter_parent=False on the joint suppresses the auto-filter
        when shapes already exist on both sides at joint-creation time."""

        builder = ModelBuilder()
        body = builder.add_link()
        mesh_shape = builder.add_shape_sphere(body=body, radius=0.5)
        ground_shape = builder.add_ground_plane()
        builder.add_joint_fixed(parent=-1, child=body, collision_filter_parent=False)
        pair = (min(mesh_shape, ground_shape), max(mesh_shape, ground_shape))
        self.assertNotIn(pair, builder.shape_collision_filter_pairs)

    def test_validate_structure_invalid_shape_body(self):
        """Test that _validate_structure catches invalid shape_body references."""
        builder = ModelBuilder()
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body=body, radius=0.5, label="test_shape")

        # Manually set invalid body reference
        builder.shape_body[0] = 999  # Invalid body index

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid body reference", error_msg)
        self.assertIn("shape_body", error_msg)
        self.assertIn("test_shape", error_msg)
        self.assertIn("999", error_msg)


class TestModelJoints(unittest.TestCase):
    def test_add_builder_xform_updates_root_free_joint_coordinates(self):
        parent_xform = wp.transform(wp.vec3(0.4, -0.2, 0.1), wp.quat_rpy(0.3, -0.4, 0.2))
        child_xform = wp.transform(wp.vec3(-0.1, 0.3, 0.2), wp.quat_rpy(-0.2, 0.1, 0.4))
        body_xform = wp.transform(wp.vec3(1.0, -2.0, 0.5), wp.quat_rpy(0.1, 0.2, -0.3))
        offset = wp.transform(wp.vec3(-0.5, 0.7, 1.2), wp.quat_rpy(-0.3, 0.2, 0.1))

        source = ModelBuilder()
        body = source.add_link(xform=body_xform)
        joint = source.add_joint_free(
            child=body,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
        source.add_articulation([joint])

        builder = ModelBuilder()
        builder.add_builder(source, xform=offset)

        expected_body_xform = offset * body_xform
        expected_joint_q = wp.transform_inverse(parent_xform) * expected_body_xform * child_xform
        q_start = builder.joint_q_start[joint]
        assert_np_equal(np.array(builder.joint_X_p[joint]), np.array(parent_xform), tol=1.0e-6)
        assert_np_equal(
            np.array(builder.joint_q[q_start : q_start + 7]),
            np.array(expected_joint_q),
            tol=1.0e-6,
        )

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        assert_np_equal(state.body_q.numpy()[body], np.array(expected_body_xform), tol=1.0e-5)

    def test_add_builder_xform_preserves_parented_free_joint_coordinates(self):
        parent_body_xform = wp.transform(wp.vec3(0.5, -0.4, 0.2), wp.quat_rpy(0.2, -0.1, 0.3))
        child_body_xform = wp.transform(wp.vec3(-0.6, 0.8, 1.1), wp.quat_rpy(-0.3, 0.4, -0.2))
        root_parent_xform = wp.transform(wp.vec3(0.1, 0.2, -0.3), wp.quat_rpy(0.1, 0.3, -0.2))
        parent_xform = wp.transform(wp.vec3(-0.2, 0.5, 0.1), wp.quat_rpy(-0.2, 0.1, 0.4))
        child_xform = wp.transform(wp.vec3(0.3, -0.1, 0.2), wp.quat_rpy(0.3, -0.4, 0.1))
        offset = wp.transform(wp.vec3(1.0, -0.5, 0.7), wp.quat_rpy(0.4, 0.2, -0.3))

        source = ModelBuilder()
        parent = source.add_link(xform=parent_body_xform)
        child = source.add_link(xform=child_body_xform)
        root_joint = source.add_joint_free(child=parent, parent_xform=root_parent_xform)
        child_joint = source.add_joint_free(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
        source.add_articulation([root_joint, child_joint])

        q_start = source.joint_q_start[child_joint]
        expected_joint_q = np.array(source.joint_q[q_start : q_start + 7])

        builder = ModelBuilder()
        builder.add_builder(source, xform=offset)

        q_start = builder.joint_q_start[child_joint]
        assert_np_equal(np.array(builder.joint_X_p[child_joint]), np.array(parent_xform), tol=1.0e-6)
        assert_np_equal(np.array(builder.joint_q[q_start : q_start + 7]), expected_joint_q, tol=1.0e-6)

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        assert_np_equal(state.body_q.numpy()[parent], np.array(offset * parent_body_xform), tol=1.0e-5)
        assert_np_equal(state.body_q.numpy()[child], np.array(offset * child_body_xform), tol=1.0e-5)

    def test_add_joint_free_initializes_relative_transform(self):
        parent_body_xform = wp.transform(wp.vec3(1.0, -2.0, 0.5), wp.quat_rpy(0.2, -0.3, 0.4))
        child_body_xform = wp.transform(wp.vec3(-0.5, 1.5, 2.0), wp.quat_rpy(-0.4, 0.1, 0.3))
        parent_xform = wp.transform(wp.vec3(0.3, -0.2, 0.1), wp.quat_rpy(0.1, 0.2, -0.1))
        child_xform = wp.transform(wp.vec3(-0.1, 0.4, 0.2), wp.quat_rpy(-0.2, 0.3, 0.1))

        builder = ModelBuilder()
        parent = builder.add_link(xform=parent_body_xform)
        child = builder.add_link(xform=child_body_xform)
        joint = builder.add_joint_free(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
        builder.add_articulation([joint])

        parent_anchor_world = parent_body_xform * parent_xform
        expected_joint_q = wp.transform_inverse(parent_anchor_world) * child_body_xform * child_xform
        q_start = builder.joint_q_start[joint]
        assert_np_equal(
            np.array(builder.joint_q[q_start : q_start + 7]),
            np.array(expected_joint_q),
            tol=1.0e-6,
        )

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        assert_np_equal(state.body_q.numpy()[child], np.array(child_body_xform), tol=1.0e-5)

    def test_joint_target_q_qd_shape_with_free_and_ball_joints(self):
        """``joint_target_q`` follows ``joint_q`` (coord) under
        ``use_coord_layout_targets``; ``joint_target_qd`` always follows
        ``joint_qd`` (DOF). Free and ball joints are where the two layouts
        diverge. Multi-articulation builder also exercises the per-env start
        arrays."""
        for use_coord in (False, True):
            prev = newton.use_coord_layout_targets
            newton.use_coord_layout_targets = use_coord
            try:
                builder = ModelBuilder()
                # env 0: free + revolute (7 coords / 6 DOFs from free)
                b0 = builder.add_link(mass=1.0)
                j0_free = builder.add_joint_free(child=b0)
                b1 = builder.add_link(mass=1.0)
                j0_rev = builder.add_joint_revolute(parent=b0, child=b1, axis=newton.Axis.Z)
                builder.add_articulation([j0_free, j0_rev])
                # env 1: ball + revolute (4 coords / 3 DOFs from ball)
                b2 = builder.add_link(mass=1.0)
                j1_ball = builder.add_joint_ball(parent=-1, child=b2)
                b3 = builder.add_link(mass=1.0)
                j1_rev = builder.add_joint_revolute(parent=b2, child=b3, axis=newton.Axis.Z)
                builder.add_articulation([j1_ball, j1_rev])
                model = builder.finalize()

                self.assertEqual(model.joint_dof_count, 7 + 4)
                self.assertEqual(model.joint_coord_count, 8 + 5)

                target_q_size = model.joint_coord_count if use_coord else model.joint_dof_count
                self.assertEqual(model.joint_target_q.shape[0], target_q_size)
                self.assertEqual(model.joint_target_qd.shape[0], model.joint_dof_count)

                control = model.control()
                self.assertEqual(control.joint_target_q.shape[0], target_q_size)
                self.assertEqual(control.joint_target_qd.shape[0], model.joint_dof_count)

                expected_start = model.joint_q_start.numpy() if use_coord else model.joint_qd_start.numpy()
                np.testing.assert_array_equal(model.joint_target_q_start.numpy(), expected_start)

                if use_coord:
                    target_q = model.joint_target_q.numpy()
                    q_starts = model.joint_q_start.numpy()
                    # env 0 free joint: w-component at offset 6 (3 lin + 3 quat-xyz)
                    self.assertAlmostEqual(float(target_q[int(q_starts[0]) + 6]), 1.0)
                    # env 1 ball joint: w-component at offset 3 (3 quat-xyz)
                    self.assertAlmostEqual(float(target_q[int(q_starts[2]) + 3]), 1.0)
            finally:
                newton.use_coord_layout_targets = prev

    def test_ball_free_per_axis_target_pos_preserved(self):
        """``JointDofConfig.target_pos`` on BALL/FREE angular axes must flow
        into ``joint_target_q`` under both flag values.

        - Flag=False (legacy DOF): the 3 angular scalars are projected verbatim
          into the DOF slice (matching the pre-coord-layout behavior).
        - Flag=True (coord): the 3 angular scalars are interpreted as extrinsic
          ZYX Euler angles and converted to a unit quaternion via
          :meth:`ModelBuilder._quat_from_euler_zyx`, matching kamino's
          DOF→coord conversion.
        """
        ang_targets = (0.1, 0.2, -0.3)

        def _make_axes():
            return [
                ModelBuilder.JointDofConfig(axis=newton.Axis.X, target_pos=ang_targets[0]),
                ModelBuilder.JointDofConfig(axis=newton.Axis.Y, target_pos=ang_targets[1]),
                ModelBuilder.JointDofConfig(axis=newton.Axis.Z, target_pos=ang_targets[2]),
            ]

        lin_targets = (1.5, -2.5, 3.5)

        def _make_linear_axes():
            return [
                ModelBuilder.JointDofConfig(axis=newton.Axis.X, target_pos=lin_targets[0]),
                ModelBuilder.JointDofConfig(axis=newton.Axis.Y, target_pos=lin_targets[1]),
                ModelBuilder.JointDofConfig(axis=newton.Axis.Z, target_pos=lin_targets[2]),
            ]

        expected_quat = ModelBuilder._quat_from_axis_targets(*ang_targets)

        for use_coord in (False, True):
            prev = newton.use_coord_layout_targets
            newton.use_coord_layout_targets = use_coord
            try:
                builder = ModelBuilder()
                # BALL via low-level add_joint with per-axis targets
                b_ball = builder.add_link(mass=1.0)
                j_ball = builder.add_joint(
                    newton.JointType.BALL,
                    parent=-1,
                    child=b_ball,
                    angular_axes=_make_axes(),
                )
                # FREE via low-level add_joint with per-axis linear+angular targets
                b_free = builder.add_link(mass=1.0)
                j_free = builder.add_joint(
                    newton.JointType.FREE,
                    parent=-1,
                    child=b_free,
                    linear_axes=_make_linear_axes(),
                    angular_axes=_make_axes(),
                )
                builder.add_articulation([j_ball])
                builder.add_articulation([j_free])
                model = builder.finalize()

                target_q = model.joint_target_q.numpy()

                if use_coord:
                    # BALL coord slice = (qx, qy, qz, qw) — full unit quaternion
                    q_starts = model.joint_q_start.numpy()
                    b = int(q_starts[j_ball])
                    np.testing.assert_allclose(target_q[b : b + 4], expected_quat, rtol=0, atol=1e-6)
                    # FREE coord slice = (px, py, pz, qx, qy, qz, qw)
                    f = int(q_starts[j_free])
                    np.testing.assert_allclose(target_q[f : f + 3], lin_targets, rtol=0, atol=1e-6)
                    np.testing.assert_allclose(target_q[f + 3 : f + 7], expected_quat, rtol=0, atol=1e-6)
                    # Verify unit norm (would only hold post-conversion)
                    self.assertAlmostEqual(float(np.linalg.norm(target_q[b : b + 4])), 1.0, places=5)
                    self.assertAlmostEqual(float(np.linalg.norm(target_q[f + 3 : f + 7])), 1.0, places=5)
                else:
                    # DOF projection: BALL → 3 raw angular floats; FREE → 3 lin + 3 raw ang
                    qd_starts = model.joint_qd_start.numpy()
                    b = int(qd_starts[j_ball])
                    np.testing.assert_allclose(target_q[b : b + 3], ang_targets, rtol=0, atol=1e-6)
                    f = int(qd_starts[j_free])
                    np.testing.assert_allclose(target_q[f : f + 3], lin_targets, rtol=0, atol=1e-6)
                    np.testing.assert_allclose(target_q[f + 3 : f + 6], ang_targets, rtol=0, atol=1e-6)
            finally:
                newton.use_coord_layout_targets = prev

    def test_collapse_keeps_attachment_anchored_rod_joints(self):
        """collapse_fixed_joints must not delete non-fixed joints: a rod anchored
        mid-chain by a ball joint (the USD attachment pattern) keeps every cable
        joint even though the anchor makes the chain a loop."""
        builder = newton.ModelBuilder()
        pts = [wp.vec3(0.1 * i, 0.0, 1.0) for i in range(4)]
        bodies, _joints = builder.add_rod(
            positions=pts, radius=0.02, label="cable", wrap_in_articulation=True, body_frame_origin="com"
        )
        builder.add_joint_ball(parent=-1, child=bodies[1], label="att")
        labels_before = sorted(builder.joint_label)
        builder.collapse_fixed_joints()
        self.assertEqual(sorted(builder.joint_label), labels_before)
        # Topological joint ordering survives: every joint's parent index is below its child.
        for j in range(builder.joint_count):
            parent, child = builder.joint_parent[j], builder.joint_child[j]
            if parent >= 0:
                self.assertLess(parent, child, builder.joint_label[j])

    def test_collapse_keeps_parallel_joints(self):
        """Two joints between the same body pair (e.g. an attachment with two point
        sites) both survive collapse."""
        builder = newton.ModelBuilder()
        pts = [wp.vec3(0.1 * i, 0.0, 1.0) for i in range(4)]
        bodies, _joints = builder.add_rod(
            positions=pts, radius=0.02, label="cable", wrap_in_articulation=True, body_frame_origin="com"
        )
        builder.add_joint_ball(parent=-1, child=bodies[1], label="att_a")
        builder.add_joint_ball(parent=-1, child=bodies[1], label="att_b")
        count_before = builder.joint_count
        builder.collapse_fixed_joints()
        self.assertEqual(builder.joint_count, count_before)

    def test_collapse_parallel_joints_with_fixed_ordering(self):
        """Parallel joints between one body pair survive collapse regardless of which
        joint comes first; a fixed joint among them still merges the pair, and the
        surviving non-fixed joint is remapped onto the merged body."""
        for order in ("fixed_first", "fixed_second", "fixed_kept_first"):
            with self.subTest(order=order):
                builder = newton.ModelBuilder()
                p = builder.add_body(label="parent")
                c = builder.add_body(label="child")
                builder.add_shape_sphere(p, radius=0.1)
                builder.add_shape_sphere(c, radius=0.1)
                if order == "fixed_second":
                    builder.add_joint_ball(parent=p, child=c, label="ball")
                    builder.add_joint_fixed(parent=p, child=c, label="fix")
                else:
                    builder.add_joint_fixed(parent=p, child=c, label="fix")
                    builder.add_joint_ball(parent=p, child=c, label="ball")
                keep = ["fix"] if order == "fixed_kept_first" else []
                builder.collapse_fixed_joints(joints_to_keep=keep)
                labels = list(builder.joint_label)
                # add_body() gives each body a free joint; only assert on ours.
                if order == "fixed_kept_first":
                    # Nothing merged: both parallel joints survive.
                    self.assertIn("fix", labels)
                    self.assertIn("ball", labels)
                    self.assertEqual(builder.body_count, 2)
                else:
                    # The fixed pair merged (or the redundant fixed loop joint dropped);
                    # the ball joint survives with valid endpoints.
                    self.assertIn("ball", labels)
                for j in range(builder.joint_count):
                    self.assertLess(builder.joint_child[j], builder.body_count)

    def test_collapse_reindexes_bodies_in_original_order(self):
        """Retained bodies keep their original relative order after collapse, so an
        anchor joint reaching a rod mid-chain cannot scramble recorded body ranges."""
        builder = newton.ModelBuilder()
        # A rigid pair joined by a fixed joint: something real to collapse.
        b0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), label="base")
        b1 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), label="tool")
        builder.add_shape_sphere(b0, radius=0.1)
        builder.add_shape_sphere(b1, radius=0.1)
        builder.add_joint_free(b0)
        builder.add_joint_fixed(b0, b1)
        pts = [wp.vec3(0.1 * i, 0.0, 1.0) for i in range(4)]
        bodies, joints = builder.add_rod(
            positions=pts, radius=0.02, label="cable", wrap_in_articulation=True, body_frame_origin="com"
        )
        # Record the group the way the USD importer does, so the range remap is exercised.
        builder._record_cable_group("cable", (bodies[0], bodies[-1] + 1), (joints[0], joints[-1] + 1))
        builder.add_joint_ball(parent=-1, child=bodies[-1], label="att")
        cable_labels_before = [builder.body_label[b] for b in bodies]
        builder.collapse_fixed_joints()
        # The fixed pair merged into one body; the cable bodies stay contiguous and ordered.
        start, end = builder._cable_body_start[0], builder._cable_body_end[0]
        self.assertEqual(end - start, len(bodies))
        self.assertEqual([builder.body_label[b] for b in range(start, end)], cable_labels_before)

    def test_collapse_fixed_joints(self):
        shape_cfg = ModelBuilder.ShapeConfig(density=1.0)

        def add_three_cubes(builder: ModelBuilder, parent_body=-1):
            unit_cube = {"hx": 0.5, "hy": 0.5, "hz": 0.5, "cfg": shape_cfg}
            b0 = builder.add_link()
            builder.add_shape_box(body=b0, **unit_cube)
            j0 = builder.add_joint_fixed(
                parent=parent_body, child=b0, parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0))
            )
            b1 = builder.add_link()
            builder.add_shape_box(body=b1, **unit_cube)
            j1 = builder.add_joint_fixed(
                parent=parent_body, child=b1, parent_xform=wp.transform(wp.vec3(0.0, 1.0, 0.0))
            )
            b2 = builder.add_link()
            builder.add_shape_box(body=b2, **unit_cube)
            j2 = builder.add_joint_fixed(
                parent=parent_body, child=b2, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0))
            )
            return b2, [j0, j1, j2]

        builder = ModelBuilder()
        # only fixed joints
        last_body, joints = add_three_cubes(builder)
        builder.add_articulation(joints)
        assert builder.joint_count == 3
        assert builder.body_count == 3

        # fixed joints followed by a non-fixed joint
        last_body, joints = add_three_cubes(builder)
        assert builder.joint_count == 6
        assert builder.body_count == 6
        assert builder.articulation_count == 1  # Only one articulation created so far
        b3 = builder.add_link()
        builder.add_shape_box(
            body=b3, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg, xform=wp.transform(wp.vec3(1.0, 2.0, 3.0))
        )
        joints.append(builder.add_joint_revolute(parent=last_body, child=b3, axis=wp.vec3(0.0, 1.0, 0.0)))
        builder.add_articulation(joints)
        assert builder.articulation_count == 2  # Now we have two articulations

        # a non-fixed joint followed by fixed joints
        free_xform = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_rpy(0.4, 0.5, 0.6))
        free_parent_xform = wp.transform(wp.vec3(0.0, -1.0, 0.0))
        b4 = builder.add_link(xform=free_xform)
        builder.add_shape_box(body=b4, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
        j_free = builder.add_joint_free(parent=-1, child=b4, parent_xform=free_parent_xform)
        assert_np_equal(builder.body_q[b4], np.array(free_xform))
        expected_joint_q = wp.transform_inverse(free_parent_xform) * free_xform
        assert_np_equal(builder.joint_q[-7:], np.array(expected_joint_q))
        assert builder.joint_count == 8
        assert builder.body_count == 8
        _last_body2, joints2 = add_three_cubes(builder, parent_body=b4)
        all_joints = [j_free, *joints2]
        builder.add_articulation(all_joints)
        assert builder.articulation_count == 3  # Three articulations total

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="articulation_name",
                dtype=str,
                frequency=newton.Model.AttributeFrequency.ARTICULATION,
                default="",
                values={0: "fixed", 1: "revolute", 2: "free"},
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="articulation_ref",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.ONCE,
                references="articulation",
                default=-1,
                values={0: [1, (2, -1)]},
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="articulation_ref_wp",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.ONCE,
                references="articulation",
                default=wp.int32(-1),
                values={0: [wp.int32(1), (wp.int32(2), wp.int32(-1))]},
            )
        )

        collapse_results = builder.collapse_fixed_joints()

        assert builder.joint_count == 2
        assert builder.articulation_count == 2
        assert collapse_results["articulation_remap"] == {1: 0, 2: 1}
        assert builder.articulation_start == [0, 1]
        assert builder.articulation_label == ["articulation_1", "articulation_2"]
        assert builder.articulation_world == [-1, -1]
        assert builder.joint_articulation == [0, 1]
        assert builder.custom_attributes["articulation_name"].values == {0: "revolute", 1: "free"}
        assert builder.custom_attributes["articulation_ref"].values == {0: [0, (1, -1)]}
        assert builder.custom_attributes["articulation_ref_wp"].values == {0: [0, (1, -1)]}
        assert builder.joint_type == [newton.JointType.REVOLUTE, newton.JointType.FREE]
        assert builder.shape_count == 11
        assert builder.shape_body == [-1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1]
        assert builder.body_count == 2
        assert builder.body_com[0] == wp.vec3(1.0, 2.0, 3.0)
        assert builder.body_com[1] == wp.vec3(0.25, 0.25, 0.25)
        assert builder.body_mass == [1.0, 4.0]
        assert builder.body_inv_mass == [1.0, 0.25]

        # create another builder, test add_builder function
        builder2 = ModelBuilder()
        builder2.add_builder(builder)
        assert builder2.articulation_count == builder.articulation_count
        assert builder2.joint_count == builder.joint_count
        assert builder2.body_count == builder.body_count
        assert builder2.shape_count == builder.shape_count
        assert builder2.articulation_start == builder.articulation_start
        # add the same builder again
        builder2.add_builder(builder)
        assert builder2.articulation_count == 2 * builder.articulation_count
        assert builder2.articulation_start == [0, 1, 2, 3]

    def test_collapse_fixed_joints_remaps_custom_body_and_joint_references(self):
        # A custom attribute declaring references="body"/"joint" must have its indices remapped
        # when fixed joints are collapsed, just like the built-in arrays. This is the generic path
        # that also covers MuJoCo tendons and equality constraints (which reference joints/bodies).
        shape_cfg = ModelBuilder.ShapeConfig(density=1.0)
        builder = ModelBuilder()

        # b0 is fixed to the world -> collapsed away (its body and joint are removed).
        b0 = builder.add_link()
        builder.add_shape_box(body=b0, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
        j_fixed = builder.add_joint_fixed(parent=-1, child=b0)

        # b1 has a revolute joint -> survives; its body and joint indices shift down by one.
        b1 = builder.add_link()
        builder.add_shape_box(body=b1, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
        j_rev = builder.add_joint_revolute(parent=-1, child=b1, axis=wp.vec3(0.0, 1.0, 0.0))
        builder.add_articulation([j_fixed, j_rev])

        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="thing", namespace="test"))
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ref_body",
                dtype=wp.int32,
                frequency="test:thing",
                namespace="test",
                references="body",
                default=-1,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ref_joint",
                dtype=wp.int32,
                frequency="test:thing",
                namespace="test",
                references="joint",
                default=-1,
            )
        )
        # Row 0 references the survivors; row 1 references the collapsed-away body and joint.
        builder.add_custom_values(**{"test:ref_body": b1, "test:ref_joint": j_rev})
        builder.add_custom_values(**{"test:ref_body": b0, "test:ref_joint": j_fixed})

        builder.collapse_fixed_joints()

        # Survivors remap to their new indices; references to removed entities collapse to -1.
        self.assertEqual(builder.custom_attributes["test:ref_body"].values, [0, -1])
        self.assertEqual(builder.custom_attributes["test:ref_joint"].values, [0, -1])

    def test_collapse_fixed_joints_with_locked_inertia(self):
        builder = ModelBuilder()
        b0 = builder.add_link(mass=1.0, lock_inertia=True)
        j0 = builder.add_joint_free(b0)
        b1 = builder.add_link(mass=2.0, lock_inertia=True)
        j1 = builder.add_joint_fixed(parent=b0, child=b1)
        builder.add_articulation([j0, j1])

        builder.collapse_fixed_joints()

        self.assertEqual(builder.body_count, 1)
        self.assertAlmostEqual(builder.body_mass[0], 3.0)
        self.assertTrue(builder.body_lock_inertia[0])

    def test_collapse_fixed_joints_massless_chain(self):
        """Collapsing a chain of massless bodies into a positive-mass body must yield a finite center of mass."""
        for use_articulation in (True, False):
            with self.subTest(use_articulation=use_articulation):
                builder = ModelBuilder()
                root = builder.add_link(mass=0.0, label="massless_root")
                dummy = builder.add_link(mass=0.0, label="massless_dummy")
                mass_body = builder.add_link(mass=2.0, com=wp.vec3(1.0, 0.0, 0.0), label="mass_body")

                joints = [
                    builder.add_joint_free(parent=-1, child=root, label="floating_base"),
                    builder.add_joint_fixed(parent=root, child=dummy, label="root_to_dummy"),
                    builder.add_joint_fixed(parent=dummy, child=mass_body, label="dummy_to_mass_body"),
                ]

                if use_articulation:
                    builder.add_articulation(joints)
                builder.collapse_fixed_joints()

                self.assertEqual(builder.body_count, 1)
                self.assertAlmostEqual(builder.body_mass[0], 2.0)
                assert_np_equal(np.array(builder.body_com[0]), np.array([1.0, 0.0, 0.0]))

    def test_collapse_fixed_joints_with_groups(self):
        """Test that collapse_fixed_joints correctly preserves world groups."""
        # Optionally enable debug printing
        verbose = False  # Set to True to enable debug output

        # Create builder with multiple worlds and fixed joints
        builder = ModelBuilder()

        # World 0: Chain with fixed joints
        builder.begin_world()
        b0_0 = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        b0_1 = builder.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        b0_2 = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)

        # Connect to world so collapse_fixed_joints processes this chain
        j0_0 = builder.add_joint_revolute(
            parent=-1,
            child=b0_0,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0.0, 0.0, 1.0),
        )

        # Add fixed joint (will be collapsed)
        j0_1 = builder.add_joint_fixed(
            parent=b0_0, child=b0_1, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
        )

        # Add revolute joint (will be retained)
        j0_2 = builder.add_joint_revolute(
            parent=b0_1,
            child=b0_2,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0.0, 1.0, 0.0),
        )
        # Create articulation for world 0
        builder.add_articulation([j0_0, j0_1, j0_2])

        builder.end_world()

        # World 1: Another chain
        builder.begin_world()
        b1_0 = builder.add_link(xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()), mass=1.0)
        b1_1 = builder.add_link(xform=wp.transform(wp.vec3(1.0, 2.0, 0.0), wp.quat_identity()), mass=1.0)

        # Connect to world
        j1_0 = builder.add_joint_revolute(
            parent=-1,
            child=b1_0,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(1.0, 0.0, 0.0),
        )

        # Add revolute joint
        j1_1 = builder.add_joint_revolute(
            parent=b1_0,
            child=b1_1,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0.0, 0.0, 1.0),
        )

        # Create articulation for world 1
        builder.add_articulation([j1_0, j1_1])

        builder.end_world()

        # Global body (connected to world via free joint)
        # Using add_body for a standalone body with free joint
        builder.add_body(xform=wp.transform(wp.vec3(0.0, -5.0, 0.0), wp.quat_identity()), mass=0.0)

        # Check worlds before collapse
        self.assertEqual(builder.body_world, [0, 0, 0, 1, 1, -1])
        self.assertEqual(builder.joint_world, [0, 0, 0, 1, 1, -1])  # 6 joints now (includes free joint from add_body)

        # Collapse fixed joints
        builder.collapse_fixed_joints(verbose=verbose)

        # After collapse:
        # - b0_0 and b0_1 are merged (b0_1 removed)
        # - Fixed joint is removed
        # - Remaining bodies: b0_0 (merged), b0_2, b1_0, b1_1, global_body
        # - Note: global_body is now retained because it's connected to world via free joint
        # - Remaining joints: world->b0_0, b0_0->b0_2, world->b1_0, b1_0->b1_1, world->global_body (free joint)

        self.assertEqual(builder.body_count, 5)  # One body removed (b0_1 merged)
        self.assertEqual(builder.joint_count, 5)  # One joint removed (fixed joint)

        # Check that groups are preserved correctly
        self.assertEqual(builder.body_world, [0, 0, 1, 1, -1])  # Groups preserved for retained bodies
        self.assertEqual(builder.joint_world, [0, 0, 1, 1, -1])  # Groups preserved for retained joints

        # Finalize and verify
        model = builder.finalize()
        body_groups = model.body_world.numpy()
        joint_worlds = model.joint_world.numpy()

        # Verify body groups
        self.assertEqual(body_groups[0], 0)  # Merged b0_0
        self.assertEqual(body_groups[1], 0)  # b0_2
        self.assertEqual(body_groups[2], 1)  # b1_0
        self.assertEqual(body_groups[3], 1)  # b1_1

        # Verify joint groups (world connections and body-to-body joints)
        self.assertEqual(joint_worlds[0], 0)  # world->b0_0 from world 0
        self.assertEqual(joint_worlds[1], 0)  # b0_0->b0_2 from world 0
        self.assertEqual(joint_worlds[2], 1)  # world->b1_0 from world 1
        self.assertEqual(joint_worlds[3], 1)  # b1_0->b1_1 from world 1

        # Verify world start indices
        particle_world_start = model.particle_world_start.numpy() if model.particle_world_start is not None else []
        body_world_start = model.body_world_start.numpy() if model.body_world_start is not None else []
        shape_world_start = model.shape_world_start.numpy() if model.shape_world_start is not None else []
        joint_world_start = model.joint_world_start.numpy() if model.joint_world_start is not None else []
        articulation_world_start = (
            model.articulation_world_start.numpy() if model.articulation_world_start is not None else []
        )
        joint_dof_world_start = model.joint_dof_world_start.numpy() if model.joint_dof_world_start is not None else []
        joint_coord_world_start = (
            model.joint_coord_world_start.numpy() if model.joint_coord_world_start is not None else []
        )
        joint_constraint_world_start = (
            model.joint_constraint_world_start.numpy() if model.joint_constraint_world_start is not None else []
        )

        # Optional console-output for debugging
        if verbose:
            print(f"particle_world_start: {particle_world_start}")
            print(f"body_world_start: {body_world_start}")
            print(f"shape_world_start: {shape_world_start}")
            print(f"joint_world_start: {joint_world_start}")
            print(f"articulation_world_start: {articulation_world_start}")
            print(f"joint_dof_world_start: {joint_dof_world_start}")
            print(f"joint_coord_world_start: {joint_coord_world_start}")
            print(f"joint_constraint_world_start: {joint_constraint_world_start}")

        # Verify total counts
        self.assertEqual(builder.particle_count, 0)
        self.assertEqual(builder.body_count, 5)
        self.assertEqual(builder.shape_count, 0)
        self.assertEqual(builder.joint_count, 5)
        self.assertEqual(builder.articulation_count, 3)
        self.assertEqual(builder._equality_constraint_count, 0)
        self.assertEqual(builder.joint_dof_count, 10)
        self.assertEqual(builder.joint_coord_count, 11)
        self.assertEqual(builder.joint_constraint_count, 20)
        self.assertEqual(particle_world_start[-1], builder.particle_count)
        self.assertEqual(body_world_start[-1], builder.body_count)
        self.assertEqual(shape_world_start[-1], builder.shape_count)
        self.assertEqual(joint_world_start[-1], builder.joint_count)
        self.assertEqual(articulation_world_start[-1], builder.articulation_count)
        self.assertEqual(joint_dof_world_start[-1], builder.joint_dof_count)
        self.assertEqual(joint_coord_world_start[-1], builder.joint_coord_count)
        self.assertEqual(joint_constraint_world_start[-1], builder.joint_constraint_count)

        # Check that sizes match world_count + 2, i.e. conforms to spec
        self.assertEqual(particle_world_start.size, model.world_count + 2)
        self.assertEqual(body_world_start.size, model.world_count + 2)
        self.assertEqual(shape_world_start.size, model.world_count + 2)
        self.assertEqual(joint_world_start.size, model.world_count + 2)
        self.assertEqual(articulation_world_start.size, model.world_count + 2)
        self.assertEqual(joint_dof_world_start.size, model.world_count + 2)
        self.assertEqual(joint_coord_world_start.size, model.world_count + 2)
        self.assertEqual(joint_constraint_world_start.size, model.world_count + 2)

        # Check that the last elements match total counts
        self.assertEqual(particle_world_start[-1], model.particle_count)
        self.assertEqual(body_world_start[-1], model.body_count)
        self.assertEqual(shape_world_start[-1], model.shape_count)
        self.assertEqual(joint_world_start[-1], model.joint_count)
        self.assertEqual(articulation_world_start[-1], model.articulation_count)
        self.assertEqual(joint_dof_world_start[-1], model.joint_dof_count)
        self.assertEqual(joint_coord_world_start[-1], model.joint_coord_count)
        self.assertEqual(joint_constraint_world_start[-1], model.joint_constraint_count)

        # Check that world starts are non-decreasing
        for i in range(model.world_count + 1):
            self.assertLessEqual(particle_world_start[i], particle_world_start[i + 1])
            self.assertLessEqual(body_world_start[i], body_world_start[i + 1])
            self.assertLessEqual(shape_world_start[i], shape_world_start[i + 1])
            self.assertLessEqual(joint_world_start[i], joint_world_start[i + 1])
            self.assertLessEqual(articulation_world_start[i], articulation_world_start[i + 1])
            self.assertLessEqual(joint_dof_world_start[i], joint_dof_world_start[i + 1])
            self.assertLessEqual(joint_coord_world_start[i], joint_coord_world_start[i + 1])
            self.assertLessEqual(joint_constraint_world_start[i], joint_constraint_world_start[i + 1])

        # Check exact values of world starts for this specific case
        self.assertTrue(np.array_equal(particle_world_start, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(body_world_start, np.array([0, 2, 4, 5])))
        self.assertTrue(np.array_equal(shape_world_start, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(joint_world_start, np.array([0, 2, 4, 5])))
        self.assertTrue(np.array_equal(articulation_world_start, np.array([0, 1, 2, 3])))
        self.assertTrue(np.array_equal(joint_dof_world_start, np.array([0, 2, 4, 10])))
        self.assertTrue(np.array_equal(joint_coord_world_start, np.array([0, 2, 4, 11])))
        self.assertTrue(np.array_equal(joint_constraint_world_start, np.array([0, 10, 20, 20])))

    def test_collapse_fixed_joints_with_selective_fixed_joint_collapsing(self):
        """Test that joints listed in joints_to_keep are not collapsed."""

        def add_joints_and_links(builder: ModelBuilder):
            b0 = builder.add_link(label="body_1", mass=1.0)
            b1 = builder.add_link(label="body_2", mass=1.0)
            j1 = builder.add_joint_fixed(parent=b0, child=b1, label="fixed_1")
            b2 = builder.add_link(label="body_3", mass=1.0)
            j2 = builder.add_joint_revolute(parent=b1, child=b2, label="rev_1")
            b3 = builder.add_link(label="body_4", mass=1.0)
            j3 = builder.add_joint_fixed(parent=b2, child=b3, label="fixed_2")
            builder.add_articulation([j1, j2, j3])

        # Testing default behaviour when the list joints_to_keep is empty
        builder_1 = ModelBuilder()
        add_joints_and_links(builder_1)

        builder_1.collapse_fixed_joints(joints_to_keep=[])

        # After collapse:
        # - body_1 and body_2 are merged (fixed_1 removed)
        # - body_3 and body_4 are merged (fixed_2 removed)
        # - Remaining bodies : body_1 (merged) and body_3 (merged)
        # - Remaining joints : rev_1

        self.assertEqual(builder_1.body_count, 2)
        self.assertEqual(builder_1.joint_count, 1)
        self.assertAlmostEqual(builder_1.body_mass[0], 2.0)
        self.assertAlmostEqual(builder_1.body_mass[1], 2.0)

        # Testing behaviour when joints_to_keep contains a joint
        builder_2 = ModelBuilder()
        add_joints_and_links(builder_2)

        builder_2.collapse_fixed_joints(joints_to_keep=["fixed_1"])

        # After collapse:
        # - fixed_1 is retained
        # - body_3 and body_4 are merged (fixed_2 removed)
        # - Remaining bodies : body_1, body_2 and body_3 (merged)
        # - Remaining joints : fixed_1 , rev_1

        self.assertIn("fixed_1", builder_2.joint_label)
        self.assertEqual(builder_2.body_count, 3)
        self.assertEqual(builder_2.joint_count, 2)
        self.assertAlmostEqual(builder_2.body_mass[0], 1.0)
        self.assertAlmostEqual(builder_2.body_mass[1], 1.0)
        self.assertAlmostEqual(builder_2.body_mass[2], 2.0)

        # Testing behaviour when joints_to_keep contains a hierarchical joint
        builder_3 = ModelBuilder()
        add_joints_and_links(builder_3)

        # Adding a nested builder in builder_3 to test hierarchical joints
        builder_nested = ModelBuilder()
        add_joints_and_links(builder_nested)
        builder_3.add_builder(builder_nested, label_prefix="builder_nested")

        builder_3.collapse_fixed_joints(joints_to_keep=["fixed_2", "builder_nested/fixed_1"])

        # After collapse:
        # - builder_nested/fixed_1 is retained
        # - body_1 and body_2 are merged (fixed_1 removed)
        # - builder_nested/body_3 and builder_nested/body_4 are merged (builder_nested/fixed_2 removed)
        # - Remaining bodies : body_1 (merged), body_3, body_4, builder_nested/body_1, builder_nested/body_2, builder_nested/body_3 (merged)
        # - Remaining joints : rev_1, fixed_2, builder_nested/fixed_1, builder_nested/rev_1

        self.assertIn("fixed_2", builder_3.joint_label)
        self.assertIn("builder_nested/fixed_1", builder_3.joint_label)
        self.assertEqual(builder_3.body_count, 6)
        self.assertEqual(builder_3.joint_count, 4)
        self.assertAlmostEqual(builder_3.body_mass[0], 2.0)
        self.assertAlmostEqual(builder_3.body_mass[1], 1.0)
        self.assertAlmostEqual(builder_3.body_mass[2], 1.0)
        self.assertAlmostEqual(builder_3.body_mass[3], 1.0)
        self.assertAlmostEqual(builder_3.body_mass[4], 1.0)
        self.assertAlmostEqual(builder_3.body_mass[5], 2.0)

        # Testing the warning when joints_to_keep contains a joint whose child has zero or negative mass
        builder_4 = ModelBuilder()
        b0 = builder_4.add_link(label="body_1", mass=1.0)
        b1 = builder_4.add_link(label="body_2", mass=0.0)
        j1 = builder_4.add_joint_fixed(parent=b0, child=b1, label="fixed_1")
        builder_4.add_articulation([j1])

        with self.assertWarns(UserWarning) as cm:
            builder_4.collapse_fixed_joints(joints_to_keep=["fixed_1"])
        self.assertIn("Skipped joint fixed_1 has a child body_2 with zero or negative mass", str(cm.warning))

    def test_collapse_fixed_joints_preserves_loop_closure(self):
        """Test that collapse_fixed_joints retains loop-closing joints.

        Covers two symmetric cases:
        1. The merged-away body is the loop joint's *parent* (parent remapping).
        2. The merged-away body is the loop joint's *child* (child remapping).
        """

        # --- Case 1: merged body is the loop joint's parent ---
        # world --(free)--> b0 --(revolute)--> b1 --(fixed)--> b2 --(revolute, loop)--> b0
        # After collapse b2 merges into b1; loop joint parent must remap b2 -> b1
        builder = ModelBuilder()
        b0 = builder.add_link(label="b0", mass=1.0)
        j0 = builder.add_joint_free(parent=-1, child=b0)
        b1 = builder.add_link(label="b1", mass=1.0)
        j1 = builder.add_joint_revolute(parent=b0, child=b1, axis=wp.vec3(0, 0, 1))
        b2 = builder.add_link(label="b2", mass=1.0)
        j2 = builder.add_joint_fixed(parent=b1, child=b2)
        builder.add_joint_revolute(parent=b2, child=b0, axis=wp.vec3(0, 0, 1), label="loop_b2_b0")
        builder.add_articulation([j0, j1, j2])

        builder.collapse_fixed_joints()

        self.assertEqual(builder.body_count, 2)
        self.assertEqual(builder.joint_count, 3)
        self.assertIn("loop_b2_b0", builder.joint_label)
        loop_i = builder.joint_label.index("loop_b2_b0")
        self.assertEqual(
            builder.joint_parent[loop_i],
            builder.body_label.index("b1"),
            "Loop joint parent should be remapped from b2 to b1",
        )
        self.assertEqual(
            builder.joint_child[loop_i], builder.body_label.index("b0"), "Loop joint child (b0) should be unchanged"
        )

        # --- Case 2: merged body is the loop joint's child ---
        # world --(free)--> b0 --(fixed)--> b1
        # world --(free)--> b2 --(revolute, loop)--> b1
        # After collapse b1 merges into b0; loop joint child must remap b1 -> b0
        builder = ModelBuilder()
        b0 = builder.add_link(label="b0", mass=1.0)
        j0 = builder.add_joint_free(parent=-1, child=b0)
        b1 = builder.add_link(label="b1", mass=1.0)
        j_fixed = builder.add_joint_fixed(parent=b0, child=b1, label="fixed_b0_b1")
        b2 = builder.add_link(label="b2", mass=1.0)
        j2 = builder.add_joint_free(parent=-1, child=b2)
        builder.add_joint_revolute(parent=b2, child=b1, axis=wp.vec3(0, 0, 1), label="loop_b2_b1")
        builder.add_articulation([j0, j_fixed])
        builder.add_articulation([j2])

        builder.collapse_fixed_joints()

        # b1 is merged into b0 -> 2 bodies (b0, b2)
        self.assertEqual(builder.body_count, 2)
        # the loop joint survives and is remapped from b2 -> b1 to b2 -> b0
        self.assertIn("loop_b2_b1", builder.joint_label)
        loop_i = builder.joint_label.index("loop_b2_b1")
        self.assertEqual(builder.joint_parent[loop_i], builder.body_label.index("b2"))
        self.assertEqual(builder.joint_child[loop_i], builder.body_label.index("b0"))

    def test_articulation_validation_contiguous(self):
        """Test that articulation requires contiguous joint indices"""
        builder = ModelBuilder()

        # Create links
        link1 = builder.add_link(mass=1.0)
        link2 = builder.add_link(mass=1.0)
        link3 = builder.add_link(mass=1.0)
        link4 = builder.add_link(mass=1.0)

        # Create joints
        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        joint2 = builder.add_joint_revolute(parent=link1, child=link2)
        joint3 = builder.add_joint_revolute(parent=link2, child=link3)
        joint4 = builder.add_joint_revolute(parent=link3, child=link4)

        # Test valid contiguous articulation
        builder.add_articulation([joint1, joint2, joint3, joint4])  # Should work

        # Test non-contiguous articulation should fail
        builder2 = ModelBuilder()
        link1 = builder2.add_link(mass=1.0)
        link2 = builder2.add_link(mass=1.0)
        link3 = builder2.add_link(mass=1.0)

        j1 = builder2.add_joint_revolute(parent=-1, child=link1)
        j2 = builder2.add_joint_revolute(parent=link1, child=link2)
        # Create a joint for another articulation to create a gap
        other_link = builder2.add_link(mass=1.0)
        _j_other = builder2.add_joint_revolute(parent=-1, child=other_link)
        j3 = builder2.add_joint_revolute(parent=link2, child=link3)

        # This should fail because [j1, j2, j3] are not contiguous (j_other is in between)
        with self.assertRaises(ValueError) as context:
            builder2.add_articulation([j1, j2, j3])
        self.assertIn("contiguous", str(context.exception))

    def test_articulation_validation_monotonic(self):
        """Test that articulation requires monotonically increasing joint indices"""
        builder = ModelBuilder()

        # Create links
        link1 = builder.add_link(mass=1.0)
        link2 = builder.add_link(mass=1.0)

        # Create joints
        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        joint2 = builder.add_joint_revolute(parent=link1, child=link2)

        # Test joints in wrong order (not monotonic)
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([joint2, joint1])  # Wrong order
        self.assertIn("monotonically increasing", str(context.exception))

    def test_articulation_validation_empty(self):
        """Test that articulation requires at least one joint"""
        builder = ModelBuilder()

        # Test empty articulation should fail
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([])
        self.assertIn("no joints", str(context.exception))

    def test_articulation_validation_world_mismatch(self):
        """Test that all joints in articulation must belong to same world"""
        builder = ModelBuilder()

        # Create joints in world 0
        builder.begin_world()
        link1 = builder.add_link(mass=1.0)
        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        builder.end_world()

        # Create joint in world 1
        builder.begin_world()
        link2 = builder.add_link(mass=1.0)
        joint2 = builder.add_joint_revolute(parent=-1, child=link2)

        # Try to create articulation from joints in different worlds (while still in world 1)
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([joint1, joint2])
        self.assertIn("world", str(context.exception).lower())
        builder.end_world()

    def test_articulation_validation_tree_structure(self):
        """Test that articulation validates tree structure (no multiple parents)"""
        builder = ModelBuilder()

        # Create links
        link1 = builder.add_link(mass=1.0)
        link2 = builder.add_link(mass=1.0)
        link3 = builder.add_link(mass=1.0)

        # Create joints that would form invalid tree (link2 has two parents)
        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        joint2 = builder.add_joint_revolute(parent=link1, child=link2)
        joint3 = builder.add_joint_revolute(parent=link3, child=link2)  # link2 already has parent link1

        # This should fail because link2 has multiple parents
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([joint1, joint2, joint3])
        self.assertIn("multiple parents", str(context.exception))

    def test_articulation_validation_duplicate_joint(self):
        """Test that adding a joint to multiple articulations raises an error"""
        builder = ModelBuilder()

        # Create links and joints
        link1 = builder.add_link(mass=1.0)
        link2 = builder.add_link(mass=1.0)

        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        joint2 = builder.add_joint_revolute(parent=link1, child=link2)

        # Add joints to first articulation
        builder.add_articulation([joint1, joint2])

        # Create another joint
        link3 = builder.add_link(mass=1.0)
        joint3 = builder.add_joint_revolute(parent=link2, child=link3)

        # Try to add joint2 (already in articulation) to a new articulation
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([joint2, joint3])
        self.assertIn("already belongs to articulation", str(context.exception))
        self.assertIn("joint_2", str(context.exception))  # joint2's key

    def test_joint_world_validation(self):
        """Test that joints validate parent/child bodies belong to current world"""
        builder = ModelBuilder()

        # Create body in world 0
        builder.begin_world()
        link1 = builder.add_link(mass=1.0)
        builder.end_world()

        # Switch to world 1 and try to create joint with body from world 0
        builder.begin_world()
        link2 = builder.add_link(mass=1.0)

        # This should fail because link1 is in world 0 but we're in world 1
        with self.assertRaises(ValueError) as context:
            builder.add_joint_revolute(parent=link1, child=link2)
        self.assertIn("world", str(context.exception).lower())
        builder.end_world()

    def test_articulation_validation_orphan_joint(self):
        """Test that joints not belonging to an articulation raise an error on finalize."""
        builder = ModelBuilder()
        parent = builder.add_link()
        child = builder.add_link()

        # World-root joints are intentionally allowed without articulation
        # metadata, so use a non-root joint to exercise orphan validation.
        builder.add_joint_revolute(parent=parent, child=child, label="orphan_joint")

        # finalize() should raise ValueError about orphan joints
        with self.assertRaises(ValueError) as context:
            builder.finalize()

        self.assertIn("not belonging to any articulation", str(context.exception))
        self.assertIn("orphan_joint", str(context.exception))

    def test_articulation_validation_allows_standalone_world_root(self):
        """Test that a standalone world-root joint does not require an articulation."""
        builder = ModelBuilder()
        body = builder.add_link()
        joint = builder.add_joint_fixed(parent=-1, child=body, label="standalone_root")

        model = builder.finalize()

        self.assertEqual(model.articulation_count, 0)
        self.assertEqual(model.joint_articulation.numpy()[joint], -1)

    def test_articulation_validation_multiple_orphan_joints(self):
        """Test error message shows multiple orphan joints."""
        builder = ModelBuilder()
        body1 = builder.add_link()
        body2 = builder.add_link()
        body3 = builder.add_link()

        # Add multiple non-root joints without articulations.
        builder.add_joint_revolute(parent=body1, child=body2, label="first_joint")
        builder.add_joint_revolute(parent=body2, child=body3, label="second_joint")

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("2 joint(s)", error_msg)
        self.assertIn("first_joint", error_msg)
        self.assertIn("second_joint", error_msg)

    def test_validate_structure_invalid_joint_parent(self):
        """Test that _validate_structure catches invalid joint_parent references."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body, label="test_joint")
        builder.add_articulation([joint])

        # Manually set invalid parent body reference
        builder.joint_parent[0] = 999  # Invalid body index

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid body reference", error_msg)
        self.assertIn("joint_parent", error_msg)
        self.assertIn("test_joint", error_msg)

    def test_validate_structure_invalid_joint_child(self):
        """Test that _validate_structure catches invalid joint_child references."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body, label="test_joint")
        builder.add_articulation([joint])

        # Manually set invalid child body reference (child cannot be -1)
        builder.joint_child[0] = -1  # Invalid: child cannot be world

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid body reference", error_msg)
        self.assertIn("joint_child", error_msg)
        self.assertIn("Child cannot be the world", error_msg)

    def test_validate_structure_self_referential_joint(self):
        """Test that _validate_structure catches self-referential joints."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body, label="self_ref_joint")
        builder.add_articulation([joint])

        # Manually set parent == child (self-referential)
        builder.joint_parent[0] = body
        builder.joint_child[0] = body

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Self-referential joint", error_msg)
        self.assertIn("self_ref_joint", error_msg)

    def test_validate_joint_ordering_correct_order(self):
        """Test that validate_joint_ordering passes for correctly ordered joints."""
        builder = ModelBuilder()

        # Create a simple chain in DFS order
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        body3 = builder.add_link(mass=1.0)

        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        joint3 = builder.add_joint_revolute(parent=body2, child=body3)
        builder.add_articulation([joint1, joint2, joint3])

        # Should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.validate_joint_ordering()
            ordering_warnings = [warning for warning in w if "DFS topological order" in str(warning.message)]
            self.assertEqual(len(ordering_warnings), 0)

        self.assertTrue(result)

    def test_validate_joint_ordering_incorrect_order(self):
        """Test that validate_joint_ordering warns for incorrectly ordered joints."""
        builder = ModelBuilder()

        # Create a chain: world -> body1 -> body2 -> body3
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        body3 = builder.add_link(mass=1.0)

        # Create joints in WRONG order: joint3 (body2->body3) comes BEFORE joint2 (body1->body2)
        # This is invalid because body2 hasn't been processed yet when we try to process joint3
        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint3 = builder.add_joint_revolute(parent=body2, child=body3)  # Out of order - parent not processed
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        builder.add_articulation([joint1, joint3, joint2])  # Wrong order: should be [joint1, joint2, joint3]

        # Should warn about non-DFS order
        with self.assertWarns(UserWarning) as cm:
            result = builder.validate_joint_ordering()

        self.assertFalse(result)
        self.assertIn("DFS topological order", str(cm.warning))

    def test_skip_validation_joint_ordering_default(self):
        """Test that joint ordering validation is skipped by default."""
        builder = ModelBuilder()

        # Create a chain: world -> body1 -> body2 -> body3
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        body3 = builder.add_link(mass=1.0)

        # Create joints in WRONG order for the chain
        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint3 = builder.add_joint_revolute(parent=body2, child=body3)  # Out of order
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        builder.add_articulation([joint1, joint3, joint2])

        # By default (skip_validation_joint_ordering=True), should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.finalize()
            ordering_warnings = [warning for warning in w if "DFS topological order" in str(warning.message)]
            self.assertEqual(len(ordering_warnings), 0)

    def test_enable_validation_joint_ordering(self):
        """Test that joint ordering validation can be enabled."""
        builder = ModelBuilder()

        # Create a chain: world -> body1 -> body2 -> body3
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        body3 = builder.add_link(mass=1.0)

        # Create joints in WRONG order for the chain
        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint3 = builder.add_joint_revolute(parent=body2, child=body3)  # Out of order
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        builder.add_articulation([joint1, joint3, joint2])

        # With skip_validation_joint_ordering=False, should warn
        with self.assertWarns(UserWarning) as cm:
            builder.finalize(skip_validation_joint_ordering=False)

        self.assertIn("DFS topological order", str(cm.warning))

    def test_mimic_constraint_programmatic(self):
        """Test programmatic creation of mimic constraints."""
        builder = newton.ModelBuilder()

        # Create two joints
        b0 = builder.add_body()
        b1 = builder.add_body()
        b2 = builder.add_body()

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b0,
            axis=(0, 0, 1),
            label="j1",
        )
        j2 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=(0, 0, 1),
            label="j2",
        )
        j3 = builder.add_joint_revolute(
            parent=-1,
            child=b2,
            axis=(0, 0, 1),
            label="j3",
        )

        # Add mimic constraints
        _c1 = builder.add_constraint_mimic(
            joint0=j2,
            joint1=j1,
            coef0=-0.25,
            coef1=1.5,
            label="mimic1",
        )
        _c2 = builder.add_constraint_mimic(
            joint0=j3,
            joint1=j1,
            coef0=0.0,
            coef1=-1.0,
            enabled=False,
            label="mimic2",
        )

        model = builder.finalize()

        self.assertEqual(model.constraint_mimic_count, 2)

        # Check first constraint
        self.assertEqual(model.constraint_mimic_joint0.numpy()[0], j2)
        self.assertEqual(model.constraint_mimic_joint1.numpy()[0], j1)
        self.assertAlmostEqual(model.constraint_mimic_coef0.numpy()[0], -0.25)
        self.assertAlmostEqual(model.constraint_mimic_coef1.numpy()[0], 1.5)
        self.assertTrue(model.constraint_mimic_enabled.numpy()[0])
        self.assertEqual(model.constraint_mimic_label[0], "mimic1")

        # Check second constraint
        self.assertEqual(model.constraint_mimic_joint0.numpy()[1], j3)
        self.assertEqual(model.constraint_mimic_joint1.numpy()[1], j1)
        self.assertAlmostEqual(model.constraint_mimic_coef0.numpy()[1], 0.0)
        self.assertAlmostEqual(model.constraint_mimic_coef1.numpy()[1], -1.0)
        self.assertFalse(model.constraint_mimic_enabled.numpy()[1])
        self.assertEqual(model.constraint_mimic_label[1], "mimic2")

    def test_add_base_joint_fixed_to_parent(self):
        """Test that add_base_joint with parent creates fixed joint."""
        builder = ModelBuilder()
        parent_body = builder.add_body(xform=wp.transform((0, 0, 0), wp.quat_identity()), mass=1.0)
        parent_joint = builder.add_joint_fixed(parent=-1, child=parent_body)
        builder.add_articulation([parent_joint])  # Register parent body into an articulation

        child_body = builder.add_body(xform=wp.transform((1, 0, 0), wp.quat_identity()), mass=0.5)
        joint_id = builder._add_base_joint(child_body, parent=parent_body, floating=False)

        self.assertEqual(builder.joint_type[joint_id], newton.JointType.FIXED)
        self.assertEqual(builder.joint_parent[joint_id], parent_body)


class TestModelWorld(unittest.TestCase):
    def test_add_world_with_open_edges(self):
        builder = ModelBuilder()

        dim_x = 16
        dim_y = 16

        world_builder = ModelBuilder()
        world_builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.1, 0.1, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.25),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=1.0 / dim_x,
            cell_y=1.0 / dim_y,
            mass=1.0,
        )

        world_count = 2
        world_offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        builder_open_edge_count = np.sum(np.array(builder.edge_indices) == -1)
        world_builder_open_edge_count = np.sum(np.array(world_builder.edge_indices) == -1)

        for i in range(world_count):
            xform = wp.transform(world_offsets[i], wp.quat_identity())
            builder.add_world(world_builder, xform)

        self.assertEqual(
            np.sum(np.array(builder.edge_indices) == -1),
            builder_open_edge_count + world_count * world_builder_open_edge_count,
            "builder does not have the expected number of open edges",
        )

    def test_add_particles_grouping(self):
        """Test that add_particles correctly assigns world groups."""
        builder = ModelBuilder()

        # Test with default group (-1)
        builder.add_particles(
            pos=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], vel=[(0.0, 0.0, 0.0)] * 3, mass=[1.0] * 3
        )

        # Change to world 0 and add more particles
        builder.begin_world()
        builder.add_particles(pos=[(3.0, 0.0, 0.0), (4.0, 0.0, 0.0)], vel=[(0.0, 0.0, 0.0)] * 2, mass=[1.0] * 2)
        builder.end_world()

        # Finalize and check groups
        model = builder.finalize()
        particle_groups = model.particle_world.numpy()

        # First 3 particles should be in group -1
        self.assertTrue(np.all(particle_groups[0:3] == -1))
        # Next 2 particles should be in group 0
        self.assertTrue(np.all(particle_groups[3:5] == 0))

    def test_world_grouping(self):
        """Test world grouping functionality for Model entities."""
        # Optionally enable debug printing
        verbose = False  # Set to True to enable debug output

        # Create builder with a mix of global and world-specific entities
        main_builder = ModelBuilder()

        # Create global entities (world -1)
        ground_body = main_builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_identity()), mass=0.0)
        main_builder.add_shape_box(
            body=ground_body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=5.0, hy=5.0, hz=0.1
        )
        main_builder.add_particle((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), mass=1.0)

        # Create a simple builder for worlds
        def create_world_builder():
            world_builder = ModelBuilder()
            # Add particles
            p1 = world_builder.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), mass=1.0)
            p2 = world_builder.add_particle((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), mass=1.0)
            world_builder.add_spring(p1, p2, ke=100.0, kd=1.0, control=0.0)

            # Add articulated body
            b1 = world_builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=10.0)
            b2 = world_builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), mass=5.0)
            b3 = world_builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=2.5)
            j1 = world_builder.add_joint_revolute(parent=b1, child=b2, axis=(0, 1, 0))
            j2 = world_builder.add_joint_revolute(parent=b2, child=b3, axis=(0, 1, 0))
            world_builder.add_articulation([j1, j2])
            world_builder.add_shape_sphere(
                body=b1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.1
            )
            world_builder.add_shape_sphere(
                body=b2, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.05
            )
            world_builder.add_shape_sphere(
                body=b3, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.025
            )

            return world_builder

        # Add world 0
        world0_builder = create_world_builder()
        main_builder.add_world(world0_builder, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))

        # Add world 1
        world1_builder = create_world_builder()
        main_builder.add_world(world1_builder, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))

        # Add world 2
        world2_builder = create_world_builder()
        main_builder.add_world(world2_builder, xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()))

        # Add more global entities to end of the model
        floor_body = main_builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_identity()), mass=0.0)
        main_builder.add_shape_box(
            body=floor_body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=5.0, hy=5.0, hz=0.1
        )
        ball_body = main_builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
        main_builder.add_shape_sphere(
            body=ball_body, xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()), radius=0.5
        )
        main_builder.add_particle((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), mass=1.0)
        main_builder.add_particle((0.0, 0.0, 5.5), (0.0, 0.0, 0.0), mass=1.0)

        # Finalize the model
        model = main_builder.finalize()

        # Verify counts
        self.assertEqual(model.world_count, 3)
        self.assertEqual(model.particle_count, 9)  # 3 global + 2*3 = 9
        self.assertEqual(model.body_count, 12)  # 3 global + 3*3 = 12
        self.assertEqual(model.shape_count, 12)  # 3 global + 3*3 = 12
        self.assertEqual(model.joint_count, 9)  # 3 global + 2*3 = 9
        self.assertEqual(model.articulation_count, 6)  # 3 global + 1*3 = 6

        # Verify group assignments
        particle_world = model.particle_world.numpy() if model.particle_world is not None else []
        body_world = model.body_world.numpy() if model.body_world is not None else []
        shape_world = model.shape_world.numpy() if model.shape_world is not None else []
        joint_world = model.joint_world.numpy() if model.joint_world is not None else []
        articulation_world = model.articulation_world.numpy() if model.articulation_world is not None else []

        if len(particle_world) > 0:
            # Check global entities
            self.assertEqual(particle_world[0], -1)  # global particle at front
            self.assertEqual(particle_world[-2], -1)  # global particle at back
            self.assertEqual(particle_world[-1], -1)  # global particle at back

            # Check world 0 entities (indices for particles)
            self.assertTrue(np.all(particle_world[1:3] == 0))

            # Check world 1 entities (auto-assigned)
            self.assertTrue(np.all(particle_world[3:5] == 1))

            # Check world 2 entities (auto-assigned)
            self.assertTrue(np.all(particle_world[5:7] == 2))

        if len(body_world) > 0:
            self.assertEqual(body_world[0], -1)  # ground body
            self.assertTrue(np.all(body_world[1:4] == 0))
            self.assertTrue(np.all(body_world[4:7] == 1))
            self.assertTrue(np.all(body_world[7:10] == 2))
            self.assertEqual(body_world[10], -1)  # floor body
            self.assertEqual(body_world[11], -1)  # ball body

        if len(shape_world) > 0:
            self.assertEqual(shape_world[0], -1)  # ground shape
            self.assertTrue(np.all(shape_world[1:4] == 0))
            self.assertTrue(np.all(shape_world[4:7] == 1))
            self.assertTrue(np.all(shape_world[7:10] == 2))
            self.assertEqual(shape_world[10], -1)  # floor shape
            self.assertEqual(shape_world[11], -1)  # ball shape

        if len(joint_world) > 0:
            self.assertEqual(joint_world[0], -1)  # ground body's free joint
            self.assertEqual(joint_world[1], 0)
            self.assertEqual(joint_world[2], 0)
            self.assertEqual(joint_world[3], 1)
            self.assertEqual(joint_world[4], 1)
            self.assertEqual(joint_world[5], 2)
            self.assertEqual(joint_world[6], 2)
            self.assertEqual(joint_world[7], -1)  # floor body's free joint
            self.assertEqual(joint_world[8], -1)  # ball body's free joint

        if len(articulation_world) > 0:
            self.assertEqual(articulation_world[0], -1)  # ground body's articulation
            self.assertEqual(articulation_world[1], 0)
            self.assertEqual(articulation_world[2], 1)
            self.assertEqual(articulation_world[3], 2)
            self.assertEqual(articulation_world[4], -1)  # floor body's articulation
            self.assertEqual(articulation_world[5], -1)  # ball body's articulation

        # Verify world start indices
        particle_world_start = model.particle_world_start.numpy() if model.particle_world_start is not None else []
        body_world_start = model.body_world_start.numpy() if model.body_world_start is not None else []
        shape_world_start = model.shape_world_start.numpy() if model.shape_world_start is not None else []
        joint_world_start = model.joint_world_start.numpy() if model.joint_world_start is not None else []
        articulation_world_start = (
            model.articulation_world_start.numpy() if model.articulation_world_start is not None else []
        )
        joint_dof_world_start = model.joint_dof_world_start.numpy() if model.joint_dof_world_start is not None else []
        joint_coord_world_start = (
            model.joint_coord_world_start.numpy() if model.joint_coord_world_start is not None else []
        )
        joint_constraint_world_start = (
            model.joint_constraint_world_start.numpy() if model.joint_constraint_world_start is not None else []
        )

        # Optional console-output for debugging
        if verbose:
            print(f"particle_world_start: {particle_world_start}")
            print(f"body_world_start: {body_world_start}")
            print(f"shape_world_start: {shape_world_start}")
            print(f"joint_world_start: {joint_world_start}")
            print(f"articulation_world_start: {articulation_world_start}")
            print(f"joint_dof_world_start: {joint_dof_world_start}")
            print(f"joint_coord_world_start: {joint_coord_world_start}")
            print(f"joint_constraint_world_start: {joint_constraint_world_start}")

        # Check that sizes match world_count + 2, i.e. conforms to spec
        self.assertEqual(particle_world_start.size, model.world_count + 2)
        self.assertEqual(body_world_start.size, model.world_count + 2)
        self.assertEqual(shape_world_start.size, model.world_count + 2)
        self.assertEqual(joint_world_start.size, model.world_count + 2)
        self.assertEqual(articulation_world_start.size, model.world_count + 2)
        self.assertEqual(joint_dof_world_start.size, model.world_count + 2)
        self.assertEqual(joint_coord_world_start.size, model.world_count + 2)
        self.assertEqual(joint_constraint_world_start.size, model.world_count + 2)

        # Check that the last elements match total counts
        self.assertEqual(particle_world_start[-1], model.particle_count)
        self.assertEqual(body_world_start[-1], model.body_count)
        self.assertEqual(shape_world_start[-1], model.shape_count)
        self.assertEqual(joint_world_start[-1], model.joint_count)
        self.assertEqual(articulation_world_start[-1], model.articulation_count)
        self.assertEqual(joint_dof_world_start[-1], model.joint_dof_count)
        self.assertEqual(joint_coord_world_start[-1], model.joint_coord_count)
        self.assertEqual(joint_constraint_world_start[-1], model.joint_constraint_count)

        # Check that world starts are non-decreasing
        for i in range(model.world_count + 1):
            self.assertLessEqual(particle_world_start[i], particle_world_start[i + 1])
            self.assertLessEqual(body_world_start[i], body_world_start[i + 1])
            self.assertLessEqual(shape_world_start[i], shape_world_start[i + 1])
            self.assertLessEqual(joint_world_start[i], joint_world_start[i + 1])
            self.assertLessEqual(articulation_world_start[i], articulation_world_start[i + 1])
            self.assertLessEqual(joint_dof_world_start[i], joint_dof_world_start[i + 1])
            self.assertLessEqual(joint_coord_world_start[i], joint_coord_world_start[i + 1])
            self.assertLessEqual(joint_constraint_world_start[i], joint_constraint_world_start[i + 1])

        # Check exact values of world starts for this specific case
        self.assertTrue(np.array_equal(particle_world_start, np.array([1, 3, 5, 7, 9])))
        self.assertTrue(np.array_equal(body_world_start, np.array([1, 4, 7, 10, 12])))
        self.assertTrue(np.array_equal(shape_world_start, np.array([1, 4, 7, 10, 12])))
        self.assertTrue(np.array_equal(joint_world_start, np.array([1, 3, 5, 7, 9])))
        self.assertTrue(np.array_equal(articulation_world_start, np.array([1, 2, 3, 4, 6])))
        self.assertTrue(np.array_equal(joint_dof_world_start, np.array([6, 8, 10, 12, 24])))
        self.assertTrue(np.array_equal(joint_coord_world_start, np.array([7, 9, 11, 13, 27])))
        self.assertTrue(np.array_equal(joint_constraint_world_start, np.array([0, 10, 20, 30, 30])))

    def test_world_count_tracking(self):
        """Test that world_count is properly tracked when using add_world."""
        main_builder = ModelBuilder()

        # Create a simple sub-builder
        sub_builder = ModelBuilder()
        sub_builder.add_body(mass=1.0)

        # Test 1: Global entities should not increment world_count
        self.assertEqual(main_builder.world_count, 0)
        main_builder.add_builder(sub_builder)  # Adds to global world (-1)
        self.assertEqual(main_builder.world_count, 0)  # Should still be 0

        # Test 2: Using add_world() for automatic world management
        main_builder.add_world(sub_builder)
        self.assertEqual(main_builder.world_count, 1)

        main_builder.add_world(sub_builder)
        self.assertEqual(main_builder.world_count, 2)

        # Test 3: Using begin_world/end_world
        main_builder2 = ModelBuilder()

        # Add worlds in sequence
        main_builder2.begin_world()
        main_builder2.add_builder(sub_builder)
        main_builder2.end_world()
        self.assertEqual(main_builder2.world_count, 1)

        main_builder2.begin_world()
        main_builder2.add_builder(sub_builder)
        main_builder2.end_world()
        self.assertEqual(main_builder2.world_count, 2)

        # Test 4: Adding to same world using begin_world with existing index
        main_builder2.begin_world()
        main_builder2.add_builder(sub_builder)  # Adds to world 2
        main_builder2.add_builder(sub_builder)  # Also adds to world 2
        main_builder2.end_world()
        self.assertEqual(main_builder2.world_count, 3)  # Should now be 3

    def test_world_validation_errors(self):
        """Test that world validation catches non-contiguous and non-monotonic world indices."""
        # Test non-contiguous worlds
        builder1 = ModelBuilder()
        sub_builder = ModelBuilder()
        sub_builder.add_body(mass=1.0)

        # Create world 0 and world 2, skipping world 1
        # We need to manually manipulate world indices to create invalid cases
        builder1.add_world(sub_builder)  # Creates world 0
        # Manually skip world 1 by incrementing world_count
        builder1.world_count = 2
        builder1.begin_world()  # This will be world 2
        builder1.add_builder(sub_builder)
        builder1.end_world()

        # Should raise error about non-contiguous worlds
        with self.assertRaises(ValueError) as cm:
            builder1.finalize()
        self.assertIn("not contiguous", str(cm.exception))

        # Test non-monotonic worlds
        # This is harder to create with the new API since worlds are always added in order
        # We'll have to directly manipulate the world arrays
        builder2 = ModelBuilder()
        builder2.add_world(sub_builder)  # World 0
        builder2.add_world(sub_builder)  # World 1
        # Manually swap world indices to create non-monotonic ordering
        builder2.body_world[0], builder2.body_world[1] = builder2.body_world[1], builder2.body_world[0]

        # Should raise error about non-monotonic ordering
        with self.assertRaises(ValueError) as cm:
            builder2.finalize()
        self.assertIn("monotonic", str(cm.exception))

    def test_world_context_errors(self):
        """Test error handling for begin_world() and end_world()."""
        # Test calling begin_world() twice without end_world()
        builder1 = ModelBuilder()
        builder1.begin_world()
        with self.assertRaises(RuntimeError) as cm:
            builder1.begin_world()
        self.assertIn("Cannot begin a new world", str(cm.exception))
        self.assertIn("already in world context", str(cm.exception))

        # Test calling end_world() without begin_world()
        builder2 = ModelBuilder()
        with self.assertRaises(RuntimeError) as cm:
            builder2.end_world()
        self.assertIn("Cannot end world", str(cm.exception))
        self.assertIn("not currently in a world context", str(cm.exception))

        # Test that we can still use the builder correctly after proper usage
        builder3 = ModelBuilder()
        builder3.begin_world()
        builder3.add_body()
        builder3.end_world()
        model = builder3.finalize()
        self.assertEqual(model.world_count, 1)

        # Test world index out of range (above world_count-1)
        builder4 = ModelBuilder()
        builder4.begin_world()  # Creates world 0
        builder4.add_body()
        builder4.end_world()
        # Manually set world index above valid range
        builder4.body_world[0] = 5  # world_count=1, so valid range is -1 to 0
        with self.assertRaises(ValueError) as cm:
            builder4.finalize()
        self.assertIn("Invalid world index", str(cm.exception))

        # Test world index below -1 (invalid)
        builder5 = ModelBuilder()
        builder5.begin_world()
        builder5.add_body()
        builder5.end_world()
        # Manually set an invalid world index below -1
        builder5.body_world[0] = -2
        with self.assertRaises(ValueError) as cm:
            builder5.finalize()
        self.assertIn("Invalid world index", str(cm.exception))

    def test_add_world(self):
        orig_xform = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_rpy(0.5, 0.6, 0.7))
        offset_xform = wp.transform(wp.vec3(4.0, 5.0, 6.0), wp.quat_rpy(-0.7, 0.8, -0.9))

        fixed_base = ModelBuilder()
        b0 = fixed_base.add_link(xform=orig_xform)
        j0 = fixed_base.add_joint_revolute(parent=-1, child=b0, parent_xform=orig_xform)
        fixed_base.add_articulation([j0])
        fixed_base.add_shape_sphere(body=b0, xform=orig_xform)

        floating_base = ModelBuilder()
        b1 = floating_base.add_link(xform=orig_xform)
        j1 = floating_base.add_joint_free(parent=-1, child=b1)
        floating_base.add_articulation([j1])
        floating_base.add_shape_sphere(body=b1, xform=orig_xform)

        static_shape = ModelBuilder()
        static_shape.add_shape_sphere(body=-1, xform=orig_xform)

        builder = ModelBuilder()
        builder.add_world(fixed_base, xform=offset_xform)
        builder.add_world(floating_base, xform=offset_xform)
        builder.add_world(static_shape, xform=offset_xform)

        self.assertEqual(builder.body_count, 2)
        self.assertEqual(builder.joint_count, 2)
        self.assertEqual(builder.articulation_count, 2)
        self.assertEqual(builder.shape_count, 3)
        self.assertEqual(builder.body_world, [0, 1])
        self.assertEqual(builder.joint_world, [0, 1])
        self.assertEqual(builder.joint_type, [newton.JointType.REVOLUTE, newton.JointType.FREE])
        self.assertEqual(builder.joint_parent, [-1, -1])
        self.assertEqual(builder.joint_child, [0, 1])
        self.assertEqual(builder.joint_q_start, [0, 1])
        self.assertEqual(builder.joint_qd_start, [0, 1])
        self.assertEqual(builder.shape_world, [0, 1, 2])
        self.assertEqual(builder.shape_body, [0, 1, -1])
        self.assertEqual(builder.body_shapes, {0: [0], 1: [1], -1: [2]})
        self.assertEqual(builder.body_q[0], offset_xform * orig_xform)
        self.assertEqual(builder.body_q[1], offset_xform * orig_xform)
        # fixed base has updated parent transform
        assert_np_equal(np.array(builder.joint_X_p[0]), np.array(offset_xform * orig_xform), tol=1.0e-6)
        # floating base has updated joint coordinates
        assert_np_equal(np.array(builder.joint_q[1:]), np.array(offset_xform * orig_xform), tol=1.0e-6)
        # shapes with a parent body keep the original transform
        assert_np_equal(np.array(builder.shape_transform[0]), np.array(orig_xform), tol=1.0e-6)
        assert_np_equal(np.array(builder.shape_transform[1]), np.array(orig_xform), tol=1.0e-6)
        # static shape receives the offset transform
        assert_np_equal(np.array(builder.shape_transform[2]), np.array(offset_xform * orig_xform), tol=1.0e-6)


class TestModelValidation(unittest.TestCase):
    def test_lock_inertia_on_shape_addition(self):
        builder = ModelBuilder()
        shape_cfg = ModelBuilder.ShapeConfig(density=1000.0)
        base_com = wp.vec3(0.1, 0.2, 0.3)
        base_inertia = wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4)

        locked_body = builder.add_link(mass=2.0, com=base_com, inertia=base_inertia, lock_inertia=True)
        unlocked_body = builder.add_link(mass=2.0, com=base_com, inertia=base_inertia, lock_inertia=False)

        locked_mass = builder.body_mass[locked_body]
        locked_com = builder.body_com[locked_body]
        locked_inertia = builder.body_inertia[locked_body]

        unlocked_mass = builder.body_mass[unlocked_body]

        builder.add_shape_box(body=locked_body, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
        builder.add_shape_box(body=unlocked_body, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)

        self.assertEqual(builder.body_mass[locked_body], locked_mass)
        assert_np_equal(np.array(builder.body_com[locked_body]), np.array(locked_com))
        assert_np_equal(np.array(builder.body_inertia[locked_body]), np.array(locked_inertia))
        self.assertNotEqual(builder.body_mass[unlocked_body], unlocked_mass)

    def test_validate_structure_invalid_equality_constraint_body(self):
        """Test that _validate_structure catches invalid equality constraint body references."""
        builder = ModelBuilder()
        body1 = builder.add_body(mass=1.0)
        body2 = builder.add_body(mass=1.0)
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.WELD,
            body1=body1,
            body2=body2,
            label="test_constraint",
        )

        # Manually set invalid body reference
        _eq_set_value(builder, "equality_constraint_body1", 0, 999)

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid body reference", error_msg)
        self.assertIn("equality_constraint_body1", error_msg)
        self.assertIn("test_constraint", error_msg)

    def test_validate_structure_invalid_equality_constraint_joint(self):
        """Test that _validate_structure catches invalid equality constraint joint references."""
        builder = ModelBuilder()
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        builder.add_articulation([joint1, joint2])

        # Add a joint equality constraint
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.JOINT,
            joint1=joint1,
            joint2=joint2,
            label="joint_constraint",
        )

        # Manually set invalid joint reference
        _eq_set_value(builder, "equality_constraint_joint1", 0, 999)

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid joint reference", error_msg)
        self.assertIn("equality_constraint_joint1", error_msg)
        self.assertIn("joint_constraint", error_msg)

    def test_validate_structure_array_length_mismatch(self):
        """Test that _validate_structure catches array length mismatches."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body)
        builder.add_articulation([joint])

        # Manually corrupt array length
        builder.joint_armature.append(0.0)  # Add extra element

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Array length mismatch", error_msg)
        self.assertIn("joint_armature", error_msg)

    def test_skip_all_validations(self):
        """Test that skip_all_validations skips all validation checks."""
        builder = ModelBuilder()
        parent = builder.add_link(mass=1.0)
        child = builder.add_link(mass=1.0)
        builder.add_joint_revolute(parent=parent, child=child, label="orphan_joint")
        # Don't add articulation - this would normally fail _validate_joints

        # Without skip_all_validations, should raise ValueError about orphan joint
        with self.assertRaises(ValueError) as context:
            builder.finalize(skip_all_validations=False)
        self.assertIn("orphan_joint", str(context.exception))

        # With skip_all_validations=True, should NOT raise the validation error
        # Create a fresh builder for clean test
        builder2 = ModelBuilder()
        parent2 = builder2.add_link(mass=1.0)
        child2 = builder2.add_link(mass=1.0)
        builder2.add_joint_revolute(parent=parent2, child=child2, label="orphan_joint2")
        # This should succeed (validation skipped)
        model = builder2.finalize(skip_all_validations=True)
        self.assertIsNotNone(model)

    def test_skip_validation_structure(self):
        """Test that skip_validation_structure skips structural validation."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body)
        builder.add_articulation([joint])

        # Manually corrupt array length to trigger structure validation error
        builder.joint_armature.append(0.0)  # Add extra element

        # Without skip_validation_structure, should raise ValueError
        with self.assertRaises(ValueError) as context:
            builder.finalize(skip_validation_structure=False)
        self.assertIn("Array length mismatch", str(context.exception))

        # Create fresh builder with same corruption
        builder2 = ModelBuilder()
        body2 = builder2.add_link(mass=1.0)
        joint2 = builder2.add_joint_revolute(parent=-1, child=body2)
        builder2.add_articulation([joint2])
        builder2.joint_armature.append(0.0)

        # With skip_validation_structure=True, should skip the structure check
        # Model creation will likely fail, but not from structure validation
        try:
            builder2.finalize(skip_validation_structure=True)
        except ValueError as e:
            # If it raises ValueError, it should NOT be about array length mismatch
            self.assertNotIn("Array length mismatch", str(e))

    def test_control_clear(self):
        """Test that Control.clear() works without errors."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])

        model = builder.finalize()
        control = model.control()
        try:
            control.clear()
        except Exception as e:
            self.fail(f"control.clear() raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
