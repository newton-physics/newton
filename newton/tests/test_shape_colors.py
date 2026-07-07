# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock

import numpy as np
import warp as wp

import newton
from newton._src.viewer.viewer import MAX_TRIANGLE_OPACITY_GROUPS
from newton._src.viewer.viewer_gl import ViewerGL, _compute_shape_vbo_xforms
from newton.viewer import ViewerNull


class _ShapeColorProbe(ViewerNull):
    """Captures per-batch appearance values passed through ``log_instances``."""

    def __init__(self):
        """Initialize the probe with storage for the latest appearance values."""
        super().__init__(num_frames=1)
        self.last_colors = None
        self.last_opacities = None

    def log_instances(self, name, mesh, xforms, scales, colors, materials, *, opacities=None, hidden=False):
        """Capture the most recent instance appearance values sent to the viewer."""
        del name, mesh, xforms, scales, materials, hidden
        self.last_colors = None if colors is None else colors.numpy().copy()
        self.last_opacities = None if opacities is None else opacities.numpy().copy()


class _TriangleOpacityProbe(ViewerNull):
    """Captures mesh opacity values passed through ``log_mesh``."""

    def __init__(self):
        """Initialize the probe with storage for mesh opacity values."""
        super().__init__(num_frames=1)
        self.mesh_opacities = {}

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
        opacity=None,
    ):
        """Capture opacity for visible triangle mesh logs."""
        del points, indices, normals, uvs, texture, backface_culling
        if not hidden:
            self.mesh_opacities[name] = opacity


class TestShapeColors(unittest.TestCase):
    """Regression tests for shape color storage and viewer synchronization."""

    def setUp(self):
        """Cache the active Warp device for model finalization."""
        self.device = wp.get_device()

    def _make_tetra_mesh(self, color=None, opacity=None):
        """Create a small tetrahedral mesh with optional display appearance."""
        vertices = np.array(
            [
                (-0.5, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, 0.0, 0.5),
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32)
        return newton.Mesh(vertices, indices, color=color, opacity=opacity)

    def _make_soft_tet_mesh(self, opacity=None):
        """Create a one-tet deformable mesh with optional surface opacity."""
        vertices = np.array(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 3], dtype=np.int32)
        return newton.TetMesh(vertices, indices, opacity=opacity)

    def test_collision_shape_without_explicit_color_uses_palette_by_default(self):
        """Verify collision shapes use the per-shape palette sequence by default."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_box(body=body, hx=0.1, hy=0.2, hz=0.3)

        model = builder.finalize(device=self.device)
        viewer = ViewerNull()
        expected = np.array(viewer._shape_color_map(shape), dtype=np.float32)

        np.testing.assert_allclose(model.shape_color.numpy()[shape], expected, atol=1e-6, rtol=1e-6)

    def test_add_shape_mesh_uses_mesh_color_when_color_is_none(self):
        """Verify mesh shapes inherit embedded mesh colors when no override is given."""
        mesh = self._make_tetra_mesh(color=(0.2, 0.4, 0.6))
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_mesh(body=body, mesh=mesh)

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.shape_color.numpy()[shape], [0.2, 0.4, 0.6], atol=1e-6, rtol=1e-6)

    def test_explicit_shape_color_overrides_mesh_color(self):
        """Verify explicit shape colors override colors embedded in meshes."""
        mesh = self._make_tetra_mesh(color=(0.2, 0.4, 0.6))
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_mesh(
            body=body,
            mesh=mesh,
            color=(0.9, 0.1, 0.3),
        )

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.shape_color.numpy()[shape], [0.9, 0.1, 0.3], atol=1e-6, rtol=1e-6)

    def test_shape_opacity_defaults_to_opaque(self):
        """Verify shapes default to fully opaque display opacity."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_box(body=body, hx=0.1, hy=0.2, hz=0.3)

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.shape_opacity.numpy()[shape], 1.0, atol=1e-6, rtol=1e-6)

    def test_add_shape_mesh_uses_mesh_opacity_when_opacity_is_none(self):
        """Verify mesh shapes inherit embedded mesh opacity when no override is given."""
        mesh = self._make_tetra_mesh(opacity=0.35)
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_mesh(body=body, mesh=mesh)

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.shape_opacity.numpy()[shape], 0.35, atol=1e-6, rtol=1e-6)

    def test_explicit_shape_opacity_overrides_mesh_opacity(self):
        """Verify explicit shape opacity overrides opacity embedded in meshes."""
        mesh = self._make_tetra_mesh(opacity=0.35)
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_mesh(body=body, mesh=mesh, opacity=0.8)

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.shape_opacity.numpy()[shape], 0.8, atol=1e-6, rtol=1e-6)

    def test_shape_opacity_rejects_invalid_values(self):
        """Verify shape opacity is finite and in the display opacity range."""
        for invalid_opacity in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(opacity=invalid_opacity):
                builder = newton.ModelBuilder()
                body = builder.add_body(mass=1.0)
                with self.assertRaisesRegex(ValueError, "Shape opacity"):
                    builder.add_shape_box(body=body, hx=0.1, hy=0.2, hz=0.3, opacity=invalid_opacity)
                self.assertEqual(builder.shape_count, 0)

    def test_triangle_opacity_rejects_invalid_values_before_mutation(self):
        """Triangle opacity follows the same finite [0, 1] contract as shapes."""
        for invalid_opacity in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(opacity=invalid_opacity):
                builder = newton.ModelBuilder()
                for position in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
                    builder.add_particle(pos=position, vel=(0.0, 0.0, 0.0), mass=1.0)

                with self.assertRaisesRegex(ValueError, "Triangle opacity"):
                    builder.add_triangle(0, 1, 2, opacity=invalid_opacity)

                self.assertEqual(len(builder.tri_indices), 0)
                self.assertEqual(len(builder.tri_opacity), 0)

    def test_triangle_opacity_array_rejects_wrong_length_before_mutation(self):
        builder = newton.ModelBuilder()
        for position in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
            builder.add_particle(pos=position, vel=(0.0, 0.0, 0.0), mass=1.0)

        with self.assertRaisesRegex(ValueError, "exactly 1 values"):
            builder.add_triangles([0], [1], [2], opacity=[0.2, 0.4])

        self.assertEqual(len(builder.tri_indices), 0)
        self.assertEqual(len(builder.tri_opacity), 0)

    def test_cloth_opacity_defaults_to_opaque(self):
        """Verify cloth triangles default to fully opaque display opacity."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
        )

        model = builder.finalize(device=self.device)

        self.assertEqual(model.tri_count, 2)
        np.testing.assert_allclose(
            model.tri_opacity.numpy(),
            np.ones(2, dtype=np.float32),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_cloth_grid_stores_explicit_opacity(self):
        """Verify cloth helper opacity is stored per generated surface triangle."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
            opacity=0.4,
        )

        model = builder.finalize(device=self.device)

        self.assertEqual(model.tri_count, 2)
        np.testing.assert_allclose(model.tri_opacity.numpy(), [0.4, 0.4], atol=1e-6, rtol=1e-6)

    def test_soft_mesh_uses_tet_mesh_opacity_when_opacity_is_none(self):
        """Verify soft meshes inherit display opacity from their TetMesh."""
        builder = newton.ModelBuilder()
        mesh = self._make_soft_tet_mesh(opacity=0.35)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=mesh,
        )

        model = builder.finalize(device=self.device)

        self.assertEqual(model.tri_count, 4)
        np.testing.assert_allclose(
            model.tri_opacity.numpy(),
            np.full(4, 0.35, dtype=np.float32),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_explicit_soft_mesh_opacity_overrides_tet_mesh_opacity(self):
        """Verify explicit soft mesh opacity overrides opacity embedded in TetMesh."""
        builder = newton.ModelBuilder()
        mesh = self._make_soft_tet_mesh(opacity=0.35)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=mesh,
            opacity=0.75,
        )

        model = builder.finalize(device=self.device)

        self.assertEqual(model.tri_count, 4)
        np.testing.assert_allclose(
            model.tri_opacity.numpy(),
            np.full(4, 0.75, dtype=np.float32),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_viewer_logs_triangle_mesh_opacity_from_model(self):
        """Verify triangle mesh logging passes model triangle opacity to viewers."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
            opacity=0.4,
        )
        model = builder.finalize(device=self.device)
        state = model.state()

        viewer = _TriangleOpacityProbe()
        viewer.set_model(model)
        viewer.log_state(state)

        self.assertIn("/model/triangles", viewer.mesh_opacities)
        np.testing.assert_allclose(viewer.mesh_opacities["/model/triangles"], 0.4, atol=1e-6, rtol=1e-6)

    def test_viewer_warns_for_wrong_triangle_opacity_count(self):
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
        )
        model = builder.finalize(device=self.device)
        model.tri_opacity = wp.array([0.2, 0.4, 0.6], dtype=wp.float32, device=self.device)
        viewer = ViewerNull()
        viewer.set_model(model)

        with self.assertWarnsRegex(UserWarning, "3 values for 2 triangles"):
            groups, _ = viewer._get_triangle_opacity_groups()

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][2], 1.0)

    def test_viewer_caps_continuous_triangle_opacity_groups(self):
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=5,
            dim_y=4,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
        )
        model = builder.finalize(device=self.device)
        model.tri_opacity = wp.array(
            np.linspace(0.0, 1.0, model.tri_count, dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        viewer = ViewerNull()
        viewer.set_model(model)

        with self.assertWarnsRegex(UserWarning, "quantizing"):
            groups, _ = viewer._get_triangle_opacity_groups()

        self.assertLessEqual(len(groups), MAX_TRIANGLE_OPACITY_GROUPS)

    def test_viewer_caches_triangle_opacity_groups_until_mutated(self):
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=5,
            dim_y=4,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
        )
        model = builder.finalize(device=self.device)
        opacities = np.full(model.tri_count, 0.5, dtype=np.float32)
        opacities[: model.tri_count // 2] = 0.25
        model.tri_opacity = wp.array(opacities, dtype=wp.float32, device=self.device)
        viewer = ViewerNull()
        viewer.set_model(model)

        groups_first, _ = viewer._get_triangle_opacity_groups()
        groups_second, stale_second = viewer._get_triangle_opacity_groups()
        self.assertIs(groups_second, groups_first)
        self.assertEqual(stale_second, [])

        # An in-place mutation must invalidate the cached groups.
        opacities[:] = 0.75
        wp.copy(model.tri_opacity, wp.array(opacities, dtype=wp.float32, device=self.device))
        groups_third, stale_third = viewer._get_triangle_opacity_groups()
        self.assertIsNot(groups_third, groups_first)
        self.assertEqual(len(groups_third), 1)
        self.assertAlmostEqual(groups_third[0][2], 0.75, places=6)
        self.assertEqual(stale_third, groups_first)

    def test_opaque_and_transparent_shapes_use_separate_batches(self):
        builder = newton.ModelBuilder()
        body0 = builder.add_body(mass=1.0)
        body1 = builder.add_body(mass=1.0)
        builder.add_shape_box(body=body0, hx=0.1, hy=0.2, hz=0.3, opacity=1.0)
        builder.add_shape_box(body=body1, hx=0.1, hy=0.2, hz=0.3, opacity=0.5)
        model = builder.finalize(device=self.device)
        viewer = ViewerNull()

        viewer.set_model(model)

        self.assertEqual(sorted(batch.transparent for batch in viewer._shape_instances.values()), [False, True])

    def test_viewer_gl_opacity_kernel_sets_dirty_and_regroup_flags(self):
        device = wp.get_device("cpu")
        common_inputs = [
            wp.array([wp.transform_identity()], dtype=wp.transformf, device=device),
            wp.array([-1], dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.transformf, device=device),
            wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=device),
            wp.array([int(newton.GeoType.BOX)], dtype=wp.int32, device=device),
            wp.array([-1], dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.transform_identity(),
            wp.array([0], dtype=wp.int32, device=device),
        ]
        out_world_xforms = wp.empty(1, dtype=wp.transformf, device=device)
        out_vbo_xforms = wp.empty(1, dtype=wp.mat44, device=device)

        def get_flags(current_opacity, previous_opacity):
            flags = wp.zeros(2, dtype=wp.int32, device=device)
            wp.launch(
                _compute_shape_vbo_xforms,
                dim=1,
                inputs=[
                    *common_inputs,
                    wp.array([current_opacity], dtype=wp.float32, device=device),
                    wp.array([previous_opacity], dtype=wp.float32, device=device),
                    flags,
                    1,
                ],
                outputs=[out_world_xforms, out_vbo_xforms],
                device=device,
            )
            return flags.numpy()

        np.testing.assert_array_equal(get_flags(0.5, 1.0), [1, 1])
        np.testing.assert_array_equal(get_flags(0.5, 0.4), [1, 0])

    def test_viewer_gl_rebuilds_opacity_dependent_caches(self):
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.objects = {}
        viewer._shape_instances = {"stale": object()}
        viewer._gaussian_instances = [object()]
        viewer._sdf_isomesh_instances = {0: object()}
        viewer._sdf_isomesh_populated = True
        viewer.model_shape_color = object()
        viewer.model_shape_opacity = object()
        viewer._shape_to_slot = np.array([0], dtype=np.int32)
        viewer._slot_to_shape = np.array([0], dtype=np.int32)
        viewer._slot_to_shape_wp = object()
        viewer._shape_to_batch = [object()]
        viewer._shape_transparent_mask = np.array([False])
        viewer._populate_shapes = Mock()
        viewer._rebuild_gl_shape_caches = Mock()

        viewer._rebuild_shape_batches_for_opacity_groups()

        viewer._populate_shapes.assert_called_once_with()
        viewer._rebuild_gl_shape_caches.assert_called_once_with()
        self.assertTrue(viewer.model_changed)

    def test_ground_plane_keeps_checkerboard_material_with_resolved_shape_colors(self):
        """Verify the ground plane keeps its checkerboard material after color resolution."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)

        viewer = ViewerNull()
        viewer.set_model(model)

        batch = next(iter(viewer._shape_instances.values()))
        np.testing.assert_allclose(batch.materials.numpy()[0], [0.5, 0.0, 1.0, 0.0], atol=1e-6, rtol=1e-6)

    def test_viewer_syncs_runtime_shape_colors_from_model(self):
        """Verify the viewer reflects runtime updates written to ``model.shape_color``."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_box(
            body=body,
            hx=0.1,
            hy=0.2,
            hz=0.3,
            color=(0.1, 0.2, 0.3),
        )
        model = builder.finalize(device=self.device)
        state = model.state()

        viewer = _ShapeColorProbe()
        viewer.set_model(model)
        viewer.log_state(state)
        np.testing.assert_allclose(viewer.last_colors[0], [0.1, 0.2, 0.3], atol=1e-6, rtol=1e-6)

        viewer.last_colors = None
        model.shape_color[shape : shape + 1].fill_(wp.vec3(0.8, 0.2, 0.1))
        viewer.log_state(state)

        self.assertIsNotNone(viewer.last_colors)
        np.testing.assert_allclose(viewer.last_colors[0], [0.8, 0.2, 0.1], atol=1e-6, rtol=1e-6)

    def test_viewer_syncs_runtime_shape_opacities_from_model(self):
        """Verify the viewer reflects runtime updates written to ``model.shape_opacity``."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_box(
            body=body,
            hx=0.1,
            hy=0.2,
            hz=0.3,
            opacity=0.4,
        )
        model = builder.finalize(device=self.device)
        state = model.state()

        viewer = _ShapeColorProbe()
        viewer.set_model(model)
        viewer.log_state(state)
        np.testing.assert_allclose(viewer.last_opacities[0], 0.4, atol=1e-6, rtol=1e-6)

        viewer.last_opacities = None
        model.shape_opacity[shape : shape + 1].fill_(0.7)
        viewer.log_state(state)

        self.assertIsNotNone(viewer.last_opacities)
        np.testing.assert_allclose(viewer.last_opacities[0], 0.7, atol=1e-6, rtol=1e-6)

    def test_viewer_builds_inverse_shape_color_slot_mapping(self):
        """Verify packed color slots can be mapped back to model shape indices."""
        builder = newton.ModelBuilder()
        body0 = builder.add_body(mass=1.0)
        body1 = builder.add_body(mass=1.0)
        builder.add_shape_box(body=body0, hx=0.1, hy=0.2, hz=0.3)
        builder.add_shape_box(body=body1, hx=0.2, hy=0.1, hz=0.3)
        builder.add_shape_sphere(body=body1, radius=0.15)

        model = builder.finalize(device=self.device)
        viewer = ViewerNull()
        viewer.set_model(model)

        packed_shape_colors = viewer.model_shape_color
        shape_to_slot = viewer._shape_to_slot
        slot_to_shape = viewer._slot_to_shape

        self.assertIsNotNone(packed_shape_colors)
        self.assertIsNotNone(shape_to_slot)
        self.assertIsNotNone(slot_to_shape)
        assert packed_shape_colors is not None
        assert shape_to_slot is not None
        assert slot_to_shape is not None
        self.assertEqual(len(slot_to_shape), len(packed_shape_colors))

        rendered_shapes = np.flatnonzero(shape_to_slot >= 0)
        self.assertEqual(len(rendered_shapes), len(slot_to_shape))
        np.testing.assert_array_equal(np.sort(slot_to_shape), rendered_shapes)
        for shape_idx in rendered_shapes:
            slot = int(shape_to_slot[shape_idx])
            self.assertEqual(int(slot_to_shape[slot]), int(shape_idx))

    def test_viewer_repacks_runtime_shape_colors_into_packed_order(self):
        """Verify runtime color sync repacks model colors into packed viewer order."""
        builder = newton.ModelBuilder()
        body0 = builder.add_body(mass=1.0)
        body1 = builder.add_body(mass=1.0)
        body2 = builder.add_body(mass=1.0)
        shape0 = builder.add_shape_box(body=body0, hx=0.1, hy=0.2, hz=0.3)
        shape1 = builder.add_shape_sphere(body=body1, radius=0.15)
        # Reuse the same box geometry so shapes 0 and 2 share a render batch.
        shape2 = builder.add_shape_box(body=body2, hx=0.1, hy=0.2, hz=0.3)

        model = builder.finalize(device=self.device)
        viewer = ViewerNull()
        viewer.set_model(model)

        packed_shape_colors = viewer.model_shape_color
        slot_to_shape = viewer._slot_to_shape
        self.assertIsNotNone(packed_shape_colors)
        self.assertIsNotNone(slot_to_shape)
        assert packed_shape_colors is not None
        assert slot_to_shape is not None

        expected_slot_order = np.array([shape0, shape2, shape1], dtype=np.int32)
        np.testing.assert_array_equal(slot_to_shape, expected_slot_order)

        updated_colors = {
            shape0: (0.8, 0.1, 0.2),
            shape1: (0.1, 0.9, 0.3),
            shape2: (0.2, 0.3, 0.95),
        }
        for shape_idx, color in updated_colors.items():
            model.shape_color[shape_idx : shape_idx + 1].fill_(wp.vec3(*color))

        viewer._sync_shape_colors_from_model()

        expected_colors = model.shape_color.numpy()[slot_to_shape]
        np.testing.assert_allclose(packed_shape_colors.numpy(), expected_colors, atol=1e-6, rtol=1e-6)

    def test_viewer_repacks_runtime_shape_opacities_into_packed_order(self):
        """Verify runtime opacity sync repacks model opacities into packed viewer order."""
        builder = newton.ModelBuilder()
        body0 = builder.add_body(mass=1.0)
        body1 = builder.add_body(mass=1.0)
        body2 = builder.add_body(mass=1.0)
        shape0 = builder.add_shape_box(body=body0, hx=0.1, hy=0.2, hz=0.3, opacity=0.3)
        shape1 = builder.add_shape_sphere(body=body1, radius=0.15, opacity=0.4)
        # Reuse the same box geometry so shapes 0 and 2 share a render batch.
        shape2 = builder.add_shape_box(body=body2, hx=0.1, hy=0.2, hz=0.3, opacity=0.5)

        model = builder.finalize(device=self.device)
        viewer = ViewerNull()
        viewer.set_model(model)

        packed_shape_opacities = viewer.model_shape_opacity
        slot_to_shape = viewer._slot_to_shape
        self.assertIsNotNone(packed_shape_opacities)
        self.assertIsNotNone(slot_to_shape)
        assert packed_shape_opacities is not None
        assert slot_to_shape is not None

        expected_slot_order = np.array([shape0, shape2, shape1], dtype=np.int32)
        np.testing.assert_array_equal(slot_to_shape, expected_slot_order)

        updated_opacities = {
            shape0: 0.8,
            shape1: 0.6,
            shape2: 0.7,
        }
        for shape_idx, opacity in updated_opacities.items():
            model.shape_opacity[shape_idx : shape_idx + 1].fill_(opacity)

        viewer._sync_shape_opacities_from_model()

        expected_opacities = model.shape_opacity.numpy()[slot_to_shape]
        np.testing.assert_allclose(packed_shape_opacities.numpy(), expected_opacities, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
