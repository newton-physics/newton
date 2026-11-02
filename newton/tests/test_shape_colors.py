# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest
import warnings
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import warp as wp

import newton
from newton._src.viewer.gl.opengl import RendererGL
from newton._src.viewer.gl.shaders import _with_shader_define, shape_fragment_shader
from newton._src.viewer.viewer import (
    _DEFAULT_LAYER_ID,
    MAX_TRIANGLE_APPEARANCE_GROUPS,
    MAX_TRIANGLE_OPACITY_GROUPS,
    Layer,
)
from newton._src.viewer.viewer_gl import ViewerGL, _compute_shape_vbo_xforms
from newton.viewer import ViewerNull


class _ShapeColorProbe(ViewerNull):
    """Captures per-batch appearance values passed through ``log_instances``."""

    def __init__(self):
        """Initialize the probe with storage for the latest appearance values."""
        super().__init__(num_frames=1)
        self.last_colors = None
        self.last_opacities = None

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False, opacities=None):
        """Capture the most recent instance appearance values sent to the viewer."""
        self.last_colors = None if colors is None else colors.numpy().copy()
        self.last_opacities = None if opacities is None else opacities.numpy().copy()


class _TriangleAppearanceProbe(ViewerNull):
    """Capture mesh appearance values passed through ``log_mesh``."""

    def __init__(self):
        """Initialize the probe with storage for mesh appearance values."""
        super().__init__(num_frames=1)
        self.mesh_colors = {}
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
        color=None,
        opacity=None,
    ):
        """Capture color and opacity for visible triangle mesh logs."""
        if not hidden:
            self.mesh_colors[name] = color
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

    def _make_soft_tet_mesh(self):
        """Create a one-tet deformable mesh."""
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
        return newton.TetMesh(vertices, indices)

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

    def test_triangle_color_rejects_invalid_values_before_mutation(self):
        """Reject malformed triangle colors before appending geometry."""
        invalid_colors = ((0.1, 0.2), (-0.1, 0.2, 0.3), (0.1, 1.1, 0.3), (0.1, float("nan"), 0.3))
        for invalid_color in invalid_colors:
            with self.subTest(color=invalid_color):
                builder = newton.ModelBuilder()
                for position in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
                    builder.add_particle(pos=position, vel=(0.0, 0.0, 0.0), mass=1.0)

                with self.assertRaisesRegex(ValueError, "Triangle color"):
                    builder.add_triangle(0, 1, 2, color=invalid_color)

                self.assertEqual(len(builder.tri_indices), 0)
                self.assertEqual(len(builder.tri_color), 0)

    def test_triangle_surface_appearance_arguments_are_adjacent_keyword_only(self):
        """Keep color immediately before opacity on every surface-triangle builder API."""
        method_names = (
            "add_triangle",
            "add_triangles",
            "add_cloth_grid",
            "add_cloth_mesh",
            "add_soft_grid",
            "add_soft_mesh",
        )
        for method_name in method_names:
            with self.subTest(method=method_name):
                parameters = list(inspect.signature(getattr(newton.ModelBuilder, method_name)).parameters.values())
                color_index = next(index for index, parameter in enumerate(parameters) if parameter.name == "color")
                self.assertEqual(parameters[color_index + 1].name, "opacity")
                self.assertEqual(parameters[color_index].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertEqual(parameters[color_index + 1].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_triangle_opacity_array_rejects_wrong_length_before_mutation(self):
        """Reject mismatched triangle opacity arrays before appending geometry."""
        builder = newton.ModelBuilder()
        for position in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
            builder.add_particle(pos=position, vel=(0.0, 0.0, 0.0), mass=1.0)

        with self.assertRaisesRegex(ValueError, "exactly 1 values"):
            builder.add_triangles([0], [1], [2], opacity=[0.2, 0.4])

        self.assertEqual(len(builder.tri_indices), 0)
        self.assertEqual(len(builder.tri_opacity), 0)

    def test_cloth_opacity_defaults_to_opaque(self):
        """Use the canonical color and opaque alpha for unstyled cloth triangles."""
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
            model.tri_color.numpy(),
            np.tile([0.7, 0.5, 0.3], (2, 1)),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            model.tri_opacity.numpy(),
            np.ones(2, dtype=np.float32),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_cloth_grid_stores_explicit_surface_appearance(self):
        """Store cloth color and opacity on every generated surface triangle."""
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
            color=(0.3, 0.5, 0.7),
            opacity=0.4,
        )

        model = builder.finalize(device=self.device)

        self.assertEqual(model.tri_count, 2)
        np.testing.assert_allclose(
            model.tri_color.numpy(),
            np.tile([0.3, 0.5, 0.7], (2, 1)),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(model.tri_opacity.numpy(), [0.4, 0.4], atol=1e-6, rtol=1e-6)

    def test_cloth_grid_stores_per_triangle_surface_appearance(self):
        """Store one color and opacity value per generated cloth triangle."""
        colors = [(0.1, 0.2, 0.3), (0.7, 0.8, 0.9)]
        opacities = [0.25, 0.75]
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
            color=colors,
            opacity=opacities,
        )

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.tri_color.numpy(), colors, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(model.tri_opacity.numpy(), opacities, atol=1e-6, rtol=1e-6)

    def test_soft_grid_stores_per_triangle_surface_appearance(self):
        """Store one color and opacity value per generated soft-grid face."""
        colors = np.linspace(0.1, 0.9, 36, dtype=np.float32).reshape(12, 3)
        opacities = np.linspace(0.2, 0.8, 12, dtype=np.float32)
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=1.0,
            cell_y=1.0,
            cell_z=1.0,
            density=1.0,
            k_mu=1.0,
            k_lambda=1.0,
            k_damp=0.0,
            color=colors,
            opacity=opacities,
        )

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.tri_color.numpy(), colors, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(model.tri_opacity.numpy(), opacities, atol=1e-6, rtol=1e-6)

    def test_soft_mesh_stores_explicit_surface_appearance(self):
        """Store soft-mesh color and opacity on every generated surface triangle."""
        builder = newton.ModelBuilder()
        mesh = self._make_soft_tet_mesh()
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=mesh,
            color=(0.2, 0.4, 0.6),
            opacity=0.35,
        )

        model = builder.finalize(device=self.device)

        self.assertEqual(model.tri_count, 4)
        np.testing.assert_allclose(
            model.tri_color.numpy(),
            np.tile([0.2, 0.4, 0.6], (4, 1)),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            model.tri_opacity.numpy(),
            np.full(4, 0.35, dtype=np.float32),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_soft_mesh_stores_per_triangle_surface_appearance(self):
        """Store one color and opacity value per extracted soft-mesh face."""
        colors = np.array(
            ((0.1, 0.2, 0.3), (0.2, 0.3, 0.4), (0.3, 0.4, 0.5), (0.4, 0.5, 0.6)),
            dtype=np.float32,
        )
        opacities = np.array((0.2, 0.4, 0.6, 0.8), dtype=np.float32)
        builder = newton.ModelBuilder()
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=self._make_soft_tet_mesh(),
            color=colors,
            opacity=opacities,
        )

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.tri_color.numpy(), colors, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(model.tri_opacity.numpy(), opacities, atol=1e-6, rtol=1e-6)

    def test_soft_mesh_defaults_to_opaque_surface_appearance(self):
        """Use the canonical color and opaque alpha for an unstyled soft mesh."""
        builder = newton.ModelBuilder()
        mesh = self._make_soft_tet_mesh()
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
            model.tri_color.numpy(),
            np.tile([0.7, 0.5, 0.3], (4, 1)),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            model.tri_opacity.numpy(),
            np.ones(4, dtype=np.float32),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_viewer_logs_triangle_mesh_appearance_from_model(self):
        """Pass model triangle color and opacity to viewers."""
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
            color=(0.2, 0.4, 0.6),
            opacity=0.4,
        )
        model = builder.finalize(device=self.device)
        state = model.state()

        viewer = _TriangleAppearanceProbe()
        viewer.set_model(model)
        viewer.log_state(state)

        self.assertIn("/model/triangles", viewer.mesh_opacities)
        np.testing.assert_allclose(viewer.mesh_colors["/model/triangles"], (0.2, 0.4, 0.6), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(viewer.mesh_opacities["/model/triangles"], 0.4, atol=1e-6, rtol=1e-6)

    def test_viewer_warns_for_wrong_triangle_opacity_count(self):
        """Fall back to opaque triangles when the model array length is invalid."""
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
            groups, _ = viewer._get_triangle_appearance_groups()

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][3], 1.0)

    def test_viewer_caps_continuous_triangle_opacity_groups(self):
        """Bound draw-call growth for continuously varying triangle opacity."""
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
            groups, _ = viewer._get_triangle_appearance_groups()

        self.assertLessEqual(len(groups), MAX_TRIANGLE_OPACITY_GROUPS)

    def test_viewer_caps_continuous_triangle_appearance_groups(self):
        """Bound draw-call growth for continuously varying triangle colors and opacity."""
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
        values = np.linspace(0.0, 1.0, model.tri_count, dtype=np.float32)
        model.tri_color = wp.array(
            np.column_stack((values, values[::-1], np.mod(values * 3.0, 1.0))),
            dtype=wp.vec3,
            device=self.device,
        )
        model.tri_opacity = wp.array(values, dtype=wp.float32, device=self.device)
        viewer = ViewerNull()
        viewer.set_model(model)

        with self.assertWarnsRegex(UserWarning, "quantizing"):
            groups, _ = viewer._get_triangle_appearance_groups()

        self.assertLessEqual(len(groups), MAX_TRIANGLE_APPEARANCE_GROUPS)

    def test_viewer_caches_triangle_appearance_groups_until_opacity_mutates(self):
        """Reuse triangle groups until an in-place opacity mutation occurs."""
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

        groups_first, _ = viewer._get_triangle_appearance_groups()
        groups_second, stale_second = viewer._get_triangle_appearance_groups()
        self.assertIs(groups_second, groups_first)
        self.assertEqual(stale_second, [])

        # An in-place mutation must invalidate the cached groups.
        opacities[:] = 0.75
        wp.copy(model.tri_opacity, wp.array(opacities, dtype=wp.float32, device=self.device))
        groups_third, stale_third = viewer._get_triangle_appearance_groups()
        self.assertIsNot(groups_third, groups_first)
        self.assertEqual(len(groups_third), 1)
        self.assertAlmostEqual(groups_third[0][3], 0.75, places=6)
        self.assertEqual(stale_third, groups_first)

    def test_viewer_invalidates_triangle_appearance_groups_when_color_mutates(self):
        """Rebuild triangle groups after an in-place color mutation."""
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
        viewer = ViewerNull()
        viewer.set_model(model)

        groups_first, _ = viewer._get_triangle_appearance_groups()
        colors = np.tile([0.1, 0.2, 0.3], (model.tri_count, 1)).astype(np.float32)
        wp.copy(model.tri_color, wp.array(colors, dtype=wp.vec3, device=self.device))
        groups_second, stale_second = viewer._get_triangle_appearance_groups()

        self.assertIsNot(groups_second, groups_first)
        np.testing.assert_allclose(groups_second[0][2], (0.1, 0.2, 0.3), atol=1e-6, rtol=1e-6)
        self.assertEqual(stale_second, groups_first)

    def test_opaque_and_transparent_shapes_use_separate_batches(self):
        """Separate opaque and transparent instances into render-pass batches."""
        builder = newton.ModelBuilder()
        body0 = builder.add_body(mass=1.0)
        body1 = builder.add_body(mass=1.0)
        builder.add_shape_box(body=body0, hx=0.1, hy=0.2, hz=0.3, opacity=1.0)
        builder.add_shape_box(body=body1, hx=0.1, hy=0.2, hz=0.3, opacity=0.5)
        model = builder.finalize(device=self.device)
        viewer = ViewerNull()

        viewer.set_model(model)

        self.assertEqual(sorted(batch.transparent for batch in viewer._shape_instances.values()), [False, True])

    def test_viewer_gl_splits_mixed_opacity_instance_batches(self):
        """Split public GL instance batches across opaque and transparent passes."""

        class FakeMesh:
            pass

        class FakeMeshInstancer:
            def __init__(self, num_instances, mesh, *, enable_cuda_interop=False):
                self.num_instances = num_instances
                self.mesh = mesh
                self.hidden = False
                self.active_instances = num_instances
                self.last_xforms = None
                self.last_opacities = None

            def update_from_transforms(self, xforms, scales, colors, materials, opacities):
                self.active_instances = 0 if xforms is None else len(xforms)
                self.last_xforms = None if xforms is None else xforms.numpy().copy()
                self.last_opacities = None if opacities is None else opacities.numpy().copy()

            def has_transparency(self):
                return bool(self.last_opacities is not None and np.any(self.last_opacities < 0.999))

        viewer = ViewerGL.__new__(ViewerGL)
        viewer._enable_cuda_interop = ViewerGL.CudaInterop.NONE
        viewer._layers = {_DEFAULT_LAYER_ID: Layer(_DEFAULT_LAYER_ID)}
        viewer._active_layer_id = _DEFAULT_LAYER_ID
        viewer.objects = {"/mesh": FakeMesh()}

        xforms = wp.array(
            [
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
            ],
            dtype=wp.transform,
            device="cpu",
        )
        scales = wp.array([wp.vec3(1.0)] * 3, dtype=wp.vec3, device="cpu")
        colors = wp.array([wp.vec3(0.5)] * 3, dtype=wp.vec3, device="cpu")
        materials = wp.array([wp.vec4(0.5)] * 3, dtype=wp.vec4, device="cpu")
        opacities = wp.array([1.0, 0.5, 1.0], dtype=wp.float32, device="cpu")

        with (
            patch("newton._src.viewer.viewer_gl.MeshGL", FakeMesh),
            patch("newton._src.viewer.viewer_gl.MeshInstancerGL", FakeMeshInstancer),
        ):
            viewer.log_instances("/instances", "/mesh", xforms, scales, colors, materials, opacities=opacities)

        opaque = viewer.objects["/instances"]
        transparent = viewer.objects["/instances/__transparent__"]
        np.testing.assert_allclose(opaque.last_xforms[:, 0], [0.0, 2.0])
        np.testing.assert_allclose(opaque.last_opacities, [1.0, 1.0])
        np.testing.assert_allclose(transparent.last_xforms[:, 0], [1.0])
        np.testing.assert_allclose(transparent.last_opacities, [0.5])
        self.assertFalse(opaque.hidden)
        self.assertFalse(transparent.hidden)

        renderer = RendererGL.__new__(RendererGL)
        opaque_objects, transparent_objects = renderer._split_transparent_objects(
            {"opaque": opaque, "transparent": transparent}, scene_has_transparency=True
        )
        self.assertEqual(opaque_objects, {"opaque": opaque})
        self.assertEqual(transparent_objects, [("transparent", transparent)])

    def test_viewer_gl_opacity_kernel_sets_dirty_and_regroup_flags(self):
        """Flag opacity changes and opaque-threshold crossings on the device."""
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
        """Rebuild all shape caches after an opacity pass transition."""

        class FakeMeshInstancer:
            pass

        viewer = ViewerGL.__new__(ViewerGL)
        viewer._layers = {"solverA": Layer("solverA")}
        viewer._active_layer_id = "solverA"
        viewer.objects = {
            "/layers/solverA/model/shapes/shape_0": FakeMeshInstancer(),
            "/layers/solverB/model/shapes/shape_0": FakeMeshInstancer(),
        }
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

        with patch("newton._src.viewer.gl.opengl.MeshInstancerGL", FakeMeshInstancer):
            viewer._rebuild_shape_batches_for_opacity_groups()

        viewer._populate_shapes.assert_called_once_with()
        viewer._rebuild_gl_shape_caches.assert_called_once_with()
        self.assertTrue(viewer.model_changed)
        self.assertNotIn("/layers/solverA/model/shapes/shape_0", viewer.objects)
        self.assertIn("/layers/solverB/model/shapes/shape_0", viewer.objects)

    @staticmethod
    def _make_transparency_renderer(oit_supported: bool):
        """Build a bare ``RendererGL`` with just the transparency state populated."""
        renderer = RendererGL.__new__(RendererGL)
        renderer._shape_transparent_shader = None
        renderer._oit_resolve_shader = None
        renderer._oit_fbo = None
        renderer._oit_supported = oit_supported
        renderer._oit_fallback_warned = False
        renderer._setup_oit_buffer = Mock(side_effect=lambda: setattr(renderer, "_oit_fbo", object()))
        return renderer

    def test_renderer_gl_lazily_creates_transparency_resources(self):
        """Defer transparency shaders and framebuffers until first use."""
        renderer = self._make_transparency_renderer(oit_supported=True)

        transparent_shader = object()
        resolve_shader = object()
        with (
            patch("newton._src.viewer.gl.opengl.ShaderShape", return_value=transparent_shader) as shape_shader,
            patch("newton._src.viewer.gl.opengl.OITResolveShader", return_value=resolve_shader) as oit_shader,
        ):
            self.assertTrue(renderer._ensure_transparency_resources())
            self.assertTrue(renderer._ensure_transparency_resources())

        shape_shader.assert_called_once_with(RendererGL.gl, enable_transparency=True)
        oit_shader.assert_called_once_with(RendererGL.gl)
        renderer._setup_oit_buffer.assert_called_once_with()

    def test_renderer_gl_warns_once_when_oit_is_unsupported(self):
        """Report unsupported weighted OIT once instead of failing the frame."""
        renderer = self._make_transparency_renderer(oit_supported=False)

        with (
            patch("newton._src.viewer.gl.opengl.ShaderShape", return_value=object()),
            patch("newton._src.viewer.gl.opengl.OITResolveShader", return_value=object()),
        ):
            with self.assertWarnsRegex(UserWarning, "Falling back to unsorted alpha blending"):
                self.assertFalse(renderer._ensure_transparency_resources())

            # A second frame must not re-warn, but must still report no OIT.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertFalse(renderer._ensure_transparency_resources())

        self.assertEqual(caught, [])

    def test_render_scene_falls_back_to_alpha_blending_without_oit(self):
        """Render transparency with alpha blending when weighted OIT is unavailable."""
        renderer = RendererGL.__new__(RendererGL)
        transparent_shader = MagicMock()
        renderer._shape_shader = MagicMock()
        renderer._shape_transparent_shader = transparent_shader
        renderer.draw_sky = False
        renderer.draw_wireframe = False
        renderer.draw_edges = False
        renderer.msaa_samples = 4
        renderer._frame_msaa_fbo = object()
        renderer._ensure_transparency_resources = Mock(return_value=False)
        renderer._update_shape_shader = Mock()
        renderer._draw_objects = Mock()
        renderer._resolve_msaa_frame = Mock()
        renderer._render_transparent_objects = Mock()
        renderer._render_blended_transparent_objects = Mock()

        transparent_object = Mock(hidden=False)
        transparent_object.has_transparency = Mock(return_value=True)
        opaque_object = Mock(hidden=False)
        opaque_object.has_transparency = Mock(return_value=False)
        objects = {"transparent": transparent_object, "opaque": opaque_object}

        with patch.object(RendererGL, "gl", MagicMock()):
            msaa_resolved = renderer._render_scene(objects, scene_has_transparency=True)

        self.assertFalse(msaa_resolved)
        transparent_shader.set_oit_enabled.assert_called_once_with(False)
        renderer._render_blended_transparent_objects.assert_called_once_with([("transparent", transparent_object)])
        renderer._render_transparent_objects.assert_not_called()
        # The opaque MSAA target stays multisampled; no early resolve is needed.
        renderer._resolve_msaa_frame.assert_not_called()
        renderer._draw_objects.assert_called_once_with({"opaque": opaque_object})

    def test_oit_depth_weight_discriminates_depth(self):
        """Weight nearer transparent fragments above farther ones.

        Mirrors the weight computed by ``shape_fragment_shader`` in
        ``newton/_src/viewer/gl/shaders.py``. The previous weight saturated its
        clamp ceiling at every depth, which reduced the resolve to a plain
        alpha-weighted average with no depth ordering.
        """

        def oit_weight(alpha: float, normalized_depth: float) -> float:
            depth_weight = float(
                np.clip(
                    10.0 / (1e-5 + (2.0 * normalized_depth) ** 2 + (0.6 * normalized_depth) ** 6),
                    1e-2,
                    3e3,
                )
            )
            return alpha * depth_weight

        for alpha in (0.35, 0.5, 0.55):
            with self.subTest(alpha=alpha):
                near = oit_weight(alpha, 0.2)
                far = oit_weight(alpha, 0.9)
                self.assertGreater(near, far)
                # Neither sample may sit on a clamp bound, or ordering is lost again.
                self.assertLess(near, 3e3 * alpha)
                self.assertGreater(far, 1e-2 * alpha)

    def test_transparency_shader_normalizes_depth_by_reference_distance(self):
        """Build the OIT weight from the scene-scale-independent depth uniform."""
        transparent_source = _with_shader_define(shape_fragment_shader, "ENABLE_TRANSPARENCY")

        self.assertIn("uniform float oit_inv_depth_reference;", transparent_source)
        self.assertIn("ViewDepth * oit_inv_depth_reference", transparent_source)
        # The window-space depth weight that saturated its clamp must not come back.
        self.assertNotIn("1e8", transparent_source)
        self.assertNotIn("gl_FragCoord.z * 0.9", transparent_source)
        # The opaque variant compiles the same source without the transparency block.
        self.assertNotIn("#define ENABLE_TRANSPARENCY", shape_fragment_shader)

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
