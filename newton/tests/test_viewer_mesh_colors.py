# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock

import numpy as np
import warp as wp

import newton
from newton._src.viewer.gl.opengl import RenderVertex, fill_vertex_data
from newton._src.viewer.viewer_gl import ViewerGL
from newton._src.viewer.viewer_viser import ViewerViser
from newton.viewer import ViewerNull


class _LogMeshProbe(ViewerNull):
    """Capture mesh colors forwarded by the shared deformable viewer path."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.logged_colors = None

    def log_mesh(self, _name, _points, _indices, *args, colors=None, **kwargs):
        self.logged_colors = colors


class _LegacyLogMeshProbe(ViewerNull):
    """Represent a third-party viewer with the pre-colors mesh signature."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.called = False

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
        self.called = True


class TestViewerMeshColors(unittest.TestCase):
    """Verify per-vertex colors throughout the shared mesh viewer path."""

    def test_log_triangles_forwards_distinct_deformable_colors(self):
        """Forward distinct colors for two disconnected deformable surfaces."""
        builder = newton.ModelBuilder()
        for pos in (
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(2.0, 0.0, 0.0),
            wp.vec3(3.0, 0.0, 0.0),
            wp.vec3(2.0, 1.0, 0.0),
        ):
            builder.add_particle(pos, wp.vec3(0.0), 1.0)
        builder.add_triangle(0, 1, 2)
        builder.add_triangle(3, 4, 5)
        model = builder.finalize(device="cpu")
        model.particle_display_color = wp.array(
            [(1.0, 0.0, 0.0)] * 3 + [(0.0, 0.0, 1.0)] * 3,
            dtype=wp.vec3,
        )

        viewer = _LogMeshProbe()
        viewer.set_model(model)
        viewer._log_triangles(model.state())

        self.assertIs(viewer.logged_colors, model.particle_display_color)
        np.testing.assert_array_equal(
            viewer.logged_colors.numpy(),
            np.array([(1.0, 0.0, 0.0)] * 3 + [(0.0, 0.0, 1.0)] * 3, dtype=np.float32),
        )

    def test_log_triangles_preserves_legacy_uncolored_viewer_call(self):
        """Avoid sending the new keyword for an uncolored model."""
        builder = newton.ModelBuilder()
        builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0), 1.0)
        builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0), 1.0)
        builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0), 1.0)
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device="cpu")
        self.assertIsNone(model.particle_display_color)

        viewer = _LegacyLogMeshProbe()
        viewer.set_model(model)
        viewer._log_triangles(model.state())

        self.assertTrue(viewer.called)

    def test_fill_vertex_data_packs_colors_and_white_fallback(self):
        """Pack authored mesh colors while preserving a white no-color multiplier."""
        points = wp.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], dtype=wp.vec3)
        normals = wp.array([(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)], dtype=wp.vec3)
        uvs = wp.array([(0.0, 0.0), (1.0, 0.0)], dtype=wp.vec2)
        colors = wp.array([(1.0, 0.25, 0.0), (0.0, 0.5, 1.0)], dtype=wp.vec3)

        vertices = wp.zeros(2, dtype=RenderVertex)
        wp.launch(fill_vertex_data, dim=2, inputs=[points, normals, uvs, colors], outputs=[vertices])
        np.testing.assert_allclose(vertices.numpy()["color"], colors.numpy(), atol=1e-6)

        wp.launch(fill_vertex_data, dim=2, inputs=[points, normals, uvs, None], outputs=[vertices])
        np.testing.assert_allclose(vertices.numpy()["color"], np.ones((2, 3)), atol=1e-6)

    def test_gl_vertex_colors_override_uniform_without_tint(self):
        """Use a white mesh multiplier for authored vertex colors."""
        viewer = ViewerGL.__new__(ViewerGL)
        mesh = Mock()
        mesh.base_color = (0.7, 0.5, 0.3)
        mesh.material = (0.5, 0.0, 0.0, 0.0)
        viewer.objects = {"mesh": mesh}
        viewer._qualify = lambda name: name

        points = wp.array([(0.0, 0.0, 0.0)], dtype=wp.vec3)
        indices = wp.array([0, 0, 0], dtype=wp.int32)
        colors = wp.array([(1.0, 0.0, 0.0)], dtype=wp.vec3)
        viewer.log_mesh("mesh", points, indices, color=(0.2, 0.4, 0.6), colors=colors)

        self.assertEqual(mesh.base_color, (0.2, 0.4, 0.6))
        self.assertEqual(mesh.color, (1.0, 1.0, 1.0))
        self.assertIs(mesh.update.call_args.args[-1], colors)

        viewer.log_mesh("mesh", points, indices)
        self.assertEqual(mesh.color, mesh.base_color)
        self.assertIsNone(mesh.update.call_args.args[-1])

    def test_gl_texture_takes_precedence_over_vertex_colors(self):
        """Suppress vertex colors when a usable texture is supplied."""
        viewer = ViewerGL.__new__(ViewerGL)
        mesh = Mock()
        mesh.base_color = (0.7, 0.5, 0.3)
        mesh.material = (0.5, 0.0, 0.0, 0.0)
        viewer.objects = {"mesh": mesh}
        viewer._qualify = lambda name: name

        points = wp.array([(0.0, 0.0, 0.0)], dtype=wp.vec3)
        indices = wp.array([0, 0, 0], dtype=wp.int32)
        uvs = wp.array([(0.0, 0.0)], dtype=wp.vec2)
        colors = wp.array([(1.0, 0.0, 0.0)], dtype=wp.vec3)
        texture = np.full((1, 1, 3), 255, dtype=np.uint8)
        viewer.log_mesh("mesh", points, indices, uvs=uvs, texture=texture, colors=colors)

        self.assertIsNone(mesh.update.call_args.args[-1])
        self.assertEqual(mesh.color, mesh.base_color)

    def test_viser_color_mesh_preserves_vertex_colors(self):
        """Preserve per-vertex RGB values in the Viser trimesh path."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        mesh = ViewerViser._build_color_trimesh_mesh(points, indices, colors)

        self.assertIsNotNone(mesh)
        np.testing.assert_array_equal(
            mesh.visual.vertex_colors,
            np.array([[255, 0, 0, 255], [0, 128, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
