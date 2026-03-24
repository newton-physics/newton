# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull


class _ShapeColorProbe(ViewerNull):
    """Captures per-batch colors passed through ``log_instances``."""

    def __init__(self):
        """Initialize the probe with storage for the latest colors."""
        super().__init__(num_frames=1)
        self.last_colors = None

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """Capture the most recent instance colors sent to the viewer."""
        del name, mesh, xforms, scales, materials, hidden
        self.last_colors = None if colors is None else colors.numpy().copy()


class TestShapeColors(unittest.TestCase):
    """Regression tests for shape color storage and viewer synchronization."""

    def setUp(self):
        """Cache the active Warp device for model finalization."""
        self.device = wp.get_device()

    def _make_tetra_mesh(self, color=None):
        """Create a small tetrahedral mesh with an optional display color."""
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
        return newton.Mesh(vertices, indices, color=color)

    def test_collision_shape_without_explicit_color_uses_default_palette(self):
        """Verify collision shapes fall back to the default viewer palette."""
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

    def test_update_shape_colors_warns_and_writes_model_shape_color(self):
        """Verify deprecated viewer color updates warn and write through to the model."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_box(body=body, hx=0.1, hy=0.2, hz=0.3)
        model = builder.finalize(device=self.device)
        state = model.state()

        viewer = _ShapeColorProbe()
        viewer.set_model(model)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            viewer.update_shape_colors({shape: (0.7, 0.2, 0.9)})

        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        np.testing.assert_allclose(model.shape_color.numpy()[shape], [0.7, 0.2, 0.9], atol=1e-6, rtol=1e-6)

        viewer.last_colors = None
        viewer.log_state(state)
        self.assertIsNotNone(viewer.last_colors)
        np.testing.assert_allclose(viewer.last_colors[0], [0.7, 0.2, 0.9], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
