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
        super().__init__(num_frames=1)
        self.last_colors = None

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        del name, mesh, xforms, scales, materials, hidden
        self.last_colors = None if colors is None else colors.numpy().copy()


class TestShapeColors(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device()

    def _make_tetra_mesh(self, color=None):
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

    def test_add_shape_methods_place_color_before_label(self):
        method_names = [
            "add_shape_plane",
            "add_shape_sphere",
            "add_shape_ellipsoid",
            "add_shape_box",
            "add_shape_capsule",
            "add_shape_cylinder",
            "add_shape_cone",
            "add_shape_mesh",
            "add_shape_convex_hull",
            "add_shape_heightfield",
            "add_shape_gaussian",
        ]

        for method_name in method_names:
            with self.subTest(method=method_name):
                parameters = list(inspect.signature(getattr(newton.ModelBuilder, method_name)).parameters.values())
                parameter_names = [parameter.name for parameter in parameters]

                self.assertIn("color", parameter_names)
                self.assertIn("label", parameter_names)

                color_index = parameter_names.index("color")
                label_index = parameter_names.index("label")

                self.assertEqual(color_index + 1, label_index)
                self.assertEqual(parameters[color_index].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def test_add_shape_box_accepts_positional_color_before_label(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_box(
            body,
            None,
            0.1,
            0.2,
            0.3,
            None,
            False,
            (0.25, 0.5, 0.75),
            "box_shape",
        )

        model = builder.finalize(device=self.device)

        self.assertEqual(model.shape_label[shape], "box_shape")
        np.testing.assert_allclose(model.shape_color.numpy()[shape], [0.25, 0.5, 0.75], atol=1e-6, rtol=1e-6)

    def test_explicit_shape_color_is_stored_on_model(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_box(
            body=body,
            hx=0.1,
            hy=0.2,
            hz=0.3,
            color=(0.25, 0.5, 0.75),
        )

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.shape_color.numpy()[shape], [0.25, 0.5, 0.75], atol=1e-6, rtol=1e-6)

    def test_collision_shape_without_explicit_color_uses_default_palette(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_box(body=body, hx=0.1, hy=0.2, hz=0.3)

        model = builder.finalize(device=self.device)
        viewer = ViewerNull()
        expected = np.array(viewer._shape_color_map(shape), dtype=np.float32)

        np.testing.assert_allclose(model.shape_color.numpy()[shape], expected, atol=1e-6, rtol=1e-6)

    def test_add_shape_mesh_uses_mesh_color_when_color_is_none(self):
        mesh = self._make_tetra_mesh(color=(0.2, 0.4, 0.6))
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        shape = builder.add_shape_mesh(body=body, mesh=mesh)

        model = builder.finalize(device=self.device)

        np.testing.assert_allclose(model.shape_color.numpy()[shape], [0.2, 0.4, 0.6], atol=1e-6, rtol=1e-6)

    def test_explicit_shape_color_overrides_mesh_color(self):
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

    def test_viewer_syncs_runtime_shape_colors_from_model(self):
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
