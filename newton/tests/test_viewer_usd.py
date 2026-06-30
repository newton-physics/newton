# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerUSD

if USD_AVAILABLE:
    from pxr import UsdGeom, UsdShade


def _build_box_model() -> newton.Model:
    builder = newton.ModelBuilder()
    builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        label="b",
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    builder.add_shape(
        body=0,
        type=newton.GeoType.BOX,
        scale=wp.vec3(0.5, 0.5, 0.5),
        cfg=cfg,
    )
    return builder.finalize()


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestViewerUSD(unittest.TestCase):
    def _make_viewer(self):
        temp_file = tempfile.NamedTemporaryFile(suffix=".usda", delete=False)
        temp_file.close()
        self.addCleanup(lambda: os.path.exists(temp_file.name) and os.remove(temp_file.name))
        viewer = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer.close)
        self.addCleanup(lambda: setattr(viewer, "output_path", ""))
        return viewer

    def _logged_texture_path(self, viewer, mesh_name: str) -> str:
        safe = mesh_name.replace("/", "_").lstrip("_")
        shader = UsdShade.Shader.Get(viewer.stage, f"/root/Materials/mat_{safe}/DiffuseTexture")
        asset_path = shader.GetInput("file").Get()
        return asset_path.path if hasattr(asset_path, "path") else str(asset_path)

    def test_log_points_keeps_per_point_wp_vec3_colors_for_three_points(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=wp.vec3,
        )
        colors = wp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=wp.vec3,
        )

        viewer.begin_frame(0.0)
        path = viewer.log_points("/points_per_point", points, radii=0.01, colors=colors)

        points_prim = UsdGeom.Points.Get(viewer.stage, path)
        display_color = np.asarray(points_prim.GetDisplayColorAttr().Get(viewer._frame_index), dtype=np.float32)
        interpolation = UsdGeom.Primvar(points_prim.GetDisplayColorAttr()).GetInterpolation()

        self.assertEqual(interpolation, UsdGeom.Tokens.vertex)
        np.testing.assert_allclose(display_color, colors.numpy(), atol=1e-6)

    def test_reuses_existing_layer_for_same_output_path(self):
        temp_file = tempfile.NamedTemporaryFile(suffix=".usda", delete=False)
        temp_file.close()
        self.addCleanup(lambda: os.path.exists(temp_file.name) and os.remove(temp_file.name))

        # Create first viewer and write some data into the stage.
        viewer1 = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer1.close)
        self.addCleanup(lambda: setattr(viewer1, "output_path", ""))

        viewer1.begin_frame(0.0)
        points = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
        colors = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3)
        path = viewer1.log_points("/points_from_viewer1", points, radii=0.01, colors=colors)

        # Ensure the prim written by viewer1 is present before creating viewer2.
        prim_before = UsdGeom.Points.Get(viewer1.stage, path).GetPrim()
        self.assertTrue(prim_before.IsValid())

        # Create second viewer for the same output path; this should reuse the same
        # underlying layer and clear any previous contents.
        viewer2 = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer2.close)
        self.addCleanup(lambda: setattr(viewer2, "output_path", ""))

        # Verify that the stage/layer reuse actually occurred.
        self.assertIsNotNone(viewer2.stage)
        self.assertIs(viewer1.stage.GetRootLayer(), viewer2.stage.GetRootLayer())

        # Verify that viewer2 cleared/overwrote viewer1's data.
        prim_after = UsdGeom.Points.Get(viewer2.stage, path).GetPrim()
        self.assertFalse(prim_after.IsValid())
        self.assertTrue(os.path.exists(temp_file.name))

    def test_generated_texture_path_stays_stable_after_clear_model(self):
        viewer = self._make_viewer()
        mesh_name = "/textured_mesh"
        points = wp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=wp.vec3,
        )
        indices = wp.array([0, 1, 2], dtype=wp.int32)
        uvs = wp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=wp.vec2)
        texture = np.array(
            [
                [[255, 0, 0, 255], [0, 255, 0, 255]],
                [[0, 0, 255, 255], [255, 255, 255, 255]],
            ],
            dtype=np.uint8,
        )

        viewer.begin_frame(0.0)
        viewer.log_mesh(mesh_name, points, indices, uvs=uvs, texture=texture)
        first_texture_path = self._logged_texture_path(viewer, mesh_name)

        viewer.clear_model()
        viewer.begin_frame(0.0)
        viewer.log_mesh(mesh_name, points, indices, uvs=uvs, texture=texture)
        second_texture_path = self._logged_texture_path(viewer, mesh_name)

        self.assertEqual(first_texture_path, second_texture_path)
        self.assertTrue(os.path.exists(first_texture_path))
        self.assertFalse([name for name in os.listdir(os.path.dirname(second_texture_path)) if name.endswith(".tmp")])

    def test_log_points_treats_wp_float_triplet_as_single_constant_color(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=wp.vec3,
        )
        color_triplet = wp.array([0.25, 0.5, 0.75], dtype=wp.float32)

        viewer.begin_frame(0.0)
        path = viewer.log_points("/points_constant", points, radii=0.01, colors=color_triplet)

        points_prim = UsdGeom.Points.Get(viewer.stage, path)
        display_color = np.asarray(points_prim.GetDisplayColorAttr().Get(viewer._frame_index), dtype=np.float32)
        interpolation = UsdGeom.Primvar(points_prim.GetDisplayColorAttr()).GetInterpolation()

        self.assertEqual(interpolation, UsdGeom.Tokens.constant)
        np.testing.assert_allclose(display_color, np.array([[0.25, 0.5, 0.75]], dtype=np.float32), atol=1e-6)

    def test_log_points_defaults_radii_when_omitted(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=wp.vec3,
        )

        viewer.begin_frame(0.0)
        path = viewer.log_points("/points_default_radii", points)

        points_prim = UsdGeom.Points.Get(viewer.stage, path)
        widths = np.asarray(points_prim.GetWidthsAttr().Get(viewer._frame_index), dtype=np.float32)
        interpolation = UsdGeom.Primvar(points_prim.GetWidthsAttr()).GetInterpolation()

        self.assertEqual(interpolation, UsdGeom.Tokens.constant)
        np.testing.assert_allclose(widths, np.array([0.2], dtype=np.float32), atol=1e-6)

    def test_named_layers_write_distinct_prim_namespaces(self):
        viewer = self._make_viewer()

        viewer.activate("solverA")
        viewer.set_model(_build_box_model())
        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        viewer.activate("solverB")
        viewer.set_model(_build_box_model())
        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        prim_a = viewer.stage.GetPrimAtPath("/root/layers/solverA/model/shapes/shape_0/instance_0")
        prim_b = viewer.stage.GetPrimAtPath("/root/layers/solverB/model/shapes/shape_0/instance_0")

        self.assertTrue(prim_a.IsValid())
        self.assertTrue(prim_b.IsValid())

    def test_remove_layer_preserves_sibling_usd_prims(self):
        viewer = self._make_viewer()

        viewer.activate("solverA")
        viewer.set_model(_build_box_model())
        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        viewer.activate("solverB")
        viewer.set_model(_build_box_model())
        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        viewer.remove_layer("solverA")

        prim_a = viewer.stage.GetPrimAtPath("/root/layers/solverA/model/shapes/shape_0/instance_0")
        prim_b = viewer.stage.GetPrimAtPath("/root/layers/solverB/model/shapes/shape_0/instance_0")

        self.assertFalse(prim_a.IsValid())
        self.assertTrue(prim_b.IsValid())

    def test_layer_visibility_hides_usd_instances(self):
        viewer = self._make_viewer()
        viewer.activate("solverA")
        viewer.set_model(_build_box_model())

        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        viewer.set_layer_visible("solverA", False)
        viewer.begin_frame(0.1)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        prim = viewer.stage.GetPrimAtPath("/root/layers/solverA/model/shapes/shape_0/instance_0")
        visibility = UsdGeom.Imageable(prim).GetVisibilityAttr().Get(viewer._frame_index)

        self.assertEqual(visibility, "invisible")


if __name__ == "__main__":
    unittest.main(verbosity=2)
