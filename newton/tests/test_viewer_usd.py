# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerRTX, ViewerUSD

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

    def _get_bound_preview_surface(self, prim):
        material, _binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        self.assertTrue(material)
        shader = UsdShade.Shader(material.GetPrim().GetChild("PreviewSurface"))
        self.assertTrue(shader)
        return shader

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

    def test_log_instances_authors_display_opacity(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
            dtype=wp.vec3,
        )
        indices = wp.array([0, 1, 2], dtype=wp.int32)
        xforms = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform)
        scales = wp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=wp.vec3)
        opacities = wp.array([0.25, 0.75], dtype=wp.float32)

        viewer.begin_frame(0.0)
        viewer.log_mesh("/opacity_mesh", points, indices)
        viewer.log_instances("/opacity_instances", "/opacity_mesh", xforms, scales, None, None, opacities=opacities)

        for i, expected in enumerate((0.25, 0.75)):
            prim = viewer.stage.GetPrimAtPath(f"/root/opacity_instances/instance_{i}")
            display_opacity = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayOpacity")

            self.assertTrue(display_opacity)
            np.testing.assert_allclose(
                np.asarray(display_opacity.Get(viewer._frame_index), dtype=np.float32),
                np.array([expected], dtype=np.float32),
                atol=1e-6,
            )

    def test_log_instances_authors_preview_surface_opacity(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
            dtype=wp.vec3,
        )
        indices = wp.array([0, 1, 2], dtype=wp.int32)
        xforms = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform)
        scales = wp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=wp.vec3)
        colors = wp.array([[0.2, 0.4, 0.6], [0.8, 0.3, 0.1]], dtype=wp.vec3)
        materials = wp.array([[0.25, 0.1, 0.0, 0.0], [0.75, 0.4, 0.0, 0.0]], dtype=wp.vec4)
        opacities = wp.array([0.25, 0.75], dtype=wp.float32)

        viewer.begin_frame(0.0)
        viewer.log_mesh("/opacity_mesh", points, indices)
        viewer.log_instances(
            "/opacity_instances",
            "/opacity_mesh",
            xforms,
            scales,
            colors,
            materials,
            opacities=opacities,
        )

        for i, expected in enumerate((0.25, 0.75)):
            prim = viewer.stage.GetPrimAtPath(f"/root/opacity_instances/instance_{i}")
            shader = self._get_bound_preview_surface(prim)
            self.assertAlmostEqual(shader.GetInput("opacity").Get(), expected, places=6)

        shader0 = self._get_bound_preview_surface(viewer.stage.GetPrimAtPath("/root/opacity_instances/instance_0"))
        np.testing.assert_allclose(np.asarray(shader0.GetInput("diffuseColor").Get()), [0.2, 0.4, 0.6], atol=1e-6)
        self.assertAlmostEqual(shader0.GetInput("roughness").Get(), 0.25, places=6)
        self.assertAlmostEqual(shader0.GetInput("metallic").Get(), 0.1, places=6)

    def test_log_instances_rejects_mismatched_opacity_count(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
            dtype=wp.vec3,
        )
        indices = wp.array([0, 1, 2], dtype=wp.int32)
        xforms = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform)
        scales = wp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=wp.vec3)
        opacities = wp.array([0.25, 0.5, 0.75], dtype=wp.float32)

        viewer.begin_frame(0.0)
        viewer.log_mesh("/opacity_mesh", points, indices)
        with self.assertRaisesRegex(ValueError, "Opacity arrays"):
            viewer.log_instances("/opacity_instances", "/opacity_mesh", xforms, scales, None, None, opacities=opacities)

    def test_log_mesh_authors_display_opacity(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
            dtype=wp.vec3,
        )
        indices = wp.array([0, 1, 2], dtype=wp.int32)

        viewer.begin_frame(0.0)
        viewer.log_mesh("/opacity_mesh_standalone", points, indices, opacity=0.35)

        prim = viewer.stage.GetPrimAtPath("/root/opacity_mesh_standalone")
        display_opacity = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayOpacity")

        self.assertTrue(display_opacity)
        np.testing.assert_allclose(
            np.asarray(display_opacity.Get(viewer._frame_index), dtype=np.float32),
            np.array([0.35], dtype=np.float32),
            atol=1e-6,
        )

    def test_log_mesh_authors_preview_surface_opacity(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
            dtype=wp.vec3,
        )
        indices = wp.array([0, 1, 2], dtype=wp.int32)

        viewer.begin_frame(0.0)
        viewer.log_mesh(
            "/opacity_mesh_standalone",
            points,
            indices,
            opacity=0.35,
            color=(0.1, 0.2, 0.3),
            roughness=0.2,
            metallic=0.4,
        )

        prim = viewer.stage.GetPrimAtPath("/root/opacity_mesh_standalone")
        shader = self._get_bound_preview_surface(prim)

        self.assertAlmostEqual(shader.GetInput("opacity").Get(), 0.35, places=6)
        self.assertEqual(shader.GetInput("opacityMode").Get(), "transparent")
        self.assertAlmostEqual(shader.GetInput("opacityThreshold").Get(), 0.0, places=6)
        np.testing.assert_allclose(np.asarray(shader.GetInput("diffuseColor").Get()), [0.1, 0.2, 0.3], atol=1e-6)
        self.assertAlmostEqual(shader.GetInput("roughness").Get(), 0.2, places=6)
        self.assertAlmostEqual(shader.GetInput("metallic").Get(), 0.4, places=6)

    def test_viewer_rtx_accepts_opacity_arguments(self):
        log_mesh_params = inspect.signature(ViewerRTX.log_mesh).parameters
        log_instances_params = inspect.signature(ViewerRTX.log_instances).parameters

        self.assertIn("opacity", log_mesh_params)
        self.assertIn("opacities", log_instances_params)

    def test_viewer_rtx_compensates_preview_surface_opacity(self):
        viewer = ViewerRTX.__new__(ViewerRTX)

        opacity = viewer._preview_surface_opacity_value(0.35)

        self.assertLess(opacity, 0.35)
        self.assertAlmostEqual(1.0 - (1.0 - opacity) ** viewer._PREVIEW_SURFACE_OPACITY_LAYERS, 0.35, places=6)
        self.assertEqual(viewer._preview_surface_ior_value(0.35), 1.0)
        self.assertAlmostEqual(viewer._preview_surface_opacity_value(1.0), 1.0, places=6)

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
