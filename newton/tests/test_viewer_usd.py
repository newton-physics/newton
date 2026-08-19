# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

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
        # Allocate a private work dir per test so the texture PNG and any
        # mkstemp siblings stay isolated from the system temp dir. Without
        # this, the .tmp check below is flaky on hosts where other tests or
        # processes also call mkstemp into the shared temp dir.
        work_dir = tempfile.mkdtemp(prefix="newton_test_viewer_usd_")
        self.addCleanup(lambda: shutil.rmtree(work_dir, ignore_errors=True))
        output_path = os.path.join(work_dir, "scene.usda")
        viewer = ViewerUSD(output_path=output_path, num_frames=1, points_as_spheres=False)
        self.addCleanup(viewer.close)
        self.addCleanup(lambda: setattr(viewer, "output_path", ""))
        viewer._test_work_dir = work_dir
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
        viewer1 = ViewerUSD(output_path=temp_file.name, num_frames=1, points_as_spheres=False)
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
        viewer2 = ViewerUSD(output_path=temp_file.name, num_frames=1, points_as_spheres=False)
        self.addCleanup(viewer2.close)
        self.addCleanup(lambda: setattr(viewer2, "output_path", ""))

        # Verify that the stage/layer reuse actually occurred.
        self.assertIsNotNone(viewer2.stage)
        self.assertIs(viewer1.stage.GetRootLayer(), viewer2.stage.GetRootLayer())

        # Verify that viewer2 cleared/overwrote viewer1's data.
        prim_after = UsdGeom.Points.Get(viewer2.stage, path).GetPrim()
        self.assertFalse(prim_after.IsValid())
        self.assertTrue(os.path.exists(temp_file.name))

    def test_partial_texture_rewrite_keeps_published_png_intact(self):
        """A crash mid-write must not corrupt the previously published PNG."""
        from PIL import Image

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
        tex_path = self._logged_texture_path(viewer, mesh_name)
        with open(tex_path, "rb") as published_file:
            published = published_file.read()

        real_save = Image.Image.save

        def partial_save(image, file_path, *args, **kwargs):
            buffer = io.BytesIO()
            kwargs.setdefault("format", "PNG")
            real_save(image, buffer, *args, **kwargs)
            partial_png = buffer.getvalue()
            with open(file_path, "wb") as partial_file:
                partial_file.write(partial_png[: len(partial_png) // 2])
            raise OSError("simulated crash mid-write")

        viewer.clear_model()
        viewer.begin_frame(0.0)
        with mock.patch("PIL.Image.Image.save", partial_save), self.assertWarns(UserWarning):
            viewer.log_mesh(mesh_name, points, indices, uvs=uvs, texture=texture)

        with open(tex_path, "rb") as published_file:
            self.assertEqual(published_file.read(), published)
        leaked = sorted(name for name in os.listdir(viewer._test_work_dir) if name.endswith(".tmp"))
        self.assertEqual(leaked, [])

    def test_save_texture_atomic_cleans_up_tmp_on_failure(self):
        """A failure during the temp-file write must not leave a `.tmp` sibling behind."""
        viewer = self._make_viewer()
        tex_path = os.path.join(viewer._test_work_dir, "tex.png")
        tex_array = np.zeros((2, 2, 3), dtype=np.uint8)

        # Force os.replace to fail after the PNG has been written to the
        # temp file, so the finally cleanup branch in _save_texture_atomic
        # must run to remove the .tmp sibling.
        with mock.patch("newton._src.viewer.viewer_usd.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                ViewerUSD._save_texture_atomic(tex_array, tex_path)

        leaked = [name for name in os.listdir(viewer._test_work_dir) if name.endswith(".tmp")]
        self.assertEqual(leaked, [])

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

    def test_log_points_hides_existing_prim_with_empty_points_and_hidden(self):
        viewer = self._make_viewer()

        viewer.begin_frame(0.0)
        points = wp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=wp.vec3)
        path = viewer.log_points("/particles", points, radii=0.01)

        points_prim = UsdGeom.Points.Get(viewer.stage, path)
        self.assertEqual(
            points_prim.GetVisibilityAttr().Get(viewer._frame_index),
            UsdGeom.Tokens.inherited,
        )

        viewer.begin_frame(1.0 / 60.0)
        empty = wp.empty(0, dtype=wp.vec3)
        viewer.log_points("/particles", empty, hidden=True)

        self.assertEqual(
            points_prim.GetVisibilityAttr().Get(viewer._frame_index),
            UsdGeom.Tokens.invisible,
        )

    def test_log_points_renders_as_point_instancer_by_default(self):
        temp_file = tempfile.NamedTemporaryFile(suffix=".usda", delete=False)
        temp_file.close()
        self.addCleanup(lambda: os.path.exists(temp_file.name) and os.remove(temp_file.name))
        viewer = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer.close)
        self.addCleanup(lambda: setattr(viewer, "output_path", ""))

        viewer.begin_frame(0.0)
        points = wp.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=wp.vec3,
        )
        path = viewer.log_points("/spheres", points, radii=0.05)

        instancer = UsdGeom.PointInstancer.Get(viewer.stage, path)
        self.assertTrue(instancer)
        self.assertFalse(UsdGeom.Points.Get(viewer.stage, path))
        sphere = UsdGeom.Sphere.Get(viewer.stage, path.AppendChild("sphere"))
        self.assertTrue(sphere)
        self.assertEqual(sphere.GetRadiusAttr().Get(), 1.0)

        scales = np.asarray(instancer.GetScalesAttr().Get(viewer._frame_index), dtype=np.float32)
        np.testing.assert_allclose(scales, np.full((3, 3), 0.05, dtype=np.float32), atol=1e-6)

        hidden_path = viewer.log_points("/spheres", None)
        self.assertEqual(hidden_path, path)
        self.assertEqual(instancer.GetVisibilityAttr().Get(viewer._frame_index), UsdGeom.Tokens.invisible)

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
