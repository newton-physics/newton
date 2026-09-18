# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerRTX

if USD_AVAILABLE:
    from pxr import Sdf, Usd, UsdGeom, UsdShade


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestViewerRTXMarkers(unittest.TestCase):
    def setUp(self):
        self.ovrtx = mock.MagicMock()
        patcher = mock.patch.dict("sys.modules", {"ovrtx": self.ovrtx})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.viewer = ViewerRTX(headless=True)
        self.addCleanup(self.viewer.close)
        self.viewer._phase = ViewerRTX._PHASE_RENDER
        self.viewer._rtx = self.ovrtx.Renderer()

    def _log_spheres(self, count, *, hidden=False, color=(1.0, 0.0, 0.0)):
        xforms = wp.array([wp.transform_identity()] * count, dtype=wp.transform, device="cpu")
        colors = wp.array([color] * count, dtype=wp.vec3, device="cpu")
        self.viewer.log_shapes("/markers/spheres", newton.GeoType.SPHERE, 0.1, xforms, colors, hidden=hidden)

    def test_add_markers_after_first_frame(self):
        """Register mesh prototypes and instances after rendering starts."""
        self._log_spheres(2)
        self.assertEqual(len(self.viewer._instance_prim_paths["/markers/spheres"]), 2)
        self.assertTrue(self.viewer._rtx.add_usd_reference_from_string.called)
        self.assertTrue(self.viewer._rtx.bind_attribute.called)

    def test_resize_and_clear_marker_batch(self):
        """Grow, shrink, and hide an existing runtime marker batch."""
        for count in (1, 3, 1, 0, 2):
            self._log_spheres(count)
            self.assertEqual(len(self.viewer._instance_prim_paths["/markers/spheres"]), count)
        self._log_spheres(2, hidden=True)
        self.viewer._update_ovrtx_instance_visibility()
        self.assertEqual(self.viewer._rtx.write_attribute.call_args.kwargs["tensor"], ["invisible"] * 2)
        self._log_spheres(2)
        self.viewer._update_ovrtx_instance_visibility()
        self.assertEqual(self.viewer._rtx.write_attribute.call_args.kwargs["tensor"], ["inherited"] * 2)

    def test_reuse_geometry_until_appearance_changes(self):
        """Reuse runtime geometry for motion and refresh changed colors."""
        self._log_spheres(2)
        self.viewer._rtx.add_usd_reference_from_string.reset_mock()
        self._log_spheres(2)
        self.viewer._rtx.add_usd_reference_from_string.assert_not_called()
        self._log_spheres(2, color=(0.0, 1.0, 0.0))
        self.viewer._rtx.add_usd_reference_from_string.assert_called_once()

    def test_runtime_materials_are_self_contained(self):
        """Keep material bindings valid when a runtime batch is referenced elsewhere."""
        self._log_spheres(2, color=(0.0, 1.0, 0.0))
        content = self.viewer._rtx.add_usd_reference_from_string.call_args.args[0]
        layer = Sdf.Layer.CreateAnonymous()
        self.assertTrue(layer.ImportFromString(content))
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Referenced")
        root.GetReferences().AddReference(layer.identifier)
        mesh = stage.GetPrimAtPath("/Referenced/instance_0")
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        self.assertTrue(material)
        shader = UsdShade.Shader(material.GetPrim().GetChild("PreviewSurface"))
        np.testing.assert_allclose(shader.GetInput("diffuseColor").Get(), [0.0, 1.0, 0.0])
        self.assertEqual(material.GetSurfaceOutput().GetConnectedSource()[0].GetPrim(), shader.GetPrim())

    def test_hidden_build_markers_can_be_shown(self):
        """Preserve initial visibility and allow markers to be enabled after startup."""
        self.viewer._phase = ViewerRTX._PHASE_BUILD
        self._log_spheres(2, hidden=True)
        group = UsdGeom.Imageable(self.viewer.stage.GetPrimAtPath("/root/markers/spheres"))
        self.assertEqual(group.ComputeVisibility(self.viewer._frame_index), "invisible")
        self.viewer._phase = ViewerRTX._PHASE_RENDER
        self._log_spheres(2)
        self.viewer._update_ovrtx_instance_visibility()
        self.viewer._rtx.write_attribute.assert_any_call(
            prim_paths=["/root/markers/spheres"], attribute_name="visibility", tensor=["inherited"]
        )

    def test_points_can_be_added_after_startup(self):
        """Create sphere markers after the first rendered frame."""
        points = wp.array([[0, 0, 1]], dtype=wp.vec3, device="cpu")
        path = self.viewer.log_points("/points", points, radii=0.1, colors=(1.0, 0.8, 0.0))
        self.assertEqual(str(path), self.viewer._point_batch_paths["/points"])
        self.viewer._rtx.add_usd_reference_from_string.assert_called_once()
        self.viewer.log_points("/points", points, radii=0.2)
        self.assertIn("/points", self.viewer._pending_point_batches)

    def test_replace_build_batch_and_reset_render_history(self):
        """Replace an initial batch without colliding with its prims or retaining old samples."""
        self.viewer._phase = ViewerRTX._PHASE_BUILD
        self._log_spheres(3)
        path = "/root/markers/spheres"
        self.viewer._runtime_prim_paths[path] = path
        self.viewer._phase = ViewerRTX._PHASE_RENDER
        self._log_spheres(1)
        runtime_path = self.viewer._rtx.add_usd_reference_from_string.call_args.kwargs["prefix_path"]
        self.assertNotEqual(path, runtime_path)
        self.assertEqual(self.viewer._instance_prim_paths["/markers/spheres"], [f"{runtime_path}/instance_0"])
        self.viewer._rtx.write_attribute.assert_any_call(
            prim_paths=[path], attribute_name="visibility", tensor=["invisible"]
        )
        with (
            mock.patch.object(self.viewer, "_update_ovrtx_camera"),
            mock.patch.object(self.viewer, "_update_ovrtx_transforms"),
            mock.patch.object(self.viewer, "_render_and_display"),
        ):
            self.viewer.end_frame()
            self.viewer.end_frame()
        self.viewer._rtx.reset.assert_called_once_with(time=0.0)

    def test_arrows_have_heads_and_correct_endpoints(self):
        """Render arrow meshes with tips at their endpoints, including reversed arrows."""
        starts = wp.array([[0, 0, 0], [1, 2, 3], [1, 0, 0]], dtype=wp.vec3, device="cpu")
        ends = wp.array([[0, 0, 2], [1, 2, 1], [1, 0, 0]], dtype=wp.vec3, device="cpu")
        self.viewer.log_arrows("/arrows", starts, ends, (1.0, 0.0, 0.0), width=0.02)
        paths = self.viewer._instance_prim_paths["/arrows"]
        xforms, scales = self.viewer._pending_xforms["/arrows"]
        for i in range(len(paths)):
            mesh = UsdGeom.Mesh.Get(self.viewer.stage, f"/root/arrows/instance_{i}")
            self.assertTrue(mesh)
            points = np.asarray(mesh.GetPointsAttr().Get(self.viewer._frame_index))
            # The cone is wider than the shaft and converges to a single tip.
            tip = points[np.argmax(points[:, 2])]
            self.assertAlmostEqual(float(np.linalg.norm(tip[:2])), 0.0, places=6)
            self.assertGreater(float(np.max(np.linalg.norm(points[:, :2], axis=1))), 1.5)
            tip_world = wp.transform_point(wp.transform(*xforms.numpy()[i]), wp.vec3(*(tip * scales.numpy()[i])))
            np.testing.assert_allclose(np.asarray(tip_world), ends.numpy()[i], atol=1e-6)

    def test_lines_added_at_runtime_reuse_geometry(self):
        """Keep runtime lines colored and update motion without recreating the scene."""
        starts = wp.array([[0, 0, 0]], dtype=wp.vec3, device="cpu")
        ends = wp.array([[0, 0, 1]], dtype=wp.vec3, device="cpu")
        for _ in range(2):
            self.viewer.log_lines("/lines", starts, ends, (0.0, 1.0, 0.0))
        self.assertEqual(len(self.viewer._instance_prim_paths["/lines"]), 1)
        self.assertEqual(self.viewer._rtx.add_usd_reference_from_string.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
