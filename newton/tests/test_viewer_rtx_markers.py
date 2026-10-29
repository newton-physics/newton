# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE, get_strict_warning_args, patch_sys_module
from newton.viewer import ViewerRTX

if USD_AVAILABLE:
    from pxr import Gf, Usd, UsdGeom, UsdShade


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestViewerRTXMarkers(unittest.TestCase):
    def setUp(self):
        self.ovrtx = mock.MagicMock()
        self.ovstage = mock.MagicMock()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch_sys_module("ovrtx", self.ovrtx))
        stack.enter_context(patch_sys_module("ovstage", self.ovstage))
        self.viewer = ViewerRTX(headless=True)
        self.addCleanup(self.viewer.close)
        self.viewer._phase = ViewerRTX._PHASE_RENDER
        self.viewer._rtx = self.ovrtx.Renderer()
        self.viewer._ovstage = self.ovstage.Stage()
        self.viewer._make_laned_array_dltensor = mock.Mock(side_effect=lambda values, lanes: values)
        self.writes = []
        self.viewer._write_runtime_attribute = mock.Mock(side_effect=self._record_write)

    def _record_write(self, prim_paths, attribute_name, values, *, is_array=False, **kwargs):
        self.writes.append((list(prim_paths), attribute_name, values, is_array))

    def _array_writes(self):
        """Return the latest array value written per attribute."""
        return {name: values for _, name, values, is_array in self.writes if is_array}

    def _attribute_writes(self, attribute_name):
        """Return ``(prim_paths, values)`` of the non-array writes to an attribute."""
        return [
            (paths, values) for paths, name, values, is_array in self.writes if name == attribute_name and not is_array
        ]

    @property
    def _add_reference(self):
        return self.ovstage.population.add_usd_reference_from_string

    def test_mock_cleanup_preserves_cpu_capture_imports(self):
        """Keep lazily imported Warp launch types alive after mocked RTX initialization."""
        # A fresh runtime must initialize inside the mock to expose module-cache rollback.
        script = textwrap.dedent(
            """
            import unittest
            import warp as wp
            from newton.tests.test_viewer_rtx_markers import TestViewerRTXMarkers
            from newton.examples.basic.example_basic_pendulum import Example
            from newton.viewer import ViewerNull

            result = unittest.TestResult()
            TestViewerRTXMarkers("test_add_markers_after_first_frame").run(result)
            assert result.wasSuccessful(), result.errors or result.failures
            with wp.ScopedDevice("cpu"):
                example = Example(ViewerNull(), None)
                wp.capture_launch(example.graph)
            """
        )
        result = subprocess.run(
            [sys.executable, *get_strict_warning_args(), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _log_spheres(self, count, *, hidden=False, color=(1.0, 0.0, 0.0)):
        xforms = wp.array([wp.transform_identity()] * count, dtype=wp.transform, device="cpu")
        colors = wp.array([color] * count, dtype=wp.vec3, device="cpu")
        self.viewer.log_shapes("/markers/spheres", newton.GeoType.SPHERE, 0.1, xforms, colors, hidden=hidden)

    def _stage_from_usda(self, content):
        stage = Usd.Stage.CreateInMemory()
        self.assertTrue(stage.GetRootLayer().ImportFromString(content))
        return stage

    def test_add_markers_after_first_frame(self):
        """Register mesh prototypes and instances after rendering starts."""
        self._log_spheres(2)
        self.assertEqual(len(self.viewer._instance_prim_paths["/markers/spheres"]), 2)
        self.assertTrue(self._add_reference.called)
        self.assertIn("/markers/spheres", self.viewer._pending_xforms)

    def test_resize_and_clear_marker_batch(self):
        """Grow, shrink, and hide an existing runtime marker batch."""
        for count in (1, 3, 1, 0, 2):
            self._log_spheres(count)
            self.assertEqual(len(self.viewer._instance_prim_paths["/markers/spheres"]), count)
        self._log_spheres(2, hidden=True)
        self.viewer._update_ovrtx_instance_visibility()
        self.assertEqual(self._attribute_writes("visibility")[-1][1], ["invisible"] * 2)
        self._log_spheres(2)
        self.viewer._update_ovrtx_instance_visibility()
        self.assertEqual(self._attribute_writes("visibility")[-1][1], ["inherited"] * 2)

    def test_reuse_geometry_until_appearance_changes(self):
        """Reuse runtime geometry for motion and refresh changed colors."""
        self._log_spheres(2)
        self._add_reference.reset_mock()
        self._log_spheres(2)
        self._add_reference.assert_not_called()
        self._log_spheres(2, color=(0.0, 1.0, 0.0))
        self._add_reference.assert_called_once()

    def test_omitted_instance_transforms_preserve_visibility(self):
        """Keep an existing runtime batch visible when transforms are omitted."""
        self._log_spheres(1)
        mesh = self.viewer._instance_specs["/markers/spheres"][0]

        self.viewer.log_instances("/markers/spheres", mesh, None, None, None, None)

        self.assertTrue(self.viewer._pending_instance_visibility["/markers/spheres"])

    def test_runtime_batch_changes_do_not_flatten_the_scene(self):
        """Limit runtime batch replacement to its USD subtree."""
        with mock.patch.object(self.viewer.stage, "Flatten", side_effect=AssertionError("flattened full stage")):
            self._log_spheres(2)
            self._log_spheres(3)
        self.assertTrue(self._add_reference.called)

    def test_runtime_materials_are_self_contained(self):
        """Keep material bindings valid when a runtime batch is referenced elsewhere."""
        self._log_spheres(2, color=(0.0, 1.0, 0.0))
        content = self._add_reference.call_args.args[1]
        source_stage = self._stage_from_usda(content)
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Referenced")
        root.GetReferences().AddReference(source_stage.GetRootLayer().identifier)
        mesh = stage.GetPrimAtPath("/Referenced/instance_0")
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        self.assertTrue(material)
        shader = UsdShade.Shader(material.GetPrim().GetChild("PreviewSurface"))
        np.testing.assert_allclose(shader.GetInput("diffuseColor").Get(), [0.0, 1.0, 0.0])
        self.assertFalse(shader.GetInput("emissiveColor"))
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
        self.assertIn((["/root/markers/spheres"], ["inherited"]), self._attribute_writes("visibility"))

    def test_points_can_be_added_after_startup(self):
        """Create sphere markers after the first rendered frame."""
        points = wp.array([[0, 0, 1]], dtype=wp.vec3, device="cpu")
        path = self.viewer.log_points("/points", points, radii=0.1, colors=(1.0, 0.8, 0.0))
        self.assertEqual(str(path), self.viewer._point_batch_paths["/points"])
        self._add_reference.assert_called_once()
        self.viewer.log_points("/points", points, radii=0.2)
        self.assertIn("/points", self.viewer._pending_point_batches)

    def test_point_resize_reuses_per_point_colors(self):
        """Preserve colors and valid scales through paused growth and shrinkage."""
        viewer = self.viewer
        points = wp.array([[0, 0, 0], [1, 0, 0]], dtype=wp.vec3, device="cpu")
        colors = wp.array([[1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device="cpu")
        radii = wp.array([0.2, 0.3], dtype=float, device="cpu")
        viewer.log_points("/points", points, colors=colors)

        for paused in (False, True):
            with self.subTest(paused=paused):
                viewer.set_rendering_paused(paused)
                viewer.begin_frame(0.0)
                viewer.log_points("/points", points, radii=radii, colors=colors)
                viewer.end_frame()
                for count in (3, 1):
                    self.writes.clear()
                    if paused and count == 1:
                        viewer.begin_frame(0.5)
                        viewer.log_points("/points", points, radii=radii, colors=colors)
                        viewer.end_frame()
                    viewer.begin_frame(float(count))
                    viewer.log_points("/points", wp.zeros(count, dtype=wp.vec3, device="cpu"))
                    viewer.end_frame()
                    if paused:
                        viewer.set_rendering_paused(False)
                        viewer.end_frame()
                        viewer.set_rendering_paused(True)
                    attributes = self._array_writes()
                    expected = colors.numpy()[[0, 1, 1] if count == 3 else [0]]
                    np.testing.assert_allclose(attributes["primvars:displayColor"], expected)
                    np.testing.assert_allclose(attributes["scales"], np.full((count, 3), 0.1))
                    self.assertEqual(viewer._point_batch_synced_counts["/points"], count)

    def test_final_point_log_does_not_merge_stale_appearance(self):
        """Use cached colors and default radii when the final log omits appearance."""
        viewer = self.viewer
        points = wp.zeros(2, dtype=wp.vec3, device="cpu")
        viewer.log_points("/points", points, colors=(1.0, 0.0, 0.0))
        viewer.begin_frame(0.0)
        viewer.log_points("/points", points, radii=0.3, colors=(0.0, 1.0, 0.0))
        viewer.log_points("/points", wp.zeros(3, dtype=wp.vec3, device="cpu"))
        viewer.end_frame()
        attributes = self._array_writes()
        np.testing.assert_allclose(attributes["positions"], np.zeros((3, 3)))
        np.testing.assert_allclose(attributes["scales"], np.full((3, 3), 0.1))
        np.testing.assert_allclose(attributes["primvars:displayColor"], [[1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(attributes["primvars:displayColor:indices"], [0, 0, 0])

    def test_hidden_point_updates_preserve_appearance(self):
        """Apply hidden point appearance updates before revealing the batch."""
        points = wp.array([[0, 0, 0]], dtype=wp.vec3, device="cpu")
        self.viewer.log_points("/points", points, radii=0.1, colors=(1.0, 0.0, 0.0))

        self.viewer.log_points("/points", points, radii=0.2, colors=(0.0, 1.0, 0.0), hidden=True)
        self.viewer._update_ovrtx_point_batches()

        np.testing.assert_allclose(self.viewer._point_batch_colors["/points"], [[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(self._array_writes()["scales"], [[0.2, 0.2, 0.2]])

        self.writes.clear()
        self.viewer.log_points("/points", points)
        self.viewer._update_ovrtx_point_batches()
        self.assertEqual(
            [(paths, name, values) for paths, name, values, is_array in self.writes if not is_array],
            [([self.viewer._point_batch_paths["/points"]], "visibility", ["inherited"])],
        )

    def test_replace_build_batch_and_reset_render_history(self):
        """Replace an initial batch without colliding with its prims or retaining old samples."""
        self.viewer._phase = ViewerRTX._PHASE_BUILD
        self._log_spheres(3)
        path = "/root/markers/spheres"
        self.viewer._runtime_prim_paths[path] = path
        self.viewer._phase = ViewerRTX._PHASE_RENDER
        self._log_spheres(1)
        runtime_path = self._add_reference.call_args.args[2]
        self.assertNotEqual(path, runtime_path)
        self.assertEqual(self.viewer._instance_prim_paths["/markers/spheres"], [f"{runtime_path}/instance_0"])
        with (
            mock.patch.object(self.viewer, "_update_ovrtx_camera"),
            mock.patch.object(self.viewer, "_update_ovrtx_transforms"),
        ):
            self.viewer.end_frame()
            self.viewer.end_frame()
        self.assertIn(([path], ["invisible"]), self._attribute_writes("visibility"))
        self.viewer._rtx.reset.assert_called_once_with(time=0.0)

    def test_gaussian_field_replaces_runtime_prim_when_asset_changes(self):
        """Author a Gaussian field once per asset and queue batched placement writes otherwise."""
        first = newton.Gaussian([[0, 0, 0]], scales=[[0.2, 0.2, 0.2]])
        second = newton.Gaussian([[0, 0, 0]], scales=[[0.4, 0.2, 0.2]])
        path = "/root/field"
        self.viewer.log_gaussian("/field", first)
        self.assertEqual(self._add_reference.call_count, 1)
        runtime_path = self.viewer._gaussian_paths["/field"]
        self.assertNotEqual(runtime_path, path)
        prim = self.viewer.stage.GetPrimAtPath(path)
        samples = prim.GetAttribute("visibility").GetNumTimeSamples()
        for frame in range(1, 3):
            self.viewer.begin_frame(frame / 60.0)
            self.viewer.log_gaussian("/field", first, wp.transform(wp.vec3(frame, 0, 0), wp.quat_identity()), True)
        # Same asset: no new runtime prim and no new USD samples, only a queued placement.
        self.assertEqual(self._add_reference.call_count, 1)
        self.assertEqual(prim.GetAttribute("visibility").GetNumTimeSamples(), samples)
        # A hidden call without a placement keeps the last one; a shown one resets to the origin.
        self.viewer.log_gaussian("/field", first, hidden=True)
        np.testing.assert_allclose(self.viewer._gaussian_last_placement["/field"][:3], [2, 0, 0])
        self.viewer.log_gaussian("/field", second)
        self.assertEqual(self._add_reference.call_count, 2)
        self.assertNotEqual(self.viewer._gaussian_paths["/field"], runtime_path)
        with (
            mock.patch.object(self.viewer, "_update_ovrtx_camera"),
            mock.patch.object(self.viewer, "_update_ovrtx_transforms"),
        ):
            self.viewer.end_frame()
        xform_writes = [(paths, values) for paths, name, values, _ in self.writes if name == "omni:xform"]
        self.assertEqual(xform_writes[-1][0], [self.viewer._gaussian_paths["/field"]])
        self.assertEqual(xform_writes[-1][1].shape, (1, 4, 4))
        # The batched matrix carries the latest placement: the shown call's origin.
        np.testing.assert_allclose(xform_writes[-1][1][0, 3, :3], [0, 0, 0])
        self.assertIn(([self.viewer._gaussian_paths["/field"]], ["inherited"]), self._attribute_writes("visibility"))

    def test_gaussian_field_replacement_is_deferred_while_rendering(self):
        """Remap a Gaussian field to its runtime prim once a deferred replacement is published."""
        gaussian = newton.Gaussian([[0, 0, 0]], scales=[[0.2, 0.2, 0.2]])
        self.viewer.log_gaussian("/field", gaussian)
        runtime_path = self.viewer._gaussian_paths["/field"]
        self.viewer.set_rendering_paused(True)
        self.viewer.log_gaussian("/field", newton.Gaussian([[0, 0, 0]], scales=[[0.4, 0.2, 0.2]]))
        self.assertEqual(self.viewer._gaussian_paths["/field"], "/root/field")
        self.assertIn("/root/field", self.viewer._deferred_prims)
        self.viewer.set_rendering_paused(False)
        self.viewer._flush_deferred_prims()
        self.assertNotIn(self.viewer._gaussian_paths["/field"], ("/root/field", runtime_path))

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
        """Keep runtime lines emissive and update motion without recreating the scene."""
        starts = wp.array([[0, 0, 0]], dtype=wp.vec3, device="cpu")
        ends = wp.array([[0, 0, 1]], dtype=wp.vec3, device="cpu")
        self.viewer.log_lines("/lines", starts, ends, (0.0, 1.0, 0.0))
        first_xforms = self.viewer._pending_xforms["/lines"][0].numpy()
        content = self._add_reference.call_args.args[1]
        stage = self._stage_from_usda(content)
        emissive_colors = [
            UsdShade.Shader(prim).GetInput("emissiveColor").Get()
            for prim in stage.Traverse()
            if prim.IsA(UsdShade.Shader) and UsdShade.Shader(prim).GetInput("emissiveColor")
        ]
        self.assertEqual(emissive_colors, [Gf.Vec3f(0.0, 1.0, 0.0)])

        moved_starts = wp.array([[1, 0, 0]], dtype=wp.vec3, device="cpu")
        moved_ends = wp.array([[1, 0, 2]], dtype=wp.vec3, device="cpu")
        self.viewer.log_lines("/lines", moved_starts, moved_ends, (0.0, 1.0, 0.0))
        moved_xforms = self.viewer._pending_xforms["/lines"][0].numpy()
        self.assertEqual(len(self.viewer._instance_prim_paths["/lines"]), 1)
        self.assertEqual(self._add_reference.call_count, 2)
        self.assertFalse(np.array_equal(first_xforms, moved_xforms))


if __name__ == "__main__":
    unittest.main(verbosity=2)
