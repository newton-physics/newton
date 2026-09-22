# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test ViewerRTX compatibility and runtime scene updates."""

import builtins
import importlib.util
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

from newton.viewer import ViewerRTX

OVRTX_AVAILABLE = importlib.util.find_spec("ovrtx") is not None
OVSTAGE_AVAILABLE = importlib.util.find_spec("ovstage") is not None


@unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
class TestViewerRTXVersionCompatibility(unittest.TestCase):
    def test_legacy_ovrtx_does_not_require_ovstage(self):
        """Construct ViewerRTX with legacy OVRTX without importing OVStage."""
        import ovrtx

        original_import = builtins.__import__

        def import_without_ovstage(name, *args, **kwargs):
            if name == "ovstage":
                raise AssertionError("legacy OVRTX must not import ovstage")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.object(ovrtx, "__version__", "0.3.0"),
            mock.patch("builtins.__import__", side_effect=import_without_ovstage),
        ):
            viewer = ViewerRTX(headless=True)

        viewer.close()

    def test_modern_ovrtx_requires_ovstage(self):
        """Require OVStage when constructing ViewerRTX with OVRTX 0.4 or newer."""
        import ovrtx

        original_import = builtins.__import__

        def import_without_ovstage(name, *args, **kwargs):
            if name == "ovstage":
                raise ImportError("ovstage unavailable")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.object(ovrtx, "__version__", "0.4.0"),
            mock.patch("builtins.__import__", side_effect=import_without_ovstage),
            self.assertRaisesRegex(ImportError, "OVRTX 0.4 or newer"),
        ):
            ViewerRTX(headless=True)


@unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
class TestViewerRTXWindowCleanup(unittest.TestCase):
    def test_init_failure_closes_partial_window(self):
        """Close and clear a partially initialized presentation window."""
        import ovrtx

        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer.stage = mock.Mock()
        viewer._use_ovstage = False
        viewer._headless = False
        viewer._window = None
        viewer.gui = None
        viewer._instance_prim_paths = {}
        viewer._mesh_prim_paths = {}
        viewer._point_batch_paths = {}
        viewer._all_instance_paths = []
        viewer._bound_instance_prim_paths = {}
        viewer._runtime_transform_bindings = {}
        viewer._transform_binding = None
        viewer._rtx = None
        viewer._tex_resource = None
        viewer._gl_texture = None
        viewer._gl_program = None
        viewer._gl_vao = None
        partial_window = mock.Mock()

        def fail_after_window_creation():
            viewer._window = partial_window
            viewer._tex_resource = mock.Mock()
            viewer._gl_texture = 1
            viewer._gl_program = 2
            viewer._gl_vao = 3
            raise RuntimeError("window initialization failed")

        with (
            mock.patch.object(viewer, "_add_camera_lights_and_render_product"),
            mock.patch.object(viewer, "_apply_ground_material"),
            mock.patch.object(viewer, "_init_window", side_effect=fail_after_window_creation),
            mock.patch.object(viewer, "_release_runtime_scene"),
            mock.patch.object(viewer, "_destroy_ovrtx"),
            mock.patch.object(ovrtx, "RendererConfig"),
            mock.patch.object(ovrtx, "Renderer"),
            self.assertRaisesRegex(RuntimeError, "Failed to create window"),
        ):
            viewer._init_ovrtx()

        partial_window.close.assert_called_once_with()
        self.assertIsNone(viewer._window)
        self.assertIsNone(viewer._tex_resource)
        self.assertIsNone(viewer._gl_texture)
        self.assertIsNone(viewer._gl_program)
        self.assertIsNone(viewer._gl_vao)


@unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
class TestViewerRTXOvstage(unittest.TestCase):
    def setUp(self):
        """Create an ovstage-backed ViewerRTX runtime stub."""
        import ovstage

        self.ovstage = ovstage
        self.viewer = ViewerRTX.__new__(ViewerRTX)
        self.viewer._rtx = None
        self.viewer._use_ovstage = True
        self.viewer._ovstage = self.ovstage.Stage("newton.test.ViewerRTX")
        self.viewer._ovstage_attached = False
        self.viewer._ovstage_paths = self.ovstage.PathDictionary(self.viewer._ovstage)
        self.viewer._ovstage_queries = {}
        self.viewer._ovstage_ordinal = 1
        self.viewer._ovstage_population_dirty = False
        self.viewer._runtime_scene_changed = False
        self.ovstage.population.open_usd_from_string(
            self.viewer._ovstage,
            """#usda 1.0
def Xform "World"
{
    def Xform "A"
    {
    }
    def Xform "B"
    {
    }
}
""",
            ordinal=self.viewer._ovstage_ordinal,
            time_code=0.0,
        )
        self.viewer._ovstage.advance_write_floor(self.viewer._ovstage_ordinal, self.ovstage.Scope.ALL).wait()

    def tearDown(self):
        """Release the ovstage runtime stub."""
        self.viewer._release_ovstage()

    def test_write_visibility_uses_reusable_query_without_deprecation(self):
        """Write token attributes through one reusable ordered query."""
        prim_paths = ["/World/A", "/World/B"]
        self.viewer._ovstage_ordinal = 2

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.viewer._write_runtime_attribute(
                prim_paths,
                "visibility",
                ["inherited", "invisible"],
            )

        self.assertFalse([warning for warning in caught if warning.category is DeprecationWarning])
        query = self.viewer._get_ovstage_query(prim_paths)
        self.assertIs(query, self.viewer._get_ovstage_query(prim_paths))
        self.assertEqual(len(self.viewer._ovstage_queries), 1)

        self.viewer._ovstage.advance_write_floor(self.viewer._ovstage_ordinal, self.ovstage.Scope.ALL).wait()
        attribute = self.viewer._ovstage_paths.intern_token("visibility")
        with self.viewer._ovstage.read_attributes(
            query,
            [attribute],
            self.ovstage.OrdinalRange.latest(self.viewer._ovstage_ordinal),
        ) as read:
            group = read.fetch_next()
            data_rows = np.from_dlpack(group.dlpack(0)).copy()
            values = [data_rows[group.data_row_index(index)] for index in range(group.data_count)]
            self.viewer._ovstage.release_group(group)

        self.assertEqual(
            [self.viewer._ovstage_paths.token_to_string(int(value)) for value in values],
            ["inherited", "invisible"],
        )

    def test_write_array_attribute_preserves_vector_lanes(self):
        """Write a runtime vector array with its element layout intact."""
        points = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        self.viewer._ovstage_ordinal = 2
        self.viewer._write_runtime_array_attribute("/World/A", "points", points)
        self.viewer._ovstage.advance_write_floor(self.viewer._ovstage_ordinal, self.ovstage.Scope.ALL).wait()

        query = self.viewer._get_ovstage_query(["/World/A"])
        attribute = self.viewer._ovstage_paths.intern_token("points")
        with self.viewer._ovstage.read_attributes(
            query,
            [attribute],
            self.ovstage.OrdinalRange.latest(self.viewer._ovstage_ordinal),
        ) as read:
            group = read.fetch_next()
            self.assertEqual(group.tensor(0).dtype.lanes, 3)
            values = np.from_dlpack(group.dlpack(0)).copy()
            self.viewer._ovstage.release_group(group)

        np.testing.assert_allclose(values, points)

    def test_runtime_prim_population_uses_ovstage(self):
        """Add and replace runtime USD through OVStage without legacy calls."""
        from pxr import Usd, UsdGeom

        self.viewer._rtx = mock.Mock()
        self.viewer._rtx.add_usd_reference_from_string.side_effect = DeprecationWarning
        self.viewer._rtx.remove_usd.side_effect = DeprecationWarning
        self.viewer.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.viewer.stage, "/World/RuntimeMarker")
        self.viewer._frame_index = 0
        self.viewer._runtime_prim_handles = {}
        self.viewer._runtime_prim_paths = {}
        self.viewer._runtime_prim_serial = 0

        runtime_path = self.viewer._replace_runtime_prim("/World/RuntimeMarker")
        self.assertTrue(self.viewer._ovstage_population_dirty)
        self.assertEqual(runtime_path, "/World/RuntimeMarker_rtx_1")
        self.assertIsInstance(self.viewer._runtime_prim_handles["/World/RuntimeMarker"], int)
        self.viewer._ovstage_ordinal = 2
        self.viewer._apply_ovstage_population_changes()
        self.assertFalse(self.viewer._ovstage_population_dirty)

        replacement_path = self.viewer._replace_runtime_prim("/World/RuntimeMarker")
        self.assertTrue(self.viewer._ovstage_population_dirty)
        self.assertEqual(replacement_path, "/World/RuntimeMarker_rtx_2")
        self.viewer._ovstage_ordinal = 3
        self.viewer._apply_ovstage_population_changes()
        self.assertFalse(self.viewer._ovstage_population_dirty)
        self.viewer._rtx.add_usd_reference_from_string.assert_not_called()
        self.viewer._rtx.remove_usd.assert_not_called()

    def test_end_frame_waits_for_async_render_before_stage_writes(self):
        """Finish the previous async stage read before publishing the next frame."""
        events = []
        self.viewer._phase = self.viewer._PHASE_RENDER
        self.viewer._async = True
        self.viewer._render_result = mock.Mock()
        self.viewer._render_result.wait.side_effect = lambda: events.append("wait")

        with (
            mock.patch.object(
                self.viewer,
                "_apply_ovstage_population_changes",
                side_effect=lambda: events.append("write"),
            ),
            mock.patch.object(self.viewer, "_update_ovrtx_camera"),
            mock.patch.object(self.viewer, "_update_ovrtx_transforms"),
            mock.patch.object(self.viewer, "_update_ovrtx_instance_visibility"),
            mock.patch.object(self.viewer, "_update_ovrtx_point_batches"),
            mock.patch.object(self.viewer, "_update_ovrtx_mesh_points"),
            mock.patch.object(self.viewer, "_render_and_display"),
        ):
            self.viewer.end_frame()

        self.assertEqual(events[:2], ["wait", "write"])


class TestViewerRTXRenderOutput(unittest.TestCase):
    def test_ldr_color_lookup_accepts_legacy_and_ovrtx_05_names(self):
        """Find the color output returned by legacy and OVRTX 0.5 renderers."""
        for name in ("LdrColor", "/Render/Vars/LdrColor"):
            with self.subTest(name=name):
                render_var = object()
                frame = mock.Mock(render_vars={name: render_var})
                self.assertIs(ViewerRTX._get_ldr_color_render_var(frame), render_var)

    @unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
    def test_display_uses_ovrtx_05_color_output(self):
        """Blit the fully qualified OVRTX 0.5 color output to the window."""
        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer._rtx = mock.Mock()
        viewer._should_close = False
        viewer._async = False
        viewer._use_ovstage = True
        viewer._ovstage_ordinal = 1
        viewer._render_product_path = "/Render/Product"
        viewer.fps = 60
        viewer._window = mock.Mock(context=object())

        render_var = mock.MagicMock()
        mapping = render_var.map.return_value.__enter__.return_value
        pixels = mock.Mock()
        pixels.device.stream.cuda_stream = 17
        frame = mock.Mock(render_vars={"/Render/Vars/LdrColor": render_var})
        viewer._rtx.step.return_value = {"product": mock.Mock(frames=[frame])}

        with (
            mock.patch.object(wp, "from_dlpack", return_value=pixels),
            mock.patch.object(viewer, "_blit_to_window") as blit,
        ):
            viewer._render_and_display()

        blit.assert_called_once_with(pixels)
        mapping.unmap.assert_called_once_with(stream=17)

    @unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
    def test_screenshot_uses_ovrtx_05_color_output(self):
        """Capture the fully qualified OVRTX 0.5 color output."""
        viewer = ViewerRTX.__new__(ViewerRTX)
        expected = np.zeros((2, 3, 4), dtype=np.uint8)
        render_var = mock.MagicMock()
        render_var.map.return_value.__enter__.return_value = expected
        frame = mock.Mock(render_vars={"/Render/Vars/LdrColor": render_var})
        viewer._render_products = {"product": mock.Mock(frames=[frame])}
        viewer._render_result = None

        np.testing.assert_array_equal(viewer._capture_screenshot_pixels(), expected)


@unittest.skipUnless(OVRTX_AVAILABLE and OVSTAGE_AVAILABLE and wp.is_cuda_available(), "Requires OVRTX and CUDA")
class TestViewerRTXRendering(unittest.TestCase):
    def test_runtime_line_batch_has_no_deprecation_warnings(self):
        """Render a line batch first created after the runtime scene is active."""
        viewer = ViewerRTX(headless=True, async_rendering=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                viewer.begin_frame(0.0)
                viewer.end_frame()
                viewer.begin_frame(1.0 / 60.0)
                viewer.log_lines(
                    "/runtime_line",
                    wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3),
                    wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
                    wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
                )
                viewer.end_frame()
        finally:
            viewer.close()


class TestViewerRTXMeshUpdates(unittest.TestCase):
    def _make_runtime_viewer(self):
        """Capture mesh attribute writes at the OVRTX boundary."""
        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer._phase = viewer._PHASE_RENDER
        viewer._qualify = mock.Mock(side_effect=lambda name: name)
        viewer._mesh_prim_paths = {"/mesh": "/root/mesh"}
        viewer._pending_mesh_points = {}
        viewer._pending_mesh_normals = {}
        viewer._pending_mesh_topology = {}
        viewer._pending_mesh_visibility = {}
        viewer._rtx = mock.Mock()
        viewer._use_ovstage = False
        # Avoid the optional OVRTX DLPack adapter, but retain the arrays sent to it.
        viewer._make_point3f_dltensor = mock.Mock(side_effect=np.copy)
        attributes = {}

        def write_array_attribute(prim_paths, attribute_name, tensors):
            self.assertEqual(prim_paths, ["/root/mesh"])
            self.assertEqual(len(tensors), 1)
            attributes[attribute_name] = np.array(tensors[0], copy=True)

        viewer._rtx.write_array_attribute.side_effect = write_array_attribute
        return viewer, attributes

    def test_dynamic_mesh_generates_runtime_normals_after_topology_change(self):
        """Send fresh smooth or sharp normals to RTX after replacing mesh topology."""
        for index_dtype in (wp.int32, wp.uint32):
            with self.subTest(index_dtype=index_dtype):
                viewer, attributes = self._make_runtime_viewer()
                points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device="cpu")
                indices = wp.array([0, 1, 2], dtype=index_dtype, device="cpu")
                normals = wp.array([[1, 0, 0]] * 3, dtype=wp.vec3, device="cpu")
                viewer.log_mesh("/mesh", points, indices, normals=normals, dynamic=True)
                viewer._update_ovrtx_mesh_points()
                np.testing.assert_array_equal(attributes["normals"], normals.numpy())

                folded_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
                folded_indices = np.array([0, 1, 2, 0, 3, 1], dtype=np.int32)
                for split in (False, True):
                    with self.subTest(split=split):
                        if split:
                            vertices = folded_points[folded_indices]
                            triangles = np.arange(6, dtype=np.int32)
                            expected_normals = [[0, 0, 1]] * 3 + [[0, 1, 0]] * 3
                        else:
                            vertices, triangles = folded_points, folded_indices
                            expected_normals = [[0, 2**-0.5, 2**-0.5]] * 2 + [[0, 0, 1], [0, 1, 0]]
                        viewer.log_mesh(
                            "/mesh",
                            wp.array(vertices, dtype=wp.vec3, device="cpu"),
                            wp.array(triangles, dtype=index_dtype, device="cpu"),
                            dynamic=True,
                        )
                        viewer._update_ovrtx_mesh_points()
                        np.testing.assert_allclose(attributes["normals"], expected_normals, atol=1e-6)
                        np.testing.assert_array_equal(attributes["points"], vertices)
                        np.testing.assert_array_equal(attributes["faceVertexIndices"], triangles)
                        np.testing.assert_array_equal(attributes["faceVertexCounts"], [3, 3])

                viewer.log_mesh(
                    "/mesh",
                    wp.empty(0, dtype=wp.vec3, device="cpu"),
                    wp.empty(0, dtype=index_dtype, device="cpu"),
                    dynamic=True,
                )
                viewer._update_ovrtx_mesh_points()
                self.assertEqual(attributes["normals"].shape, (0, 3))
                self.assertEqual(attributes["points"].shape, (0, 3))
                self.assertEqual(attributes["faceVertexIndices"].size, 0)

    def test_deforming_mesh_recomputes_runtime_normals(self):
        """Refresh omitted normals when points change without a topology update."""
        viewer, attributes = self._make_runtime_viewer()
        indices = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
        for vertices, normal in (
            ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 0, 1]),
            ([[0, 0, 0], [0, 0, 1], [1, 0, 0]], [0, 1, 0]),
        ):
            with self.subTest(normal=normal):
                viewer.log_mesh("/mesh", wp.array(vertices, dtype=wp.vec3, device="cpu"), indices)
                viewer._update_ovrtx_mesh_points()
                self.assertIn("normals", attributes)
                np.testing.assert_allclose(attributes["normals"], [normal] * 3, atol=1e-6)
                self.assertNotIn("faceVertexIndices", attributes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
