# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib.util
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

from newton._src.viewer.viewer_rtx import ViewerRTX

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
        viewer._all_instance_paths = []
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
        self.viewer._line_batch_handles = {}
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

    def test_runtime_line_batch_population_uses_ovstage(self):
        """Add and remove runtime line USD through OVStage without legacy calls."""
        self.viewer._rtx = mock.Mock()
        self.viewer._rtx.add_usd_reference_from_string.side_effect = DeprecationWarning
        self.viewer._rtx.remove_usd.side_effect = DeprecationWarning
        self.viewer._get_path = mock.Mock(return_value="/World/RuntimeLines")
        starts = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        ends = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

        self.assertTrue(self.viewer._rebuild_runtime_line_batch_layer("runtime", starts, ends, colors, 0.1, False))
        self.assertTrue(self.viewer._ovstage_population_dirty)
        self.assertIsInstance(self.viewer._line_batch_handles["runtime"], int)
        self.viewer._ovstage_ordinal = 2
        self.viewer._apply_ovstage_population_changes()
        self.assertFalse(self.viewer._ovstage_population_dirty)

        self.viewer._remove_runtime_line_batch_layer("runtime")
        self.assertTrue(self.viewer._ovstage_population_dirty)
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
            mock.patch.object(self.viewer, "_update_ovrtx_line_batches"),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
