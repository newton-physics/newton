# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from newton._src.viewer import viewer_gl
from newton._src.viewer.plot_logger import PlotLogger
from newton._src.viewer.viewer_gui import ViewerGui
from newton._src.viewer.viewer_rtx import ViewerRTX
from newton._src.viewer.viewer_usd import UsdGeom


class _ViewerPlottingTests:
    def setUp(self):
        with wp.ScopedDevice("cpu"):
            self.viewer = self._make_viewer()
        self.gl = mock.MagicMock()
        self.gl.GLuint = ctypes.c_uint
        self.gl.glGenTextures.side_effect = lambda count, out: out.__setitem__(0, self.gl.glGenTextures.call_count)
        patch = mock.patch.object(PlotLogger, "_get_gl", return_value=self.gl)
        self.get_gl = patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.viewer.close)
        self.logger = self.viewer._plot_logger
        self.imgui = mock.MagicMock()
        self.gui = ViewerGui.__new__(ViewerGui)
        self.gui._viewer = self.viewer
        self.gui.ui = SimpleNamespace(imgui=self.imgui, io=SimpleNamespace(display_size=(1280, 720)), dpi_scale=1.0)

    def test_scalar_reaches_plot_window(self):
        """Display scalar values logged before the first rendered frame."""
        self.viewer.log_scalar("reward", np.float32(2.5))
        self.gui._render_scalar_plots()
        self.imgui.plot_lines.assert_called_once()
        args, kwargs = self.imgui.plot_lines.call_args
        self.assertEqual(args[0], "##reward")
        self.assertEqual(args[1][-1], 2.5)
        self.assertEqual(kwargs["overlay_text"], "2.5")

    def test_array_creates_plot_window(self):
        """Make logged arrays available to the plot window."""
        self.viewer.log_array("observations", np.arange(6).reshape(2, 3))
        self.imgui.begin.return_value = False
        self.gui._render_scalar_plots()
        self.imgui.begin.assert_called_once_with("Plots")
        self.get_gl.assert_not_called()

    def test_history_rolls_and_refreshes(self):
        """Bound scalar history and refresh cached plots after new samples."""
        for value in range(5):
            self.viewer.log_scalar("reward", value)
        self.gui._render_scalar_plots()
        np.testing.assert_array_equal(self.imgui.plot_lines.call_args.args[1], [2, 3, 4])
        self.viewer.log_scalar("reward", True)
        self.gui._render_scalar_plots()
        np.testing.assert_array_equal(self.imgui.plot_lines.call_args.args[1], [3, 4, 1])

    def test_smoothing_and_clear(self):
        """Average complete sample groups and discard pending samples on clear."""
        self.viewer.log_scalar("reward", 2, smoothing=2)
        self.gui._render_scalar_plots()
        self.assertTrue(np.isnan(self.imgui.plot_lines.call_args.args[1]).all())
        self.viewer.log_scalar("reward", 4, smoothing=2)
        self.viewer.log_scalar("reward", 100, smoothing=2)
        self.assertEqual(list(self.logger._scalar_buffers["reward"]), [3])
        self.viewer.log_scalar("reward", 6, clear=True, smoothing=2)
        self.viewer.log_scalar("reward", 10, smoothing=2)
        self.gui._render_scalar_plots()
        values = self.imgui.plot_lines.call_args.args[1]
        self.assertTrue(np.isnan(values[:-1]).all())
        self.assertEqual(values[-1], 8)
        with self.assertRaisesRegex(ValueError, "smoothing must be >= 1"):
            self.viewer.log_scalar("reward", 1, smoothing=0)

    def test_array_shapes_and_warp_input(self):
        """Normalize scalar, vector, strided, and Warp arrays for heatmaps."""
        inputs = [
            (np.array(3), [[3]]),
            (np.array([1, 2]), [[1, 2]]),
            (np.arange(12).reshape(3, 4)[:, ::2], [[0, 2], [4, 6], [8, 10]]),
            (wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"), [[1, 2]]),
        ]
        for value, expected in inputs:
            with self.subTest(value=type(value)):
                self.viewer.log_array("observations", value)
                actual = self.logger._array_buffers["observations"]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(actual.flags.c_contiguous)
        with self.assertRaisesRegex(ValueError, "scalar, 1-D, or 2-D"):
            self.viewer.log_array("observations", np.zeros((2, 2, 2)))
        self.get_gl.assert_not_called()

    def test_heatmap_updates_resizes_and_removes(self):
        """Upload changed heatmaps and release textures on resize and removal."""
        self.viewer.log_array("observations", np.arange(6).reshape(2, 3))
        self.gui._render_scalar_plots()
        self.imgui.image.assert_called_once()
        self.gl.glTexSubImage2D.assert_called_once()
        texture_id = self.logger._array_textures["observations"]["texture_id"]
        self.gui._render_scalar_plots()
        self.gl.glTexSubImage2D.assert_called_once()
        self.viewer.log_array("observations", np.arange(6).reshape(2, 3) * 2)
        self.gui._render_scalar_plots()
        self.assertEqual(self.gl.glTexSubImage2D.call_count, 2)
        self.assertEqual(self.logger._array_textures["observations"]["texture_id"], texture_id)
        self.viewer.log_array("observations", np.arange(8).reshape(2, 4))
        self.gui._render_scalar_plots()
        self.assertEqual(self.gl.glGenTextures.call_count, 2)
        self.assertEqual(self.gl.glDeleteTextures.call_args.args[1][0], texture_id)
        self.viewer.log_array("observations", None)
        self.assertEqual(self.gl.glDeleteTextures.call_count, 2)
        self.assertNotIn("observations", self.logger._array_buffers)
        self.assertNotIn("observations", self.logger._array_dirty)
        self.assertNotIn("observations", self.logger._array_textures)
        self.imgui.reset_mock()
        self.gui._render_scalar_plots()
        self.imgui.begin.assert_not_called()

    def test_empty_heatmap(self):
        """Display empty array metadata without allocating invalid GL textures."""
        self.viewer.log_array("empty", np.zeros((0, 3)))
        self.gui._render_scalar_plots()
        self.imgui.text.assert_called_once_with("shape 0x3 (empty)")
        self.get_gl.assert_not_called()

    def test_layer_names_and_clear_all(self):
        """Keep same-name layer signals distinct and clear all layer resources."""
        self.viewer.log_scalar("reward", 1)
        self.viewer.log_array("observations", np.ones((2, 2)))
        self.viewer.activate("training")
        self.viewer.log_scalar("reward", 2)
        self.viewer.log_array("observations", np.zeros((2, 2)))
        self.gui._render_scalar_plots()
        self.assertEqual(set(self.logger._scalar_buffers), {"reward", "/layers/training/reward"})
        self.viewer.log_scalar("/layers/training/reward", 3)
        self.assertEqual(list(self.logger._scalar_buffers["/layers/training/reward"]), [2, 3])
        self.viewer.log_array("observations", None)
        self.assertIn("observations", self.logger._array_buffers)
        self.viewer.clear_all_layers()
        self.assertFalse(self.logger._scalar_buffers)
        self.assertFalse(self.logger._scalar_arrays)
        self.assertFalse(self.logger._array_buffers)
        self.assertFalse(self.logger._array_textures)
        self.assertEqual(self.gl.glDeleteTextures.call_count, 2)

    def test_clear_model_releases_plots(self):
        """Clear plot history, pending smoothing, and textures on model reset."""
        self.viewer.log_scalar("reward", 99, smoothing=2)
        self.viewer.log_array("observations", np.ones((2, 2)))
        self.gui._render_scalar_plots()
        self.viewer.clear_model()
        self.gl.glDeleteTextures.assert_called_once()
        self.viewer.log_scalar("reward", 2, smoothing=2)
        self.viewer.log_scalar("reward", 4, smoothing=2)
        self.gui._render_scalar_plots()
        values = self.imgui.plot_lines.call_args.args[1]
        self.assertTrue(np.isnan(values[:-1]).all())
        self.assertEqual(values[-1], 3)
        self.assertFalse(self.logger._array_textures)

    def test_close_releases_textures(self):
        """Release heatmap textures exactly once when closing the viewer."""
        self.viewer.log_array("observations", np.ones((2, 2)))
        self.gui._render_scalar_plots()
        self.viewer.close()
        self.gl.glDeleteTextures.assert_called_once()
        self.logger.clear()
        self.gl.glDeleteTextures.assert_called_once()


@unittest.skipIf(UsdGeom is None, "usd-core is required")
class TestViewerRTXPlotting(_ViewerPlottingTests, unittest.TestCase):
    def _make_viewer(self):
        # OVRTX is not instantiated until the first rendered frame.
        with mock.patch.dict("sys.modules", {"ovrtx": SimpleNamespace()}):
            return ViewerRTX(headless=True, plot_history_size=3)


class TestViewerGLPlotting(_ViewerPlottingTests, unittest.TestCase):
    def _make_viewer(self):
        renderer = mock.MagicMock()
        renderer.window.get_framebuffer_size.return_value = (1280, 720)
        renderer.window.get_size.return_value = (1280, 720)
        with (
            mock.patch.object(viewer_gl, "RendererGL", return_value=renderer),
            mock.patch.object(viewer_gl, "ImageLogger"),
        ):
            return viewer_gl.ViewerGL(headless=True, plot_history_size=3)

    def test_clear_model_preserves_other_layers(self):
        """Release only the active GL layer's plots during a model reset."""
        self.viewer.log_scalar("reward", 1)
        self.viewer.log_array("observations", np.ones((2, 2)))
        self.viewer.activate("training")
        self.viewer.log_scalar("reward", 2)
        self.viewer.log_array("observations", np.zeros((2, 2)))
        self.gui._render_scalar_plots()
        self.viewer.clear_model()
        self.assertEqual(set(self.logger._scalar_buffers), {"reward"})
        self.assertEqual(set(self.logger._array_textures), {"observations"})
        self.gl.glDeleteTextures.assert_called_once()


class TestPlotLogger(unittest.TestCase):
    def test_invalid_history_size(self):
        """Reject invalid history sizes before either backend initializes."""
        for viewer_class in (viewer_gl.ViewerGL, ViewerRTX):
            for value, error in ((0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)):
                with self.subTest(viewer=viewer_class.__name__, value=value), self.assertRaises(error):
                    viewer_class(plot_history_size=value)

    def test_texture_context_is_owned_by_viewer(self):
        """Activate the owning window before deleting its heatmap texture."""
        window = mock.Mock()
        gl = mock.MagicMock(GLuint=ctypes.c_uint)
        logger = PlotLogger(3, get_window=lambda: window)
        logger._array_textures["observations"] = {"texture_id": 42}
        with mock.patch.dict("sys.modules", {"pyglet": SimpleNamespace(gl=gl)}):
            logger.clear()
        window.switch_to.assert_called_once()
        self.assertEqual(gl.glDeleteTextures.call_args.args[1][0], 42)

    def test_clear_after_window_close(self):
        """Release heatmaps when the window or its context is already gone."""
        closed_window = mock.Mock()
        closed_window.switch_to.side_effect = AttributeError("context is already destroyed")
        for window in (closed_window, None):
            with self.subTest(window=window):
                gl = mock.MagicMock(GLuint=ctypes.c_uint)
                logger = PlotLogger(3, get_window=lambda window=window: window)
                logger.log_scalar("reward", 1, smoothing=2)
                for index, name in enumerate(("observations", "actions"), start=42):
                    logger.log_array(name, np.ones((2, 2)))
                    logger._array_textures[name] = {"texture_id": index}
                with mock.patch.dict("sys.modules", {"pyglet": SimpleNamespace(gl=gl)}):
                    logger.clear()
                    logger.clear()
                self.assertFalse(logger._scalar_buffers)
                self.assertFalse(logger._scalar_accumulators)
                self.assertFalse(logger._array_buffers)
                self.assertFalse(logger._array_dirty)
                self.assertFalse(logger._array_textures)
                self.assertEqual([call.args[1][0] for call in gl.glDeleteTextures.call_args_list], [42, 43])

    def test_heatmap_nonfinite_values(self):
        """Color nonfinite cells separately and compute finite heatmap bounds."""
        logger = PlotLogger(3, get_window=lambda: None)
        rgba, low, high = logger._colorize_heatmap(np.array([[0, 2, np.nan, np.inf]], dtype=np.float32))
        self.assertEqual((low, high), (0, 2))
        np.testing.assert_array_equal(rgba[0, 0], logger._heatmap_color_lut[0])
        np.testing.assert_array_equal(rgba[0, 1], logger._heatmap_color_lut[-1])
        np.testing.assert_array_equal(rgba[0, 2:], np.tile(logger._heatmap_nan_rgba, (2, 1)))
        rgba, low, high = logger._colorize_heatmap(np.array([[np.nan]], dtype=np.float32))
        self.assertTrue(np.isnan(low) and np.isnan(high))
        np.testing.assert_array_equal(rgba[0, 0], logger._heatmap_nan_rgba)
        reduced = logger._downsample_heatmap(np.array([[1, np.nan], [3, np.inf]]), 1, 1)
        np.testing.assert_array_equal(reduced, [[2]])


if __name__ == "__main__":
    unittest.main()
