# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise synchronous RTX rendering and independent rendering pause."""

import importlib.util
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerRTX


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestRenderingPauseRTX(unittest.TestCase):
    def setUp(self):
        """Create a viewer without initializing a renderer or a window."""
        with wp.ScopedDevice("cpu"), mock.patch.dict("sys.modules", {"ovrtx": mock.Mock(__version__="0.3.0")}):
            self.viewer = ViewerRTX(headless=True, num_frames=4)
        self.addCleanup(self.viewer.close)

    def test_pause_does_not_change_simulation_controls(self):
        """Preserve simulation pause and single-step behavior across render toggles."""
        viewer = self.viewer
        viewer.set_rendering_paused(True)
        self.assertTrue(viewer.should_step())
        viewer._paused = True
        viewer._step_requested = True
        self.assertTrue(viewer.should_step())
        self.assertFalse(viewer.should_step())
        viewer.set_rendering_paused(False)
        self.assertTrue(viewer.is_paused())

    def test_complete_each_frame_before_returning(self):
        """Complete one current render per unpaused loop in both window modes."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._update_scene = mock.Mock()
        viewer._accept_render = mock.Mock()
        viewer._present = mock.Mock()
        for headless in (False, True):
            with self.subTest(headless=headless):
                viewer._headless = headless
                viewer._window = None if headless else mock.Mock()
                for i in range(3):
                    products = {"frame": i}
                    viewer._rtx.step.return_value = products
                    viewer.end_frame()
                    viewer._rtx.step.assert_called_once()
                    viewer._accept_render.assert_called_once_with(products)
                    viewer._rtx.step_async.assert_not_called()
                    viewer._rtx.step.reset_mock()
                    viewer._accept_render.reset_mock()
                    viewer.set_rendering_paused(True)
                    viewer.end_frame()
                    viewer._rtx.step.assert_not_called()
                    viewer._accept_render.assert_not_called()
                    viewer.set_rendering_paused(False)

    def test_gui_pause_keeps_presentation_active(self):
        """Apply the UI toggle before rendering and keep presenting while paused."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._window = mock.Mock()
        viewer._present = mock.Mock()
        viewer.gui = mock.Mock()
        viewer.gui.prepare_frame.side_effect = lambda: viewer.set_rendering_paused(True)
        for _ in range(3):
            viewer.end_frame()
        viewer._rtx.step.assert_not_called()
        viewer._rtx.step_async.assert_not_called()
        self.assertEqual(viewer.gui.prepare_frame.call_count, 3)
        self.assertEqual(viewer._present.call_count, 3)

    def test_legacy_async_argument_warns_and_renders_synchronously(self):
        """Keep old constructor calls working without retaining an async pipeline."""
        with (
            wp.ScopedDevice("cpu"),
            mock.patch.dict("sys.modules", {"ovrtx": mock.Mock(__version__="0.3.0")}),
            self.assertWarnsRegex(DeprecationWarning, "async_rendering"),
        ):
            viewer = ViewerRTX(headless=True, async_rendering=True)
        self.addCleanup(viewer.close)
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._accept_render = mock.Mock()
        viewer.end_frame()
        viewer._rtx.step.assert_called_once()
        viewer._rtx.step_async.assert_not_called()

    def test_point_resize_while_paused_retains_valid_colors(self):
        """Resize a paused point batch without carrying mismatched pending arrays."""
        viewer = self.viewer
        colors = wp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device="cpu")
        radii = wp.array([0.1, 0.2], dtype=float, device="cpu")
        points = wp.zeros(2, dtype=wp.vec3, device="cpu")
        viewer.log_points("/points", points, colors=colors)
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._write_runtime_array_attribute = mock.Mock()
        viewer.set_rendering_paused(True)
        viewer.begin_frame(0.0)
        viewer.log_points("/points", points, radii=radii, colors=colors)
        viewer.end_frame()
        for count in (3, 1):
            viewer.begin_frame(float(count))
            viewer.log_points("/points", wp.zeros(count, dtype=wp.vec3, device="cpu"))
            viewer.end_frame()
            self.assertEqual(viewer._point_batch_synced_counts["/points"], count)
            expected = colors.numpy()[[0, 1, 1] if count == 3 else [0]]
            np.testing.assert_array_equal(viewer._point_batch_colors["/points"], expected)
        viewer._rtx.step.assert_not_called()
        viewer._rtx.step_async.assert_not_called()
        viewer.set_rendering_paused(False)
        viewer._accept_render = mock.Mock()
        viewer.end_frame()
        viewer._rtx.step.assert_called_once()

    def test_point_resize_with_multiple_logs_in_one_frame(self):
        """Let the final point log use cached colors without merging stale radii."""
        viewer = self.viewer
        points = wp.zeros(2, dtype=wp.vec3, device="cpu")
        viewer.log_points("/points", points, colors=(1.0, 0.0, 0.0))
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._accept_render = mock.Mock()
        viewer._write_runtime_array_attribute = mock.Mock()
        viewer.begin_frame(0.0)
        viewer.log_points(
            "/points",
            points,
            radii=wp.array([0.1, 0.2], dtype=float, device="cpu"),
            colors=wp.ones(2, dtype=wp.vec3, device="cpu"),
        )
        viewer.log_points("/points", wp.zeros(3, dtype=wp.vec3, device="cpu"))
        viewer.end_frame()
        self.assertEqual(viewer._point_batch_synced_counts["/points"], 3)

    def test_paused_headless_capture_and_frame_budget(self):
        """Capture the last completed image while paused loops consume the budget."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer.set_rendering_paused(True)
        with self.assertRaisesRegex(RuntimeError, "frame"):
            viewer._capture_screenshot_pixels()
        pixels = np.full((3, 4, 4), 37, dtype=np.uint8)
        render_var = mock.MagicMock()
        render_var.map.return_value.__enter__.return_value = pixels
        viewer._render_products = {"product": mock.Mock(frames=[mock.Mock(render_vars={"LdrColor": render_var})])}
        with mock.patch.dict("sys.modules", {"ovrtx": mock.Mock()}):
            for i in range(4):
                self.assertTrue(viewer.is_running())
                viewer.begin_frame(float(i))
                viewer.end_frame()
                np.testing.assert_array_equal(viewer._capture_screenshot_pixels(), pixels)
        self.assertFalse(viewer.is_running())
        viewer._rtx.step.assert_not_called()
        viewer._rtx.step_async.assert_not_called()

    def test_clear_and_close_release_renderer_once(self):
        """Clear cached output and preserve pause without asynchronous cleanup."""
        viewer = self.viewer
        viewer.set_rendering_paused(True)
        renderer = mock.Mock()
        viewer._rtx = renderer
        viewer._render_products = {"old": object()}
        viewer.clear_model()
        renderer.destroy.assert_called_once_with()
        self.assertTrue(viewer.is_rendering_paused())
        self.assertIsNone(viewer._render_products)
        viewer.close()
        viewer.close()
        renderer.destroy.assert_called_once_with()


@unittest.skipUnless(
    USD_AVAILABLE and importlib.util.find_spec("ovrtx") is not None and wp.is_cuda_available(),
    "Requires the rtx extra and a CUDA device",
)
class TestRenderingPauseRTXIntegration(unittest.TestCase):
    def test_moving_scene_capture_stays_frozen_until_resume(self):
        """Show the first frame immediately and resume directly to the latest state."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.3, hy=0.3, hz=0.3, color=(1.0, 0.1, 0.0))
        model = builder.finalize()
        viewer = ViewerRTX(width=64, height=48, headless=True)
        try:
            viewer.set_model(model)
            viewer.set_camera(wp.vec3(3.0, -4.0, 2.0), pitch=-20.0, yaw=125.0)
            state = model.state()
            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.end_frame()
            frozen = viewer._capture_screenshot_pixels().copy()
            self.assertGreater(np.ptp(frozen[:, :, :3]), 0)
            viewer.set_rendering_paused(True)
            for i in range(1, 4):
                state.body_q.assign([wp.transform(wp.vec3(float(i), 0.0, 0.0), wp.quat_identity())])
                viewer.begin_frame(i / 60.0)
                viewer.log_state(state)
                # Exercise runtime geometry replacement and renderer reset
                # while retaining the previous render products for capture.
                viewer.log_points("/runtime_points", wp.zeros(i, dtype=wp.vec3), radii=0.1)
                viewer.end_frame()
                np.testing.assert_array_equal(viewer._capture_screenshot_pixels(), frozen)
            viewer.set_rendering_paused(False)
            viewer.begin_frame(4.0 / 60.0)
            viewer.log_state(state)
            viewer.end_frame()
            self.assertFalse(np.array_equal(viewer._capture_screenshot_pixels(), frozen))
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
