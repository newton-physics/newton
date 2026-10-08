# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise RTX rendering modes and independent rendering pause."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerRTX


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestRenderingPauseRTX(unittest.TestCase):
    def setUp(self):
        """Create a viewer without initializing a renderer or a window."""
        ovrtx = mock.patch.dict("sys.modules", {"ovrtx": mock.Mock(__version__="0.3.0")})
        ovrtx.start()
        self.addCleanup(ovrtx.stop)
        with wp.ScopedDevice("cpu"):
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

    def test_synchronous_mode_completes_each_frame(self):
        """Complete one current render per unpaused loop in both window modes."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._async = False
        viewer._rtx = mock.Mock()
        viewer._update_scene = mock.Mock()
        viewer._accept_render = mock.Mock()
        viewer._present = mock.Mock()
        for headless in (False, True):
            with self.subTest(headless=headless):
                viewer._headless = headless
                viewer._window = None if headless else mock.Mock()
                products = {"frame": object()}
                viewer._rtx.step.return_value = products
                viewer.end_frame()
                viewer._accept_render.assert_called_once_with(products)
                viewer.set_rendering_paused(True)
                viewer.end_frame()
                viewer._accept_render.assert_called_once_with(products)
                viewer.set_rendering_paused(False)
                viewer.end_frame()
                self.assertEqual(viewer._rtx.step.call_count, 2)
                self.assertEqual(viewer._accept_render.call_count, 2)
                viewer._rtx.step_async.assert_not_called()
                viewer._rtx.step.reset_mock()
                viewer._accept_render.reset_mock()

    def test_gui_pause_keeps_presentation_active(self):
        """Apply the UI toggle before rendering and keep presenting while paused."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._window = mock.Mock()
        viewer._present = mock.Mock()
        pending = mock.Mock()
        viewer._render_result = pending
        viewer.gui = mock.Mock()
        viewer.gui.prepare_frame.side_effect = lambda: viewer.set_rendering_paused(True)
        viewer.end_frame()
        pending.wait.assert_not_called()
        viewer._rtx.step.assert_not_called()
        viewer._rtx.step_async.assert_not_called()
        viewer.gui.prepare_frame.assert_called_once()
        viewer._present.assert_called_once()

    def test_async_mode_waits_normally_and_discards_paused_result(self):
        """Keep blocking frame cadence, and never display a pre-pause result on resume."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._window = mock.Mock()
        viewer._update_scene = mock.Mock()
        viewer._present = mock.Mock(side_effect=lambda *_: viewer._rtx.step_async.assert_not_called())
        viewer._accept_render = mock.Mock()
        pending = viewer._rtx.step_async.return_value
        viewer.end_frame()
        viewer._present.assert_called_once()
        viewer._present.side_effect = None
        pending.wait.assert_not_called()
        viewer._rtx.step_async.assert_called_once()
        viewer._accept_render.side_effect = lambda _: self.assertEqual(viewer._update_scene.call_count, 2)
        viewer.end_frame()
        pending.wait.assert_called_once_with()
        viewer._accept_render.assert_called_once_with(pending.wait.return_value.fetch.return_value)
        viewer._accept_render.side_effect = None
        viewer._accept_render.reset_mock()
        pending.wait.reset_mock()
        viewer._rtx.step_async.reset_mock()
        viewer.set_rendering_paused(True)
        viewer.end_frame()
        pending.wait.assert_not_called()
        viewer._rtx.step_async.assert_not_called()
        viewer.set_rendering_paused(False)
        viewer.end_frame()
        pending.wait.assert_called_once_with()
        viewer._accept_render.assert_not_called()
        viewer._rtx.step_async.assert_called_once()
        viewer.end_frame()
        viewer._accept_render.assert_called_once_with(pending.wait.return_value.fetch.return_value)
        viewer._rtx.step.assert_not_called()

    def test_paused_headless_capture_and_frame_budget(self):
        """Capture the last completed image while paused loops consume the budget."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._render_width = 4
        viewer._render_height = 3
        pending = mock.Mock()
        viewer._render_result = pending
        viewer.set_rendering_paused(True)
        with self.assertRaisesRegex(RuntimeError, "frame"):
            viewer.get_frame()
        pixels = np.full((3, 4, 4), 37, dtype=np.uint8)
        viewer._displayed_pixels = wp.array(pixels, dtype=wp.vec4ub, device="cpu")
        for i in range(4):
            self.assertTrue(viewer.is_running())
            viewer.begin_frame(float(i))
            viewer.end_frame()
            np.testing.assert_array_equal(viewer.get_frame().numpy(), pixels[:, :, :3])
        self.assertFalse(viewer.is_running())
        pending.wait.assert_not_called()
        viewer._rtx.step.assert_not_called()
        viewer._rtx.step_async.assert_not_called()

    def test_clear_and_close_release_renderer_once(self):
        """Drain a pending frame once and preserve pause through clear and close."""
        viewer = self.viewer
        viewer.set_rendering_paused(True)
        renderer = mock.Mock()
        viewer._rtx = renderer
        pending = mock.Mock()
        viewer._render_result = pending
        viewer._render_products = {"old": object()}
        viewer.clear_model()
        renderer.destroy.assert_called_once_with()
        self.assertTrue(viewer.is_rendering_paused())
        self.assertIsNone(viewer._render_products)
        viewer.close()
        viewer.close()
        renderer.destroy.assert_called_once_with()
        pending.wait.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
