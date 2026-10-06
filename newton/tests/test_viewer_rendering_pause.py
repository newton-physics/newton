# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise rendering pause independently of simulation stepping."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip
from newton.viewer import ViewerNull


class TestRenderingPauseBase(unittest.TestCase):
    def test_unsupported_backend_rejects_pause(self):
        """Reject unsupported rendering pause without affecting simulation."""
        viewer = ViewerNull()
        self.assertFalse(viewer.is_rendering_paused())
        viewer.set_rendering_paused(False)
        with self.assertRaises(NotImplementedError):
            viewer.set_rendering_paused(True)
        self.assertTrue(viewer.should_step())


class TestRenderingPauseGL(unittest.TestCase):
    def test_freeze_capture_resize_and_resume(self):
        """Keep an image stable through logging, resize, and UI-driven toggles."""
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self)
        self.addCleanup(viewer.close)
        before = np.zeros((48, 64, 3), dtype=np.uint8)
        before[:24, :, 0] = 255
        before[24:, :, 1] = 255
        after = np.zeros_like(before)
        after[:, :, 2] = 255

        viewer.set_rendering_paused(True)
        viewer.begin_frame(0.0)
        viewer.end_frame()
        with self.assertRaisesRegex(RuntimeError, "displayed frame"):
            viewer.get_frame()
        viewer.set_rendering_paused(False)
        viewer.begin_frame(1.0)
        viewer.log_image("sensor", before, fullscreen=True)
        viewer.end_frame()
        reference = viewer.get_frame().numpy().copy()
        np.testing.assert_array_equal(reference, before)

        gui = mock.Mock()
        gui.prepare_frame.side_effect = lambda: viewer.set_rendering_paused(True)
        viewer.gui = gui
        viewer.begin_frame(2.0)
        viewer.log_image("sensor", after, fullscreen=True)
        with mock.patch.object(viewer.renderer, "render", wraps=viewer.renderer.render) as render:
            viewer.end_frame()
            render.assert_not_called()
        np.testing.assert_array_equal(viewer.get_frame().numpy(), reference)
        np.testing.assert_array_equal(viewer._displayed_frame.pixels()[::-1, :, :3], reference)
        gui.render_prepared_frame.assert_called_once()
        viewer.gui = None

        viewer.renderer.window.set_size(128, 96)
        viewer.renderer._on_window_resize(128, 96)
        viewer.begin_frame(3.0)
        viewer.end_frame()
        resized = viewer.get_frame().numpy()
        self.assertEqual(resized.shape, (96, 128, 3))
        np.testing.assert_array_equal(resized[10, 10], [255, 0, 0])
        np.testing.assert_array_equal(resized[80, 10], [0, 255, 0])

        viewer.set_rendering_paused(False)
        viewer.begin_frame(4.0)
        viewer.log_image("sensor", after, fullscreen=True)
        viewer.end_frame()
        np.testing.assert_array_equal(viewer.get_frame().numpy()[10, 10], [0, 0, 255])
        viewer.set_rendering_paused(True)
        viewer.clear_model()
        viewer.begin_frame(5.0)
        viewer.end_frame()
        self.assertTrue(viewer.is_rendering_paused())
        with self.assertRaises(RuntimeError):
            viewer.get_frame()


if __name__ == "__main__":
    unittest.main()
