# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise rendering pause independently of simulation stepping."""

import unittest
from unittest import mock

import numpy as np

from newton._src.viewer.viewer_gui import ViewerGui
from newton.viewer import ViewerNull


class TestRenderingPauseBase(unittest.TestCase):
    def test_unsupported_backend_ignores_pause(self):
        """Ignore unsupported rendering pause without affecting simulation."""
        viewer = ViewerNull()
        self.assertFalse(viewer.is_rendering_paused())
        viewer.set_rendering_paused(False)
        viewer.set_rendering_paused(True)
        self.assertFalse(viewer.is_rendering_paused())
        self.assertTrue(viewer.should_step())


class TestRenderingPauseGui(unittest.TestCase):
    def test_pause_releases_interaction_and_blocks_navigation(self):
        """Stop camera inertia and active gizmos until rendering resumes."""
        viewer = mock.Mock()
        viewer.is_rendering_paused.return_value = True
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.ui = None
        gui._cam_vel = np.ones(3)
        gui._gizmo_active = {"body": True}
        gui.on_rendering_paused()
        np.testing.assert_array_equal(gui._cam_vel, np.zeros(3))
        self.assertEqual(gui._gizmo_active, {})
        self.assertFalse(viewer.gizmo_is_using)
        self.assertTrue(gui.should_ignore_mouse_input(allow_active_pick_drag=True))
        gui.update_camera_from_keys(0.1, lambda _: True)
        viewer.camera.get_front.assert_not_called()
        viewer.is_rendering_paused.return_value = False
        self.assertFalse(gui.should_ignore_mouse_input())


if __name__ == "__main__":
    unittest.main()
