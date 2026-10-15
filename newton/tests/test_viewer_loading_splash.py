# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from newton._src.viewer.viewer_gl import ViewerGL


class TestViewerGLLoadingSplashState(unittest.TestCase):
    """Direct state tests for ``show_loading_splash`` / ``hide_loading_splash``."""

    def _make_viewer(self):
        # Bypass ``ViewerGL.__init__`` (which would open a GL window) and
        # hand-initialize only the state the splash API touches. State lives on
        # ViewerGui; the viewer just delegates to ``self.gui``.
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.gui = SimpleNamespace(_loading_splash_active=False, _loading_splash_text=None)
        viewer.gui.show_loading_splash = lambda text=None: (
            setattr(viewer.gui, "_loading_splash_active", True),
            setattr(viewer.gui, "_loading_splash_text", text),
        )
        viewer.gui.hide_loading_splash = lambda: (
            setattr(viewer.gui, "_loading_splash_active", False),
            setattr(viewer.gui, "_loading_splash_text", None),
        )
        return viewer

    def test_show_sets_active_and_text(self):
        """Verify showing the splash records its active state and text."""
        viewer = self._make_viewer()
        viewer.show_loading_splash("Loading...")
        self.assertTrue(viewer.gui._loading_splash_active)
        self.assertEqual(viewer.gui._loading_splash_text, "Loading...")

    def test_hide_clears_state(self):
        """Verify hiding the splash clears its active state and text."""
        viewer = self._make_viewer()
        viewer.show_loading_splash("Loading...")
        viewer.hide_loading_splash()
        self.assertFalse(viewer.gui._loading_splash_active)
        self.assertIsNone(viewer.gui._loading_splash_text)

    def test_headless_no_gui_is_noop(self):
        """Verify splash operations are no-ops when no GUI exists."""
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.gui = None
        # Must not raise even though there is no GUI to drive.
        viewer.show_loading_splash("Loading...")
        viewer.hide_loading_splash()


if __name__ == "__main__":
    unittest.main()
