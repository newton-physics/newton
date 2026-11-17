# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton._src.viewer.viewer_gl import ViewerGL


class TestViewerGLLoadingSplashState(unittest.TestCase):
    """Splash calls on a viewer without a GUI."""

    def test_headless_no_gui_is_noop(self):
        """Verify splash operations are no-ops when no GUI exists."""
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.gui = None
        # Must not raise even though there is no GUI to drive.
        viewer.show_loading_splash("Loading...")
        viewer.hide_loading_splash()


if __name__ == "__main__":
    unittest.main()
