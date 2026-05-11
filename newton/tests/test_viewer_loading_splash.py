# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest import mock

import newton.examples
from newton._src.viewer.viewer_gl import ViewerGL


class TestViewerGLLoadingSplashState(unittest.TestCase):
    """Direct state tests for ``show_loading_splash`` / ``hide_loading_splash``."""

    def _make_viewer(self):
        # Bypass ``ViewerGL.__init__`` (which would open a GL window) and
        # hand-initialize only the state the splash API touches.
        viewer = ViewerGL.__new__(ViewerGL)
        viewer._loading_splash_active = False
        viewer._loading_splash_text = None
        return viewer

    def test_show_sets_active_and_text(self):
        viewer = self._make_viewer()
        viewer.show_loading_splash("Loading...")
        self.assertTrue(viewer._loading_splash_active)
        self.assertEqual(viewer._loading_splash_text, "Loading...")

    def test_hide_clears_state(self):
        viewer = self._make_viewer()
        viewer.show_loading_splash("Loading...")
        viewer.hide_loading_splash()
        self.assertFalse(viewer._loading_splash_active)
        self.assertIsNone(viewer._loading_splash_text)


class _RecordingViewer:
    """Stub viewer recording observable calls, used by the lifecycle tests."""

    def __init__(self):
        self.calls = []

    def show_loading_splash(self, text=None):
        self.calls.append(("show_loading_splash", text))

    def hide_loading_splash(self):
        self.calls.append(("hide_loading_splash",))

    def begin_frame(self, t):
        self.calls.append(("begin_frame", t))

    def end_frame(self):
        self.calls.append(("end_frame",))

    def is_running(self):
        return False

    def is_paused(self):
        return False

    def close(self):
        self.calls.append(("close",))


class TestLoadingSplashLifecycle(unittest.TestCase):
    """``init()`` shows the splash for visible GL viewers; ``run()`` hides it."""

    def _args(self, **overrides):
        defaults = {
            "viewer": "gl",
            "headless": False,
            "device": None,
            "quiet": True,
            "warp_config": [],
            "benchmark": False,
            "realtime": False,
            "output_path": None,
            "num_frames": 1,
            "rerun_address": None,
            "test": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _run_init(self, args):
        stub = _RecordingViewer()
        parser = mock.MagicMock()
        parser.parse_args.return_value = args
        with (
            mock.patch("newton.viewer.ViewerGL", return_value=stub),
            mock.patch("newton.examples._apply_warp_config"),
        ):
            newton.examples.init(parser=parser)
        return stub

    def test_init_shows_splash_for_visible_gl(self):
        viewer = self._run_init(self._args())
        self.assertIn(("show_loading_splash", "Loading..."), viewer.calls)

    def test_init_skips_splash_for_headless(self):
        viewer = self._run_init(self._args(headless=True))
        self.assertNotIn(("show_loading_splash", "Loading..."), viewer.calls)

    def test_run_hides_splash(self):
        viewer = _RecordingViewer()
        example = SimpleNamespace(
            viewer=viewer,
            step=lambda: None,
            render=lambda: None,
        )
        args = SimpleNamespace(test=False)
        newton.examples.run(example, args)
        self.assertIn(("hide_loading_splash",), viewer.calls)


if __name__ == "__main__":
    unittest.main()
