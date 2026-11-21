# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest import mock

import newton.examples


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


class TestExampleRunnerLifecycle(unittest.TestCase):
    """Test example initialization and execution lifecycle behavior."""

    def _args(self, **overrides):
        defaults = {
            "viewer": "gl",
            "headless": False,
            "paused": False,
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
        """Verify initialization shows the loading splash for a visible GL viewer."""
        viewer = self._run_init(self._args())
        self.assertIn(("show_loading_splash", "Loading..."), viewer.calls)

    def test_init_skips_splash_for_headless(self):
        """Verify initialization skips the loading splash for a headless viewer."""
        viewer = self._run_init(self._args(headless=True))
        self.assertNotIn(("show_loading_splash", "Loading..."), viewer.calls)

    def test_run_hides_splash(self):
        """Verify the runner hides the loading splash before execution."""
        viewer = _RecordingViewer()
        example = SimpleNamespace(
            viewer=viewer,
            step=lambda: None,
            render=lambda: None,
        )
        args = SimpleNamespace(test=False)
        newton.examples.run(example, args)
        self.assertIn(("hide_loading_splash",), viewer.calls)

    def test_run_requires_test_final_in_test_mode(self):
        """Ensure per-step checks cannot replace the required completion check."""
        viewer = _RecordingViewer()
        example = SimpleNamespace(
            viewer=viewer,
            step=lambda: None,
            render=lambda: None,
            test_post_step=lambda: None,
        )
        args = SimpleNamespace(test=True)

        with self.assertRaisesRegex(NotImplementedError, "test_final"):
            newton.examples.run(example, args)


if __name__ == "__main__":
    unittest.main()
