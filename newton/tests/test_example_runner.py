# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import newton.examples
import newton.tests.unittest_utils

# Runs an example process in the mode given as its argument and prints, as its last line of
# output, whether collections still visit the objects it made, at each point of its life.
_EXIT_PROBE = """
import atexit
import gc
import json
import sys
from types import SimpleNamespace

markers = {}
report = {}


class Marker:
    pass


def is_collectable(obj):
    # gc.freeze() moves objects to a generation that collections skip and gc.get_objects() omits.
    return any(o is obj for o in gc.get_objects())


def report_at_exit():
    for name, marker in markers.items():
        report[name + "_at_exit"] = is_collectable(marker)
    print(json.dumps(report))


# Registered first, so it runs last: only the interpreter's shutdown comes after it.
atexit.register(report_at_exit)

import newton.examples
from newton.viewer import ViewerNull

mode = sys.argv[1]


class Example:
    def __init__(self, viewer):
        self.viewer = viewer

    def step(self):
        if self.viewer.frame_count == 1:
            markers["late"] = Marker()
            report["early_in_last_frame"] = is_collectable(markers["early"])
            if mode == "exit":
                sys.exit(3)

    def render(self):
        self.viewer.begin_frame(0.0)
        self.viewer.end_frame()


markers["early"] = Marker()
if mode == "list":
    sys.argv = ["newton.examples", "--list"]
    newton.examples.main()
else:
    newton.examples.run(Example(ViewerNull(num_frames=2)), SimpleNamespace(test=False))
    report["early_after_run"] = is_collectable(markers["early"])
    report["late_after_run"] = is_collectable(markers["late"])
"""


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


class TestExampleRunnerExit(unittest.TestCase):
    """Test that an example process spares interpreter shutdown its full collections."""

    def _run_probe(self, mode):
        env = os.environ.copy()
        env.pop("PYTHONWARNINGS", None)
        result = subprocess.run(
            [sys.executable, *newton.tests.unittest_utils.get_strict_warning_args(), "-c", _EXIT_PROBE, mode],
            capture_output=True,
            env=env,
            text=True,
            check=False,
        )
        try:
            report = json.loads(result.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            self.fail(f"The probe exited with status {result.returncode} and no report: {result.stderr}")
        return result, report

    def test_run_freezes_objects_at_exit_only(self):
        """Verify a finished run freezes its objects when the process exits, not before."""
        result, report = self._run_probe("run")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # The example browser and the batch runner collect between examples.
        self.assertTrue(report["early_in_last_frame"])
        self.assertTrue(report["early_after_run"])
        self.assertTrue(report["late_after_run"])
        self.assertFalse(report["early_at_exit"])
        self.assertFalse(report["late_at_exit"])

    def test_run_freezes_objects_when_example_exits(self):
        """Verify a run that an example ends with ``sys.exit()`` freezes its objects at exit."""
        result, report = self._run_probe("exit")
        self.assertEqual(result.returncode, 3, msg=result.stderr)
        self.assertTrue(report["early_in_last_frame"])
        self.assertFalse(report["early_at_exit"])
        self.assertFalse(report["late_at_exit"])

    def test_main_freezes_objects_when_listing_examples(self):
        """Verify ``main()`` freezes objects at exit when it exits without running an example."""
        result, report = self._run_probe("list")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("basic_pendulum", result.stdout)
        self.assertFalse(report["early_at_exit"])


if __name__ == "__main__":
    unittest.main()
