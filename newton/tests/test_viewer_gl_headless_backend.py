# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from newton._src.viewer.gl import opengl

# Run the viewer in a session that offers no display and has not already put
# pyglet in headless mode, which is the configuration the selection exists for.
_DISPLAY_VARS = ("DISPLAY", "WAYLAND_DISPLAY", "PYGLET_HEADLESS")

# Exit code the subprocess reserves for a host that cannot render at all, which
# the backend selection can do nothing about. Every other non-zero exit is a
# failure of the behavior under test.
_UNAVAILABLE_EXIT_CODE = 3

_CONSTRUCT_HEADLESS_VIEWER = textwrap.dedent(
    f"""
    import sys

    import newton.viewer

    # The pyglet errors that mean the machine offers no usable GL, kept apart
    # from a real failure. NoSuchDisplayException is deliberately absent: it is
    # exactly the failure this test exists to catch.
    UNAVAILABLE = {{"ConfigException", "ContextException", "MissingFunctionException", "NoSuchConfigException"}}

    try:
        viewer = newton.viewer.ViewerGL(width=64, height=48, headless=True)
    except Exception as exc:
        error_type = type(exc)
        if error_type.__module__.startswith("pyglet.") and error_type.__name__ in UNAVAILABLE:
            print(f"VIEWER_UNAVAILABLE: {{error_type.__name__}}: {{exc}}")
            sys.exit({_UNAVAILABLE_EXIT_CODE})
        raise

    try:
        print("VIEWER_CONSTRUCTED")
    finally:
        viewer.close()
    """
)


class TestPygletHeadlessBackendSelection(unittest.TestCase):
    # Resolved per call rather than imported by name, so that the behavioral
    # test below, which reaches the same selection through ViewerGL, still
    # runs when the selection is missing.
    @staticmethod
    def _needs_headless_backend(headless):
        return opengl.needs_pyglet_headless_backend(headless)

    def test_headless_viewer_without_a_display_needs_the_headless_backend(self):
        """Verify a headless viewer on a display-less Linux session selects the headless backend."""
        with mock.patch.object(sys, "platform", "linux"), mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(self._needs_headless_backend(True))

    def test_windowed_viewer_keeps_the_default_backend(self):
        """Verify a windowed viewer is left on pyglet's default backend."""
        with mock.patch.object(sys, "platform", "linux"), mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self._needs_headless_backend(False))
            self.assertFalse(self._needs_headless_backend(None))

    def test_an_available_display_keeps_the_default_backend(self):
        """Verify a session with a display keeps the default backend, headless viewer or not.

        Headless rendering is about not showing a window, not about the
        session lacking one, so a viewer on a machine that has a display must
        keep the backend that can drive it.
        """
        for variable in ("DISPLAY", "WAYLAND_DISPLAY"):
            with self.subTest(variable=variable):
                with (
                    mock.patch.object(sys, "platform", "linux"),
                    mock.patch.dict(os.environ, {variable: ":0"}, clear=True),
                ):
                    self.assertFalse(self._needs_headless_backend(True))

    def test_other_platforms_keep_the_default_backend(self):
        """Verify the selection is confined to Linux, whose default backend is Xlib."""
        for platform in ("darwin", "win32"):
            with self.subTest(platform=platform):
                with (
                    mock.patch.object(sys, "platform", platform),
                    mock.patch.dict(os.environ, {}, clear=True),
                ):
                    self.assertFalse(self._needs_headless_backend(True))


class TestViewerGLHeadlessWithoutDisplay(unittest.TestCase):
    def test_headless_viewer_constructs_without_a_display(self):
        """Verify ``ViewerGL(headless=True)`` opens with no display and no pyglet setup.

        pyglet binds its display backend once per process, so this runs in a
        subprocess with the display variables scrubbed. Without the backend
        selection the constructor raises ``NoSuchDisplayException``.
        """
        if any(os.environ.get(variable) for variable in _DISPLAY_VARS[:2]):
            self.skipTest("session has a display, so the headless backend is not selected")
        if not sys.platform.startswith("linux"):
            self.skipTest("pyglet only defaults to Xlib on Linux")
        if importlib.util.find_spec("pyglet") is None:
            self.skipTest("ViewerGL dependencies not available: pyglet is not installed")

        env = {key: value for key, value in os.environ.items() if key not in _DISPLAY_VARS}
        result = subprocess.run(
            [sys.executable, "-c", _CONSTRUCT_HEADLESS_VIEWER],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )

        if result.returncode == _UNAVAILABLE_EXIT_CODE:
            self.skipTest(f"ViewerGL backend not available:\n{result.stdout}")

        if result.returncode != 0:
            self.assertNotIn(
                "NoSuchDisplayException",
                result.stderr,
                "ViewerGL(headless=True) must not ask pyglet for a display it cannot have",
            )
            self.fail(f"ViewerGL(headless=True) failed without a display:\n{result.stdout}\n{result.stderr}")

        self.assertIn("VIEWER_CONSTRUCTED", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
