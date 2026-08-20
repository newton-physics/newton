# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import types
import unittest
from unittest import mock

from newton._src.viewer.gl import opengl


def _pyglet_module(*, headless: bool = False) -> types.ModuleType:
    """Return a minimal pyglet module for backend configuration tests."""
    pyglet = types.ModuleType("pyglet")
    pyglet.options = {"headless": headless, "debug_gl": True}
    return pyglet


def _backend_module(name: str, backend_module: str) -> types.ModuleType:
    """Return a pyglet submodule that reports a selected backend."""
    module = types.ModuleType(name)
    backend_class = type("Backend", (), {})
    backend_class.__module__ = backend_module
    if name == "pyglet.window":
        module.Window = backend_class
    elif name == "pyglet.gl":
        module.Config = backend_class
    else:
        module.Display = backend_class
    return module


class TestPygletBackendSelection(unittest.TestCase):
    def test_selects_egl_only_for_displayless_linux(self):
        """Verify EGL is selected only for headless Linux without DISPLAY."""
        environments = (
            ("linux", {}, True),
            ("linux", {"DISPLAY": ":0"}, False),
            ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, True),
            ("darwin", {}, False),
            ("win32", {}, False),
        )

        for platform, environment, expected in environments:
            with self.subTest(platform=platform, environment=environment):
                with (
                    mock.patch.object(sys, "platform", platform),
                    mock.patch.dict(os.environ, environment, clear=True),
                ):
                    self.assertEqual(opengl._requests_pyglet_egl(True), expected)

    def test_configures_native_backend_for_displayed_linux(self):
        """Verify displayed Linux preserves pyglet's native hidden-window backend."""
        pyglet = _pyglet_module()
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True),
            mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
        ):
            effective_headless, configured_pyglet = opengl._configure_pyglet_backend(True)

        self.assertTrue(effective_headless)
        self.assertIs(configured_pyglet, pyglet)
        self.assertFalse(pyglet.options["headless"])

    def test_honors_pyglet_headless_default(self):
        """Verify an unspecified request preserves pyglet's headless default."""
        pyglet = _pyglet_module(headless=True)
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
        ):
            effective_headless, _ = opengl._configure_pyglet_backend(None)

        self.assertTrue(effective_headless)
        self.assertTrue(pyglet.options["headless"])

    def test_rejects_native_backend_after_egl_import(self):
        """Verify a native request cannot follow an EGL pyglet import."""
        pyglet = _pyglet_module(headless=True)
        window = _backend_module("pyglet.window", "pyglet.window.headless")
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True),
            mock.patch.dict(sys.modules, {"pyglet": pyglet, "pyglet.window": window}, clear=False),
            self.assertRaisesRegex(RuntimeError, "cannot switch pyglet from its egl backend to native"),
        ):
            opengl._configure_pyglet_backend(False)

    def test_rejects_egl_backend_after_native_import(self):
        """Verify an EGL request cannot follow a native pyglet import."""
        pyglet = _pyglet_module()
        gl = _backend_module("pyglet.gl", "pyglet.gl.xlib")
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(sys.modules, {"pyglet": pyglet, "pyglet.gl": gl}, clear=False),
            self.assertRaisesRegex(RuntimeError, "cannot switch pyglet from its native backend to egl"),
        ):
            opengl._configure_pyglet_backend(True)


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux")
class TestPygletBackendSelectionSubprocess(unittest.TestCase):
    def test_initializes_egl_without_display(self):
        """Verify headless RendererGL initializes without DISPLAY."""
        environment = {key: value for key, value in os.environ.items() if key not in {"DISPLAY", "WAYLAND_DISPLAY"}}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from newton._src.viewer.gl.opengl import RendererGL; "
                    "renderer = RendererGL(headless=True, screen_width=64, screen_height=48, vsync=False); "
                    "assert renderer.headless; renderer.close()"
                ),
            ],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
