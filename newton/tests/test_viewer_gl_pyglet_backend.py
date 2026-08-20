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


def _backend_module(name: str, attribute_name: str, backend_module: str) -> types.ModuleType:
    """Return a pyglet submodule whose backend class reports ``backend_module``."""
    module = types.ModuleType(name)
    backend_class = type("Backend", (), {})
    backend_class.__module__ = backend_module
    setattr(module, attribute_name, backend_class)
    return module


class TestPygletBackendSelection(unittest.TestCase):
    def test_needs_egl_only_for_displayless_linux(self):
        """Verify EGL is required only for headless Linux without DISPLAY."""
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
                    self.assertEqual(opengl._needs_egl_backend(True), expected)

    def test_get_pyglet_backend_reads_bound_submodule(self):
        """Verify the bound backend is read from an imported pyglet submodule."""
        window = _backend_module("pyglet.window", "Window", "pyglet.window.headless")
        with mock.patch.dict(sys.modules, {"pyglet.window": window}, clear=False):
            self.assertEqual(opengl._get_pyglet_backend(), "egl")

        gl = _backend_module("pyglet.gl", "Config", "pyglet.gl.xlib")
        with mock.patch.dict(
            sys.modules,
            {"pyglet.window": None, "pyglet.gl": gl, "pyglet.display": None},
            clear=False,
        ):
            self.assertEqual(opengl._get_pyglet_backend(), "native")

    def test_get_pyglet_backend_none_when_unbound(self):
        """Verify no backend is reported before any pyglet submodule imports."""
        with mock.patch.dict(
            sys.modules,
            {"pyglet.window": None, "pyglet.gl": None, "pyglet.display": None},
            clear=False,
        ):
            self.assertIsNone(opengl._get_pyglet_backend())

    def test_resolve_headless_returns_explicit_request(self):
        """Verify an explicit headless argument is returned unchanged."""
        pyglet = _pyglet_module(headless=True)
        with mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False):
            self.assertTrue(opengl._resolve_headless(True))
            self.assertFalse(opengl._resolve_headless(False))

    def test_resolve_headless_honors_configured_default(self):
        """Verify an unspecified request follows pyglet's configured headless flag.

        Covers displayed and non-Linux environments, where the honored default
        must survive rather than being reset to windowed.
        """
        environments = (
            ("linux", {}),
            ("linux", {"DISPLAY": ":0"}),
            ("darwin", {}),
            ("win32", {}),
        )

        for platform, environment in environments:
            with self.subTest(platform=platform, environment=environment):
                pyglet = _pyglet_module(headless=True)
                with (
                    mock.patch.object(sys, "platform", platform),
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
                ):
                    self.assertTrue(opengl._resolve_headless(None))

                pyglet = _pyglet_module(headless=False)
                with (
                    mock.patch.object(sys, "platform", platform),
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
                ):
                    self.assertFalse(opengl._resolve_headless(None))

    def test_selects_native_backend_for_displayed_linux(self):
        """Verify displayed Linux keeps pyglet's native hidden-window backend."""
        pyglet = _pyglet_module()
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True),
            mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
            mock.patch.object(opengl, "_get_pyglet_backend", return_value=None),
        ):
            opengl._select_pyglet_backend(True)

        self.assertFalse(pyglet.options["headless"])

    def test_preserves_configured_egl_backend(self):
        """Verify a configured EGL backend is not reset on displayed or non-Linux hosts.

        Regression test: selecting the native path must never overwrite an
        explicitly configured ``pyglet.options["headless"] = True``.
        """
        for platform, environment in (("linux", {"DISPLAY": ":0"}), ("darwin", {}), ("win32", {})):
            with self.subTest(platform=platform, environment=environment):
                pyglet = _pyglet_module(headless=True)
                with (
                    mock.patch.object(sys, "platform", platform),
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
                    mock.patch.object(opengl, "_get_pyglet_backend", return_value=None),
                ):
                    opengl._select_pyglet_backend(True)

                self.assertTrue(pyglet.options["headless"])

    def test_native_selection_leaves_headless_option_unset(self):
        """Verify native selection does not write a False headless option."""
        pyglet = _pyglet_module(headless=False)
        with (
            mock.patch.object(sys, "platform", "darwin"),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
            mock.patch.object(opengl, "_get_pyglet_backend", return_value=None),
        ):
            opengl._select_pyglet_backend(False)

        self.assertFalse(pyglet.options["headless"])

    def test_rejects_native_backend_after_egl_import(self):
        """Verify a native request cannot follow an EGL pyglet import."""
        pyglet = _pyglet_module()
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True),
            mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
            mock.patch.object(opengl, "_get_pyglet_backend", return_value="egl"),
            self.assertRaisesRegex(RuntimeError, "cannot switch pyglet from its egl backend to native"),
        ):
            opengl._select_pyglet_backend(False)

    def test_rejects_windowed_request_when_egl_configured(self):
        """Verify a windowed request is rejected when pyglet is configured for EGL."""
        pyglet = _pyglet_module(headless=True)
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True),
            mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
            mock.patch.object(opengl, "_get_pyglet_backend", return_value=None),
            self.assertRaisesRegex(RuntimeError, "cannot create a windowed renderer"),
        ):
            opengl._select_pyglet_backend(False)

    def test_rejects_egl_backend_after_native_import(self):
        """Verify an EGL request cannot follow a native pyglet import."""
        pyglet = _pyglet_module()
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(sys.modules, {"pyglet": pyglet}, clear=False),
            mock.patch.object(opengl, "_get_pyglet_backend", return_value="native"),
            self.assertRaisesRegex(RuntimeError, "cannot switch pyglet from its native backend to egl"),
        ):
            opengl._select_pyglet_backend(True)


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux")
class TestPygletBackendSelectionSubprocess(unittest.TestCase):
    _UNAVAILABLE_EXIT_CODE = 99

    def test_initializes_egl_without_display(self):
        """Verify headless RendererGL initializes without DISPLAY, or skips if EGL is absent."""
        environment = {key: value for key, value in os.environ.items() if key not in {"DISPLAY", "WAYLAND_DISPLAY"}}
        code = (
            "import sys\n"
            "from newton.tests.unittest_utils import is_viewer_gl_unavailable_error\n"
            "from newton._src.viewer.gl.opengl import RendererGL\n"
            "try:\n"
            "    renderer = RendererGL(headless=True, screen_width=64, screen_height=48, vsync=False)\n"
            "    assert renderer.headless\n"
            "    renderer.close()\n"
            "except Exception as exc:\n"
            f"    sys.exit({self._UNAVAILABLE_EXIT_CODE} if is_viewer_gl_unavailable_error(exc) else 1)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )
        if result.returncode == self._UNAVAILABLE_EXIT_CODE:
            self.skipTest(f"headless GL backend unavailable:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
