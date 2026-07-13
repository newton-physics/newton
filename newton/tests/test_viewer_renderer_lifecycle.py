# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the OpenGL renderer lifecycle."""

import unittest
from types import SimpleNamespace
from unittest import mock

from newton._src.viewer.gl.opengl import RendererGL


class TestRendererGLClose(unittest.TestCase):
    def test_close_keeps_pyglet_event_loop_available_for_next_viewer(self):
        """Closing one viewer must not stop Pyglet's shared event loop."""
        renderer = RendererGL.__new__(RendererGL)
        renderer._make_current = mock.Mock()
        renderer.window = mock.Mock()
        renderer.app = SimpleNamespace(event_loop=mock.Mock(), platform_event_loop=mock.Mock())
        renderer.headless = False
        RendererGL._fallback_texture = object()

        renderer.close()

        renderer._make_current.assert_called_once_with()
        renderer.window.close.assert_called_once_with()
        renderer.app.event_loop.dispatch_event.assert_not_called()
        renderer.app.platform_event_loop.stop.assert_not_called()
        self.assertIsNone(RendererGL._fallback_texture)


if __name__ == "__main__":
    unittest.main()
