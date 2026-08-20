# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

from newton.viewer import ViewerGL


def _make_headless_viewer_or_skip(test: unittest.TestCase) -> ViewerGL:
    try:
        return ViewerGL(width=64, height=48, headless=True)
    except Exception as exc:
        test.skipTest(f"ViewerGL not available: {exc}")
        raise


class TestViewerGLMultisampling(unittest.TestCase):
    def test_window_config_requests_the_scene_sample_count(self):
        """Verify the window config is negotiated at the scene's sample count.

        The scene is never drawn into the window's own buffer: it is rendered
        into the multi-sampled framebuffer allocated at ``msaa_samples`` and
        resolved into a texture. Requesting a different count from the window
        therefore buys a default framebuffer that goes unused, and narrows the
        set of drivers that can match the request.
        """
        viewer = _make_headless_viewer_or_skip(self)

        try:
            granted = viewer.renderer.window.config.samples
            # A driver offering no multi-sampled config at all reports 0; any
            # other value has to agree with what the scene is rendered at.
            self.assertIn(granted, (0, viewer.renderer.msaa_samples))
        finally:
            viewer.close()

    def test_scene_keeps_msaa_without_a_multisampled_window_config(self):
        """Verify the scene stays anti-aliased when the window config has no MSAA."""
        try:
            import pyglet.window
        except Exception as exc:
            self.skipTest(f"pyglet window backend not available: {exc}")
            return

        real_window = pyglet.window.Window

        def _reject_multisampled_config(*args, **kwargs):
            if kwargs.pop("config", None) is not None:
                raise pyglet.window.NoSuchConfigException("no multi-sampled config")
            return real_window(*args, **kwargs)

        with mock.patch.object(pyglet.window, "Window", _reject_multisampled_config):
            viewer = _make_headless_viewer_or_skip(self)

        try:
            self.assertGreater(viewer.renderer.msaa_samples, 0)
            self.assertIsNotNone(viewer.renderer._frame_msaa_fbo)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
