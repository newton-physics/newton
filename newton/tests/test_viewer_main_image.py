# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import time
import unittest

from newton._src.viewer.gl.image_logger import LoggedImageTexture
from newton._src.viewer.gl.opengl import _texture_tile_uv_rect
from newton._src.viewer.viewer_gl import ViewerGL


class _DummyRenderer:
    def __init__(self):
        self.render_calls = []
        self.render_texture_calls = []
        self.present_calls = 0

    def update(self):
        pass

    def has_exit(self):
        return False

    def render(self, *args):
        self.render_calls.append(args)

    def render_texture(self, *args, **kwargs):
        self.render_texture_calls.append((args, kwargs))

    def present(self):
        self.present_calls += 1


class _DummyImageLogger:
    def __init__(self, texture):
        self.texture = texture
        self.queries = []
        self.log_calls = []

    def get_texture(self, name):
        self.queries.append(name)
        return self.texture

    def log(self, name, image):
        self.log_calls.append((name, image))


class _DummyGui:
    def __init__(self):
        self.render_frame_calls = []

    def render_frame(self, *, update_fps):
        self.render_frame_calls.append(update_fps)


def _make_viewer(texture=None, main_image_name="color"):
    if texture is None:
        texture = LoggedImageTexture(
            texture_id=7,
            texture_width=64,
            texture_height=32,
            tile_count=2,
            tile_width=32,
            tile_height=32,
            atlas_cols=2,
        )
    viewer = ViewerGL.__new__(ViewerGL)
    object.__setattr__(viewer, "renderer", _DummyRenderer())
    object.__setattr__(viewer, "_image_logger", _DummyImageLogger(texture))
    object.__setattr__(viewer, "gui", _DummyGui())
    object.__setattr__(viewer, "_main_image_name", main_image_name)
    object.__setattr__(viewer, "_last_time", time.perf_counter())
    object.__setattr__(viewer, "_update_camera", lambda _dt: None)
    object.__setattr__(viewer, "wind", None)
    object.__setattr__(viewer, "camera", object())
    object.__setattr__(viewer, "objects", {})
    object.__setattr__(viewer, "lines", {})
    object.__setattr__(viewer, "wireframe_shapes", {})
    object.__setattr__(viewer, "arrows", {})
    return viewer


class TestViewerGLMainImage(unittest.TestCase):
    def test_texture_tile_uv_rect_flips_logged_image_rows(self):
        """Verify logged image atlas UVs flip vertically for OpenGL drawing."""
        self.assertEqual(_texture_tile_uv_rect(0, 32, 32, 64, 64, 2), (0.0, 0.5, 0.5, 0.0))
        self.assertEqual(_texture_tile_uv_rect(3, 32, 32, 64, 64, 2), (0.5, 1.0, 1.0, 0.5))

    def test_update_uses_main_image_texture_without_scene_rendering(self):
        """Verify main-image mode bypasses the 3D scene renderer."""
        viewer = _make_viewer(main_image_name="color")

        ViewerGL._update(viewer)

        self.assertEqual(viewer._image_logger.queries, ["color"])
        self.assertEqual(
            viewer.renderer.render_texture_calls,
            [
                (
                    (7, 64, 32),
                    {
                        "tile_count": 2,
                        "tile_width": 32,
                        "tile_height": 32,
                        "atlas_cols": 2,
                    },
                )
            ],
        )
        self.assertEqual(viewer.renderer.render_calls, [])
        self.assertEqual(viewer.gui.render_frame_calls, [True])
        self.assertEqual(viewer.renderer.present_calls, 1)
        self.assertIsNone(viewer._main_image_name)

    def test_update_clears_window_when_main_image_is_not_logged(self):
        """Verify missing main images clear instead of rendering stale scene pixels."""
        viewer = _make_viewer(texture=None, main_image_name="color")
        viewer._image_logger.texture = None

        ViewerGL._update(viewer)

        self.assertEqual(viewer._image_logger.queries, ["color"])
        self.assertEqual(viewer.renderer.render_texture_calls, [((None, 0, 0), {})])
        self.assertEqual(viewer.renderer.render_calls, [])
        self.assertEqual(viewer.renderer.present_calls, 1)
        self.assertIsNone(viewer._main_image_name)

    def test_update_uses_scene_renderer_without_main_image(self):
        """Verify normal scene rendering remains active when no main image is set."""
        viewer = _make_viewer(main_image_name=None)

        ViewerGL._update(viewer)

        self.assertEqual(viewer._image_logger.queries, [])
        self.assertEqual(viewer.renderer.render_texture_calls, [])
        self.assertEqual(len(viewer.renderer.render_calls), 1)
        self.assertEqual(viewer.renderer.present_calls, 1)
        self.assertIsNone(viewer._main_image_name)

    def test_log_main_image_logs_and_selects_current_frame(self):
        """Verify the public main-image logger uploads and selects the image."""
        viewer = ViewerGL.__new__(ViewerGL)
        image = object()
        logger = _DummyImageLogger(texture=None)
        object.__setattr__(viewer, "_qualify", lambda name: f"/layers/test/{name}")
        object.__setattr__(viewer, "_image_logger", logger)

        ViewerGL.log_main_image(viewer, "color", image)

        self.assertEqual(logger.log_calls, [("/layers/test/color", image)])
        self.assertEqual(viewer._main_image_name, "/layers/test/color")

    def test_log_main_image_rejects_empty_name(self):
        """Verify the public main-image logger rejects invalid names."""
        viewer = ViewerGL.__new__(ViewerGL)
        object.__setattr__(viewer, "_qualify", lambda name: name)
        object.__setattr__(viewer, "_image_logger", _DummyImageLogger(texture=None))

        with self.assertRaises(ValueError):
            ViewerGL.log_main_image(viewer, "", object())


if __name__ == "__main__":
    unittest.main()
