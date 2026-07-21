# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from typing import get_type_hints
from unittest import mock

import numpy as np

from newton._src.viewer.gl.opengl import RendererGL
from newton._src.viewer.viewer_gl import ViewerGL


class _FakeGL:
    GL_READ_FRAMEBUFFER = 0x8CA8
    GL_DRAW_FRAMEBUFFER = 0x8CA9
    GL_COLOR_ATTACHMENT0 = 0x8CE0

    def __init__(self):
        self.calls = []

    def glBindFramebuffer(self, target, framebuffer):
        self.calls.append(("bind_framebuffer", target, int(framebuffer)))

    def glReadBuffer(self, buffer):
        self.calls.append(("read_buffer", buffer))

    def glDrawBuffer(self, buffer):
        self.calls.append(("draw_buffer", buffer))

    def glViewport(self, x, y, width, height):
        self.calls.append(("viewport", x, y, width, height))


class _FakeRenderer:
    def __init__(self):
        self._screen_width = 640
        self._screen_height = 360
        self._frame_fbo = 1
        self._frame_texture = 2
        self._frame_depth_texture = 3
        self._post_process_fbo = 4
        self._post_process_texture = 5
        self._post_process_depth_texture = 6
        self._view_matrix = np.arange(16, dtype=np.float32)
        self._projection_matrix = np.arange(16, 32, dtype=np.float32)
        self.ensure_count = 0
        self.make_current_count = 0

    def _ensure_post_process_target(self):
        self.ensure_count += 1

    def _make_current(self):
        self.make_current_count += 1


def _make_viewer():
    viewer = ViewerGL.__new__(ViewerGL)
    viewer.renderer = _FakeRenderer()
    viewer.camera = SimpleNamespace(
        pos=(1.0, 2.0, 3.0),
        get_right=lambda: (1.0, 0.0, 0.0),
        get_up=lambda: (0.0, 1.0, 0.0),
        get_front=lambda: (0.0, 0.0, -1.0),
        fov=55.0,
        near=0.05,
        far=500.0,
        up_axis=2,
    )
    viewer._post_process_registrations = []
    viewer._post_process_closed = False
    return viewer


class TestViewerGLPostProcessRegistration(unittest.TestCase):
    def test_registration_close_is_idempotent_and_reentrant(self):
        viewer = _make_viewer()
        cleanup_count = 0
        registration = None

        def cleanup():
            nonlocal cleanup_count
            cleanup_count += 1
            registration.close()

        registration = viewer.register_post_process(lambda context: None, cleanup=cleanup)
        self.assertFalse(registration.closed)

        registration.close()
        registration.close()

        self.assertTrue(registration.closed)
        self.assertEqual(cleanup_count, 1)
        self.assertEqual(viewer._post_process_registrations, [])
        self.assertEqual(viewer.renderer.make_current_count, 1)

    def test_viewer_cleanup_closes_registrations_in_reverse_order(self):
        viewer = _make_viewer()
        events = []
        registrations = [
            viewer.register_post_process(lambda context: None, cleanup=lambda name=name: events.append(name))
            for name in ("a", "b", "c")
        ]

        viewer._close_post_processes()
        viewer._close_post_processes()

        self.assertEqual(events, ["c", "b", "a"])
        self.assertTrue(all(registration.closed for registration in registrations))
        self.assertEqual(viewer._post_process_registrations, [])

    def test_rejects_invalid_callbacks(self):
        viewer = _make_viewer()

        with self.assertRaises(TypeError):
            viewer.register_post_process(None)
        with self.assertRaises(TypeError):
            viewer.register_post_process(lambda context: None, cleanup=object())

    def test_public_annotations_resolve_at_runtime(self):
        context_hints = get_type_hints(ViewerGL.PostProcessContext)
        registration_hints = get_type_hints(ViewerGL.register_post_process)

        self.assertIs(context_hints["camera"], ViewerGL.CameraSnapshot)
        self.assertIs(registration_hints["return"], ViewerGL.PostProcessRegistration)


class TestViewerGLPostProcessDispatch(unittest.TestCase):
    def test_callbacks_receive_color_depth_ping_pong_and_camera_snapshot(self):
        viewer = _make_viewer()
        fake_gl = _FakeGL()
        contexts = []

        for _ in range(3):
            viewer.register_post_process(contexts.append)

        with mock.patch.object(RendererGL, "gl", fake_gl):
            final_target = viewer._render_post_processes()

        self.assertEqual(final_target, (4, 5, 6))
        self.assertEqual(
            [
                (
                    context.input_framebuffer_id,
                    context.input_color_texture_id,
                    context.input_depth_texture_id,
                    context.output_framebuffer_id,
                    context.output_color_texture_id,
                    context.output_depth_texture_id,
                )
                for context in contexts
            ],
            [
                (1, 2, 3, 4, 5, 6),
                (4, 5, 6, 1, 2, 3),
                (1, 2, 3, 4, 5, 6),
            ],
        )

        for context in contexts:
            self.assertEqual(context.viewport, (0, 0, 640, 360))
            self.assertEqual(context.width, 640)
            self.assertEqual(context.height, 360)
            self.assertEqual(context.camera.position, (1.0, 2.0, 3.0))
            self.assertEqual(context.camera.forward, (0.0, 0.0, -1.0))
            self.assertEqual(context.camera.fov_y, 55.0)
            with self.assertRaises(ValueError):
                context.camera.view_matrix[0] = 0.0
            with self.assertRaises(ValueError):
                context.camera.projection_matrix[0] = 0.0
        self.assertIs(contexts[0].camera, contexts[1].camera)
        self.assertIs(contexts[1].camera, contexts[2].camera)

        self.assertEqual(
            [call for call in fake_gl.calls if call[0] == "viewport"],
            [("viewport", 0, 0, 640, 360)] * 3,
        )

    def test_closed_callback_is_skipped_without_consuming_a_target(self):
        viewer = _make_viewer()
        fake_gl = _FakeGL()
        events = []
        registration_b = None

        def callback_a(context):
            events.append(("a", context.input_framebuffer_id, context.output_framebuffer_id))
            registration_b.close()

        def callback_b(context):
            events.append(("b", context.input_framebuffer_id, context.output_framebuffer_id))

        def callback_c(context):
            events.append(("c", context.input_framebuffer_id, context.output_framebuffer_id))

        viewer.register_post_process(callback_a)
        registration_b = viewer.register_post_process(callback_b)
        viewer.register_post_process(callback_c)

        with mock.patch.object(RendererGL, "gl", fake_gl):
            final_target = viewer._render_post_processes()

        self.assertEqual(events, [("a", 1, 4), ("c", 4, 1)])
        self.assertEqual(final_target, (1, 2, 3))

    def test_callback_exception_stops_dispatch(self):
        viewer = _make_viewer()
        fake_gl = _FakeGL()
        events = []

        viewer.register_post_process(lambda context: events.append("a"))

        def fail(context):
            events.append("b")
            raise RuntimeError("callback failed")

        viewer.register_post_process(fail)
        viewer.register_post_process(lambda context: events.append("c"))

        with mock.patch.object(RendererGL, "gl", fake_gl), self.assertRaisesRegex(RuntimeError, "callback failed"):
            viewer._render_post_processes()

        self.assertEqual(events, ["a", "b"])


class TestRendererGLPostProcessLifecycle(unittest.TestCase):
    def test_close_runs_all_callbacks_before_window_close_and_is_idempotent(self):
        renderer = RendererGL.__new__(RendererGL)
        renderer._closed = False
        renderer._close_callbacks = []
        renderer.headless = True
        events = []
        renderer._make_current = lambda: events.append("current")
        renderer.window = SimpleNamespace(close=lambda: events.append("window"))

        renderer.register_close(lambda: events.append("a"))

        def cleanup_b():
            events.append("b")
            raise RuntimeError("cleanup failed")

        renderer.register_close(cleanup_b)

        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            renderer.close()
        renderer.close()

        self.assertEqual(events, ["current", "b", "a", "window"])

    def test_close_still_closes_window_when_event_shutdown_fails(self):
        renderer = RendererGL.__new__(RendererGL)
        renderer._closed = False
        renderer._close_callbacks = []
        renderer.headless = False
        events = []
        renderer._make_current = lambda: events.append("current")

        def fail_on_exit(event):
            events.append(event)
            raise RuntimeError("event shutdown failed")

        renderer.app = SimpleNamespace(
            event_loop=SimpleNamespace(dispatch_event=fail_on_exit),
            platform_event_loop=SimpleNamespace(stop=lambda: events.append("stop")),
        )
        renderer.window = SimpleNamespace(close=lambda: events.append("window"))

        with self.assertRaisesRegex(RuntimeError, "event shutdown failed"):
            renderer.close()

        self.assertEqual(events, ["current", "on_exit", "stop", "window"])

    def test_resize_rebuilds_all_existing_frame_targets(self):
        renderer = RendererGL.__new__(RendererGL)
        renderer.window = SimpleNamespace(get_framebuffer_size=lambda: (800, 450))
        renderer._setup_frame_buffer = mock.Mock()

        renderer.resize(10, 20)

        self.assertEqual((renderer._screen_width, renderer._screen_height), (800, 450))
        renderer._setup_frame_buffer.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
