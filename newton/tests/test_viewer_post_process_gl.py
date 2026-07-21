# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import unittest

import numpy as np
import warp as wp

from newton._src.viewer.viewer_gl import ViewerGL


class TestViewerGLPostProcessGL(unittest.TestCase):
    def test_headless_color_depth_ping_pong_and_frame_capture(self):
        try:
            viewer = ViewerGL(width=32, height=24, headless=True)
        except Exception as error:
            self.skipTest(f"ViewerGL not available: {error}")
            return

        cleanup_events = []
        try:
            from pyglet import gl

            viewer.device = wp.get_device("cpu")
            sampled_depth = []

            def clear_output(red: float, green: float, blue: float, depth: float) -> None:
                gl.glDisable(gl.GL_SCISSOR_TEST)
                gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glClearColor(red, green, blue, 1.0)
                gl.glClearDepth(depth)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            def pass_a(context):
                clear_output(1.0, 0.0, 0.0, 0.25)

            viewer.register_post_process(pass_a, cleanup=lambda: cleanup_events.append("a"))
            viewer.begin_frame(0.0)
            viewer.end_frame()
            frame = viewer.get_frame().numpy()
            expected = np.broadcast_to(np.array([255, 0, 0], dtype=np.uint8), frame.shape)
            np.testing.assert_allclose(frame, expected, atol=1)

            def pass_b(context):
                depth = (gl.GLfloat * 1)()
                gl.glReadPixels(0, 0, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, ctypes.byref(depth))
                sampled_depth.append(float(depth[0]))
                clear_output(0.0, 1.0, 0.0, 0.75)

            viewer.register_post_process(pass_b, cleanup=lambda: cleanup_events.append("b"))
            viewer.begin_frame(1.0)
            viewer.end_frame()
            frame = viewer.get_frame().numpy()

            self.assertAlmostEqual(sampled_depth[0], 0.25, places=4)
            expected = np.broadcast_to(np.array([0, 255, 0], dtype=np.uint8), frame.shape)
            np.testing.assert_allclose(frame, expected, atol=1)
        finally:
            viewer.close()
        self.assertEqual(cleanup_events, ["b", "a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
