# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton._src.viewer.gl.opengl import RendererGL
from newton._src.viewer.viewer_gl import ViewerGL


def _make_box_model(device: str | wp.Device):
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25, color=(1.0, 0.0, 0.0))
    return builder.finalize(device=device)


class _FakeGL:
    GL_PIXEL_PACK_BUFFER = 0x88EB
    GL_STREAM_READ = 0x88E1
    GL_PACK_ALIGNMENT = 0x0D05
    GL_FRAMEBUFFER = 0x8D40
    GL_RGB = 0x1907
    GL_UNSIGNED_BYTE = 0x1401

    GLuint = ctypes.c_uint
    GLsizeiptr = ctypes.c_size_t

    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels
        self.bound_buffer = 0
        self.readback_count = 0

    def glGenBuffers(self, count, buffers):
        buffers[0] = 17

    def glBindBuffer(self, target, buffer):
        self.bound_buffer = int(buffer)

    def glBufferData(self, target, size, data, usage):
        pass

    def glPixelStorei(self, name, value):
        pass

    def glBindFramebuffer(self, target, framebuffer):
        pass

    def glReadPixels(self, x, y, width, height, pixel_format, pixel_type, data):
        pass

    def glGetBufferSubData(self, target, offset, size, data):
        if self.bound_buffer == 0:
            raise RuntimeError("pixel buffer must be bound during readback")
        ctypes.memmove(data, self.pixels.ctypes.data, self.pixels.nbytes)
        self.readback_count += 1


class TestViewerGLGetFrame(unittest.TestCase):
    def test_headless_frame_capture_across_devices(self):
        cuda_devices = wp.get_cuda_devices()
        if cuda_devices:
            wp.zeros(1, dtype=wp.float32, device=cuda_devices[0])

        try:
            viewer = newton.viewer.ViewerGL(width=64, height=48, headless=True)
        except Exception as exc:
            self.skipTest(f"ViewerGL not available: {exc}")
            return

        try:
            cpu_device = wp.get_device("cpu")
            cpu_model = _make_box_model(cpu_device)
            viewer.set_model(cpu_model)
            self.assertEqual(viewer.device, cpu_device)

            viewer.set_camera(pos=wp.vec3(2.0, -3.0, 2.0), pitch=-25.0, yaw=35.0)
            viewer.begin_frame(0.0)
            viewer.log_state(cpu_model.state())
            viewer.end_frame()

            frame = viewer.get_frame()
            self.assertEqual(frame.shape, (48, 64, 3))
            self.assertEqual(frame.dtype, wp.uint8)
            self.assertEqual(frame.device, cpu_device)
            self.assertGreater(np.ptp(frame.numpy()), 0)

            target = wp.empty(shape=(48, 64, 3), dtype=wp.uint8, device=cpu_device)
            self.assertIs(viewer.get_frame(target_image=target), target)

            viewer._invalidate_pbo()
            self.assertEqual(viewer.get_frame().shape, (48, 64, 3))

            for cuda_device in cuda_devices[:2]:
                viewer.set_model(_make_box_model(cuda_device))
                self.assertEqual(viewer.device, cuda_device)

                # Capture the existing framebuffer to isolate PBO rebinding
                # from model-geometry updates.
                cuda_frame = viewer.get_frame()
                self.assertEqual(cuda_frame.shape, (48, 64, 3))
                self.assertEqual(cuda_frame.dtype, wp.uint8)
                self.assertEqual(cuda_frame.device, cuda_device)
                self.assertGreater(np.ptp(cuda_frame.numpy()), 0)
        finally:
            viewer.close()

    def test_cpu_viewer_uses_host_pbo_readback(self):
        pixels = np.array(
            [
                [10, 11, 12],
                [20, 21, 22],
                [30, 31, 32],
                [40, 41, 42],
            ],
            dtype=np.uint8,
        ).reshape(-1)
        fake_gl = _FakeGL(pixels)
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.device = wp.get_device("cpu")
        viewer.renderer = SimpleNamespace(
            _screen_width=2,
            _screen_height=2,
            _frame_fbo=3,
        )
        viewer.gui = None
        viewer._pbo = None
        viewer._wp_pbo = None
        viewer._pbo_host_buffer = None

        with (
            mock.patch.object(RendererGL, "gl", fake_gl),
            mock.patch.object(
                wp,
                "RegisteredGLBuffer",
                side_effect=AssertionError("CPU readback must not use CUDA-GL interop"),
            ),
        ):
            frame = viewer.get_frame()

        np.testing.assert_array_equal(
            frame.numpy(),
            np.array(
                [
                    [[30, 31, 32], [40, 41, 42]],
                    [[10, 11, 12], [20, 21, 22]],
                ],
                dtype=np.uint8,
            ),
        )
        self.assertEqual(frame.device, wp.get_device("cpu"))
        self.assertEqual(fake_gl.readback_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
