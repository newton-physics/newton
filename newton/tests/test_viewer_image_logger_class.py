# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the ImageLogger class with stubbed pyglet.gl.

These tests cover orchestration, state management, and cleanup paths that
the per-helper tests in ``test_viewer_image_logger.py`` cannot reach
without instantiating ``ImageLogger`` itself.
"""

import sys
import types
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp


class _GLuintArray(list):
    """Stand-in for a ctypes ``GLuint * n`` array: integer-indexable."""

    def __init__(self, size: int, *initial):
        data = list(initial) + [0] * (size - len(initial))
        super().__init__(data)


class _GLuintHolder:
    """Stand-in for a ctypes ``GLuint()`` holder with a ``.value`` slot."""

    def __init__(self, v: int = 0):
        self.value = v


class _GLTypeFactory:
    """Supports both ``gl.GLuint()`` (holder) and ``gl.GLuint * n`` (array type)."""

    def __call__(self, *args) -> _GLuintHolder:
        return _GLuintHolder(args[0] if args else 0)

    def __mul__(self, n: int):
        def ctor(*values):
            return _GLuintArray(n, *values)

        return ctor


class _FakeGL:
    """Minimal fake of ``pyglet.gl`` sufficient for ImageLogger upload paths.

    Records every GL call so tests can assert on them, fabricates unique
    texture/buffer ids, and lets tests drive failures by patching specific
    functions.
    """

    GL_TEXTURE_2D = 0x0DE1
    GL_RGBA = 0x1908
    GL_RGBA8 = 0x8058
    GL_UNSIGNED_BYTE = 0x1401
    GL_TEXTURE_MIN_FILTER = 0x2801
    GL_TEXTURE_MAG_FILTER = 0x2800
    GL_TEXTURE_WRAP_S = 0x2802
    GL_TEXTURE_WRAP_T = 0x2803
    GL_LINEAR = 0x2601
    GL_CLAMP_TO_EDGE = 0x812F
    GL_PIXEL_UNPACK_BUFFER = 0x88EC
    GL_STREAM_DRAW = 0x88E0
    GL_MAX_TEXTURE_SIZE = 0x0D33

    def __init__(self, max_texture_size: int = 16384):
        self.max_texture_size = max_texture_size
        self.next_id = 100
        self.calls: list[tuple[str, tuple]] = []
        self.deleted_textures: list[int] = []
        self.deleted_buffers: list[int] = []

        self.GLuint = _GLTypeFactory()
        self.GLint = _GLTypeFactory()
        self.GLsizeiptr = int

    # --- GL entry points ---

    def glGetIntegerv(self, pname, out):
        self.calls.append(("glGetIntegerv", (pname,)))
        out[0] = self.max_texture_size

    def glGenTextures(self, n, out):
        for i in range(n):
            out[i] = self.next_id
            self.next_id += 1
        self.calls.append(("glGenTextures", (n,)))

    def glGenBuffers(self, n, out):
        # Caller passes a `GLuint()` single-slot holder: set `.value`.
        out.value = self.next_id
        self.next_id += 1
        self.calls.append(("glGenBuffers", (n,)))

    def glBindTexture(self, target, tex_id):
        self.calls.append(("glBindTexture", (target, tex_id)))

    def glTexParameteri(self, *args):
        self.calls.append(("glTexParameteri", args))

    def glTexImage2D(self, *args):
        self.calls.append(("glTexImage2D", args))

    def glTexSubImage2D(self, *args):
        self.calls.append(("glTexSubImage2D", args))

    def glBindBuffer(self, target, buf_id):
        self.calls.append(("glBindBuffer", (target, buf_id)))

    def glBufferData(self, *args):
        self.calls.append(("glBufferData", args))

    def glDeleteTextures(self, n, arr):
        self.deleted_textures.append(int(arr[0]))
        self.calls.append(("glDeleteTextures", (n,)))

    def glDeleteBuffers(self, n, arr):
        self.deleted_buffers.append(int(arr[0]))
        self.calls.append(("glDeleteBuffers", (n,)))


def _install_fake_pyglet(fake_gl: _FakeGL) -> None:
    """Make ``from pyglet import gl`` resolve to *fake_gl* for this test."""
    fake_pyglet = types.ModuleType("pyglet")
    fake_pyglet.gl = fake_gl  # type: ignore[attr-defined]
    sys.modules["pyglet"] = fake_pyglet
    sys.modules["pyglet.gl"] = fake_gl  # type: ignore[assignment]


def _uninstall_fake_pyglet() -> None:
    sys.modules.pop("pyglet", None)
    sys.modules.pop("pyglet.gl", None)


class _ImageLoggerFixture(unittest.TestCase):
    """Sets up a fake GL context and builds an ImageLogger on the CPU device."""

    def setUp(self):
        self.fake_gl = _FakeGL()
        _install_fake_pyglet(self.fake_gl)

        from newton._src.viewer.gl.image_logger import ImageLogger  # noqa: PLC0415

        self.ImageLogger = ImageLogger
        self.cpu_device = wp.get_device("cpu")
        self.logger = ImageLogger(device=self.cpu_device)

    def tearDown(self):
        # Avoid leaking the fake module into other tests.
        _uninstall_fake_pyglet()


class TestImageLoggerOrchestration(_ImageLoggerFixture):
    def test_first_log_creates_entry_and_auto_selects(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        self.logger.log("cam0", img)

        self.assertIn("cam0", self.logger._images)
        self.assertEqual(self.logger._selected, "cam0")
        entry = self.logger._images["cam0"]
        self.assertEqual((entry.n, entry.h, entry.w, entry.c), (1, 8, 8, 1))
        self.assertNotEqual(entry.tex_id, 0)

    def test_subsequent_new_name_does_not_switch_selection(self):
        self.logger.log("cam0", np.zeros((8, 8), dtype=np.uint8))
        self.logger.log("cam1", np.zeros((8, 8), dtype=np.uint8))
        self.assertEqual(self.logger._selected, "cam0")
        self.assertEqual(set(self.logger._images.keys()), {"cam0", "cam1"})

    def test_repeated_same_name_updates_in_place(self):
        self.logger.log("cam0", np.zeros((8, 8), dtype=np.uint8))
        tex_id_1 = self.logger._images["cam0"].tex_id
        self.logger.log("cam0", np.ones((8, 8), dtype=np.uint8))
        tex_id_2 = self.logger._images["cam0"].tex_id
        self.assertEqual(tex_id_1, tex_id_2)

    def test_shape_change_triggers_reallocation(self):
        self.logger.log("cam0", np.zeros((8, 8), dtype=np.uint8))
        # Second call with different shape -> _ensure_texture re-runs glTexImage2D.
        tex_image_2d_before = sum(1 for c in self.fake_gl.calls if c[0] == "glTexImage2D")
        self.logger.log("cam0", np.zeros((16, 16), dtype=np.uint8))
        tex_image_2d_after = sum(1 for c in self.fake_gl.calls if c[0] == "glTexImage2D")
        self.assertGreater(tex_image_2d_after, tex_image_2d_before)
        entry = self.logger._images["cam0"]
        self.assertEqual((entry.h, entry.w), (16, 16))

    def test_upload_failure_rolls_back_metadata(self):
        self.logger.log("cam0", np.zeros((8, 8), dtype=np.uint8))
        entry = self.logger._images["cam0"]
        pre_shape = (entry.n, entry.h, entry.w, entry.c)

        # Force the next glTexSubImage2D (inside _upload_cpu) to blow up.
        original = self.fake_gl.glTexSubImage2D

        def _boom(*args):
            raise RuntimeError("simulated GL failure")

        self.fake_gl.glTexSubImage2D = _boom
        try:
            with self.assertRaises(RuntimeError):
                self.logger.log("cam0", np.zeros((32, 32), dtype=np.uint8))
        finally:
            self.fake_gl.glTexSubImage2D = original

        # Metadata (n, h, w, c) must not reflect the failed 32x32 upload.
        entry = self.logger._images["cam0"]
        self.assertEqual((entry.n, entry.h, entry.w, entry.c), pre_shape)

    def test_clear_releases_all_entries_and_is_idempotent(self):
        self.logger.log("cam0", np.zeros((8, 8), dtype=np.uint8))
        self.logger.log("cam1", np.zeros((8, 8), dtype=np.uint8))
        self.assertEqual(len(self.logger._images), 2)

        self.logger.clear()
        self.assertEqual(len(self.logger._images), 0)
        self.assertIsNone(self.logger._selected)
        self.assertEqual(len(self.fake_gl.deleted_textures), 2)

        # Second clear must not raise.
        self.logger.clear()


class TestImageLoggerDeviceWarnings(_ImageLoggerFixture):
    """_maybe_warn_cross_device dedupes per (name, device) and re-warns on change."""

    def _make_logger_on_fake_cuda(self, device_name: str = "cuda:0"):
        """Construct an ImageLogger whose _device pretends to be CUDA.

        We don't need a real CUDA context — only the ``is_cuda`` flag and
        identity comparison against incoming arrays' devices.
        """
        fake_device = mock.MagicMock()
        fake_device.is_cuda = True
        fake_device.__str__ = lambda self: device_name  # type: ignore[misc]
        fake_device.__eq__ = lambda self, other: other is fake_device  # type: ignore[misc]
        fake_device.__hash__ = lambda self: id(fake_device)  # type: ignore[misc]
        return self.ImageLogger(device=fake_device), fake_device

    def _fake_cuda_array(self, device_name: str):
        """Build a stand-in wp.array whose .device pretends to be CUDA."""
        arr = mock.MagicMock(spec=wp.array)
        dev = mock.MagicMock()
        dev.is_cuda = True
        dev.__str__ = lambda self: device_name  # type: ignore[misc]
        arr.device = dev
        return arr, dev

    def test_warns_once_per_same_device(self):
        logger, _ = self._make_logger_on_fake_cuda()
        arr, _ = self._fake_cuda_array("cuda:1")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logger._maybe_warn_cross_device("img", arr)
            logger._maybe_warn_cross_device("img", arr)
        cross_dev_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(cross_dev_warnings), 1)

    def test_rewarns_when_device_changes(self):
        logger, _ = self._make_logger_on_fake_cuda()
        arr1, _ = self._fake_cuda_array("cuda:1")
        arr2, _ = self._fake_cuda_array("cuda:2")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logger._maybe_warn_cross_device("img", arr1)
            logger._maybe_warn_cross_device("img", arr2)
        cross_dev_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(cross_dev_warnings), 2)


class TestImageLoggerCleanup(_ImageLoggerFixture):
    def test_free_entry_warns_on_gl_failure(self):
        self.logger.log("cam0", np.zeros((8, 8), dtype=np.uint8))

        def _boom(*args):
            raise RuntimeError("simulated driver error")

        self.fake_gl.glDeleteTextures = _boom
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.logger.clear()
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("GL cleanup failed" in m for m in messages), messages)


@unittest.skipUnless(wp.is_cuda_available(), "GPU-path test requires CUDA")
class TestImageLoggerEnsurePboRollback(unittest.TestCase):
    """If wp.RegisteredGLBuffer construction raises, _ensure_pbo must roll
    back the GL buffer id so the next upload attempt starts from a clean slate.
    """

    def setUp(self):
        self.fake_gl = _FakeGL()
        _install_fake_pyglet(self.fake_gl)

        from newton._src.viewer.gl.image_logger import ImageLogger, LoggedImage  # noqa: PLC0415

        self.ImageLogger = ImageLogger
        self.LoggedImage = LoggedImage
        self.cuda_device = wp.get_device("cuda:0")
        self.logger = ImageLogger(device=self.cuda_device)

    def tearDown(self):
        _uninstall_fake_pyglet()

    def test_registered_gl_buffer_failure_rolls_back_pbo(self):
        entry = self.LoggedImage(name="cam0")

        with mock.patch.object(wp, "RegisteredGLBuffer", side_effect=RuntimeError("CUDA-GL interop unavailable")):
            with self.assertRaises(RuntimeError):
                self.logger._ensure_pbo(entry, byte_size=1024, realloc=True)

        # Rollback contract: pbo_id cleared, buffer deleted, next call starts fresh.
        self.assertIsNone(entry.pbo_id)
        self.assertIsNone(entry.wp_pbo)
        self.assertEqual(len(self.fake_gl.deleted_buffers), 1)


class TestViewerGLClearModelClearsImageLogger(unittest.TestCase):
    """Regression test for issue #2731: example-browser switching must not
    leave stale image-logger entries behind.

    Two symptoms originally observed in the GUI:
      1) The image window opened by the previous example stayed visible after
         switching to one that does not call ``log_image``.
      2) Closing that window manually and re-entering the camera example did
         not re-open it (because logger entries from the prior run survived,
         so the auto-select branch in ``ImageLogger.log`` was skipped).
    Both stem from ``ViewerGL.clear_model`` not clearing the image logger.
    """

    def test_clear_model_clears_image_logger(self):
        try:
            import newton.viewer  # noqa: PLC0415

            viewer = newton.viewer.ViewerGL(headless=True)
        except Exception as exc:
            self.skipTest(f"ViewerGL not available: {exc}")
            return

        try:
            viewer.log_image("color", np.zeros((4, 4), dtype=np.uint8))
            logger = viewer._image_logger
            self.assertIn("color", logger._images)
            self.assertEqual(logger._selected, "color")

            # Simulate user closing the image window (the X button maps to
            # this internally in ImageLogger.draw).
            logger._selected = None

            viewer.clear_model()

            self.assertEqual(logger._images, {})
            self.assertIsNone(logger._selected)

            # Re-logging the same name after clear_model must auto-select
            # again so the window re-opens, matching the behavior on first
            # entry into a camera example.
            viewer.log_image("color", np.zeros((4, 4), dtype=np.uint8))
            self.assertEqual(logger._selected, "color")
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
