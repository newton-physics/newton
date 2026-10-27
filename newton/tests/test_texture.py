# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for texture loading, assets packaged inside USD (.usdz) archives, and linear-to-sRGB conversion."""

import importlib.util
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

from newton._src.utils.texture import linear_texture_to_srgb, load_texture
from newton.tests.unittest_utils import USD_AVAILABLE

_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


class TestLinearTextureToSrgb(unittest.TestCase):
    def test_uint8_lut_matches_reference_bit_exactly(self):
        """Match the reference conversion bit-for-bit for every uint8 value."""
        values = np.arange(256, dtype=np.uint8)
        image = np.empty((1, 256, 4), dtype=np.uint8)
        # Offset each channel so a wrong-channel LUT index is detectable; each still spans all 256 values.
        for channel in range(3):
            image[0, :, channel] = np.roll(values, channel * 64)
        image[..., 3] = values[::-1]
        original = image.copy()

        converted = linear_texture_to_srgb(image)

        expected = original.copy()
        linear_rgb = np.clip(original[..., :3].astype(np.float32) / 255.0, 0.0, 1.0)
        expected_rgb = np.where(
            linear_rgb <= 0.0031308,
            linear_rgb * 12.92,
            1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055,
        )
        expected[..., :3] = np.clip(np.round(expected_rgb * 255.0), 0.0, 255.0).astype(np.uint8)
        np.testing.assert_array_equal(converted, expected)
        np.testing.assert_array_equal(image, original)
        self.assertTrue(converted.flags.c_contiguous)
        # The shared LUT is read-only; the result must not inherit that. "W" is NumPy's short key for the write flag.
        self.assertTrue(converted.flags["W"])


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    """Write a small solid-color RGB PNG to *path*."""
    from PIL import Image

    Image.fromarray(np.full((4, 4, 3), color, dtype=np.uint8)).save(str(path))


def _build_usdz_with_texture(tmpdir: str, color: tuple[int, int, int]) -> Path:
    """Package a texture into a .usdz and return the archive path."""
    from pxr import Sdf, Usd, UsdShade, UsdUtils

    tex_path = Path(tmpdir) / "tex.png"
    _write_png(tex_path, color)

    stage_path = Path(tmpdir) / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    shader = UsdShade.Shader.Define(stage, "/Looks/Material/Texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./tex.png")
    stage.GetRootLayer().Save()

    usdz_path = Path(tmpdir) / "scene.usdz"
    UsdUtils.CreateNewUsdzPackage(str(stage_path), str(usdz_path))
    return usdz_path


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
@unittest.skipUnless(_PIL_AVAILABLE, "Requires Pillow")
class TestPackagedTextureLoading(unittest.TestCase):
    def test_load_texture_from_usdz_package(self):
        """Load a texture addressed with USD package-relative syntax (``scene.usdz[tex.png]``).

        Regression test: package-relative paths are not valid filesystem paths, so
        the loader must resolve them through USD's asset resolver rather than
        handing them straight to Pillow.
        """
        color = (10, 20, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            usdz_path = _build_usdz_with_texture(tmpdir, color)
            packaged = f"{usdz_path}[tex.png]"

            image = load_texture(packaged)

            self.assertIsNotNone(image, "packaged texture should load")
            self.assertEqual(image.shape, (4, 4, 4))
            self.assertEqual(tuple(int(c) for c in image[0, 0]), (*color, 255))

    def test_load_texture_missing_package_member_returns_none(self):
        """Return ``None`` (not raise) when the named member is absent from the archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            usdz_path = _build_usdz_with_texture(tmpdir, (1, 2, 3))
            with self.assertWarns(UserWarning):
                self.assertIsNone(load_texture(f"{usdz_path}[does_not_exist.png]"))


@unittest.skipUnless(_PIL_AVAILABLE, "Requires Pillow")
class TestMeshGLTextureCaching(unittest.TestCase):
    def setUp(self):
        from newton._src.viewer.gl import opengl  # noqa: PLC0415

        stack = ExitStack()
        self.addCleanup(stack.close)
        self.mesh = opengl.MeshGL.__new__(opengl.MeshGL)
        self.mesh.texture_id = None
        self.mesh._texture_file_signature = None
        self.gl = mock.Mock()
        stack.enter_context(mock.patch.object(opengl.RendererGL, "gl", self.gl))
        self.upload = stack.enter_context(mock.patch.object(opengl, "_upload_texture_from_file", return_value=1))
        self.load = stack.enter_context(mock.patch("newton._src.utils.texture.load_texture", wraps=load_texture))

    def test_unchanged_file_reuses_texture_until_file_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "texture.png"
            _write_png(path, (10, 20, 30))
            self.mesh.update_texture(str(path))
            self.mesh.update_texture(path)
            self.assertEqual(self.load.call_count, 1)
            self.assertEqual(self.upload.call_count, 1)
            self.gl.glDeleteTextures.assert_not_called()

            previous_mtime = path.stat().st_mtime_ns
            _write_png(path, (30, 20, 10))
            os.utime(path, ns=(previous_mtime + 1_000_000_000, previous_mtime + 1_000_000_000))
            self.mesh.update_texture(path)
            self.assertEqual(self.upload.call_count, 2)
            self.gl.glDeleteTextures.assert_called_once_with(1, 1)
            np.testing.assert_array_equal(self.upload.call_args.args[1][0, 0, :3], [30, 20, 10])

            other = Path(tmpdir) / "other.png"
            _write_png(other, (40, 50, 60))
            self.mesh.update_texture(other)
            self.assertEqual(self.upload.call_count, 3)

            self.mesh.update_texture(None)
            self.assertIsNone(self.mesh.texture_id)
            self.mesh.update_texture(other)
            self.assertEqual(self.upload.call_count, 4)

    def test_array_updates_are_not_cached(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        self.mesh.update_texture(image)
        image[:] = 255
        self.mesh.update_texture(image)
        self.assertEqual(self.upload.call_count, 2)
        np.testing.assert_array_equal(self.upload.call_args.args[1], image)

    def test_failed_upload_is_retried(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "texture.png"
            _write_png(path, (10, 20, 30))
            self.upload.side_effect = [None, 1]
            self.mesh.update_texture(path)
            self.assertIsNone(self.mesh.texture_id)
            self.mesh.update_texture(path)
            self.assertEqual(self.mesh.texture_id, 1)
            self.assertEqual(self.upload.call_count, 2)


if __name__ == "__main__":
    unittest.main()
