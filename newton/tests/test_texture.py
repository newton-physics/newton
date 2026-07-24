# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for texture loading, including assets packaged inside USD (.usdz) archives."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton._src.utils.texture import load_texture
from newton.tests.unittest_utils import USD_AVAILABLE

_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    """Write a small solid-color RGB PNG to *path*."""
    from PIL import Image

    Image.fromarray(np.full((4, 4, 3), color, dtype=np.uint8), "RGB").save(str(path))


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


if __name__ == "__main__":
    unittest.main()
