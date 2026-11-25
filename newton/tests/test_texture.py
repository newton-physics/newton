# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for texture loading, assets packaged inside USD (.usdz) archives, and linear-to-sRGB conversion."""

import concurrent.futures
import contextlib
import sys
import tempfile
import threading
import unittest
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

import newton._src.utils.texture as texture_utils
from newton import NewtonWarning
from newton._src.utils.texture import _load_textures, linear_texture_to_srgb, load_texture
from newton.tests.unittest_utils import patch_sys_module, requires


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


@requires("pxr.Usd", "PIL")
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


def _save_noise_png(path: Path, width: int, height: int) -> np.ndarray:
    """Save an RGBA image of noise as a PNG file and return its pixels."""
    from PIL import Image

    pixels = np.random.default_rng(width * 1000 + height).integers(0, 256, (height, width, 4), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return pixels


def _count_texture_threads() -> int:
    """Count the worker threads of ``_load_textures()`` that are alive."""
    return sum(thread.name.startswith("newton-texture") for thread in threading.enumerate())


@contextlib.contextmanager
def _record_worker_submissions(cpu_count: int, *, workers_keep_up: bool) -> Iterator[list]:
    """Run ``_load_textures()`` as on *cpu_count* CPUs, and yield the list of the images it submits to workers.

    How far ahead the loader decodes depends on whether it finds a file unfinished when it reaches
    it. The pool settles that, rather than leaving it to timing:

    - with *workers_keep_up*, ``submit()`` returns once the file is decoded, so the loader never waits;
    - without, the first file submitted is held back until the second ``submit()``, which only comes
      after the loader has found that file unfinished, so the loader waits for it.
    """
    submitted = []
    second_submit = threading.Event()
    convert_to_rgba = texture_utils._convert_to_rgba

    def convert_first_late(image):
        if image is submitted[0]:
            second_submit.wait(timeout=30.0)
        return convert_to_rgba(image)

    class RecordingPool(ThreadPoolExecutor):
        def submit(self, fn, image):
            submitted.append(image)
            if len(submitted) == 2:
                second_submit.set()
            future = super().submit(fn, image)
            if workers_keep_up:
                concurrent.futures.wait((future,))
            return future

    with (
        mock.patch.object(texture_utils, "_count_usable_cpus", return_value=cpu_count),
        mock.patch.object(texture_utils, "ThreadPoolExecutor", RecordingPool),
        mock.patch.object(
            texture_utils, "_convert_to_rgba", convert_to_rgba if workers_keep_up else convert_first_late
        ),
    ):
        yield submitted


@requires("PIL")
class TestLoadTextures(unittest.TestCase):
    """Tests for ``_load_textures()``, which decodes image files ahead on worker threads."""

    def test_matches_load_texture(self):
        """Yield, in order, what ``load_texture()`` returns for files of each kind and for arrays."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            rng = np.random.default_rng(7)
            rgba = str(directory / "rgba.png")
            rgba_pixels = _save_noise_png(Path(rgba), 40, 24)
            jpeg = str(directory / "rgb.jpg")
            Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)).save(jpeg)
            gray = str(directory / "gray.png")
            gray_pixels = rng.integers(0, 256, (8, 16), dtype=np.uint8)
            Image.fromarray(gray_pixels).save(gray)
            palette = str(directory / "palette.png")
            palette_image = Image.fromarray(np.arange(240, dtype=np.uint8).reshape(20, 12) % 3)
            palette_image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255])
            palette_image.save(palette)
            array = rng.integers(0, 256, (3, 5, 3), dtype=np.uint8)
            path_like = directory / "path_like.png"
            path_like_pixels = _save_noise_png(path_like, 8, 10)
            textures = (rgba, jpeg, gray, palette, array, None, path_like)

            # The thread that converts each file, and what reading its source image raises right after.
            convert_to_rgba = texture_utils._convert_to_rgba
            thread_names = []
            source_errors = []

            def convert_and_record(image):
                converted = convert_to_rgba(image)
                thread_names.append(threading.current_thread().name)
                try:
                    image.getpixel((0, 0))
                except ValueError as error:
                    source_errors.append(str(error))
                else:
                    source_errors.append("")
                return converted

            with (
                mock.patch.object(texture_utils, "_count_usable_cpus", return_value=4),
                mock.patch.object(texture_utils, "_convert_to_rgba", convert_and_record),
            ):
                loaded = list(_load_textures(textures))

            self.assertEqual(len(loaded), 7)
            # Every file but the first texture, which the calling thread loads itself, decodes on a worker.
            self.assertEqual(len(thread_names), 4)
            self.assertRegex(thread_names[0], "^newton-texture")
            self.assertRegex(thread_names[1], "^newton-texture")
            self.assertRegex(thread_names[2], "^newton-texture")
            self.assertRegex(thread_names[3], "^newton-texture")
            # Each worker releases the pixels of its source image, rather than leaving them to the calling thread.
            self.assertEqual(len(source_errors), 4)
            self.assertIn("closed image", source_errors[0])
            self.assertIn("closed image", source_errors[1])
            self.assertIn("closed image", source_errors[2])
            self.assertIn("closed image", source_errors[3])

            self.assertTrue(loaded[0].flags.c_contiguous)
            np.testing.assert_array_equal(loaded[0], rgba_pixels)

            self.assertEqual(loaded[1].shape, (64, 96, 4))
            self.assertEqual(loaded[1].dtype, np.uint8)
            self.assertTrue(loaded[1].flags.c_contiguous)
            np.testing.assert_array_equal(loaded[1], load_texture(jpeg))
            np.testing.assert_array_equal(loaded[1][..., 3], 255)

            self.assertEqual(loaded[2].shape, (8, 16, 4))
            np.testing.assert_array_equal(loaded[2][..., 0], gray_pixels)
            np.testing.assert_array_equal(loaded[2][..., 1], gray_pixels)
            np.testing.assert_array_equal(loaded[2][..., 2], gray_pixels)
            np.testing.assert_array_equal(loaded[2][..., 3], 255)

            self.assertEqual(loaded[3].shape, (20, 12, 4))
            np.testing.assert_array_equal(loaded[3][0, 0], (255, 0, 0, 255))
            np.testing.assert_array_equal(loaded[3][0, 1], (0, 255, 0, 255))
            np.testing.assert_array_equal(loaded[3][0, 2], (0, 0, 255, 255))
            np.testing.assert_array_equal(loaded[3], load_texture(palette))

            np.testing.assert_array_equal(loaded[4], array)
            self.assertIsNone(loaded[5])
            np.testing.assert_array_equal(loaded[6], path_like_pixels)

    @unittest.skipIf(sys.platform == "win32", "load_texture() can't open a file:// URL that has a drive letter")
    def test_loads_file_urls(self):
        """Decode the file that a ``file://`` URL names on a worker, like any other file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            first = directory / "first.png"
            first_pixels = _save_noise_png(first, 10, 6)
            # A name that its URL quotes.
            second = directory / "second file.png"
            second_pixels = _save_noise_png(second, 6, 10)

            with (
                mock.patch.object(texture_utils, "_count_usable_cpus", return_value=4),
                mock.patch.object(
                    texture_utils, "_convert_to_rgba", wraps=texture_utils._convert_to_rgba
                ) as convert_to_rgba,
            ):
                loaded = list(_load_textures((first.as_uri(), second.as_uri())))

            self.assertEqual(len(loaded), 2)
            # The calling thread loads the first texture itself. The second decodes on a worker only
            # if the loader resolved its URL to a path.
            self.assertEqual(len(convert_to_rgba.call_args_list), 1)
            np.testing.assert_array_equal(loaded[0], first_pixels)
            np.testing.assert_array_equal(loaded[1], second_pixels)

    def test_reports_failures_in_order(self):
        """Warn about a file that fails to load when its texture is reached, not when a worker decodes it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            good = directory / "good.png"
            good_pixels = _save_noise_png(good, 32, 32)
            missing = directory / "missing.png"
            truncated = directory / "truncated.png"
            truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
            other = directory / "other.png"
            other_pixels = _save_noise_png(other, 16, 16)

            # The truncated file opens, so a worker decodes it while the first two textures load.
            decoded = threading.Event()
            convert_to_rgba = texture_utils._convert_to_rgba

            def convert_and_notify(image):
                try:
                    return convert_to_rgba(image)
                finally:
                    decoded.set()

            with (
                mock.patch.object(texture_utils, "_count_usable_cpus", return_value=4),
                mock.patch.object(texture_utils, "_convert_to_rgba", convert_and_notify),
            ):
                loader = _load_textures((str(good), str(missing), str(truncated), str(other)))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    first = next(loader)
                    second = next(loader)
                    self.assertTrue(decoded.wait(timeout=30.0))
                np.testing.assert_array_equal(first, good_pixels)
                self.assertIsNone(second)
                messages = [str(w.message) for w in caught if issubclass(w.category, NewtonWarning)]
                self.assertEqual(len(messages), 1)
                self.assertIn(f"Failed to load texture image: {missing}", messages[0])

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    third = next(loader)
                self.assertIsNone(third)
                messages = [str(w.message) for w in caught if issubclass(w.category, NewtonWarning)]
                self.assertEqual(len(messages), 1)
                self.assertIn(f"Failed to load texture image: {truncated}", messages[0])

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fourth = next(loader)
                    with self.assertRaises(StopIteration):
                        next(loader)
                np.testing.assert_array_equal(fourth, other_pixels)
                messages = [str(w.message) for w in caught if issubclass(w.category, NewtonWarning)]
                self.assertEqual(len(messages), 0)

    def test_reports_unparsable_path_in_order(self):
        """Warn about a path that ``urlparse()`` rejects when its texture is reached, as ``load_texture()`` does."""
        # Brackets in the host part of a path make urlparse() raise.
        unparsable = "//[x"
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            first = directory / "first.png"
            first_pixels = _save_noise_png(first, 8, 4)
            last = directory / "last.png"
            last_pixels = _save_noise_png(last, 4, 8)

            with (
                mock.patch.object(texture_utils, "_count_usable_cpus", return_value=4),
                contextlib.closing(_load_textures((str(first), unparsable, str(last)))) as loader,
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loaded_first = next(loader)
                np.testing.assert_array_equal(loaded_first, first_pixels)
                messages = [str(w.message) for w in caught if issubclass(w.category, NewtonWarning)]
                self.assertEqual(len(messages), 0)

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loaded_unparsable = next(loader)
                self.assertIsNone(loaded_unparsable)
                messages = [str(w.message) for w in caught if issubclass(w.category, NewtonWarning)]
                self.assertEqual(len(messages), 1)
                self.assertIn("Failed to load texture image: //[x", messages[0])

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loaded_last = next(loader)
                    with self.assertRaises(StopIteration):
                        next(loader)
                np.testing.assert_array_equal(loaded_last, last_pixels)
                messages = [str(w.message) for w in caught if issubclass(w.category, NewtonWarning)]
                self.assertEqual(len(messages), 0)

            # Alone, it takes the path without workers.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = list(_load_textures((unparsable,)))
            self.assertEqual(loaded, [None])
            messages = [str(w.message) for w in caught if issubclass(w.category, NewtonWarning)]
            self.assertEqual(len(messages), 1)
            self.assertIn("Failed to load texture image: //[x", messages[0])

    def test_uses_no_thread_without_spare_core_or_second_file(self):
        """Load on the calling thread when it has the only usable CPU, or when a single texture is a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            first = directory / "first.png"
            first_pixels = _save_noise_png(first, 8, 4)
            second = directory / "second.png"
            second_pixels = _save_noise_png(second, 4, 8)
            third = directory / "third.png"
            third_pixels = _save_noise_png(third, 6, 6)
            array = np.zeros((2, 2, 4), dtype=np.uint8)

            with (
                mock.patch.object(texture_utils, "_count_usable_cpus", return_value=1),
                mock.patch.object(texture_utils, "ThreadPoolExecutor", side_effect=AssertionError("pool created")),
            ):
                loaded = list(_load_textures((str(first), str(second), str(third))))
            self.assertEqual(len(loaded), 3)
            np.testing.assert_array_equal(loaded[0], first_pixels)
            np.testing.assert_array_equal(loaded[1], second_pixels)
            np.testing.assert_array_equal(loaded[2], third_pixels)

            with (
                mock.patch.object(texture_utils, "_count_usable_cpus", return_value=8),
                mock.patch.object(texture_utils, "ThreadPoolExecutor", side_effect=AssertionError("pool created")),
            ):
                loaded = list(_load_textures((array, str(first), array)))
            self.assertEqual(len(loaded), 3)
            np.testing.assert_array_equal(loaded[0], array)
            np.testing.assert_array_equal(loaded[1], first_pixels)
            np.testing.assert_array_equal(loaded[2], array)

    def test_decodes_further_ahead_only_after_waiting(self):
        """Decode one more file ahead each time the calling thread waits for a worker, and no more otherwise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index in range(6):
                path = Path(tmpdir) / f"{index}.png"
                _save_noise_png(path, 4 + index, 4)
                paths.append(str(path))

            # The workers keep up: one file ahead, although three workers are allowed.
            with (
                _record_worker_submissions(4, workers_keep_up=True) as submitted,
                contextlib.closing(_load_textures(paths)) as loader,
            ):
                next(loader)
                self.assertEqual(len(submitted), 1)
                next(loader)
                self.assertEqual(len(submitted), 2)
                next(loader)
                self.assertEqual(len(submitted), 3)
                next(loader)
                self.assertEqual(len(submitted), 4)
                next(loader)
                self.assertEqual(len(submitted), 5)
                next(loader)
                self.assertEqual(len(submitted), 5)

            # The calling thread waits for the second texture: from there, two files ahead.
            with (
                _record_worker_submissions(3, workers_keep_up=False) as submitted,
                contextlib.closing(_load_textures(paths)) as loader,
            ):
                next(loader)
                self.assertEqual(len(submitted), 1)
                next(loader)
                self.assertEqual(len(submitted), 3)
                next(loader)
                self.assertEqual(len(submitted), 4)
                next(loader)
                self.assertEqual(len(submitted), 5)
                next(loader)
                self.assertEqual(len(submitted), 5)
                next(loader)
                self.assertEqual(len(submitted), 5)

            # One wait brings a single file more, however many workers are allowed.
            with (
                _record_worker_submissions(8, workers_keep_up=False) as submitted,
                contextlib.closing(_load_textures(paths)) as loader,
            ):
                next(loader)
                self.assertEqual(len(submitted), 1)
                next(loader)
                self.assertEqual(len(submitted), 3)

    def test_lookahead_stays_within_workers_and_memory_cap(self):
        """Never decode further ahead than there are workers, nor past the cap on the pixels waiting.

        The calling thread waits for the second texture, so the lookahead would grow if it were allowed to.
        """
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index in range(6):
                path = Path(tmpdir) / f"{index}.png"
                _save_noise_png(path, 4 + index, 4)
                paths.append(str(path))

            # One worker: a single file ahead.
            with (
                _record_worker_submissions(2, workers_keep_up=False) as submitted,
                contextlib.closing(_load_textures(paths)) as loader,
            ):
                next(loader)
                self.assertEqual(len(submitted), 1)
                next(loader)
                self.assertEqual(len(submitted), 2)
                next(loader)
                self.assertEqual(len(submitted), 3)
                next(loader)
                self.assertEqual(len(submitted), 4)
                next(loader)
                self.assertEqual(len(submitted), 5)
                next(loader)
                self.assertEqual(len(submitted), 5)

            # A cap below one image: a single file ahead, whatever the number of workers.
            with (
                _record_worker_submissions(8, workers_keep_up=False) as submitted,
                mock.patch.object(texture_utils, "_MAX_PENDING_TEXTURE_BYTES", 1),
                contextlib.closing(_load_textures(paths)) as loader,
            ):
                next(loader)
                self.assertEqual(len(submitted), 1)
                next(loader)
                self.assertEqual(len(submitted), 2)
                next(loader)
                self.assertEqual(len(submitted), 3)
                next(loader)
                self.assertEqual(len(submitted), 4)
                next(loader)
                self.assertEqual(len(submitted), 5)
                next(loader)
                self.assertEqual(len(submitted), 5)

            # The files hold 64, 80, 96, 112, 128, and 144 bytes of pixels. A cap of 208 bytes: the third and
            # fourth files fit together, to the byte, and no later pair does. A file that doesn't fit with
            # those waiting is closed, and submitted once nothing waits.
            image_open = Image.open
            opened = []

            def open_and_record(*args, **kwargs):
                image = image_open(*args, **kwargs)
                opened.append((image, image.fp))
                return image

            with (
                _record_worker_submissions(3, workers_keep_up=False) as submitted,
                mock.patch.object(texture_utils, "_MAX_PENDING_TEXTURE_BYTES", 208),
                mock.patch.object(Image, "open", open_and_record),
                contextlib.closing(_load_textures(paths)) as loader,
            ):
                next(loader)
                self.assertEqual(len(submitted), 1)
                next(loader)
                self.assertEqual(len(submitted), 3)
                next(loader)
                self.assertEqual(len(submitted), 3)
                image, file = opened[-1]
                self.assertEqual(image.filename, paths[4])
                self.assertTrue(file.closed)
                next(loader)
                self.assertEqual(len(submitted), 4)
                self.assertEqual(submitted[3].filename, paths[4])
                image, file = opened[-1]
                self.assertEqual(image.filename, paths[5])
                self.assertTrue(file.closed)
                next(loader)
                self.assertEqual(len(submitted), 5)
                self.assertEqual(submitted[4].filename, paths[5])
                next(loader)
                self.assertEqual(len(submitted), 5)

    def test_loads_on_calling_thread_without_pillow(self):
        """Report every file the way ``load_texture()`` does when Pillow is missing, rather than raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            first = directory / "first.png"
            _save_noise_png(first, 8, 4)
            second = directory / "second.png"
            _save_noise_png(second, 4, 8)
            array = np.zeros((2, 2, 4), dtype=np.uint8)

            with (
                mock.patch.object(texture_utils, "_count_usable_cpus", return_value=4),
                mock.patch.object(texture_utils, "ThreadPoolExecutor", side_effect=AssertionError("pool created")),
                patch_sys_module("PIL", None),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                loaded = list(_load_textures((str(first), array, str(second))))

            self.assertEqual(len(loaded), 3)
            self.assertIsNone(loaded[0])
            np.testing.assert_array_equal(loaded[1], array)
            self.assertIsNone(loaded[2])
            messages = [str(w.message) for w in caught if issubclass(w.category, NewtonWarning)]
            self.assertEqual(len(messages), 2)
            self.assertIn(f"Failed to load texture image: {first}", messages[0])
            self.assertIn(f"Failed to load texture image: {second}", messages[1])

    def test_leaves_no_thread_behind(self):
        """Stop the worker threads once the textures are loaded, and when the iterator is closed early."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index in range(4):
                path = Path(tmpdir) / f"{index}.png"
                _save_noise_png(path, 8, 8)
                paths.append(str(path))
            self.assertEqual(_count_texture_threads(), 0)

            with mock.patch.object(texture_utils, "_count_usable_cpus", return_value=4):
                loader = _load_textures(paths)
                next(loader)
                self.assertGreaterEqual(_count_texture_threads(), 1)
                self.assertEqual(len(list(loader)), 3)
                self.assertEqual(_count_texture_threads(), 0)

                loader = _load_textures(paths)
                next(loader)
                self.assertGreaterEqual(_count_texture_threads(), 1)
                loader.close()
                self.assertEqual(_count_texture_threads(), 0)


if __name__ == "__main__":
    unittest.main()
