# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.viewer.gl.image_logger import (
    _atlas_layout,
    _convert_to_packed_rgba_numpy,
    _pack_rgba_warp,
    _to_canonical_4d_numpy,
    _validate,
    compute_grid_layout,
)


def _pack(src: np.ndarray) -> np.ndarray:
    """Pack via the NumPy reference using the production atlas layout."""
    _, n, _, _, _ = _to_canonical_4d_numpy(src)
    cols, _ = _atlas_layout(n)
    return _convert_to_packed_rgba_numpy(src, cols)


class TestComputeGridLayout(unittest.TestCase):
    def test_n_zero_returns_zeros(self):
        rows, cols, cw, ch = compute_grid_layout(0, 1.0, 400, 400)
        self.assertEqual((rows, cols), (0, 0))
        self.assertEqual((cw, ch), (0.0, 0.0))

    def test_single_tile_fills_window(self):
        rows, cols, cw, ch = compute_grid_layout(1, 1.0, 400, 400)
        self.assertEqual((rows, cols), (1, 1))
        self.assertAlmostEqual(cw, 400)
        self.assertAlmostEqual(ch, 400)

    def test_square_tiles_prefer_sqrt_grid(self):
        rows, cols, _, _ = compute_grid_layout(16, 1.0, 800, 800)
        self.assertEqual(cols, 4)
        self.assertEqual(rows, 4)

    def test_wide_window_packs_more_cols(self):
        # 4 square tiles in a very wide window should be one row of 4
        rows, cols, _, _ = compute_grid_layout(4, 1.0, 1600, 100)
        self.assertEqual(cols, 4)
        self.assertEqual(rows, 1)

    def test_tall_window_packs_more_rows(self):
        # 4 square tiles in a very tall window should be one col of 4
        rows, cols, _, _ = compute_grid_layout(4, 1.0, 100, 1600)
        self.assertEqual(cols, 1)
        self.assertEqual(rows, 4)

    def test_n_seven_yields_three_by_three_with_gaps(self):
        rows, cols, _, _ = compute_grid_layout(7, 1.0, 600, 600)
        self.assertEqual(cols, 3)
        self.assertEqual(rows, 3)
        # 3*3 = 9 cells, 2 empty

    def test_height_limited_clamps_cells(self):
        # 4 square tiles, narrow+short window makes cells height-limited
        _rows, _cols, cw, ch = compute_grid_layout(4, 1.0, 800, 100)
        # Expected: 4x1 layout, cell_h <= 100
        self.assertLessEqual(ch, 100.0 + 1e-6)
        # Cell must be square given tile_aspect=1.0
        self.assertAlmostEqual(cw, ch, places=6)

    def test_extreme_aspect_tall(self):
        # very tall tiles (aspect=10) still return positive cell sizes
        _, _, cw, ch = compute_grid_layout(4, 10.0, 400, 800)
        self.assertGreater(cw, 0)
        self.assertGreater(ch, 0)

    def test_extreme_aspect_wide(self):
        # very wide tiles (aspect=0.1) still return positive cell sizes
        _, _, cw, ch = compute_grid_layout(4, 0.1, 400, 800)
        self.assertGreater(cw, 0)
        self.assertGreater(ch, 0)

    def test_spacing_reduces_cell_size(self):
        # Without spacing: 4 cells in an 800-wide window give 200-wide cells.
        _, _, cw_no_spacing, _ = compute_grid_layout(4, 1.0, 800.0, 200.0)
        # With 10 px spacing, 3 gaps * 10 = 30 px removed, so cells shrink.
        _, _, cw_with_spacing, _ = compute_grid_layout(4, 1.0, 800.0, 200.0, spacing_x=10.0, spacing_y=0.0)
        self.assertLess(cw_with_spacing, cw_no_spacing)

    def test_spacing_never_produces_negative_cells(self):
        # Extreme spacing that would push the usable area below zero.
        _rows, _cols, cw, ch = compute_grid_layout(9, 1.0, 20.0, 20.0, spacing_x=100.0, spacing_y=100.0)
        # Should still return positive (or zero) finite cell sizes, not negative.
        self.assertGreaterEqual(cw, 0.0)
        self.assertGreaterEqual(ch, 0.0)


class TestValidate(unittest.TestCase):
    def test_rejects_non_array(self):
        with self.assertRaises(ValueError) as cm:
            _validate("x", [[1, 2], [3, 4]])  # plain list
        self.assertIn("expected wp.array or np.ndarray", str(cm.exception))

    def test_rejects_empty_name(self):
        with self.assertRaises(ValueError) as cm:
            _validate("", np.zeros((4, 4), dtype=np.uint8))
        self.assertIn("name must be a non-empty string", str(cm.exception))

    def test_rejects_1d_array(self):
        with self.assertRaises(ValueError) as cm:
            _validate("x", np.zeros(16, dtype=np.uint8))
        self.assertIn("expected 2D, 3D, or 4D", str(cm.exception))

    def test_rejects_5d_array(self):
        with self.assertRaises(ValueError) as cm:
            _validate("x", np.zeros((2, 3, 4, 5, 6), dtype=np.uint8))
        self.assertIn("expected 2D, 3D, or 4D", str(cm.exception))

    def test_rejects_4d_channel_count_2(self):
        with self.assertRaises(ValueError) as cm:
            _validate("x", np.zeros((2, 8, 8, 2), dtype=np.uint8))
        self.assertIn("C in (1, 3, 4)", str(cm.exception))

    def test_rejects_int16_dtype(self):
        with self.assertRaises(ValueError) as cm:
            _validate("x", np.zeros((8, 8), dtype=np.int16))
        self.assertIn("expected uint8 or float32", str(cm.exception))

    def test_rejects_zero_dimension(self):
        with self.assertRaises(ValueError) as cm:
            _validate("x", np.zeros((0, 8, 8), dtype=np.uint8))
        self.assertIn("all dimensions must be positive", str(cm.exception))

    def test_accepts_uint8_2d(self):
        kind = _validate("x", np.zeros((16, 16), dtype=np.uint8))
        self.assertEqual(kind, (1, 16, 16, 1))

    def test_accepts_float32_2d(self):
        kind = _validate("x", np.zeros((16, 16), dtype=np.float32))
        self.assertEqual(kind, (1, 16, 16, 1))

    def test_accepts_uint8_3d_color(self):
        kind = _validate("x", np.zeros((16, 16, 3), dtype=np.uint8))
        self.assertEqual(kind, (1, 16, 16, 3))

    def test_accepts_uint8_3d_rgba(self):
        kind = _validate("x", np.zeros((16, 16, 4), dtype=np.uint8))
        self.assertEqual(kind, (1, 16, 16, 4))

    def test_accepts_float32_3d_batch_grayscale(self):
        # Last dim 8 not in {1,3,4}, so treated as (N=16, H=16, W=8)
        kind = _validate("x", np.zeros((16, 16, 8), dtype=np.float32))
        self.assertEqual(kind, (16, 16, 8, 1))

    def test_accepts_uint8_4d_batched(self):
        kind = _validate("x", np.zeros((5, 16, 16, 4), dtype=np.uint8))
        self.assertEqual(kind, (5, 16, 16, 4))

    def test_accepts_warp_array(self):
        arr = wp.zeros((8, 8), dtype=wp.uint8)
        kind = _validate("x", arr)
        self.assertEqual(kind, (1, 8, 8, 1))


class TestAtlasLayout(unittest.TestCase):
    """``_atlas_layout`` picks square-ish ``(cols, rows)`` keeping cols >= rows."""

    def test_zero(self):
        self.assertEqual(_atlas_layout(0), (0, 0))

    def test_one(self):
        self.assertEqual(_atlas_layout(1), (1, 1))

    def test_perfect_square(self):
        self.assertEqual(_atlas_layout(9), (3, 3))
        self.assertEqual(_atlas_layout(16), (4, 4))

    def test_non_square_packs_remainder(self):
        # 7 -> ceil(sqrt(7))=3 cols, 3 rows (last row has 1 tile, 2 empty slots).
        self.assertEqual(_atlas_layout(7), (3, 3))
        # 5 -> 3 cols, 2 rows.
        self.assertEqual(_atlas_layout(5), (3, 2))

    def test_avoids_strip_for_moderate_batches(self):
        # Eric's case: 65 tiles must NOT pack as a (1, 65) strip, which would
        # blow past GL_MAX_TEXTURE_SIZE for moderate tile sizes.
        cols, rows = _atlas_layout(65)
        self.assertGreater(cols, 1)
        self.assertGreater(rows, 1)
        self.assertGreaterEqual(cols * rows, 65)


class TestConvertToPackedRgbaNumpy(unittest.TestCase):
    def test_grayscale_2d_uint8_replicates_to_rgb_full_alpha(self):
        src = np.array([[10, 20], [30, 40]], dtype=np.uint8)  # (H=2, W=2)
        out = _pack(src)
        # Canonical shape (1,2,2,1), N=1 -> 1x1 atlas: (H=2, W=2, 4).
        self.assertEqual(out.shape, (2, 2, 4))
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out[..., 0], [[10, 20], [30, 40]])
        np.testing.assert_array_equal(out[..., 1], [[10, 20], [30, 40]])
        np.testing.assert_array_equal(out[..., 2], [[10, 20], [30, 40]])
        np.testing.assert_array_equal(out[..., 3], np.full((2, 2), 255, dtype=np.uint8))

    def test_rgb_3d_uint8_preserves_channels_full_alpha(self):
        src = np.zeros((2, 2, 3), dtype=np.uint8)
        src[..., 0] = 10
        src[..., 1] = 20
        src[..., 2] = 30
        out = _pack(src)
        self.assertEqual(out.shape, (2, 2, 4))
        np.testing.assert_array_equal(out[..., 0], np.full((2, 2), 10, dtype=np.uint8))
        np.testing.assert_array_equal(out[..., 3], np.full((2, 2), 255, dtype=np.uint8))

    def test_rgba_3d_uint8_preserves_alpha(self):
        src = np.zeros((2, 2, 4), dtype=np.uint8)
        src[..., 3] = 77
        out = _pack(src)
        np.testing.assert_array_equal(out[..., 3], np.full((2, 2), 77, dtype=np.uint8))

    def test_batched_4d_packs_into_atlas(self):
        # 3D (3, 2, 2) -> (N=3, H=2, W=2) grayscale batch (last dim 2 not in {1,3,4}).
        # N=3 -> atlas cols=2, rows=2 -> (4, 4, 4); slot (1,1) is empty (zeroed).
        src = np.zeros((3, 2, 2), dtype=np.uint8)
        src[0] = 10
        src[1] = 20
        src[2] = 30
        out = _pack(src)
        self.assertEqual(out.shape, (4, 4, 4))
        # Tile 0 -> (row 0, col 0)
        np.testing.assert_array_equal(out[0:2, 0:2, 0], np.full((2, 2), 10, dtype=np.uint8))
        # Tile 1 -> (row 0, col 1)
        np.testing.assert_array_equal(out[0:2, 2:4, 0], np.full((2, 2), 20, dtype=np.uint8))
        # Tile 2 -> (row 1, col 0)
        np.testing.assert_array_equal(out[2:4, 0:2, 0], np.full((2, 2), 30, dtype=np.uint8))
        # Empty slot -> zeros (RGBA all zero, including alpha).
        np.testing.assert_array_equal(out[2:4, 2:4], np.zeros((2, 2, 4), dtype=np.uint8))

    def test_float32_clips_and_scales(self):
        src = np.array([[-0.5, 0.0, 0.5, 1.5]], dtype=np.float32).reshape(1, 4)
        out = _pack(src)
        expected_luma = np.array([0, 0, 127, 255], dtype=np.uint8).reshape(1, 4)
        np.testing.assert_array_equal(out[..., 0], expected_luma)
        np.testing.assert_array_equal(out[..., 3], np.full((1, 4), 255, dtype=np.uint8))

    def test_float32_rgba_preserves_alpha(self):
        src = np.zeros((1, 1, 4), dtype=np.float32)
        src[0, 0] = [0.0, 0.5, 1.0, 0.25]
        out = _pack(src)
        self.assertIn(int(out[0, 0, 3]), (63, 64))


class TestPackRgbaWarp(unittest.TestCase):
    """Warp kernel output must match NumPy reference (exact for uint8, 1-ULP for float32)."""

    def _run(self, src_np: np.ndarray) -> np.ndarray:
        device = wp.get_preferred_device()
        arr_np, n, h, w, c = _to_canonical_4d_numpy(src_np)
        cols, rows = _atlas_layout(n)
        dtype = wp.uint8 if src_np.dtype == np.uint8 else wp.float32
        src_4d = wp.from_numpy(arr_np, dtype=dtype, device=device)
        out_wp = wp.zeros((rows * h, cols * w, 4), dtype=wp.uint8, device=device)
        _pack_rgba_warp(src_4d, c, cols, out_wp)
        return out_wp.numpy()

    def test_uint8_grayscale_matches_numpy(self):
        src = (np.arange(16, dtype=np.uint8)).reshape(4, 4)
        got = self._run(src)
        want = _pack(src)
        np.testing.assert_array_equal(got, want)

    def test_uint8_rgb_matches_numpy(self):
        src = np.random.default_rng(0).integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
        got = self._run(src)
        want = _pack(src)
        np.testing.assert_array_equal(got, want)

    def test_uint8_rgba_batched_matches_numpy(self):
        src = np.random.default_rng(1).integers(0, 256, size=(5, 4, 4, 4), dtype=np.uint8)
        got = self._run(src)
        want = _pack(src)
        np.testing.assert_array_equal(got, want)

    def test_float32_grayscale_matches_numpy_within_1(self):
        src = np.random.default_rng(2).random((4, 4), dtype=np.float32)
        got = self._run(src)
        want = _pack(src)
        self.assertTrue(np.all(np.abs(got.astype(int) - want.astype(int)) <= 1))

    def test_float32_rgba_batched_clips_out_of_range(self):
        # All 4 tiles fully populate the 2x2 atlas, so every pixel is touched.
        src = np.full((4, 2, 2, 4), 1.5, dtype=np.float32)
        got = self._run(src)
        self.assertTrue(np.all(got == 255))


if __name__ == "__main__":
    unittest.main()
