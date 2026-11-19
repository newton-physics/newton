# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton.sensors import SensorCamera

Utils = SensorCamera.Utils


def _device():
    return wp.get_preferred_device()


class TestToRgba(unittest.TestCase):
    def test_unpacks_uint32_to_uint8(self):
        # Packed RGBA: R=10, G=20, B=30, A=40 -> 0x28_1E_14_0A
        packed = (10) | (20 << 8) | (30 << 16) | (40 << 24)
        inp = np.full((6, 2, 2), packed, dtype=np.uint32)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=_device())

        out = Utils.to_rgba_from_color(inp_wp)

        got = out.numpy()
        self.assertEqual(got.shape, (6, 2, 2, 4))
        np.testing.assert_array_equal(got[..., 0], np.full((6, 2, 2), 10, dtype=np.uint8))
        np.testing.assert_array_equal(got[..., 1], np.full((6, 2, 2), 20, dtype=np.uint8))
        np.testing.assert_array_equal(got[..., 2], np.full((6, 2, 2), 30, dtype=np.uint8))
        np.testing.assert_array_equal(got[..., 3], np.full((6, 2, 2), 40, dtype=np.uint8))

    def test_view_and_pixel_axis_ordering(self):
        """Distinct values per (view, row, column) catch axis-swap bugs that a uniform input wouldn't."""
        # Encode view, row and column in R, G and B so a swap would flip which channel holds which index.
        view_count, height, width = 3, 2, 4
        inp = np.zeros((view_count, height, width), dtype=np.uint32)
        for v in range(view_count):
            for y in range(height):
                for x in range(width):
                    inp[v, y, x] = v | (y << 8) | (x << 16) | (255 << 24)

        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=_device())
        got = Utils.to_rgba_from_color(inp_wp).numpy()

        for v in range(view_count):
            for y in range(height):
                for x in range(width):
                    np.testing.assert_array_equal(got[v, y, x], (v, y, x, 255), err_msg=f"view={v} y={y} x={x}")


class TestToRgbaFromNormal(unittest.TestCase):
    def test_normal_maps_vec3_to_rgb(self):
        # (1, 2, 2) normal input.
        # Pixel (0,0): (1, 0, 0)     -> R=255, G=127 or 128, B=127 or 128
        # Pixel (0,1): (0, 1, 0)     -> R=127 or 128, G=255, B=127 or 128
        # Pixel (1,0): (0, 0, 1)     -> R=127 or 128, G=127 or 128, B=255
        # Pixel (1,1): (-1, -1, -1)  -> R=0, G=0, B=0
        inp = np.zeros((1, 2, 2, 3), dtype=np.float32)
        inp[0, 0, 0] = (1.0, 0.0, 0.0)
        inp[0, 0, 1] = (0.0, 1.0, 0.0)
        inp[0, 1, 0] = (0.0, 0.0, 1.0)
        inp[0, 1, 1] = (-1.0, -1.0, -1.0)
        inp_wp = wp.from_numpy(inp, dtype=wp.vec3f, device=_device())

        got = Utils.to_rgba_from_normal(inp_wp).numpy()
        self.assertEqual(got.shape, (1, 2, 2, 4))

        # (1, 0, 0) -> (255, ~127, ~127, 255)
        self.assertEqual(got[0, 0, 0, 0], 255)
        self.assertIn(got[0, 0, 0, 1], (127, 128))
        self.assertIn(got[0, 0, 0, 2], (127, 128))
        self.assertEqual(got[0, 0, 0, 3], 255)

        # (0, 1, 0) -> (~127, 255, ~127, 255)
        self.assertIn(got[0, 0, 1, 0], (127, 128))
        self.assertEqual(got[0, 0, 1, 1], 255)
        self.assertIn(got[0, 0, 1, 2], (127, 128))
        self.assertEqual(got[0, 0, 1, 3], 255)

        # (0, 0, 1) -> (~127, ~127, 255, 255)
        self.assertIn(got[0, 1, 0, 0], (127, 128))
        self.assertIn(got[0, 1, 0, 1], (127, 128))
        self.assertEqual(got[0, 1, 0, 2], 255)
        self.assertEqual(got[0, 1, 0, 3], 255)

        # (-1, -1, -1) -> (0, 0, 0, 255)
        self.assertEqual(got[0, 1, 1, 0], 0)
        self.assertEqual(got[0, 1, 1, 1], 0)
        self.assertEqual(got[0, 1, 1, 2], 0)
        self.assertEqual(got[0, 1, 1, 3], 255)

        # Regression: a component swap (say swapping R and B) would flip
        # pixel(0,0) from (255, ~127, ~127) to (~127, ~127, 255). Verify the
        # R channel of pixel(0,0) is strictly greater than its B channel so a
        # swap would be caught.
        self.assertGreater(got[0, 0, 0, 0], got[0, 0, 0, 2])


class TestToRgbaFromDepth(unittest.TestCase):
    def test_depth_normalizes_to_grayscale(self):
        # (1, 1, 4) depth input with near=1, far=10:
        # d=1.0   (near)        -> bright (255)
        # d=10.0  (far)         -> dim (50)
        # d=-1.0  (miss)        -> black (0, 0, 0, 255)
        # d=0.0   (clear-depth) -> black (matches the default
        #                         ClearData.clear_depth = 0.0 sentinel)
        inp = np.array([[[1.0, 10.0, -1.0, 0.0]]], dtype=np.float32)
        inp_wp = wp.from_numpy(inp, dtype=wp.float32, device=_device())
        depth_range = wp.array([1.0, 10.0], dtype=wp.float32, device=inp_wp.device)

        got = Utils.to_rgba_from_depth(inp_wp, depth_range=depth_range).numpy()
        self.assertEqual(got.shape, (1, 1, 4, 4))

        # Near pixel: bright.
        self.assertEqual(got[0, 0, 0, 0], 255)
        # Grayscale: R == G == B.
        self.assertEqual(got[0, 0, 0, 0], got[0, 0, 0, 1])
        self.assertEqual(got[0, 0, 0, 1], got[0, 0, 0, 2])
        self.assertEqual(got[0, 0, 0, 3], 255)

        # Far pixel: dim.
        self.assertEqual(got[0, 0, 1, 0], 50)
        self.assertEqual(got[0, 0, 1, 0], got[0, 0, 1, 1])
        self.assertEqual(got[0, 0, 1, 1], got[0, 0, 1, 2])
        self.assertEqual(got[0, 0, 1, 3], 255)

        # Negative-depth miss pixel: (0, 0, 0, 255).
        np.testing.assert_array_equal(got[0, 0, 2], (0, 0, 0, 255))

        # clear_depth=0.0 also renders as miss (regression guard for the
        # ClearData sentinel).
        np.testing.assert_array_equal(got[0, 0, 3], (0, 0, 0, 255))

        # Near pixel brighter than far pixel (closer = brighter).
        self.assertGreater(got[0, 0, 0, 0], got[0, 0, 1, 0])

    def test_depth_accepts_tuple_range(self):
        # Use d=1.0 (not 0.0) for the "near" pixel: d<=0 is the miss sentinel.
        inp = wp.from_numpy(np.array([[[1.0, 10.0]]], dtype=np.float32), dtype=wp.float32, device=_device())
        got = Utils.to_rgba_from_depth(inp, depth_range=(1.0, 10.0)).numpy()
        self.assertEqual(got[0, 0, 0, 0], 255)  # near -> bright
        self.assertEqual(got[0, 0, 1, 0], 50)  # far -> dim

    def test_depth_rejects_near_ge_far(self):
        inp = wp.zeros((1, 2, 2), dtype=wp.float32, device=_device())
        with self.assertRaises(ValueError) as cm:
            Utils.to_rgba_from_depth(inp, depth_range=(5.0, 3.0))
        self.assertIn("near < far", str(cm.exception))


class TestToRgbaFromShapeIndex(unittest.TestCase):
    def test_shape_index_hash_colors_differ_by_index(self):
        # (1, 2, 2) shape-index input with four distinct indices.
        inp = np.array([[[1, 2], [3, 4]]], dtype=np.uint32)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=_device())

        got = Utils.to_rgba_from_shape_index(inp_wp).numpy()

        # Collect the four RGB tuples; they should all be distinct.
        rgbs = {tuple(int(c) for c in got[0, y, x, :3]) for y in range(2) for x in range(2)}
        self.assertEqual(len(rgbs), 4, f"Expected 4 distinct hash colors, got: {rgbs}")

        # All alpha channels 255.
        np.testing.assert_array_equal(got[0, :, :, 3], np.full((2, 2), 255, dtype=np.uint8))

    def test_shape_index_hash_zero_is_not_black(self):
        # Shape index 0 must hash to a non-black color so a real shape doesn't
        # collide with the miss color. The miss sentinel ``0xFFFFFFFF`` must
        # still render black (wraps to 0 after the +1 bias).
        inp = np.array([[[0, 0xFFFFFFFF]]], dtype=np.uint32)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=_device())

        got = Utils.to_rgba_from_shape_index(inp_wp).numpy()

        zero_rgb = tuple(int(c) for c in got[0, 0, 0, :3])
        miss_rgb = tuple(int(c) for c in got[0, 0, 1, :3])
        self.assertNotEqual(zero_rgb, (0, 0, 0))
        self.assertEqual(miss_rgb, (0, 0, 0))

    def test_shape_index_palette_lookup(self):
        # (1, 1, 3) with indices [0, 1, 2], palette [red, green, blue].
        inp = np.array([[[0, 1, 2]]], dtype=np.uint32)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=_device())

        palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        palette_wp = wp.from_numpy(palette, dtype=wp.uint8, device=inp_wp.device)

        got = Utils.to_rgba_from_shape_index(inp_wp, colors=palette_wp).numpy()

        np.testing.assert_array_equal(got[0, 0, 0], (255, 0, 0, 255))
        np.testing.assert_array_equal(got[0, 0, 1], (0, 255, 0, 255))
        np.testing.assert_array_equal(got[0, 0, 2], (0, 0, 255, 255))

    def test_shape_index_palette_out_of_range_is_black(self):
        # Shape index 5 with a palette of length 3 -> black (0, 0, 0, 255).
        inp = np.array([[[5]]], dtype=np.uint32)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=_device())

        palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        palette_wp = wp.from_numpy(palette, dtype=wp.uint8, device=inp_wp.device)

        got = Utils.to_rgba_from_shape_index(inp_wp, colors=palette_wp).numpy()
        np.testing.assert_array_equal(got[0, 0, 0], (0, 0, 0, 255))


if __name__ == "__main__":
    unittest.main()
