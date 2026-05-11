# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import types
import unittest
import warnings

import numpy as np
import warp as wp


def _make_utils(world_count: int = 1, device=None):
    """Construct a Utils with a minimal stand-in RenderContext.

    Exercises the public adapter methods (argument conversion, buffer
    allocation) without requiring a full SensorTiledCamera + Model setup.
    """
    from newton._src.sensors.warp_raytrace.utils import Utils  # noqa: PLC0415

    render_context = types.SimpleNamespace(
        world_count=world_count,
        device=device or wp.get_preferred_device(),
    )
    return Utils(render_context)


class TestToRgba(unittest.TestCase):
    def test_unpacks_uint32_to_uint8(self):
        # 4D uint32 input: (W=2 worlds, C=3 cams, H=2, Wpix=2).
        # Packed RGBA: R=10, G=20, B=30, A=40 -> 0x28_1E_14_0A
        packed = (10) | (20 << 8) | (30 << 16) | (40 << 24)
        inp = np.full((2, 3, 2, 2), packed, dtype=np.uint32)
        utils = _make_utils(world_count=2)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=utils._Utils__render_context.device)

        out = utils.to_rgba_from_color(inp_wp)

        got = out.numpy()
        self.assertEqual(got.shape, (6, 2, 2, 4))
        np.testing.assert_array_equal(got[..., 0], np.full((6, 2, 2), 10, dtype=np.uint8))
        np.testing.assert_array_equal(got[..., 1], np.full((6, 2, 2), 20, dtype=np.uint8))
        np.testing.assert_array_equal(got[..., 2], np.full((6, 2, 2), 30, dtype=np.uint8))
        np.testing.assert_array_equal(got[..., 3], np.full((6, 2, 2), 40, dtype=np.uint8))

    def test_world_camera_axis_ordering(self):
        """Distinct values per (world, camera) catch axis-swap bugs that a uniform input wouldn't."""
        # world_count=3, camera_count=2, H=1, W=1.
        # Encode world & camera in R and G so a swap would flip which channel holds which index.
        # world index -> R; camera index -> G; B=0; A=255
        world_count, camera_count = 3, 2
        inp = np.zeros((world_count, camera_count, 1, 1), dtype=np.uint32)
        for w in range(world_count):
            for c in range(camera_count):
                packed = (w) | (c << 8) | (0 << 16) | (255 << 24)
                inp[w, c, 0, 0] = packed

        utils = _make_utils(world_count=world_count)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=utils._Utils__render_context.device)
        out = utils.to_rgba_from_color(inp_wp)
        got = out.numpy()

        # Expected: tile i = w * camera_count + c; R = w, G = c
        for w in range(world_count):
            for c in range(camera_count):
                tile = w * camera_count + c
                self.assertEqual(got[tile, 0, 0, 0], w, f"R channel mismatch at world={w} camera={c}")
                self.assertEqual(got[tile, 0, 0, 1], c, f"G channel mismatch at world={w} camera={c}")
                self.assertEqual(got[tile, 0, 0, 2], 0)
                self.assertEqual(got[tile, 0, 0, 3], 255)


class TestToRgbaFromNormal(unittest.TestCase):
    def test_normal_maps_vec3_to_rgb(self):
        from newton._src.sensors.warp_raytrace.utils import unpack_normal_to_rgba_kernel  # noqa: PLC0415

        # (1, 1, 2, 2) normal input.
        # Pixel (0,0): (1, 0, 0)     -> R=255, G=127 or 128, B=127 or 128
        # Pixel (0,1): (0, 1, 0)     -> R=127 or 128, G=255, B=127 or 128
        # Pixel (1,0): (0, 0, 1)     -> R=127 or 128, G=127 or 128, B=255
        # Pixel (1,1): (-1, -1, -1)  -> R=0, G=0, B=0
        inp = np.zeros((1, 1, 2, 2, 3), dtype=np.float32)
        inp[0, 0, 0, 0] = (1.0, 0.0, 0.0)
        inp[0, 0, 0, 1] = (0.0, 1.0, 0.0)
        inp[0, 0, 1, 0] = (0.0, 0.0, 1.0)
        inp[0, 0, 1, 1] = (-1.0, -1.0, -1.0)
        inp_wp = wp.from_numpy(inp, dtype=wp.vec3f, device=wp.get_preferred_device())

        out = wp.empty((1, 2, 2, 4), dtype=wp.uint8, device=inp_wp.device)
        wp.launch(
            unpack_normal_to_rgba_kernel,
            dim=(1, 1, 2, 2),
            inputs=[inp_wp],
            outputs=[out],
            device=inp_wp.device,
        )
        got = out.numpy()
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
        from newton._src.sensors.warp_raytrace.utils import unpack_depth_to_rgba_kernel  # noqa: PLC0415

        # (1, 1, 1, 4) depth input with near=1, far=10:
        # d=1.0   (near)        -> bright (255)
        # d=10.0  (far)         -> dim (50)
        # d=-1.0  (miss)        -> black (0, 0, 0, 255)
        # d=0.0   (clear-depth) -> black (matches the legacy
        #                         flatten_depth_image kernel and the default
        #                         ClearData.clear_depth = 0.0 sentinel)
        inp = np.array([[[[1.0, 10.0, -1.0, 0.0]]]], dtype=np.float32)
        inp_wp = wp.from_numpy(inp, dtype=wp.float32, device=wp.get_preferred_device())
        depth_range = wp.array([1.0, 10.0], dtype=wp.float32, device=inp_wp.device)

        out = wp.empty((1, 1, 4, 4), dtype=wp.uint8, device=inp_wp.device)
        wp.launch(
            unpack_depth_to_rgba_kernel,
            dim=(1, 1, 1, 4),
            inputs=[inp_wp, depth_range],
            outputs=[out],
            device=inp_wp.device,
        )
        got = out.numpy()
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
        self.assertEqual(got[0, 0, 2, 0], 0)
        self.assertEqual(got[0, 0, 2, 1], 0)
        self.assertEqual(got[0, 0, 2, 2], 0)
        self.assertEqual(got[0, 0, 2, 3], 255)

        # clear_depth=0.0 also renders as miss (regression guard for the
        # ClearData sentinel; was bright in earlier revisions of this PR).
        self.assertEqual(got[0, 0, 3, 0], 0)
        self.assertEqual(got[0, 0, 3, 1], 0)
        self.assertEqual(got[0, 0, 3, 2], 0)
        self.assertEqual(got[0, 0, 3, 3], 255)

        # Near pixel brighter than far pixel (closer = brighter).
        self.assertGreater(got[0, 0, 0, 0], got[0, 0, 1, 0])


class TestToRgbaFromShapeIndex(unittest.TestCase):
    def test_shape_index_hash_colors_differ_by_index(self):
        from newton._src.sensors.warp_raytrace.utils import (  # noqa: PLC0415
            unpack_shape_index_hash_to_rgba_kernel,
        )

        # (1, 1, 2, 2) shape-index input with four distinct indices.
        inp = np.array([[[[1, 2], [3, 4]]]], dtype=np.uint32)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=wp.get_preferred_device())

        out = wp.empty((1, 2, 2, 4), dtype=wp.uint8, device=inp_wp.device)
        wp.launch(
            unpack_shape_index_hash_to_rgba_kernel,
            dim=(1, 1, 2, 2),
            inputs=[inp_wp],
            outputs=[out],
            device=inp_wp.device,
        )
        got = out.numpy()

        # Collect the four RGB tuples; they should all be distinct.
        rgbs = {
            (int(got[0, 0, 0, 0]), int(got[0, 0, 0, 1]), int(got[0, 0, 0, 2])),
            (int(got[0, 0, 1, 0]), int(got[0, 0, 1, 1]), int(got[0, 0, 1, 2])),
            (int(got[0, 1, 0, 0]), int(got[0, 1, 0, 1]), int(got[0, 1, 0, 2])),
            (int(got[0, 1, 1, 0]), int(got[0, 1, 1, 1]), int(got[0, 1, 1, 2])),
        }
        self.assertEqual(len(rgbs), 4, f"Expected 4 distinct hash colors, got: {rgbs}")

        # All alpha channels 255.
        for y in range(2):
            for x in range(2):
                self.assertEqual(got[0, y, x, 3], 255)

    def test_shape_index_hash_zero_is_not_black(self):
        # Shape index 0 must hash to a non-black color so a real shape doesn't
        # collide with the miss color. The miss sentinel ``0xFFFFFFFF`` must
        # still render black (wraps to 0 after the +1 bias).
        from newton._src.sensors.warp_raytrace.utils import (  # noqa: PLC0415
            unpack_shape_index_hash_to_rgba_kernel,
        )

        inp = np.array([[[[0, 0xFFFFFFFF]]]], dtype=np.uint32)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=wp.get_preferred_device())

        out = wp.empty((1, 1, 2, 4), dtype=wp.uint8, device=inp_wp.device)
        wp.launch(
            unpack_shape_index_hash_to_rgba_kernel,
            dim=(1, 1, 1, 2),
            inputs=[inp_wp],
            outputs=[out],
            device=inp_wp.device,
        )
        got = out.numpy()

        zero_rgb = (int(got[0, 0, 0, 0]), int(got[0, 0, 0, 1]), int(got[0, 0, 0, 2]))
        miss_rgb = (int(got[0, 0, 1, 0]), int(got[0, 0, 1, 1]), int(got[0, 0, 1, 2]))
        self.assertNotEqual(zero_rgb, (0, 0, 0))
        self.assertEqual(miss_rgb, (0, 0, 0))

    def test_shape_index_palette_lookup(self):
        from newton._src.sensors.warp_raytrace.utils import colorize_shape_index_with_palette_kernel  # noqa: PLC0415

        # (1, 1, 1, 3) with indices [0, 1, 2], palette [red, green, blue].
        inp = np.array([[[[0, 1, 2]]]], dtype=np.uint32)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=wp.get_preferred_device())

        palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        palette_wp = wp.from_numpy(palette, dtype=wp.uint8, device=inp_wp.device)

        out = wp.empty((1, 1, 3, 4), dtype=wp.uint8, device=inp_wp.device)
        wp.launch(
            colorize_shape_index_with_palette_kernel,
            dim=(1, 1, 1, 3),
            inputs=[inp_wp, palette_wp],
            outputs=[out],
            device=inp_wp.device,
        )
        got = out.numpy()

        # Pixel 0: red.
        self.assertEqual(got[0, 0, 0, 0], 255)
        self.assertEqual(got[0, 0, 0, 1], 0)
        self.assertEqual(got[0, 0, 0, 2], 0)
        self.assertEqual(got[0, 0, 0, 3], 255)
        # Pixel 1: green.
        self.assertEqual(got[0, 0, 1, 0], 0)
        self.assertEqual(got[0, 0, 1, 1], 255)
        self.assertEqual(got[0, 0, 1, 2], 0)
        self.assertEqual(got[0, 0, 1, 3], 255)
        # Pixel 2: blue.
        self.assertEqual(got[0, 0, 2, 0], 0)
        self.assertEqual(got[0, 0, 2, 1], 0)
        self.assertEqual(got[0, 0, 2, 2], 255)
        self.assertEqual(got[0, 0, 2, 3], 255)

    def test_shape_index_palette_out_of_range_is_black(self):
        from newton._src.sensors.warp_raytrace.utils import colorize_shape_index_with_palette_kernel  # noqa: PLC0415

        # Shape index 5 with a palette of length 3 -> black (0, 0, 0, 255).
        inp = np.array([[[[5]]]], dtype=np.uint32)
        inp_wp = wp.from_numpy(inp, dtype=wp.uint32, device=wp.get_preferred_device())

        palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        palette_wp = wp.from_numpy(palette, dtype=wp.uint8, device=inp_wp.device)

        out = wp.empty((1, 1, 1, 4), dtype=wp.uint8, device=inp_wp.device)
        wp.launch(
            colorize_shape_index_with_palette_kernel,
            dim=(1, 1, 1, 1),
            inputs=[inp_wp, palette_wp],
            outputs=[out],
            device=inp_wp.device,
        )
        got = out.numpy()
        self.assertEqual(got[0, 0, 0, 0], 0)
        self.assertEqual(got[0, 0, 0, 1], 0)
        self.assertEqual(got[0, 0, 0, 2], 0)
        self.assertEqual(got[0, 0, 0, 3], 255)


class TestUtilsPublicAdapterAPI(unittest.TestCase):
    """Exercise the public Utils.to_rgba_from_* wrappers (not just raw kernels)."""

    def test_color_returns_canonical_shape_and_dtype(self):
        utils = _make_utils(world_count=2)
        inp = wp.zeros((2, 3, 4, 4), dtype=wp.uint32, device=utils._Utils__render_context.device)
        out = utils.to_rgba_from_color(inp)
        self.assertEqual(tuple(out.shape), (6, 4, 4, 4))
        self.assertEqual(out.dtype, wp.uint8)

    def test_depth_accepts_tuple_range(self):
        utils = _make_utils(world_count=1)
        device = utils._Utils__render_context.device
        # Use d=1.0 (not 0.0) for the "near" pixel: d<=0 is the miss sentinel.
        inp = wp.from_numpy(
            np.array([[[[1.0, 10.0]]]], dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )
        out = utils.to_rgba_from_depth(inp, depth_range=(1.0, 10.0))
        got = out.numpy()
        self.assertEqual(got[0, 0, 0, 0], 255)  # near -> bright
        self.assertEqual(got[0, 0, 1, 0], 50)  # far -> dim

    def test_depth_rejects_near_ge_far(self):
        utils = _make_utils(world_count=1)
        device = utils._Utils__render_context.device
        inp = wp.zeros((1, 1, 2, 2), dtype=wp.float32, device=device)
        with self.assertRaises(ValueError) as cm:
            utils.to_rgba_from_depth(inp, depth_range=(5.0, 3.0))
        self.assertIn("near < far", str(cm.exception))

    def test_worlds_per_row_zero_warns_and_auto_layouts(self):
        # Legacy callers passed worlds_per_row=0 to mean "auto layout"
        # because the original gate was a falsy check. Behavior preserved
        # behind a DeprecationWarning; outcome must match worlds_per_row=None.
        utils = _make_utils(world_count=4)
        device = utils._Utils__render_context.device
        inp = wp.zeros((4, 1, 3, 5), dtype=wp.uint32, device=device)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            out_zero = utils.flatten_color_image_to_rgba(inp, worlds_per_row=0)
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "expected DeprecationWarning for worlds_per_row=0",
        )

        out_auto = utils.flatten_color_image_to_rgba(inp, worlds_per_row=None)
        self.assertEqual(tuple(out_zero.shape), tuple(out_auto.shape))

    def test_worlds_per_row_negative_raises(self):
        utils = _make_utils(world_count=1)
        device = utils._Utils__render_context.device
        inp = wp.zeros((1, 1, 2, 2), dtype=wp.uint32, device=device)
        with self.assertRaises(ValueError):
            utils.flatten_color_image_to_rgba(inp, worlds_per_row=-1)


if __name__ == "__main__":
    unittest.main()
