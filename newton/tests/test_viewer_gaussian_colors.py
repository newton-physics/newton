# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton._src.geometry.types import SH_C0
from newton._src.viewer.utils import gaussian_display_colors


class TestGaussianDisplayColors(unittest.TestCase):
    @staticmethod
    def _sh_for_display_rgb(rgb):
        """SH DC coefficients whose evaluated radiance is ``rgb``."""
        return (np.asarray([rgb], dtype=np.float32) - 0.5) / SH_C0

    def test_srgb_radiance_is_displayed_as_is(self):
        colors = gaussian_display_colors(self._sh_for_display_rgb([0.2, 0.6, 0.8]), newton.utils.ColorSpace.SRGB)
        self.assertEqual(colors.dtype, np.float32)
        np.testing.assert_allclose(colors, [[0.2, 0.6, 0.8]], atol=1e-6)

    def test_linear_radiance_is_encoded_to_srgb(self):
        rgb = [0.0, 0.2, 0.6]
        colors = gaussian_display_colors(self._sh_for_display_rgb(rgb), newton.utils.ColorSpace.LINEAR)
        np.testing.assert_allclose(colors, [newton.utils.color_linear_to_srgb(rgb)], atol=1e-5)

    def test_colors_are_clamped_to_display_range(self):
        sh = np.array([[-10.0, 0.0, 10.0]], dtype=np.float32)
        for color_space in (newton.utils.ColorSpace.SRGB, newton.utils.ColorSpace.LINEAR):
            colors = gaussian_display_colors(sh, color_space)
            self.assertEqual(colors[0, 0], 0.0)
            self.assertEqual(colors[0, 2], 1.0)

    def test_higher_order_coefficients_are_ignored(self):
        dc = self._sh_for_display_rgb([0.3, 0.4, 0.5])
        sh = np.concatenate([dc, np.full((1, 6), 5.0, dtype=np.float32)], axis=1)
        np.testing.assert_allclose(
            gaussian_display_colors(sh, newton.utils.ColorSpace.SRGB), [[0.3, 0.4, 0.5]], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
