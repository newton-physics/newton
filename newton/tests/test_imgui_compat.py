# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton._src.viewer.gl.imgui_compat import color_edit3_tuple, to_imgui_color4, to_rgb_tuple


class _FakeImVec4:
    def __init__(self, x: float, y: float, z: float, w: float):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class _FakeImgui:
    ImVec4 = _FakeImVec4

    def __init__(self):
        self.last_color = None

    def color_edit3(self, label, color, flags=0):
        self.last_color = color
        return True, self.ImVec4(color.x * 0.5, color.y * 0.5, color.z * 0.5, color.w)


class TestImguiCompat(unittest.TestCase):
    def test_to_imgui_color4_accepts_rgb_tuple(self):
        imgui = _FakeImgui()
        color = to_imgui_color4(imgui, (0.1, 0.2, 0.3))
        self.assertIsInstance(color, _FakeImVec4)
        self.assertEqual((color.x, color.y, color.z, color.w), (0.1, 0.2, 0.3, 1.0))

    def test_to_rgb_tuple_accepts_imvec4_like(self):
        color = _FakeImVec4(0.4, 0.5, 0.6, 1.0)
        self.assertEqual(to_rgb_tuple(color), (0.4, 0.5, 0.6))

    def test_color_edit3_tuple_round_trips_to_rgb_tuple(self):
        imgui = _FakeImgui()
        changed, color = color_edit3_tuple(imgui, "Light Color", (1.0, 0.8, 0.6))
        self.assertTrue(changed)
        self.assertIsInstance(imgui.last_color, _FakeImVec4)
        self.assertEqual(color, (0.5, 0.4, 0.3))

    def test_invalid_color_arity_raises(self):
        imgui = _FakeImgui()
        with self.assertRaises(ValueError):
            to_imgui_color4(imgui, (0.1, 0.2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
