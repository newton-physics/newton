# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import pyglet

from newton._src.viewer.camera import Camera
from newton._src.viewer.viewer_gl import ViewerGL

pyglet.options["headless"] = True


def _as_np(value):
    return np.array((value[0], value[1], value[2]), dtype=float)


def _assert_vec_close(test, actual, expected, tol=1.0e-6):
    np.testing.assert_allclose(_as_np(actual), np.array(expected, dtype=float), atol=tol, rtol=0.0)


class TestViewerCameraOrbit(unittest.TestCase):
    def test_sync_pivot_to_view_tracks_fps_look(self):
        camera = Camera(pos=(0.0, 0.0, 0.0), up_axis="Z")
        camera.sync_pivot_to_view(distance=10.0)

        camera.yaw = 90.0
        camera.pitch = 30.0
        camera.sync_pivot_to_view()

        expected_pivot = _as_np(camera.pos) + _as_np(camera.get_front()) * 10.0
        _assert_vec_close(self, camera.pivot, expected_pivot)
        self.assertAlmostEqual(camera.pivot_distance, 10.0)

    def test_translate_moves_camera_and_pivot_together(self):
        camera = Camera(pos=(0.0, 0.0, 0.0), up_axis="Z")
        camera.sync_pivot_to_view(distance=7.0)
        start_offset = _as_np(camera.pivot) - _as_np(camera.pos)

        camera.translate((1.0, -2.0, 3.0))

        _assert_vec_close(self, camera.pos, (1.0, -2.0, 3.0))
        _assert_vec_close(self, _as_np(camera.pivot) - _as_np(camera.pos), start_offset)
        self.assertAlmostEqual(camera.pivot_distance, 7.0)

    def test_orbit_keeps_pivot_fixed_and_points_at_pivot(self):
        camera = Camera(pos=(10.0, 0.0, 0.0), up_axis="Z")
        camera.look_at((0.0, 0.0, 0.0))
        camera.set_pivot((0.0, 0.0, 0.0))

        camera.orbit(delta_yaw=45.0, delta_pitch=30.0)

        _assert_vec_close(self, camera.pivot, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(camera.pivot_distance, 10.0)
        direction_to_pivot = (_as_np(camera.pivot) - _as_np(camera.pos)) / camera.pivot_distance
        np.testing.assert_allclose(_as_np(camera.get_front()), direction_to_pivot, atol=1.0e-6, rtol=0.0)

    def test_look_at_points_front_at_pivot_for_all_up_axes(self):
        for up_axis in ("X", "Y", "Z"):
            with self.subTest(up_axis=up_axis):
                camera = Camera(pos=(1.0, 2.0, 3.0), up_axis=up_axis)
                camera.look_at((-4.0, 6.0, 2.0))

                direction_to_pivot = (_as_np(camera.pivot) - _as_np(camera.pos)) / camera.pivot_distance
                np.testing.assert_allclose(_as_np(camera.get_front()), direction_to_pivot, atol=1.0e-6, rtol=0.0)

    def test_orbit_clamps_pitch_to_89_degrees(self):
        camera = Camera(pos=(10.0, 0.0, 0.0), up_axis="Z")
        camera.look_at((0.0, 0.0, 0.0))
        camera.set_pivot((0.0, 0.0, 0.0))

        camera.orbit(delta_yaw=0.0, delta_pitch=200.0)

        self.assertEqual(camera.pitch, 89.0)
        self.assertTrue(math.isfinite(camera.pos[0]))
        self.assertAlmostEqual(camera.pivot_distance, 10.0)

    def test_pan_moves_camera_and_pivot_in_camera_plane(self):
        camera = Camera(pos=(10.0, 0.0, 0.0), up_axis="Z")
        camera.look_at((0.0, 0.0, 0.0))
        camera.set_pivot((0.0, 0.0, 0.0))

        start_pos = _as_np(camera.pos)
        start_pivot = _as_np(camera.pivot)
        right = _as_np(camera.get_right())
        up = _as_np(camera.get_up())

        camera.pan(delta_right=2.0, delta_up=-3.0)

        expected_delta = right * 2.0 + up * -3.0
        _assert_vec_close(self, camera.pos, start_pos + expected_delta)
        _assert_vec_close(self, camera.pivot, start_pivot + expected_delta)
        self.assertAlmostEqual(camera.pivot_distance, 10.0)

    def test_dolly_moves_camera_toward_pivot_without_moving_pivot(self):
        camera = Camera(pos=(10.0, 0.0, 0.0), up_axis="Z")
        camera.look_at((0.0, 0.0, 0.0))
        camera.set_pivot((0.0, 0.0, 0.0))

        camera.dolly(0.5)
        distance_after_dolly_in = camera.pivot_distance

        self.assertLess(distance_after_dolly_in, 10.0)
        _assert_vec_close(self, camera.pivot, (0.0, 0.0, 0.0))
        direction_to_pivot = (_as_np(camera.pivot) - _as_np(camera.pos)) / camera.pivot_distance
        np.testing.assert_allclose(_as_np(camera.get_front()), direction_to_pivot, atol=1.0e-6, rtol=0.0)

        camera.dolly(-0.5)

        self.assertGreater(camera.pivot_distance, distance_after_dolly_in)


class _FakeRenderer:
    def __init__(self):
        self.pressed_keys = set()

    def is_key_down(self, symbol):
        return symbol in self.pressed_keys


def _make_viewer_for_camera_input():
    viewer = ViewerGL.__new__(ViewerGL)
    viewer.ui = None
    viewer.camera = Camera(pos=(10.0, 0.0, 0.0), up_axis="Z")
    viewer.camera.look_at((0.0, 0.0, 0.0))
    viewer.camera.set_pivot((0.0, 0.0, 0.0))
    viewer.renderer = _FakeRenderer()
    viewer._camera_orbit_sensitivity = 0.1
    viewer._camera_dolly_scroll_sensitivity = 0.15
    viewer._camera_dolly_drag_sensitivity = 0.01
    viewer.picking_enabled = False
    viewer.picking = None
    return viewer


class TestViewerGLCameraInput(unittest.TestCase):
    def test_scroll_dollies_toward_pivot_by_default(self):
        viewer = _make_viewer_for_camera_input()
        start_distance = viewer.camera.pivot_distance
        start_fov = viewer.camera.fov

        viewer.on_mouse_scroll(x=0.0, y=0.0, scroll_x=0.0, scroll_y=1.0)

        self.assertLess(viewer.camera.pivot_distance, start_distance)
        self.assertEqual(viewer.camera.fov, start_fov)

    def test_ctrl_scroll_keeps_fov_zoom_escape_hatch(self):
        viewer = _make_viewer_for_camera_input()
        viewer.renderer.pressed_keys.add(pyglet.window.key.LCTRL)
        start_distance = viewer.camera.pivot_distance
        start_fov = viewer.camera.fov

        viewer.on_mouse_scroll(x=0.0, y=0.0, scroll_x=0.0, scroll_y=1.0)

        self.assertEqual(viewer.camera.pivot_distance, start_distance)
        self.assertLess(viewer.camera.fov, start_fov)

    def test_middle_mouse_drag_orbits_about_pivot(self):
        viewer = _make_viewer_for_camera_input()
        start_pos = _as_np(viewer.camera.pos)

        viewer.on_mouse_drag(
            x=0.0,
            y=0.0,
            dx=25.0,
            dy=10.0,
            buttons=pyglet.window.mouse.MIDDLE,
            modifiers=0,
        )

        self.assertFalse(np.allclose(_as_np(viewer.camera.pos), start_pos))
        _assert_vec_close(self, viewer.camera.pivot, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(viewer.camera.pivot_distance, 10.0)

    def test_shift_middle_mouse_drag_pans_camera_and_pivot(self):
        viewer = _make_viewer_for_camera_input()
        start_pos = _as_np(viewer.camera.pos)
        start_pivot = _as_np(viewer.camera.pivot)

        viewer.on_mouse_drag(
            x=0.0,
            y=0.0,
            dx=25.0,
            dy=10.0,
            buttons=pyglet.window.mouse.MIDDLE,
            modifiers=pyglet.window.key.MOD_SHIFT,
        )

        self.assertFalse(np.allclose(_as_np(viewer.camera.pos), start_pos))
        np.testing.assert_allclose(
            _as_np(viewer.camera.pivot) - start_pivot,
            _as_np(viewer.camera.pos) - start_pos,
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_ctrl_middle_mouse_drag_dollies_without_moving_pivot(self):
        viewer = _make_viewer_for_camera_input()
        start_distance = viewer.camera.pivot_distance
        start_pivot = _as_np(viewer.camera.pivot)

        viewer.on_mouse_drag(
            x=0.0,
            y=0.0,
            dx=0.0,
            dy=10.0,
            buttons=pyglet.window.mouse.MIDDLE,
            modifiers=pyglet.window.key.MOD_CTRL,
        )

        self.assertLess(viewer.camera.pivot_distance, start_distance)
        _assert_vec_close(self, viewer.camera.pivot, start_pivot)


if __name__ == "__main__":
    unittest.main()
