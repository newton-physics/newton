# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np

from newton._src.viewer.camera import Camera


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

    def test_pivot_distance_is_derived_from_pivot_and_position(self):
        camera = Camera(pos=(0.0, 0.0, 0.0), up_axis="Z")
        self.assertFalse(hasattr(camera, "_pivot_distance"))

        camera.pivot = camera._as_vec3((0.0, 0.0, 0.0))
        self.assertAlmostEqual(camera.pivot_distance, camera.MIN_PIVOT_DISTANCE)

        camera.pos = camera._as_vec3((0.0, 0.0, 2.0))
        self.assertAlmostEqual(camera.pivot_distance, 2.0)

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


if __name__ == "__main__":
    unittest.main()
