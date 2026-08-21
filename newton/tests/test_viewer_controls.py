# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from newton._src.viewer.viewer_gl import ViewerGL
from newton._src.viewer.viewer_gui import ViewerGui
from newton._src.viewer.viewer_null import ViewerNull

_Vec3 = namedtuple("_Vec3", ("x", "y", "z"))


def _make_gl_state(paused: bool = False, step_requested: bool = False) -> "ViewerGL":
    # Lightweight stand-in with just the fields ViewerGL.should_step() needs.
    return SimpleNamespace(_paused=paused, _step_requested=step_requested)  # type: ignore[return-value]


class TestViewerBaseShouldStep(unittest.TestCase):
    """ViewerBase.should_step() defaults to not self.is_paused()."""

    def test_returns_true_when_not_paused(self):
        viewer = ViewerNull()
        self.assertTrue(viewer.should_step())

    def test_returns_true_on_repeated_calls(self):
        viewer = ViewerNull()
        for _ in range(3):
            self.assertTrue(viewer.should_step())


class TestViewerCameraSpeed(unittest.TestCase):
    def test_defaults_to_four_meters_per_second(self):
        self.assertEqual(ViewerNull().camera_speed, 4.0)

    def test_accepts_finite_nonnegative_values(self):
        viewer = ViewerNull()

        viewer.camera_speed = 0.2
        self.assertEqual(viewer.camera_speed, 0.2)

        viewer.camera_speed = 0.0
        self.assertEqual(viewer.camera_speed, 0.0)

    def test_rejects_negative_and_nonfinite_values(self):
        viewer = ViewerNull()

        for value in (-1.0, float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                viewer.camera_speed = value

    def test_gui_keyboard_movement_uses_viewer_camera_speed(self):
        camera = SimpleNamespace(
            pos=_Vec3(0.0, 0.0, 0.0),
            get_front=lambda: (1.0, 0.0, 0.0),
            get_right=lambda: (0.0, 1.0, 0.0),
            get_up=lambda: (0.0, 0.0, 1.0),
        )
        viewer = SimpleNamespace(camera=camera, camera_speed=2.0)
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.ui = None
        gui._cam_vel = np.zeros(3, dtype=np.float32)
        gui._cam_damp_tau = 0.1

        key = SimpleNamespace(W=1, UP=2, S=3, DOWN=4, A=5, LEFT=6, D=7, RIGHT=8, Q=9, E=10)
        pyglet = SimpleNamespace(window=SimpleNamespace(key=key))
        with patch.dict(sys.modules, {"pyglet": pyglet}):
            gui.update_camera_from_keys(0.1, lambda code: code == key.W)

        self.assertAlmostEqual(camera.pos.x, 0.2)
        self.assertAlmostEqual(camera.pos.y, 0.0)
        self.assertAlmostEqual(camera.pos.z, 0.0)


class TestViewerGLShouldStep(unittest.TestCase):
    """ViewerGL.should_step() state machine: running, paused, and single-step."""

    def test_returns_true_when_running(self):
        v = _make_gl_state(paused=False, step_requested=False)
        self.assertTrue(ViewerGL.should_step(v))

    def test_returns_false_when_paused(self):
        v = _make_gl_state(paused=True, step_requested=False)
        self.assertFalse(ViewerGL.should_step(v))

    def test_returns_true_once_after_step_request(self):
        v = _make_gl_state(paused=True, step_requested=True)
        self.assertTrue(ViewerGL.should_step(v))
        self.assertFalse(ViewerGL.should_step(v))

    def test_stale_request_cleared_when_running(self):
        # Reproduces the bug: . pressed while running, then SPACE to pause.
        # The flag must not survive into the paused state and fire a spurious step.
        v = _make_gl_state(paused=False, step_requested=True)
        ViewerGL.should_step(v)  # running frame — must clear the flag
        v._paused = True
        self.assertFalse(ViewerGL.should_step(v))

    def test_multiple_step_requests_fire_once_each(self):
        v = _make_gl_state(paused=True, step_requested=True)
        self.assertTrue(ViewerGL.should_step(v))
        v._step_requested = True
        self.assertTrue(ViewerGL.should_step(v))
        self.assertFalse(ViewerGL.should_step(v))


def _make_gl_running_state(headless: bool, num_frames: int | None, frame_count: int = 0) -> "ViewerGL":
    # Stand-in carrying only the fields ViewerGL.is_running()/end_frame() read,
    # so the frame budget can be exercised without a GL context.
    return SimpleNamespace(  # type: ignore[return-value]
        renderer=SimpleNamespace(has_exit=lambda: False),
        _headless=headless,
        num_frames=num_frames,
        _frame_count=frame_count,
        _update=lambda: None,
    )


class TestViewerGLFrameBudget(unittest.TestCase):
    """ViewerGL.is_running() honours num_frames in headless mode."""

    def test_headless_stops_once_num_frames_reached(self):
        """Verify headless rendering stops after num_frames frames."""
        v = _make_gl_running_state(headless=True, num_frames=3)
        for _ in range(3):
            self.assertTrue(ViewerGL.is_running(v))
            ViewerGL.end_frame(v)
        self.assertFalse(ViewerGL.is_running(v))

    def test_headless_without_num_frames_runs_unbounded(self):
        """Verify headless rendering is unbounded when num_frames is None."""
        v = _make_gl_running_state(headless=True, num_frames=None)
        for _ in range(5):
            ViewerGL.end_frame(v)
        self.assertTrue(ViewerGL.is_running(v))

    def test_windowed_ignores_num_frames(self):
        """Verify a visible window keeps running past num_frames."""
        v = _make_gl_running_state(headless=False, num_frames=1)
        for _ in range(3):
            ViewerGL.end_frame(v)
        self.assertTrue(ViewerGL.is_running(v))

    def test_window_close_stops_headless_run_early(self):
        """Verify an exit request wins over a remaining frame budget."""
        v = _make_gl_running_state(headless=True, num_frames=10)
        v.renderer.has_exit = lambda: True
        self.assertFalse(ViewerGL.is_running(v))

    def test_end_frame_counts_frames(self):
        """Verify end_frame() advances the frame counter used by the budget."""
        v = _make_gl_running_state(headless=True, num_frames=2)
        ViewerGL.end_frame(v)
        self.assertEqual(v._frame_count, 1)

    def test_zero_num_frames_stops_before_the_first_frame(self):
        """Verify a zero budget renders nothing at all."""
        v = _make_gl_running_state(headless=True, num_frames=0)
        self.assertFalse(ViewerGL.is_running(v))


class TestViewerGLNumFramesValidation(unittest.TestCase):
    """ViewerGL rejects num_frames values that would otherwise fail silently.

    The budget is applied as ``_frame_count < num_frames``, so a non-integer
    or negative value produces a surprising frame count rather than an error.
    These inputs are rejected before any GL context is created, so the tests
    need no display.
    """

    def test_rejects_non_integer_num_frames(self):
        """Verify a float num_frames raises TypeError rather than rendering a fractional budget."""
        with self.assertRaises(TypeError):
            ViewerGL(num_frames=1.5)  # type: ignore[arg-type]

    def test_rejects_bool_num_frames(self):
        """Verify a bool num_frames raises TypeError rather than being treated as 0 or 1."""
        with self.assertRaises(TypeError):
            ViewerGL(num_frames=True)

    def test_rejects_negative_num_frames(self):
        """Verify a negative num_frames raises ValueError rather than silently rendering nothing."""
        with self.assertRaises(ValueError):
            ViewerGL(num_frames=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
