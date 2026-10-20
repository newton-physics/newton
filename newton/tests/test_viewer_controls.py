# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton import Axis
from newton._src.viewer.camera import Camera
from newton._src.viewer.viewer_gl import ViewerGL
from newton._src.viewer.viewer_gui import ViewerGui
from newton._src.viewer.viewer_null import ViewerNull
from newton.tests.unittest_utils import patch_sys_module

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
        viewer = SimpleNamespace(camera=camera, camera_speed=2.0, is_rendering_paused=lambda: False)
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.ui = None
        gui._cam_vel = np.zeros(3, dtype=np.float32)
        gui._cam_damp_tau = 0.1

        key = SimpleNamespace(W=1, UP=2, S=3, DOWN=4, A=5, LEFT=6, D=7, RIGHT=8, Q=9, E=10)
        pyglet = SimpleNamespace(window=SimpleNamespace(key=key))
        with patch_sys_module("pyglet", pyglet):
            gui.update_camera_from_keys(0.1, lambda code: code == key.W)

        self.assertAlmostEqual(camera.pos.x, 0.2)
        self.assertAlmostEqual(camera.pos.y, 0.0)
        self.assertAlmostEqual(camera.pos.z, 0.0)

    def test_camera_deceleration_is_capped(self):
        """Check that the camera deceleration is capped to avoid flipping the camera velocity direction."""
        camera = SimpleNamespace(
            pos=_Vec3(0.0, 0.0, 0.0),
            get_front=lambda: (1.0, 0.0, 0.0),
            get_right=lambda: (0.0, 1.0, 0.0),
            get_up=lambda: (0.0, 0.0, 1.0),
        )
        viewer = SimpleNamespace(camera=camera, camera_speed=2.0, is_rendering_paused=lambda: False)
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.ui = None
        velocity_init = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Non-zero initial velocity
        gui._cam_vel = velocity_init.copy()
        gui._cam_damp_tau = 0.1

        key = SimpleNamespace(W=1, UP=2, S=3, DOWN=4, A=5, LEFT=6, D=7, RIGHT=8, Q=9, E=10)
        pyglet = SimpleNamespace(window=SimpleNamespace(key=key))
        with patch_sys_module("pyglet", pyglet):
            gui.update_camera_from_keys(2.0 * gui._cam_damp_tau, lambda _: False)

        self.assertGreaterEqual(np.dot(camera.pos, velocity_init), 0.0)


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

    def test_rejects_invalid_cuda_interop_mode(self):
        """Reject CUDA interop values outside the public flag enum."""
        for value in (True, 1, 1.5, "dynamic"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                ViewerGL(enable_cuda_interop=value)  # type: ignore[arg-type]

    def test_cuda_interop_defaults_to_dynamic_meshes(self):
        """Enable CUDA interop for dynamic meshes by default."""
        parameter = inspect.signature(ViewerGL).parameters["enable_cuda_interop"]
        self.assertEqual(parameter.default, ViewerGL.CudaInterop.DYNAMIC_MESH)

    def test_cuda_interop_flags_are_composable(self):
        """Compose independent CUDA interop categories with bitwise flags."""
        flags = ViewerGL.CudaInterop.POINTS | ViewerGL.CudaInterop.LINES
        self.assertTrue(flags & ViewerGL.CudaInterop.POINTS)
        self.assertTrue(flags & ViewerGL.CudaInterop.LINES)
        self.assertFalse(flags & ViewerGL.CudaInterop.INSTANCES)
        self.assertEqual(
            ViewerGL.CudaInterop.ALL,
            ViewerGL.CudaInterop.DYNAMIC_MESH
            | ViewerGL.CudaInterop.STATIC_MESH
            | ViewerGL.CudaInterop.POINTS
            | ViewerGL.CudaInterop.INSTANCES
            | ViewerGL.CudaInterop.LINES,
        )

    def test_rejects_unknown_cuda_interop_flags(self):
        """Reject flag bits outside the supported CUDA interop categories."""
        with self.assertRaises(ValueError):
            ViewerGL(enable_cuda_interop=ViewerGL.CudaInterop(1 << 10))


class TestViewerGizmos(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from imgui_bundle import imgui, imguizmo
        except ImportError:
            self.skipTest("imgui_bundle is required")

        self.imgui = imgui
        self.giz = imguizmo.im_guizmo
        context = imgui.create_context()
        self.addCleanup(imgui.destroy_context, context)
        self.io = imgui.get_io()
        self.io.set_ini_filename("")
        self.io.display_size = (800, 600)
        self.io.delta_time = 1.0 / 60.0
        # Build draw lists without a window or GPU renderer.
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        self.viewer = ViewerGL.__new__(ViewerGL)
        self.viewer.camera = Camera(width=800, height=600, pos=(0.0, 0.0, 5.0), up_axis="Y")
        self.viewer.camera.yaw = -90.0
        self.viewer._rendering_paused = False
        self.viewer._gizmo_log = {}
        self.viewer.gizmo_is_using = False
        self.viewer.show_ui = True
        self.gui = ViewerGui.__new__(ViewerGui)
        self.gui._viewer = self.viewer
        self.gui.ui = SimpleNamespace(is_available=True, io=self.io, giz=self.giz)
        self.gui._gizmo_active = {}
        self.gui._gizmo_hovered = False
        self.gui._camera_dolly_scroll_sensitivity = 0.15
        # ImGuizmo owns global drag state; always release it before destroying ImGui.
        self.addCleanup(self.frame, 0.0, 0.0, False)
        self.frame(0.0, 0.0, False)

    def frame(self, x: float, y: float, down: bool) -> None:
        self.io.add_mouse_pos_event(x, y)
        self.io.add_mouse_button_event(0, down)
        self.imgui.new_frame()
        self.giz.begin_frame()
        try:
            self.gui._render_gizmos()
        finally:
            self.imgui.render()

    def test_drag_and_snap_with_multiple_operations(self) -> None:
        """Capture handles, preserve drags across idle operations, and snap only on release."""
        for translate in (None, (Axis.X, Axis.Y)):
            with self.subTest(translate=translate):
                target = wp.transform((0.0, 0.0, 0.0), wp.quat_identity())
                snap = wp.transform((0.1, 0.2, 0.3), wp.quat_identity())
                other = wp.transform((1.0, 1.0, 0.0), wp.quat_identity())
                self.viewer._gizmo_log.clear()
                self.viewer.log_gizmo("target", target, translate=translate, snap_to=snap)
                self.viewer.log_gizmo("other", other)
                self.frame(0.0, 0.0, False)
                # The X arrow is right of the origin at (400, 300).
                self.frame(420.0, 300.0, False)
                camera_pos = tuple(self.viewer.camera.pos)
                self.gui.handle_mouse_scroll(1.0)
                self.assertEqual(tuple(self.viewer.camera.pos), camera_pos)
                self.assertTrue(self.gui.is_mouse_capturing())

                self.frame(420.0, 300.0, True)
                self.frame(440.0, 300.0, True)
                self.assertTrue(self.viewer.gizmo_is_using)
                self.assertTrue(self.gui._gizmo_active["target"])
                self.assertFalse(self.gui._gizmo_active["other"])
                first_x = float(target[0])
                self.assertGreater(first_x, 0.0)
                np.testing.assert_allclose(np.asarray(target)[1:], (0.0, 0.0, 0.0, 0.0, 0.0, 1.0), atol=1e-6)
                np.testing.assert_array_equal(other, (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))

                # Stable names must survive a different logging order.
                self.viewer._gizmo_log = dict(reversed(tuple(self.viewer._gizmo_log.items())))
                self.frame(440.0, 300.0, True)
                self.assertAlmostEqual(target[0], first_x, places=5)
                self.assertTrue(self.gui._gizmo_active["target"])
                self.frame(460.0, 300.0, True)
                self.assertGreater(target[0], first_x)
                self.frame(460.0, 300.0, False)
                np.testing.assert_array_equal(target, snap)
                self.assertFalse(self.viewer.gizmo_is_using)
                self.assertFalse(self.gui._gizmo_active["target"])

                # A later idle frame must not snap again.
                snap[0] = 2.0
                self.frame(0.0, 0.0, False)
                self.assertAlmostEqual(target[0], 0.1)
                self.viewer._gizmo_log.clear()
                self.frame(0.0, 0.0, False)
                self.frame(0.0, 0.0, False)
                self.assertEqual(self.gui._gizmo_active, {})
                self.assertFalse(self.gui.is_mouse_capturing())

    def test_rotation_with_restricted_axes(self) -> None:
        """Rotate around the enabled axis and snap after releasing the ring."""
        target = wp.transform((0.0, 0.0, 0.0), wp.quat_identity())
        snap = wp.transform((0.1, 0.2, 0.3), wp.quat_identity())
        self.viewer.log_gizmo("target", target, translate=(Axis.X, Axis.Y), rotate=(Axis.Z,), snap_to=snap)
        # The Z ring crosses the upper-right quadrant around the origin.
        self.frame(420.0, 280.0, False)
        self.assertTrue(self.gui.is_mouse_capturing())
        self.frame(420.0, 280.0, True)
        self.frame(430.0, 300.0, True)
        self.assertTrue(self.gui._gizmo_active["target"])
        np.testing.assert_allclose(np.asarray(target)[:5], 0.0, atol=1e-6)
        self.assertGreater(abs(target[5]), 0.1)
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(target)[3:])), 1.0, places=5)
        self.frame(430.0, 300.0, False)
        np.testing.assert_array_equal(target, snap)
        self.assertFalse(self.viewer.gizmo_is_using)

    def test_idle_transform_is_unchanged(self) -> None:
        """Preserve the caller's quaternion representation when no handle modifies it."""
        target = wp.transform((0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, -1.0))
        original = np.asarray(target).copy()
        # Also exercise the legacy raw-transform entry.
        self.viewer._gizmo_log["target"] = target
        self.frame(0.0, 0.0, False)
        np.testing.assert_array_equal(target, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
