# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import inspect
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest import mock

import numpy as np

from newton._src.viewer.gl.gui import UI
from newton._src.viewer.viewer_gl import ViewerGL
from newton._src.viewer.viewer_gui import ViewerGui
from newton._src.viewer.viewer_null import ViewerNull
from newton.tests.unittest_utils import patch_sys_module

_Vec3 = namedtuple("_Vec3", ("x", "y", "z"))


class TestViewerUI(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from imgui_bundle import imgui, imguizmo
        except ImportError as exc:
            self.skipTest(f"Viewer UI dependencies not available: {exc}")

        context = imgui.create_context()
        self.addCleanup(imgui.destroy_context, context)
        # Exercise ImGui state without an OpenGL renderer or window.
        self.ui = UI.__new__(UI)
        self.ui.imgui = imgui
        self.ui.giz = imguizmo.im_guizmo
        self.ui.io = imgui.get_io()
        self.ui.io.set_ini_filename("")
        self.ui.io.display_size = (640, 480)
        self.ui.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        self.ui.is_available = True
        self.ui._frame_time = None

    def test_frame_delta(self) -> None:
        """Advance ImGui time by elapsed frame durations, including a repeated timestamp."""
        elapsed = 0.0
        with mock.patch(
            "newton._src.viewer.gl.gui.perf_counter",
            side_effect=[0.0, 0.025, 0.050, 0.060, 0.060],
        ):
            for delta in (1.0 / 60.0, 0.025, 0.025, 0.010, 1e-6):
                with self.subTest(delta=delta):
                    self.ui.begin_frame()
                    try:
                        elapsed += delta
                        self.assertAlmostEqual(self.ui.io.delta_time, delta)
                        self.assertAlmostEqual(self.ui.imgui.get_time(), elapsed)
                    finally:
                        self.ui.imgui.end_frame()

    def test_dpi_scaling(self) -> None:
        """Keep style sizes stable on reapplication and restore them at scale one."""
        imgui = self.ui.imgui
        font = self.ui.io.fonts.add_font_default()
        self.ui._setup_dark_style()
        text_color = tuple(imgui.get_style().color_(imgui.Col_.text))
        for scale in (1.0, 2.0, 2.0, 1.5, 1.0):
            with self.subTest(scale=scale):
                expected_style = imgui.Style()
                expected_style.scale_all_sizes(scale)
                self.ui.dpi_scale = scale
                self.ui._apply_dpi_scaling()
                style = imgui.get_style()
                self.assertEqual(tuple(style.window_padding), (12.0 * scale, 12.0 * scale))
                # This size is not explicitly overridden by the Newton theme.
                self.assertEqual(tuple(style.separator_text_padding), tuple(expected_style.separator_text_padding))
                self.assertEqual(style.font_scale_dpi, scale)
                self.assertEqual(tuple(style.color_(imgui.Col_.text)), text_color)
                self.assertEqual(self.ui.io.fonts.fonts[0], font)


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


@unittest.skipUnless(importlib.util.find_spec("imgui_bundle"), "Requires imgui_bundle")
class TestViewerGuiRecovery(unittest.TestCase):
    def test_callback_failure_leaves_next_frame_usable(self):
        """Unwind real ImGui windows and styles without a renderer or model."""
        from imgui_bundle import imgui

        context = imgui.create_context()
        self.addCleanup(imgui.destroy_context, context)
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = (320, 320)
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = SimpleNamespace(show_ui=True)
        gui._loading_splash_active = False
        gui.ui = SimpleNamespace(
            is_available=True, imgui=imgui, io=io, begin_frame=imgui.new_frame, end_frame=imgui.render
        )
        error = RuntimeError("GUI callback failed")

        def fail():
            imgui.begin("Example")
            imgui.begin_child("Nested")
            imgui.push_style_color(imgui.Col_.text, (1.0, 0.0, 0.0, 1.0))
            raise error

        gui._render_ui = fail
        with self.assertRaises(RuntimeError) as raised:
            gui.prepare_frame(update_fps=False)
        self.assertIs(raised.exception, error)
        self.assertFalse(gui._frame_prepared)
        self.assertTrue(io.config_error_recovery_enable_assert)
        gui._render_ui = lambda: imgui.text("Retry")
        gui.prepare_frame(update_fps=False)
        self.assertTrue(gui._frame_prepared)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
