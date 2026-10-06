# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise rendering pause independently of simulation stepping."""

import importlib.util
import unittest
from time import perf_counter, sleep
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerNull, ViewerRTX


class TestRenderingPauseBase(unittest.TestCase):
    def test_unsupported_backend_rejects_pause(self):
        """Reject unsupported rendering pause without affecting simulation."""
        viewer = ViewerNull()
        self.assertFalse(viewer.is_rendering_paused())
        viewer.set_rendering_paused(False)
        with self.assertRaises(NotImplementedError):
            viewer.set_rendering_paused(True)
        self.assertTrue(viewer.should_step())


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestRenderingPauseRTX(unittest.TestCase):
    def setUp(self):
        """Create a viewer without initializing a renderer or a window."""
        with wp.ScopedDevice("cpu"), mock.patch.dict("sys.modules", {"ovrtx": mock.Mock()}):
            self.viewer = ViewerRTX(headless=True, num_frames=4)
        self.addCleanup(self.viewer.close)

    def test_pause_does_not_change_simulation_controls(self):
        """Preserve simulation pause and single-step behavior across render toggles."""
        viewer = self.viewer
        viewer.set_rendering_paused(True)
        self.assertTrue(viewer.is_rendering_paused())
        self.assertTrue(viewer.should_step())
        viewer._paused = True
        viewer._step_requested = True
        self.assertTrue(viewer.should_step())
        self.assertFalse(viewer.should_step())
        viewer.set_rendering_paused(False)
        self.assertTrue(viewer.is_paused())

    def test_pending_render_polls_both_phases_without_submitting(self):
        """Keep UI presentation responsive through operation and fetch timeouts."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._window = mock.Mock()
        viewer._present = mock.Mock()
        viewer._accept_render = mock.Mock()
        operation = mock.Mock()
        fetch = mock.Mock()
        operation.wait.side_effect = [None, fetch, fetch]
        fetch.fetch.side_effect = [None, {"stale": object()}]
        viewer._render_result = operation
        viewer._pending_render_generation = viewer._render_generation
        viewer.set_rendering_paused(True)
        for _ in range(3):
            viewer.end_frame()
        self.assertEqual(operation.wait.call_args_list, [mock.call(timeout_ns=0)] * 3)
        self.assertEqual(fetch.fetch.call_args_list, [mock.call(timeout_ns=0)] * 2)
        self.assertIsNone(viewer._render_result)
        viewer._rtx.step_async.assert_not_called()
        viewer._rtx.step.assert_not_called()
        viewer._accept_render.assert_not_called()
        self.assertEqual(viewer._present.call_count, 3)

    def test_quick_resume_never_accepts_pre_pause_result(self):
        """Discard old output even if rendering resumes before it completes."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer._update_scene = mock.Mock()
        viewer._accept_render = mock.Mock()
        operation = mock.Mock()
        operation.wait.return_value = None
        viewer._render_result = operation
        viewer._pending_render_generation = viewer._render_generation
        for _ in range(3):
            viewer.set_rendering_paused(True)
            viewer.set_rendering_paused(False)
        viewer.end_frame()
        viewer._update_scene.assert_not_called()
        operation.wait.return_value = mock.Mock()
        viewer.end_frame()
        viewer._accept_render.assert_not_called()
        viewer._update_scene.assert_called_once()
        viewer._rtx.step_async.assert_called_once()
        # Complete the new generation; cleanup must own no unfinished operation.
        viewer._collect_render(block=False)
        viewer._accept_render.assert_called_once()

    def test_paused_capture_uses_owned_pixels_and_requires_a_frame(self):
        """Capture the frozen image without consulting an outstanding operation."""
        viewer = self.viewer
        viewer.set_rendering_paused(True)
        with self.assertRaisesRegex(RuntimeError, "displayed frame"):
            viewer._capture_screenshot_pixels()
        expected = np.full((3, 4, 4), 37, dtype=np.uint8)
        viewer._displayed_pixels = wp.array(expected, dtype=wp.vec4ub, device="cpu")
        viewer._render_result = mock.Mock()
        np.testing.assert_array_equal(viewer._capture_screenshot_pixels(), expected)
        viewer._render_result.wait.assert_not_called()

    def test_pending_updates_survive_frames_and_budget_expires(self):
        """Retain one-off updates while headless pause still consumes its budget."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer.set_rendering_paused(True)
        viewer._pending_instance_visibility["/marker"] = False
        for i in range(4):
            self.assertTrue(viewer.is_running())
            viewer.begin_frame(float(i))
            viewer.end_frame()
        self.assertFalse(viewer.is_running())
        self.assertEqual(viewer._pending_instance_visibility, {"/marker": False})
        self.assertEqual(viewer._rtx.mock_calls, [])

    def test_runtime_geometry_defers_until_resume(self):
        """Coalesce marker additions and resize operations without calling OVRTX."""
        viewer = self.viewer
        viewer._phase = viewer._PHASE_RENDER
        viewer._rtx = mock.Mock()
        viewer.set_rendering_paused(True)
        for count in (1, 3, 2):
            viewer.begin_frame(0.0)
            viewer.log_shapes(
                "/markers",
                newton.GeoType.SPHERE,
                0.1,
                wp.array([wp.transform_identity()] * count, dtype=wp.transform, device="cpu"),
            )
            viewer.end_frame()
        self.assertEqual(viewer._rtx.mock_calls, [])
        self.assertEqual(len(viewer._instance_prim_paths["/markers"]), 2)
        self.assertEqual(len(viewer._deferred_prims), 2)  # prototype mesh and instance batch
        viewer.set_rendering_paused(False)
        with mock.patch.dict("sys.modules", {"ovrtx": mock.Mock()}):
            viewer._flush_deferred_prims()
        self.assertFalse(viewer._deferred_prims)
        self.assertEqual(viewer._rtx.add_usd_reference_from_string.call_count, 2)

    def test_clear_and_close_drain_pending_work(self):
        """Release each operation before renderer teardown and invalidate capture."""
        viewer = self.viewer
        viewer.set_rendering_paused(True)
        viewer._displayed_pixels = wp.zeros((2, 2), dtype=wp.vec4ub, device="cpu")
        operation = mock.Mock()
        viewer._render_result = operation
        viewer.clear_model()
        operation.wait.assert_called_once_with()
        operation.wait.return_value.fetch.assert_called_once_with()
        self.assertTrue(viewer.is_rendering_paused())
        self.assertIsNone(viewer._displayed_pixels)
        self.assertIsNone(viewer._render_result)
        operation = mock.Mock()
        viewer._render_result = operation
        viewer.close()
        viewer.close()
        operation.wait.assert_called_once_with()


class TestRenderingPauseGL(unittest.TestCase):
    def test_freeze_capture_resize_and_resume(self):
        """Keep an image stable through logging, resize, and UI-driven toggles."""
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self)
        self.addCleanup(viewer.close)
        before = np.zeros((48, 64, 3), dtype=np.uint8)
        before[:24, :, 0] = 255
        before[24:, :, 1] = 255
        after = np.zeros_like(before)
        after[:, :, 2] = 255

        viewer.set_rendering_paused(True)
        viewer.begin_frame(0.0)
        viewer.end_frame()
        with self.assertRaisesRegex(RuntimeError, "displayed frame"):
            viewer.get_frame()
        viewer.set_rendering_paused(False)
        viewer.begin_frame(1.0)
        viewer.log_image("sensor", before, fullscreen=True)
        viewer.end_frame()
        reference = viewer.get_frame().numpy().copy()
        np.testing.assert_array_equal(reference, before)

        gui = mock.Mock()
        gui.prepare_frame.side_effect = lambda: viewer.set_rendering_paused(True)
        viewer.gui = gui
        viewer.begin_frame(2.0)
        viewer.log_image("sensor", after, fullscreen=True)
        with mock.patch.object(viewer.renderer, "render", wraps=viewer.renderer.render) as render:
            viewer.end_frame()
            render.assert_not_called()
        np.testing.assert_array_equal(viewer.get_frame().numpy(), reference)
        np.testing.assert_array_equal(viewer._displayed_frame.pixels()[::-1, :, :3], reference)
        gui.render_prepared_frame.assert_called_once()
        viewer.gui = None

        viewer.renderer.window.set_size(128, 96)
        viewer.renderer._on_window_resize(128, 96)
        viewer.begin_frame(3.0)
        viewer.end_frame()
        resized = viewer.get_frame().numpy()
        self.assertEqual(resized.shape, (96, 128, 3))
        np.testing.assert_array_equal(resized[10, 10], [255, 0, 0])
        np.testing.assert_array_equal(resized[80, 10], [0, 255, 0])

        viewer.set_rendering_paused(False)
        viewer.begin_frame(4.0)
        viewer.log_image("sensor", after, fullscreen=True)
        viewer.end_frame()
        np.testing.assert_array_equal(viewer.get_frame().numpy()[10, 10], [0, 0, 255])
        viewer.set_rendering_paused(True)
        viewer.clear_model()
        viewer.begin_frame(5.0)
        viewer.end_frame()
        self.assertTrue(viewer.is_rendering_paused())
        with self.assertRaises(RuntimeError):
            viewer.get_frame()


@unittest.skipUnless(
    USD_AVAILABLE and importlib.util.find_spec("ovrtx") is not None and wp.is_cuda_available(),
    "Requires the rtx extra and a CUDA device",
)
class TestRenderingPauseRTXIntegration(unittest.TestCase):
    def test_moving_scene_capture_stays_frozen_until_resume(self):
        """Verify frozen pixels and resumed motion with real sync and async RTX."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.3, hy=0.3, hz=0.3, color=(1.0, 0.1, 0.0))
        model = builder.finalize()
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                viewer = ViewerRTX(width=64, height=48, headless=True, async_rendering=asynchronous)
                try:
                    viewer.set_model(model)
                    viewer.set_camera(wp.vec3(3.0, -4.0, 2.0), pitch=-20.0, yaw=125.0)
                    state = model.state()
                    for i in range(3):
                        viewer.begin_frame(i / 60.0)
                        viewer.log_state(state)
                        viewer.end_frame()
                    viewer.set_rendering_paused(True)
                    frozen = viewer._capture_screenshot_pixels().copy()
                    self.assertGreater(np.ptp(frozen[:, :, :3]), 0)
                    for i in range(3, 6):
                        state.body_q.assign([wp.transform(wp.vec3(0.5 * i, 0.0, 0.0), wp.quat_identity())])
                        viewer.begin_frame(i / 60.0)
                        viewer.log_state(state)
                        viewer.end_frame()
                        np.testing.assert_array_equal(viewer._capture_screenshot_pixels(), frozen)
                    viewer.set_rendering_paused(False)
                    deadline = perf_counter() + 10.0
                    i = 6
                    while perf_counter() < deadline:
                        viewer.begin_frame(i / 60.0)
                        viewer.log_state(state)
                        viewer.end_frame()
                        if not np.array_equal(viewer._capture_screenshot_pixels(), frozen):
                            break
                        sleep(0.01)
                        i += 1
                    self.assertFalse(np.array_equal(viewer._capture_screenshot_pixels(), frozen))
                finally:
                    viewer.close()


if __name__ == "__main__":
    unittest.main()
