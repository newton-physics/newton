# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest import mock

import newton.examples
from newton._src.viewer.viewer_gl import ViewerGL


class TestViewerGLLoadingSplashAPIShape(unittest.TestCase):
    """Pin ``ViewerGL``'s splash methods by name.

    ``newton.examples.run`` probes for the method via
    ``hasattr(viewer, "hide_loading_splash")``.  If the method is ever
    renamed on ``ViewerGL`` and the call site is not updated together,
    ``hasattr`` would silently flip to ``False`` and the splash would
    linger across the entire example run with no diagnostic — none of
    the integration tests would notice because they use
    ``_RecordingViewer`` (which always defines the method by name).
    """

    def test_show_loading_splash_exists(self):
        self.assertTrue(callable(getattr(ViewerGL, "show_loading_splash", None)))

    def test_hide_loading_splash_exists(self):
        self.assertTrue(callable(getattr(ViewerGL, "hide_loading_splash", None)))


class TestViewerGLLoadingSplashState(unittest.TestCase):
    """``show_loading_splash`` / ``hide_loading_splash`` mutate ViewerGL state correctly.

    Constructed via ``__new__`` so the test does not require a GL context;
    only the state-machine behavior is exercised.
    """

    def _make_viewer(self):
        # Deliberately bypass ``ViewerGL.__init__`` (which would open a
        # GL window) and hand-initialize only the state the splash API
        # touches.  Do not call ``__init__`` here.
        viewer = ViewerGL.__new__(ViewerGL)
        viewer._loading_splash_active = False
        viewer._loading_splash_text = None
        return viewer

    def test_show_sets_active_and_text(self):
        viewer = self._make_viewer()
        viewer.show_loading_splash("Loading basic_pendulum...")
        self.assertTrue(viewer._loading_splash_active)
        self.assertEqual(viewer._loading_splash_text, "Loading basic_pendulum...")

    def test_show_without_text_sets_active_only(self):
        viewer = self._make_viewer()
        viewer.show_loading_splash()
        self.assertTrue(viewer._loading_splash_active)
        self.assertIsNone(viewer._loading_splash_text)

    def test_hide_clears_state(self):
        viewer = self._make_viewer()
        viewer.show_loading_splash("Loading...")
        viewer.hide_loading_splash()
        self.assertFalse(viewer._loading_splash_active)
        self.assertIsNone(viewer._loading_splash_text)

    def test_show_replaces_text(self):
        viewer = self._make_viewer()
        viewer.show_loading_splash("first")
        viewer.show_loading_splash("second")
        self.assertEqual(viewer._loading_splash_text, "second")

    def test_hide_is_idempotent(self):
        # ``run()`` calls ``hide_loading_splash`` unconditionally, even when
        # no splash was raised — exercise that hot path.
        viewer = self._make_viewer()
        viewer.hide_loading_splash()
        viewer.hide_loading_splash()
        self.assertFalse(viewer._loading_splash_active)


class _FakeImGui:
    """Minimal stand-in for the ImGui module used by ``_render_loading_splash``.

    Records every draw-list call so tests can assert on counts and on the
    geometric layout (the leftmost ball must end up offset from its pivot).
    """

    class _Vec:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __iter__(self):
            yield self.x
            yield self.y

    class _DrawList:
        def __init__(self):
            self.rects = []
            self.lines = []
            self.circles = []
            self.texts = []

        def add_rect_filled(self, p1, p2, col):
            self.rects.append((tuple(p1), tuple(p2), col))

        def add_line(self, p1, p2, col, thickness):
            self.lines.append((tuple(p1), tuple(p2), col, thickness))

        def add_circle_filled(self, center, radius, col):
            self.circles.append((tuple(center), radius, col))

        def add_text(self, pos, col, text):
            self.texts.append((tuple(pos), col, text))

    class _WindowFlags:
        # Arbitrary bitmask sentinels.  These are not the real ImGui flag
        # values; they only need to be distinct ints so the production
        # code's ``OR`` builds a unique mask the fake can echo back.
        no_decoration = 1 << 0
        no_inputs = 1 << 1
        no_saved_settings = 1 << 2
        no_focus_on_appearing = 1 << 3
        no_nav = 1 << 4
        no_bring_to_front_on_focus = 1 << 5
        no_move = 1 << 6
        no_background = 1 << 7

    def __init__(
        self,
        viewport_size=(800, 600),
        viewport_pos=(0.0, 0.0),
        font_size=13.0,
        begin_returns_open=True,
    ):
        self.ImVec2 = self._Vec
        self.ImVec4 = lambda r, g, b, a: (r, g, b, a)
        self.WindowFlags_ = self._WindowFlags
        self._draw_list = self._DrawList()
        self._viewport = SimpleNamespace(
            pos=self._Vec(*viewport_pos),
            size=self._Vec(*viewport_size),
        )
        self._font_size = font_size
        self._begin_returns_open = begin_returns_open
        self.begin_calls = []  # list of (name, flags)
        self.end_calls = 0

    def get_main_viewport(self):
        return self._viewport

    def get_font_size(self):
        return self._font_size

    def color_convert_float4_to_u32(self, c):
        return c

    def calc_text_size(self, text):
        return self._Vec(7.0 * len(text), 13.0)

    def set_next_window_pos(self, pos):
        pass

    def set_next_window_size(self, size):
        pass

    def begin(self, name, p_open, flags):
        self.begin_calls.append((name, flags))
        return (self._begin_returns_open, self._begin_returns_open)

    def end(self):
        self.end_calls += 1

    def get_window_draw_list(self):
        return self._draw_list


class TestRenderLoadingSplash(unittest.TestCase):
    """Direct coverage of the ImGui draw routine without a GL context."""

    def _viewer_with_fake_ui(self, *, active, text=None, viewport=(800, 600), viewport_pos=(0.0, 0.0)):
        viewer = ViewerGL.__new__(ViewerGL)
        viewer._loading_splash_active = active
        viewer._loading_splash_text = text
        fake_imgui = _FakeImGui(viewport_size=viewport, viewport_pos=viewport_pos)
        viewer.ui = SimpleNamespace(imgui=fake_imgui)
        return viewer, fake_imgui

    def test_inactive_short_circuits(self):
        viewer, imgui = self._viewer_with_fake_ui(active=False)
        viewer._render_loading_splash()
        self.assertEqual(imgui.begin_calls, [])
        self.assertEqual(imgui._draw_list.circles, [])

    def test_no_ui_short_circuits(self):
        viewer = ViewerGL.__new__(ViewerGL)
        viewer._loading_splash_active = True
        viewer._loading_splash_text = "Loading..."
        viewer.ui = None
        viewer._render_loading_splash()  # must not raise

    def test_active_renders_dim_bar_strings_and_balls(self):
        viewer, imgui = self._viewer_with_fake_ui(active=True, text="Loading...", viewport=(800, 600))
        viewer._render_loading_splash()

        self.assertEqual([name for name, _ in imgui.begin_calls], ["##loading_splash"])
        self.assertEqual(imgui.end_calls, 1)
        # 1 dim rect + 1 cradle bar
        self.assertEqual(len(imgui._draw_list.rects), 2)
        # 5 strings, 5 balls
        self.assertEqual(len(imgui._draw_list.lines), 5)
        self.assertEqual(len(imgui._draw_list.circles), 5)
        self.assertEqual(len(imgui._draw_list.texts), 1)

        # The window flags must mark the splash as non-interactive and
        # transparent so it does not steal focus or paint a styled window
        # background over the dim rect.
        flags = imgui.begin_calls[0][1]
        wf = imgui.WindowFlags_
        for required in (wf.no_inputs, wf.no_move, wf.no_background, wf.no_decoration):
            self.assertTrue(flags & required, f"missing flag bit {required:#x} in {flags:#x}")

        # The dim rect must cover the full viewport so the underlying
        # scene does not leak through alongside the cradle.
        rects_by_area = sorted(
            imgui._draw_list.rects,
            key=lambda r: (r[1][0] - r[0][0]) * (r[1][1] - r[0][1]),
            reverse=True,
        )
        dim_p1, dim_p2, _ = rects_by_area[0]
        self.assertEqual(dim_p1, (0.0, 0.0))
        self.assertEqual(dim_p2, (800.0, 600.0))

    def test_dim_rect_honors_nonzero_viewport_origin(self):
        # On multi-monitor setups ImGui's main viewport sits at non-zero
        # ``pos``.  A regression that dropped the ``+ viewport.pos`` term
        # would still draw a dim rect at (0, 0) that visually covers the
        # primary monitor — guard the offset explicitly.
        viewer, imgui = self._viewer_with_fake_ui(active=True, viewport=(800, 600), viewport_pos=(150.0, 75.0))
        viewer._render_loading_splash()

        rects_by_area = sorted(
            imgui._draw_list.rects,
            key=lambda r: (r[1][0] - r[0][0]) * (r[1][1] - r[0][1]),
            reverse=True,
        )
        dim_p1, dim_p2, _ = rects_by_area[0]
        self.assertEqual(dim_p1, (150.0, 75.0))
        self.assertEqual(dim_p2, (950.0, 675.0))

    def test_collapsed_window_skips_drawing_but_ends_frame(self):
        # ImGui's ``begin`` returns ``(False, ...)`` when the window is
        # collapsed/clipped (e.g. user style with oversized
        # ``WindowMinSize``).  All draws are gated by that return, but
        # ``end()`` must still run unconditionally — otherwise the
        # begin/end pair is unbalanced and ImGui asserts.
        viewer = ViewerGL.__new__(ViewerGL)
        viewer._loading_splash_active = True
        viewer._loading_splash_text = "Loading..."
        fake_imgui = _FakeImGui(begin_returns_open=False)
        viewer.ui = SimpleNamespace(imgui=fake_imgui)
        viewer._render_loading_splash()

        self.assertEqual(fake_imgui.end_calls, 1)
        self.assertEqual(fake_imgui._draw_list.rects, [])
        self.assertEqual(fake_imgui._draw_list.lines, [])
        self.assertEqual(fake_imgui._draw_list.circles, [])
        self.assertEqual(fake_imgui._draw_list.texts, [])

    def test_text_omitted_when_none(self):
        viewer, imgui = self._viewer_with_fake_ui(active=True, text=None)
        viewer._render_loading_splash()
        self.assertEqual(imgui._draw_list.texts, [])

    def test_leftmost_ball_is_lifted(self):
        # The cradle visual depends on ball 0 being offset to the *left*
        # of its pivot (the swing) while the rest hang straight.  This
        # locks in the geometric intent.
        viewer, imgui = self._viewer_with_fake_ui(active=True, text=None)
        viewer._render_loading_splash()

        balls = imgui._draw_list.circles  # list of ((cx, cy), r, col)
        ball0_x = balls[0][0][0]
        ball1_x = balls[1][0][0]
        # ball 1 hangs straight — its x equals its pivot x.  ball 0's pivot
        # is one ball-spacing to the left of ball 1's, and a lifted ball
        # is offset further left by sin(swing_angle) * string_length.
        self.assertLess(ball0_x, ball1_x)
        # all "resting" balls share a y; the lifted ball is higher (smaller y)
        ball0_y = balls[0][0][1]
        rest_ys = {balls[i][0][1] for i in range(1, 5)}
        self.assertEqual(len(rest_ys), 1)
        self.assertLess(ball0_y, next(iter(rest_ys)))

    def test_scales_with_font_size(self):
        # Doubling the font size should roughly double the ball radius.
        small_viewer, small_imgui = self._viewer_with_fake_ui(active=True)
        small_imgui._font_size = 13.0
        small_viewer._render_loading_splash()
        small_radius = small_imgui._draw_list.circles[0][1]

        big_viewer, big_imgui = self._viewer_with_fake_ui(active=True)
        big_imgui._font_size = 26.0
        big_viewer._render_loading_splash()
        big_radius = big_imgui._draw_list.circles[0][1]

        self.assertAlmostEqual(big_radius, small_radius * 2.0, places=4)


class _RecordingViewer:
    """Stub viewer that records every observable call in order, for ordering tests."""

    def __init__(self):
        self.calls = []
        self._running = False

    def show_loading_splash(self, text=None):
        self.calls.append(("show_loading_splash", text))

    def hide_loading_splash(self):
        self.calls.append(("hide_loading_splash",))

    def begin_frame(self, t):
        self.calls.append(("begin_frame", t))

    def end_frame(self):
        self.calls.append(("end_frame",))

    def is_running(self):
        self.calls.append(("is_running",))
        return self._running

    def is_paused(self):
        return False

    def close(self):
        self.calls.append(("close",))


class TestInitTriggersLoadingSplash(unittest.TestCase):
    """``init()`` shows the splash for visible GL viewers and skips otherwise."""

    def _args(self, **overrides):
        defaults = {
            "viewer": "gl",
            "headless": False,
            "device": None,
            "quiet": True,
            "warp_config": [],
            "benchmark": False,
            "realtime": False,
            "output_path": None,
            "num_frames": 1,
            "rerun_address": None,
            "test": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _run_init_with_stub(self, args, viewer_attr="ViewerGL"):
        stub = _RecordingViewer()
        parser = mock.MagicMock()
        parser.parse_args.return_value = args
        with (
            mock.patch(f"newton.viewer.{viewer_attr}", return_value=stub),
            mock.patch("newton.examples._apply_warp_config"),
        ):
            viewer, _ = newton.examples.init(parser=parser)
        return viewer

    def test_visible_gl_shows_splash_before_pumping_frames(self):
        viewer = self._run_init_with_stub(self._args())
        kinds = [c[0] for c in viewer.calls]
        # show is the first observable call; the three frame pumps follow.
        self.assertEqual(kinds[0], "show_loading_splash")
        self.assertEqual(viewer.calls[0], ("show_loading_splash", "Loading..."))
        # exactly three begin/end pairs follow, in order
        self.assertEqual(
            kinds[1:],
            ["begin_frame", "end_frame"] * 3,
        )

    def test_headless_skips_splash_and_pumps(self):
        viewer = self._run_init_with_stub(self._args(headless=True))
        self.assertEqual(viewer.calls, [])

    def test_non_gl_viewer_skips_splash_and_pumps(self):
        # The ``visible_gl`` gate is ``args.viewer == "gl" and not headless``.
        # A regression dropping the viewer check would silently pump
        # frames through a USD/Rerun/Null/Viser viewer.
        for viewer_kind, viewer_attr, extra in [
            ("usd", "ViewerUSD", {"output_path": "/tmp/out.usd"}),
            ("rerun", "ViewerRerun", {}),
            ("null", "ViewerNull", {}),
            ("viser", "ViewerViser", {}),
        ]:
            with self.subTest(viewer=viewer_kind):
                viewer = self._run_init_with_stub(self._args(viewer=viewer_kind, **extra), viewer_attr=viewer_attr)
                self.assertEqual(viewer.calls, [])


class _OneShotRunningViewer(_RecordingViewer):
    """``is_running`` returns True for one iteration, then False.

    Lets the ``run()`` loop body execute exactly once so we can assert
    that ``hide_loading_splash`` ran before step/render fired.
    """

    def __init__(self):
        super().__init__()
        self._iterations_left = 1

    def is_running(self):
        self.calls.append(("is_running",))
        if self._iterations_left > 0:
            self._iterations_left -= 1
            return True
        return False


class TestRunHidesLoadingSplash(unittest.TestCase):
    """``run()`` lowers the splash before entering the main loop."""

    def test_hide_runs_before_first_is_running(self):
        viewer = _RecordingViewer()
        step_calls = []
        render_calls = []
        example = SimpleNamespace(
            viewer=viewer,
            step=lambda: step_calls.append(("step",)),
            render=lambda: render_calls.append(("render",)),
        )
        args = SimpleNamespace(test=False)
        newton.examples.run(example, args)

        kinds = [c[0] for c in viewer.calls]
        self.assertIn("hide_loading_splash", kinds)
        self.assertIn("close", kinds)
        # hide_loading_splash must precede the first is_running check
        # (which gates the step/render loop).
        self.assertLess(kinds.index("hide_loading_splash"), kinds.index("is_running"))
        # is_running returned False, so step/render never ran.
        self.assertEqual(step_calls, [])
        self.assertEqual(render_calls, [])

    def test_hide_runs_before_step_and_render_when_loop_executes(self):
        # Stronger guard than the no-loop test: a regression that moved
        # ``hide_loading_splash`` *into* the loop body (e.g. after the
        # first ``step`` call) would still pass the simpler test, but
        # would fail this one because hide must precede step/render.
        viewer = _OneShotRunningViewer()
        recorded = []
        example = SimpleNamespace(
            viewer=viewer,
            step=lambda: recorded.append("step"),
            render=lambda: recorded.append("render"),
        )
        # Cooperate with the ordering check by recording the splash
        # event into the same timeline.
        original_hide = viewer.hide_loading_splash

        def hide_and_record():
            recorded.append("hide_loading_splash")
            original_hide()

        viewer.hide_loading_splash = hide_and_record

        args = SimpleNamespace(test=False)
        newton.examples.run(example, args)

        self.assertIn("hide_loading_splash", recorded)
        self.assertIn("step", recorded)
        self.assertIn("render", recorded)
        self.assertLess(recorded.index("hide_loading_splash"), recorded.index("step"))
        self.assertLess(recorded.index("hide_loading_splash"), recorded.index("render"))


if __name__ == "__main__":
    unittest.main()
