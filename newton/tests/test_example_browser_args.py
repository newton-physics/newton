# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the example-browser switch/reset args plumbing.

The companion :mod:`newton.tests.test_example_browser` script runs every
registered example through the browser using a real GL viewer; it is a
manual smoke test and does not contain any auto-discovered ``TestCase``
classes.
"""

import types
import unittest
from unittest.mock import patch

import newton.examples
from newton.examples import _ExampleBrowser
from newton.viewer import ViewerNull


class _StubViewer:
    """Minimal viewer stub for exercising _ExampleBrowser without a UI."""

    def __init__(self):
        self.cleared = 0
        self.cleared_all_layers = 0

    def clear_all_layers(self):
        self.cleared_all_layers += 1

    def clear_model(self):
        self.cleared += 1


class _StubExample:
    """Captures the args namespace passed by the example browser."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        return parser


class _OtherStubExample:
    """Second stub example with a different parser default to exercise switch()."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=7)
        return parser


class TestExampleBrowserReset(unittest.TestCase):
    def test_runtime_failures_keep_browser_running(self):
        for stage, error in (
            ("bind", RuntimeError("history failed")),
            ("step", RuntimeError("solver failed")),
            ("render", RuntimeError("render failed")),
            ("step", SystemExit(2)),
        ):
            with self.subTest(stage=stage, error=type(error).__name__):
                self._assert_runtime_failure_recovers(stage, error)

    def _assert_runtime_failure_recovers(self, stage, error):
        viewer = _StubViewer()
        viewer._paused = False
        viewer._step_requested = True
        viewer.frames = viewer.polls = 0
        viewer.closed = False
        viewer.begin_frame = lambda time: None
        viewer.end_frame = lambda: setattr(viewer, "frames", viewer.frames + 1)
        viewer.should_step = lambda: not viewer._paused
        viewer.close = lambda: setattr(viewer, "closed", True)
        browser = _ExampleBrowser(viewer)

        class GoodExample(_StubExample):
            def step(self):
                pass

            def render(self):
                self.viewer.end_frame()

        class FailedExample(GoodExample):
            def step(self):
                if stage == "step":
                    raise error

            def render(self):
                if stage == "render":
                    raise error

        def bind(example):
            if stage == "bind" and isinstance(example, FailedExample):
                raise error

        viewer.gui = types.SimpleNamespace(_workspace_example=None)
        viewer.gui.workspace = types.SimpleNamespace(
            bind=bind,
            before_step=lambda: True,
            render_example=lambda: viewer.gui._workspace_example.render(),  # noqa: PLW0108
        )

        def is_running():
            viewer.polls += 1
            if viewer.polls == 3:
                browser.switch_target = "fake.recovered"
            return viewer.polls <= 5

        viewer.is_running = is_running
        args = _StubExample.create_parser().parse_args([])
        with (
            patch("newton.examples._ExampleBrowser", return_value=browser),
            patch(
                "newton.examples.importlib.import_module",
                return_value=types.SimpleNamespace(Example=GoodExample),
            ),
            self.assertLogs("newton.examples", level="ERROR") as logs,
        ):
            newton.examples.run(FailedExample(viewer, args), args)
        self.assertTrue(viewer.closed)
        self.assertGreaterEqual(viewer.frames, 3)
        self.assertEqual(browser.current_module, "fake.recovered")
        self.assertIsNone(browser.load_error)
        self.assertFalse(viewer._step_requested)
        self.assertEqual(len(logs.records), 1)
        self.assertIsNotNone(logs.records[0].exc_info)

    def test_constructor_exit_keeps_browser_running(self):
        viewer = _StubViewer()
        browser = _ExampleBrowser(viewer)
        browser.switch_target = "fake.exiting"

        class ExitingExample(_StubExample):
            def __init__(self, viewer, args):
                raise SystemExit(2)

        with (
            patch(
                "newton.examples.importlib.import_module", return_value=types.SimpleNamespace(Example=ExitingExample)
            ),
            self.assertLogs("newton.examples", level="ERROR"),
        ):
            example, _ = browser.switch(_StubExample)
        self.assertIsNone(example)
        self.assertIn("SystemExit: 2", browser.load_error)
        self.assertEqual(browser.failed_target, "fake.exiting")
        with self.assertLogs("newton.examples", level="ERROR"):
            self.assertIsNone(browser.reset(ExitingExample))
        self.assertIn("SystemExit: 2", browser.load_error)

    def test_reset_preserves_layer_display_options(self):
        """Preserve presentation choices on reset and across differently named layers."""

        class LayeredExample(_StubExample):
            def __init__(self, viewer, args):
                super().__init__(viewer, args)
                viewer.activate("comparison")
                viewer.show_contacts = False
                viewer.show_visual = True

        viewer = ViewerNull(num_frames=1)
        viewer.show_contacts = True
        viewer.activate("comparison")
        viewer.show_contacts = True
        viewer.show_visual = False
        viewer.show_contact_normals = False
        viewer.contact_viz_scale = 3.5
        viewer.contact_normal_scale = 0.6
        viewer.show_contact_pairs = True
        viewer.show_contact_lines = False
        viewer.show_contact_labels = False
        viewer.camera_speed = 1.5
        viewer.camera_slowdown = 3.0
        viewer.contact_force_scale = 2.0 * viewer._contact_force_scale_default
        browser = _ExampleBrowser(viewer)
        self.assertIsNotNone(browser.reset(LayeredExample))
        self.assertTrue(viewer.show_contacts)
        self.assertFalse(viewer.show_visual)
        self.assertFalse(viewer.show_contact_normals)
        self.assertEqual(viewer.contact_viz_scale, 3.5)
        self.assertEqual(viewer.contact_normal_scale, 0.6)
        self.assertTrue(viewer.show_contact_pairs)
        self.assertFalse(viewer.show_contact_lines)
        self.assertEqual(viewer.contact_force_scale, 2.0 * viewer._contact_force_scale_default)
        self.assertTrue(viewer.layers["__default__"].show_contacts)
        viewer.show_ui, viewer.vsync, viewer._paused = False, True, True
        viewer.renderer = types.SimpleNamespace(draw_sky=False, sky_upper=(0.2, 0.3, 0.4))
        contact_render_options = {
            "contact_arrow_scale": 0.7,
            "contact_normal_max_pixels": 40.0,
            "contact_force_max_pixels": 50.0,
            "contact_disk_scale": 0.6,
        }
        for name, value in contact_render_options.items():
            setattr(viewer.renderer, name, value)
        viewer.camera = types.SimpleNamespace(fov=55.0, near=0.002, far=1500.0, pos="previous scene")

        class SwitchedExample(_StubExample):
            def __init__(self, viewer, args):
                super().__init__(viewer, args)
                viewer.activate("new scene")
                viewer.show_contacts, viewer.show_visual = False, True
                viewer.contact_viz_scale = 0.1
                viewer.contact_normal_scale = 1.0
                viewer.show_contact_pairs = False
                viewer.show_contact_lines = True
                viewer.show_contact_labels = True
                viewer.camera_speed = 4.0
                viewer.camera_slowdown = 2.0
                for name in contact_render_options:
                    setattr(viewer.renderer, name, 1.0)
                viewer._contact_force_scale_default = 0.005
                viewer.contact_force_scale = 0.005
                viewer.show_ui, viewer.vsync, viewer._paused = True, False, False
                viewer.renderer.draw_sky = True
                viewer.renderer.sky_upper = (1.0, 1.0, 1.0)
                viewer.camera = types.SimpleNamespace(fov=45.0, near=0.01, far=1000.0, pos="new scene")

        browser.switch_target = "fake.module.path"
        with patch(
            "newton.examples.importlib.import_module", return_value=types.SimpleNamespace(Example=SwitchedExample)
        ):
            example, _ = browser.switch(LayeredExample)
        self.assertIsInstance(example, SwitchedExample)
        self.assertTrue(viewer.show_contacts)
        self.assertFalse(viewer.show_visual)
        self.assertFalse(viewer.show_contact_normals)
        self.assertEqual(viewer.contact_viz_scale, 3.5)
        self.assertFalse(viewer.show_ui)
        self.assertTrue(viewer.vsync)
        self.assertTrue(viewer._paused)
        self.assertFalse(viewer.renderer.draw_sky)
        self.assertEqual(viewer.renderer.sky_upper, (0.2, 0.3, 0.4))
        self.assertEqual(viewer.camera.pos, "new scene")
        self.assertEqual((viewer.camera.fov, viewer.camera.near, viewer.camera.far), (55.0, 0.002, 1500.0))
        self.assertEqual(viewer.camera_speed, 1.5)
        self.assertEqual(viewer.camera_slowdown, 3.0)
        self.assertFalse(viewer.show_contact_labels)
        self.assertAlmostEqual(viewer.contact_force_scale, 0.01)
        self.assertEqual(viewer.contact_normal_scale, 0.6)
        self.assertTrue(viewer.show_contact_pairs)
        self.assertFalse(viewer.show_contact_lines)
        for name, value in contact_render_options.items():
            self.assertEqual(getattr(viewer.renderer, name), value)

        # Switching from mesh vertices (off by default) to sand must not
        # mistake the old example's default for a user's visibility choice.
        class ParticleExample(_StubExample):
            def __init__(self, viewer, args):
                super().__init__(viewer, args)
                viewer.show_particles = True

        particle_module = types.SimpleNamespace(Example=ParticleExample)
        viewer.show_particles = False
        browser.switch_target = "fake.particles"
        with patch("newton.examples.importlib.import_module", return_value=particle_module):
            example, _ = browser.switch(SwitchedExample)
        self.assertIsInstance(example, ParticleExample)
        self.assertTrue(viewer.show_particles)
        # Explicit GUI choices survive both switching and resetting.
        viewer._particle_visibility_user = viewer.show_particles = False
        browser.switch_target = "fake.particles"
        with patch("newton.examples.importlib.import_module", return_value=particle_module):
            browser.switch(ParticleExample)
        self.assertFalse(viewer.show_particles)
        browser.reset(ParticleExample)
        self.assertFalse(viewer.show_particles)

    def test_reset_preserves_user_provided_args(self):
        # Simulate the user invoking the example with `--world-count 2`.
        args = _StubExample.create_parser().parse_args(["--world-count", "2"])
        self.assertEqual(args.world_count, 2)

        viewer = _StubViewer()
        browser = _ExampleBrowser(viewer, args)

        new_example = browser.reset(_StubExample)

        self.assertIsNotNone(new_example)
        self.assertEqual(new_example.args.world_count, 2)
        self.assertEqual(viewer.cleared_all_layers, 1)
        self.assertEqual(viewer.cleared, 0)

    def test_reset_falls_back_to_defaults_when_no_args(self):
        viewer = _StubViewer()
        browser = _ExampleBrowser(viewer)

        new_example = browser.reset(_StubExample)

        self.assertIsNotNone(new_example)
        self.assertEqual(new_example.args.world_count, 4)

    def test_reset_snapshots_args_against_later_mutation(self):
        # The browser should snapshot the args at construction time so that
        # later mutations of the caller's namespace (or of nested mutable
        # fields) do not leak into the reset behavior.
        args = _StubExample.create_parser().parse_args(["--world-count", "2"])
        args.warp_config.append("dummy=1")

        browser = _ExampleBrowser(_StubViewer(), args)

        args.world_count = 99
        args.warp_config.append("mutated=1")

        new_example = browser.reset(_StubExample)

        self.assertIsNotNone(new_example)
        self.assertEqual(new_example.args.world_count, 2)
        self.assertEqual(new_example.args.warp_config, ["dummy=1"])

    def test_reset_after_switch_uses_new_example_args(self):
        # Original launch: _StubExample with --world-count 2.
        args = _StubExample.create_parser().parse_args(["--world-count", "2"])
        viewer = _StubViewer()
        browser = _ExampleBrowser(viewer, args)

        # Simulate the user picking _OtherStubExample from the browser tree.
        browser.switch_target = "fake.module.path"
        fake_module = types.SimpleNamespace(Example=_OtherStubExample)
        with patch("newton.examples.importlib.import_module", return_value=fake_module):
            new_example, new_class = browser.switch(_StubExample)

        self.assertIs(new_class, _OtherStubExample)
        self.assertEqual(browser.current_module, "fake.module.path")
        self.assertEqual(new_example.args.world_count, 7)
        self.assertEqual(viewer.cleared_all_layers, 1)
        self.assertEqual(viewer.cleared, 0)

        # Reset after switch must use the new example's args (its parser
        # defaults), not the originally launched _StubExample's args.
        reset_example = browser.reset(new_class)
        self.assertIsNotNone(reset_example)
        self.assertIsInstance(reset_example, _OtherStubExample)
        self.assertEqual(reset_example.args.world_count, 7)
        self.assertEqual(browser.current_module, "fake.module.path")

    def test_failed_switch_clears_partial_scene_and_reports_error(self):
        viewer = _StubViewer()
        browser = _ExampleBrowser(viewer)
        browser.current_module = "previous.example"
        browser.switch_target = "fake.module.path"

        class FailedExample(_StubExample):
            def __init__(self, viewer, args):
                raise RuntimeError("asset missing")

        failed_module = types.SimpleNamespace(Example=FailedExample)
        with (
            patch("newton.examples.importlib.import_module", return_value=failed_module),
            self.assertLogs("newton.examples", level="ERROR") as logs,
        ):
            example, example_class = browser.switch(_StubExample)
        self.assertIsNone(example)
        self.assertIsNone(browser.current_module)
        self.assertIs(example_class, _StubExample)
        self.assertEqual(viewer.cleared_all_layers, 2)
        self.assertIn("RuntimeError: asset missing", browser.load_error)
        self.assertIn("RuntimeError: asset missing", logs.output[0])
        self.assertIsNotNone(logs.records[0].exc_info)
        self.assertEqual(browser.failed_target, "fake.module.path")

        browser.switch_target = browser.failed_target
        with patch("newton.examples.importlib.import_module", return_value=types.SimpleNamespace(Example=_StubExample)):
            example, _ = browser.switch(_StubExample)
        self.assertIsInstance(example, _StubExample)
        self.assertEqual(browser.current_module, "fake.module.path")
        self.assertIsNone(browser.load_error)
        self.assertIsNone(browser.failed_target)


if __name__ == "__main__":
    unittest.main()
