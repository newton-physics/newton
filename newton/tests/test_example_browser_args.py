# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for browser arguments, failure recovery, and display choices.

The companion :mod:`newton.tests.test_example_browser` script runs every
registered example through the browser using a real GL viewer; it is a
manual smoke test and does not contain any auto-discovered ``TestCase``
classes.
"""

import argparse
import gc
import types
import unittest
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from unittest.mock import patch

import warp as wp

import newton
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

    def _preserve_display_options(self, *, switching=False):
        return nullcontext()


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


class _BrowserViewer(ViewerNull):
    """Use real layer/model management with an in-memory browser UI."""

    def __init__(self) -> None:
        self.callbacks: dict[str, list[Callable]] = {}
        self.splash: str | None = None
        self.closed = False
        self.on_frame: Callable[[], None] = lambda: None
        self.reset_callback: Callable[[], None] | None = None
        super().__init__(num_frames=6)

    def register_ui_callback(self, callback: Callable, position: str = "side") -> None:
        self.callbacks.setdefault(position, []).append(callback)

    def set_reset_callback(self, callback: Callable[[], None]) -> None:
        self.reset_callback = callback

    def clear_model(self) -> None:
        self.callbacks.pop("side", None)
        self.callbacks.pop("free", None)
        super().clear_model()

    def show_loading_splash(self, text: str) -> None:
        self.splash = text

    def hide_loading_splash(self) -> None:
        self.splash = None

    def end_frame(self) -> None:
        super().end_frame()
        if self.splash is None:
            self.on_frame()

    def close(self) -> None:
        self.closed = True


class TestExampleBrowserReset(unittest.TestCase):
    def test_display_options_survive_reset_and_switch(self) -> None:
        """Keep layer choices while recalibrating forces for the new model."""
        models = []
        for mass in (1.0, 10.0):
            builder = newton.ModelBuilder()
            builder.add_body(mass=mass, inertia=wp.mat33(mass, 0.0, 0.0, 0.0, mass, 0.0, 0.0, 0.0, mass))
            models.append(builder.finalize(device="cpu"))

        contact_render_options = {
            "contact_arrow_scale": 0.7,
            "contact_normal_max_pixels": 40.0,
            "contact_force_max_pixels": 50.0,
            "contact_disk_scale": 0.6,
        }

        def contact_defaults(viewer: ViewerNull) -> None:
            viewer.show_contact_pairs = False
            viewer.show_contact_lines = True
            viewer.show_contact_labels = True
            viewer.camera_slowdown = 2.0
            for name in contact_render_options:
                setattr(viewer.renderer, name, 1.0)

        class Scene(_StubExample):
            def __init__(self, viewer: ViewerNull, args: argparse.Namespace) -> None:
                super().__init__(viewer, args)
                viewer.activate("scene")
                viewer.set_model(models[0])
                viewer.show_joints = True
                viewer.camera = types.SimpleNamespace(
                    fov=40.0, near=0.01, far=1000.0, pos=(1.0, 2.0, 3.0), pitch=0.0, yaw=0.0, pivot=(0.0, 0.0, 0.0)
                )
                viewer.renderer = types.SimpleNamespace(draw_wireframe=True, joint_scale=2.0)
                viewer.renderer.wireframe_line_width = 1.0
                viewer.camera_speed = 4.0
                viewer.vsync = False
                viewer._paused = False
                viewer.show_ui = False
                contact_defaults(viewer)

        class OtherScene(_StubExample):
            def __init__(self, viewer: ViewerNull, args: argparse.Namespace) -> None:
                super().__init__(viewer, args)
                viewer.activate("other")
                viewer.set_model(models[1])
                viewer.show_particles = True
                viewer.show_hydro_contact_surface = True
                viewer.camera = types.SimpleNamespace(fov=90.0, near=0.1, far=100.0)
                viewer.renderer = types.SimpleNamespace(draw_wireframe=False, joint_scale=4.0)
                viewer.renderer.wireframe_line_width = 2.0
                viewer.camera_speed = 0.2
                viewer.vsync = True
                viewer._paused = True
                viewer.show_ui = True
                contact_defaults(viewer)

        with wp.ScopedDevice("cpu"):
            viewer = ViewerNull(num_frames=1)
            Scene(viewer, newton.examples.default_args())
            viewer.show_contacts = True
            viewer.show_visual = False
            viewer.show_contact_normals = False
            viewer.contact_viz_scale = 3.5
            viewer.contact_mode_eps_force = 0.2
            viewer.contact_force_scale *= 2.0
            viewer.camera.fov = 55.0
            viewer.camera.pos = (4.0, 5.0, 6.0)
            viewer.camera.pitch, viewer.camera.yaw = -20.0, 30.0
            viewer.camera.pivot = (1.0, 1.0, 1.0)
            viewer.camera_speed = 0.75
            viewer.renderer.joint_scale = 3.0
            viewer.renderer.wireframe_line_width = 3.0
            viewer._display_options_user.update(
                show_contacts=True,
                show_visual=False,
                show_contact_normals=False,
                contact_viz_scale=3.5,
                contact_mode_eps_force=0.2,
                contact_force_scale=2.0,
                joint_scale=3.0,
                wireframe_line_width=3.0,
            )
            # Exercise the same preference recording boundary as the workspace,
            # without constructing an OpenGL renderer or duplicating its UI.
            with viewer._track_display_options():
                viewer.contact_normal_scale = 0.6
                viewer.show_contact_pairs = True
                viewer.show_contact_lines = False
                viewer.show_contact_labels = False
                viewer.camera_slowdown = 3.0
                viewer.camera.near, viewer.camera.far = 0.002, 1500.0
                for name, value in contact_render_options.items():
                    setattr(viewer.renderer, name, value)
            original_force_scale = viewer.contact_force_scale
            viewer.layer.visible = False
            browser = _ExampleBrowser(viewer)
            self.assertIsNotNone(browser.reset(Scene))
            self.assertFalse(viewer.layer.visible)
            self.assertAlmostEqual(viewer.contact_force_scale, original_force_scale)
            self.assertEqual(viewer.camera.fov, 55.0)
            self.assertEqual(viewer.camera.pos, (4.0, 5.0, 6.0))
            self.assertEqual((viewer.camera.pitch, viewer.camera.yaw), (-20.0, 30.0))
            self.assertEqual(viewer.camera.pivot, (1.0, 1.0, 1.0))
            self.assertEqual(viewer.camera_speed, 0.75)
            with self.subTest(operation="reset", option="wireframe_line_width"):
                self.assertEqual(viewer.renderer.wireframe_line_width, 3.0)
            self.assertTrue(viewer.show_joints)

            browser.switch_target = "test.other"
            with patch(
                "newton.examples.importlib.import_module", return_value=types.SimpleNamespace(Example=OtherScene)
            ):
                example, example_class = browser.switch(Scene)
            self.assertIsInstance(example, OtherScene)
            self.assertTrue(viewer.layer.visible)
            self.assertTrue(viewer.show_contacts)
            self.assertFalse(viewer.show_visual)
            self.assertFalse(viewer.show_contact_normals)
            self.assertEqual(viewer.contact_viz_scale, 3.5)
            self.assertEqual(viewer.contact_mode_eps_force, 0.2)
            self.assertAlmostEqual(viewer.contact_force_scale, original_force_scale / 10.0)
            self.assertFalse(viewer.show_joints)
            self.assertTrue(viewer.show_hydro_contact_surface)
            self.assertFalse(viewer.renderer.draw_wireframe)
            self.assertEqual(viewer.renderer.joint_scale, 3.0)
            self.assertEqual(viewer.camera.fov, 90.0)
            with self.subTest(operation="switch", option="camera_speed"):
                self.assertEqual(viewer.camera_speed, 0.2)
            with self.subTest(operation="switch", option="wireframe_line_width"):
                self.assertEqual(viewer.renderer.wireframe_line_width, 3.0)
            self.assertTrue(viewer.vsync)
            self.assertTrue(viewer._paused)
            self.assertTrue(viewer.show_ui)
            self.assertEqual(viewer.contact_normal_scale, 0.6)
            self.assertTrue(viewer.show_contact_pairs)
            self.assertFalse(viewer.show_contact_lines)
            self.assertFalse(viewer.show_contact_labels)
            self.assertEqual(viewer.camera_slowdown, 3.0)
            self.assertEqual((viewer.camera.near, viewer.camera.far), (0.002, 1500.0))
            for name, value in contact_render_options.items():
                self.assertEqual(getattr(viewer.renderer, name), value)
            # A previous scene's default must not hide loose particles.
            self.assertTrue(viewer.show_particles)
            viewer.show_particles = False
            viewer._display_options_user["show_particles"] = False
            self.assertIsNotNone(browser.reset(example_class))
            self.assertFalse(viewer.show_particles)
            # Combine explicit particle and joint choices in one switch, including
            # a choice equal to the old default and a different active layer.
            viewer.show_joints = False
            viewer._display_options_user.update(show_joints=False, _paused=False, show_ui=False)
            viewer.activate("unused")
            viewer.show_joints = True
            with viewer._track_display_options():
                viewer.camera.fov = 46.0
                viewer.camera_speed = 1.5
            browser.switch_target = "test.other"
            with patch(
                "newton.examples.importlib.import_module", return_value=types.SimpleNamespace(Example=OtherScene)
            ):
                browser.switch(example_class)
            self.assertFalse(viewer.show_particles)
            self.assertFalse(viewer.show_joints)
            self.assertFalse(viewer._paused)
            self.assertFalse(viewer.show_ui)

            self.assertEqual(viewer.camera.fov, 46.0)
            self.assertEqual(viewer.camera_speed, 1.5)

    def test_load_failures_clear_partial_scenes_and_allow_retry(self) -> None:
        """Recover from parser exits and partial constructors on switch and reset."""
        builder = newton.ModelBuilder()
        model = builder.finalize(device="cpu")

        class FailingScene(_StubExample):
            failure: str | None = None
            last_instance: weakref.ReferenceType | None = None

            @staticmethod
            def create_parser() -> argparse.ArgumentParser:
                parser = _StubExample.create_parser()
                if FailingScene.failure == "parser":
                    parser.add_argument("--required", required=True)
                return parser

            def __init__(self, viewer: _BrowserViewer, args: argparse.Namespace) -> None:
                super().__init__(viewer, args)
                FailingScene.last_instance = weakref.ref(self)
                self.cycle = self
                viewer.activate("partial")
                viewer.set_model(model)
                viewer.register_ui_callback(lambda ui: self.args)
                if self.failure == "constructor":
                    raise RuntimeError("construction failed")
                if self.failure == "exit":
                    raise SystemExit(2)

        # Cover each failure kind and both entry points without their full product.
        for operation, failure in (("switch", "parser"), ("reset", "constructor"), ("switch", "exit")):
            with self.subTest(operation=operation, failure=failure), wp.ScopedDevice("cpu"):
                viewer = _BrowserViewer()
                browser = _ExampleBrowser(viewer)
                module = types.SimpleNamespace(Example=FailingScene)
                FailingScene.failure = failure
                with patch("newton.examples.importlib.import_module", return_value=module):
                    with self.assertLogs("newton", level="ERROR") as logs:
                        if operation == "switch":
                            browser.switch_target = "test.failing"
                            result, _ = browser.switch(_StubExample)
                        else:
                            result = browser.reset(FailingScene)
                    self.assertIsNone(result)
                    self.assertIn("SystemExit" if failure != "constructor" else "construction failed", logs.output[0])
                    self.assertIsNone(viewer.splash)
                    self.assertIsNone(viewer.model)
                    self.assertEqual(len(viewer.layers), 1)
                    self.assertNotIn("side", viewer.callbacks)
                    self.assertIn("panel", viewer.callbacks)
                    self.assertFalse(viewer.closed)
                    # LogRecord.exc_info also owns the failed constructor's traceback.
                    del logs
                    gc.collect()
                    if failure != "parser":
                        self.assertIsNone(FailingScene.last_instance())
                    FailingScene.failure = None
                    if operation == "switch":
                        browser.switch_target = "test.failing"
                        result, _ = browser.switch(_StubExample)
                    else:
                        result = browser.reset(FailingScene)
                    self.assertIsInstance(result, FailingScene)
                    self.assertIs(viewer.model, model)
                    self.assertIsNone(viewer.splash)

    def test_cleanup_precedes_new_constructor(self) -> None:
        """Collect callback-owned cycles before a replacement starts graph capture."""

        class Scene(_StubExample):
            def __init__(self, viewer: _BrowserViewer, args: argparse.Namespace) -> None:
                self.cycle = self
                self.previous_released = not viewer.old_refs or viewer.old_refs[-1]() is None
                viewer.old_refs.append(weakref.ref(self))
                super().__init__(viewer, args)

            def gui(self, ui: object) -> None:
                pass

            def step(self) -> None:
                pass

            def render(self) -> None:
                self.viewer.begin_frame(0.0)
                self.viewer.end_frame()

        def make_browser(viewer: _BrowserViewer, args: argparse.Namespace) -> _ExampleBrowser:
            browser = _ExampleBrowser(viewer, args)

            def replace() -> None:
                if args.operation == "switch":
                    browser.switch_target = "test.scene"
                else:
                    viewer.reset_callback()
                viewer.on_frame = lambda: None

            viewer.on_frame = replace
            return browser

        for operation in ("switch", "reset"):
            with self.subTest(operation=operation), wp.ScopedDevice("cpu"):
                viewer = _BrowserViewer()
                viewer.old_refs = []
                args = newton.examples.default_args()
                args.operation = operation

                with (
                    patch("newton.examples._ExampleBrowser", side_effect=make_browser),
                    patch("newton.examples.importlib.import_module", return_value=types.SimpleNamespace(Example=Scene)),
                ):
                    newton.examples.run(Scene(viewer, args), args)
                self.assertEqual(len(viewer.old_refs), 2)
                self.assertIsNone(viewer.old_refs[0]())
                self.assertTrue(viewer.old_refs[1]().previous_released)

    def test_runtime_failure_recovers_but_automated_runs_raise(self) -> None:
        """Keep browsing after step/render errors and propagate automated failures."""

        class Scene(_StubExample):
            fail = True

            def __init__(self, viewer: _BrowserViewer, args: argparse.Namespace) -> None:
                super().__init__(viewer, args)
                viewer.activate("runtime")

            def step(self) -> None:
                if self.args.failure_phase == "step" and Scene.fail:
                    raise self.args.failure

            def render(self) -> None:
                self.viewer.begin_frame(0.0)
                if self.args.failure_phase == "render" and Scene.fail:
                    raise self.args.failure
                self.viewer.render_count += 1
                self.viewer.end_frame()

            def test_final(self) -> None:
                pass

        # Recovery covers both phases; each automated mode needs only one failure.
        for phase, mode in (
            ("step", "interactive"),
            ("render", "interactive"),
            ("step", "test"),
            ("render", "headless"),
            ("step", "benchmark"),
            ("render", "null"),
        ):
            with self.subTest(phase=phase, mode=mode), wp.ScopedDevice("cpu"):
                viewer = ViewerNull(num_frames=6) if mode == "null" else _BrowserViewer()
                viewer.render_count = 0
                args = newton.examples.default_args()
                if mode == "null":
                    args.viewer = "null"
                elif mode != "interactive":
                    setattr(args, mode, None if mode == "benchmark" else True)
                args.failure_phase = phase
                args.failure = RuntimeError(f"{phase} failed")
                Scene.fail = True

                if mode != "interactive":
                    with self.assertRaises(RuntimeError) as raised:
                        newton.examples.run(Scene(viewer, args), args)
                    self.assertIs(raised.exception, args.failure)
                else:

                    def retry(viewer: _BrowserViewer = viewer) -> None:
                        self.assertIsNone(viewer.model)
                        self.assertEqual(len(viewer.layers), 1)
                        Scene.fail = False
                        viewer.reset_callback()
                        viewer.on_frame = lambda: None

                    viewer.on_frame = retry
                    with self.assertLogs("newton", level="ERROR") as logs:
                        newton.examples.run(Scene(viewer, args), args)
                    self.assertIn(f"{phase} failed", logs.output[0])
                    self.assertGreater(viewer.render_count, 0)
                    self.assertTrue(viewer.closed)
                    self.assertIsNone(viewer.splash)

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

        viewer = _StubViewer()
        browser = _ExampleBrowser(viewer, args)

        args.world_count = 99
        args.warp_config.append("mutated=1")

        new_example = browser.reset(_StubExample)

        self.assertIsNotNone(new_example)
        self.assertEqual(new_example.args.world_count, 2)
        self.assertEqual(new_example.args.warp_config, ["dummy=1"])
        self.assertEqual(viewer.cleared_all_layers, 1)
        self.assertEqual(viewer.cleared, 0)

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


if __name__ == "__main__":
    unittest.main()
