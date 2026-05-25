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


class _StubViewer:
    """Minimal viewer stub for exercising _ExampleBrowser without a UI."""

    def __init__(self):
        self.cleared = 0

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
    def test_reset_preserves_user_provided_args(self):
        # Simulate the user invoking the example with `--world-count 2`.
        args = _StubExample.create_parser().parse_args(["--world-count", "2"])
        self.assertEqual(args.world_count, 2)

        viewer = _StubViewer()
        browser = _ExampleBrowser(viewer, args)

        new_example = browser.reset(_StubExample)

        self.assertIsNotNone(new_example)
        self.assertEqual(new_example.args.world_count, 2)
        self.assertEqual(viewer.cleared, 1)

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
        browser = _ExampleBrowser(_StubViewer(), args)

        # Simulate the user picking _OtherStubExample from the browser tree.
        browser.switch_target = "fake.module.path"
        fake_module = types.SimpleNamespace(Example=_OtherStubExample)
        with patch("newton.examples.importlib.import_module", return_value=fake_module):
            new_example, new_class = browser.switch(_StubExample)

        self.assertIs(new_class, _OtherStubExample)
        self.assertEqual(new_example.args.world_count, 7)

        # Reset after switch must use the new example's args (its parser
        # defaults), not the originally launched _StubExample's args.
        reset_example = browser.reset(new_class)
        self.assertIsNotNone(reset_example)
        self.assertIsInstance(reset_example, _OtherStubExample)
        self.assertEqual(reset_example.args.world_count, 7)


if __name__ == "__main__":
    unittest.main()
