# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the load_usd example's interactive file loading."""

import unittest

import newton.examples
from newton.examples.basic.example_load_usd import Example
from newton.viewer import ViewerNull


class _FakeUI:
    """Stand-in for the viewer's UI wrapper, yielding one picked path."""

    def __init__(self, picked):
        self._picked = picked

    def consume_file_dialog_result(self):
        picked, self._picked = self._picked, None
        return picked

    def open_load_file_dialog(self, title=""):
        pass


class _FakeImgui:
    """Stand-in for the imgui module passed to Example.gui()."""

    def text(self, *args):
        pass

    def button(self, *args):
        return False

    def same_line(self):
        pass

    def separator(self):
        pass


class TestLoadUsdDeferredLoad(unittest.TestCase):
    def test_gui_defers_load_out_of_the_render_pass(self):
        """A file picked in gui() is loaded by render(), not during gui() itself.

        gui() is invoked from inside the viewer's render pass, so rebuilding the
        scene there releases renderer resources that the pass is still using --
        ViewerRTX drives its UI from within _render_and_display() and then
        dereferences the renderer that set_model() just tore down.
        """
        first = newton.examples.get_asset("cartpole.usda")
        second = newton.examples.get_asset("ant.usda")

        args = Example.create_parser().parse_args([str(first)])
        example = Example(ViewerNull(), args)
        self.assertEqual(example.current_path, str(first))

        example.viewer.ui = _FakeUI(str(second))
        example.gui(_FakeImgui())

        # The pick is recorded but must not have rebuilt anything yet.
        self.assertEqual(example.current_path, str(first))
        self.assertEqual(example._pending_path, str(second))

        example.render()

        self.assertEqual(example.current_path, str(second))
        self.assertIsNone(example._pending_path)


if __name__ == "__main__":
    unittest.main()
