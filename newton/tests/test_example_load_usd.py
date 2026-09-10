# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the load_usd example's interactive file loading."""

import unittest

import numpy as np

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


class TestLoadUsdReset(unittest.TestCase):
    def test_reset_sim_survives_the_next_step(self):
        """Reset restores the imported pose and the restored pose survives stepping.

        The default solver is reduced-coordinate: it steps from
        ``state.joint_q``/``joint_qd`` and re-derives ``body_q`` by forward
        kinematics. A reset that only rewrote ``body_q`` therefore looked correct
        until the next step overwrote it from the stale joint coordinates.
        """
        asset = newton.examples.get_asset("ant.usda")
        args = Example.create_parser().parse_args([str(asset)])
        example = Example(ViewerNull(), args)

        initial = example.state_0.body_q.numpy().copy()
        for _ in range(60):
            example.step()
        moved = example.state_0.body_q.numpy().copy()
        self.assertGreater(np.abs(moved - initial).max(), 0.05, "asset did not move; the test would be vacuous")

        example.reset_sim()
        np.testing.assert_allclose(example.state_0.body_q.numpy(), initial)

        example.step()
        after = example.state_0.body_q.numpy()
        self.assertLess(np.abs(after - initial).max(), np.abs(after - moved).max())


class TestLoadUsdSolverWiring(unittest.TestCase):
    def test_solver_native_contacts_replace_the_newton_pipeline(self):
        """The example runs Newton's collision pipeline only for solvers that consume it.

        :class:`~newton.solvers.SolverMuJoCo` runs its own collision detection and
        ignores the contacts passed to ``step()``, so the example must hand it a
        buffer to report into instead of drawing contacts it never used.
        """
        asset = str(newton.examples.get_asset("cartpole.usda"))

        mujoco = Example(ViewerNull(), Example.create_parser().parse_args([asset, "--solver", "mujoco"]))
        self.assertIsNone(mujoco.collision_pipeline)

        xpbd = Example(ViewerNull(), Example.create_parser().parse_args([asset, "--solver", "xpbd"]))
        self.assertIsNotNone(xpbd.collision_pipeline)

        # Both must report contacts for the viewer after a step.
        for example in (mujoco, xpbd):
            example.step()
            self.assertIsNotNone(example.contacts.rigid_contact_count)


if __name__ == "__main__":
    unittest.main()
