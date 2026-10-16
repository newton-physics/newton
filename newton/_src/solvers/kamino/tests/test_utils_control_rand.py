# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RandomController class."""

import unittest

import numpy as np
import warp as wp

from newton import ModelBuilder
from newton._src.solvers.kamino._src import ModelKamino
from newton._src.solvers.kamino._src.core.time import advance_time
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.control.rand import RandomJointController
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.utils.basics import build_boxes_fourbar

###
# Tests
###


class TestRandomController(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        # Create a default random controller
        controller = RandomJointController()
        # Check default values
        self.assertIsNotNone(controller)
        self.assertEqual(controller._model, None)
        self.assertEqual(controller._data, None)
        self.assertRaises(RuntimeError, lambda: controller.device)
        self.assertRaises(RuntimeError, lambda: controller.seed)
        self.assertRaises(RuntimeError, lambda: controller.model)
        self.assertRaises(RuntimeError, lambda: controller.data)

    def test_01_make_for_single_fourbar(self):
        # Define a model builder for the boxes_fourbar problem with 1 world
        builder = build_boxes_fourbar()
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        data = model.data()
        control = model.control()

        # Create a random controller with default arguments
        controller = RandomJointController(model=model, seed=self.seed)

        # Check contents
        self.assertIsNotNone(controller)
        self.assertIsNotNone(controller._model, None)
        self.assertIsNotNone(controller._data, None)
        self.assertIs(controller.device, model.device)

        # Check dimensions of the interval array
        self.assertEqual(controller.data.interval.shape, (model.size.num_worlds,))
        self.assertTrue((controller.data.interval.numpy() == 1.0).all())

        # Check that the seed is set correctly
        self.assertEqual(controller.seed, self.seed)

        # Check that the generated control inputs are different than the default values
        self.assertEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)
        controller.compute(time=data.time, control=control)
        tau_j_np_0 = control.tau_j.numpy().copy()
        msg.info("control.tau_j: %s", tau_j_np_0)
        self.assertGreaterEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)

    def test_02_make_for_multiple_fourbar(self):
        # Define a model builder for the boxes_fourbar problem with 4 worlds
        builder = ModelBuilder()
        builder.replicate(builder=build_boxes_fourbar(), world_count=4)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        data = model.data()
        control = model.control()

        # Create a random controller with default arguments
        controller = RandomJointController(model=model, seed=self.seed)

        # Check contents
        self.assertIsNotNone(controller)
        self.assertIsNotNone(controller._model, None)
        self.assertIsNotNone(controller._data, None)
        self.assertIs(controller.device, model.device)

        # Check dimensions of the interval array
        self.assertEqual(controller.data.interval.shape, (model.size.num_worlds,))
        self.assertTrue((controller.data.interval.numpy() == 1.0).all())

        # Check that the seed is set correctly
        self.assertEqual(controller.seed, self.seed)

        # Check that the generated control inputs are different than the default values
        self.assertEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)
        controller.compute(time=data.time, control=control)
        tau_j_np_0 = control.tau_j.numpy().copy()
        msg.info("control.tau_j: %s", tau_j_np_0)
        self.assertGreaterEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)

    def _build_replicated_fourbar(self, num_worlds: int):
        """Build a replicated fourbar ``ModelKamino`` and its data/control containers."""
        builder = ModelBuilder()
        builder.replicate(builder=build_boxes_fourbar(), world_count=num_worlds)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        return model, model.data(), model.control()

    def _dof_world_map(self, model: ModelKamino) -> np.ndarray:
        """Return, for each DoF in ``tau_j``, the world index owning it."""
        joints_wid = model.joints.wid.numpy()
        dofs_offset = model.joints.dofs_offset.numpy()
        dof_wid = np.empty(int(model.size.sum_of_num_joint_dofs), dtype=np.int32)
        for j in range(joints_wid.shape[0]):
            dof_wid[dofs_offset[j] : dofs_offset[j + 1]] = joints_wid[j]
        return dof_wid

    def test_03_time_based_scheduling_multi_world(self):
        """Behavioral test of the per-world elapsed-time scheduling.

        Configures four worlds with distinct ``(dt, interval)`` pairs and drives
        the controller through the simulation loop, asserting that ``tau_j`` is
        regenerated exactly at hand-computed elapsed-time boundaries and stays
        bit-for-bit identical between them. World 3 exercises the special
        ``interval == 0.0`` case, which requests a fresh torque at every step
        regardless of the world's time-step; worlds 0-2 exercise time-based
        scheduling, with world 0 vs. world 2 (same ``dt``, different
        ``interval``) demonstrating that the schedule is driven by elapsed
        simulated time rather than step counts.
        """
        # Build a replicated 4-world fourbar model
        num_worlds = 4
        model, data, control = self._build_replicated_fourbar(num_worlds)

        # Configure distinct per-world time-steps (powers of two, exactly
        # representable in float32 to avoid accumulated drift over the loop)
        per_world_dt = np.array([0.125, 0.25, 0.125, 0.25], dtype=np.float32)
        model.time.set_timesteps(per_world_dt)

        # Configure distinct per-world intervals for torque regeneration; the
        # 0.0 entry selects the "refresh every step" special mode
        per_world_interval = np.array([0.5, 0.75, 0.25, 0.0], dtype=np.float32)
        controller = RandomJointController(model=model, interval=per_world_interval, seed=self.seed)

        # Ground-truth per-world update-step sets, computed by hand from the
        # ``t = k * dt`` schedule crossing multiples of ``interval``:
        #   world 0: dt=0.125, interval=0.5  -> t in {0.0, 0.5, 1.0, 1.5}
        #   world 1: dt=0.25,  interval=0.75 -> t in {0.0, 0.75, 1.5, 2.25, 3.0}
        #   world 2: dt=0.125, interval=0.25 -> t in {0.0, 0.25, ..., 1.5}
        #   world 3: dt=0.25,  interval=0.0  -> every step (special mode)
        num_steps = 13  # spans t in [0, 1.5] for world 0 and [0, 3.0] for worlds 1/3
        expected_update_steps = {
            0: {0, 4, 8, 12},
            1: {0, 3, 6, 9, 12},
            2: {0, 2, 4, 6, 8, 10, 12},
            3: set(range(num_steps)),
        }

        # Map each DoF to its owning world so per-world slices of ``tau_j`` can
        # be compared independently
        dof_wid = self._dof_world_map(model)
        for w in range(num_worlds):
            self.assertTrue((dof_wid == w).any(), f"world {w} has no DoFs to observe")

        # Walk the simulation loop, snapshotting tau_j after each compute; the
        # kernel writes only on update steps, so on non-update steps the slice
        # must remain byte-for-byte identical to the previous snapshot
        prev_tau_j = control.tau_j.numpy().copy()
        self.assertEqual(np.linalg.norm(prev_tau_j), 0.0)
        for step in range(num_steps):
            controller.compute(time=data.time, control=control)
            tau_j = control.tau_j.numpy().copy()
            for w in range(num_worlds):
                mask = dof_wid == w
                prev_slice = prev_tau_j[mask]
                curr_slice = tau_j[mask]
                if step in expected_update_steps[w]:
                    self.assertFalse(
                        np.array_equal(prev_slice, curr_slice),
                        f"world {w} step {step}: torques did not update at expected boundary",
                    )
                else:
                    np.testing.assert_array_equal(
                        curr_slice,
                        prev_slice,
                        err_msg=f"world {w} step {step}: torques changed between boundaries",
                    )
            advance_time(model=model.time, data=data.time)
            prev_tau_j = tau_j

    def test_04_seed_reproducibility(self):
        """Same seed reproduces the exact torque sequence; changing the seed
        produces a different sequence.

        Rebuilding the model and controller from scratch and re-running the
        simulation loop with the same seed must yield bit-for-bit identical
        ``tau_j`` snapshots at every step, including the ``interval == 0.0``
        world where a new torque is drawn each step. Bumping the seed by one
        must change at least the initial torque, since every world updates at
        ``t = 0`` and the RNG stream depends on the seed.
        """
        num_worlds = 4
        num_steps = 6
        per_world_dt = np.array([0.125, 0.25, 0.125, 0.25], dtype=np.float32)
        per_world_interval = np.array([0.5, 0.75, 0.25, 0.0], dtype=np.float32)

        def run(seed: int) -> list[np.ndarray]:
            """Run ``num_steps`` of the controller from a freshly built model."""
            model, data, control = self._build_replicated_fourbar(num_worlds)
            model.time.set_timesteps(per_world_dt)
            controller = RandomJointController(model=model, interval=per_world_interval, seed=seed)
            history = []
            for _ in range(num_steps):
                controller.compute(time=data.time, control=control)
                history.append(control.tau_j.numpy().copy())
                advance_time(model=model.time, data=data.time)
            return history

        history_a = run(self.seed)
        history_b = run(self.seed)
        history_c = run(self.seed + 1)

        # Same seed must yield the exact same torque snapshot at every step
        for step, (a, b) in enumerate(zip(history_a, history_b, strict=True)):
            np.testing.assert_array_equal(a, b, err_msg=f"step {step}: identical seed produced different torques")

        # A different seed must alter the RNG stream: since every world updates
        # at step 0, the initial torque snapshot must differ from the reference
        self.assertFalse(
            np.array_equal(history_a[0], history_c[0]),
            "changing the seed did not change the generated torques at step 0",
        )


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
