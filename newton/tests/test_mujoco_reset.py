# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverMuJoCo.reset() clearing stale internal buffers per world."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import StateFlags
from newton.solvers import SolverMuJoCo


def _build_two_world_model(world_count: int = 2) -> newton.Model:
    """Build a model of ``world_count`` identical free + revolute articulations."""
    template = newton.ModelBuilder()
    body0 = template.add_link(mass=0.2, xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    template.add_shape_box(body=body0, hx=0.1, hy=0.1, hz=0.1)
    joint0 = template.add_joint_free(child=body0)

    body1 = template.add_link(mass=0.1)
    template.add_shape_capsule(body=body1, radius=0.05, half_height=0.15)
    joint1 = template.add_joint_revolute(
        parent=body0,
        child=body1,
        parent_xform=wp.transform((0.0, 0.15, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, -0.15, 0.0), wp.quat_identity()),
        axis=(0.0, 0.0, 1.0),
    )
    template.add_articulation([joint0, joint1])

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for i in range(world_count):
        builder.add_world(template, xform=wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity()))
    return builder.finalize()


class TestMuJoCoReset(unittest.TestCase):
    def setUp(self):
        self.model = _build_two_world_model(world_count=2)
        self.solver = SolverMuJoCo(self.model, iterations=2, ls_iterations=2)
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        newton.eval_fk(self.model, self.state_in.joint_q, self.state_in.joint_qd, self.state_in)
        # One step populates the internal buffers we then expect reset to clear.
        self.collision_pipeline.collide(self.state_in, self.contacts)
        self.solver.step(self.state_in, self.state_out, self.control, self.contacts, 1.0 / 60.0)

    def _cleared_buffers(self):
        """MuJoCo buffers that reset() must zero, skipping any that are empty.

        ``qacc`` is excluded: the solver overwrites it from ``qacc_warmstart``
        at the start of every step, so reset() deliberately leaves it alone.
        """
        d = self.solver.mjw_data
        buffers = {
            "qacc_warmstart": d.qacc_warmstart,
            "qfrc_applied": d.qfrc_applied,
            "ctrl": d.ctrl,
            "act": d.act,
            "xfrc_applied": d.xfrc_applied,
        }
        return {name: buf for name, buf in buffers.items() if buf.shape[1] > 0}

    def _poison(self):
        """Fill all reset-managed buffers with a non-zero sentinel in every world."""
        for buf in self._cleared_buffers().values():
            buf.fill_(7.0)

    def test_reset_masked_world_only(self):
        """A per-world mask clears the selected world and leaves the others intact."""
        self._poison()
        mask = wp.array([True, False], dtype=wp.bool, device=self.model.device)
        self.solver.reset(self.state_out, world_mask=mask)

        for name, buf in self._cleared_buffers().items():
            values = buf.numpy()
            self.assertTrue(np.all(values[0] == 0.0), f"{name} not cleared in masked world 0")
            self.assertTrue(np.all(values[1] == 7.0), f"{name} wrongly cleared in unmasked world 1")

    def test_reset_all_worlds(self):
        """A ``None`` mask clears every world."""
        self._poison()
        self.solver.reset(self.state_out, world_mask=None)

        for name, buf in self._cleared_buffers().items():
            values = buf.numpy()
            self.assertTrue(np.all(values == 0.0), f"{name} not cleared in all worlds")

    def test_reset_rejects_wrong_length_mask(self):
        mask = wp.array([True, False, True], dtype=wp.bool, device=self.model.device)
        with self.assertRaises(ValueError):
            self.solver.reset(self.state_out, world_mask=mask)

    def test_reset_recovers_from_nan_warmstart(self):
        """A NaN warm-start in one world is cleared so the next step stays finite."""
        warmstart = self.solver.mjw_data.qacc_warmstart
        if warmstart.shape[1] == 0:
            self.skipTest("model has no DOFs to warm-start")
        poisoned = warmstart.numpy()
        poisoned[0, :] = np.nan
        warmstart.assign(poisoned)

        mask = wp.array([True, False], dtype=wp.bool, device=self.model.device)
        self.solver.reset(self.state_out, world_mask=mask)
        self.assertTrue(np.all(np.isfinite(self.solver.mjw_data.qacc_warmstart.numpy())))

    def test_joint_q_reset_to_model_default_masked(self):
        """JOINT_Q resets joint_q to model defaults for masked worlds only."""
        defaults = self.model.joint_q.numpy()
        coords_per_world = self.model.joint_coord_count // self.model.world_count
        # Corrupt the live joint coordinates in both worlds.
        self.state_out.joint_q.assign(np.full_like(defaults, 9.0))

        mask = wp.array([True, False], dtype=wp.bool, device=self.model.device)
        self.solver.reset(self.state_out, world_mask=mask, flags=StateFlags.JOINT_Q)

        result = self.state_out.joint_q.numpy()
        np.testing.assert_allclose(result[:coords_per_world], defaults[:coords_per_world])
        np.testing.assert_allclose(result[coords_per_world:], np.full(coords_per_world, 9.0))

    def test_joint_qd_reset_to_model_default(self):
        """JOINT_QD resets joint_qd to model defaults across all worlds."""
        defaults = self.model.joint_qd.numpy()
        self.state_out.joint_qd.assign(np.full_like(defaults, 5.0))

        self.solver.reset(self.state_out, world_mask=None, flags=StateFlags.JOINT_QD)

        np.testing.assert_allclose(self.state_out.joint_qd.numpy(), defaults)

    def test_flags_zero_preserves_joint_state_but_clears_buffers(self):
        """flags=0 keeps the Newton state untouched while still clearing buffers."""
        self._poison()
        corrupted = np.full_like(self.model.joint_q.numpy(), 9.0)
        self.state_out.joint_q.assign(corrupted)

        self.solver.reset(self.state_out, world_mask=None, flags=0)

        # Joint state preserved...
        np.testing.assert_allclose(self.state_out.joint_q.numpy(), corrupted)
        # ...but internal buffers still cleared.
        for name, buf in self._cleared_buffers().items():
            self.assertTrue(np.all(buf.numpy() == 0.0), f"{name} not cleared with flags=0")

    def test_body_flags_are_ignored(self):
        """BODY_Q/BODY_QD do not touch joint state (body poses are FK-derived)."""
        corrupted = np.full_like(self.model.joint_q.numpy(), 9.0)
        self.state_out.joint_q.assign(corrupted)

        self.solver.reset(self.state_out, world_mask=None, flags=StateFlags.BODY_Q | StateFlags.BODY_QD)

        np.testing.assert_allclose(self.state_out.joint_q.numpy(), corrupted)

    def test_reset_defers_qpos_sync_at_default_interval(self):
        """At the default interval (1), reset leaves qpos for the next step to sync."""
        qpos = self.solver.mjw_data.qpos
        qpos.assign(np.full(qpos.shape, 3.0, dtype=np.float32))

        self.solver.reset(self.state_out, world_mask=None, flags=StateFlags.JOINT_Q)

        # reset does not touch qpos at interval == 1 (step() syncs it every step).
        np.testing.assert_allclose(self.solver.mjw_data.qpos.numpy(), 3.0)

    def test_reset_syncs_qpos_when_interval_not_default(self):
        """With update_data_interval != 1, reset pushes joint state into qpos now."""
        model = _build_two_world_model(world_count=2)
        solver = SolverMuJoCo(model, iterations=2, ls_iterations=2, update_data_interval=0)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        # The reference is whatever syncing the current state produces (the same
        # path reset uses); capture it, then poison qpos with NaNs.
        solver._update_mjc_data(solver.mjw_data, model, state)
        qpos_synced = solver.mjw_data.qpos.numpy().copy()
        solver.mjw_data.qpos.assign(np.full_like(qpos_synced, np.nan))

        # With the per-step sync disabled, reset must push the state into qpos now.
        solver.reset(state, world_mask=None, flags=StateFlags.JOINT_Q)

        result = solver.mjw_data.qpos.numpy()
        self.assertTrue(np.all(np.isfinite(result)), "reset did not overwrite NaN qpos")
        np.testing.assert_allclose(result, qpos_synced, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
