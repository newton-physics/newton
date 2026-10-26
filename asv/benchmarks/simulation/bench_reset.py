# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA ASV timings for RL-style MuJoCo Warp resets and the first step.

Humanoid uses 2048 worlds. Full reset selects all worlds; partial reset selects
every fourth world (512/2048). The PR gate measures partial reset plus the first
step; full reset-only remains available to ordinary ASV collection. Prepared
nondefault joint positions and velocities are written through ArticulationView,
selected solver buffers are cleared without restoring model joint defaults,
and FK updates selected body state. Step series also clear Newton forces and
take exactly one direct SolverMuJoCo.step at the example's sim_dt, including
Newton-to-MuJoCo qpos/qvel synchronization. All series wait for CUDA completion.

Model construction, targets, selection masks, dirty-state preparation, warm-up,
and verification are outside timing. No CUDA graph wraps the measured call.
Model-parameter randomization and the examples' clocks are outside this scope.
"""

import os
import sys

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented

import newton
from newton.selection import ArticulationView

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_config import pr_gate_repeat


class _Reset:
    repeat = pr_gate_repeat(10)
    number = 1
    # asv_runner 0.2.x repeats warm-up calls without re-running setup.
    warmup_time = 0
    param_names = ["world_count", "reset_count"]
    _buffer_names = ("qacc_warmstart", "qfrc_applied", "xfrc_applied", "act", "ctrl")

    def _create_example(self, world_count):
        from benchmark_mujoco import Example as MujocoExample  # noqa: PLC0415

        example = MujocoExample(
            robot="humanoid",
            world_count=world_count,
            headless=True,
            randomize=False,
            actuation="random",
            use_cuda_graph=False,
        )
        view = ArticulationView(example.model, "*")
        example.step()
        return example, view

    def _make_joint_states(self, model, world_count):
        initial_q = self.view.get_dof_positions(model).numpy().copy()
        initial_qd = self.view.get_dof_velocities(model).numpy().copy()
        assert initial_q.shape == (world_count, 1, 28)
        assert initial_qd.shape == (world_count, 1, 27)
        lower = model.joint_limit_lower.numpy().reshape((world_count, 27))[:, 6:]
        upper = model.joint_limit_upper.numpy().reshape((world_count, 27))[:, 6:]
        offset = np.arange(world_count, dtype=np.float32) * 0.0001
        target_q = initial_q.copy()
        target_q[:, 0, 0] += 0.04 + offset
        target_q[:, 0, 2] += 0.03
        target_q[:, 0, 7:] = np.clip(target_q[:, 0, 7:] + 0.04, lower + 0.005, upper - 0.005)
        target_qd = initial_qd.copy()
        target_qd[:, 0, 0] = 0.05
        target_qd[:, 0, 6:] = 0.02

        dirty_q = initial_q.copy()
        dirty_q[:, 0, 0] -= 0.03 + offset
        dirty_q[:, 0, 2] += 0.01
        dirty_q[:, 0, 7:] = np.clip(dirty_q[:, 0, 7:] - 0.04, lower + 0.005, upper - 0.005)
        dirty_qd = initial_qd.copy()
        dirty_qd[:, 0, 0] = -0.03
        dirty_qd[:, 0, 6:] = -0.02
        np.testing.assert_array_equal(target_q[:, 0, 3:7], initial_q[:, 0, 3:7])
        np.testing.assert_array_equal(dirty_q[:, 0, 3:7], initial_q[:, 0, 3:7])
        assert np.all(target_q[:, 0, 7:] >= lower) and np.all(target_q[:, 0, 7:] <= upper)
        assert np.all(dirty_q[:, 0, 7:] >= lower) and np.all(dirty_q[:, 0, 7:] <= upper)

        return target_q, target_qd, dirty_q, dirty_qd

    def setup(self, world_count, reset_count):
        if not wp.get_device().is_cuda:
            raise SkipNotImplemented

        self.example, self.view = self._create_example(world_count)
        self.solver = self.example.solver
        self.state = self.example.state_0
        self.state_out = self.example.state_1
        self.device = self.example.model.device
        assert not self.solver.use_mujoco_cpu
        assert self.solver.update_data_interval == 1
        assert not self.solver.enable_sleeping

        self.selected = np.zeros(world_count, dtype=bool)
        self.selected[:: world_count // reset_count] = True
        assert np.count_nonzero(self.selected) == reset_count
        self.view_mask = (
            None if reset_count == world_count else wp.array(self.selected, dtype=wp.bool, device=self.device)
        )
        self.world_mask = (
            None
            if reset_count == world_count
            else wp.array(np.append(self.selected, False), dtype=wp.bool, device=self.device)
        )
        self.fk_mask = None if self.view_mask is None else self.view.get_model_articulation_mask(self.view_mask)

        model = self.example.model
        self.target_q_np, self.target_qd_np, self.dirty_q, self.dirty_qd = self._make_joint_states(model, world_count)
        self.target_q = wp.array(self.target_q_np, dtype=wp.float32, device=self.device)
        self.target_qd = wp.array(self.target_qd_np, dtype=wp.float32, device=self.device)

        target_state = model.state()
        self.view.set_dof_positions(target_state, self.target_q)
        self.view.set_dof_velocities(target_state, self.target_qd)
        newton.eval_fk(model, target_state.joint_q, target_state.joint_qd, target_state)
        self.target_body = {
            name: getattr(target_state, name).numpy().reshape((world_count, -1)) for name in ("body_q", "body_qd")
        }

        # Compile the reset path, then make its timed input different from the target.
        self._reset()
        self._prepare_dirty(world_count)
        self._snapshot_reset_expected(world_count)
        wp.synchronize_device(self.device)

    def _reset(self):
        self.view.set_dof_positions(self.state, self.target_q, mask=self.view_mask)
        self.view.set_dof_velocities(self.state, self.target_qd, mask=self.view_mask)
        self.solver.reset(self.state, world_mask=self.world_mask, flags=newton.StateFlags.NONE)
        newton.eval_fk(self.example.model, self.state.joint_q, self.state.joint_qd, self.state, mask=self.fk_mask)

    def _prepare_dirty(self, world_count):
        self.state.joint_q.assign(self.dirty_q.reshape(self.state.joint_q.shape))
        self.state.joint_qd.assign(self.dirty_qd.reshape(self.state.joint_qd.shape))
        newton.eval_fk(self.example.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.dirty_buffers = {}
        for name in self._buffer_names:
            array = getattr(self.solver.mjw_data, name)
            snapshot = array.numpy()
            dirty = (np.arange(snapshot.size, dtype=np.float32) * 0.001 + 0.25).reshape(snapshot.shape)
            array.assign(dirty)
            self.dirty_buffers[name] = dirty

    def _snapshot_reset_expected(self, world_count):
        self.arrays = {
            "joint_q": self.state.joint_q,
            "joint_qd": self.state.joint_qd,
            **{name: getattr(self.solver.mjw_data, name) for name in self._buffer_names},
            "body_q": self.state.body_q,
            "body_qd": self.state.body_qd,
        }
        self.before = {name: array.numpy().reshape((world_count, -1)) for name, array in self.arrays.items()}
        self.expected = {
            "joint_q": self.target_q_np.reshape((world_count, -1)),
            "joint_qd": self.target_qd_np.reshape((world_count, -1)),
            **{
                name: np.zeros_like(getattr(self.solver.mjw_data, name).numpy()).reshape((world_count, -1))
                for name in self._buffer_names
            },
            **self.target_body,
        }

    def _verify_reset(self, world_count):
        for name, array in self.arrays.items():
            actual = array.numpy().reshape((world_count, -1))
            if name in ("body_q", "body_qd"):
                np.testing.assert_allclose(
                    actual[self.selected],
                    self.expected[name][self.selected],
                    atol=2e-6,
                    rtol=2e-6,
                    err_msg=f"{name}: selected worlds",
                )
            else:
                np.testing.assert_array_equal(
                    actual[self.selected], self.expected[name][self.selected], err_msg=f"{name}: selected worlds"
                )
            np.testing.assert_array_equal(
                actual[~self.selected], self.before[name][~self.selected], err_msg=f"{name}: unselected worlds"
            )


class _ResetOnly(_Reset):
    def time_reset(self, world_count, reset_count):
        self._reset()
        wp.synchronize_device(self.device)

    def teardown(self, world_count, reset_count):
        self._verify_reset(world_count)


class _ResetWithStep(_Reset):
    def setup(self, world_count, reset_count):
        super().setup(world_count, reset_count)
        # Check the reset state before any integration, then restore dirty input.
        self._reset()
        self._verify_reset(world_count)
        self._prepare_dirty(world_count)
        self.reference, _ = self._create_example(world_count)
        np.testing.assert_array_equal(self.reference.model.joint_q.numpy(), self.example.model.joint_q.numpy())
        np.testing.assert_array_equal(self.reference.model.joint_qd.numpy(), self.example.model.joint_qd.numpy())
        assert self.reference.sim_dt == self.example.sim_dt

        # Compile one complete measured path and one direct reference step.
        self._reset_and_step()
        self.reference.state_0.clear_forces()
        self.reference.solver.step(
            self.reference.state_0,
            self.reference.state_1,
            self.reference.control,
            getattr(self.reference, "contacts", None),
            self.reference.sim_dt,
        )
        self.reference.state_0, self.reference.state_1 = self.reference.state_1, self.reference.state_0

        self._prepare_dirty(world_count)
        self._snapshot_reset_expected(world_count)
        ref_state = self.reference.state_0
        expected_q = self.dirty_q.copy()
        expected_q[self.selected] = self.target_q_np[self.selected]
        expected_qd = self.dirty_qd.copy()
        expected_qd[self.selected] = self.target_qd_np[self.selected]
        ref_state.joint_q.assign(expected_q.reshape(ref_state.joint_q.shape))
        ref_state.joint_qd.assign(expected_qd.reshape(ref_state.joint_qd.shape))
        newton.eval_fk(self.reference.model, ref_state.joint_q, ref_state.joint_qd, ref_state)
        for name in self._buffer_names:
            dirty = self.dirty_buffers[name].copy()
            dirty.reshape((world_count, -1))[self.selected] = 0.0
            getattr(self.reference.solver.mjw_data, name).assign(dirty)

        # The independent expected operation never calls _reset or a selection setter.
        ref_state.clear_forces()
        self.reference.solver.step(
            ref_state,
            self.reference.state_1,
            self.reference.control,
            getattr(self.reference, "contacts", None),
            self.reference.sim_dt,
        )
        self.reference.state_0, self.reference.state_1 = self.reference.state_1, ref_state
        self.poststep_expected = {
            name: getattr(self.reference.state_0, name).numpy().reshape((world_count, -1))
            for name in ("joint_q", "joint_qd", "body_q", "body_qd")
        }
        self.poststep_expected.update(
            {
                name: getattr(self.reference.solver.mjw_data, name).numpy().reshape((world_count, -1))
                for name in ("qpos", "qvel")
            }
        )
        wp.synchronize_device(self.device)

    def _reset_and_step(self):
        self._reset()
        self.state.clear_forces()
        self.solver.step(
            self.state,
            self.state_out,
            self.example.control,
            getattr(self.example, "contacts", None),
            self.example.sim_dt,
        )
        self.state, self.state_out = self.state_out, self.state

    def time_reset_and_first_step(self, world_count, reset_count):
        self._reset_and_step()
        wp.synchronize_device(self.device)

    def teardown(self, world_count, reset_count):
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "qpos", "qvel"):
            owner = self.solver.mjw_data if name in ("qpos", "qvel") else self.state
            actual = getattr(owner, name).numpy().reshape((world_count, -1))
            assert np.isfinite(actual).all(), name
            for selected, label in ((self.selected, "selected"), (~self.selected, "unselected")):
                np.testing.assert_allclose(
                    actual[selected],
                    self.poststep_expected[name][selected],
                    atol=2e-5,
                    rtol=2e-5,
                    err_msg=f"{name}: {label} worlds after first step",
                )


class FullResetHumanoidMuJoCo(_ResetOnly):
    params = [[2048], [2048]]


class FastPartialResetStepHumanoidMuJoCo(_ResetWithStep):
    params = [[2048], [512]]
