# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Absolute elapsed RL-style reset time for batched MuJoCo Warp cartpoles.

Both series use 128 worlds. The full reset selects all 128; the partial reset
selects every fourth world (32/128, 25%). The timed operation writes prepared,
nondefault joint positions and velocities through ArticulationView, clears the
selected worlds' persistent MuJoCo solver buffers with SolverMuJoCo.reset, runs
forward kinematics for their body poses and velocities, and waits for CUDA.
The cleared buffers are qacc_warmstart, qfrc_applied, xfrc_applied, act and ctrl;
empty actuator buffers contain no values. With update_data_interval=1 and
sleeping disabled, the next step synchronizes MuJoCo qpos/qvel from Newton state.
Model construction, target generation, warm-up, dirty-state preparation and
verification are outside timing. No CUDA graph wraps the measured reset. This
does not reset the Example clock, its second state buffer, or model parameters.
Run with ASV: ``uvx --with virtualenv asv run --launch-method spawn HEAD^! -b ResetCartpole``.
The standalone run_benchmark helper repeats warm-up without setup and is unsuitable here.
"""

import os
import sys

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented

import newton
import newton.examples
from newton.examples.selection.example_selection_cartpole import Example

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_config import pr_gate_repeat


class _ResetCartpole:
    repeat = pr_gate_repeat(10)
    number = 1
    # asv_runner 0.2.x repeats warm-up calls without re-running setup.
    warmup_time = 0
    param_names = ["world_count", "reset_count"]

    def setup(self, world_count, reset_count):
        if not wp.get_device().is_cuda:
            raise SkipNotImplemented

        args = newton.examples.default_args(Example.create_parser())
        args.world_count = world_count
        args.solver = "mujoco"
        self.example = Example(newton.viewer.ViewerNull(), args)
        self.example.step()
        self.solver = self.example.solver
        self.state = self.example.state_0
        self.view = self.example.cartpoles
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
        target_q = self.view.get_dof_positions(model).numpy().copy()
        target_qd = self.view.get_dof_velocities(model).numpy().copy()
        target_q += np.array([0.2, 0.125, -0.125], dtype=np.float32)
        target_q[:, :, 0] += np.linspace(-0.4, 0.4, world_count, dtype=np.float32)[:, None]
        target_qd += np.array([0.05, -0.075, 0.1], dtype=np.float32)
        self.target_q = wp.array(target_q, dtype=wp.float32, device=self.device)
        self.target_qd = wp.array(target_qd, dtype=wp.float32, device=self.device)

        # Prepare the selected-body FK reference and compile the measured path.
        target_state = model.state()
        self.view.set_dof_positions(target_state, self.target_q)
        self.view.set_dof_velocities(target_state, self.target_qd)
        newton.eval_fk(model, target_state.joint_q, target_state.joint_qd, target_state)
        self.time_reset(world_count, reset_count)

        data = self.solver.mjw_data
        self.arrays = {
            "joint_q": self.state.joint_q,
            "joint_qd": self.state.joint_qd,
            "qacc_warmstart": data.qacc_warmstart,
            "qfrc_applied": data.qfrc_applied,
            "xfrc_applied": data.xfrc_applied,
            "act": data.act,
            "ctrl": data.ctrl,
        }
        self.expected = {}
        self.before = {}
        for name, array in self.arrays.items():
            if name == "joint_q":
                initial = target_q.reshape(array.shape)
            elif name == "joint_qd":
                initial = target_qd.reshape(array.shape)
            else:
                initial = np.zeros_like(array.numpy())
            # Every populated row differs from its reset target and other worlds.
            offset = np.arange(initial.size, dtype=np.float32).reshape(initial.shape) * 0.001 + 0.25
            dirty = initial + offset
            array.assign(dirty)
            self.before[name] = dirty.reshape((world_count, -1))
            self.expected[name] = initial.reshape((world_count, -1))

        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)
        for name in ("body_q", "body_qd"):
            self.arrays[name] = getattr(self.state, name)
            self.before[name] = self.arrays[name].numpy().reshape((world_count, -1))
            self.expected[name] = getattr(target_state, name).numpy().reshape((world_count, -1))

        # Drain the warm-up and dirty-state preparation before ASV starts its timer.
        wp.synchronize_device(self.device)

    def time_reset(self, world_count, reset_count):
        self.view.set_dof_positions(self.state, self.target_q, mask=self.view_mask)
        self.view.set_dof_velocities(self.state, self.target_qd, mask=self.view_mask)
        self.solver.reset(self.state, world_mask=self.world_mask, flags=newton.StateFlags.NONE)
        newton.eval_fk(self.example.model, self.state.joint_q, self.state.joint_qd, self.state, mask=self.fk_mask)
        wp.synchronize_device(self.device)

    def teardown(self, world_count, reset_count):
        """Verify the selected reset state and preserve unselected worlds."""
        for name, array in self.arrays.items():
            actual = array.numpy().reshape((world_count, -1))
            if name in ("body_q", "body_qd"):
                np.testing.assert_allclose(
                    actual[self.selected], self.expected[name][self.selected], err_msg=f"{name}: selected worlds"
                )
            else:
                np.testing.assert_array_equal(
                    actual[self.selected], self.expected[name][self.selected], err_msg=f"{name}: selected worlds"
                )
            np.testing.assert_array_equal(
                actual[~self.selected], self.before[name][~self.selected], err_msg=f"{name}: unselected worlds"
            )


class FastFullResetCartpoleMuJoCo(_ResetCartpole):
    params = [[128], [128]]


class FastPartialResetCartpoleMuJoCo(_ResetCartpole):
    params = [[128], [32]]
