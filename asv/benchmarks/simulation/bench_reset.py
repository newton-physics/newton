# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Absolute elapsed reset time for the batched selection cartpole on MuJoCo Warp.

Both series time SolverMuJoCo.reset on state_0, including CUDA completion.
Full means all worlds within this contract: joint_q and joint_qd are restored
from model defaults; MuJoCo qacc_warmstart, qfrc_applied, xfrc_applied, act and
ctrl are cleared. Empty actuator buffers are included but contain no values.
Body poses/velocities, Newton body forces and Control are not restored. MuJoCo
qpos/qvel and derived body state are synchronized by the next simulation step,
which is excluded here (update_data_interval=1, sleeping disabled). This is not
a reset of the entire Example, its second state buffer or its simulation clock.

The partial reset selects every fourth world: 32 of 128 (25%). Model creation,
a simulation step, reset compilation/warm-up, deterministic dirtying and host
verification are outside timing. No CUDA graph wraps the measured reset.
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
        self.device = self.example.model.device
        assert not self.solver.use_mujoco_cpu
        assert self.solver.update_data_interval == 1
        assert not self.solver.enable_sleeping

        self.selected = np.zeros(world_count, dtype=bool)
        self.selected[:: world_count // reset_count] = True
        assert np.count_nonzero(self.selected) == reset_count
        self.world_mask = (
            None
            if reset_count == world_count
            else wp.array(np.append(self.selected, False), dtype=wp.bool, device=self.device)
        )
        self.solver.reset(self.state, world_mask=self.world_mask)

        model = self.example.model
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
            if name in ("joint_q", "joint_qd"):
                initial = getattr(model, name).numpy()
            else:
                initial = np.zeros_like(array.numpy())
            # Every populated row differs from its reset target and other worlds.
            offset = np.arange(initial.size, dtype=np.float32).reshape(initial.shape) * 0.001 + 0.25
            dirty = initial + offset
            array.assign(dirty)
            self.before[name] = dirty.reshape((world_count, -1))
            self.expected[name] = initial.reshape((world_count, -1))

        control = self.example.control
        self.untouched = {
            "body_q": self.state.body_q,
            "body_qd": self.state.body_qd,
            "body_f": self.state.body_f,
            "joint_f": control.joint_f,
            "joint_target_q": control.joint_target_q,
            "joint_target_qd": control.joint_target_qd,
        }
        self.untouched_before = {name: array.numpy() for name, array in self.untouched.items()}
        # Drain the warm-up reset and all preparation before ASV starts its timer.
        wp.synchronize_device(self.device)

    def time_reset(self, world_count, reset_count):
        self.solver.reset(self.state, world_mask=self.world_mask)
        wp.synchronize_device(self.device)

    def teardown(self, world_count, reset_count):
        for name, array in self.arrays.items():
            actual = array.numpy().reshape((world_count, -1))
            np.testing.assert_array_equal(
                actual[self.selected], self.expected[name][self.selected], err_msg=f"{name}: selected worlds"
            )
            np.testing.assert_array_equal(
                actual[~self.selected], self.before[name][~self.selected], err_msg=f"{name}: unselected worlds"
            )
        for name, array in self.untouched.items():
            np.testing.assert_array_equal(
                array.numpy(), self.untouched_before[name], err_msg=f"{name}: outside reset scope"
            )


class FastFullResetCartpoleMuJoCo(_ResetCartpole):
    params = [[128], [128]]


class FastPartialResetCartpoleMuJoCo(_ResetCartpole):
    params = [[128], [32]]
