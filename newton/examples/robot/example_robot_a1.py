# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot A1
#
# Shows how to load and position-control the OneRobotics A1 left arm,
# right arm, or bimanual stand from structured USD assets.
#
# Command: python -m newton.examples robot_a1 --model bimanual
#          python -m newton.examples robot_a1 --model left
#          python -m newton.examples robot_a1 --model right
#          python -m newton.examples robot_a1 --demo wave
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode

A1_ASSET_PATHS = {
    "left": Path("usd/a1_left/onerobotics_a1_left.usda"),
    "right": Path("usd/a1_right/onerobotics_a1_right.usda"),
    "bimanual": Path("usd/a1_bimanual_stand/onerobotics_a1_bimanual_stand.usda"),
}
A1_DEMOS = ("hold", "joint_sine", "wave")


@wp.kernel
def update_a1_joint_targets(
    base_targets: wp.array[wp.float32],
    amplitudes: wp.array[wp.float32],
    phases: wp.array[wp.float32],
    motion_time: wp.array[wp.float32],
    angular_frequency: wp.float32,
    joint_targets: wp.array[wp.float32],
):
    """Update A1 position targets with smooth sinusoidal offsets."""
    dof = wp.tid()
    phase = phases[dof]
    offset = amplitudes[dof] * (wp.sin(angular_frequency * motion_time[0] + phase) - wp.sin(phase))
    joint_targets[dof] = base_targets[dof] + offset


@wp.kernel
def advance_a1_motion_time(motion_time: wp.array[wp.float32], dt: wp.float32):
    """Advance the device-side trajectory clock."""
    motion_time[0] = motion_time[0] + dt


class Example:
    def __init__(self, viewer, args):
        """Initialize the selected A1 model, solver, state, and viewer."""
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.world_count = args.world_count
        self.demo = args.demo
        self.viewer = viewer
        self.device = wp.get_device()

        a1 = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(a1)
        a1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1)

        asset_root = newton.utils.download_asset("onerobotics_a1")
        a1.add_usd(
            str(asset_root / A1_ASSET_PATHS[args.model]),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )

        for dof_id in range(a1.joint_dof_count):
            a1.joint_target_ke[dof_id] = 500.0
            a1.joint_target_kd[dof_id] = 25.0
            a1.joint_target_mode[dof_id] = int(JointTargetMode.POSITION)
            a1.joint_target_q[dof_id] = a1.joint_q[dof_id]

        builder = newton.ModelBuilder()
        builder.replicate(a1, self.world_count, spacing=(2.0, 2.0, 0.0))

        base_targets = np.asarray(builder.joint_target_q, dtype=np.float32)
        amplitudes = np.zeros_like(base_targets)
        phases = np.zeros_like(base_targets)
        dofs_per_world = a1.joint_dof_count

        if self.demo == "joint_sine":
            local_amplitudes = np.full(dofs_per_world, 0.15, dtype=np.float32)
            local_phases = np.linspace(0.0, np.pi, dofs_per_world, endpoint=False, dtype=np.float32)
            amplitudes[:] = np.tile(local_amplitudes, self.world_count)
            phases[:] = np.tile(local_phases, self.world_count)
        elif self.demo == "wave":
            local_amplitudes = np.zeros(dofs_per_world, dtype=np.float32)
            local_phases = np.zeros(dofs_per_world, dtype=np.float32)
            for arm_start in range(0, dofs_per_world, 7):
                arm_dofs = min(7, dofs_per_world - arm_start)
                wave_dofs = min(3, arm_dofs)
                start = arm_start + arm_dofs - wave_dofs
                direction = 1.0 if (arm_start // 7) % 2 == 0 else -1.0
                local_amplitudes[start : start + wave_dofs] = (
                    direction * np.array([0.20, 0.30, 0.40], dtype=np.float32)[-wave_dofs:]
                )
                local_phases[start : start + wave_dofs] = np.linspace(0.0, np.pi / 2.0, wave_dofs, dtype=np.float32)
            amplitudes[:] = np.tile(local_amplitudes, self.world_count)
            phases[:] = np.tile(local_phases, self.world_count)

        self.model = builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.base_targets = wp.array(base_targets, dtype=wp.float32, device=self.device)
        self.motion_amplitudes = wp.array(amplitudes, dtype=wp.float32, device=self.device)
        self.motion_phases = wp.array(phases, dtype=wp.float32, device=self.device)
        self.motion_time = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.angular_frequency = 0.8
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((2.0, 2.0, 0.0))

        self.capture()

    def capture(self):
        """Capture one simulation step as a Warp graph when supported."""
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        """Advance the A1 simulation by one frame using fixed substeps."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.demo != "hold":
                wp.launch(
                    update_a1_joint_targets,
                    dim=self.model.joint_dof_count,
                    inputs=[
                        self.base_targets,
                        self.motion_amplitudes,
                        self.motion_phases,
                        self.motion_time,
                        self.angular_frequency,
                    ],
                    outputs=[self.control.joint_target_q],
                    device=self.device,
                )
                wp.launch(
                    advance_a1_motion_time,
                    dim=1,
                    inputs=[self.motion_time, self.sim_dt],
                    device=self.device,
                )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the simulation and update the elapsed simulation time."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current A1 state in the configured viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the selected A1 model remains finite while holding its pose."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        if not np.isfinite(joint_q).all() or not np.isfinite(joint_qd).all():
            raise ValueError("A1 joint state contains non-finite values")
        if self.demo != "hold":
            joint_targets = self.control.joint_target_q.numpy()
            base_targets = self.base_targets.numpy()
            if np.allclose(joint_targets, base_targets):
                raise ValueError(f"A1 {self.demo} demo did not update any joint targets")

    @staticmethod
    def create_parser():
        """Create the command-line parser for selecting the A1 configuration."""
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("--model", choices=A1_ASSET_PATHS, default="bimanual")
        parser.add_argument("--demo", choices=A1_DEMOS, default="joint_sine")
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
