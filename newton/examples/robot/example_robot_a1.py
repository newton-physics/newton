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


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.world_count = args.world_count
        self.viewer = viewer

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
        self.model = builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((2.0, 2.0, 0.0))

        self.capture()

    def capture(self):
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the selected A1 model remains finite while holding its pose."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        if not np.isfinite(joint_q).all() or not np.isfinite(joint_qd).all():
            raise ValueError("A1 joint state contains non-finite values")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("--model", choices=A1_ASSET_PATHS, default="bimanual")
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
