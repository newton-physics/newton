# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Robot Anymal D
#
# Shows how to simulate Anymal D with multiple worlds using SolverMuJoCo.
#
# Command: python -m newton.examples robot_anymal_d --num-worlds 16
#
###########################################################################

import mujoco
import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
    def __init__(self, viewer, num_worlds=8):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        self.viewer = viewer

        self.device = wp.get_device()

        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e2, limit_kd=1.0e0, friction=1e-5
        )
        articulation_builder.default_shape_cfg.ke = 5.0e4
        articulation_builder.default_shape_cfg.kd = 5.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")
        articulation_builder.add_usd(
            asset_file,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_non_physics_prims=True,
            hide_collision_shapes=True,
        )

        articulation_builder.joint_q[:3] = [0.0, 0.0, 0.62]
        if len(articulation_builder.joint_q) > 6:
            articulation_builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]

        for i in range(len(articulation_builder.joint_dof_mode)):
            articulation_builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            articulation_builder.joint_target_ke[i] = 150
            articulation_builder.joint_target_kd[i] = 5

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.num_worlds):
            builder.add_builder(articulation_builder)

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
            impratio=100,
            iterations=100,
            ls_iterations=50,
            ncon_per_world=20,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # ensure this is called at the end of the Example constructor
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((3.0, 3.0, 0.0))

        # put graph capture into it's own function
        self.capture()

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
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
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.006,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are small",
            lambda q, qd: max(abs(qd)) < 0.1,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=8, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds)

    newton.examples.run(example, args)
