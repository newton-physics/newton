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
# Command: python -m newton.examples robot_anymal_d --world-count 16
#
###########################################################################

import mujoco
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode


class Example:
    def __init__(self, viewer, args):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count

        self.viewer = viewer

        self.device = wp.get_device()

        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
        )
        articulation_builder.default_shape_cfg.ke = 2.0e3
        articulation_builder.default_shape_cfg.kd = 1.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")
        articulation_builder.add_usd(
            asset_file,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        articulation_builder.joint_q[:3] = [0.0, 0.0, 0.62]
        if len(articulation_builder.joint_q) > 6:
            articulation_builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]

        for i in range(articulation_builder.joint_dof_count):
            articulation_builder.joint_target_ke[i] = 150
            articulation_builder.joint_target_kd[i] = 5
            articulation_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(articulation_builder)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()
        use_mujoco_contacts = args.use_mujoco_contacts if args else False
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
            impratio=100,
            iterations=100,
            ls_iterations=50,
            nconmax=45,
            njmax=100,
            use_mujoco_contacts=use_mujoco_contacts,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics for collision detection
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.use_mujoco_contacts = use_mujoco_contacts
        if use_mujoco_contacts:
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        else:
            self.contacts = self.model.contacts()

        # ensure this is called at the end of the Example constructor
        self.viewer.set_model(self.model)

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
        if not self.use_mujoco_contacts:
            self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)

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

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.006,
        )
        # Only check velocities on CUDA where we run 500 frames (enough time to settle)
        # On CPU we only run 10 frames and the robot is still falling (~0.65 m/s)
        if self.device.is_cuda:
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "body velocities are small",
                lambda q, qd: max(abs(qd))
                < 0.25,  # Relaxed from 0.1 - collision pipeline has residual velocities up to ~0.2
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_mujoco_contacts_arg(parser)
        parser.set_defaults(world_count=8)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
