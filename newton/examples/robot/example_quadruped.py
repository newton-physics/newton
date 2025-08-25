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
# Example Quadruped
#
# Shows how to set up a simulation of a quadruped articulation
# from URDF using newton.ModelBuilder.add_urdf().
#
# Command: python -m newton.examples quadruped --num-envs 100
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, num_envs=4):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        self.viewer = viewer

        quadruped = newton.ModelBuilder()
        quadruped.default_body_armature = 0.01
        quadruped.default_joint_cfg.armature = 0.01
        quadruped.default_joint_cfg.mode = newton.JointMode.TARGET_POSITION
        quadruped.default_joint_cfg.target_ke = 2000.0
        quadruped.default_joint_cfg.target_kd = 20.0
        quadruped.default_shape_cfg.ke = 1.0e4
        quadruped.default_shape_cfg.kd = 1.0e2
        quadruped.default_shape_cfg.kf = 1.0e2
        quadruped.default_shape_cfg.mu = 1.0
        quadruped.add_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.7)),
            floating=True,
            enable_self_collisions=False,
        )
        quadruped.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
        quadruped.joint_target[-12:] = quadruped.joint_q[-12:]

        builder = newton.ModelBuilder()
        builder.replicate(quadruped, self.num_envs, spacing=(3, 2, 0))

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

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
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=100, help="Total number of simulated environments.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_envs)

    newton.examples.run(example)
