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
# Example Robot Humanoid
#
# Shows how to set up a simulation of a humanoid articulation
# from MJCF using newton.ModelBuilder.add_mjcf().
#
# Command: python -m newton.examples robot_humanoid --num-worlds 16
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton import ActuatorMode


class Example:
    def __init__(self, viewer, num_worlds=4, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        self.viewer = viewer

        humanoid = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(humanoid)
        humanoid.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        humanoid.default_shape_cfg.ke = 5.0e4
        humanoid.default_shape_cfg.kd = 5.0e2
        humanoid.default_shape_cfg.kf = 1.0e3
        humanoid.default_shape_cfg.mu = 0.75

        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")

        humanoid.add_mjcf(
            mjcf_filename,
            ignore_names=["floor", "ground"],
            xform=wp.transform(wp.vec3(0, 0, 1.3)),
            parse_sites=False,  # AD: remove once asset is fixed
        )

        for i in range(len(humanoid.joint_target_ke)):
            humanoid.joint_target_ke[i] = 150
            humanoid.joint_target_kd[i] = 5
            humanoid.joint_act_mode[i] = int(ActuatorMode.POSITION_VELOCITY)

        builder = newton.ModelBuilder()
        builder.replicate(humanoid, self.num_worlds)

        builder.add_ground_plane()

        self.model = builder.finalize()
        use_mujoco_contacts = args.use_mujoco_contacts if args is not None else False
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=100,
            nconmax=50,
            use_mujoco_contacts=use_mujoco_contacts,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics for collision detection
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline from command-line args (default: CollisionPipelineUnified with EXPLICIT)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
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

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.01,
        )
        threshold = 0.1
        if self.sim_time < 0.2:
            threshold = 1.0
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all humanoids have come to a rest",
            lambda q, qd: max(abs(qd)) < threshold,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=4, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds, args)

    newton.examples.run(example, args)
