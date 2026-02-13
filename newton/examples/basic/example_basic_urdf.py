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
# Example Basic URDF
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
# from a URDF using the newton.ModelBuilder().
# Note this example does not include a trained policy.
#
# Users can pick bodies by right-clicking and dragging with the mouse.
#
# Command: python -m newton.examples basic_urdf
#
###########################################################################


import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, num_worlds, args=None):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        self.viewer = viewer

        quadruped = newton.ModelBuilder()

        # set default parameters for the quadruped
        quadruped.default_body_armature = 0.01
        quadruped.default_joint_cfg.armature = 0.01
        quadruped.default_joint_cfg.target_ke = 2000.0
        quadruped.default_joint_cfg.target_kd = 1.0
        quadruped.default_shape_cfg.ke = 1.0e4
        quadruped.default_shape_cfg.kd = 1.0e2
        quadruped.default_shape_cfg.kf = 1.0e2
        quadruped.default_shape_cfg.mu = 1.0

        # parse the URDF file
        quadruped.add_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
            ignore_inertial_definitions=True,  # Use geometry-based inertia for stability
        )

        # set initial joint positions
        quadruped.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
        quadruped.joint_target_pos[-12:] = quadruped.joint_q[-12:]

        # use "scene" for the entire set of worlds
        scene = newton.ModelBuilder()

        # use the builder.replicate() function to create N copies of the world
        scene.replicate(quadruped, self.num_worlds)
        scene.add_ground_plane()

        # finalize model
        self.model = scene.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        # put graph capture into it's own function
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "quadruped links are not moving too fast",
            lambda q, qd: max(abs(qd)) < 0.15,
        )

        bodies_per_world = self.model.body_count // self.num_worlds
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "quadrupeds have reached the terminal height",
            lambda q, qd: wp.abs(q[2] - 0.46) < 0.01,
            # only select the root body of each world
            indices=[i * bodies_per_world for i in range(self.num_worlds)],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=100, help="Total number of simulated worlds.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create viewer and run
    example = Example(viewer, args.num_worlds, args)

    newton.examples.run(example, args)
