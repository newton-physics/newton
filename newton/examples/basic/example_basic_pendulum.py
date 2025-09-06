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
# Example Basic Pendulum
#
# Shows how to set up a simulation of a simple double pendulum using the
# newton.ModelBuilder() class.
#
# Command: python -m newton.examples basic_pendulum
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        builder.add_articulation(key="pendulum")

        hx = 1.0
        hy = 0.1
        hz = 0.1

        # create first link
        link_0 = builder.add_body()
        builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

        link_1 = builder.add_body()
        builder.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

        # add joints
        builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        # add ground plane
        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

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

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser with common arguments
    parser = newton.examples.create_parser(add_help=True)
    parser.description = "Shows a simple pendulum simulation using Newton physics."

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example)
