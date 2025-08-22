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
# Example Hello World
#
# Shows how to set up a simulation of a simple pendulum using the
# newton.ModelBuilder() class. This is a minimal example that does not
# require any additional dependencies.
#
# Example usage:
# uv run newton/examples/basic/example_basic_hello_world.py
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self):
        # setup simulation time and timestep
        self.sim_time = 0.0
        self.sim_dt = 1.0 / 60.0

        # create double pendulum model using the builder
        builder = newton.ModelBuilder()

        builder.add_articulation(key="double_pendulum")

        hx = 1.0
        hy = 0.1
        hz = 0.1

        # create pendulum link
        pendulum_link = builder.add_body()
        builder.add_shape_box(pendulum_link, hx=hx, hy=hy, hz=hz)

        # add joint to world
        builder.add_joint_revolute(
            parent=-1,  # parent is world
            child=pendulum_link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        # finalize model
        self.model = builder.finalize()

        # set gravity to 9.81 m/s^2 in the negative z direction
        self.model.gravity = wp.vec3(0.0, 0.0, -9.81)

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # set reduced-coordinate initial angles and velocities
        host_model_joint_q = self.model.joint_q.to("cpu")
        host_model_joint_q.numpy()[0] = 0
        wp.copy(self.model.joint_q, host_model_joint_q)

        host_model_joint_qd = self.model.joint_qd.to("cpu")
        host_model_joint_qd.numpy()[0] = -1.0
        wp.copy(self.model.joint_qd, host_model_joint_qd)

        # Set initial pendulum position and velocity from reduced coordinates
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
        self.solver.step(self.state_0, self.state_0, None, None, self.sim_dt)
        # copy states for graph capture because we are only taking a single step
        # wp.copy(self.state_0.body_q, self.state_1.body_q)
        # wp.copy(self.state_0.body_qd, self.state_1.body_qd)

    def step(self):
        print(f"[Time {self.sim_time:.2f}s] Pendulum angular velocity {self.state_0.body_qd.numpy()[0, 1]}")
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.sim_dt

    def test(self):
        pass


if __name__ == "__main__":
    # Create viewer and run
    example = Example()

    # run 100 steps
    for _ in range(100):
        example.step()
