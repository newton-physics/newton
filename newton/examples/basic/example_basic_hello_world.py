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
# python -m newton.examples basic_hello_world
# uv run newton/examples/basic/example_basic_hello_world.py
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        # setup simulation time and timestep
        self.sim_time = 0.0
        self.sim_dt = 1.0 / 60.0

        self.viewer = viewer

        # create a pendulum model using the builder
        #   by default, the builder uses up_axis=newton.Axis.Z and gravity=-9.81 m/s^2
        #   but we make it explicit here for clarity (see newton.ModelBuilder for more details).
        pendulum = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        pendulum.add_articulation(key="pendulum")

        hx = 1.0
        hy = 0.1
        hz = 0.1

        # create pendulum link
        pendulum_link = pendulum.add_body()
        pendulum.add_shape_box(pendulum_link, hx=hx, hy=hy, hz=hz)

        # add joint to world
        pendulum.add_joint_revolute(
            parent=-1,  # parent is world
            child=pendulum_link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        # set reduced-coordinate initial angles and velocities
        pendulum.joint_q[0] = 0.0
        pendulum.joint_qd[0] = -1.0

        # finalize model
        #   this method transfers all data to the memory of the target device (eg. gpu) ready for simulation.
        self.model = pendulum.finalize()

        # setup viewer, also works for null viewer
        self.viewer.set_model(self.model)

        # create solver.
        #   see newton/docs/solvers.rst for more details.
        self.solver = newton.solvers.SolverXPBD(self.model)

        # create state objects for simulation
        #   state_0 is initialized with the initial configuration given in the model description.
        #   the attributes of state_0, like body_q, body_qd, etc., are all on the target device (eg. gpu).
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # create control and contact objects for simulation
        #   unused in this example, but presented here for completeness
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # capture graph if a cuda device is available to improve performance.
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        # clear forces
        self.state_0.clear_forces()

        # solve for the next state
        self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)

        # swap states
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        # body_qd is a warp array of shape (body_count, ) and type spatial_vector that lives on the target device (eg. gpu)
        # to access the data via slicing (eg. array[0,0:3]) from the host (eg. cpu), we need to create a copy of the warp array
        # to the host using the .numpy() method.
        # - for more details about the spatial vector conventions that are used in Newton, see newton/docs/conventions.rst.
        # - for more details about warp arrays and interoperability with numpy and pytorch, see the notebook tutorials from warp:
        #   https://github.com/NVIDIA/warp?tab=readme-ov-file#running-notebooks
        print(f"[Time {self.sim_time:.2f}s] Pendulum angular velocity {self.state_0.body_qd.numpy()[0, 1]}")

        self.sim_time += self.sim_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer)

    newton.examples.run(example)
