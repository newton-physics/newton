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
# Example Sim Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a USD stage using newton.ModelBuilder().
#
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
    def __init__(self, viewer, num_envs=8):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = 10  # renamed from num_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps  # renamed from dt

        # unpack any example specific args
        self.num_envs = num_envs

        # save a reference to the viewer
        self.viewer = viewer

        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1

        newton.utils.parse_usd(
            newton.examples.get_asset("cartpole.usda"),
            articulation_builder,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        builder = newton.ModelBuilder()

        positions = newton.examples.compute_env_offsets(num_envs, env_offset=(1.0, 2.0, 0.0))

        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(positions[i], wp.quat_identity()))

            # joint initial positions
            builder.joint_q[-3:] = [0.0, 0.3, 0.0]

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model)
        # self.solver = newton.solvers.SolverSemiImplicit(self.model, joint_attach_ke=1600.0, joint_attach_kd=20.0)
        # self.solver = newton.solvers.SolverFeatherstone(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # we do not need to evaluate contacts for this example
        self.contacts = None

        # ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ensure this is called at the end of the Example constructor
        self.viewer.set_model(self.model)

        # put graph capture into it's own function
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
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
    # Create parser that inherits common arguments and adds example-specific ones
    # keep example options short, don't overload user with options
    # device, viewer type, and other options are created by default
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=100, help="Total number of simulated environments.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args.num_envs)

    newton.examples.run(example)
