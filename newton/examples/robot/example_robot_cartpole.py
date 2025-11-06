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
# Example Robot Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a USD stage using newton.ModelBuilder.add_usd().
#
# Command: python -m newton.examples robot_cartpole --num-worlds 100
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, num_worlds=8):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        self.viewer = viewer

        cartpole = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(cartpole)
        cartpole.default_shape_cfg.density = 100.0
        cartpole.default_joint_cfg.armature = 0.1
        cartpole.default_body_armature = 0.1

        cartpole.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )
        # set initial joint positions
        cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]

        builder = newton.ModelBuilder()
        builder.replicate(cartpole, self.num_worlds, spacing=(1.0, 2.0, 0.0))

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

        # Evaluating forward kinematics is needed only for maximal-coordinate solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

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
        self.viewer.end_frame()

    def test(self):
        num_bodies_per_world = self.model.body_count // self.num_worlds
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart is at ground level and has correct orientation",
            lambda q, qd: q[2] == 0.0 and newton.utils.vec_allclose(q.q, wp.quat_identity()),
            indices=[i * num_bodies_per_world for i in range(self.num_worlds)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart only moves along y direction",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and qd[2] == 0.0
            and wp.length_sq(wp.spatial_bottom(qd)) == 0.0,
            indices=[i * num_bodies_per_world for i in range(self.num_worlds)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 only has y-axis linear velocity and x-axis angular velocity",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and qd[2] == 0.0
            and abs(qd[3]) > 0.3
            and qd[4] == 0.0
            and qd[5] == 0.0,
            indices=[i * num_bodies_per_world + 1 for i in range(self.num_worlds)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole2 only has yz-plane linear velocity and x-axis angular velocity",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and abs(qd[2]) > 0.05
            and abs(qd[3]) > 0.2
            and qd[4] == 0.0
            and qd[5] == 0.0,
            indices=[i * num_bodies_per_world + 2 for i in range(self.num_worlds)],
        )
        qd = self.state_0.body_qd.numpy()
        world0_cart_vel = wp.spatial_vector(*qd[0])
        world0_pole1_vel = wp.spatial_vector(*qd[1])
        world0_pole2_vel = wp.spatial_vector(*qd[2])
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart velocities match across worlds",
            lambda q, qd: newton.utils.vec_allclose(qd, world0_cart_vel),
            indices=[i * num_bodies_per_world for i in range(self.num_worlds)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 velocities match across worlds",
            lambda q, qd: newton.utils.vec_allclose(qd, world0_pole1_vel),
            indices=[i * num_bodies_per_world + 1 for i in range(self.num_worlds)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole2 velocities match across worlds",
            lambda q, qd: newton.utils.vec_allclose(qd, world0_pole2_vel),
            indices=[i * num_bodies_per_world + 2 for i in range(self.num_worlds)],
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=100, help="Total number of simulated worlds.")
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds)

    newton.examples.run(example, args)
