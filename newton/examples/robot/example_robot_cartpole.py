# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a USD stage using newton.ModelBuilder.add_usd().
#
# Command: python -m newton.examples robot_cartpole --world-count 100
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count

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
        builder.replicate(cartpole, self.world_count, spacing=(1.0, 2.0, 0.0))

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
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))

        # Set camera to view all the cartpoles
        self.viewer.set_camera(
            pos=wp.vec3(7.3, -14.0, 2.3),
            pitch=-5.0,
            yaw=-225.0,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 90.0

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

    def test_final(self):
        num_bodies_per_world = self.model.body_count // self.world_count
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart is at ground level and has correct orientation",
            lambda q, qd: q[2] == 0.0 and newton.math.vec_allclose(q.q, wp.quat_identity()),
            indices=[i * num_bodies_per_world for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart only moves along y direction",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and qd[2] == 0.0
            and wp.length_sq(wp.spatial_bottom(qd)) == 0.0,
            indices=[i * num_bodies_per_world for i in range(self.world_count)],
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
            indices=[i * num_bodies_per_world + 1 for i in range(self.world_count)],
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
            indices=[i * num_bodies_per_world + 2 for i in range(self.world_count)],
        )
        qd = self.state_0.body_qd.numpy()
        world0_cart_vel = wp.spatial_vector(*qd[0])
        world0_pole1_vel = wp.spatial_vector(*qd[1])
        world0_pole2_vel = wp.spatial_vector(*qd[2])
        # Replicated GPU worlds can drift by a few ulps in body twists.
        world_velocity_atol = 1e-6
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart velocities match across worlds",
            lambda q, qd: newton.math.vec_allclose(qd, world0_cart_vel, atol=world_velocity_atol),
            indices=[i * num_bodies_per_world for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 velocities match across worlds",
            lambda q, qd: newton.math.vec_allclose(qd, world0_pole1_vel, atol=world_velocity_atol),
            indices=[i * num_bodies_per_world + 1 for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole2 velocities match across worlds",
            lambda q, qd: newton.math.vec_allclose(qd, world0_pole2_vel, atol=world_velocity_atol),
            indices=[i * num_bodies_per_world + 2 for i in range(self.world_count)],
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=100)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
