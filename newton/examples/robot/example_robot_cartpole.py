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

import numpy as np
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
        self.solver_type = args.solver

        self.viewer = viewer

        cartpole = newton.ModelBuilder()
        if self.solver_type == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(cartpole)
        else:
            newton.solvers.SolverMuJoCo.register_custom_attributes(cartpole)
        cartpole.default_shape_cfg.density = 100.0
        cartpole.default_joint_cfg.armature = 0.1

        cartpole.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )
        for shape in range(cartpole.shape_count):
            cartpole.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_SHAPES

        # apply additional inertia to the bodies for better stability
        body_armature = 0.1
        for body in range(cartpole.body_count):
            inertia_np = np.asarray(cartpole.body_inertia[body], dtype=np.float32).reshape(3, 3)
            inertia_np += np.eye(3, dtype=np.float32) * body_armature
            cartpole.body_inertia[body] = wp.mat33(inertia_np)

        # set initial joint positions
        cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]

        builder = newton.ModelBuilder()
        builder.replicate(cartpole, self.world_count, spacing=(1.0, 2.0, 0.0))

        # finalize model
        self.model = builder.finalize()

        if self.solver_type == "kamino":
            solver_config = newton.solvers.SolverKamino.Config.from_model(
                self.model, dynamics_solver="dvi", sparse_dynamics=False, sparse_jacobian=False
            )
            solver_config.dvi.max_alternating_iterations = 8
            solver_config.dvi.bilateral_solve_interval = 8
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
        else:
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
        constrained_atol = 1.0e-6
        num_bodies_per_world = self.model.body_count // self.world_count
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart is at ground level and has correct orientation",
            lambda q, qd: (
                abs(q[2]) < constrained_atol
                and newton.math.vec_allclose(q.q, wp.quat_identity(), atol=constrained_atol)
            ),
            indices=[i * num_bodies_per_world for i in range(self.world_count)],
        )
        # fmt: off
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart only moves along y direction",
            lambda q, qd: abs(qd[0]) < constrained_atol
            and abs(qd[1]) > 0.05
            and abs(qd[2]) < constrained_atol
            and wp.length_sq(wp.spatial_bottom(qd)) < constrained_atol * constrained_atol,
            indices=[i * num_bodies_per_world for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 only has y-axis linear velocity and x-axis angular velocity",
            lambda q, qd: abs(qd[0]) < constrained_atol
            and abs(qd[1]) > 0.05
            and abs(qd[2]) < constrained_atol
            and abs(qd[3]) > 0.3
            and abs(qd[4]) < constrained_atol
            and abs(qd[5]) < constrained_atol,
            indices=[i * num_bodies_per_world + 1 for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole2 only has yz-plane linear velocity and x-axis angular velocity",
            lambda q, qd: abs(qd[0]) < constrained_atol
            and abs(qd[1]) > 0.05
            and abs(qd[2]) > 0.05
            and abs(qd[3]) > 0.2
            and abs(qd[4]) < constrained_atol
            and abs(qd[5]) < constrained_atol,
            indices=[i * num_bodies_per_world + 2 for i in range(self.world_count)],
        )
        # fmt: on
        qd = self.state_0.body_qd.numpy()
        world0_cart_vel = wp.spatial_vector(*qd[0])
        world0_pole1_vel = wp.spatial_vector(*qd[1])
        world0_pole2_vel = wp.spatial_vector(*qd[2])
        # Replicated GPU worlds can accumulate small relative differences in body twists.
        world_velocity_atol = 1e-6
        world_velocity_rtol = 2.5e-4
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart velocities match across worlds",
            lambda q, qd: newton.math.vec_allclose(
                qd, world0_cart_vel, rtol=world_velocity_rtol, atol=world_velocity_atol
            ),
            indices=[i * num_bodies_per_world for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 velocities match across worlds",
            lambda q, qd: newton.math.vec_allclose(
                qd, world0_pole1_vel, rtol=world_velocity_rtol, atol=world_velocity_atol
            ),
            indices=[i * num_bodies_per_world + 1 for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole2 velocities match across worlds",
            lambda q, qd: newton.math.vec_allclose(
                qd, world0_pole2_vel, rtol=world_velocity_rtol, atol=world_velocity_atol
            ),
            indices=[i * num_bodies_per_world + 2 for i in range(self.world_count)],
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("--solver", choices=["mujoco", "kamino"], default="mujoco")
        parser.set_defaults(world_count=100)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
