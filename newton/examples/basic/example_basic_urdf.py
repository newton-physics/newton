# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

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

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "xpbd"

        self.viewer = viewer

        quadruped = newton.ModelBuilder()

        # set default parameters for the quadruped
        quadruped.default_joint_cfg.armature = 0.01

        if self.solver_type == "vbd":
            quadruped.default_joint_cfg.target_ke = 1.0e4
            quadruped.default_joint_cfg.target_kd = 0.0
            quadruped.default_shape_cfg.ke = 5.0e5
            quadruped.default_shape_cfg.kd = 0.0
            quadruped.default_shape_cfg.mu = 1.0
        else:
            quadruped.default_joint_cfg.target_ke = 2000.0
            quadruped.default_joint_cfg.target_kd = 1.0
            quadruped.default_shape_cfg.mu = 1.0

        # parse the URDF file
        quadruped.add_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
            ignore_inertial_definitions=True,  # Use geometry-based inertia for stability
        )

        # apply additional inertia to the bodies for better stability
        body_armature = 0.01
        for body in range(quadruped.body_count):
            inertia_np = np.asarray(quadruped.body_inertia[body], dtype=np.float32).reshape(3, 3)
            inertia_np += np.eye(3, dtype=np.float32) * body_armature
            inertia = wp.mat33(inertia_np)
            quadruped.body_inertia[body] = inertia

        # set initial joint positions
        quadruped.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
        quadruped.joint_target_pos[-12:] = quadruped.joint_q[-12:]

        # use "scene" for the entire set of worlds
        scene = newton.ModelBuilder()

        # use the builder.replicate() function to create N copies of the world
        scene.replicate(quadruped, self.world_count)

        scene.add_ground_plane(cfg=quadruped.default_shape_cfg)
        if self.solver_type == "vbd":
            scene.color()

        self.model = scene.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        if self.solver_type == "vbd":
            self.update_step_interval = 1
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=2,
            )
        else:
            self.update_step_interval = 1
            self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

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
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Collision detection and contact refresh cadence.
            refresh_contacts = (substep % self.update_step_interval) == 0
            if refresh_contacts:
                self.model.collide(self.state_0, self.contacts)

            if self.solver_type == "vbd":
                self.solver.set_rigid_history_update(refresh_contacts)

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

        bodies_per_world = self.model.body_count // self.world_count
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "quadrupeds have reached the terminal height",
            lambda q, qd: wp.abs(q[2] - 0.46) < 0.01,
            # only select the root body of each world
            indices=[i * bodies_per_world for i in range(self.world_count)],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=100)
        parser.add_argument(
            "--solver",
            type=str,
            default="xpbd",
            choices=["vbd", "xpbd"],
            help="Solver type: xpbd (default) or vbd",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
