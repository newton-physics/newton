# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Anymal D
#
# Shows how to simulate Anymal D with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples kamino_robot_anymal_d --world-count 16
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / 0.0025))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        self.use_kamino_contacts = args.use_kamino_contacts if args else False
        self.linear_solver_type = getattr(args, "linear_solver_type", "LLTB") if args else "LLTB"
        self.linear_solver_kwargs = getattr(args, "linear_solver_kwargs", {}) if args else {}
        self.dynamics_solver = getattr(args, "dynamics_solver", "padmm") if args else "padmm"
        self.actuated = getattr(args, "actuated", False) if args else False
        self.max_iterations = getattr(args, "max_iterations", 25) if args else 25
        self.projection_iterations = getattr(args, "projection_iterations", 3) if args else 3
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0
        robot_builder.request_contact_attributes("force")  # For contact visualization

        # Load the Anymal D USD and add it to the builder
        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")
        robot_builder.add_usd(
            asset_file,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,
            enable_self_collisions=True,
            hide_collision_shapes=True,
        )

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.request_contact_attributes("force")
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Add a global ground plane applied to all worlds
        builder.add_ground_plane()

        # Create the model from the builder
        self.model = builder.finalize()
        self.model.rigid_contact_max = 1000 * self.world_count

        if self.actuated:
            # Match the public ANYmal D pose controller without actuating the floating base.
            target_ke = self.model.joint_target_ke.numpy()
            target_kd = self.model.joint_target_kd.numpy()
            target_mode = self.model.joint_target_mode.numpy()
            joint_type = self.model.joint_type.numpy()
            joint_dof_start = self.model.joint_qd_start.numpy()
            for joint in range(self.model.joint_count):
                if joint_type[joint] != newton.JointType.REVOLUTE:
                    continue
                dof_slice = slice(joint_dof_start[joint], joint_dof_start[joint + 1])
                target_ke[dof_slice] = 150.0
                target_kd[dof_slice] = 5.0
                target_mode[dof_slice] = int(newton.JointTargetMode.POSITION_VELOCITY)
            self.model.joint_target_ke.assign(target_ke)
            self.model.joint_target_kd.assign(target_kd)
            self.model.joint_target_mode.assign(target_mode)

        # Create the Kamino solver for the given model
        self.config = newton.solvers.SolverKamino.Config.from_model(
            self.model,
            dynamics_solver=self.dynamics_solver,
        )
        self.config.use_collision_detector = self.use_kamino_contacts
        self.config.dynamics.linear_solver_type = self.linear_solver_type
        self.config.dynamics.linear_solver_kwargs = self.linear_solver_kwargs
        if self.dynamics_solver == "padmm":
            self.config.padmm.warmstart_mode = "none"
            self.config.padmm.use_graph_conditionals = getattr(args, "use_graph_conditionals", True) if args else True
        else:
            # The simple row metric settles the passive robot more reliably
            # with a moderate penalty and projected structural feedback.
            self.config.lox.joint_penalty_scale = 20.0
            self.config.lox.max_iterations = self.max_iterations
            self.config.lox.projection_iterations = self.projection_iterations
        self.solver = newton.solvers.SolverKamino(self.model, config=self.config)

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Configure CD components based on whether we want to use Kamino's
        # internal contact solver or Newton's collision pipeline
        if not self.use_kamino_contacts:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
        else:
            self.collision_pipeline = None
            self.contacts = newton.CollisionPipeline(self.model).contacts()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Reset the simulation state to a valid initial configuration above the ground
        self.base_q = wp.zeros(shape=(self.world_count,), dtype=wp.transformf)
        q_b = wp.quat_identity(dtype=wp.float32)
        q_base = wp.transformf((0.0, 0.0, 1.0), q_b)
        self.base_q.assign([q_base] * self.world_count)
        reset_config = newton.solvers.SolverKamino.ResetConfig(
            base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(self.base_q),
        )
        self.solver.reset(state=self.state_0, config=reset_config)

        # Capture with CUDA graphs on GPU or APIC graphs on CPU.
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(5.0, 0.0, 2.0)
            pitch = -15.0
            yaw = -180.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def capture(self):
        self.graph = None
        if not self.device.is_cuda or not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if not self.use_kamino_contacts:
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            else:
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.solver.update_contacts(self.contacts, self.state_0)
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

    def _test_state_finite(self):
        invalid = []
        for name, array in (
            ("body_q", self.state_0.body_q),
            ("body_qd", self.state_0.body_qd),
            ("joint_q", self.state_0.joint_q),
            ("joint_qd", self.state_0.joint_qd),
            ("contact_force", self.contacts.rigid_contact_force),
        ):
            if not np.isfinite(array.numpy()).all():
                invalid.append(name)
        if self.dynamics_solver == "lox":
            solver = self.solver._solver_kamino.solver_fd
            if solver.world_failed.numpy().any():
                invalid.append("LOX backend status")
        assert not invalid, f"Non-finite or failed ANYmal state: {', '.join(invalid)}"

    def test_post_step(self):
        self._test_state_finite()

    def test_final(self):
        self._test_state_finite()
        if self.dynamics_solver == "lox":
            structural_residual = self.solver._solver_kamino.solver_fd.problem.structural_residual.numpy()
            assert np.max(np.abs(structural_residual), initial=0.0) < 0.02, (
                "ANYmal structural joint residual must settle below 0.02 m or rad"
            )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.006,
        )
        # Only check velocities on CUDA, where example tests run enough frames to settle.
        # Short CPU smoke runs may still be falling when they finish.
        if self.device.is_cuda:
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "body velocities are small",
                lambda q, qd: (
                    max(abs(qd)) < 0.25
                ),  # Relaxed from 0.1 - unified pipeline has residual velocities up to ~0.2
            )
        assert int(self.contacts.rigid_contact_count.numpy()[0]) > 0, "ANYmal must finish in contact with the ground"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_kamino_contacts_arg(parser)
        parser.add_argument(
            "--dynamics-solver",
            choices=("padmm", "lox"),
            default="padmm",
            help="Kamino rigid-body dynamics backend.",
        )
        parser.add_argument(
            "--actuated",
            action="store_true",
            help="Enable position-velocity drives that hold the imported joint pose.",
        )
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=25,
            help="Maximum LOX splitting iterations per solve.",
        )
        parser.add_argument(
            "--projection-iterations",
            type=int,
            default=3,
            help="Sequential projection sweeps per LOX splitting iteration.",
        )
        parser.add_argument(
            "--linear-solver-type",
            choices=("LLTB", "LLTBRCM", "CR"),
            default="LLTB",
            type=str.upper,
            help="Kamino dynamics linear solver to use.",
        )
        parser.add_argument(
            "--no-graph-conditionals",
            dest="use_graph_conditionals",
            action="store_false",
            help="Disable graph conditional loops in Kamino PADMM.",
        )
        parser.set_defaults(world_count=1)
        parser.set_defaults(use_kamino_contacts=True)
        parser.set_defaults(use_graph_conditionals=True)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
