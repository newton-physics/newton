# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot DR Legs
#
# Shows how to simulate DR Legs with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples kamino_robot_dr_legs --world-count 16
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        self.use_kamino_contacts = args.use_kamino_contacts if args else False
        self.dynamics_solver = getattr(args, "dynamics_solver", "padmm") if args else "padmm"
        self.linear_solver_type = getattr(args, "linear_solver_type", "LLTB") if args else "LLTB"
        self.linear_solver_kwargs = getattr(args, "linear_solver_kwargs", {}) if args else {}
        target_sim_dt = self.frame_dt / 12 if self.dynamics_solver == "dvi" else 0.01
        self.sim_substeps = max(1, round(self.frame_dt / target_sim_dt))
        self.sim_dt = self.frame_dt / self.sim_substeps
        # DVI benefits from early contact detection because it solves inequality
        # constraints slightly less accurately than PADMM. Contact forces remain
        # zero until the shapes overlap.
        dvi_contact_margin = 5.0e-4 if self.dynamics_solver == "dvi" else 1e-6
        self.dvi_contact_block_preconditioner = bool(getattr(args, "dvi_contact_block_preconditioner", False))
        self.dvi_contact_jacobi_omega = float(getattr(args, "dvi_contact_jacobi_omega", 0.45))
        self.dvi_contact_jacobi_relaxation = float(getattr(args, "dvi_contact_jacobi_relaxation", 0.9))
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = dvi_contact_margin
        robot_builder.default_shape_cfg.gap = 1e-2

        # Load the DR Legs USD and add it to the builder
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_meshes_and_boxes.usda")
        robot_builder.add_usd(
            asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,  # TODO @cavemor: Fails when True, investigate (doesn't have fixed joints)
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.request_contact_attributes("force")
        builder.default_shape_cfg.margin = dvi_contact_margin
        builder.default_shape_cfg.gap = 1e-2
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Add a global ground plane applied to all worlds
        builder.add_ground_plane()

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)
        self.model.rigid_contact_max = 72 * self.world_count

        # Create the Kamino solver for the given model
        self.config = newton.solvers.SolverKamino.Config.from_model(
            self.model,
            dynamics_solver=self.dynamics_solver,
        )
        self.config.use_fk_solver = True
        self.config.use_collision_detector = self.use_kamino_contacts
        self.config.dynamics.linear_solver_type = self.linear_solver_type
        self.config.dynamics.linear_solver_kwargs = self.linear_solver_kwargs
        self.config.constraints.delta = 1e-3
        self.config.padmm.max_iterations = 200
        self.config.padmm.primal_tolerance = 1e-4
        self.config.padmm.dual_tolerance = 1e-4
        self.config.padmm.compl_tolerance = 1e-4
        self.config.padmm.use_graph_conditionals = getattr(args, "use_graph_conditionals", True) if args else True
        if self.dynamics_solver == "dvi":
            self.config.use_fk_solver = False
            self.config.integrator = "moreau"
            self.config.constraints.alpha = 0.1
            self.config.constraints.beta = 0.011
            self.config.constraints.gamma = 0.015
            self.config.dynamics.preconditioning = False
            self.config.dynamics.linear_solver_type = "CR"
            self.config.dynamics.linear_solver_kwargs = {"maxiter": 9}
            self.config.sparse_dynamics = True
            self.config.sparse_jacobian = True
            self.config.dvi.max_iterations = 200
            self.config.dvi.tolerance = 1e-4
            self.config.dvi.regularization = 1e-5
            self.config.dvi.omega = 0.3
            self.config.dvi.block_iterations = 4
            self.config.dvi.contact_iterations = 2
            self.config.dvi.bilateral_solve_period = 1
            self.config.dvi.contact_jacobi_omega = self.dvi_contact_jacobi_omega
            self.config.dvi.contact_jacobi_relaxation = self.dvi_contact_jacobi_relaxation
            self.config.dvi.contact_block_preconditioner = self.dvi_contact_block_preconditioner
            self.config.dvi.contact_warmstart_method = "key_and_position_with_net_force_backup"
        self.solver = newton.solvers.SolverKamino(self.model, config=self.config)

        # Set joint armature and viscous damping for better
        # stability of the implicit joint-space PD controller
        # TODO: Remove this once we add Newton USD schemas in the model asset
        self.solver._solver_kamino._model.joints.a_j.fill_(0.011)  # Joint armature
        self.solver._solver_kamino._model.joints.b_j.fill_(0.044)  # Joint viscous damping
        self.solver._solver_kamino._model.joints.k_p_j.fill_(10.0)  # Proportional gain
        self.solver._solver_kamino._model.joints.k_d_j.fill_(2.0)  # Derivative gain

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

        # Warm-start the simulation
        if not self.use_kamino_contacts:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.solver.reset(self.state_0)

        # Reset the simulation state to a valid initial configuration above the ground
        self.base_q = wp.zeros(shape=(self.world_count,), dtype=wp.transformf)
        q_b = wp.quat_identity(dtype=wp.float32)
        q_base = wp.transformf((0.0, 0.0, 0.4), q_b)
        self.base_q.assign([q_base] * self.world_count)
        reset_config = newton.solvers.SolverKamino.ResetConfig(
            base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(base_q=self.base_q),
        )
        self.solver.reset(state=self.state_0, config=reset_config)
        self.solver.reset(state=self.state_1, config=reset_config)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.graph = None
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(1.34, 0.0, 0.25)
            pitch = -7.0
            yaw = -180.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

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
        self.viewer.log_contacts(self.contacts, self.state_1)
        self.viewer.end_frame()

    def test_final(self):
        pass  # TODO: Add some assertions here once we have a more meaningful test scenario

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_kamino_contacts_arg(parser)
        parser.add_argument(
            "--dynamics-solver",
            choices=("padmm", "dvi"),
            default="padmm",
            help="Kamino dynamics solver to use.",
        )
        parser.add_argument(
            "--dvi-contact-block-preconditioner",
            action="store_true",
            help="Use the opt-in full 3x3 contact block preconditioner for the Kamino DVI solver.",
        )
        parser.add_argument(
            "--dvi-contact-jacobi-omega",
            type=float,
            default=0.45,
            help="Step size for Kamino DVI non-colored contact Jacobi and block-preconditioned contact updates.",
        )
        parser.add_argument(
            "--dvi-contact-jacobi-relaxation",
            type=float,
            default=0.9,
            help="Solution mixing for Kamino DVI non-colored contact Jacobi and block-preconditioned contact updates.",
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
            help="Disable CUDA graph conditional nodes in Kamino PADMM.",
        )
        parser.set_defaults(world_count=1)
        parser.set_defaults(use_kamino_contacts=True)
        parser.set_defaults(use_graph_conditionals=True)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
