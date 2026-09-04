# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example for simulating all basic models as a single heterogeneous multi-world model with SolverKamino.
#
# Command: python -m newton.examples kamino_basic_heterogeneous
#
###########################################################################

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests import get_kamino_basics_asset
from newton.tests.utils import basics


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.sim_dt = 0.0025
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.scenario = getattr(args, "scenario", "all") if args else "all"
        self.dynamics_solver = getattr(args, "dynamics_solver", "padmm") if args else "padmm"
        self.viewer = viewer
        self.device = wp.get_device()

        # Define a helper function to load each basic model from USD and
        # add it to the builder, with consistent settings for all models
        def load_basic_asset_from_usd(asset_file: str) -> newton.ModelBuilder:
            asset_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverKamino.register_custom_attributes(asset_builder)
            asset_builder.default_shape_cfg.margin = 0.0
            asset_builder.default_shape_cfg.gap = 0.0
            asset_builder.add_usd(
                asset_file,
                joint_ordering=None,
                force_show_colliders=True,
                force_position_velocity_actuation=True,
                enable_self_collisions=False,
                hide_collision_shapes=False,
            )
            return asset_builder

        # Load the heterogeneous basic models either from USD or manually using the
        # model builder API, depending on the command-line argument `--from-usd`
        builder = newton.ModelBuilder()
        if args is not None and args.from_usd:
            if args.z_offset != 0.0:
                raise ValueError("--z-offset requires --no-from-usd.")
            # Load all basic USD assets and add them to the builder
            asset_names = [
                "boxes_fourbar",
                "boxes_nunchaku",
                "boxes_hinged",
                "box_pendulum",
                "box_on_plane",
                "cartpole",
            ]
            if self.scenario != "all":
                asset_names = [self.scenario]
            for asset_name in asset_names:
                asset_file = get_kamino_basics_asset(f"{asset_name}.usda")
                builder.add_world(builder=load_basic_asset_from_usd(asset_file))
        else:
            # Manually build the heterogeneous basic models using the builder API
            if self.scenario == "all":
                basics.make_basics_heterogeneous_builder(builder=builder, ground=True)
            else:
                build_scenario = getattr(basics, f"build_{self.scenario}")
                build_scenario(builder=builder, z_offset=args.z_offset if args else 0.0, ground=True)

        # Create the model from the builder
        builder.request_contact_attributes("force")  # For contact visualization
        self.model = builder.finalize(skip_validation_joints=True)

        # Create and configure settings for SolverKamino and the collision detector
        solver_config = newton.solvers.SolverKamino.Config.from_model(
            self.model,
            dynamics_solver=self.dynamics_solver,
        )
        solver_config.use_collision_detector = True
        solver_config.use_fk_solver = True
        solver_config.collision_detector.pipeline = "primitive" if args is None or args.from_usd else "unified"
        solver_config.collision_detector.max_contacts = 32 * self.model.world_count
        solver_config.dynamics.preconditioning = True
        if self.dynamics_solver == "padmm":
            solver_config.padmm.primal_tolerance = 1e-4
            solver_config.padmm.dual_tolerance = 1e-4
            solver_config.padmm.compl_tolerance = 1e-4
            solver_config.padmm.max_iterations = 200
            solver_config.padmm.rho_0 = 0.1
            solver_config.padmm.use_acceleration = True
            solver_config.padmm.warmstart_mode = "containers"
            solver_config.padmm.contact_warmstart_method = "geom_pair_net_force"
        else:
            tolerance = getattr(args, "lox_tolerance", 1.0e-7) if args else 1.0e-7
            solver_config.lox.position_tolerance = tolerance
            solver_config.lox.rotation_tolerance = tolerance
        # Create the Kamino solver for the given model
        self.solver = newton.solvers.SolverKamino(model=self.model, config=solver_config)

        # Create state, control, and contacts data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets(spacing=(5.0, 5.0, 0.0))

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if hasattr(self.viewer, "set_camera"):
            if self.scenario == "all":
                camera_pos = wp.vec3(0.0, -15.0, 1.6)
                pitch = -1.5
                yaw = 92.0
            else:
                camera_pos = wp.vec3(1.5, -2.5, 1.0)
                pitch = -10.0
                yaw = 115.0
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
        # Since rendering is called after stepping the simulation, the previous and next
        # states correspond to self.state_1 and self.state_0 due to the reference swaps,
        # so contacts are rendered with self.state_1 to match the body positions at the
        # time of contact generation.
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_1)
        self.viewer.end_frame()

    def test_final(self):
        assert self.graph is not None
        arrays = (self.state_0.body_q, self.state_0.body_qd, self.state_0.joint_q, self.state_0.joint_qd)
        assert all(np.isfinite(array.numpy()).all() for array in arrays)
        if self.dynamics_solver == "lox":
            solver = self.solver._solver_kamino.solver_fd
            assert not solver.world_failed.numpy().any()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--from-usd",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Load the heterogeneous basic models from USD (otherwise build them manually).",
        )
        parser.add_argument(
            "--scenario",
            choices=(
                "all",
                "box_on_plane",
                "box_pendulum",
                "boxes_hinged",
                "boxes_nunchaku",
                "boxes_fourbar",
                "cartpole",
            ),
            default="all",
            help="Run all basic worlds or one selected model.",
        )
        parser.add_argument(
            "--z-offset",
            type=float,
            default=0.0,
            help="Initial vertical offset for a manually built selected model [m].",
        )
        parser.add_argument(
            "--dynamics-solver",
            choices=("padmm", "lox"),
            default="padmm",
            help="Kamino rigid-body dynamics backend.",
        )
        parser.add_argument(
            "--lox-tolerance",
            type=float,
            default=1.0e-7,
            help="Position and rotation tolerance for the LOX backend.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
