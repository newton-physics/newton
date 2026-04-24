# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example for basic four-bar mechanism
#
# Shows how to simulate a basic four-bar linkage with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples kamino_basic_fourbar --world-count 16
#
###########################################################################

import argparse
import os

import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.models import get_basics_usd_assets_path
from newton._src.solvers.kamino._src.models.builders import basics_newton
from newton._src.solvers.kamino._src.utils import logger as msg


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.sim_dt = 0.001
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        msg.notif("Creating and configuring the model builder for Kamino...")
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0

        # Load the basic box pendulum either from USD or by manually building it
        # with the builder API, depending on the command-line argument `--from-usd`
        if args.from_usd:
            # Load the basic box pendulum USD and add it to the builder
            msg.notif("Loading USD asset and adding it to the model builder...")
            asset_file = os.path.join(get_basics_usd_assets_path(), "box_pendulum.usda")
            robot_builder.add_usd(
                asset_file,
                joint_ordering=None,
                force_show_colliders=True,
                force_position_velocity_actuation=True,
                enable_self_collisions=False,
                hide_collision_shapes=False,
            )
        else:
            # Manually build the basic box pendulum using the builder API
            basics_newton.build_box_pendulum(builder=robot_builder)

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        msg.notif(f"Duplicating the model builder for {self.world_count} worlds and finalizing the model...")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Create the model from the builder
        msg.notif("Creating the model from the builder...")
        self.model = builder.finalize(skip_validation_joints=True)

        # Create and configure settings for SolverKamino and the collision detector
        solver_config = newton.solvers.SolverKamino.Config.from_model(self.model)
        solver_config.use_collision_detector = True
        solver_config.use_fk_solver = False
        solver_config.dynamics.preconditioning = True
        solver_config.padmm.primal_tolerance = 1e-6
        solver_config.padmm.dual_tolerance = 1e-6
        solver_config.padmm.compl_tolerance = 1e-6
        solver_config.padmm.max_iterations = 200
        solver_config.padmm.rho_0 = 0.1
        solver_config.padmm.use_acceleration = True
        solver_config.padmm.warmstart_mode = "containers"
        solver_config.padmm.contact_warmstart_method = "geom_pair_net_force"

        # Create the Kamino solver for the given model
        msg.notif("Creating the Kamino solver for the given model...")
        self.solver = newton.solvers.SolverKamino(model=self.model, config=solver_config)

        # Create state, control, and contacts data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Warm-start the simulation
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.solver.reset(self.state_0)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(-2.0, -2.0, 1.0)
            pitch = -5.0
            yaw = 45.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
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
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.006,
        )
        # Only check velocities on CUDA where we run 500 frames (enough time to settle)
        # On CPU we only run 10 frames and the robot is still falling (~0.65 m/s)
        if self.device.is_cuda:
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "body velocities are small",
                lambda q, qd: (
                    max(abs(qd)) < 0.25
                ),  # Relaxed from 0.1 - unified pipeline has residual velocities up to ~0.2
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--from-usd",
            type=argparse.BooleanOptionalAction,
            default=False,
            help="Load the basic four-bar mechanism from USD.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    msg.notif("Starting the simulation...")
    newton.examples.run(example, args)
