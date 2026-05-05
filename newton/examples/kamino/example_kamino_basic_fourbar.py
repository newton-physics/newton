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
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / 0.0025))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0

        # Load the basic four-bar mechanism either from USD or by manually building it
        # with the builder API, depending on the command-line argument `--from-usd`
        if args is not None and args.from_usd:
            # Load the basic four-bar USD and add it to the builder
            asset_file = get_kamino_basics_asset("boxes_fourbar.usda")
            robot_builder.add_usd(
                asset_file,
                joint_ordering=None,
                force_show_colliders=True,
                force_position_velocity_actuation=True,
                enable_self_collisions=False,
                hide_collision_shapes=False,
            )
        else:
            # Manually build the basic four-bar mechanism using the builder API
            basics.build_boxes_fourbar(builder=robot_builder)

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)

        # Set ad-hoc softer joint PD gains to make the
        # four-bar more "squishy", i.e. more compliant
        self.model.joint_target_ke.fill_(1.0)
        self.model.joint_target_kd.fill_(0.001)

        # Create and configure settings for SolverKamino and the collision detector
        solver_config = newton.solvers.SolverKamino.Config.from_model(self.model)
        solver_config.use_collision_detector = True
        solver_config.use_fk_solver = True
        solver_config.collision_detector.pipeline = "primitive"
        solver_config.collision_detector.max_contacts = 32 * self.model.world_count
        solver_config.dynamics.preconditioning = True
        solver_config.padmm.primal_tolerance = 1e-4
        solver_config.padmm.dual_tolerance = 1e-4
        solver_config.padmm.compl_tolerance = 1e-4
        solver_config.padmm.max_iterations = 200
        solver_config.padmm.rho_0 = 0.1
        solver_config.padmm.use_acceleration = True
        solver_config.padmm.warmstart_mode = "containers"
        solver_config.padmm.contact_warmstart_method = "geom_pair_net_force"

        # Create the Kamino solver for the given model
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

        # Reset the simulation state to a valid initial configuration above the ground
        self.base_q = wp.zeros(shape=(self.world_count,), dtype=wp.transformf)
        q_b = wp.quat_identity(dtype=wp.float32)
        q_base = wp.transformf((0.0, 0.0, 0.1), q_b)
        q_base = np.array(q_base)
        q_base = np.tile(q_base, (self.world_count, 1))
        for w in range(self.world_count):
            q_base[w, :3] += np.array([0.0, 0.0, 0.2]) * float(w)
        self.base_q.assign(q_base)
        self.solver.reset(state_out=self.state_0, base_q=self.base_q)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(-0.5, -1.0, 0.2)
            pitch = -5.0
            yaw = 70.0
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
        pass  # TODO: Add some assertions here once we have a more meaningful test scenario

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--from-usd",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Load the basic four-bar mechanism from USD (otherwise build it manually).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
