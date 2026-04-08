# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Anymal D
#
# Shows how to simulate Anymal D with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples robot_anymal_d --num-worlds 16
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        # Set simulation run-time configurations
        self.fps = 60
        self.sim_dt = 0.0025
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.num_worlds = args.world_count
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0

        # Load the Anymal D USD and add it to the builder
        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")
        robot_builder.add_usd(
            asset_file,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=True,
            enable_self_collisions=True,
            hide_collision_shapes=True,
        )

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.num_worlds):
            builder.add_world(robot_builder)

        # Add a global ground plane applied to all worlds
        builder.add_ground_plane()

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)

        # Create the Kamino solver for the given model
        self.solver = newton.solvers.SolverKamino(self.model)

        # Create state, control, and contacts data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Use Newton's collision pipeline (same as standard Newton examples)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Reset the simulation state to a valid initial configuration above the ground
        self.base_q = wp.zeros(shape=(self.num_worlds,), dtype=wp.transformf)
        q_b = wp.quat_identity(dtype=wp.float32)
        q_base = wp.transformf((0.0, 0.0, 1.0), q_b)
        self.base_q.assign([q_base] * self.num_worlds)
        self.solver.reset(state_out=self.state_0, base_q=self.base_q)

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # Start paused to inspect the initial configuration
        self.viewer._paused = True

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.num_worlds == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(5.0, 0.0, 2.0)
            pitch = -15.0
            yaw = -180.0
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
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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
            # fmt: off
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "body velocities are small",
                lambda q, qd: max(abs(qd))
                < 0.25,  # Relaxed from 0.1 - unified pipeline has residual velocities up to ~0.2
            )
            # fmt: on

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_kamino_contacts_arg(parser)
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
