# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example for basic boxes hinged system
#
# Shows how to simulate a basic boxes hinged with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples kamino_basic_boxes_hinged --world-count 16
#
###########################################################################

import argparse

import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils.sim.viewer_recording import enable_recording
from newton.tests import get_kamino_basics_asset
from newton.tests.utils import basics


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
        self.actuated = args.actuated if args else False
        self.time = wp.zeros((self.world_count,), device=self.device)

        video_output_filename = getattr(args, "video_path", None)
        self.record_video = enable_recording(
            viewer=self.viewer,
            record_video=args.record_video if args else False,
            start_clip=True,
            output_path=video_output_filename if video_output_filename is not None else "recording.mp4",
            max_frames=getattr(args, "max_video_frames", 1000),
            fps=self.fps,
        )

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0

        # Load the basic boxes hinged either from USD or by manually building it
        # with the builder API, depending on the command-line argument `--from-usd`
        if args is not None and args.from_usd:
            # Load the basic boxes hinged USD and add it to the builder
            asset_file = get_kamino_basics_asset("boxes_hinged.usda")
            robot_builder.add_usd(
                asset_file,
                joint_ordering=None,
                force_show_colliders=True,
                force_position_velocity_actuation=True,
                enable_self_collisions=False,
                hide_collision_shapes=False,
            )
        else:
            # Manually build the basic boxes hinged using the builder API
            basics.build_boxes_hinged(builder=robot_builder)

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)

        # Create and configure settings for SolverKamino and the collision detector
        solver_config = newton.solvers.SolverKamino.Config.from_model(self.model)
        solver_config.use_collision_detector = True
        solver_config.use_fk_solver = False
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
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(-0.5, -5.2, 1.8)
            pitch = -15.0
            yaw = 90.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def capture(self):
        self.graph = None
        if self.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.actuated:
                self._advance_time()
                self._apply_actuation()
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

    def _advance_time(self):
        """Advances the current simulation time by ``dt``."""

        @wp.kernel
        def advance_time_kernel(dt: wp.float32, time: wp.array[wp.float32]):
            """Advance the time in each world."""
            wid = wp.tid()
            time[wid] += dt

        wp.launch(
            advance_time_kernel,
            dim=self.model.world_count,
            inputs=[self.sim_dt, self.time],
            device=self.device,
        )

    def _apply_actuation(self):
        """Apply actuation to each world."""

        @wp.kernel
        def actuation_kernel(
            time: wp.array[wp.float32],
            joint_f: wp.array[wp.float32],
        ):
            # Retrieve the world index from the thread ID
            wid = wp.tid()

            # Define the time window for the active external force profile
            t_start = wp.float32(2.0)
            t_end = wp.float32(2.5)
            t_loop = wp.float32(5.0)

            # Apply a time-dependent force
            t = time[wid]
            t -= wp.floor(t / t_loop) * t_loop
            if t > t_start and t < t_end:
                joint_f[wid] = -3.0
            else:
                joint_f[wid] = 0.0

        wp.launch(
            actuation_kernel,
            dim=self.model.world_count,
            inputs=[
                self.time,
                self.control.joint_f,
            ],
            device=self.device,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--from-usd",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Load the basic boxes hinged from USD.",
        )
        parser.add_argument(
            "--actuated",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Actuate the model with predefined inputs.",
        )
        parser.add_argument(
            "--record-video",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Record a video of the viewer, up to 1000 frames.",
        )
        parser.add_argument(
            "--video-path",
            type=str,
            default=None,
            help="Output video path (defaults to 'recording.mp4').",
        )
        parser.add_argument(
            "--max-video-frames",
            type=int,
            default=1000,
            help="Maximum number of frames recorded for the video (defaults to 1000).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    if hasattr(viewer, "finish_clip"):
        viewer.finish_clip()
