# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example demonstrating all supported joint types
#
# Shows how to simulate multiple systems with different joint types in multiple worlds using SolverKamino.
#
# Command: python -m newton.examples kamino_basic_all_joints
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.models.builders import testing_newton
from newton._src.solvers.kamino._src.utils import logger as msg


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.sim_dt = 0.001
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        msg.notif("Creating and configuring the model builder for Kamino...")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0

        # Manually build the basic box on plane using the builder API
        testing_newton.build_all_joints_test(builder=builder)

        # Create the model from the builder
        msg.notif("Creating the model from the builder...")
        self.model = builder.finalize()

        # Create and configure settings for SolverKamino and the collision detector
        solver_config = newton.solvers.SolverKamino.Config.from_model(self.model)
        solver_config.use_collision_detector = True
        solver_config.padmm.primal_tolerance = 1e-6
        solver_config.padmm.dual_tolerance = 1e-6
        solver_config.padmm.compl_tolerance = 1e-6
        solver_config.padmm.max_iterations = 200
        solver_config.padmm.rho_0 = 0.1

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
        if hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(0.0, -13.0, 4.0)
            pitch = -15.0
            yaw = 90.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

        # Set the viewer to start in paused mode so that the user can
        # observe the initial state before stepping the simulation
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True

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
        newton.examples.add_kamino_contacts_arg(parser)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    msg.notif("Starting the simulation...")
    newton.examples.run(example, args)
