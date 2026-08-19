# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Initial and Rest Poses
#
# Demonstrates two cables whose initial and structural-rest poses differ:
#
# - a helical initial pose with zero intrinsic bend and twist;
# - a straight initial pose with a helical structural-rest pose.
#
# The initial and rest segment lengths match in both cases, isolating the
# angular material response from initial stretch.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.cable.example_cable_initial_rest_pose
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.cable.example_cable_initial_rest_pose --test --viewer null
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


def _create_helix_points(
    center: wp.vec3,
    *,
    num_segments: int,
    axis_length: float,
    radius: float,
    turns: float,
) -> list[wp.vec3]:
    """Create uniformly sampled helix points around the X-axis."""
    return [
        wp.vec3(
            float(center[0]) + axis_length * (i / num_segments - 0.5),
            float(center[1]) + radius * math.cos(2.0 * math.pi * turns * i / num_segments),
            float(center[2]) + radius * math.sin(2.0 * math.pi * turns * i / num_segments),
        )
        for i in range(num_segments + 1)
    ]


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_iterations = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        num_segments = 60
        cable_radius = 0.025
        helix_radius = 0.22
        helix_axis_length = 2.6
        helix_turns = 2.5
        stretch_stiffness = 1.0e6
        bend_stiffness = 4.0e2
        bend_damping = 5.0e-2

        builder = newton.ModelBuilder()
        cable_cfg = builder.default_shape_cfg.copy()
        cable_cfg.density = 800.0
        cable_cfg.ke = 5.0e4
        cable_cfg.kd = 0.0
        cable_cfg.mu = 0.6

        cable_height = cable_radius + helix_radius
        left_center = wp.vec3(0.0, -0.75, cable_height)
        right_center = wp.vec3(0.0, 0.75, cable_height)

        left_initial_points = _create_helix_points(
            left_center,
            num_segments=num_segments,
            axis_length=helix_axis_length,
            radius=helix_radius,
            turns=helix_turns,
        )
        left_initial_quaternions = newton.utils.rod_parallel_transport_quaternions(left_initial_points)
        # Keep the helical initial geometry but author zero intrinsic bend and twist.
        left_bodies, _ = builder.add_rod(
            positions=left_initial_points,
            quaternions=left_initial_quaternions,
            rest_straight=True,
            radius=cable_radius,
            cfg=cable_cfg,
            stretch_stiffness=stretch_stiffness,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            label="straight_rest_helix_initial",
            color=(0.95, 0.35, 0.12),
            body_frame_origin="com",
        )

        right_rest_points = _create_helix_points(
            right_center,
            num_segments=num_segments,
            axis_length=helix_axis_length,
            radius=helix_radius,
            turns=helix_turns,
        )
        right_rest_quaternions = newton.utils.rod_parallel_transport_quaternions(right_rest_points)
        segment_length = float(wp.length(right_rest_points[1] - right_rest_points[0]))
        right_initial_points, right_initial_quaternions = newton.utils.rod_straight_points_and_quaternions(
            start=wp.vec3(-0.5 * num_segments * segment_length, right_center[1], right_center[2]),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=num_segments * segment_length,
            num_segments=num_segments,
        )
        # Keep the straight initial geometry but author an explicit helical rest shape.
        right_bodies, _ = builder.add_rod(
            positions=right_initial_points,
            quaternions=right_initial_quaternions,
            rest_positions=right_rest_points,
            rest_quaternions=right_rest_quaternions,
            radius=cable_radius,
            cfg=cable_cfg,
            stretch_stiffness=stretch_stiffness,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            label="helix_rest_straight_initial",
            color=(0.15, 0.65, 1.0),
            body_frame_origin="com",
        )
        self.cable_bodies = [left_bodies, right_bodies]

        builder.add_ground_plane()
        builder.color(balance_colors=False)
        self.model = builder.finalize(device=args.device)

        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_compliant_alm=True,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -4.5, 4.2), pitch=-35.0, yaw=90.0)
        if hasattr(self.viewer, "camera"):
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.15))
            if hasattr(self.viewer.camera, "fov"):
                self.viewer.camera.fov = 48.0

        self.capture()

    def capture(self):
        """Capture the simulation loop for CUDA replay."""
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Advance the simulation by one frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the example by one frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify both cable simulations remain finite and bounded."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        body_indices = [body for cable_bodies in self.cable_bodies for body in cable_bodies]
        if not np.isfinite(body_q[body_indices]).all() or not np.isfinite(body_qd[body_indices]).all():
            raise ValueError("Cable state contains non-finite values.")
        if np.max(np.abs(body_q[body_indices, :3])) >= 10.0:
            raise ValueError("Cable body positions exceeded the example bounds.")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
