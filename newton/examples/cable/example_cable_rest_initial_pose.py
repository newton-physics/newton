# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Rest Initial Pose
#
# Shows that cable rest pose and simulation initial pose are separate:
# - left cable: straight rest pose, spiral initial pose
# - right cable: spiral rest pose, straight initial pose
#
# In both cases, rest and initial segment lengths are matched so the demo is
# about bend/twist rest-shape mismatch, not accidental initial stretch.
#
# Command: python -m newton.examples cable_rest_initial_pose
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples


def _pin_body(builder: newton.ModelBuilder, body: int) -> None:
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_iterations = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.num_segments = 60
        self.cable_radius = 0.015
        self.spiral_radius = 0.22
        self.bend_stiffness = 4.0e2
        self.bend_damping = 5.0e-3
        self.stretch_stiffness = 1.0e6
        self.stretch_damping = 0.0

        builder = newton.ModelBuilder()

        cable_cfg = builder.default_shape_cfg.copy()
        cable_cfg.density = 800.0
        cable_cfg.ke = 5.0e4
        cable_cfg.kd = 0.0
        cable_cfg.mu = 0.6
        cable_cfg.has_shape_collision = True
        cable_cfg.has_particle_collision = False

        cable_height = self.cable_radius
        left_start = wp.vec3(-1.35, -0.45, cable_height)
        right_start = wp.vec3(-1.35, 0.45, cable_height)

        left_initial_points = self._create_spiral_points(left_start)
        left_rest_points, left_rest_quats = newton.utils.create_straight_cable_rest_from_initial(
            left_initial_points,
            start=left_start,
            direction=wp.vec3(1.0, 0.0, 0.0),
        )
        left_initial_xforms = newton.utils.create_cable_body_transforms(left_initial_points)

        right_rest_points = self._create_spiral_points(right_start)
        right_rest_quats = newton.utils.create_parallel_transport_cable_quaternions(right_rest_points)
        right_initial_points = newton.utils.create_straight_cable_points_from_lengths(
            right_start,
            wp.vec3(1.0, 0.0, 0.0),
            newton.utils.compute_cable_segment_lengths(right_rest_points),
        )
        right_initial_xforms = newton.utils.create_cable_body_transforms(right_initial_points)
        newton.utils.validate_cable_segment_lengths_match(right_rest_points, right_initial_points)

        self.case_body_ids: list[list[int]] = []

        left_bodies = self._add_cable(
            builder,
            left_rest_points,
            left_rest_quats,
            cable_cfg,
            label="straight_rest_spiral_initial",
        )
        right_bodies = self._add_cable(
            builder,
            right_rest_points,
            right_rest_quats,
            cable_cfg,
            label="spiral_rest_straight_initial",
        )
        self.case_body_ids.extend([left_bodies, right_bodies])

        builder.add_ground_plane()

        builder.color()
        self.model = builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_initial = self.model.state()
        self.control = self.model.control()
        pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.model.contacts(collision_pipeline=pipeline)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
        )

        newton.utils.apply_cable_body_transforms(
            [self.state_initial, self.state_0, self.state_1],
            left_bodies,
            left_initial_xforms,
            body_q_prev=self.solver.body_q_prev,
        )
        newton.utils.apply_cable_body_transforms(
            [self.state_initial, self.state_0, self.state_1],
            right_bodies,
            right_initial_xforms,
            body_q_prev=self.solver.body_q_prev,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-0.05, -2.1, 2.4), pitch=-52.0, yaw=90.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 45.0

        self.capture()

    def _create_spiral_points(self, start: wp.vec3) -> list[wp.vec3]:
        axis_length = 2.6
        turns = 2.5

        points: list[wp.vec3] = []
        for i in range(self.num_segments + 1):
            u = i / self.num_segments
            theta = 2.0 * math.pi * turns * u
            points.append(
                wp.vec3(
                    float(start[0]) + axis_length * u,
                    float(start[1]) + self.spiral_radius * (math.cos(theta) - 1.0),
                    float(start[2]),
                )
            )
        return points

    def _add_cable(
        self,
        builder: newton.ModelBuilder,
        points: list[wp.vec3],
        quaternions: list[wp.quat],
        cfg: newton.ModelBuilder.ShapeConfig,
        *,
        label: str,
    ) -> list[int]:
        body_ids, _joint_ids = builder.add_rod(
            positions=points,
            quaternions=quaternions,
            radius=self.cable_radius,
            cfg=cfg,
            stretch_stiffness=self.stretch_stiffness,
            stretch_damping=self.stretch_damping,
            bend_stiffness=self.bend_stiffness,
            bend_damping=self.bend_damping,
            label=label,
        )
        return body_ids

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def reset_simulation(self):
        self.state_0.assign(self.state_initial)
        self.state_1.assign(self.state_initial)
        self.solver.body_q_prev.assign(self.state_initial.body_q)
        self.sim_time = 0.0

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
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
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        body_ids = [body for case_bodies in self.case_body_ids for body in case_bodies]
        assert np.isfinite(body_q[body_ids]).all(), "Non-finite cable body transforms"
        assert np.isfinite(body_qd[body_ids]).all(), "Non-finite cable body velocities"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
