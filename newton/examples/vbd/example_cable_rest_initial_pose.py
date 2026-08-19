# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Rest Initial Pose
#
# Shows two cables whose structural rest and simulation initial poses differ:
# - left: straight rest pose, helical initial pose
# - right: helical rest pose, straight initial pose
#
# Each initial segment has the same length as its corresponding rest segment,
# isolating bend/twist mismatch from initial stretch/shear.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_rest_initial_pose
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_rest_initial_pose --test --viewer null
#
###########################################################################

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.num_segments = 60
        self.cable_radius = 0.015
        self.helix_radius = 0.22
        self.helix_axis_length = 2.6
        self.helix_turns = 2.5

        builder = newton.ModelBuilder()
        cable_cfg = builder.default_shape_cfg.copy()
        cable_cfg.density = 800.0
        cable_cfg.ke = 5.0e4
        cable_cfg.kd = 0.0
        cable_cfg.mu = 0.6
        cable_cfg.has_shape_collision = True
        cable_cfg.has_particle_collision = False

        cable_height = self.helix_radius + self.cable_radius
        left_center = wp.vec3(0.0, -0.45, cable_height)
        right_center = wp.vec3(0.0, 0.45, cable_height)

        left_initial_points = self._create_helix_points(left_center)
        left_rest_points = self._create_length_matched_straight_points(left_initial_points, left_center)

        right_rest_points = self._create_helix_points(right_center)
        right_initial_points = self._create_length_matched_straight_points(right_rest_points, right_center)

        self.cable_body_ids = [
            *self._add_cable(
                builder,
                initial_points=left_initial_points,
                rest_points=left_rest_points,
                cfg=cable_cfg,
                label="straight_rest_helix_initial",
            ),
            *self._add_cable(
                builder,
                initial_points=right_initial_points,
                rest_points=right_rest_points,
                cfg=cable_cfg,
                label="helix_rest_straight_initial",
            ),
        ]

        builder.add_ground_plane()
        builder.color()

        self.model = builder.finalize()
        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.solver = newton.solvers.SolverVBD(self.model, iterations=8)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, 0.0, 5.8), pitch=-89.0, yaw=90.0)

        self.capture()

    def _create_helix_points(self, center: wp.vec3) -> list[wp.vec3]:
        points = []
        for i in range(self.num_segments + 1):
            u = i / self.num_segments
            theta = 2.0 * math.pi * self.helix_turns * u
            points.append(
                wp.vec3(
                    float(center[0]) + self.helix_axis_length * (u - 0.5),
                    float(center[1]) + self.helix_radius * math.cos(theta),
                    float(center[2]) + self.helix_radius * math.sin(theta),
                )
            )
        return points

    @staticmethod
    def _create_length_matched_straight_points(points: list[wp.vec3], center: wp.vec3) -> list[wp.vec3]:
        segment_lengths = [float(wp.length(p1 - p0)) for p0, p1 in pairwise(points)]
        x = float(center[0]) - 0.5 * sum(segment_lengths)
        straight_points = [wp.vec3(x, float(center[1]), float(center[2]))]
        for segment_length in segment_lengths:
            x += segment_length
            straight_points.append(wp.vec3(x, float(center[1]), float(center[2])))
        return straight_points

    def _add_cable(
        self,
        builder: newton.ModelBuilder,
        *,
        initial_points: list[wp.vec3],
        rest_points: list[wp.vec3],
        cfg: newton.ModelBuilder.ShapeConfig,
        label: str,
    ) -> list[int]:
        initial_quaternions = newton.utils.create_parallel_transport_cable_quaternions(initial_points)
        rest_quaternions = newton.utils.create_parallel_transport_cable_quaternions(rest_points)
        body_ids, _joint_ids = builder.add_rod(
            positions=initial_points,
            quaternions=initial_quaternions,
            rest_positions=rest_points,
            rest_quaternions=rest_quaternions,
            radius=self.cable_radius,
            cfg=cfg,
            stretch_stiffness=1.0e6,
            stretch_damping=0.0,
            bend_stiffness=4.0e2,
            bend_damping=5.0e-2,
            label=label,
            body_frame_origin="com",
        )
        return body_ids

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
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
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()[self.cable_body_ids]
        body_qd = self.state_0.body_qd.numpy()[self.cable_body_ids]
        assert np.isfinite(body_q).all(), "Non-finite cable body transforms"
        assert np.isfinite(body_qd).all(), "Non-finite cable body velocities"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
