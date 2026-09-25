# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Adaptive Collision Scheduling
#
# Fires a fast capsule into a staggered wall of box-shaped bricks. The
# collision scheduler refreshes speculative contacts often enough to prevent
# the projectile from tunneling while preserving a fixed solver substep count.
#
# Command: python -m newton.examples adaptive_collision
#
###########################################################################

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples

FPS = 60
SUBSTEPS = 10
SPECULATIVE_GAP_MAX = 0.2

BRICK_DEPTH = 0.18
BRICK_LENGTH = 0.58
HALF_BRICK_LENGTH = 0.28
BRICK_HEIGHT = 0.24
MORTAR_GAP = 0.02
BRICK_COURSES = 7
BRICK_LAYERS = 1
BRICK_DENSITY = 1800.0

PROJECTILE_START = (-2.5, 0.0, 0.9)
PROJECTILE_SPEED = 100.0
PROJECTILE_RADIUS = 0.1
PROJECTILE_HALF_HEIGHT = 0.2
PROJECTILE_DENSITY = 1200.0

BRICK_COLORS = (
    wp.vec3(0.56, 0.12, 0.07),
    wp.vec3(0.68, 0.18, 0.09),
    wp.vec3(0.76, 0.25, 0.12),
    wp.vec3(0.62, 0.15, 0.08),
)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.use_scheduler = args.scheduler
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SUBSTEPS
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.0

        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.mu = 0.8
        ground_cfg.restitution = 0.05
        builder.add_ground_plane(cfg=ground_cfg)

        brick_cfg = builder.default_shape_cfg.copy()
        brick_cfg.density = BRICK_DENSITY
        brick_cfg.gap = 0.01
        brick_cfg.mu = 0.75
        brick_cfg.restitution = 0.08

        self.brick_bodies = []
        course_pitch = BRICK_HEIGHT + 0.25 * MORTAR_GAP
        brick_pitch = BRICK_LENGTH + MORTAR_GAP
        for course in range(BRICK_COURSES):
            z = 0.5 * BRICK_HEIGHT + course * course_pitch
            if course % 2 == 0:
                bricks = [(index * brick_pitch, BRICK_LENGTH) for index in range(-2, 3)]
            else:
                bricks = [
                    (-2.25 * brick_pitch, HALF_BRICK_LENGTH),
                    (-1.5 * brick_pitch, BRICK_LENGTH),
                    (-0.5 * brick_pitch, BRICK_LENGTH),
                    (0.5 * brick_pitch, BRICK_LENGTH),
                    (1.5 * brick_pitch, BRICK_LENGTH),
                    (2.25 * brick_pitch, HALF_BRICK_LENGTH),
                ]

            for layer in range(BRICK_LAYERS):
                x = (layer - 0.5 * (BRICK_LAYERS - 1)) * (BRICK_DEPTH + MORTAR_GAP)
                for column, (y, length) in enumerate(bricks):
                    body = builder.add_body(
                        xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()),
                        label=f"brick_{layer}_{course}_{column}",
                    )
                    builder.add_shape_box(
                        body,
                        hx=0.5 * BRICK_DEPTH,
                        hy=0.5 * length,
                        hz=0.5 * BRICK_HEIGHT,
                        cfg=brick_cfg,
                        color=BRICK_COLORS[(layer + course + 2 * column) % len(BRICK_COLORS)],
                    )
                    self.brick_bodies.append(body)

        projectile_cfg = builder.default_shape_cfg.copy()
        projectile_cfg.density = PROJECTILE_DENSITY
        projectile_cfg.gap = 0.0
        projectile_cfg.mu = 0.2
        projectile_cfg.restitution = 0.1
        self.projectile_body = builder.add_body(
            xform=wp.transform(wp.vec3(*PROJECTILE_START), wp.quat_identity()),
            label="projectile",
        )
        builder.add_shape_capsule(
            self.projectile_body,
            xform=wp.transform(
                wp.vec3(0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi),
            ),
            radius=PROJECTILE_RADIUS,
            half_height=PROJECTILE_HALF_HEIGHT,
            cfg=projectile_cfg,
            color=wp.vec3(0.08, 0.32, 0.75),
        )
        builder.body_qd[self.projectile_body] = (PROJECTILE_SPEED, 0.0, 0.0, 0.0, 0.0, 0.0)

        self.model = builder.finalize()
        self.states = (self.model.state(), self.model.state())
        self.state_0, self.state_1 = self.states
        self.control = self.model.control()
        self.initial_brick_positions = self.state_0.body_q.numpy()[self.brick_bodies, :3].copy()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            speculative_contact_gap_max=SPECULATIVE_GAP_MAX,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=10,
            rigid_contact_relaxation=0.8,
            enable_restitution=True,
        )

        self.scheduler = newton.CollisionSubstepScheduler(
            self.collision_pipeline,
            self.states,
            collision_callback=self._collide,
            substep_callback=self._substep,
            frame_dt=self.frame_dt,
            substeps=SUBSTEPS,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-4.5, -5.0, 2.7), pitch=-13.0, yaw=48.0)

        # Preallocate lazy collision-pipeline storage before the pipeline is
        # invoked from a CUDA conditional graph body.
        self._collide(self.state_0, self.frame_dt)

        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _collide(self, state, collision_dt):
        self.collision_pipeline.collide(state, self.contacts, dt=collision_dt)

    def _substep(self, state_in, state_out, dt):
        state_in.clear_forces()
        self.viewer.apply_forces(state_in)
        self.solver.step(state_in, state_out, self.control, self.contacts, dt)

    def simulate(self):
        if self.use_scheduler:
            self.scheduler.step()
        else:
            self._collide(self.state_0, self.frame_dt)
            for substep in range(SUBSTEPS):
                self._substep(self.states[substep % 2], self.states[1 - substep % 2], self.sim_dt)

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        brick_positions = body_q[self.brick_bodies, :3]
        max_brick_displacement = float(np.linalg.norm(brick_positions - self.initial_brick_positions, axis=1).max())
        projectile_x = float(body_q[self.projectile_body, 0])
        if self.use_scheduler:
            assert max_brick_displacement > 0.2, (
                f"Scheduled collision detection did not disrupt the brick wall: {max_brick_displacement:.3f} m"
            )
        else:
            assert projectile_x > 0.5, f"The unscheduled projectile did not pass the wall: x={projectile_x:.3f} m"
            assert max_brick_displacement < 0.1, (
                f"The unscheduled projectile unexpectedly disrupted the brick wall: {max_brick_displacement:.3f} m"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--scheduler",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Adapt collision-detection frequency to the projectile speed.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
