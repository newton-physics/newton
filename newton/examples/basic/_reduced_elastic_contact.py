# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton.examples.basic._reduced_elastic import elastic_shape_deformed_vertices


def identity_inertia(scale: float = 0.02) -> wp.mat33:
    return wp.mat33(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale)


def contact_shape_config(ke: float = 8.0e4, kd: float = 0.04, mu: float = 2.2) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = ke
    cfg.kd = kd
    cfg.mu = mu
    cfg.margin = 0.0
    cfg.gap = 0.0
    return cfg


def visual_shape_config() -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    cfg.density = 0.0
    return cfg


def rubber_contact_modes(half_extent: float, contact_side: float, axis: int = 0, poisson: float = 0.35):
    """Return two local compression modes for one pad face."""

    def mode_shape(x):
        xi = float(np.clip(contact_side * x[axis] / half_extent, -1.0, 1.0))
        s = 0.5 * (xi + 1.0)
        s2 = s * s
        phi0 = np.zeros(3, dtype=np.float32)
        phi1 = np.zeros(3, dtype=np.float32)
        phi0[axis] = -contact_side * s
        phi1[axis] = -contact_side * s2
        for i in range(3):
            if i == axis:
                continue
            phi0[i] = poisson * float(x[i]) * s
            phi1[i] = 0.5 * poisson * float(x[i]) * s2
        return np.array([phi0, phi1], dtype=np.float32)

    return mode_shape


def scraper_modes(height: float):
    """Return vertical compression and lateral bending modes for a floor scraper."""

    def mode_shape(x):
        eta = float(np.clip((x[2] + height) / (2.0 * height), 0.0, 1.0))
        bottom = 1.0 - eta
        return np.array(
            [
                [0.0, 0.0, bottom],
                [-bottom * bottom, 0.0, 0.12 * bottom],
            ],
            dtype=np.float32,
        )

    return mode_shape


def owner_q_starts(model: newton.Model, bodies: list[int]) -> dict[int, int]:
    starts = {}
    for body in bodies:
        elastic_index = int(model.body_elastic_index.numpy()[body])
        owner = int(model.elastic_joint.numpy()[elastic_index])
        starts[body] = int(model.joint_q_start.numpy()[owner])
    return starts


def owner_qd_starts(model: newton.Model, bodies: list[int]) -> dict[int, int]:
    starts = {}
    for body in bodies:
        elastic_index = int(model.body_elastic_index.numpy()[body])
        owner = int(model.elastic_joint.numpy()[elastic_index])
        starts[body] = int(model.joint_qd_start.numpy()[owner])
    return starts


def apply_kinematic_targets(
    state: newton.State,
    q_starts: dict[int, int],
    targets: dict[int, tuple[wp.vec3, wp.quat]],
    velocities: dict[int, tuple[wp.vec3, wp.vec3]] | None = None,
    qd_starts: dict[int, int] | None = None,
):
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    for body, (pos, quat) in targets.items():
        start = q_starts[body]
        q[start : start + 7] = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
        body_q[body] = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
        if velocities is not None and body in velocities:
            linear, angular = velocities[body]
            if qd_starts is not None and body in qd_starts:
                qd_start = qd_starts[body]
                qd[qd_start : qd_start + 6] = [
                    linear[0],
                    linear[1],
                    linear[2],
                    angular[0],
                    angular[1],
                    angular[2],
                ]
            body_qd[body] = [linear[0], linear[1], linear[2], angular[0], angular[1], angular[2]]
    state.joint_q.assign(q)
    if velocities is not None:
        state.joint_qd.assign(qd)
    state.body_q.assign(body_q)
    if velocities is not None:
        state.body_qd.assign(body_qd)


def finite_difference_target_velocities(
    targets: dict[int, tuple[wp.vec3, wp.quat]],
    previous_targets: dict[int, tuple[wp.vec3, wp.quat]],
    dt: float,
) -> dict[int, tuple[wp.vec3, wp.vec3]]:
    inv_dt = 1.0 / dt if dt > 0.0 else 0.0
    velocities = {}
    for body, (pos, _quat) in targets.items():
        prev_pos = previous_targets[body][0]
        linear = (pos - prev_pos) * inv_dt
        velocities[body] = (linear, wp.vec3(0.0, 0.0, 0.0))
    return velocities


def validate_elastic_vertices(model: newton.Model, state: newton.State):
    for elastic_shape in range(int(model.elastic_shape_count)):
        vertices = elastic_shape_deformed_vertices(model, state, elastic_shape)
        if not np.isfinite(vertices).all():
            raise AssertionError(f"elastic shape {elastic_shape} has non-finite deformed vertices")


def step_example(example, frame_count: int):
    for _ in range(frame_count):
        example.step()
        example.render()
    example.test_final()
    return example


def run_example_test(example_cls, frame_count: int, device: str | wp.context.Devicelike | None = None):
    context = wp.ScopedDevice(device) if device is not None else nullcontext()
    with context:
        viewer = newton.viewer.ViewerNull()
        example = example_cls(viewer, None)
        return step_example(example, frame_count)
