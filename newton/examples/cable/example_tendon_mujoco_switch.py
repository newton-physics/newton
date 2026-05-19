# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon MuJoCo Switch
#
# Active-set tendon routing with one optional wrap candidate.  A tendon is
# anchored to the end of a rotating top capsule.  When the capsule swings left,
# the straight lower-guide-to-endpoint span intersects the middle capstan and
# the solver activates that rolling link.  When the capsule swings right, the
# solver skips the middle capstan and the tendon wraps only around the lower
# guide.
#
# Command: python -m newton.examples tendon_mujoco_switch
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable.cable import get_tendon_cable_lines


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        self.lower_radius = 0.09
        self.middle_radius = 0.085
        self.lower_orientation = -1
        self.middle_orientation = -1
        self.lower_z = 0.05
        self.middle_z = 0.62
        self.top_length = 0.60
        self.right_angle = 1.08
        self.left_angle = -0.27
        self._initial_angle = self._top_angle(0.0)
        self._identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

        self.anchor_idx = self._add_site_body(builder, (-0.72, 0.0, self.lower_z - 0.035), (0.90, 0.15, 0.15))
        self.lower_guide_idx = self._add_lower_guide_body(builder)
        self.top_idx = self._add_top_capsule_body(builder, self._initial_angle)

        builder.add_tendon()
        builder.add_tendon_link(
            body=self.anchor_idx,
            link_type=int(newton.TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.0),
            axis=(0.0, 1.0, 0.0),
        )
        builder.add_tendon_link(
            body=self.lower_guide_idx,
            link_type=int(newton.TendonLinkType.ROLLING),
            radius=self.lower_radius,
            orientation=self.lower_orientation,
            mu=0.0,
            offset=(0.0, 0.0, self.lower_z),
            axis=(0.0, 1.0, 0.0),
            compliance=1.0e-5,
            damping=0.2,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=self.lower_guide_idx,
            link_type=int(newton.TendonLinkType.ROLLING),
            radius=self.middle_radius,
            orientation=self.middle_orientation,
            mu=0.0,
            active=False,
            offset=(0.0, 0.0, self.middle_z),
            axis=(0.0, 1.0, 0.0),
            compliance=1.0e-5,
            damping=0.2,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=self.top_idx,
            link_type=int(newton.TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, self.top_length),
            axis=(0.0, 1.0, 0.0),
            compliance=1.0e-5,
            damping=0.2,
            rest_length=-1.0,
        )

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=12,
            joint_linear_relaxation=0.9,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.lower_link = 1
        self.middle_link = 2
        self.endpoint_link = 3
        self.lower_seg = 0
        self.middle_left_seg = 1
        self.middle_right_seg = 2

        self._active_history = []
        self._top_x_history = []
        self._transition_count = 0
        self._last_middle_active = False
        self._activation_mismatch_count = 0
        self._max_inactive_middle_penetration = 0.0
        self._min_active_tangent_radius = float("inf")
        self._max_active_tangent_radius = 0.0
        self._saw_middle_segment_disabled = False
        self._saw_middle_segment_enabled = False

        self._update_kinematic_bodies(0.0)
        self._update_active_set()

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.18, -1.75, 0.58), pitch=0.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def _add_site_body(self, builder, pos, color):
        body = builder.add_body(xform=wp.transform(p=wp.vec3(*pos)), mass=0.0, is_kinematic=True)
        builder.add_shape_sphere(body, radius=0.018, as_site=True, color=color)
        return body

    def _add_lower_guide_body(self, builder):
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)), mass=0.0, is_kinematic=True)
        q_cyl_y = wp.quat(math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0))
        builder.add_shape_capsule(
            body,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5 * self.middle_z)),
            radius=0.040,
            half_height=0.5 * self.middle_z - 0.04,
            color=(0.80, 0.80, 0.78),
        )
        builder.add_shape_cylinder(
            body,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.lower_z), q=q_cyl_y),
            radius=self.lower_radius,
            half_height=0.10,
            color=(0.22, 0.62, 0.95),
        )
        builder.add_shape_cylinder(
            body,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.middle_z), q=q_cyl_y),
            radius=self.middle_radius,
            half_height=0.10,
            color=(0.35, 0.75, 0.45),
        )
        return body

    def _add_top_capsule_body(self, builder, angle):
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.middle_z), q=_quat_y(angle)),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_capsule(
            body,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5 * self.top_length)),
            radius=0.038,
            half_height=0.5 * self.top_length - 0.038,
            color=(0.88, 0.86, 0.80),
        )
        builder.add_shape_sphere(
            body,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.top_length)),
            radius=0.020,
            as_site=True,
            color=(0.95, 0.20, 0.48),
        )
        return body

    def _top_angle(self, t):
        start = 0.25
        duration = 3.00
        if t <= start or t >= start + duration:
            return self.right_angle
        phase = (t - start) / duration
        u = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)
        return self.right_angle + (self.left_angle - self.right_angle) * u

    def _update_kinematic_bodies(self, t):
        angle = self._top_angle(t)
        body_q = self.state_0.body_q.numpy()
        body_q[self.lower_guide_idx, :3] = (0.0, 0.0, 0.0)
        body_q[self.lower_guide_idx, 3:] = self._identity_q
        body_q[self.top_idx, :3] = (0.0, 0.0, self.middle_z)
        body_q[self.top_idx, 3:] = _quat_y_np(angle)
        self.state_0.body_q.assign(body_q)
        self.state_1.body_q.assign(body_q)

    def _world_point_for_link(self, link_idx, body_q):
        body_idx = int(self.model.tendon_link_body.numpy()[link_idx])
        offset = self.model.tendon_link_offset.numpy()[link_idx]
        return _transform_point_np(body_q[body_idx], offset)

    def _middle_span_projection(self):
        body_q = self.state_0.body_q.numpy()
        endpoint = self._world_point_for_link(self.endpoint_link, body_q)
        lower_departure = _link_route_point_np(
            self.model,
            body_q,
            self.lower_link,
            endpoint,
            outward_orientation=True,
        )
        middle = self._world_point_for_link(self.middle_link, body_q)
        span = endpoint - lower_departure
        alpha = float(
            np.clip(np.dot(middle - lower_departure, span) / max(float(np.dot(span, span)), 1.0e-12), 0.0, 1.0)
        )
        closest = lower_departure + alpha * span
        distance = float(np.linalg.norm((middle - closest)[[0, 2]]))
        return distance, alpha

    def _middle_should_wrap(self):
        distance, alpha = self._middle_span_projection()
        return 0.0 < alpha < 1.0 and distance <= self.middle_radius

    def _update_active_set(self):
        link_active = self.solver.tendon_link_active.numpy()
        link_active[self.middle_link] = 1 if self._middle_should_wrap() else 0
        self.solver.tendon_link_active.assign(link_active)

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self._update_kinematic_bodies(t)
            self._update_active_set()
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._record_metrics()

    def _record_metrics(self):
        link_active = self.solver.tendon_link_active.numpy()
        seg_active = self.solver.tendon_seg_active.numpy()
        middle_active = bool(link_active[self.middle_link])
        self._active_history.append(middle_active)
        if middle_active != self._last_middle_active:
            self._transition_count += 1
            self._last_middle_active = middle_active

        expected_active = self._middle_should_wrap()
        if middle_active != expected_active:
            self._activation_mismatch_count += 1

        body_q = self.state_0.body_q.numpy()
        endpoint = self._world_point_for_link(self.endpoint_link, body_q)
        middle = self._world_point_for_link(self.middle_link, body_q)
        self._top_x_history.append(float(endpoint[0]))

        if middle_active:
            self._saw_middle_segment_enabled = (
                self._saw_middle_segment_enabled or seg_active[self.middle_right_seg] != 0
            )
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            radii = [
                float(np.linalg.norm((att_r[self.middle_left_seg] - middle)[[0, 2]])),
                float(np.linalg.norm((att_l[self.middle_right_seg] - middle)[[0, 2]])),
            ]
            self._min_active_tangent_radius = min(self._min_active_tangent_radius, *radii)
            self._max_active_tangent_radius = max(self._max_active_tangent_radius, *radii)
        else:
            self._saw_middle_segment_disabled = (
                self._saw_middle_segment_disabled or seg_active[self.middle_right_seg] == 0
            )
            distance, _alpha = self._middle_span_projection()
            self._max_inactive_middle_penetration = max(
                self._max_inactive_middle_penetration,
                0.0,
                self.middle_radius - distance,
            )

    def test_post_step(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in switch-wrap body state"

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in switch-wrap body state"
        active_history = np.array(self._active_history, dtype=np.int32)
        top_x_history = np.array(self._top_x_history)
        link_type = self.model.tendon_link_type.numpy()
        initial_active = self.model.tendon_link_active.numpy()

        assert link_type[self.lower_link] == int(newton.TendonLinkType.ROLLING)
        assert link_type[self.middle_link] == int(newton.TendonLinkType.ROLLING)
        assert initial_active[self.lower_link] == 1
        assert initial_active[self.middle_link] == 0
        assert len(active_history) > 0, "No switch-wrap active-set samples were recorded"
        assert active_history[0] == 0 and active_history[-1] == 0, (
            f"Middle candidate should start/end inactive: {active_history}"
        )
        assert np.max(active_history) == 1, (
            f"Middle candidate should activate while top capsule swings left: {active_history}"
        )
        assert self._transition_count >= 2, f"Expected activate/deactivate transitions: {self._transition_count}"
        assert self._activation_mismatch_count == 0, (
            f"Active flag diverged from lower-tangent-to-endpoint intersection test: {self._activation_mismatch_count}"
        )
        assert np.max(top_x_history) > 0.45 and np.min(top_x_history) < -0.12, (
            f"Top endpoint did not sweep both sides: min/max x={np.min(top_x_history):.4f}/{np.max(top_x_history):.4f}"
        )
        assert self._saw_middle_segment_disabled, (
            "Inactive middle candidate should collapse to lower-tangent endpoint span"
        )
        assert self._saw_middle_segment_enabled, "Active middle candidate should restore its second segment"
        assert self._max_inactive_middle_penetration < 1.0e-5, (
            "Inactive route should only skip the middle link after line-of-sight clears: "
            f"penetration={self._max_inactive_middle_penetration:.6f}"
        )
        assert self._min_active_tangent_radius > self.middle_radius * 0.80, (
            f"Active middle route did not reach the capstan surface: min={self._min_active_tangent_radius:.6f}"
        )
        assert self._max_active_tangent_radius < self.middle_radius * 1.20, (
            f"Active middle route drifted off the capstan surface: max={self._max_active_tangent_radius:.6f}"
        )

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
        self.viewer.log_lines("cable", starts, ends, colors=(0.98, 0.22, 0.55), width=0.008)
        guide_starts, guide_ends, guide_colors = self._diagnostic_lines()
        self.viewer.log_lines("switch_diagnostics", guide_starts, guide_ends, colors=guide_colors, width=0.005)
        self.viewer.end_frame()

    def _diagnostic_lines(self):
        body_q = self.state_0.body_q.numpy()
        middle = self._world_point_for_link(self.middle_link, body_q)
        endpoint = self._world_point_for_link(self.endpoint_link, body_q)
        active = bool(self.solver.tendon_link_active.numpy()[self.middle_link])
        color = (0.15, 1.0, 0.25) if active else (0.55, 0.55, 0.55)
        starts = [
            (-0.78, -0.008, self.lower_z - 0.035),
            (middle[0], -0.008, middle[2] + self.middle_radius + 0.03),
            (0.0, -0.008, self.middle_z),
        ]
        ends = [
            (0.55, -0.008, self.lower_z - 0.035),
            (middle[0] + 0.10, -0.008, middle[2] + self.middle_radius + 0.03),
            (endpoint[0], -0.008, endpoint[2]),
        ]
        colors = [
            (0.45, 0.45, 0.45),
            color,
            (0.28, 0.28, 0.28),
        ]
        return (
            wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(colors, dtype=np.float32), dtype=wp.vec3),
        )


def _quat_y(angle):
    return wp.quat(0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle))


def _quat_y_np(angle):
    return np.array([0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)], dtype=np.float32)


def _transform_point_np(pose, point):
    p = pose[:3]
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], point)
    return p + point + q[3] * t + np.cross(q[:3], t)


def _link_route_point_np(model, body_q, link_idx, other_point, outward_orientation):
    link_type = int(model.tendon_link_type.numpy()[link_idx])
    body_idx = int(model.tendon_link_body.numpy()[link_idx])
    offset = model.tendon_link_offset.numpy()[link_idx]
    center = _transform_point_np(body_q[body_idx], offset)
    if link_type != int(newton.TendonLinkType.ROLLING):
        return center

    axis = model.tendon_link_axis.numpy()[link_idx]
    normal = _transform_vector_np(body_q[body_idx], axis)
    orientation = int(model.tendon_link_orientation.numpy()[link_idx])
    if outward_orientation:
        orientation = -orientation
    radius = float(model.tendon_link_radius.numpy()[link_idx])
    return _tangent_point_circle_np(other_point, center, radius, normal, orientation)


def _transform_vector_np(pose, vector):
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], vector)
    return vector + q[3] * t + np.cross(q[:3], t)


def _tangent_point_circle_np(p, center, radius, plane_normal, orientation):
    p = np.asarray(p, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)
    d = center - p
    d_proj = d - np.dot(d, normal) * normal
    dist = float(np.linalg.norm(d_proj))
    if dist <= radius:
        if dist < 1.0e-8:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            fallback -= np.dot(fallback, normal) * normal
            return center + radius * fallback / max(float(np.linalg.norm(fallback)), 1.0e-12)
        return center - radius * d_proj / dist

    u = d_proj / dist
    v = np.cross(normal, u)
    phi = math.asin(min(radius / dist, 1.0))
    angle = -0.5 * math.pi - phi if orientation > 0 else 0.5 * math.pi + phi
    return center + radius * (math.cos(angle) * u + math.sin(angle) * v)


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
