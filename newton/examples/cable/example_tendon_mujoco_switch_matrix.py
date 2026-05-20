# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon MuJoCo Switch Matrix
#
# Four active-set route switching lanes cover all optional-wrap neighbor
# combinations:
#
# - attachment -> optional rolling -> attachment
# - rolling    -> optional rolling -> attachment
# - attachment -> optional rolling -> rolling
# - rolling    -> optional rolling -> rolling
#
# The candidate capstan in every lane is driven horizontally through the
# bypass span.  Rolling neighbors must be tested using tangent points, so the
# matrix catches centerline predicates that activate/deactivate too late.
#
# Command: python -m newton.examples tendon_mujoco_switch_matrix
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable.cable import get_tendon_cable_lines
from newton.examples.cable.example_tendon_mujoco_switch import _link_route_point_np, _transform_point_np


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        self.radius = 0.055
        self.site_z_low = 0.04
        self.site_z_high = 0.56
        self.candidate_z = 0.30
        self.candidate_rest_x = 0.22
        self.anchor_x = 0.20
        self.anchor_z_margin = 0.08
        self._identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.lane_specs = [
            ("attachment-attachment", False, False, -0.72),
            ("rolling-attachment", True, False, -0.24),
            ("attachment-rolling", False, True, 0.24),
            ("rolling-rolling", True, True, 0.72),
        ]

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

        self.lanes = []
        q_cyl_y = wp.quat(math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0))
        for name, prev_rolling, next_rolling, y in self.lane_specs:
            target_x = self._candidate_target_x(prev_rolling, next_rolling)
            start_x = self._candidate_start_x(target_x)
            prev_body = self._add_site_body(builder, (0.0, y, self.site_z_low), (0.90, 0.15, 0.15))
            next_body = self._add_site_body(builder, (0.0, y, self.site_z_high), (0.90, 0.15, 0.15))
            prev_anchor_body = None
            next_anchor_body = None
            if prev_rolling:
                prev_anchor_body = self._add_site_body(
                    builder,
                    (-self.anchor_x, y, self.site_z_low - self.anchor_z_margin),
                    (0.90, 0.15, 0.15),
                )
            if next_rolling:
                next_anchor_body = self._add_site_body(
                    builder,
                    (self.anchor_x, y, self.site_z_high + self.anchor_z_margin),
                    (0.90, 0.15, 0.15),
                )
            candidate_body = builder.add_body(
                xform=wp.transform(p=wp.vec3(start_x, y, self.candidate_z)),
                mass=0.0,
                is_kinematic=True,
            )
            builder.add_shape_cylinder(
                candidate_body,
                xform=wp.transform(q=q_cyl_y),
                radius=self.radius,
                half_height=0.075,
                color=(0.35, 0.75, 0.45),
            )

            if prev_rolling:
                builder.add_shape_cylinder(
                    prev_body,
                    xform=wp.transform(q=q_cyl_y),
                    radius=self.radius,
                    half_height=0.075,
                    color=(0.22, 0.62, 0.95),
                )
            if next_rolling:
                builder.add_shape_cylinder(
                    next_body,
                    xform=wp.transform(q=q_cyl_y),
                    radius=self.radius,
                    half_height=0.075,
                    color=(0.95, 0.58, 0.20),
                )

            tendon_id = builder.add_tendon()
            if prev_rolling:
                builder.add_tendon_link(
                    body=prev_anchor_body,
                    link_type=int(newton.TendonLinkType.ATTACHMENT),
                    offset=(0.0, 0.0, 0.0),
                    axis=(0.0, 1.0, 0.0),
                )
                prev_link = builder.add_tendon_link(
                    body=prev_body,
                    link_type=int(newton.TendonLinkType.ROLLING),
                    radius=self.radius,
                    orientation=-1,
                    mu=0.0,
                    offset=(0.0, 0.0, 0.0),
                    axis=(0.0, 1.0, 0.0),
                    compliance=1.0e-5,
                    damping=0.2,
                    rest_length=-1.0,
                )
            else:
                prev_link = builder.add_tendon_link(
                    body=prev_body,
                    link_type=int(newton.TendonLinkType.ATTACHMENT),
                    offset=(0.0, 0.0, 0.0),
                    axis=(0.0, 1.0, 0.0),
                )
            candidate_link = builder.add_tendon_link(
                body=candidate_body,
                link_type=int(newton.TendonLinkType.ROLLING),
                radius=self.radius,
                orientation=self._candidate_orientation(start_x),
                mu=0.0,
                active=False,
                offset=(0.0, 0.0, 0.0),
                axis=(0.0, 1.0, 0.0),
                compliance=1.0e-5,
                damping=0.2,
                rest_length=-1.0,
            )
            if next_rolling:
                next_link = builder.add_tendon_link(
                    body=next_body,
                    link_type=int(newton.TendonLinkType.ROLLING),
                    radius=self.radius,
                    orientation=1,
                    mu=0.0,
                    offset=(0.0, 0.0, 0.0),
                    axis=(0.0, 1.0, 0.0),
                    compliance=1.0e-5,
                    damping=0.2,
                    rest_length=-1.0,
                )
                builder.add_tendon_link(
                    body=next_anchor_body,
                    link_type=int(newton.TendonLinkType.ATTACHMENT),
                    offset=(0.0, 0.0, 0.0),
                    axis=(0.0, 1.0, 0.0),
                    compliance=1.0e-5,
                    damping=0.2,
                    rest_length=-1.0,
                )
            else:
                next_link = builder.add_tendon_link(
                    body=next_body,
                    link_type=int(newton.TendonLinkType.ATTACHMENT),
                    offset=(0.0, 0.0, 0.0),
                    axis=(0.0, 1.0, 0.0),
                    compliance=1.0e-5,
                    damping=0.2,
                    rest_length=-1.0,
                )
            self.lanes.append(
                {
                    "name": name,
                    "prev_body": prev_body,
                    "next_body": next_body,
                    "prev_anchor_body": prev_anchor_body,
                    "next_anchor_body": next_anchor_body,
                    "candidate_body": candidate_body,
                    "prev_link": prev_link,
                    "candidate_link": candidate_link,
                    "next_link": next_link,
                    "seg_left": candidate_link - tendon_id - 1,
                    "seg_right": candidate_link - tendon_id,
                    "prev_rolling": prev_rolling,
                    "next_rolling": next_rolling,
                    "expected_side": -float(np.sign(start_x)),
                    "start_x": start_x,
                    "target_x": target_x,
                    "y": y,
                }
            )

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=12, joint_linear_relaxation=0.9)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._active_history = [[] for _ in self.lanes]
        self._transition_counts = np.zeros(len(self.lanes), dtype=np.int32)
        self._last_active = np.zeros(len(self.lanes), dtype=bool)
        self._activation_mismatch_count = np.zeros(len(self.lanes), dtype=np.int32)
        self._max_inactive_penetration = np.zeros(len(self.lanes), dtype=np.float64)
        self._min_active_tangent_radius = np.full(len(self.lanes), float("inf"), dtype=np.float64)
        self._max_active_tangent_radius = np.zeros(len(self.lanes), dtype=np.float64)
        self._min_expected_side_clearance = np.full(len(self.lanes), float("inf"), dtype=np.float64)
        self._saw_disabled_segment = np.zeros(len(self.lanes), dtype=bool)
        self._saw_enabled_segment = np.zeros(len(self.lanes), dtype=bool)

        self._update_kinematic_bodies(0.0)
        self._update_active_set()

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.22, -2.35, 0.33), pitch=0.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def _add_site_body(self, builder, pos, color):
        body = builder.add_body(xform=wp.transform(p=wp.vec3(*pos)), mass=0.0, is_kinematic=True)
        builder.add_shape_sphere(body, radius=0.015, as_site=True, color=color)
        return body

    def _candidate_target_x(self, prev_rolling, next_rolling):
        if prev_rolling and not next_rolling:
            return 0.55 * self.radius
        if next_rolling and not prev_rolling:
            return -0.55 * self.radius
        return 0.0

    def _candidate_start_x(self, target_x):
        if abs(target_x) > 1.0e-8:
            return math.copysign(self.candidate_rest_x, target_x)
        return self.candidate_rest_x

    def _candidate_orientation(self, start_x):
        return int(math.copysign(1.0, start_x))

    def _candidate_x(self, t, lane):
        start = 0.18
        duration = 3.00
        if t <= start or t >= start + duration:
            return lane["start_x"]
        phase = (t - start) / duration
        u = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)
        return lane["start_x"] + (lane["target_x"] - lane["start_x"]) * u

    def _update_kinematic_bodies(self, t):
        body_q = self.state_0.body_q.numpy()
        for lane in self.lanes:
            x = self._candidate_x(t, lane)
            if lane["prev_anchor_body"] is not None:
                body_q[lane["prev_anchor_body"], :3] = (
                    -self.anchor_x,
                    lane["y"],
                    self.site_z_low - self.anchor_z_margin,
                )
                body_q[lane["prev_anchor_body"], 3:] = self._identity_q
            body_q[lane["prev_body"], :3] = (0.0, lane["y"], self.site_z_low)
            body_q[lane["prev_body"], 3:] = self._identity_q
            body_q[lane["next_body"], :3] = (0.0, lane["y"], self.site_z_high)
            body_q[lane["next_body"], 3:] = self._identity_q
            if lane["next_anchor_body"] is not None:
                body_q[lane["next_anchor_body"], :3] = (
                    self.anchor_x,
                    lane["y"],
                    self.site_z_high + self.anchor_z_margin,
                )
                body_q[lane["next_anchor_body"], 3:] = self._identity_q
            body_q[lane["candidate_body"], :3] = (x, lane["y"], self.candidate_z)
            body_q[lane["candidate_body"], 3:] = self._identity_q
        self.state_0.body_q.assign(body_q)
        self.state_1.body_q.assign(body_q)

    def _world_point_for_link(self, link_idx, body_q):
        body_idx = int(self.model.tendon_link_body.numpy()[link_idx])
        offset = self.model.tendon_link_offset.numpy()[link_idx]
        return _transform_point_np(body_q[body_idx], offset)

    def _bypass_points(self, lane):
        body_q = self.state_0.body_q.numpy()
        candidate = self._world_point_for_link(lane["candidate_link"], body_q)
        prev_center = self._world_point_for_link(lane["prev_link"], body_q)
        next_center = self._world_point_for_link(lane["next_link"], body_q)
        if lane["prev_rolling"] and lane["next_rolling"]:
            p0 = prev_center
            p1 = next_center
            for _iter in range(10):
                p1 = _link_route_point_np(self.model, body_q, lane["next_link"], p0, outward_orientation=False)
                p0 = _link_route_point_np(self.model, body_q, lane["prev_link"], p1, outward_orientation=True)
            return p0, p1, candidate

        p0 = _link_route_point_np(self.model, body_q, lane["prev_link"], next_center, outward_orientation=True)
        p1 = _link_route_point_np(self.model, body_q, lane["next_link"], p0, outward_orientation=False)
        return p0, p1, candidate

    def _candidate_span_projection(self, lane):
        p0, p1, candidate = self._bypass_points(lane)
        span = p1 - p0
        alpha = float(np.clip(np.dot(candidate - p0, span) / max(float(np.dot(span, span)), 1.0e-12), 0.0, 1.0))
        closest = p0 + alpha * span
        distance = float(np.linalg.norm((candidate - closest)[[0, 2]]))
        return distance, alpha

    def _candidate_should_wrap(self, lane):
        distance, alpha = self._candidate_span_projection(lane)
        return 0.0 < alpha < 1.0 and distance <= self.radius

    def _update_active_set(self):
        link_active = self.solver.tendon_link_active.numpy()
        for lane in self.lanes:
            link_active[lane["candidate_link"]] = 1 if self._candidate_should_wrap(lane) else 0
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
        att_l = self.solver.tendon_seg_attachment_l.numpy()
        att_r = self.solver.tendon_seg_attachment_r.numpy()
        body_q = self.state_0.body_q.numpy()

        for i, lane in enumerate(self.lanes):
            active = bool(link_active[lane["candidate_link"]])
            self._active_history[i].append(active)
            if active != self._last_active[i]:
                self._transition_counts[i] += 1
                self._last_active[i] = active

            expected_active = self._candidate_should_wrap(lane)
            if active != expected_active:
                self._activation_mismatch_count[i] += 1

            candidate = self._world_point_for_link(lane["candidate_link"], body_q)
            if active:
                self._saw_enabled_segment[i] = self._saw_enabled_segment[i] or seg_active[lane["seg_right"]] != 0
                radii = [
                    float(np.linalg.norm((att_r[lane["seg_left"]] - candidate)[[0, 2]])),
                    float(np.linalg.norm((att_l[lane["seg_right"]] - candidate)[[0, 2]])),
                ]
                side_clearance = [
                    lane["expected_side"] * float(att_r[lane["seg_left"], 0] - candidate[0]),
                    lane["expected_side"] * float(att_l[lane["seg_right"], 0] - candidate[0]),
                ]
                self._min_active_tangent_radius[i] = min(self._min_active_tangent_radius[i], *radii)
                self._max_active_tangent_radius[i] = max(self._max_active_tangent_radius[i], *radii)
                self._min_expected_side_clearance[i] = min(self._min_expected_side_clearance[i], *side_clearance)
            else:
                self._saw_disabled_segment[i] = self._saw_disabled_segment[i] or seg_active[lane["seg_right"]] == 0
                distance, _alpha = self._candidate_span_projection(lane)
                self._max_inactive_penetration[i] = max(
                    self._max_inactive_penetration[i],
                    0.0,
                    self.radius - distance,
                )

    def test_post_step(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in switch-matrix body state"

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in switch-matrix body state"
        link_type = self.model.tendon_link_type.numpy()
        initial_active = self.model.tendon_link_active.numpy()

        for i, lane in enumerate(self.lanes):
            history = np.array(self._active_history[i], dtype=np.int32)
            assert link_type[lane["candidate_link"]] == int(newton.TendonLinkType.ROLLING)
            assert initial_active[lane["candidate_link"]] == 0
            assert history[0] == 0 and history[-1] == 0, f"{lane['name']} should start/end inactive: {history}"
            assert np.max(history) == 1, f"{lane['name']} should activate: {history}"
            active_frames = np.flatnonzero(history)
            assert np.all(history[active_frames[0] : active_frames[-1] + 1] == 1), (
                f"{lane['name']} should have one contiguous active interval: {history}"
            )
            assert self._transition_counts[i] == 2, f"{lane['name']} should enter and leave once: {history}"
            assert self._activation_mismatch_count[i] == 0, (
                f"{lane['name']} active flag diverged from tangent-aware bypass predicate: "
                f"{self._activation_mismatch_count[i]}"
            )
            assert self._saw_disabled_segment[i], f"{lane['name']} never disabled the skipped segment"
            assert self._saw_enabled_segment[i], f"{lane['name']} never restored the skipped segment"
            assert self._max_inactive_penetration[i] < 1.0e-5, (
                f"{lane['name']} remained inactive while bypass span intersected candidate: "
                f"{self._max_inactive_penetration[i]:.6f}"
            )
            assert self._min_active_tangent_radius[i] > self.radius * 0.80, (
                f"{lane['name']} active route did not reach capstan surface: {self._min_active_tangent_radius[i]:.6f}"
            )
            assert self._max_active_tangent_radius[i] < self.radius * 1.20, (
                f"{lane['name']} active route drifted off capstan surface: {self._max_active_tangent_radius[i]:.6f}"
            )
            assert self._min_expected_side_clearance[i] > self.radius * 0.20, (
                f"{lane['name']} active route jumped to wrong capstan side: {self._min_expected_side_clearance[i]:.6f}"
            )

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
        self.viewer.log_lines("cable", starts, ends, colors=(0.98, 0.22, 0.55), width=0.006)
        guide_starts, guide_ends, guide_colors = self._diagnostic_lines()
        self.viewer.log_lines("switch_matrix_diagnostics", guide_starts, guide_ends, colors=guide_colors, width=0.004)
        self.viewer.end_frame()

    def _diagnostic_lines(self):
        starts = []
        ends = []
        colors = []
        for lane in self.lanes:
            p0, p1, candidate = self._bypass_points(lane)
            active = bool(self.solver.tendon_link_active.numpy()[lane["candidate_link"]])
            starts.append((p0[0], p0[1] - 0.008, p0[2]))
            ends.append((p1[0], p1[1] - 0.008, p1[2]))
            colors.append((0.18, 0.95, 0.30) if active else (0.55, 0.55, 0.55))
            starts.append((candidate[0], candidate[1] - 0.008, candidate[2] + self.radius + 0.025))
            ends.append((candidate[0] + 0.075, candidate[1] - 0.008, candidate[2] + self.radius + 0.025))
            colors.append((0.18, 0.95, 0.30) if active else (0.55, 0.55, 0.55))
        return (
            wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(colors, dtype=np.float32), dtype=wp.vec3),
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
