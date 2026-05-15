# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon MuJoCo Wrap
#
# Minimal MuJoCo-style fixed-topology wrap prototype.  A vertical cable starts
# as a straight inactive route.  Three candidate capstans move horizontally into
# the cable at the same time from alternating sides.  Each candidate activates
# as a rolling wrap while the capstan intersects its local vertical span and
# switches back to a kinematic bypass once the straight span has line-of-sight.
#
# Command: python -m newton.examples tendon_mujoco_wrap
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

        self.radius = 0.08
        self._identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        site_zs = [1.02, 0.68, 0.36, 0.02]
        capstan_zs = [0.5 * (site_zs[i] + site_zs[i + 1]) for i in range(len(site_zs) - 1)]
        out_xs = [0.28, -0.28, 0.28]
        colors = [(0.20, 0.62, 0.95), (0.35, 0.75, 0.45), (0.95, 0.58, 0.20)]
        self.capstan_specs = [
            {
                "z": z,
                "out_x": out_x,
                "start": 0.25,
                "duration": 2.80,
                "orientation": -int(math.copysign(1.0, out_x)),
                "color": color,
            }
            for z, out_x, color in zip(capstan_zs, out_xs, colors, strict=True)
        ]
        self.candidate_count = len(self.capstan_specs)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

        self.site_idx = [self._add_site_body(builder, (0.0, 0.0, z), (0.90, 0.15, 0.15)) for z in site_zs]

        self.capstan_idx = []
        self.bypass_idx = []
        for spec in self.capstan_specs:
            self.bypass_idx.append(self._add_site_body(builder, (0.0, 0.0, spec["z"]), (0.3, 0.3, 0.3), hidden=True))
            self.capstan_idx.append(self._add_capstan_body(builder, spec))

        builder.add_tendon()
        builder.add_tendon_link(
            body=self.site_idx[0],
            link_type=int(newton.TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.0),
            axis=(0.0, 1.0, 0.0),
        )
        for i in range(self.candidate_count):
            builder.add_tendon_link(
                body=self.bypass_idx[i],
                link_type=int(newton.TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.0),
                axis=(0.0, 1.0, 0.0),
                compliance=1.0e-5,
                damping=0.2,
                rest_length=-1.0,
            )
            builder.add_tendon_link(
                body=self.site_idx[i + 1],
                link_type=int(newton.TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.0),
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

        self.candidate_link_indices = np.arange(1, 1 + 2 * self.candidate_count, 2, dtype=np.int32)
        self.candidate_seg_pairs = [(2 * i, 2 * i + 1) for i in range(self.candidate_count)]
        self.candidate_prev_links = np.arange(0, 2 * self.candidate_count, 2, dtype=np.int32)
        self.candidate_next_links = np.arange(2, 2 * self.candidate_count + 2, 2, dtype=np.int32)
        self.active = np.zeros(self.candidate_count, dtype=bool)
        self._active_history = []
        self._transition_counts = np.zeros(self.candidate_count, dtype=np.int32)
        self._activation_mismatch_count = 0
        self._max_inactive_x_error = 0.0
        self._max_active_lateral = np.zeros(self.candidate_count, dtype=np.float64)
        self._max_active_centerline_overshoot = np.zeros(self.candidate_count, dtype=np.float64)
        self._min_expected_side_clearance = np.full(self.candidate_count, float("inf"), dtype=np.float64)
        self._min_rest_length = float("inf")
        self._max_rest_length = 0.0
        self._initial_rest_sum = float(np.sum(self.solver.tendon_seg_rest_length.numpy()))
        self._max_rest_sum_error = 0.0

        self._update_kinematic_bodies(0.0)
        self._apply_route_state(allow_switch=False)

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.44, -1.65, 0.52), pitch=0.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def _add_site_body(self, builder, pos, color, hidden=False):
        body = builder.add_body(xform=wp.transform(p=wp.vec3(*pos)), mass=0.0, is_kinematic=True)
        if not hidden:
            builder.add_shape_sphere(body, radius=0.018, as_site=True, color=color)
        return body

    def _add_capstan_body(self, builder, spec):
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(float(spec["out_x"]), 0.0, float(spec["z"])), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        q_cyl_y = wp.quat(math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0))
        builder.add_shape_cylinder(
            body,
            xform=wp.transform(q=q_cyl_y),
            radius=self.radius,
            half_height=0.10,
            color=spec["color"],
        )
        return body

    def _capstan_x(self, spec, t):
        start = float(spec["start"])
        duration = float(spec["duration"])
        if t <= start or t >= start + duration:
            return float(spec["out_x"])
        phase = (t - start) / duration
        u = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)
        return float(spec["out_x"]) * (1.0 - u)

    def _update_kinematic_bodies(self, t):
        body_q = self.state_0.body_q.numpy()
        for i, spec in enumerate(self.capstan_specs):
            capstan_x = self._capstan_x(spec, t)
            body_q[self.capstan_idx[i], :3] = (capstan_x, 0.0, float(spec["z"]))
            body_q[self.capstan_idx[i], 3:] = self._identity_q
            body_q[self.bypass_idx[i], :3] = (0.0, 0.0, float(spec["z"]))
            body_q[self.bypass_idx[i], 3:] = self._identity_q
        self.state_0.body_q.assign(body_q)
        self.state_1.body_q.assign(body_q)

    def _world_point_for_link(self, link_idx, body_q):
        body_idx = int(self.model.tendon_link_body.numpy()[link_idx])
        offset = self.model.tendon_link_offset.numpy()[link_idx]
        return _transform_point_np(body_q[body_idx], offset)

    def _candidate_span_projection(self, i):
        body_q = self.state_0.body_q.numpy()
        capstan = body_q[self.capstan_idx[i], :3]
        p0 = self._world_point_for_link(int(self.candidate_prev_links[i]), body_q)
        p1 = self._world_point_for_link(int(self.candidate_next_links[i]), body_q)
        span = p1 - p0
        alpha = float(np.clip(np.dot(capstan - p0, span) / max(float(np.dot(span, span)), 1.0e-12), 0.0, 1.0))
        closest = p0 + alpha * span
        distance = float(np.linalg.norm((capstan - closest)[[0, 2]]))
        return distance, alpha

    def _candidate_should_wrap(self, i):
        distance, alpha = self._candidate_span_projection(i)
        return 0.0 < alpha < 1.0 and distance <= self.radius

    def _apply_route_state(self, allow_switch=True):
        for i in range(self.candidate_count):
            new_active = self._candidate_should_wrap(i)
            if allow_switch and new_active != bool(self.active[i]):
                self.active[i] = new_active
                self._transition_counts[i] += 1
                self._set_candidate_link_active(i, new_active)

            self._redistribute_rest_lengths(i)

    def _set_candidate_link_active(self, i, active):
        link_idx = int(self.candidate_link_indices[i])
        link_body = self.model.tendon_link_body.numpy()
        link_type = self.model.tendon_link_type.numpy()
        link_radius = self.model.tendon_link_radius.numpy()
        link_mu = self.model.tendon_link_mu.numpy()
        link_orientation = self.model.tendon_link_orientation.numpy()

        if active:
            link_body[link_idx] = self.capstan_idx[i]
            link_type[link_idx] = int(newton.TendonLinkType.ROLLING)
            link_radius[link_idx] = self.radius
            link_mu[link_idx] = 0.0
            link_orientation[link_idx] = int(self.capstan_specs[i]["orientation"])
        else:
            link_body[link_idx] = self.bypass_idx[i]
            link_type[link_idx] = int(newton.TendonLinkType.ATTACHMENT)
            link_radius[link_idx] = 0.0
            link_mu[link_idx] = 0.0
            link_orientation[link_idx] = int(self.capstan_specs[i]["orientation"])

        self.model.tendon_link_body.assign(link_body)
        self.model.tendon_link_type.assign(link_type)
        self.model.tendon_link_radius.assign(link_radius)
        self.model.tendon_link_mu.assign(link_mu)
        self.model.tendon_link_orientation.assign(link_orientation)

        if self.solver.tendon_seg_lambda is not None:
            seg_lambda = self.solver.tendon_seg_lambda.numpy()
            seg_delta = self.solver.tendon_seg_delta_lambda.numpy()
            seg_lambda[list(self.candidate_seg_pairs[i])] = 0.0
            seg_delta[list(self.candidate_seg_pairs[i])] = 0.0
            self.solver.tendon_seg_lambda.assign(seg_lambda)
            self.solver.tendon_seg_delta_lambda.assign(seg_delta)

    def _redistribute_rest_lengths(self, i):
        body_q = self.state_0.body_q.numpy()
        p0 = self._world_point_for_link(int(self.candidate_prev_links[i]), body_q).astype(np.float64)
        p1 = self._world_point_for_link(int(self.candidate_next_links[i]), body_q).astype(np.float64)
        middle = body_q[self.capstan_idx[i] if self.active[i] else self.bypass_idx[i], :3].astype(np.float64)
        len_top = float(np.linalg.norm(middle - p0))
        len_bottom = float(np.linalg.norm(p1 - middle))
        if self.active[i]:
            t0, t1, _arc = _wrap_geometry_np(
                p0,
                p1,
                middle,
                self.radius,
                np.array([0.0, 1.0, 0.0]),
                int(self.capstan_specs[i]["orientation"]),
            )
            len_top = float(np.linalg.norm(t0 - p0))
            len_bottom = float(np.linalg.norm(p1 - t1))

        rest = self.solver.tendon_seg_rest_length.numpy()
        seg_l, seg_r = self.candidate_seg_pairs[i]
        rest[seg_l] = max(len_top, 1.0e-6)
        rest[seg_r] = max(len_bottom, 1.0e-6)
        self.solver.tendon_seg_rest_length.assign(rest)

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self._update_kinematic_bodies(t)
            self._apply_route_state(allow_switch=True)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._record_metrics()

    def _record_metrics(self):
        self._active_history.append(self.active.copy())
        rest = self.solver.tendon_seg_rest_length.numpy()
        self._min_rest_length = min(self._min_rest_length, float(np.min(rest)))
        self._max_rest_length = max(self._max_rest_length, float(np.max(rest)))
        self._max_rest_sum_error = max(self._max_rest_sum_error, abs(float(np.sum(rest)) - self._initial_rest_sum))

        att_l = self.solver.tendon_seg_attachment_l.numpy()
        att_r = self.solver.tendon_seg_attachment_r.numpy()
        body_q = self.state_0.body_q.numpy()
        for i in range(self.candidate_count):
            expected_active = self._candidate_should_wrap(i)
            if bool(self.active[i]) != expected_active:
                self._activation_mismatch_count += 1

            seg_l, seg_r = self.candidate_seg_pairs[i]
            candidate_points = np.vstack((att_l[seg_l], att_r[seg_l], att_l[seg_r], att_r[seg_r]))
            if self.active[i]:
                tangent_x = np.array([att_r[seg_l, 0], att_l[seg_r, 0]], dtype=np.float64)
                capstan_x = float(body_q[self.capstan_idx[i], 0])
                expected_sign = -float(np.sign(float(self.capstan_specs[i]["out_x"])))
                rel = tangent_x - capstan_x
                self._max_active_lateral[i] = max(self._max_active_lateral[i], float(np.max(np.abs(tangent_x))))
                self._min_expected_side_clearance[i] = min(
                    self._min_expected_side_clearance[i],
                    float(np.min(rel * expected_sign)),
                )
                self._max_active_centerline_overshoot[i] = max(
                    self._max_active_centerline_overshoot[i],
                    float(np.max(-expected_sign * tangent_x)),
                )
            else:
                self._max_inactive_x_error = max(
                    self._max_inactive_x_error,
                    float(np.max(np.abs(candidate_points[:, 0]))),
                )

    def test_post_step(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in dynamic-wrap body state"

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in dynamic-wrap body state"
        active_history = np.array(self._active_history, dtype=np.int32)
        assert len(active_history) > 0, "No route-state samples were recorded"
        assert self.candidate_count == 3, f"Expected three dynamic wrap candidates: {self.candidate_count}"
        assert np.all(active_history[0] == 0), f"Cable should start inactive and straight: {active_history[:8]}"
        assert np.all(active_history[-1] == 0), f"Cable should end inactive and straight: {active_history[-8:]}"
        assert np.all(np.max(active_history, axis=0) == 1), f"Each capstan should activate once: {active_history}"
        assert np.any(np.all(active_history == 1, axis=1)), (
            f"All three capstans should be active simultaneously: {active_history}"
        )
        assert np.all(self._transition_counts >= 2), (
            f"Expected activate/deactivate transitions for each capstan: {self._transition_counts}"
        )
        assert self._activation_mismatch_count == 0, (
            f"Route active flags diverged from straight-span intersection test: {self._activation_mismatch_count}"
        )
        assert self._max_inactive_x_error < 2.0e-3, (
            f"Inactive cable should be the original vertical line: x_error={self._max_inactive_x_error:.6f}"
        )
        assert np.all(self._max_active_lateral > self.radius * 0.35), (
            f"Active wraps did not visibly route around every capstan: lateral={self._max_active_lateral}"
        )
        assert np.all(self._max_active_centerline_overshoot < 2.0e-3), (
            "Active return path crossed beyond the straight-route centerline before deactivation: "
            f"x={self._max_active_centerline_overshoot}"
        )
        assert np.all(self._min_expected_side_clearance > self.radius * 0.20), (
            "Active cable wrapped on the wrong side of at least one capstan: "
            f"min_clearance={self._min_expected_side_clearance}, radius={self.radius:.6f}"
        )
        assert self._min_rest_length > 0.0, (
            f"Route switching produced non-positive rest length: {self._min_rest_length}"
        )
        assert self._max_rest_length < 1.2, f"Route switching produced excessive rest length: {self._max_rest_length}"

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
        self.viewer.log_lines("cable", starts, ends, colors=(0.10, 0.88, 0.30), width=0.008)
        guide_starts, guide_ends, guide_colors = self._diagnostic_lines()
        self.viewer.log_lines("wrap_diagnostics", guide_starts, guide_ends, colors=guide_colors, width=0.006)
        self.viewer.end_frame()

    def _diagnostic_lines(self):
        body_q = self.state_0.body_q.numpy()
        starts = [(0.0, -0.006, 0.02)]
        ends = [(0.0, -0.006, 1.02)]
        colors = [(0.55, 0.55, 0.55)]
        for i, spec in enumerate(self.capstan_specs):
            z = float(spec["z"])
            out_x = float(spec["out_x"])
            capstan_x = float(body_q[self.capstan_idx[i], 0])
            color = (0.15, 1.0, 0.25) if self.active[i] else (0.65, 0.65, 0.65)
            starts.append((-abs(out_x), -0.006, z))
            ends.append((abs(out_x), -0.006, z))
            colors.append((0.45, 0.45, 0.45))
            starts.append((capstan_x, -0.006, z + self.radius + 0.035))
            ends.append((capstan_x + 0.09 * np.sign(out_x), -0.006, z + self.radius + 0.035))
            colors.append(color)
        return (
            wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(colors, dtype=np.float32), dtype=wp.vec3),
        )


def _transform_point_np(pose, point):
    p = pose[:3]
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], point)
    return p + point + q[3] * t + np.cross(q[:3], t)


def _tangent_point_circle_np(p, center, radius, normal, orientation):
    p = np.asarray(p, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)
    d = center - p
    d_proj = d - np.dot(d, normal) * normal
    dist = float(np.linalg.norm(d_proj))
    if dist <= radius:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fallback -= np.dot(fallback, normal) * normal
        return center + radius * fallback / max(float(np.linalg.norm(fallback)), 1.0e-12)

    u = d_proj / dist
    v = np.cross(normal, u)
    phi = math.asin(min(radius / dist, 1.0))
    angle = -0.5 * math.pi - phi if orientation > 0 else 0.5 * math.pi + phi
    return center + radius * (math.cos(angle) * u + math.sin(angle) * v)


def _wrap_geometry_np(p0, p1, center, radius, normal, orientation):
    t0 = _tangent_point_circle_np(p0, center, radius, normal, orientation)
    t1 = _tangent_point_circle_np(p1, center, radius, normal, -orientation)
    r0 = t0 - center
    r1 = t1 - center
    normal = np.asarray(normal, dtype=np.float64)
    theta = abs(math.atan2(float(np.dot(np.cross(r0, r1), normal)), float(np.dot(r0, r1))))
    return t0, t1, theta * radius


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
