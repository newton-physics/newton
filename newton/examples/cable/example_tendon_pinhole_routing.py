# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Pinhole Routing
#
# Alternating prismatic sliders pull a cable back and forth through four
# zero-radius pinholes on a smooth non-spherical cam body.  Only one endpoint is
# actively driven at a time, and the cam is free to rotate about its hinge under
# the routed cable loads from the frictional pinholes.  White/yellow diagnostic
# tick marks are fixed cable-material coordinates, so pinhole slip is visible as
# ticks moving through the body-local guide points.
#
# Command: python -m newton.examples tendon_pinhole_routing
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable.cable import (
    assert_tendon_total_length,
    get_tendon_attachment_worlds,
)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        self.guide_origin = wp.vec3(0.0, 0.0, 2.55)
        self.cam_half_width = 0.12
        self.pinhole_offsets = _cam_pinhole_offsets(self.cam_half_width + 0.035)

        guide_mass = 1.8
        guide_hx = 0.80
        guide_hy = self.cam_half_width
        guide_hz = 0.46
        guide_inertia = wp.mat33(
            guide_mass * (guide_hy * guide_hy + guide_hz * guide_hz) / 3.0,
            0.0,
            0.0,
            0.0,
            guide_mass * (guide_hx * guide_hx + guide_hz * guide_hz) / 3.0,
            0.0,
            0.0,
            0.0,
            guide_mass * (guide_hx * guide_hx + guide_hy * guide_hy) / 3.0,
        )
        self.guide_idx = guide = builder.add_link(
            xform=wp.transform(p=self.guide_origin, q=wp.quat_identity()),
            mass=guide_mass,
            inertia=guide_inertia,
            lock_inertia=True,
        )
        self._add_smooth_guide_shapes(builder, guide)
        j_guide = builder.add_joint_revolute(
            parent=-1,
            child=guide,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(p=self.guide_origin),
            child_xform=wp.transform(),
            limit_lower=-0.65,
            limit_upper=0.65,
            limit_ke=5000.0,
            limit_kd=180.0,
            armature=0.05,
        )
        builder.add_articulation([j_guide])

        self.left_idx, j_left = self._add_slider(
            builder,
            wp.vec3(-1.18, -0.155, 2.05),
            color=(0.95, 0.20, 0.55),
        )
        self.right_idx, j_right = self._add_slider(
            builder,
            wp.vec3(1.55, -0.155, 2.05),
            color=(0.95, 0.70, 0.15),
        )
        self.follower_idx = self.left_idx
        self.driver_idx = self.right_idx

        axis = (0.0, 1.0, 0.0)
        builder.add_tendon()
        builder.add_tendon_link(
            body=self.left_idx,
            link_type=int(newton.TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.07),
            axis=axis,
        )
        for offset in self.pinhole_offsets:
            builder.add_tendon_link(
                body=guide,
                link_type=int(newton.TendonLinkType.PINHOLE),
                mu=0.45,
                offset=tuple(float(v) for v in offset),
                axis=axis,
                compliance=2.0e-5,
                damping=1.0,
                rest_length=-1.0,
            )
        builder.add_tendon_link(
            body=self.right_idx,
            link_type=int(newton.TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.07),
            axis=axis,
            compliance=2.0e-5,
            damping=1.0,
            rest_length=-1.0,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=24,
            joint_linear_relaxation=0.8,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        dof_starts = self.model.joint_qd_start.numpy()
        self.guide_dof_start = int(dof_starts[j_guide])
        self.left_dof_start = int(dof_starts[j_left])
        self.right_dof_start = int(dof_starts[j_right])
        self.driver_dof_start = self.right_dof_start
        body_q = self.state_0.body_q.numpy()
        self._initial_left_x = float(body_q[self.left_idx][0])
        self._initial_right_x = float(body_q[self.right_idx][0])
        self._initial_follower_x = self._initial_left_x
        self._initial_driver_x = self._initial_right_x
        self._initial_seg_rest_lengths = self.solver.tendon_seg_rest_length.numpy().copy()
        self._initial_total_rest_length = float(np.sum(self._initial_seg_rest_lengths))
        self._tick_material_coords = np.arange(0.08, self._initial_total_rest_length - 0.04, 0.14, dtype=np.float32)
        self._max_material_slip = 0.0
        self._max_total_rest_error = 0.0
        self._max_visual_slack = 0.0
        self._guide_angle_history = []
        self._follower_x_history = []
        self._driver_x_history = []

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -5.0, 2.28), pitch=0.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def _add_smooth_guide_shapes(self, builder, guide):
        builder.add_shape_convex_hull(
            guide,
            mesh=_make_cam_mesh(self.cam_half_width),
            cfg=builder.default_site_cfg,
            color=(0.2, 0.55, 0.9),
        )
        for offset in self.pinhole_offsets:
            builder.add_shape_sphere(
                guide,
                xform=wp.transform(p=wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))),
                radius=0.035,
                as_site=True,
            )

    def _add_slider(self, builder, pos, color):
        dof = newton.ModelBuilder.JointDofConfig
        body = builder.add_link(xform=wp.transform(p=pos, q=wp.quat_identity()), mass=0.9)
        builder.add_shape_ellipsoid(body, rx=0.08, ry=0.035, rz=0.08, color=color)
        axis_cfg = dof(
            axis=newton.Axis.X,
            limit_lower=-0.48,
            limit_upper=0.55,
            limit_ke=5000.0,
            limit_kd=120.0,
            target_ke=0.0,
            target_kd=0.0,
            effort_limit=1600.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        joint = builder.add_joint_d6(
            parent=-1,
            child=body,
            linear_axes=[axis_cfg],
            angular_axes=[],
            parent_xform=wp.transform(p=pos),
            child_xform=wp.transform(),
        )
        builder.add_articulation([joint])
        return body, joint

    @staticmethod
    def _smoothstep(x):
        x = min(max(x, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _set_drive_targets(self, t):
        drive_ke = 4200.0
        drive_kd = 320.0
        right_pull = 0.34
        left_pull = 0.48
        right_pull_time = 1.35
        switch_time = 1.65
        left_pull_time = 1.60

        if t < switch_time:
            u = self._smoothstep(t / right_pull_time)
            du = (6.0 * (t / right_pull_time) - 6.0 * (t / right_pull_time) ** 2) / right_pull_time
            if t >= right_pull_time:
                du = 0.0

            left_target = 0.0
            left_velocity = 0.0
            right_target = right_pull * u
            right_velocity = right_pull * du
            left_ke = 0.0
            left_kd = 0.0
            right_ke = drive_ke
            right_kd = drive_kd
        else:
            local_t = t - switch_time
            u = self._smoothstep(local_t / left_pull_time)
            du = (6.0 * (local_t / left_pull_time) - 6.0 * (local_t / left_pull_time) ** 2) / left_pull_time
            if local_t >= left_pull_time:
                du = 0.0

            left_target = left_pull - 2.0 * left_pull * u
            left_velocity = -2.0 * left_pull * du
            right_target = right_pull
            right_velocity = 0.0
            left_ke = drive_ke
            left_kd = drive_kd
            right_ke = 0.0
            right_kd = 0.0

        self.model.joint_target_ke[self.left_dof_start : self.left_dof_start + 1].fill_(left_ke)
        self.model.joint_target_kd[self.left_dof_start : self.left_dof_start + 1].fill_(left_kd)
        self.model.joint_target_ke[self.right_dof_start : self.right_dof_start + 1].fill_(right_ke)
        self.model.joint_target_kd[self.right_dof_start : self.right_dof_start + 1].fill_(right_kd)
        self.control.joint_target_pos[self.left_dof_start : self.left_dof_start + 1].fill_(left_target)
        self.control.joint_target_vel[self.left_dof_start : self.left_dof_start + 1].fill_(left_velocity)
        self.control.joint_target_pos[self.right_dof_start : self.right_dof_start + 1].fill_(right_target)
        self.control.joint_target_vel[self.right_dof_start : self.right_dof_start + 1].fill_(right_velocity)

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self._set_drive_targets(t)

            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._record_metrics()

    def _record_metrics(self):
        body_q = self.state_0.body_q.numpy()
        q_guide = body_q[self.guide_idx]
        self._guide_angle_history.append(2.0 * math.atan2(float(q_guide[4]), float(q_guide[6])))
        self._follower_x_history.append(float(body_q[self.left_idx][0]))
        self._driver_x_history.append(float(body_q[self.right_idx][0]))
        seg_rest = self.solver.tendon_seg_rest_length.numpy()
        self._max_material_slip = max(
            self._max_material_slip, float(np.max(np.abs(seg_rest - self._initial_seg_rest_lengths)))
        )
        self._max_total_rest_error = max(
            self._max_total_rest_error,
            abs(float(np.sum(seg_rest)) - self._initial_total_rest_length),
        )
        att_l, att_r = get_tendon_attachment_worlds(self.solver, self.model, self.state_0)
        geom = np.linalg.norm(att_r - att_l, axis=1)
        self._max_visual_slack = max(self._max_visual_slack, float(np.max(np.maximum(seg_rest - geom, 0.0))))

    def test_post_step(self):
        assert_tendon_total_length(self, rel_tol=0.12)

    def test_final(self):
        assert_tendon_total_length(self, rel_tol=0.12)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in multi-pinhole routing body state"
        if not self._guide_angle_history:
            self._record_metrics()

        link_type = self.model.tendon_link_type.numpy()
        link_body = self.model.tendon_link_body.numpy()
        link_mu = self.model.tendon_link_mu.numpy()
        pinhole_links = np.nonzero(link_type == int(newton.TendonLinkType.PINHOLE))[0]
        assert len(pinhole_links) == len(self.pinhole_offsets), f"Unexpected pinhole count: {pinhole_links}"
        assert np.all(link_body[pinhole_links] == self.guide_idx), "All pinholes should be on the smooth cam body"
        assert np.all(link_mu[pinhole_links] > 0.0), "Cam pinholes should be frictional in this routing example"

        att_l, att_r = get_tendon_attachment_worlds(self.solver, self.model, self.state_0)
        guide_pose = body_q[self.guide_idx]
        for i, offset in enumerate(self.pinhole_offsets):
            link_idx = int(pinhole_links[i])
            seg_left = link_idx - 1
            seg_right = link_idx
            world = _transform_point_np(guide_pose, offset)
            np.testing.assert_allclose(att_r[seg_left], world, rtol=0.0, atol=2.0e-3)
            np.testing.assert_allclose(att_l[seg_right], world, rtol=0.0, atol=2.0e-3)

        guide_angles = np.array(self._guide_angle_history, dtype=np.float64)
        assert np.max(np.abs(guide_angles)) > 0.035, f"Passive cam did not rotate enough: {guide_angles[-5:]}"
        assert abs(float(self.control.joint_target_pos.numpy()[self.guide_dof_start])) < 1.0e-6, (
            "Cam hinge should remain passive, with no prescribed target"
        )
        assert self._max_material_slip > 0.10, (
            f"Pinhole material slip did not become visible: {self._max_material_slip:.5f}"
        )
        assert self._max_visual_slack > 0.02, (
            f"Slack curve interval did not become visible: {self._max_visual_slack:.5f}"
        )
        assert self._max_total_rest_error < 5.0e-3, (
            f"Total tendon rest length should stay conserved: err={self._max_total_rest_error:.5f}"
        )
        left_history = np.array(self._follower_x_history, dtype=np.float64)
        right_history = np.array(self._driver_x_history, dtype=np.float64)
        assert np.max(right_history) - self._initial_right_x > 0.25, (
            f"Right prismatic drive did not pull outward enough: {right_history[-5:]}"
        )
        assert np.max(left_history) - self._initial_left_x > 0.08, (
            f"Left slider did not follow the right-side pull: {left_history[-5:]}"
        )
        assert self._initial_left_x - np.min(left_history) > 0.15, (
            f"Left prismatic drive did not pull back enough: {left_history[-5:]}"
        )
        assert np.max(right_history) - np.min(right_history) > 0.18, (
            f"Right slider did not release/follow during left-side pull: {right_history[-5:]}"
        )
        assert abs(float(body_q[self.left_idx][2]) - 2.05) < 0.04, "Left slider should stay on its guide rail"
        assert abs(float(body_q[self.right_idx][2]) - 2.05) < 0.04, "Right slider should stay on its guide rail"

    def _material_tick_lines(self):
        att_l, att_r = get_tendon_attachment_worlds(self.solver, self.model, self.state_0)
        rest_lengths = self.solver.tendon_seg_rest_length.numpy()
        total = float(np.sum(rest_lengths))
        if total <= 1.0e-6 or len(self._tick_material_coords) == 0:
            empty = wp.array(np.empty((0, 3), dtype=np.float32), dtype=wp.vec3)
            return empty, empty, empty

        cumulative = np.concatenate(([0.0], np.cumsum(rest_lengths, dtype=np.float64)))
        tick_starts = []
        tick_ends = []
        tick_colors = []
        view_dir = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        half_tick = 0.032
        for i, coord in enumerate(self._tick_material_coords):
            material_coord = float(coord) % total
            seg = int(np.searchsorted(cumulative, material_coord, side="right") - 1)
            seg = min(max(seg, 0), len(rest_lengths) - 1)
            seg_rest = max(float(rest_lengths[seg]), 1.0e-6)
            alpha = np.clip((material_coord - cumulative[seg]) / seg_rest, 0.0, 1.0)
            p0 = att_l[seg].astype(np.float64)
            p1 = att_r[seg].astype(np.float64)
            pos = self._slack_curve_point(p0, p1, seg_rest, alpha)
            tangent = self._slack_curve_tangent(p0, p1, seg_rest, alpha)
            tick_dir = np.cross(view_dir, tangent)
            norm = float(np.linalg.norm(tick_dir))
            if norm < 1.0e-8:
                tick_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                tick_dir = tick_dir / norm

            tick_starts.append(pos - half_tick * tick_dir)
            tick_ends.append(pos + half_tick * tick_dir)
            tick_colors.append((1.0, 0.95, 0.20) if i % 2 == 0 else (1.0, 1.0, 1.0))

        return (
            wp.array(np.asarray(tick_starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(tick_ends, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(tick_colors, dtype=np.float32), dtype=wp.vec3),
        )

    @staticmethod
    def _slack_curve_offset(p0, p1, rest_length, alpha):
        vec = p1 - p0
        geom_length = float(np.linalg.norm(vec))
        if geom_length < 1.0e-8 or rest_length <= geom_length + 1.0e-4:
            return np.zeros(3, dtype=np.float64)

        h = min(0.5 * math.sqrt(max(rest_length * rest_length - geom_length * geom_length, 0.0)), 0.2 * geom_length)
        sag = 4.0 * h * alpha * (1.0 - alpha)
        return np.array([0.0, 0.0, -sag], dtype=np.float64)

    @classmethod
    def _slack_curve_point(cls, p0, p1, rest_length, alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return p0 + alpha * (p1 - p0) + cls._slack_curve_offset(p0, p1, rest_length, alpha)

    @classmethod
    def _slack_curve_tangent(cls, p0, p1, rest_length, alpha):
        eps = 1.0e-3
        a0 = max(0.0, alpha - eps)
        a1 = min(1.0, alpha + eps)
        tangent = cls._slack_curve_point(p0, p1, rest_length, a1) - cls._slack_curve_point(p0, p1, rest_length, a0)
        if float(np.linalg.norm(tangent)) < 1.0e-8:
            tangent = p1 - p0
        return tangent

    def _visual_cable_lines(self):
        att_l, att_r = get_tendon_attachment_worlds(self.solver, self.model, self.state_0)
        rest_lengths = self.solver.tendon_seg_rest_length.numpy()
        starts = []
        ends = []
        for seg, rest_length in enumerate(rest_lengths):
            p0 = att_l[seg].astype(np.float64)
            p1 = att_r[seg].astype(np.float64)
            geom_length = float(np.linalg.norm(p1 - p0))
            slack = max(float(rest_length) - geom_length, 0.0)
            num = 1
            if slack > 1.0e-4:
                num = max(8, int(math.ceil(max(geom_length, float(rest_length)) / 0.08)))

            prev = self._slack_curve_point(p0, p1, float(rest_length), 0.0)
            for j in range(1, num + 1):
                cur = self._slack_curve_point(p0, p1, float(rest_length), j / num)
                starts.append(prev)
                ends.append(cur)
                prev = cur

        return (
            wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3),
        )

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = self._visual_cable_lines()
            self.viewer.log_lines("cable", starts, ends, colors=(0.2, 0.75, 0.9), width=0.008)
            tick_starts, tick_ends, tick_colors = self._material_tick_lines()
            self.viewer.log_lines("cable_material_ticks", tick_starts, tick_ends, colors=tick_colors, width=0.012)
            rail_starts = wp.array([(-1.70, -0.155, 2.05), (1.18, -0.155, 2.05)], dtype=wp.vec3)
            rail_ends = wp.array([(-0.62, -0.155, 2.05), (2.14, -0.155, 2.05)], dtype=wp.vec3)
            self.viewer.log_lines("slider_rails", rail_starts, rail_ends, colors=(0.7, 0.7, 0.7), width=0.006)
            marker_starts, marker_ends, marker_colors = self._active_marker_lines()
            self.viewer.log_lines("active_slider_marker", marker_starts, marker_ends, colors=marker_colors, width=0.012)
            self.viewer.end_frame()

    def _active_marker_lines(self):
        body_q = self.state_0.body_q.numpy()
        active_idx = self.right_idx if self.sim_time < 1.65 else self.left_idx
        pos = body_q[active_idx][:3].astype(np.float64)
        color = (0.95, 0.70, 0.15) if active_idx == self.right_idx else (0.95, 0.20, 0.55)
        starts = np.array(
            [
                pos + np.array([-0.055, -0.001, 0.19]),
                pos + np.array([0.055, -0.001, 0.19]),
            ],
            dtype=np.float32,
        )
        ends = np.array(
            [
                pos + np.array([0.055, -0.001, 0.19]),
                pos + np.array([0.0, -0.001, 0.28]),
            ],
            dtype=np.float32,
        )
        colors = np.array([color, color], dtype=np.float32)
        return wp.array(starts, dtype=wp.vec3), wp.array(ends, dtype=wp.vec3), wp.array(colors, dtype=wp.vec3)


def _transform_point_np(pose, point):
    p = pose[:3]
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], point)
    return p + point + q[3] * t + np.cross(q[:3], t)


def _cam_profile(theta):
    radius = 0.50 * (1.0 + 0.27 * np.cos(theta - 0.45) + 0.09 * np.cos(2.0 * theta + 1.1))
    x = 1.18 * radius * np.cos(theta)
    z = 0.86 * radius * np.sin(theta)
    return x, z


def _cam_pinhole_offsets(front_y):
    angles = np.deg2rad(np.array([180.0, 126.0, 58.0, 0.0], dtype=np.float32))
    offsets = []
    for theta in angles:
        x, z = _cam_profile(theta)
        offsets.append((float(x), -float(front_y), float(z)))
    return np.array(offsets, dtype=np.float32)


def _make_cam_mesh(half_width):
    n = 96
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    front_center = len(angles) * 2
    back_center = front_center + 1
    vertices = []
    for y in (-half_width, half_width):
        for theta in angles:
            x, z = _cam_profile(theta)
            vertices.append((x, y, z))
    vertices.append((0.0, -half_width, 0.0))
    vertices.append((0.0, half_width, 0.0))

    indices = []
    for i in range(n):
        j = (i + 1) % n
        front_i = i
        front_j = j
        back_i = n + i
        back_j = n + j
        indices.extend((front_i, front_j, back_j))
        indices.extend((front_i, back_j, back_i))
        indices.extend((front_center, front_i, front_j))
        indices.extend((back_center, back_j, back_i))

    return newton.Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        indices=np.array(indices, dtype=np.int32),
        compute_inertia=False,
    )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
