# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Compound Pulley
#
# Compound pulley system with two dynamic pulleys and balanced weights.
# The main cable routes: two linked left weights -> P1 (rolling) -> P2
# (rolling) -> right weight.  The two equal left weights are connected by a
# short secondary cable, and their combined mass equals the right-side mass.
#
# Command: python -m newton.examples tendon_compound_pulley
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Axis, TendonLinkType
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_attachment_worlds, get_tendon_cable_lines


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

        self.r1 = 0.25
        self.r2 = 0.20
        self.p1_pos = wp.vec3(-0.5, 0.0, 4.6)
        self.p2_pos = wp.vec3(0.5, 0.0, 4.2)
        pulley_mass = 80.0
        self.pulley_mass = pulley_mass
        self.mass_heavy = 4.0
        self.mass_left_each = 0.5 * self.mass_heavy
        self.mass_left_total = 2.0 * self.mass_left_each
        self.tendon_compliance = 1.0e-6
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        pulley_half_height = 0.05
        self.left_half_extent = 0.10
        self.right_half_extent = 0.13
        self.left_upper_pos = wp.vec3(-0.75, 0.0, 2.85)
        self.left_lower_pos = wp.vec3(-0.75, 0.0, 2.45)
        self.right_pos = wp.vec3(0.70, 0.0, 2.85)

        def cylinder_inertia(mass, radius, half_height):
            inertia_y = 0.5 * mass * radius * radius
            inertia_xz = (1.0 / 12.0) * mass * (3.0 * radius * radius + (2.0 * half_height) ** 2)
            return wp.mat33(
                inertia_xz,
                0.0,
                0.0,
                0.0,
                inertia_y,
                0.0,
                0.0,
                0.0,
                inertia_xz,
            )

        def box_inertia(mass, hx, hy, hz):
            sx = 2.0 * hx
            sy = 2.0 * hy
            sz = 2.0 * hz
            return wp.mat33(
                (1.0 / 12.0) * mass * (sy * sy + sz * sz),
                0.0,
                0.0,
                0.0,
                (1.0 / 12.0) * mass * (sx * sx + sz * sz),
                0.0,
                0.0,
                0.0,
                (1.0 / 12.0) * mass * (sx * sx + sy * sy),
            )

        p1 = builder.add_body(
            xform=wp.transform(p=self.p1_pos, q=wp.quat_identity()),
            mass=pulley_mass,
            inertia=cylinder_inertia(pulley_mass, self.r1, pulley_half_height),
            lock_inertia=True,
        )
        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
        builder.add_shape_cylinder(
            p1, xform=wp.transform(q=q_cyl), radius=self.r1, half_height=pulley_half_height, cfg=shape_cfg
        )
        self.p1_idx = p1
        self._p1_theta = 0.0
        self._p2_theta = 0.0
        self._last_p1_angle = None
        self._last_p2_angle = None
        self._pulley_rotation_history = []
        self._left_z_history = []
        self._right_z_history = []
        self._left_position_history = []
        self._left_lower_position_history = []
        self._right_position_history = []
        self._left_attachment_history = []
        self._pulley_axis_error_history = []
        self._direction_validation_frames = 60
        self._axis_validation_frames = 60

        p2 = builder.add_body(
            xform=wp.transform(p=self.p2_pos, q=wp.quat_identity()),
            mass=pulley_mass,
            inertia=cylinder_inertia(pulley_mass, self.r2, pulley_half_height),
            lock_inertia=True,
        )
        builder.add_shape_cylinder(
            p2, xform=wp.transform(q=q_cyl), radius=self.r2, half_height=pulley_half_height, cfg=shape_cfg
        )
        self.p2_idx = p2

        Dof = newton.ModelBuilder.JointDofConfig
        j_p1 = builder.add_joint_revolute(
            parent=-1,
            child=p1,
            axis=Axis.Y,
            parent_xform=wp.transform(p=self.p1_pos),
            child_xform=wp.transform(),
            label="pulley_1_y",
        )
        j_p2 = builder.add_joint_revolute(
            parent=-1,
            child=p2,
            axis=Axis.Y,
            parent_xform=wp.transform(p=self.p2_pos),
            child_xform=wp.transform(),
            label="pulley_2_y",
        )

        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = []

        self.left_idx = left = builder.add_link(
            xform=wp.transform(p=self.left_upper_pos, q=wp.quat_identity()),
            mass=self.mass_left_each,
            inertia=box_inertia(
                self.mass_left_each, self.left_half_extent, self.left_half_extent, self.left_half_extent
            ),
            lock_inertia=True,
        )
        builder.add_shape_box(
            left,
            hx=self.left_half_extent,
            hy=self.left_half_extent,
            hz=self.left_half_extent,
            cfg=shape_cfg,
            color=(0.28, 0.78, 0.32),
        )
        j1 = builder.add_joint_d6(
            parent=-1,
            child=left,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=self.left_upper_pos, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        self.left_lower_idx = left_lower = builder.add_link(
            xform=wp.transform(p=self.left_lower_pos, q=wp.quat_identity()),
            mass=self.mass_left_each,
            inertia=box_inertia(
                self.mass_left_each, self.left_half_extent, self.left_half_extent, self.left_half_extent
            ),
            lock_inertia=True,
        )
        builder.add_shape_box(
            left_lower,
            hx=self.left_half_extent,
            hy=self.left_half_extent,
            hz=self.left_half_extent,
            cfg=shape_cfg,
            color=(0.28, 0.78, 0.32),
        )
        j_left_lower = builder.add_joint_d6(
            parent=-1,
            child=left_lower,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=self.left_lower_pos, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            label="left_lower_planar",
        )

        self.right_idx = right = builder.add_link(
            xform=wp.transform(p=self.right_pos, q=wp.quat_identity()),
            mass=self.mass_heavy,
            inertia=box_inertia(
                self.mass_heavy, self.right_half_extent, self.right_half_extent, self.right_half_extent
            ),
            lock_inertia=True,
        )
        builder.add_shape_box(
            right,
            hx=self.right_half_extent,
            hy=self.right_half_extent,
            hz=self.right_half_extent,
            cfg=shape_cfg,
            color=(0.86, 0.42, 0.50),
        )
        j2 = builder.add_joint_d6(
            parent=-1,
            child=right,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=self.right_pos, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        builder.add_articulation([j_p1])
        builder.add_articulation([j_p2])
        builder.add_articulation([j1])
        builder.add_articulation([j_left_lower])
        builder.add_articulation([j2])

        axis = (0.0, 1.0, 0.0)
        drive_mu = 1000.0
        builder.add_tendon()
        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.09),
            axis=axis,
        )
        builder.add_tendon_link(
            body=p1,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r1,
            orientation=1,
            mu=drive_mu,
            offset=(0.0, 0.0, 0.0),
            axis=axis,
            compliance=self.tendon_compliance,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=p2,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r2,
            orientation=1,
            mu=drive_mu,
            offset=(0.0, 0.0, 0.0),
            axis=axis,
            compliance=self.tendon_compliance,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=right,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.13),
            axis=axis,
            compliance=self.tendon_compliance,
            damping=0.1,
            rest_length=-1.0,
        )

        builder.add_tendon()
        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, -self.left_half_extent),
            axis=axis,
        )
        builder.add_tendon_link(
            body=left_lower,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, self.left_half_extent),
            axis=axis,
            compliance=self.tendon_compliance,
            damping=0.1,
            rest_length=-1.0,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=32,
            joint_linear_relaxation=0.8,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        body_q = self.state_0.body_q.numpy()
        self._initial_left_z = self._left_center_of_mass_z(body_q)
        self._initial_left_lower_z = float(body_q[self.left_lower_idx][2])
        self._initial_right_z = float(body_q[self.right_idx][2])
        self._pulley_anchor_positions = np.array(
            [
                body_q[self.p1_idx][:3],
                body_q[self.p2_idx][:3],
            ],
            dtype=float,
        )
        self._expected_no_slip_accel = self._compute_expected_no_slip_acceleration()

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -5.0, 2.5), pitch=5.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._record_motion_sample()

    @staticmethod
    def _hinge_y_angle(q):
        return float(2.0 * np.arctan2(q[4], q[6]))

    @staticmethod
    def _angle_delta(prev_angle, angle):
        return float((angle - prev_angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _left_center_of_mass_z(self, body_q):
        return float(
            (self.mass_left_each * body_q[self.left_idx][2] + self.mass_left_each * body_q[self.left_lower_idx][2])
            / self.mass_left_total
        )

    def _assert_balanced_masses(self):
        body_mass = self.model.body_mass.numpy()
        left_total = float(body_mass[self.left_idx] + body_mass[self.left_lower_idx])
        right_mass = float(body_mass[self.right_idx])
        assert abs(left_total - right_mass) < 1.0e-6, (
            f"Compound pulley balanced setup requires left mass sum to equal right mass: "
            f"left_total={left_total:.6f}, right={right_mass:.6f}"
        )

    def _compute_expected_no_slip_acceleration(self):
        att_l = self.solver.tendon_seg_attachment_l.numpy()
        att_r = self.solver.tendon_seg_attachment_r.numpy()
        left_dir = att_r[0] - att_l[0]
        right_dir = att_r[2] - att_l[2]
        left_dir = left_dir / np.linalg.norm(left_dir)
        right_dir = right_dir / np.linalg.norm(right_dir)

        body_mass = self.model.body_mass.numpy()
        body_inertia = self.model.body_inertia.numpy()
        m_light = float(body_mass[self.left_idx] + body_mass[self.left_lower_idx])
        m_heavy = float(body_mass[self.right_idx])
        i1 = float(body_inertia[self.p1_idx][1][1])
        i2 = float(body_inertia[self.p2_idx][1][1])
        generalized_force = -9.81 * (m_light * left_dir[2] + m_heavy * right_dir[2])
        effective_mass = m_light + m_heavy + i1 / (self.r1 * self.r1) + i2 / (self.r2 * self.r2)
        return generalized_force / effective_mass

    def _expected_no_slip_acceleration(self):
        return self._expected_no_slip_accel

    def _fit_rim_acceleration(self, frame_count=16):
        history = np.array(self._pulley_rotation_history[:frame_count], dtype=float)
        if len(history) < frame_count:
            return None
        times = np.arange(len(history), dtype=float) * self.frame_dt
        A = np.column_stack([np.ones_like(times), times, 0.5 * times * times])
        p1_accel = np.linalg.lstsq(A, history[:, 0] * self.r1, rcond=None)[0][2]
        p2_accel = np.linalg.lstsq(A, history[:, 1] * self.r2, rcond=None)[0][2]
        return float(p1_accel), float(p2_accel)

    def _record_motion_sample(self):
        body_q = self.state_0.body_q.numpy()
        p1_angle = self._hinge_y_angle(body_q[self.p1_idx])
        p2_angle = self._hinge_y_angle(body_q[self.p2_idx])
        if self._last_p1_angle is not None:
            self._p1_theta += self._angle_delta(self._last_p1_angle, p1_angle)
            self._p2_theta += self._angle_delta(self._last_p2_angle, p2_angle)
        self._last_p1_angle = p1_angle
        self._last_p2_angle = p2_angle
        self._pulley_rotation_history.append((self._p1_theta, self._p2_theta))
        self._left_z_history.append(self._left_center_of_mass_z(body_q))
        self._right_z_history.append(float(body_q[self.right_idx][2]))
        self._left_position_history.append(np.array(body_q[self.left_idx][:3], dtype=np.float64))
        self._left_lower_position_history.append(np.array(body_q[self.left_lower_idx][:3], dtype=np.float64))
        self._right_position_history.append(np.array(body_q[self.right_idx][:3], dtype=np.float64))
        att_l, _ = get_tendon_attachment_worlds(self.solver, self.model, self.state_0)
        self._left_attachment_history.append(np.array(att_l[0], dtype=np.float64))
        pulley_positions = np.array(
            [
                body_q[self.p1_idx][:3],
                body_q[self.p2_idx][:3],
            ],
            dtype=float,
        )
        axis_error = np.linalg.norm(pulley_positions - self._pulley_anchor_positions, axis=1)
        self._pulley_axis_error_history.append(tuple(axis_error))

    def _assert_light_attachment_stays_below_p1(self, attachment, body_q):
        p1_center = body_q[self.p1_idx][:3]
        crown_limit = float(p1_center[2] + self.r1 + 0.04)
        side_limit = float(p1_center[0] + self.r1)
        assert float(attachment[2]) <= crown_limit, (
            f"Compound pulley light-side cable attachment should not crest over P1: "
            f"attachment_z={attachment[2]:.4f}, crown_limit={crown_limit:.4f}"
        )
        assert float(attachment[0]) <= side_limit, (
            f"Compound pulley light weight should stay on the near side of P1: "
            f"attachment_x={attachment[0]:.4f}, side_limit={side_limit:.4f}"
        )

    def test_post_step(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Compound pulley produced non-finite body state"
        assert float(np.max(np.abs(body_q[:, :3]))) < 20.0, "Compound pulley body state became unbounded"
        self._assert_balanced_masses()

        att_l = self.solver.tendon_seg_attachment_l.numpy()
        att_r = self.solver.tendon_seg_attachment_r.numpy()
        p1_center = body_q[self.p1_idx][:3]
        p2_center = body_q[self.p2_idx][:3]
        left_clearance = float(np.linalg.norm(att_l[0] - p1_center) - self.r1)
        right_clearance = float(np.linalg.norm(att_r[2] - p2_center) - self.r2)
        assert min(left_clearance, right_clearance) > 0.01, (
            f"Weight attachments should stay outside pulley tangent singularities: "
            f"left={left_clearance:.4f}, right={right_clearance:.4f}"
        )
        if self._left_attachment_history:
            self._assert_light_attachment_stays_below_p1(self._left_attachment_history[-1], body_q)

        if self._pulley_axis_error_history:
            frame = len(self._pulley_axis_error_history)
            axis_error = np.array(self._pulley_axis_error_history[-1])
            assert np.isfinite(axis_error).all(), f"Compound pulley axis drift became non-finite at frame {frame}"
            assert axis_error.max() < 0.05, (
                f"Compound pulleys should stay on revolute axes through frame {frame}: "
                f"P1={axis_error[0]:.4f}, P2={axis_error[1]:.4f}"
            )
        left_clearance = float(body_q[self.left_idx][2]) - self.left_half_extent
        left_lower_clearance = float(body_q[self.left_lower_idx][2]) - self.left_half_extent
        right_clearance = float(body_q[self.right_idx][2]) - self.right_half_extent
        assert left_clearance > -0.03 and left_lower_clearance > -0.03 and right_clearance > -0.03, (
            f"Compound pulley weights should remain supported by ground contact: "
            f"left_clearance={left_clearance:.4f}, left_lower_clearance={left_lower_clearance:.4f}, "
            f"right_clearance={right_clearance:.4f}"
        )
        if len(self._pulley_rotation_history) <= self._direction_validation_frames:
            assert_tendon_total_length(self, rel_tol=0.30)
        if self.sim_time < self.frame_dt * 1.5:
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            body_q = self.state_0.body_q.numpy()
            p1_z = body_q[self.p1_idx][2]
            p2_z = body_q[self.p2_idx][2]
            assert att_r[0][2] >= p1_z - 1.0e-3, (
                f"Cable should wrap onto P1 without crossing below its centerline: "
                f"arrival tangent z={att_r[0][2]:.3f}, center z={p1_z:.3f}"
            )
            assert att_l[2][2] >= p2_z - 1.0e-3, (
                f"Cable should wrap off P2 without crossing below its centerline: "
                f"departure tangent z={att_l[2][2]:.3f}, center z={p2_z:.3f}"
            )

    def test_final(self):
        if not self._pulley_rotation_history:
            self._record_motion_sample()

        self._assert_balanced_masses()
        sample = len(self._pulley_rotation_history) - 1
        left_z = self._left_z_history[sample]
        right_z = self._right_z_history[sample]
        left_dz = left_z - self._initial_left_z
        right_dz = right_z - self._initial_right_z
        history = np.array(self._pulley_rotation_history[: sample + 1])
        left_prefix = np.array(self._left_z_history[: sample + 1])
        right_prefix = np.array(self._right_z_history[: sample + 1])
        left_positions = np.array(self._left_position_history)
        left_lower_positions = np.array(self._left_lower_position_history)
        right_positions = np.array(self._right_position_history)
        axis_prefix_end = min(len(self._pulley_axis_error_history), self._axis_validation_frames)
        axis_prefix = np.array(self._pulley_axis_error_history[:axis_prefix_end])
        assert np.isfinite(history).all(), "Non-finite compound pulley rotation inside validated prefix"
        assert np.isfinite(left_prefix).all() and np.isfinite(right_prefix).all(), (
            "Non-finite compound pulley body motion inside validated prefix"
        )
        assert axis_prefix_end >= self._axis_validation_frames, (
            f"Compound pulley axis validation expected {self._axis_validation_frames} frames, got {axis_prefix_end}"
        )
        assert np.isfinite(axis_prefix).all(), "Non-finite compound pulley axis drift inside validated window"
        max_axis_error = float(axis_prefix.max())
        assert max_axis_error < 0.05, (
            f"Compound pulleys should not leave their revolute axes over the long run: "
            f"max axis drift={max_axis_error:.4f}"
        )
        expected_accel = self._expected_no_slip_acceleration()
        assert abs(expected_accel) < 0.005, (
            f"Compound pulley balanced setup should have near-zero no-slip acceleration: expected={expected_accel:.6f}"
        )
        max_left_drift = float(np.max(np.abs(left_prefix - self._initial_left_z)))
        max_right_drift = float(np.max(np.abs(right_prefix - self._initial_right_z)))
        assert max_left_drift < 0.04 and max_right_drift < 0.04, (
            f"Balanced compound pulley weights should stay near equilibrium: "
            f"left_final_dz={left_dz:.4f}, right_final_dz={right_dz:.4f}, "
            f"max_left={max_left_drift:.4f}, max_right={max_right_drift:.4f}"
        )
        body_q = self.state_0.body_q.numpy()
        left_gap = float(
            body_q[self.left_idx][2] - self.left_half_extent - (body_q[self.left_lower_idx][2] + self.left_half_extent)
        )
        initial_gap = float(self.left_upper_pos[2] - self.left_lower_pos[2] - 2.0 * self.left_half_extent)
        assert abs(left_gap - initial_gap) < 0.02, (
            f"Short cable between equal left weights should remain taut: gap={left_gap:.4f}, expected={initial_gap:.4f}"
        )
        for attachment in self._left_attachment_history:
            self._assert_light_attachment_stays_below_p1(attachment, body_q)
        if len(left_positions) > 1 and len(left_lower_positions) > 1 and len(right_positions) > 1:
            left_step = float(np.max(np.linalg.norm(np.diff(left_positions, axis=0), axis=1)))
            left_lower_step = float(np.max(np.linalg.norm(np.diff(left_lower_positions, axis=0), axis=1)))
            right_step = float(np.max(np.linalg.norm(np.diff(right_positions, axis=0), axis=1)))
            assert max(left_step, left_lower_step, right_step) < 0.20, (
                f"Compound pulley weights should not jump per frame: "
                f"left_step={left_step:.4f}, left_lower_step={left_lower_step:.4f}, "
                f"right_step={right_step:.4f}"
            )

        max_theta = np.max(np.abs(history), axis=0)
        max_rim = max_theta * np.array([self.r1, self.r2], dtype=float)
        assert max_rim[0] < 0.015 and max_rim[1] < 0.015, (
            f"Balanced compound pulley should not wind up the dynamic pulleys: "
            f"P1_theta={max_theta[0]:.4f}, P2_theta={max_theta[1]:.4f}, "
            f"P1_rim={max_rim[0]:.4f}, P2_rim={max_rim[1]:.4f}"
        )
        rim_accel = self._fit_rim_acceleration()
        assert rim_accel is not None, "Compound pulley acceleration validation needs at least 16 samples"
        assert max(abs(rim_accel[0]), abs(rim_accel[1])) < 0.03, (
            f"Balanced compound pulley rim acceleration should be near zero: "
            f"P1={rim_accel[0]:.4f}, P2={rim_accel[1]:.4f}"
        )

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.2, 0.7, 1.0), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
