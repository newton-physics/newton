# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Cable Machine
#
# Cable machine with three pulleys of varying sizes routing a single
# tendon from a light capsule weight to a heavy box weight.  The box
# descends under gravity, pulling the capsule upward through the pulley
# chain.  All three pulleys rotate through the frictional rolling
# constraints.
#
# Demonstrates complex multi-pulley routing with diverse body shapes
# (capsules, boxes, cylinders).
#
# Command: python -m newton.examples tendon_cable_machine
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_attachment_worlds, get_tendon_cable_lines


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

        self.r1 = 0.25
        self.r2 = 0.15
        self.r3 = 0.22
        self.p1_pos = wp.vec3(-0.6, 0.0, 3.8)
        self.p2_pos = wp.vec3(0.5, 0.0, 4.4)
        self.p3_pos = wp.vec3(1.5, 0.0, 3.6)
        pulley_mass = 40.0
        pulley_shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.7, margin=0.01, gap=0.02)
        weight_shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.7, margin=0.01, gap=0.02)

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

        p1 = builder.add_body(
            xform=wp.transform(p=self.p1_pos, q=wp.quat_identity()),
            mass=pulley_mass,
            inertia=cylinder_inertia(pulley_mass, self.r1, 0.05),
            lock_inertia=True,
        )
        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
        builder.add_shape_cylinder(
            p1,
            xform=wp.transform(q=q_cyl),
            radius=self.r1,
            half_height=0.05,
            cfg=pulley_shape_cfg,
        )
        self.p1_idx = p1

        p2 = builder.add_body(
            xform=wp.transform(p=self.p2_pos, q=wp.quat_identity()),
            mass=pulley_mass,
            inertia=cylinder_inertia(pulley_mass, self.r2, 0.04),
            lock_inertia=True,
        )
        builder.add_shape_cylinder(
            p2,
            xform=wp.transform(q=q_cyl),
            radius=self.r2,
            half_height=0.04,
            cfg=pulley_shape_cfg,
        )
        self.p2_idx = p2

        p3 = builder.add_body(
            xform=wp.transform(p=self.p3_pos, q=wp.quat_identity()),
            mass=pulley_mass,
            inertia=cylinder_inertia(pulley_mass, self.r3, 0.05),
            lock_inertia=True,
        )
        builder.add_shape_cylinder(
            p3,
            xform=wp.transform(q=q_cyl),
            radius=self.r3,
            half_height=0.05,
            cfg=pulley_shape_cfg,
        )
        self.p3_idx = p3

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
        j_p3 = builder.add_joint_revolute(
            parent=-1,
            child=p3,
            axis=Axis.Y,
            parent_xform=wp.transform(p=self.p3_pos),
            child_xform=wp.transform(),
            label="pulley_3_y",
        )

        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = []

        capsule_pos = wp.vec3(-0.9, 0.0, 3.25)
        q_vert = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
        left = builder.add_link(
            xform=wp.transform(p=capsule_pos, q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_capsule(
            left,
            xform=wp.transform(q=q_vert),
            radius=0.14,
            half_height=0.18,
            cfg=weight_shape_cfg,
            color=(0.86, 0.72, 0.22),
        )
        self.left_idx = left
        j1 = builder.add_joint_d6(
            parent=-1,
            child=left,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=capsule_pos, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        box_pos = wp.vec3(1.8, 0.0, 2.7)
        right = builder.add_link(
            xform=wp.transform(p=box_pos, q=wp.quat_identity()),
            mass=2.0,
        )
        builder.add_shape_box(right, hx=0.12, hy=0.15, hz=0.10, cfg=weight_shape_cfg)
        self.right_idx = right
        j2 = builder.add_joint_d6(
            parent=-1,
            child=right,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=box_pos, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        builder.add_articulation([j_p1])
        builder.add_articulation([j_p2])
        builder.add_articulation([j_p3])
        builder.add_articulation([j1])
        builder.add_articulation([j2])

        axis = (0.0, 1.0, 0.0)
        drive_mu = 1000.0
        builder.add_tendon()

        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.32),
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
            compliance=1.0e-5,
            damping=5.0,
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
            compliance=1.0e-5,
            damping=5.0,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=p3,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r3,
            orientation=1,
            mu=drive_mu,
            offset=(0.0, 0.0, 0.0),
            axis=axis,
            compliance=1.0e-5,
            damping=5.0,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=right,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.15),
            axis=axis,
            compliance=1.0e-5,
            damping=5.0,
            rest_length=-1.0,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=32,
            joint_linear_relaxation=0.55,
            rigid_contact_relaxation=0.25,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        body_q = self.state_0.body_q.numpy()
        self._initial_capsule_z = float(body_q[self.left_idx][2])
        self._initial_box_z = float(body_q[self.right_idx][2])
        self._pulley_anchor_positions = np.array(
            [
                body_q[self.p1_idx][:3],
                body_q[self.p2_idx][:3],
                body_q[self.p3_idx][:3],
            ],
            dtype=float,
        )
        self._pulley_theta = np.zeros(3, dtype=np.float64)
        self._last_pulley_angle = [None, None, None]
        self._pulley_rotation_history = []
        self._capsule_z_history = []
        self._box_z_history = []
        self._capsule_attachment_history = []
        self._capsule_position_history = []
        self._box_position_history = []
        self._pulley_axis_error_history = []

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.4, -7.0, 2.5), pitch=5.0, yaw=90.0)
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

    def _record_motion_sample(self):
        body_q = self.state_0.body_q.numpy()
        for i, pulley_idx in enumerate((self.p1_idx, self.p2_idx, self.p3_idx)):
            angle = self._hinge_y_angle(body_q[pulley_idx])
            if self._last_pulley_angle[i] is not None:
                self._pulley_theta[i] += self._angle_delta(self._last_pulley_angle[i], angle)
            self._last_pulley_angle[i] = angle
        self._pulley_rotation_history.append(self._pulley_theta.copy())
        self._capsule_z_history.append(float(body_q[self.left_idx][2]))
        self._box_z_history.append(float(body_q[self.right_idx][2]))
        att_l, _ = get_tendon_attachment_worlds(self.solver, self.model, self.state_0)
        self._capsule_attachment_history.append(np.array(att_l[0], dtype=np.float64))
        self._capsule_position_history.append(np.array(body_q[self.left_idx][:3], dtype=np.float64))
        self._box_position_history.append(np.array(body_q[self.right_idx][:3], dtype=np.float64))
        pulley_positions = np.array(
            [
                body_q[self.p1_idx][:3],
                body_q[self.p2_idx][:3],
                body_q[self.p3_idx][:3],
            ],
            dtype=float,
        )
        self._pulley_axis_error_history.append(
            tuple(np.linalg.norm(pulley_positions - self._pulley_anchor_positions, axis=1))
        )

    def _assert_capsule_attachment_stays_below_p1(self, attachment, body_q):
        p1_center = body_q[self.p1_idx][:3]
        crown_limit = float(p1_center[2] + self.r1 + 0.04)
        side_limit = float(p1_center[0] + self.r1)
        assert float(attachment[2]) <= crown_limit, (
            f"Cable machine light-side cable attachment should not crest over P1: "
            f"attachment_z={attachment[2]:.4f}, crown_limit={crown_limit:.4f}"
        )
        assert float(attachment[0]) <= side_limit, (
            f"Cable machine light capsule should stay on the near side of P1: "
            f"attachment_x={attachment[0]:.4f}, side_limit={side_limit:.4f}"
        )

    def test_post_step(self):
        assert_tendon_total_length(self, rel_tol=0.30)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Cable machine produced non-finite body state"
        if self._capsule_attachment_history:
            self._assert_capsule_attachment_stays_below_p1(self._capsule_attachment_history[-1], body_q)
        if self._pulley_axis_error_history:
            axis_error = np.array(self._pulley_axis_error_history[-1])
            assert axis_error.max() < 0.05, (
                f"Cable machine pulleys should stay on revolute axes: "
                f"P1={axis_error[0]:.4f}, P2={axis_error[1]:.4f}, P3={axis_error[2]:.4f}"
            )
        if self.sim_time < self.frame_dt * 1.5:
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            p1_z = body_q[self.p1_idx][2]
            p3_z = body_q[self.p3_idx][2]
            assert att_r[0][2] > p1_z, (
                f"Cable should wrap over P1: arrival tangent z={att_r[0][2]:.3f} <= center z={p1_z:.3f}"
            )
            assert att_l[3][2] > p3_z, (
                f"Cable should wrap over P3: departure tangent z={att_l[3][2]:.3f} <= center z={p3_z:.3f}"
            )

    def test_final(self):
        assert_tendon_total_length(self, rel_tol=0.30)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"

        capsule_z = body_q[self.left_idx][2]
        box_z = body_q[self.right_idx][2]
        assert abs(box_z - self._initial_box_z) > 0.1 or abs(capsule_z - self._initial_capsule_z) > 0.1, (
            f"Cable machine bodies should move: capsule_z={capsule_z}, box_z={box_z}"
        )

        rotations = np.array(self._pulley_rotation_history, dtype=np.float64)
        assert np.isfinite(rotations).all(), "Non-finite cable machine pulley rotation history"
        max_rotations = np.max(np.abs(rotations), axis=0)
        assert np.all(max_rotations > 0.03), (
            f"All cable machine pulleys should rotate from the same high-mu tendon: {max_rotations}"
        )
        body_q = self.state_0.body_q.numpy()
        for attachment in self._capsule_attachment_history:
            self._assert_capsule_attachment_stays_below_p1(attachment, body_q)
        capsule_positions = np.array(self._capsule_position_history)
        box_positions = np.array(self._box_position_history)
        if len(capsule_positions) > 1 and len(box_positions) > 1:
            capsule_step = float(np.max(np.linalg.norm(np.diff(capsule_positions, axis=0), axis=1)))
            box_step = float(np.max(np.linalg.norm(np.diff(box_positions, axis=0), axis=1)))
            assert max(capsule_step, box_step) < 0.40, (
                f"Cable machine weights should not jump per frame: "
                f"capsule_step={capsule_step:.4f}, box_step={box_step:.4f}"
            )

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(1.0, 0.6, 0.1), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
