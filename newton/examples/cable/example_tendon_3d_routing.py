# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon 3D Routing
#
# Right-angle pulley drive with three cylinders.  P1 and P3 have their
# axes along Y (wrapping in XZ) while P2 at 90 degrees has its axis
# along X (wrapping in YZ).  The cable routes over P1, under P2, over
# P3 with weights on both ends.  The inter-pulley cable segments run
# approximately vertically, along the intersection of adjacent wrapping
# planes.
#
# Command: python -m newton.examples tendon_3d_routing
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.cable import (
    assert_tendon_total_length,
    get_tendon_attachment_worlds,
    get_tendon_cable_lines,
)


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

        self.r1 = 0.32
        self.r2 = 0.18
        self.r3 = 0.32
        self.p1_pos = wp.vec3(-self.r1, self.r1, 3.55)
        self.p2_pos = wp.vec3(0.0, 0.0, 2.25)
        self.p3_pos = wp.vec3(self.r3, -self.r3, 3.55)
        self.left_start_pos = wp.vec3(-2.0 * self.r1, self.r1, 1.55)
        self.right_start_pos = wp.vec3(2.0 * self.r3, -self.r3, 1.55)

        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)

        # P1 at (-r, +r), axis Y — wraps in XZ plane
        # Inner edge at x=0 aligns with P2 center; y=+r matches P2 tangent offset
        self.q_p1_init = np.array([-s, 0.0, 0.0, c])
        q_p1_wp = wp.quat(*self.q_p1_init.tolist())
        p1 = builder.add_body(
            xform=wp.transform(p=self.p1_pos, q=q_p1_wp),
            mass=0.5,
        )
        builder.add_shape_cylinder(p1, radius=self.r1, half_height=0.06)
        self.p1_idx = p1

        # P2 at (0, 0), axis X — wraps in YZ plane
        self.q_p2_init = np.array([0.0, s, 0.0, c])
        q_p2_wp = wp.quat(*self.q_p2_init.tolist())
        p2 = builder.add_body(
            xform=wp.transform(p=self.p2_pos, q=q_p2_wp),
            mass=0.5,
        )
        builder.add_shape_cylinder(p2, radius=self.r2, half_height=0.24)
        self.p2_idx = p2

        # P3 at (+r, -r), axis Y — wraps in XZ plane
        # Inner edge at x=0 aligns with P2 center; y=-r matches P2 tangent offset
        self.q_p3_init = np.array([-s, 0.0, 0.0, c])
        q_p3_wp = wp.quat(*self.q_p3_init.tolist())
        p3 = builder.add_body(
            xform=wp.transform(p=self.p3_pos, q=q_p3_wp),
            mass=0.5,
        )
        builder.add_shape_cylinder(p3, radius=self.r3, half_height=0.06)
        self.p3_idx = p3

        Dof = newton.ModelBuilder.JointDofConfig
        j_p1 = builder.add_joint_revolute(
            parent=-1,
            child=p1,
            axis=Axis.Z,
            parent_xform=wp.transform(p=self.p1_pos, q=q_p1_wp),
            child_xform=wp.transform(),
            label="pulley_1_axis",
        )
        j_p2 = builder.add_joint_revolute(
            parent=-1,
            child=p2,
            axis=Axis.Z,
            parent_xform=wp.transform(p=self.p2_pos, q=q_p2_wp),
            child_xform=wp.transform(),
            label="pulley_2_axis",
        )
        j_p3 = builder.add_joint_revolute(
            parent=-1,
            child=p3,
            axis=Axis.Z,
            parent_xform=wp.transform(p=self.p3_pos, q=q_p3_wp),
            child_xform=wp.transform(),
            label="pulley_3_axis",
        )

        free_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Y), Dof(axis=Axis.Z)]
        free_ang = [Dof(axis=Axis.X), Dof(axis=Axis.Y), Dof(axis=Axis.Z)]

        sphere_pos = self.left_start_pos
        left = builder.add_link(
            xform=wp.transform(p=sphere_pos, q=wp.quat_identity()),
            mass=1.5,
        )
        builder.add_shape_sphere(left, radius=0.10)
        j1 = builder.add_joint_d6(
            parent=-1,
            child=left,
            linear_axes=free_lin,
            angular_axes=free_ang,
            parent_xform=wp.transform(p=sphere_pos, q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        box_pos = self.right_start_pos
        right = builder.add_link(
            xform=wp.transform(p=box_pos, q=wp.quat_identity()),
            mass=3.5,
        )
        builder.add_shape_box(right, hx=0.12, hy=0.12, hz=0.12)
        j2 = builder.add_joint_d6(
            parent=-1,
            child=right,
            linear_axes=free_lin,
            angular_axes=free_ang,
            parent_xform=wp.transform(p=box_pos, q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        builder.add_articulation([j_p1])
        builder.add_articulation([j_p2])
        builder.add_articulation([j_p3])
        builder.add_articulation([j1])
        builder.add_articulation([j2])

        axis_z = (0.0, 0.0, 1.0)
        drive_mu = 1000.0
        builder.add_tendon()

        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.10),
            axis=(0.0, 1.0, 0.0),
        )
        builder.add_tendon_link(
            body=p1,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r1,
            orientation=1,
            mu=drive_mu,
            offset=(0.0, 0.0, 0.0),
            axis=axis_z,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=p2,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r2,
            orientation=-1,
            mu=drive_mu,
            offset=(0.0, 0.0, 0.0),
            axis=axis_z,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=p3,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r3,
            orientation=1,
            mu=drive_mu,
            offset=(0.0, 0.0, 0.0),
            axis=axis_z,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=right,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.12),
            axis=(0.0, 1.0, 0.0),
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=8,
            joint_linear_relaxation=0.8,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self._initial_body_q = self.state_0.body_q.numpy().copy()
        self._pulley_theta = np.zeros(3, dtype=np.float64)
        self._last_pulley_angles = None
        self._pulley_rotation_history = []

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(2.4, -4.8, 2.45), pitch=5.0, yaw=118.0)
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
    def _axis_angle(q, axis_index):
        return float(2.0 * np.arctan2(q[3 + axis_index], q[6]))

    @staticmethod
    def _angle_delta(prev_angle, angle):
        return float((angle - prev_angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _record_motion_sample(self):
        body_q = self.state_0.body_q.numpy()
        angles = np.array(
            [
                self._axis_angle(body_q[self.p1_idx], 2),
                self._axis_angle(body_q[self.p2_idx], 2),
                self._axis_angle(body_q[self.p3_idx], 2),
            ],
            dtype=np.float64,
        )
        if self._last_pulley_angles is not None:
            self._pulley_theta += np.array(
                [self._angle_delta(prev, cur) for prev, cur in zip(self._last_pulley_angles, angles, strict=True)],
                dtype=np.float64,
            )
        self._last_pulley_angles = angles
        self._pulley_rotation_history.append(self._pulley_theta.copy())

    def test_post_step(self):
        assert_tendon_total_length(self, rel_tol=0.06, allow_slack=True)
        if self.sim_time < self.frame_dt * 1.5:
            att_l, att_r = get_tendon_attachment_worlds(self.solver, self.model, self.state_0)
            for i in range(self.model.tendon_segment_count):
                dx = abs(att_l[i][0] - att_r[i][0])
                dy = abs(att_l[i][1] - att_r[i][1])
                dz = abs(att_l[i][2] - att_r[i][2])
                assert dx < 0.02, f"Segment {i} not vertical in x: dx={dx}"
                assert dz > 0.5, f"Segment {i} has too little vertical span: dz={dz}"
                if i in (0, 3):
                    assert dy < 0.02, f"End hanging segment {i} not vertical in y: dy={dy}"
                else:
                    assert dy < 0.20, f"Inter-pulley segment {i} drifted too far in y: dy={dy}"

    def test_final(self):
        assert_tendon_total_length(self, rel_tol=0.06, allow_slack=True)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"

        sphere_pos = body_q[3][:3]
        box_pos = body_q[4][:3]
        left_start = np.array(
            [self.left_start_pos[0], self.left_start_pos[1], self.left_start_pos[2]],
            dtype=np.float32,
        )
        right_start = np.array(
            [self.right_start_pos[0], self.right_start_pos[1], self.right_start_pos[2]],
            dtype=np.float32,
        )
        sphere_moved = np.linalg.norm(sphere_pos - left_start) > 0.1
        box_moved = np.linalg.norm(box_pos - right_start) > 0.1
        assert sphere_moved, f"Sphere should have moved from start: {sphere_pos}"
        assert box_moved, f"Box should have moved from start: {box_pos}"
        if not self._pulley_rotation_history:
            self._record_motion_sample()

        rotations = np.array(self._pulley_rotation_history)
        assert np.isfinite(rotations).all(), "Non-finite 3D routing pulley rotation"
        max_rot = np.max(np.abs(rotations), axis=0)
        assert np.all(max_rot > 0.05), f"All 3D routing pulleys should rotate under high-mu no-slip: {max_rot}"

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.9, 0.2, 0.2), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
