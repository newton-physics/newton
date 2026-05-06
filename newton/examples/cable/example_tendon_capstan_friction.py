# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Capstan Friction
#
# Three side-by-side asymmetric Atwood machines with
# different capstan friction coefficients on the pulley:
#
#   Left:   mu = 0.0   (frictionless)
#   Center: mu = 0.05  (subcritical — visible partial grip)
#   Right:  mu = 10.0  (no-slip)
#
# Each pulley is a dynamic body on a hinge joint, free to rotate about Y.
# The Euler-Eytelwein capstan equation bounds the tension ratio across
# a frictional contact: T_tight / T_slack <= exp(mu * theta).
#
# For dynamic pulleys with mu>0 the cable grips the pulley (stick mode).
# The XPBD constraint solver couples the pulley's rotational inertia
# I/R^2 to the system through the shared body.  The capstan bound
# activates only when the required tension ratio exceeds exp(mu*theta).
# The pulleys use explicit high inertia and the planar weights keep a
# fixed orientation so contact with the pulley does not tumble the
# light body over the rim.
# With dynamic pulleys, low finite friction spins the pulley only partially.
# The red tab on each pulley marks rim rotation; the frictionless pulley
# should translate the cable without spinning.
#
# Command: python -m newton.examples tendon_capstan_friction
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

        self.pulley_radius = 0.22
        self.pulley_z = 4.1
        pulley_mass = 40.0
        pulley_inertia_y = 0.90
        pulley_inertia = wp.mat33(
            0.50,
            0.0,
            0.0,
            0.0,
            pulley_inertia_y,
            0.0,
            0.0,
            0.0,
            0.50,
        )
        self.mus = [0.0, 1.0, 10.0]
        self.x_offsets = [-1.5, 0.0, 1.5]

        mass_light = 1.0
        mass_heavy = 1.2
        contact_cfg = newton.ModelBuilder.ShapeConfig(mu=0.7, margin=0.01, gap=0.02)

        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))

        Dof = newton.ModelBuilder.JointDofConfig
        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = []

        self.pulley_indices = []
        self.left_indices = []
        self.right_indices = []
        self._pulley_theta = [0.0, 0.0, 0.0]
        self._last_pulley_angle = [None, None, None]
        self._pulley_rotation_history = []
        self._left_z_history = []
        self._right_z_history = []
        self._left_attachment_history = []
        self._left_position_history = []
        self._right_position_history = []

        for mu, x_off in zip(self.mus, self.x_offsets, strict=True):
            pulley_pos = wp.vec3(x_off, 0.0, self.pulley_z)
            pulley = builder.add_body(
                xform=wp.transform(p=pulley_pos, q=wp.quat_identity()),
                mass=pulley_mass,
                inertia=pulley_inertia,
                lock_inertia=True,
            )
            builder.add_shape_cylinder(
                pulley,
                xform=wp.transform(q=q_cyl),
                radius=self.pulley_radius,
                half_height=0.04,
                cfg=contact_cfg,
            )
            marker_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=False,
                has_particle_collision=False,
            )
            builder.add_shape_box(
                pulley,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, self.pulley_radius + 0.025)),
                hx=0.035,
                hy=0.055,
                hz=0.012,
                cfg=marker_cfg,
                color=(0.95, 0.10, 0.06),
            )
            j_pulley = builder.add_joint_revolute(
                parent=-1,
                child=pulley,
                axis=Axis.Y,
                parent_xform=wp.transform(p=pulley_pos),
                child_xform=wp.transform(),
            )
            builder.add_articulation([j_pulley])
            self.pulley_indices.append(pulley)

            left_pos = wp.vec3(x_off - 0.4, 0.0, 2.0)
            left = builder.add_link(
                xform=wp.transform(p=left_pos, q=wp.quat_identity()),
                mass=mass_light,
            )
            builder.add_shape_box(left, hx=0.06, hy=0.06, hz=0.06, cfg=contact_cfg)
            j1 = builder.add_joint_d6(
                parent=-1,
                child=left,
                linear_axes=planar_lin,
                angular_axes=planar_ang,
                parent_xform=wp.transform(p=left_pos),
                child_xform=wp.transform(),
            )
            builder.add_articulation([j1])
            self.left_indices.append(left)

            right_pos = wp.vec3(x_off + 0.4, 0.0, 2.0)
            right = builder.add_link(
                xform=wp.transform(p=right_pos, q=wp.quat_identity()),
                mass=mass_heavy,
            )
            builder.add_shape_box(right, hx=0.09, hy=0.09, hz=0.09, cfg=contact_cfg)
            j2 = builder.add_joint_d6(
                parent=-1,
                child=right,
                linear_axes=planar_lin,
                angular_axes=planar_ang,
                parent_xform=wp.transform(p=right_pos),
                child_xform=wp.transform(),
            )
            builder.add_articulation([j2])
            self.right_indices.append(right)

            axis = (0.0, 1.0, 0.0)
            builder.add_tendon()
            builder.add_tendon_link(
                body=left,
                link_type=int(TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.06),
                axis=axis,
            )
            builder.add_tendon_link(
                body=pulley,
                link_type=int(TendonLinkType.ROLLING),
                radius=self.pulley_radius,
                orientation=1,
                mu=mu,
                offset=(0.0, 0.0, 0.0),
                axis=axis,
                compliance=1.0e-5,
                damping=5.0,
                rest_length=-1.0,
            )
            builder.add_tendon_link(
                body=right,
                link_type=int(TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.06),
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

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -6.0, 3.0), pitch=5.0, yaw=90.0)
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
        for i, pulley_idx in enumerate(self.pulley_indices):
            angle = self._hinge_y_angle(body_q[pulley_idx])
            if self._last_pulley_angle[i] is not None:
                self._pulley_theta[i] += self._angle_delta(self._last_pulley_angle[i], angle)
            self._last_pulley_angle[i] = angle
        self._pulley_rotation_history.append(tuple(self._pulley_theta))
        self._left_z_history.append(tuple(float(body_q[idx][2]) for idx in self.left_indices))
        self._right_z_history.append(tuple(float(body_q[idx][2]) for idx in self.right_indices))
        att_l, _ = get_tendon_attachment_worlds(self.solver, self.model, self.state_0)
        self._left_attachment_history.append(
            np.array([att_l[i * 2] for i in range(len(self.left_indices))], dtype=np.float64)
        )
        self._left_position_history.append(np.array([body_q[idx][:3] for idx in self.left_indices], dtype=np.float64))
        self._right_position_history.append(np.array([body_q[idx][:3] for idx in self.right_indices], dtype=np.float64))

    def _assert_light_attachments_stay_below_pulleys(self, attachments, body_q, label="Dynamic capstan"):
        for i, attachment in enumerate(attachments):
            pulley_center = body_q[self.pulley_indices[i]][:3]
            side_limit = float(pulley_center[0] + self.pulley_radius)
            assert float(attachment[0]) <= side_limit, (
                f"{label} case {i} light weight should stay on the near side of the pulley: "
                f"attachment_x={attachment[0]:.4f}, side_limit={side_limit:.4f}"
            )

    def test_post_step(self):
        assert_tendon_total_length(self, rel_tol=0.60)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Dynamic capstan produced non-finite body state"
        if self._left_attachment_history:
            self._assert_light_attachments_stay_below_pulleys(self._left_attachment_history[-1], body_q)
        if self.sim_time < self.frame_dt * 1.5:
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            for i, p_idx in enumerate(self.pulley_indices):
                pulley_z = body_q[p_idx][2]
                seg = i * 2
                assert att_r[seg][2] > pulley_z, (
                    f"Atwood {i}: arrival tangent z={att_r[seg][2]:.3f} <= center z={pulley_z:.3f}"
                )
                assert att_l[seg + 1][2] > pulley_z, (
                    f"Atwood {i}: departure tangent z={att_l[seg + 1][2]:.3f} <= center z={pulley_z:.3f}"
                )

    def test_final(self):
        assert_tendon_total_length(self, rel_tol=0.60)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"
        if not self._pulley_rotation_history:
            self._record_motion_sample()

        theta = np.array(self._pulley_rotation_history)
        assert np.isfinite(theta).all(), "Non-finite dynamic capstan pulley rotation history"
        capstan_idx = min(39, len(theta) - 1)
        capstan_theta = theta[capstan_idx]
        right_pos = np.array(self._right_position_history, dtype=np.float64)
        body_q = self.state_0.body_q.numpy()
        for attachments in self._left_attachment_history:
            self._assert_light_attachments_stay_below_pulleys(attachments, body_q)
        if len(right_pos) > 1:
            right_step = float(np.max(np.linalg.norm(np.diff(right_pos, axis=0), axis=2)))
            assert right_step < 0.40, f"Dynamic capstan heavy weights should not jump per frame: step={right_step:.4f}"
        left_z = np.array(self._left_z_history, dtype=np.float64)
        max_left_z = np.max(left_z, axis=0)
        light_limit = self.pulley_z + self.pulley_radius - 0.02
        assert np.all(max_left_z < light_limit), (
            f"Dynamic capstan light weights should not crest over the pulleys: "
            f"max_z={max_left_z}, limit={light_limit:.4f}"
        )
        assert abs(capstan_theta[0]) < 0.08, (
            f"Dynamic capstan zero-friction pulley should not rotate before payload contact: "
            f"theta={capstan_theta[0]:.4f}, all={capstan_theta}"
        )
        assert capstan_theta[1] > 0.25, (
            f"Dynamic capstan middle finite-friction pulley should rotate with heavy-side cable travel before contact: "
            f"theta={capstan_theta[1]:.4f}, all={capstan_theta}"
        )
        assert capstan_theta[2] > capstan_theta[1], (
            f"Dynamic capstan high-friction pulley should rotate at least as much as the mid-friction case before contact: "
            f"mid={capstan_theta[1]:.4f}, high={capstan_theta[2]:.4f}, all={capstan_theta}"
        )

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines(
                "cable",
                starts,
                ends,
                colors=(0.8, 0.5, 0.2),
                width=0.008,
            )
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
