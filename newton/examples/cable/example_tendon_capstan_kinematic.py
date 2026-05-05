# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Capstan Kinematic
#
# Three side-by-side asymmetric Atwood machines with
# kinematic (fixed) pulleys and different placeholder capstan friction
# coefficients:
#
#   Left:   mu = 0.0
#   Center: mu = 0.05
#   Right:  mu = 10.0
#
# The unified capstan projection uses the same path for all three cases:
# zero friction slips freely, finite friction reduces slip, and high friction
# locks against the fixed pulley.
#
# Command: python -m newton.examples tendon_capstan_kinematic
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_cable_lines


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

        self.pulley_radius = 0.15
        self.mus = [0.0, 0.05, 10.0]
        self.x_offsets = [-1.5, 0.0, 1.5]

        mass_light = 1.0
        mass_heavy = 1.08

        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))

        Dof = newton.ModelBuilder.JointDofConfig
        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = []

        self.pulley_indices = []
        self.left_indices = []
        self.right_indices = []
        self._left_z_history = []
        self._right_z_history = []

        for mu, x_off in zip(self.mus, self.x_offsets, strict=True):
            pulley_pos = wp.vec3(x_off, 0.0, 3.5)
            pulley = builder.add_body(
                xform=wp.transform(p=pulley_pos, q=wp.quat_identity()),
                mass=0.0,
                is_kinematic=True,
            )
            builder.add_shape_cylinder(
                pulley, xform=wp.transform(q=q_cyl),
                radius=self.pulley_radius, half_height=0.04,
            )
            self.pulley_indices.append(pulley)

            left_pos = wp.vec3(x_off - 0.4, 0.0, 1.85)
            left = builder.add_link(
                xform=wp.transform(p=left_pos, q=wp.quat_identity()),
                mass=mass_light,
            )
            builder.add_shape_box(left, hx=0.06, hy=0.06, hz=0.06)
            j1 = builder.add_joint_d6(
                parent=-1, child=left,
                linear_axes=planar_lin, angular_axes=planar_ang,
                parent_xform=wp.transform(p=left_pos),
                child_xform=wp.transform(),
            )
            builder.add_articulation([j1])
            self.left_indices.append(left)

            right_pos = wp.vec3(x_off + 0.4, 0.0, 1.85)
            right = builder.add_link(
                xform=wp.transform(p=right_pos, q=wp.quat_identity()),
                mass=mass_heavy,
            )
            builder.add_shape_box(right, hx=0.09, hy=0.09, hz=0.09)
            j2 = builder.add_joint_d6(
                parent=-1, child=right,
                linear_axes=planar_lin, angular_axes=planar_ang,
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
                compliance=2.0e-5,
                damping=8.0,
                rest_length=-1.0,
            )
            builder.add_tendon_link(
                body=right,
                link_type=int(TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.06),
                axis=axis,
                compliance=2.0e-5,
                damping=8.0,
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
        self._initial_left_z = np.array([body_q[i][2] for i in self.left_indices], dtype=np.float64)
        self._initial_right_z = np.array([body_q[i][2] for i in self.right_indices], dtype=np.float64)

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -6.0, 2.5), pitch=5.0, yaw=90.0)
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

    def _record_motion_sample(self):
        body_q = self.state_0.body_q.numpy()
        self._left_z_history.append(np.array([body_q[i][2] for i in self.left_indices], dtype=np.float64))
        self._right_z_history.append(np.array([body_q[i][2] for i in self.right_indices], dtype=np.float64))

    def test_post_step(self):
        assert_tendon_total_length(self, rel_tol=0.06)
        if self.sim_time < self.frame_dt * 1.5:
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            body_q = self.state_0.body_q.numpy()
            for i, p_idx in enumerate(self.pulley_indices):
                pulley_z = body_q[p_idx][2]
                seg = i * 2
                assert att_r[seg][2] > pulley_z, (
                    f"Atwood {i}: arrival tangent z={att_r[seg][2]:.3f} "
                    f"<= center z={pulley_z:.3f}"
                )
                assert att_l[seg + 1][2] > pulley_z, (
                    f"Atwood {i}: departure tangent z={att_l[seg + 1][2]:.3f} "
                    f"<= center z={pulley_z:.3f}"
                )

    def test_final(self):
        assert_tendon_total_length(self, rel_tol=0.06)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"
        if not self._right_z_history:
            self._record_motion_sample()

        left_z = np.array(self._left_z_history)
        right_z = np.array(self._right_z_history)
        assert np.isfinite(left_z).all() and np.isfinite(right_z).all(), "Non-finite capstan trajectory"

        left_disp = left_z[-1] - self._initial_left_z
        right_disp = self._initial_right_z - right_z[-1]
        assert np.all(left_disp > -0.04), f"Light weights should not sink in kinematic capstan: dz={left_disp}"
        assert np.all(right_disp > -0.02), f"Heavy weights should not rise in kinematic capstan: dz={right_disp}"

        assert right_disp[0] > 0.18, f"Zero-friction kinematic capstan should freely slip: dz={right_disp}"
        assert right_disp[0] > right_disp[1] + 0.03, (
            f"Mid-friction kinematic capstan should slip less than zero friction: dz={right_disp}"
        )
        assert right_disp[1] > right_disp[2] + 0.03, (
            f"High-friction kinematic capstan should lock more than mid friction: dz={right_disp}"
        )
        assert right_disp[2] < 0.08, f"High-friction kinematic capstan should lock cable motion: dz={right_disp}"
        if len(left_z) > 2:
            left_step = np.max(np.abs(np.diff(left_z[:, :2], axis=0)), axis=0)
            right_step = np.max(np.abs(np.diff(right_z[:, :2], axis=0)), axis=0)
            left_acc_step = np.max(np.abs(np.diff(left_z[:, :2], n=2, axis=0)), axis=0)
            right_acc_step = np.max(np.abs(np.diff(right_z[:, :2], n=2, axis=0)), axis=0)
            assert np.max(left_step) < 0.035 and np.max(right_step) < 0.035, (
                f"Low/mid kinematic capstan slip should move smoothly per frame: "
                f"left_step={left_step}, right_step={right_step}"
            )
            assert np.max(left_acc_step) < 0.012 and np.max(right_acc_step) < 0.012, (
                f"Low/mid kinematic capstan slip should not show stick-slip jumps: "
                f"left_ddz={left_acc_step}, right_ddz={right_acc_step}"
            )

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines(
                "cable", starts, ends,
                colors=(0.8, 0.5, 0.2), width=0.008,
            )
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
