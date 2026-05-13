# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Pinhole Friction
#
# Three side-by-side Atwood machines threaded through fixed pinholes:
#
#   Left:   mu = 0.0
#   Center: mu = 0.2
#   Right:  mu = 10.0
#
# A pinhole is a zero-radius body-local routing point.  Its friction coefficient
# bounds rest-length transfer between adjacent spans using the local bend angle:
# zero friction slips freely, finite friction reduces slip, and high friction
# locks cable transfer against the fixed guide point.
#
# Command: python -m newton.examples tendon_pinhole_friction
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_cable_lines


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 24
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        self.guide_z = 3.55
        self.weight_x = 0.1
        self.mus = [0.0, 0.2, 10.0]
        self.x_offsets = [-1.5, 0.0, 1.5]
        self.guide_indices = []
        self.left_indices = []
        self.right_indices = []
        self._left_z_history = []
        self._right_z_history = []

        mass_light = 1.0
        mass_heavy = 1.04
        axis = (0.0, 1.0, 0.0)

        for mu, x_off in zip(self.mus, self.x_offsets, strict=True):
            guide = builder.add_body(
                xform=wp.transform(p=wp.vec3(x_off, 0.0, self.guide_z), q=wp.quat_identity()),
                mass=0.0,
                is_kinematic=True,
            )
            builder.add_shape_sphere(guide, radius=0.055)
            self.guide_indices.append(guide)

            left = self._add_planar_weight(builder, wp.vec3(x_off - self.weight_x, 0.0, 2.0), mass_light, 0.06)
            right = self._add_planar_weight(builder, wp.vec3(x_off + self.weight_x, 0.0, 2.0), mass_heavy, 0.10)
            self.left_indices.append(left)
            self.right_indices.append(right)

            builder.add_tendon()
            builder.add_tendon_link(
                body=left,
                link_type=int(newton.TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.06),
                axis=axis,
            )
            builder.add_tendon_link(
                body=guide,
                link_type=int(newton.TendonLinkType.PINHOLE),
                mu=mu,
                offset=(0.0, 0.0, 0.0),
                axis=axis,
                compliance=1.0e-5,
                damping=0.5,
                rest_length=-1.0,
            )
            builder.add_tendon_link(
                body=right,
                link_type=int(newton.TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.10),
                axis=axis,
                compliance=1.0e-5,
                damping=0.5,
                rest_length=-1.0,
            )

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=16,
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
            self.viewer.set_camera(pos=wp.vec3(0.0, -5.6, 2.55), pitch=4.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def _add_planar_weight(self, builder, pos, mass, half_extent):
        dof = newton.ModelBuilder.JointDofConfig
        body = builder.add_link(xform=wp.transform(p=pos, q=wp.quat_identity()), mass=mass)
        builder.add_shape_box(body, hx=half_extent, hy=half_extent, hz=half_extent)
        joint = builder.add_joint_d6(
            parent=-1,
            child=body,
            linear_axes=[dof(axis=newton.Axis.X), dof(axis=newton.Axis.Z)],
            angular_axes=[dof(axis=newton.Axis.Y)],
            parent_xform=wp.transform(p=pos),
            child_xform=wp.transform(),
        )
        builder.add_articulation([joint])
        return body

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
        assert_tendon_total_length(self, rel_tol=0.08)

    def test_final(self):
        assert_tendon_total_length(self, rel_tol=0.08)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in pinhole friction body state"
        if not self._right_z_history:
            self._record_motion_sample()

        left_z = np.array(self._left_z_history)
        right_z = np.array(self._right_z_history)
        assert np.isfinite(left_z).all() and np.isfinite(right_z).all(), "Non-finite pinhole friction trajectory"

        right_disp_history = self._initial_right_z - right_z
        metric_idx = int(np.argmax(right_disp_history[:, 0]))
        left_disp = left_z[metric_idx] - self._initial_left_z
        right_disp = right_disp_history[metric_idx]

        assert np.max(left_z[:, :2]) < self.guide_z - 0.05, (
            f"Low/mid pinhole weights should stay below the guide: max_z={np.max(left_z[:, :2], axis=0)}"
        )

        assert right_disp[0] > 0.25, f"Zero-friction pinhole should freely slip: dz={right_disp}"
        assert left_disp[0] > 0.20, f"Zero-friction light weight should rise through the pinhole: dz={left_disp}"
        assert right_disp[0] > right_disp[1] + 0.08, (
            f"Mid-friction pinhole should slip less than zero friction: dz={right_disp}"
        )
        assert right_disp[1] > right_disp[2] + 0.04, (
            f"High-friction pinhole should lock more than mid friction: dz={right_disp}"
        )
        assert right_disp[2] < 0.14, f"High-friction pinhole should lock cable transfer: dz={right_disp}"

        if len(right_z) > 2:
            step = np.max(np.abs(np.diff(right_z, axis=0)))
            assert step < 0.08, f"Pinhole friction weights should move smoothly per frame: step={step:.4f}"

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.9, 0.25, 0.15), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
