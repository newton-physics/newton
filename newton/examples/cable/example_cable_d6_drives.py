# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Cable D6 Drives
#
# Two cables hanging from kinematic sphere anchors via D6 joints
# (1 free linear axis = X, 1 free angular axis = Y), comparing drive
# tracking with and without joint limits:
#
#   y = -0.5   DRIVE        -- tracks oscillating targets:
#                               linear +/-0.3 m, angular +/-1.2 rad
#   y = +0.5   DRIVE+LIMIT  -- same drives, clamped at
#                               +/-0.05 m linear, +/-0.3 rad angular
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def advance_time(t: wp.array(dtype=float), dt: float):
    t[0] = t[0] + dt


@wp.kernel
def update_drive_targets(
    t: wp.array(dtype=float),
    dof_indices: wp.array(dtype=int),
    lin_amplitude: float,
    ang_amplitude: float,
    omega: float,
    joint_target_pos: wp.array(dtype=float),
):
    """Write oscillating target_pos for D6 drive cables.

    Each cable has 2 DOFs (linear + angular). Thread i writes to
    dof_indices[2*i] (linear) and dof_indices[2*i+1] (angular).
    """
    i = wp.tid()
    lin_idx = dof_indices[i * 2]
    ang_idx = dof_indices[i * 2 + 1]
    joint_target_pos[lin_idx] = lin_amplitude * wp.sin(omega * t[0])
    joint_target_pos[ang_idx] = ang_amplitude * wp.sin(omega * t[0])


def _extract_d6_state(model, body_q_np, joint_index):
    """Extract free-axis displacement and angle for a D6 joint.

    Uses the same convention as the VBD D6 constraint:
    - Linear: d = dot(x_c - x_p, axis_world) for the free linear axis.
    - Angular: twist-decomposition about the free angular axis.
    """
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    parent_idx = model.joint_parent.numpy()[joint_index]
    child_idx = model.joint_child.numpy()[joint_index]
    qd_start = int(model.joint_qd_start.numpy()[joint_index])

    # D6 axes: first linear axes, then angular axes.
    # For our config: DOF 0 = linear X, DOF 1 = angular Y.
    lin_axis_local = model.joint_axis.numpy()[qd_start]
    ang_axis_local = model.joint_axis.numpy()[qd_start + 1]

    X_pj = model.joint_X_p.numpy()[joint_index]
    X_cj = model.joint_X_c.numpy()[joint_index]

    def tf_mul(a, b):
        pa, qa = a[:3], a[3:]
        pb, qb = b[:3], b[3:]
        ra = Rotation.from_quat([qa[0], qa[1], qa[2], qa[3]])
        rb = Rotation.from_quat([qb[0], qb[1], qb[2], qb[3]])
        q_out = (ra * rb).as_quat()
        p_out = pa + ra.apply(pb)
        return np.concatenate([p_out, q_out])

    parent_pose = body_q_np[parent_idx] if parent_idx >= 0 else np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    child_pose = body_q_np[child_idx]

    X_wp = tf_mul(parent_pose, X_pj)
    X_wc = tf_mul(child_pose, X_cj)

    q_wp = Rotation.from_quat([X_wp[3], X_wp[4], X_wp[5], X_wp[6]])

    # Linear displacement along free axis
    lin_a = lin_axis_local / np.linalg.norm(lin_axis_local)
    axis_world = q_wp.apply(lin_a)
    x_p = X_wp[:3]
    x_c = X_wc[:3]
    d_lin = float(np.dot(x_c - x_p, axis_world))

    # Angular rotation about free axis
    q_wc = Rotation.from_quat([X_wc[3], X_wc[4], X_wc[5], X_wc[6]])
    r_err = (q_wp.inv() * q_wc).as_quat()

    a = ang_axis_local / np.linalg.norm(ang_axis_local)
    proj = np.dot(r_err[:3], a)
    twist_quat = np.array([a[0] * proj, a[1] * proj, a[2] * proj, r_err[3]])
    twist_norm = np.linalg.norm(twist_quat)
    if twist_norm < 1e-12:
        return d_lin, 0.0
    twist_quat /= twist_norm

    w = np.clip(twist_quat[3], -1.0, 1.0)
    angle = 2.0 * np.arccos(abs(w))
    sign = np.sign(np.dot(a, twist_quat[:3]))
    d_ang = float(angle * (sign if sign != 0.0 else 1.0))

    return d_lin, d_ang


class Example:
    """Two cables comparing D6 joint drive with and without limits.

    Each cable hangs from a kinematic sphere at z=4 via a D6 joint
    (1 free linear axis X, 1 free angular axis Y):

    - **Drive**: spring-damper (linear ke=5000/kd=200, angular ke=2000/kd=100) tracks
      +/-0.3 m linear and +/-1.2 rad angular oscillation.
    - **Drive+Limit**: same drives but joint limits clamp to +/-0.05 m linear and +/-0.3 rad angular.
    """

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        # Simulation cadence.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_iterations = 1
        self.update_step_interval = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Cable parameters -- same as example_cable_d6_joints for stability.
        num_segments = 50
        segment_length = 0.05
        cable_radius = 0.015
        bend_stiffness = 1.0e1
        bend_damping = 1.0e-2
        stretch_stiffness = 1.0e9

        # Shared oscillation parameters.
        self.omega = 2.0  # rad/s
        self.lin_drive_amplitude = 0.3  # m
        self.ang_drive_amplitude = 1.2  # rad
        self.lin_limit_bound = 0.05  # m
        self.ang_limit_bound = 0.3  # rad

        # Drive parameters (shared by both cables).
        lin_drive_ke = 5000.0
        lin_drive_kd = 200.0
        ang_drive_ke = 2000.0
        ang_drive_kd = 100.0

        # Layout.
        y_positions = [-0.5, 0.5]
        anchor_z = 4.0
        anchor_radius = 0.18

        JointDofConfig = newton.ModelBuilder.JointDofConfig

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e1
        builder.default_shape_cfg.mu = 0.8

        for i in range(2):
            y = y_positions[i]

            # Kinematic sphere anchor.
            anchor = builder.add_link(xform=wp.transform(wp.vec3(0.0, y, anchor_z), wp.quat_identity()), mass=0.0)
            builder.add_shape_sphere(body=anchor, radius=anchor_radius, label=f"anchor_{i}")
            builder.body_mass[anchor] = 0.0
            builder.body_inv_mass[anchor] = 0.0
            builder.body_inertia[anchor] = wp.mat33(0.0)
            builder.body_inv_inertia[anchor] = wp.mat33(0.0)

            # Cable hanging below anchor.
            cable_start = wp.vec3(0.0, y, anchor_z - anchor_radius)
            rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
                start=cable_start,
                direction=wp.vec3(0.0, 0.0, -1.0),
                length=float(num_segments) * float(segment_length),
                num_segments=num_segments,
            )

            rod_bodies, rod_joints = builder.add_rod(
                positions=rod_points,
                quaternions=rod_quats,
                radius=cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                stretch_stiffness=stretch_stiffness,
                label=f"cable_{i}",
                wrap_in_articulation=False,
            )

            # D6 joint -- both share the same anchor geometry.
            parent_frame_q = rod_quats[0]
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), parent_frame_q)
            child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

            if i == 0:
                # Drive only.
                j = builder.add_joint_d6(
                    parent=anchor,
                    child=rod_bodies[0],
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    linear_axes=[JointDofConfig(axis=(1, 0, 0), target_ke=lin_drive_ke, target_kd=lin_drive_kd)],
                    angular_axes=[JointDofConfig(axis=(0, 1, 0), target_ke=ang_drive_ke, target_kd=ang_drive_kd)],
                    label="d6_drive",
                )
            else:
                # Drive + limit.
                j = builder.add_joint_d6(
                    parent=anchor,
                    child=rod_bodies[0],
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    linear_axes=[
                        JointDofConfig(
                            axis=(1, 0, 0),
                            target_ke=lin_drive_ke,
                            target_kd=lin_drive_kd,
                            limit_lower=-self.lin_limit_bound,
                            limit_upper=self.lin_limit_bound,
                            limit_ke=1.0e5,
                            limit_kd=1.0e2,
                        )
                    ],
                    angular_axes=[
                        JointDofConfig(
                            axis=(0, 1, 0),
                            target_ke=ang_drive_ke,
                            target_kd=ang_drive_kd,
                            limit_lower=-self.ang_limit_bound,
                            limit_upper=self.ang_limit_bound,
                            limit_ke=1.0e5,
                            limit_kd=1.0e1,
                        )
                    ],
                    label="d6_limit",
                )

            builder.add_articulation([*rod_joints, j])

        builder.add_ground_plane()
        builder.color()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        self.device = self.solver.device
        self.sim_time_array = wp.zeros(1, dtype=float, device=self.device)

        # Resolve D6 joint DOF indices after finalize().
        joint_types = self.model.joint_type.numpy()
        joint_qd_start = self.model.joint_qd_start.numpy()

        d6_indices = [i for i in range(self.model.joint_count) if int(joint_types[i]) == int(newton.JointType.D6)]
        assert len(d6_indices) == 2, f"Expected 2 D6 joints, found {len(d6_indices)}"
        self._d6_joint_indices = d6_indices

        # Each D6 joint has 2 DOFs (linear + angular). Build flat list: [lin0, ang0, lin1, ang1].
        drive_dofs = []
        for idx in d6_indices:
            qd_s = int(joint_qd_start[idx])
            drive_dofs.append(qd_s)  # linear DOF
            drive_dofs.append(qd_s + 1)  # angular DOF
        self._drive_dof_indices = wp.array(drive_dofs, dtype=int, device=self.device)

        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Drive targets for both cables.
            wp.launch(
                kernel=update_drive_targets,
                dim=2,  # one thread per cable
                inputs=[
                    self.sim_time_array,
                    self._drive_dof_indices,
                    self.lin_drive_amplitude,
                    self.ang_drive_amplitude,
                    self.omega,
                    self.control.joint_target_pos,
                ],
                device=self.device,
            )

            update_step_history = (substep % self.update_step_interval) == 0
            if update_step_history:
                self.model.collide(self.state_0, self.contacts)

            self.solver.set_rigid_history_update(update_step_history)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            wp.launch(kernel=advance_time, dim=1, inputs=[self.sim_time_array, self.sim_dt], device=self.device)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Validate drive tracking and limit clamping."""
        body_q_np = self.state_0.body_q.numpy()
        t = self.sim_time_array.numpy()[0]
        lin_target = self.lin_drive_amplitude * math.sin(self.omega * t)
        ang_target = self.ang_drive_amplitude * math.sin(self.omega * t)

        d_lin_drive, d_ang_drive = _extract_d6_state(self.model, body_q_np, self._d6_joint_indices[0])
        d_lin_limit, d_ang_limit = _extract_d6_state(self.model, body_q_np, self._d6_joint_indices[1])

        # Drive cable should track the targets.
        if abs(d_lin_drive - lin_target) > 0.15:
            raise AssertionError(f"Linear drive tracking: d={d_lin_drive:.3f}, target={lin_target:.3f}")
        if abs(d_ang_drive - ang_target) > 0.5:
            raise AssertionError(f"Angular drive tracking: theta={d_ang_drive:.3f}, target={ang_target:.3f}")

        # Limit cable should be clamped within bounds.
        if abs(d_lin_limit) > self.lin_limit_bound + 0.05:
            raise AssertionError(f"Linear limit violated: |d|={abs(d_lin_limit):.3f}, bound={self.lin_limit_bound}")
        if abs(d_ang_limit) > self.ang_limit_bound + 0.15:
            raise AssertionError(
                f"Angular limit violated: |theta|={abs(d_ang_limit):.3f}, bound={self.ang_limit_bound}"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
