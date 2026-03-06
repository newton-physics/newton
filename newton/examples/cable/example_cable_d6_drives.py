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
# Four cables hanging from kinematic sphere anchors via D6 joints,
# separated into linear and angular groups with drive vs. drive+limit:
#
#   Linear group (free linear X only, all angular locked):
#     lin_drive       -- drive tracks +/-0.3 m oscillation
#     lin_drive_limit -- same drive, clamped at +/-0.05 m
#
#   Angular group (free angular Y only, all linear locked):
#     ang_drive       -- drive tracks +/-1.2 rad oscillation
#     ang_drive_limit -- same drive, clamped at +/-0.3 rad
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
    amplitudes: wp.array(dtype=float),
    omega: float,
    joint_target_pos: wp.array(dtype=float),
):
    """Write oscillating target_pos for each driven DOF."""
    i = wp.tid()
    idx = dof_indices[i]
    joint_target_pos[idx] = amplitudes[i] * wp.sin(omega * t[0])


def _extract_d6_linear(model, body_q_np, joint_index):
    """Extract linear displacement along a D6 joint's free linear axis."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    parent_idx = model.joint_parent.numpy()[joint_index]
    child_idx = model.joint_child.numpy()[joint_index]
    qd_start = int(model.joint_qd_start.numpy()[joint_index])

    lin_axis_local = model.joint_axis.numpy()[qd_start]

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
    lin_a = lin_axis_local / np.linalg.norm(lin_axis_local)
    axis_world = q_wp.apply(lin_a)

    return float(np.dot(X_wc[:3] - X_wp[:3], axis_world))


def _extract_d6_angular(model, body_q_np, joint_index):
    """Extract angular rotation about a D6 joint's free angular axis."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    parent_idx = model.joint_parent.numpy()[joint_index]
    child_idx = model.joint_child.numpy()[joint_index]
    qd_start = int(model.joint_qd_start.numpy()[joint_index])

    ang_axis_local = model.joint_axis.numpy()[qd_start]

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
    q_wc = Rotation.from_quat([X_wc[3], X_wc[4], X_wc[5], X_wc[6]])
    r_err = (q_wp.inv() * q_wc).as_quat()

    a = ang_axis_local / np.linalg.norm(ang_axis_local)
    proj = np.dot(r_err[:3], a)
    twist_quat = np.array([a[0] * proj, a[1] * proj, a[2] * proj, r_err[3]])
    twist_norm = np.linalg.norm(twist_quat)
    if twist_norm < 1e-12:
        return 0.0
    twist_quat /= twist_norm

    w = np.clip(twist_quat[3], -1.0, 1.0)
    angle = 2.0 * np.arccos(abs(w))
    sign = np.sign(np.dot(a, twist_quat[:3]))
    return float(angle * (sign if sign != 0.0 else 1.0))


class Example:
    """Four cables comparing D6 joint drives with and without limits,
    separated into linear and angular groups.

    Linear group (free linear X only, all angular locked):
      lin_drive       -- spring-damper tracks +/-0.3 m oscillation
      lin_drive_limit -- same drive, clamped at +/-0.05 m

    Angular group (free angular Y only, all linear locked):
      ang_drive       -- spring-damper tracks +/-1.2 rad oscillation
      ang_drive_limit -- same drive, clamped at +/-0.3 rad
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

        # Cable parameters.
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

        # Drive stiffness/damping.
        lin_drive_ke = 5000.0
        lin_drive_kd = 200.0
        ang_drive_ke = 2000.0
        ang_drive_kd = 100.0

        anchor_z = 4.0
        anchor_radius = 0.18

        JointDofConfig = newton.ModelBuilder.JointDofConfig

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 0.8

        # Four configs: linear drive/limit pair, then angular drive/limit pair.
        y_spacing = 1.0
        y0 = -1.5 * y_spacing
        drive_configs = [
            {
                "label": "lin_drive",
                "y": y0 + 0 * y_spacing,
                "dof_type": "lin",
                "lin_axes": [JointDofConfig(axis=(1, 0, 0), target_ke=lin_drive_ke, target_kd=lin_drive_kd)],
                "ang_axes": [],
                "amplitude": self.lin_drive_amplitude,
            },
            {
                "label": "lin_drive_limit",
                "y": y0 + 1 * y_spacing,
                "dof_type": "lin",
                "lin_axes": [
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
                "ang_axes": [],
                "amplitude": self.lin_drive_amplitude,
            },
            {
                "label": "ang_drive",
                "y": y0 + 2 * y_spacing,
                "dof_type": "ang",
                "lin_axes": [],
                "ang_axes": [JointDofConfig(axis=(0, 1, 0), target_ke=ang_drive_ke, target_kd=ang_drive_kd)],
                "amplitude": self.ang_drive_amplitude,
            },
            {
                "label": "ang_drive_limit",
                "y": y0 + 3 * y_spacing,
                "dof_type": "ang",
                "lin_axes": [],
                "ang_axes": [
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
                "amplitude": self.ang_drive_amplitude,
            },
        ]

        # Track per-cable info for DOF resolution and testing.
        self._cable_configs = drive_configs
        drive_amplitudes: list[float] = []

        for cfg in drive_configs:
            y = cfg["y"]

            anchor = builder.add_link(xform=wp.transform(wp.vec3(0.0, y, anchor_z), wp.quat_identity()), mass=0.0)
            builder.add_shape_sphere(body=anchor, radius=anchor_radius, label=f"anchor_{cfg['label']}")
            builder.body_mass[anchor] = 0.0
            builder.body_inv_mass[anchor] = 0.0
            builder.body_inertia[anchor] = wp.mat33(0.0)
            builder.body_inv_inertia[anchor] = wp.mat33(0.0)

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
                label=f"cable_{cfg['label']}",
                wrap_in_articulation=False,
            )

            parent_frame_q = rod_quats[0]
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), parent_frame_q)
            child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

            j = builder.add_joint_d6(
                parent=anchor,
                child=rod_bodies[0],
                parent_xform=parent_xform,
                child_xform=child_xform,
                linear_axes=cfg["lin_axes"],
                angular_axes=cfg["ang_axes"],
                label=f"d6_{cfg['label']}",
            )
            builder.add_articulation([*rod_joints, j])

            drive_amplitudes.append(cfg["amplitude"])

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
        assert len(d6_indices) == len(drive_configs), (
            f"Expected {len(drive_configs)} D6 joints, found {len(d6_indices)}"
        )
        self._d6_joint_indices = d6_indices

        # Each D6 joint has 1 DOF. Build flat DOF index and amplitude arrays.
        drive_dofs = []
        for idx in d6_indices:
            drive_dofs.append(int(joint_qd_start[idx]))
        self._drive_dof_indices = wp.array(drive_dofs, dtype=int, device=self.device)
        self._drive_amplitudes = wp.array(drive_amplitudes, dtype=float, device=self.device)

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

            wp.launch(
                kernel=update_drive_targets,
                dim=self._drive_dof_indices.shape[0],
                inputs=[
                    self.sim_time_array,
                    self._drive_dof_indices,
                    self._drive_amplitudes,
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
        """Validate drive tracking and limit clamping for all four cables."""
        body_q_np = self.state_0.body_q.numpy()
        t = self.sim_time_array.numpy()[0]

        for i, cfg in enumerate(self._cable_configs):
            ji = self._d6_joint_indices[i]
            dof_type = cfg["dof_type"]
            label = cfg["label"]

            if dof_type == "lin":
                d = _extract_d6_linear(self.model, body_q_np, ji)
                target = self.lin_drive_amplitude * math.sin(self.omega * t)

                if "limit" not in label:
                    if abs(d - target) > 0.15:
                        raise AssertionError(f"{label}: linear drive tracking d={d:.3f}, target={target:.3f}")
                else:
                    if abs(d) > self.lin_limit_bound + 0.05:
                        raise AssertionError(
                            f"{label}: linear limit violated |d|={abs(d):.3f}, bound={self.lin_limit_bound}"
                        )
            else:
                theta = _extract_d6_angular(self.model, body_q_np, ji)
                target = self.ang_drive_amplitude * math.sin(self.omega * t)

                if "limit" not in label:
                    if abs(theta - target) > 0.5:
                        raise AssertionError(f"{label}: angular drive tracking theta={theta:.3f}, target={target:.3f}")
                else:
                    if abs(theta) > self.ang_limit_bound + 0.15:
                        raise AssertionError(
                            f"{label}: angular limit violated |theta|={abs(theta):.3f}, bound={self.ang_limit_bound}"
                        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
