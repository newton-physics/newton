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
# Example Cable Drives
#
# Demonstrates VBD joint drives and limits for revolute, prismatic, and
# D6 joint types.  Eight cables compare drive tracking vs. drive+limit
# clamping across all three joint types.
#
# Command: uv run -m newton.examples cable_drives
#
###########################################################################

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
    joint_target_pos[dof_indices[i]] = amplitudes[i] * wp.sin(omega * t[0])


def _tf_mul(a, b):
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    pa, qa = a[:3], a[3:]
    pb, qb = b[:3], b[3:]
    ra = Rotation.from_quat([qa[0], qa[1], qa[2], qa[3]])
    rb = Rotation.from_quat([qb[0], qb[1], qb[2], qb[3]])
    q_out = (ra * rb).as_quat()
    p_out = pa + ra.apply(pb)
    return np.concatenate([p_out, q_out])


def _extract_angle(model, body_q_np, joint_index):
    """Extract free-axis rotation angle via swing-twist decomposition."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    parent_idx = model.joint_parent.numpy()[joint_index]
    child_idx = model.joint_child.numpy()[joint_index]
    qd_start = int(model.joint_qd_start.numpy()[joint_index])
    axis_local = model.joint_axis.numpy()[qd_start]

    X_pj = model.joint_X_p.numpy()[joint_index]
    X_cj = model.joint_X_c.numpy()[joint_index]

    parent_pose = body_q_np[parent_idx] if parent_idx >= 0 else np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    X_wp = _tf_mul(parent_pose, X_pj)
    X_wc = _tf_mul(body_q_np[child_idx], X_cj)

    q_wp = Rotation.from_quat([X_wp[3], X_wp[4], X_wp[5], X_wp[6]])
    q_wc = Rotation.from_quat([X_wc[3], X_wc[4], X_wc[5], X_wc[6]])
    r_err = (q_wp.inv() * q_wc).as_quat()

    a = axis_local / np.linalg.norm(axis_local)
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


def _extract_displacement(model, body_q_np, joint_index):
    """Extract free-axis linear displacement for a prismatic or D6 joint."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    parent_idx = model.joint_parent.numpy()[joint_index]
    child_idx = model.joint_child.numpy()[joint_index]
    qd_start = int(model.joint_qd_start.numpy()[joint_index])
    axis_local = model.joint_axis.numpy()[qd_start]

    X_pj = model.joint_X_p.numpy()[joint_index]
    X_cj = model.joint_X_c.numpy()[joint_index]

    parent_pose = body_q_np[parent_idx] if parent_idx >= 0 else np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    X_wp = _tf_mul(parent_pose, X_pj)
    X_wc = _tf_mul(body_q_np[child_idx], X_cj)

    q_wp = Rotation.from_quat([X_wp[3], X_wp[4], X_wp[5], X_wp[6]])
    axis_world = q_wp.apply(axis_local / np.linalg.norm(axis_local))
    return float(np.dot(X_wc[:3] - X_wp[:3], axis_world))


class Example:
    """Eight cables demonstrating joint drives and limits for revolute, prismatic,
    and D6 joint types.

    - Revolute drive:        tracks +/-1.2 rad oscillation (ke=2000, kd=100)
    - Revolute drive+limit:  same drive, clamped at +/-0.3 rad
    - Prismatic drive:       tracks +/-0.3 m oscillation (ke=5000, kd=200)
    - Prismatic drive+limit: same drive, clamped at +/-0.05 m
    - D6 linear drive:       free X, tracks +/-0.3 m
    - D6 linear drive+limit: free X, clamped at +/-0.05 m
    - D6 angular drive:      free Y rotation, tracks +/-1.2 rad
    - D6 angular drive+limit: free Y rotation, clamped at +/-0.3 rad
    """

    def __init__(self, viewer, args=None):
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_iterations = 1
        self.update_step_interval = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        num_segments = 50
        segment_length = 0.05
        cable_radius = 0.015
        bend_stiffness = 1.0e1
        bend_damping = 1.0e-2
        stretch_stiffness = 1.0e9

        self.omega = 2.0
        anchor_z = 4.0
        anchor_radius = 0.18

        JointDofConfig = newton.ModelBuilder.JointDofConfig

        self._cable_specs = [
            {
                "label": "rev_drive",
                "joint": "revolute",
                "dof": "ang",
                "amplitude": 1.2,
                "has_limit": False,
                "limit_bound": 0.0,
            },
            {
                "label": "rev_limit",
                "joint": "revolute",
                "dof": "ang",
                "amplitude": 1.2,
                "has_limit": True,
                "limit_bound": 0.3,
            },
            {
                "label": "prism_drive",
                "joint": "prismatic",
                "dof": "lin",
                "amplitude": 0.3,
                "has_limit": False,
                "limit_bound": 0.0,
            },
            {
                "label": "prism_limit",
                "joint": "prismatic",
                "dof": "lin",
                "amplitude": 0.3,
                "has_limit": True,
                "limit_bound": 0.05,
            },
            {
                "label": "d6_lin_drive",
                "joint": "d6",
                "dof": "lin",
                "amplitude": 0.3,
                "has_limit": False,
                "limit_bound": 0.0,
                "lin_axes": [
                    JointDofConfig(
                        axis=(1, 0, 0),
                        target_ke=5000.0,
                        target_kd=200.0,
                    )
                ],
                "ang_axes": [],
            },
            {
                "label": "d6_lin_limit",
                "joint": "d6",
                "dof": "lin",
                "amplitude": 0.3,
                "has_limit": True,
                "limit_bound": 0.05,
                "lin_axes": [
                    JointDofConfig(
                        axis=(1, 0, 0),
                        target_ke=5000.0,
                        target_kd=200.0,
                        limit_lower=-0.05,
                        limit_upper=0.05,
                        limit_ke=1.0e5,
                        limit_kd=1.0e2,
                    )
                ],
                "ang_axes": [],
            },
            {
                "label": "d6_ang_drive",
                "joint": "d6",
                "dof": "ang",
                "amplitude": 1.2,
                "has_limit": False,
                "limit_bound": 0.0,
                "lin_axes": [],
                "ang_axes": [
                    JointDofConfig(
                        axis=(0, 1, 0),
                        target_ke=2000.0,
                        target_kd=100.0,
                    )
                ],
            },
            {
                "label": "d6_ang_limit",
                "joint": "d6",
                "dof": "ang",
                "amplitude": 1.2,
                "has_limit": True,
                "limit_bound": 0.3,
                "lin_axes": [],
                "ang_axes": [
                    JointDofConfig(
                        axis=(0, 1, 0),
                        target_ke=2000.0,
                        target_kd=100.0,
                        limit_lower=-0.3,
                        limit_upper=0.3,
                        limit_ke=1.0e5,
                        limit_kd=1.0e1,
                    )
                ],
            },
        ]

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 0.8

        y_spacing = 1.0
        y0 = -3.5

        drive_amplitudes: list[float] = []

        for i, spec in enumerate(self._cable_specs):
            y = y0 + i * y_spacing

            anchor = builder.add_link(xform=wp.transform(wp.vec3(0.0, y, anchor_z), wp.quat_identity()), mass=0.0)
            builder.add_shape_sphere(body=anchor, radius=anchor_radius, label=f"anchor_{spec['label']}")
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
                label=f"cable_{spec['label']}",
                wrap_in_articulation=False,
            )

            parent_fq = rod_quats[0]
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, -anchor_radius), parent_fq)
            child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

            jtype = spec["joint"]
            if jtype == "revolute":
                lb = spec["limit_bound"]
                j = builder.add_joint_revolute(
                    parent=anchor,
                    child=rod_bodies[0],
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    axis=(0.0, 1.0, 0.0),
                    target_ke=2000.0,
                    target_kd=100.0,
                    limit_lower=-lb if spec["has_limit"] else None,
                    limit_upper=lb if spec["has_limit"] else None,
                    limit_ke=1.0e5 if spec["has_limit"] else 0.0,
                    limit_kd=1.0e1 if spec["has_limit"] else 0.0,
                    label=f"j_{spec['label']}",
                )
            elif jtype == "prismatic":
                lb = spec["limit_bound"]
                j = builder.add_joint_prismatic(
                    parent=anchor,
                    child=rod_bodies[0],
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    axis=(1.0, 0.0, 0.0),
                    target_ke=5000.0,
                    target_kd=200.0,
                    limit_lower=-lb if spec["has_limit"] else None,
                    limit_upper=lb if spec["has_limit"] else None,
                    limit_ke=1.0e5 if spec["has_limit"] else 0.0,
                    limit_kd=1.0e2 if spec["has_limit"] else 0.0,
                    label=f"j_{spec['label']}",
                )
            else:
                j = builder.add_joint_d6(
                    parent=anchor,
                    child=rod_bodies[0],
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    linear_axes=spec["lin_axes"],
                    angular_axes=spec["ang_axes"],
                    label=f"j_{spec['label']}",
                )

            builder.add_articulation([*rod_joints, j])
            drive_amplitudes.append(spec["amplitude"])

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

        joint_types = self.model.joint_type.numpy()
        joint_qd_start = self.model.joint_qd_start.numpy()

        driven_types = {int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC), int(newton.JointType.D6)}
        driven_joints = [i for i in range(self.model.joint_count) if int(joint_types[i]) in driven_types]
        assert len(driven_joints) == len(self._cable_specs)
        self._driven_joint_indices = driven_joints

        drive_dofs = [int(joint_qd_start[ji]) for ji in driven_joints]
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
        body_q_np = self.state_0.body_q.numpy()
        t = self.sim_time_array.numpy()[0]

        for i, spec in enumerate(self._cable_specs):
            ji = self._driven_joint_indices[i]
            dof = spec["dof"]
            label = spec["label"]
            target = spec["amplitude"] * np.sin(self.omega * t)

            if dof == "ang":
                measured = _extract_angle(self.model, body_q_np, ji)
                if spec["has_limit"]:
                    bound = spec["limit_bound"]
                    if abs(measured) > bound + 0.15:
                        raise AssertionError(
                            f"{label}: angular limit violated |theta|={abs(measured):.3f}, bound={bound}"
                        )
                else:
                    if abs(measured - target) > 0.5:
                        raise AssertionError(f"{label}: drive tracking theta={measured:.3f}, target={target:.3f}")
            else:
                measured = _extract_displacement(self.model, body_q_np, ji)
                if spec["has_limit"]:
                    bound = spec["limit_bound"]
                    if abs(measured) > bound + 0.05:
                        raise AssertionError(f"{label}: linear limit violated |d|={abs(measured):.3f}, bound={bound}")
                else:
                    if abs(measured - target) > 0.15:
                        raise AssertionError(f"{label}: drive tracking d={measured:.3f}, target={target:.3f}")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
