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
# Example Cable Revolute Joints
#
# Visual test for VBD REVOLUTE joints with kinematic anchors:
# - Create multiple kinematic anchor bodies (sphere, capsule, box)
# - Attach a cable (rod) to each anchor via a REVOLUTE joint
# - Drive anchors kinematically and verify cables follow
# - The revolute joint allows free rotation about the joint axis
#   while keeping anchors coincident and perpendicular axes aligned.
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def compute_revolute_joint_error(
    parent_ids: wp.array(dtype=wp.int32),
    child_ids: wp.array(dtype=wp.int32),
    parent_local: wp.array(dtype=wp.vec3),
    child_local: wp.array(dtype=wp.vec3),
    parent_frame_q: wp.array(dtype=wp.quat),
    child_frame_q: wp.array(dtype=wp.quat),
    joint_axis_local: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    out_err_pos: wp.array(dtype=float),
    out_err_ang_perp: wp.array(dtype=float),
):
    """Test-only: compute REVOLUTE joint error (position + perpendicular-axis angle).

    For each i:
      - Position error: ||x_p - x_c||
      - Perpendicular angular error: angle of the rotation component perpendicular
        to the joint axis.
    """
    i = wp.tid()
    pb = parent_ids[i]
    cb = child_ids[i]

    Xp = body_q[pb]
    Xc = body_q[cb]

    pp = wp.transform_get_translation(Xp)
    qp = wp.transform_get_rotation(Xp)
    pc = wp.transform_get_translation(Xc)
    qc = wp.transform_get_rotation(Xc)

    p_anchor = pp + wp.quat_rotate(qp, parent_local[i])
    c_anchor = pc + wp.quat_rotate(qc, child_local[i])
    out_err_pos[i] = wp.length(p_anchor - c_anchor)

    # Angular difference between joint frames
    qpf = wp.mul(qp, parent_frame_q[i])
    qcf = wp.mul(qc, child_frame_q[i])
    dq = wp.mul(wp.quat_inverse(qpf), qcf)
    dq = wp.normalize(dq)
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])

    # Extract rotation vector
    axis_angle, angle = wp.quat_to_axis_angle(dq)
    rot_vec = axis_angle * angle

    # Project out the free-axis component
    a = wp.normalize(joint_axis_local[i])
    rot_perp = rot_vec - wp.dot(rot_vec, a) * a
    out_err_ang_perp[i] = wp.length(rot_perp)


@wp.kernel
def advance_time(t: wp.array(dtype=float), dt: float):
    t[0] = t[0] + dt


@wp.kernel
def move_kinematic_anchors(
    t: wp.array(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
    base_pos: wp.array(dtype=wp.vec3),
    phase: wp.array(dtype=float),
    ramp_time: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Kinematically animate anchor poses with rotation about both Y and X.

    The revolute joint axis is Y.
    - Y-rotation (free DOF):  the anchor visibly spins but the cable does NOT follow.
    - X-rotation (constrained DOF): the cable tilts with the anchor.
    The combination shows "partial following" unique to the revolute joint.
    """
    i = wp.tid()
    b = anchor_ids[i]

    t0 = t[0]
    u = wp.clamp(t0 / ramp_time, 0.0, 1.0)
    ramp = u * u * (3.0 - 2.0 * u)

    ti = t0 + phase[i]
    p0 = base_pos[i]

    angle_y = ramp * 1.5 * wp.sin(1.0 * ti)
    angle_x = ramp * 0.6 * wp.sin(1.7 * ti)
    q_y = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle_y)
    q_x = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle_x)
    q = wp.mul(q_x, q_y)

    body_q[b] = wp.transform(p0, q)
    body_qd[b] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class Example:
    """Visual test for VBD REVOLUTE joints with kinematic anchors.

    High-level structure:
    - Build several kinematic "driver" bodies (sphere/capsule/box).
    - For each driver, create a straight cable (rod) hanging down in world -Z.
    - Attach the first rod segment to the driver using a REVOLUTE joint.
    - Kinematically animate the drivers and verify the REVOLUTE joint keeps
      the anchor points coincident and perpendicular rotation axes aligned
      while allowing free rotation about the joint axis.
    """

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

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
        stretch_damping = 0.0

        builder = newton.ModelBuilder()

        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e1
        builder.default_shape_cfg.mu = 0.8

        driver_specs = [
            ("sphere",),
            ("capsule",),
            ("box",),
        ]

        self.anchor_bodies = []
        anchor_base_pos: list[wp.vec3] = []
        anchor_phase: list[float] = []

        revolute_parent_ids: list[int] = []
        revolute_child_ids: list[int] = []
        revolute_parent_local: list[wp.vec3] = []
        revolute_child_local: list[wp.vec3] = []
        revolute_parent_frame_q: list[wp.quat] = []
        revolute_child_frame_q: list[wp.quat] = []
        revolute_axis_local: list[wp.vec3] = []

        y0 = -0.6
        dy = 0.6
        z0 = 4.0

        for i, (kind,) in enumerate(driver_specs):
            x = 0.0
            y = y0 + dy * i
            z = z0

            if kind == "sphere":
                z = z0 - 0.05

            body = builder.add_link(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()), mass=0.0)

            r = 0.0
            hh = 0.0
            hx = 0.0
            hy = 0.0
            hz = 0.0
            attach_offset = 0.18
            key_prefix = f"revolute_{kind}_{i}"
            if kind == "sphere":
                r = 0.18
                builder.add_shape_sphere(body=body, radius=r, label=f"drv_{key_prefix}")
                attach_offset = r
            elif kind == "capsule":
                r = 0.12
                hh = 0.18
                capsule_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
                builder.add_shape_capsule(
                    body=body,
                    radius=r,
                    half_height=hh,
                    xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=capsule_q),
                    label=f"drv_{key_prefix}",
                )
                attach_offset = r
            elif kind == "box":
                hx, hy, hz = 0.18, 0.12, 0.10
                builder.add_shape_box(body=body, hx=hx, hy=hy, hz=hz, label=f"drv_{key_prefix}")
                attach_offset = hz

            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

            cable_attach_x = 0.1
            if kind in ("capsule", "box"):
                cable_attach_x = 0.2
            dz_body = z0 - z
            parent_anchor_local = wp.vec3(cable_attach_x, 0.0, -attach_offset + dz_body)
            anchor_world = wp.vec3(x + cable_attach_x, y, z0 - attach_offset)

            if kind in ("sphere", "capsule", "box"):
                x_local = cable_attach_x
                if kind == "sphere":
                    r = attach_offset
                    z_local = -math.sqrt(max(r * r - x_local * x_local, 0.0))
                elif kind == "capsule":
                    r = attach_offset
                    hh_f = hh
                    x_clamped = max(min(x_local, hh_f + r), -(hh_f + r))
                    dx = abs(x_clamped) - hh_f
                    if dx <= 0.0:
                        z_local = -r
                    else:
                        z_local = -math.sqrt(max(r * r - dx * dx, 0.0))
                    x_local = x_clamped
                else:
                    hx_f = hx
                    hz_f = hz
                    x_local = max(min(x_local, hx_f), -hx_f)
                    z_local = -hz_f

                parent_anchor_local = wp.vec3(x_local, 0.0, z_local)
                anchor_world = wp.vec3(x + x_local, y, z + z_local)

            rod_points, rod_quats = newton.utils.create_straight_cable_points_and_quaternions(
                start=anchor_world,
                direction=wp.vec3(0.0, 0.0, -1.0),
                length=float(num_segments) * float(segment_length),
                num_segments=int(num_segments),
                twist_total=0.0,
            )

            rod_bodies, rod_joints = builder.add_rod(
                positions=rod_points,
                quaternions=rod_quats,
                radius=cable_radius,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=stretch_damping,
                label=f"cable_{key_prefix}",
                wrap_in_articulation=False,
            )

            child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
            # Revolute joint axis: Y-axis in the joint frame, allowing swing in XZ
            joint_axis = (0.0, 1.0, 0.0)
            j_revolute = builder.add_joint_revolute(
                parent=body,
                child=rod_bodies[0],
                parent_xform=wp.transform(parent_anchor_local, wp.quat_identity()),
                child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
                axis=joint_axis,
                label=f"attach_{key_prefix}",
            )
            builder.add_articulation([*rod_joints, j_revolute])

            revolute_parent_ids.append(int(body))
            revolute_child_ids.append(int(rod_bodies[0]))
            revolute_parent_local.append(parent_anchor_local)
            revolute_child_local.append(child_anchor_local)
            revolute_parent_frame_q.append(wp.quat_identity())
            revolute_child_frame_q.append(wp.quat_identity())
            revolute_axis_local.append(wp.vec3(*joint_axis))

            self.anchor_bodies.append(body)
            anchor_base_pos.append(wp.vec3(x, y, z))
            anchor_phase.append(0.6 * i)

        builder.add_ground_plane()
        builder.color()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=0.1,
            rigid_joint_linear_ke=1.0e9,
            rigid_joint_angular_ke=1.0e9,
            rigid_joint_linear_k_start=1.0e5,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        self.device = self.solver.device
        self.anchor_bodies_wp = wp.array(self.anchor_bodies, dtype=wp.int32, device=self.device)
        self.anchor_base_pos_wp = wp.array(anchor_base_pos, dtype=wp.vec3, device=self.device)
        self.anchor_phase_wp = wp.array(anchor_phase, dtype=float, device=self.device)
        self.sim_time_array = wp.zeros(1, dtype=float, device=self.device)

        self._revolute_parent_ids = wp.array(revolute_parent_ids, dtype=wp.int32, device=self.device)
        self._revolute_child_ids = wp.array(revolute_child_ids, dtype=wp.int32, device=self.device)
        self._revolute_parent_local = wp.array(revolute_parent_local, dtype=wp.vec3, device=self.device)
        self._revolute_child_local = wp.array(revolute_child_local, dtype=wp.vec3, device=self.device)
        self._revolute_parent_frame_q = wp.array(revolute_parent_frame_q, dtype=wp.quat, device=self.device)
        self._revolute_child_frame_q = wp.array(revolute_child_frame_q, dtype=wp.quat, device=self.device)
        self._revolute_axis_local = wp.array(revolute_axis_local, dtype=wp.vec3, device=self.device)

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
                kernel=move_kinematic_anchors,
                dim=self.anchor_bodies_wp.shape[0],
                inputs=[
                    self.sim_time_array,
                    self.anchor_bodies_wp,
                    self.anchor_base_pos_wp,
                    self.anchor_phase_wp,
                    1.0,
                    self.state_0.body_q,
                    self.state_0.body_qd,
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
        if self._revolute_parent_ids.shape[0] == 0:
            return

        n = self._revolute_parent_ids.shape[0]
        err_pos = wp.zeros(n, dtype=float, device=self.device)
        err_ang = wp.zeros(n, dtype=float, device=self.device)
        wp.launch(
            kernel=compute_revolute_joint_error,
            dim=n,
            inputs=[
                self._revolute_parent_ids,
                self._revolute_child_ids,
                self._revolute_parent_local,
                self._revolute_child_local,
                self._revolute_parent_frame_q,
                self._revolute_child_frame_q,
                self._revolute_axis_local,
                self.state_0.body_q,
                err_pos,
                err_ang,
            ],
            device=self.device,
        )
        err_pos_max = float(np.max(err_pos.numpy()))
        err_ang_max = float(np.max(err_ang.numpy()))

        tol_pos = 1.0e-3
        tol_ang = 1.0e-2
        if err_pos_max > tol_pos or err_ang_max > tol_ang:
            raise AssertionError(
                "REVOLUTE joint error too large: "
                f"pos_max={err_pos_max:.6g} (tol={tol_pos}), "
                f"ang_perp_max={err_ang_max:.6g} (tol={tol_ang})"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
