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
# Example Cable Joints
#
# Demonstrates all VBD joint types (BALL, FIXED, REVOLUTE, PRISMATIC, D6)
# with kinematic anchors and cables.  Each row shows a different joint
# type with one or two representative configurations.
#
# Command: uv run -m newton.examples cable_joints
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

_MOTION_ROTATE_X = 0
_MOTION_ROTATE_Y = 1
_MOTION_TRANSLATE_X = 2
_MOTION_TRANSLATE_Y = 3


@wp.kernel
def compute_joint_error(
    parent_ids: wp.array(dtype=wp.int32),
    child_ids: wp.array(dtype=wp.int32),
    parent_local: wp.array(dtype=wp.vec3),
    child_local: wp.array(dtype=wp.vec3),
    parent_frame_q: wp.array(dtype=wp.quat),
    child_frame_q: wp.array(dtype=wp.quat),
    locked_lin_axis: wp.array(dtype=wp.vec3),
    locked_ang_axis: wp.array(dtype=wp.vec3),
    has_free_lin: wp.array(dtype=wp.int32),
    ang_error_mode: wp.array(dtype=wp.int32),
    free_rot_mode: wp.array(dtype=wp.int32),
    free_rot_axis: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    out_err_pos: wp.array(dtype=float),
    out_err_ang: wp.array(dtype=float),
    out_d_along: wp.array(dtype=float),
    out_rot_along_free: wp.array(dtype=float),
):
    """Compute joint constraint errors for verification.

    Position error is measured as full anchor distance (has_free_lin=0) or
    projected along a single locked axis (has_free_lin=1).

    Angular error mode:
      - 0: full relative rotation magnitude
      - 1: rotation about a locked axis only
      - 2: REVOLUTE-style perpendicular rotation magnitude

    Free rotation mode:
      - 0: no free-rotation metric
      - 1: full relative rotation magnitude
      - 2: rotation projected along ``free_rot_axis``

    d_along and rot_along_free measure the driven X / Y freedoms used by
    the dedicated per-joint example tests.
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
    C = c_anchor - p_anchor

    qpf = wp.mul(qp, parent_frame_q[i])

    if has_free_lin[i] == 1:
        la = wp.normalize(locked_lin_axis[i])
        la_world = wp.normalize(wp.quat_rotate(qpf, la))
        out_err_pos[i] = wp.abs(wp.dot(C, la_world))
        x_world = wp.normalize(wp.quat_rotate(qpf, wp.vec3(1.0, 0.0, 0.0)))
        out_d_along[i] = wp.dot(C, x_world)
    else:
        out_err_pos[i] = wp.length(C)
        out_d_along[i] = 0.0

    qcf = wp.mul(qc, child_frame_q[i])
    dq = wp.mul(wp.quat_inverse(qpf), qcf)
    dq = wp.normalize(dq)
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])

    axis_angle, angle = wp.quat_to_axis_angle(dq)
    rot_vec = axis_angle * angle

    mode = ang_error_mode[i]
    if mode == 1:
        aa = wp.normalize(locked_ang_axis[i])
        out_err_ang[i] = wp.abs(wp.dot(rot_vec, aa))
    elif mode == 2:
        a_free = wp.normalize(free_rot_axis[i])
        rot_along = wp.dot(rot_vec, a_free)
        out_err_ang[i] = wp.length(rot_vec - rot_along * a_free)
    else:
        out_err_ang[i] = wp.length(rot_vec)

    rot_mode = free_rot_mode[i]
    if rot_mode == 1:
        out_rot_along_free[i] = wp.length(rot_vec)
    elif rot_mode == 2:
        out_rot_along_free[i] = wp.dot(rot_vec, wp.normalize(free_rot_axis[i]))
    else:
        out_rot_along_free[i] = 0.0


@wp.kernel
def advance_time(t: wp.array(dtype=float), dt: float):
    t[0] = t[0] + dt


@wp.kernel
def move_kinematic_anchors(
    t: wp.array(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
    base_pos: wp.array(dtype=wp.vec3),
    phase: wp.array(dtype=float),
    motion_type: wp.array(dtype=wp.int32),
    ramp_time: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Kinematically animate anchor poses.

    motion_type 0: rotate about X.
    motion_type 1: rotate about Y.
    motion_type 2: translate along X.
    motion_type 3: translate along Y.
    """
    i = wp.tid()
    b = anchor_ids[i]

    t0 = t[0]
    u = wp.clamp(t0 / ramp_time, 0.0, 1.0)
    ramp = u * u * (3.0 - 2.0 * u)

    ti = t0 + phase[i]
    p0 = base_pos[i]
    m = motion_type[i]

    w = 1.5
    pos = p0
    q = wp.quat_identity()

    if m == 0:
        ang = (ramp * 0.8) * wp.sin(w * ti)
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), ang)
    elif m == 1:
        ang = (ramp * 1.6) * wp.sin(w * ti + 0.7)
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ang)
    elif m == 2:
        dx = (ramp * 0.4) * wp.sin(w * ti)
        pos = wp.vec3(p0[0] + dx, p0[1], p0[2])
    else:
        dy = (ramp * 0.5) * wp.sin(w * ti)
        pos = wp.vec3(p0[0], p0[1] + dy, p0[2])

    body_q[b] = wp.transform(pos, q)
    body_qd[b] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class Example:
    """Demonstrates all VBD joint types with kinematic anchors and cables.

    Eight configurations arranged in a column, one or two per joint type:

    - BALL:             anchor rotates about Y, cable ignores (angular freedom)
    - FIXED:            anchor rotates about Y, cable follows (full lock)
    - REVOLUTE (locked): anchor rotates about X (locked), cable follows
    - REVOLUTE (free):   anchor rotates about Y (free hinge), cable ignores
    - PRISMATIC (locked): anchor translates Y (locked), cable follows
    - PRISMATIC (free):   anchor translates X (free slider), cable slides
    - D6 (linear):       translate X with Z locked, cable slides along X
    - D6 (angular):      rotate about Y with Z locked, cable rotates freely
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
        stretch_damping = 0.0

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 0.8

        JointDofConfig = newton.ModelBuilder.JointDofConfig

        sphere_radius = 0.18
        z0 = 4.0
        y_spacing = 1.0
        y0 = -3.5

        configs = [
            {
                "label": "ball",
                "joint": "ball",
                "motion": _MOTION_ROTATE_Y,
                "has_free_lin": 0,
                "locked_lin": None,
                "has_free_ang": 0,
                "locked_ang": None,
            },
            {
                "label": "fixed",
                "joint": "fixed",
                "motion": _MOTION_ROTATE_Y,
                "has_free_lin": 0,
                "locked_lin": None,
                "has_free_ang": 0,
                "locked_ang": None,
            },
            {
                "label": "rev_locked",
                "joint": "revolute",
                "motion": _MOTION_ROTATE_X,
                "has_free_lin": 0,
                "locked_lin": None,
                "has_free_ang": 1,
                "locked_ang": (0, 0, 1),
            },
            {
                "label": "rev_free",
                "joint": "revolute",
                "motion": _MOTION_ROTATE_Y,
                "has_free_lin": 0,
                "locked_lin": None,
                "has_free_ang": 0,
                "locked_ang": None,
            },
            {
                "label": "prism_locked",
                "joint": "prismatic",
                "motion": _MOTION_TRANSLATE_Y,
                "has_free_lin": 0,
                "locked_lin": None,
                "has_free_ang": 0,
                "locked_ang": None,
            },
            {
                "label": "prism_free",
                "joint": "prismatic",
                "motion": _MOTION_TRANSLATE_X,
                "has_free_lin": 1,
                "locked_lin": (0, 0, 1),
                "has_free_ang": 0,
                "locked_ang": None,
            },
            {
                "label": "d6_lin",
                "joint": "d6",
                "motion": _MOTION_TRANSLATE_X,
                "has_free_lin": 1,
                "locked_lin": (0, 0, 1),
                "has_free_ang": 0,
                "locked_ang": None,
                "lin_axes": [JointDofConfig(axis=(1, 0, 0)), JointDofConfig(axis=(0, 1, 0))],
                "ang_axes": [],
            },
            {
                "label": "d6_ang",
                "joint": "d6",
                "motion": _MOTION_ROTATE_Y,
                "has_free_lin": 0,
                "locked_lin": None,
                "has_free_ang": 1,
                "locked_ang": (0, 0, 1),
                "lin_axes": [],
                "ang_axes": [JointDofConfig(axis=(1, 0, 0)), JointDofConfig(axis=(0, 1, 0))],
            },
        ]

        self.anchor_bodies: list[int] = []
        anchor_base_pos: list[wp.vec3] = []
        anchor_phase: list[float] = []
        anchor_motion_type: list[int] = []

        jt_parent_ids: list[int] = []
        jt_child_ids: list[int] = []
        jt_parent_local: list[wp.vec3] = []
        jt_child_local: list[wp.vec3] = []
        jt_parent_frame_q: list[wp.quat] = []
        jt_child_frame_q: list[wp.quat] = []
        jt_locked_lin: list[wp.vec3] = []
        jt_locked_ang: list[wp.vec3] = []
        jt_has_free_lin: list[int] = []
        jt_ang_error_mode: list[int] = []
        jt_free_rot_mode: list[int] = []
        jt_free_rot_axis: list[wp.vec3] = []

        self._ball_indices: list[int] = []
        self._fixed_indices: list[int] = []
        self._revolute_indices: list[int] = []
        self._revolute_free_indices: list[int] = []
        self._prismatic_indices: list[int] = []
        self._prismatic_free_indices: list[int] = []
        self._d6_indices: list[int] = []
        self._d6_free_lin_indices: list[int] = []
        self._d6_free_ang_indices: list[int] = []

        for idx, cfg in enumerate(configs):
            y = y0 + idx * y_spacing
            x = 0.0
            z = z0

            body = builder.add_link(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()), mass=0.0)
            builder.add_shape_sphere(body=body, radius=sphere_radius, label=f"drv_{cfg['label']}")
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

            parent_anchor_local = wp.vec3(0.0, 0.0, -sphere_radius)
            anchor_world = wp.vec3(x, y, z - sphere_radius)

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
                label=f"cable_{cfg['label']}",
                wrap_in_articulation=False,
            )

            child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
            child_fq = wp.quat_identity()
            jtype = cfg["joint"]

            if jtype == "ball":
                parent_fq = wp.quat_identity()
                j = builder.add_joint_ball(
                    parent=body,
                    child=rod_bodies[0],
                    parent_xform=wp.transform(parent_anchor_local, parent_fq),
                    child_xform=wp.transform(child_anchor_local, child_fq),
                    label=f"attach_{cfg['label']}",
                )
            elif jtype == "fixed":
                parent_fq = rod_quats[0]
                j = builder.add_joint_fixed(
                    parent=body,
                    child=rod_bodies[0],
                    parent_xform=wp.transform(parent_anchor_local, parent_fq),
                    child_xform=wp.transform(child_anchor_local, child_fq),
                    label=f"attach_{cfg['label']}",
                )
            elif jtype == "revolute":
                parent_fq = rod_quats[0]
                j = builder.add_joint_revolute(
                    parent=body,
                    child=rod_bodies[0],
                    parent_xform=wp.transform(parent_anchor_local, parent_fq),
                    child_xform=wp.transform(child_anchor_local, child_fq),
                    axis=(0.0, 1.0, 0.0),
                    label=f"attach_{cfg['label']}",
                )
            elif jtype == "prismatic":
                parent_fq = rod_quats[0]
                j = builder.add_joint_prismatic(
                    parent=body,
                    child=rod_bodies[0],
                    parent_xform=wp.transform(parent_anchor_local, parent_fq),
                    child_xform=wp.transform(child_anchor_local, child_fq),
                    axis=(1.0, 0.0, 0.0),
                    label=f"attach_{cfg['label']}",
                )
            else:
                parent_fq = rod_quats[0]
                j = builder.add_joint_d6(
                    parent=body,
                    child=rod_bodies[0],
                    parent_xform=wp.transform(parent_anchor_local, parent_fq),
                    child_xform=wp.transform(child_anchor_local, child_fq),
                    linear_axes=cfg["lin_axes"],
                    angular_axes=cfg["ang_axes"],
                    label=f"attach_{cfg['label']}",
                )

            builder.add_articulation([*rod_joints, j])

            jt_parent_ids.append(int(body))
            jt_child_ids.append(int(rod_bodies[0]))
            jt_parent_local.append(parent_anchor_local)
            jt_child_local.append(child_anchor_local)
            jt_parent_frame_q.append(parent_fq)
            jt_child_frame_q.append(child_fq)
            ll = cfg["locked_lin"]
            la = cfg["locked_ang"]
            jt_locked_lin.append(wp.vec3(*ll) if ll else wp.vec3(0.0, 0.0, 0.0))
            jt_locked_ang.append(wp.vec3(*la) if la else wp.vec3(0.0, 0.0, 0.0))
            jt_has_free_lin.append(cfg["has_free_lin"])
            ang_error_mode = 0
            free_rot_mode = 0
            free_rot_axis = wp.vec3(0.0, 0.0, 0.0)

            if jtype == "ball":
                self._ball_indices.append(idx)
                free_rot_mode = 1
            elif jtype == "fixed":
                self._fixed_indices.append(idx)
            elif jtype == "revolute":
                self._revolute_indices.append(idx)
                ang_error_mode = 2
                free_rot_axis = wp.vec3(0.0, 1.0, 0.0)
                if cfg["label"] == "rev_free":
                    self._revolute_free_indices.append(idx)
                    free_rot_mode = 2
            elif jtype == "prismatic":
                self._prismatic_indices.append(idx)
                if cfg["label"] == "prism_free":
                    self._prismatic_free_indices.append(idx)
            elif jtype == "d6":
                self._d6_indices.append(idx)
                if cfg["has_free_ang"] == 1:
                    ang_error_mode = 1
                    free_rot_mode = 2
                    free_rot_axis = wp.vec3(0.0, 1.0, 0.0)
                    self._d6_free_ang_indices.append(idx)
                if cfg["has_free_lin"] == 1:
                    self._d6_free_lin_indices.append(idx)

            jt_ang_error_mode.append(ang_error_mode)
            jt_free_rot_mode.append(free_rot_mode)
            jt_free_rot_axis.append(free_rot_axis)

            self.anchor_bodies.append(body)
            anchor_base_pos.append(wp.vec3(x, y, z))
            anchor_phase.append(0.0)
            anchor_motion_type.append(cfg["motion"])

        builder.add_ground_plane()
        builder.color()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
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
        self.anchor_motion_type_wp = wp.array(anchor_motion_type, dtype=wp.int32, device=self.device)
        self.sim_time_array = wp.zeros(1, dtype=float, device=self.device)

        self._jt_parent_ids = wp.array(jt_parent_ids, dtype=wp.int32, device=self.device)
        self._jt_child_ids = wp.array(jt_child_ids, dtype=wp.int32, device=self.device)
        self._jt_parent_local = wp.array(jt_parent_local, dtype=wp.vec3, device=self.device)
        self._jt_child_local = wp.array(jt_child_local, dtype=wp.vec3, device=self.device)
        self._jt_parent_frame_q = wp.array(jt_parent_frame_q, dtype=wp.quat, device=self.device)
        self._jt_child_frame_q = wp.array(jt_child_frame_q, dtype=wp.quat, device=self.device)
        self._jt_locked_lin = wp.array(jt_locked_lin, dtype=wp.vec3, device=self.device)
        self._jt_locked_ang = wp.array(jt_locked_ang, dtype=wp.vec3, device=self.device)
        self._jt_has_free_lin = wp.array(jt_has_free_lin, dtype=wp.int32, device=self.device)
        self._jt_ang_error_mode = wp.array(jt_ang_error_mode, dtype=wp.int32, device=self.device)
        self._jt_free_rot_mode = wp.array(jt_free_rot_mode, dtype=wp.int32, device=self.device)
        self._jt_free_rot_axis = wp.array(jt_free_rot_axis, dtype=wp.vec3, device=self.device)

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
                    self.anchor_motion_type_wp,
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

    def _compute_errors(self):
        n = self._jt_parent_ids.shape[0]
        err_pos = wp.zeros(n, dtype=float, device=self.device)
        err_ang = wp.zeros(n, dtype=float, device=self.device)
        d_along = wp.zeros(n, dtype=float, device=self.device)
        rot_along = wp.zeros(n, dtype=float, device=self.device)
        wp.launch(
            kernel=compute_joint_error,
            dim=n,
            inputs=[
                self._jt_parent_ids,
                self._jt_child_ids,
                self._jt_parent_local,
                self._jt_child_local,
                self._jt_parent_frame_q,
                self._jt_child_frame_q,
                self._jt_locked_lin,
                self._jt_locked_ang,
                self._jt_has_free_lin,
                self._jt_ang_error_mode,
                self._jt_free_rot_mode,
                self._jt_free_rot_axis,
                self.state_0.body_q,
                err_pos,
                err_ang,
                d_along,
                rot_along,
            ],
            device=self.device,
        )
        return err_pos.numpy(), err_ang.numpy(), d_along.numpy(), rot_along.numpy()

    def test_final(self):
        err_pos_np, err_ang_np, d_along_np, rot_along_np = self._compute_errors()

        tol_pos = 2.0e-3
        tol_ang = 1.0e-2
        if self._ball_indices:
            ball_pos_max = float(np.max(err_pos_np[self._ball_indices]))
            if ball_pos_max > tol_pos:
                raise AssertionError(f"BALL joint anchor error too large: max={ball_pos_max:.6g} > tol={tol_pos}")

            ball_rot_max = float(np.max(rot_along_np[self._ball_indices]))
            min_rotation_tol = 0.1
            if ball_rot_max < min_rotation_tol:
                raise AssertionError(
                    f"BALL joint angular freedom not exercised: "
                    f"max(relative_angle)={ball_rot_max:.6g} < {min_rotation_tol}"
                )

        if self._fixed_indices:
            fixed_pos_max = float(np.max(err_pos_np[self._fixed_indices]))
            fixed_ang_max = float(np.max(err_ang_np[self._fixed_indices]))
            if fixed_pos_max > tol_pos or fixed_ang_max > tol_ang:
                raise AssertionError(
                    "FIXED joint error too large: "
                    f"pos_max={fixed_pos_max:.6g} (tol={tol_pos}), "
                    f"ang_max={fixed_ang_max:.6g} (tol={tol_ang})"
                )

        if self._revolute_indices:
            rev_pos_max = float(np.max(err_pos_np[self._revolute_indices]))
            rev_ang_max = float(np.max(err_ang_np[self._revolute_indices]))
            if rev_pos_max > tol_pos:
                raise AssertionError(f"REVOLUTE joint position error too large: max={rev_pos_max:.6g} (tol={tol_pos})")
            if rev_ang_max > tol_ang:
                raise AssertionError(
                    f"REVOLUTE joint perpendicular angular error too large: max={rev_ang_max:.6g} (tol={tol_ang})"
                )

        if self._revolute_free_indices:
            rev_free_max = float(np.max(np.abs(rot_along_np[self._revolute_free_indices])))
            min_kappa_free_tol = 0.1
            if rev_free_max < min_kappa_free_tol:
                raise AssertionError(
                    f"REVOLUTE joint free-axis rotation too small in free set: "
                    f"max(|kappa_free|)={rev_free_max:.6g} < {min_kappa_free_tol}"
                )

        if self._prismatic_indices:
            prism_pos_max = float(np.max(err_pos_np[self._prismatic_indices]))
            prism_ang_max = float(np.max(err_ang_np[self._prismatic_indices]))
            if prism_pos_max > tol_pos:
                raise AssertionError(
                    f"PRISMATIC joint perpendicular position error too large: max={prism_pos_max:.6g} (tol={tol_pos})"
                )
            if prism_ang_max > tol_ang:
                raise AssertionError(
                    f"PRISMATIC joint angular error too large: max={prism_ang_max:.6g} (tol={tol_ang})"
                )

        if self._prismatic_free_indices:
            prism_free_max = float(np.max(np.abs(d_along_np[self._prismatic_free_indices])))
            min_slide_tol = 0.01
            if prism_free_max < min_slide_tol:
                raise AssertionError(
                    f"PRISMATIC joint free-axis sliding too small in free set: "
                    f"max(|C_along|)={prism_free_max:.6g} < {min_slide_tol}"
                )

        if self._d6_indices:
            d6_pos_max = float(np.max(err_pos_np[self._d6_indices]))
            d6_ang_max = float(np.max(err_ang_np[self._d6_indices]))
            if d6_pos_max > tol_pos:
                raise AssertionError(f"D6 joint position error too large: max={d6_pos_max:.6g} (tol={tol_pos})")
            if d6_ang_max > tol_ang:
                raise AssertionError(f"D6 joint angular error too large: max={d6_ang_max:.6g} (tol={tol_ang})")

        if self._d6_free_lin_indices:
            d6_free_lin_max = float(np.max(np.abs(d_along_np[self._d6_free_lin_indices])))
            if d6_free_lin_max < 0.04:
                raise AssertionError(
                    f"D6 free linear displacement too small: max(|d_along|)={d6_free_lin_max:.6g} < 0.04 m"
                )

        if self._d6_free_ang_indices:
            d6_free_ang_max = float(np.max(np.abs(rot_along_np[self._d6_free_ang_indices])))
            if d6_free_ang_max < 0.04:
                raise AssertionError(
                    f"D6 free angular rotation too small: max(|rot_along|)={d6_free_ang_max:.6g} < 0.04 rad"
                )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
