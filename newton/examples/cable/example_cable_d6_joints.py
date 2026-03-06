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
# Example Cable D6 Joints
#
# Visual test for VBD D6 joints with kinematic anchors:
# - Create 5 sphere anchors, each with a cable (rod) attached via a D6 joint
# - Translation group (translate X): lock_x, lock_z, lock_xyz
# - Rotation group (rotate about Y): lock_ang_y, lock_ang_z
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def compute_d6_joint_error(
    parent_ids: wp.array(dtype=wp.int32),
    child_ids: wp.array(dtype=wp.int32),
    parent_local: wp.array(dtype=wp.vec3),
    child_local: wp.array(dtype=wp.vec3),
    parent_frame_q: wp.array(dtype=wp.quat),
    child_frame_q: wp.array(dtype=wp.quat),
    locked_lin_axis: wp.array(dtype=wp.vec3),
    locked_ang_axis: wp.array(dtype=wp.vec3),
    has_free_lin: wp.array(dtype=wp.int32),
    has_free_ang: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    out_err_pos: wp.array(dtype=float),
    out_err_ang: wp.array(dtype=float),
    out_d_along: wp.array(dtype=float),
    out_rot_along_free: wp.array(dtype=float),
):
    """Test-only: compute D6 joint constraint errors.

    For joints with free linear axes (2 free, 1 locked), measures position error
    along the single locked axis and displacement along X (driven direction).
    For fully-locked linear DOFs, measures full position coincidence.
    Same logic applies for angular DOFs: error about the locked rotation axis
    and free rotation about Y (driven rotation axis).
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

    # Angular difference between joint frames
    qcf = wp.mul(qc, child_frame_q[i])
    dq = wp.mul(wp.quat_inverse(qpf), qcf)
    dq = wp.normalize(dq)
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])

    axis_angle, angle = wp.quat_to_axis_angle(dq)
    rot_vec = axis_angle * angle

    if has_free_ang[i] == 1:
        aa = wp.normalize(locked_ang_axis[i])
        out_err_ang[i] = wp.abs(wp.dot(rot_vec, aa))
        out_rot_along_free[i] = wp.dot(rot_vec, wp.vec3(0.0, 1.0, 0.0))
    else:
        out_err_ang[i] = wp.length(rot_vec)
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

    motion_type 0 -- translate X only.
    motion_type 1 -- rotate about Y only.
    """
    i = wp.tid()
    b = anchor_ids[i]

    t0 = t[0]
    u = wp.clamp(t0 / ramp_time, 0.0, 1.0)
    ramp = u * u * (3.0 - 2.0 * u)  # smoothstep

    ti = t0 + phase[i]
    p0 = base_pos[i]
    m = motion_type[i]

    w = 1.5

    if m == 0:
        dx = (ramp * 0.4) * wp.sin(w * ti)
        pos = wp.vec3(p0[0] + dx, p0[1], p0[2])
        q = wp.quat_identity()
    else:
        ang_y = (ramp * 0.8) * wp.sin(w * ti)
        pos = p0
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ang_y)

    body_q[b] = wp.transform(pos, q)
    body_qd[b] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class Example:
    """Visual test for VBD D6 joints with kinematic anchors.

    Five sphere anchors in a single column, each with a cable hanging below:

    Translation group (translate X only):
      lock_x   -- X locked, cable follows X translation
      lock_z   -- Z locked, cable free in X
      lock_xyz -- all locked, cable follows everything

    Rotation group (rotate about Y only):
      lock_ang_y -- angular Y locked, cable follows Y rotation
      lock_ang_z -- angular Z locked, cable free about Y
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
        stretch_damping = 0.0

        builder = newton.ModelBuilder()

        # Contacts.
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 0.8

        JointDofConfig = newton.ModelBuilder.JointDofConfig

        sphere_radius = 0.18
        z0 = 4.0

        # Five D6 configurations spread along Y.
        # Axes listed in lin_axes / ang_axes are FREE; unlisted axes are LOCKED.
        # First 3: translate X only (motion_type=0), varying linear locking.
        # Last  2: rotate about Y only (motion_type=1), varying angular locking.
        y_spacing = 1.0
        y0 = -2.0 * y_spacing
        d6_configs = [
            # --- Translation group (motion_type=0): drive along X ---
            # locked_lin = the single locked cardinal axis (None if all locked)
            {
                "label": "lock_x",
                "y": y0 + 0 * y_spacing,
                "motion": 0,
                "lin_axes": [JointDofConfig(axis=(0, 1, 0)), JointDofConfig(axis=(0, 0, 1))],
                "ang_axes": [],
                "locked_lin": (1, 0, 0),
                "locked_ang": None,
            },
            {
                "label": "lock_z",
                "y": y0 + 1 * y_spacing,
                "motion": 0,
                "lin_axes": [JointDofConfig(axis=(1, 0, 0)), JointDofConfig(axis=(0, 1, 0))],
                "ang_axes": [],
                "locked_lin": (0, 0, 1),
                "locked_ang": None,
            },
            {
                "label": "lock_xyz",
                "y": y0 + 2 * y_spacing,
                "motion": 0,
                "lin_axes": [],
                "ang_axes": [],
                "locked_lin": None,
                "locked_ang": None,
            },
            # --- Rotation group (motion_type=1): rotate about Y ---
            # locked_ang = the single locked cardinal angular axis (None if all locked)
            {
                "label": "lock_ang_y",
                "y": y0 + 3 * y_spacing,
                "motion": 1,
                "lin_axes": [],
                "ang_axes": [JointDofConfig(axis=(1, 0, 0)), JointDofConfig(axis=(0, 0, 1))],
                "locked_lin": None,
                "locked_ang": (0, 1, 0),
            },
            {
                "label": "lock_ang_z",
                "y": y0 + 4 * y_spacing,
                "motion": 1,
                "lin_axes": [],
                "ang_axes": [JointDofConfig(axis=(1, 0, 0)), JointDofConfig(axis=(0, 1, 0))],
                "locked_lin": None,
                "locked_ang": (0, 0, 1),
            },
        ]

        # Kinematic anchor bookkeeping.
        self.anchor_bodies: list[int] = []
        anchor_base_pos: list[wp.vec3] = []
        anchor_phase: list[float] = []
        anchor_motion_type: list[int] = []

        # Per-joint test data.
        d6_parent_ids: list[int] = []
        d6_child_ids: list[int] = []
        d6_parent_local: list[wp.vec3] = []
        d6_child_local: list[wp.vec3] = []
        d6_parent_frame_q: list[wp.quat] = []
        d6_child_frame_q: list[wp.quat] = []
        d6_lin_axis_local: list[wp.vec3] = []
        d6_ang_axis_local: list[wp.vec3] = []
        d6_has_free_lin: list[int] = []
        d6_has_free_ang: list[int] = []

        self._free_lin_indices: list[int] = []
        self._free_ang_indices: list[int] = []

        for joint_idx, cfg in enumerate(d6_configs):
            x = 0.0
            y = cfg["y"]
            z = z0

            body = builder.add_link(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()), mass=0.0)
            builder.add_shape_sphere(body=body, radius=sphere_radius, label=f"drv_{cfg['label']}")

            # Make the driver strictly kinematic.
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

            # Attach cable to driver with a D6 joint.
            # Parent frame quaternion matches the rod's initial segment orientation
            # so the joint rest pose aligns with the cable's hanging direction.
            child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
            parent_frame_q = rod_quats[0]
            child_frame_q = wp.quat_identity()
            j_d6 = builder.add_joint_d6(
                parent=body,
                child=rod_bodies[0],
                parent_xform=wp.transform(parent_anchor_local, parent_frame_q),
                child_xform=wp.transform(child_anchor_local, child_frame_q),
                linear_axes=cfg["lin_axes"],
                angular_axes=cfg["ang_axes"],
                label=f"attach_{cfg['label']}",
            )
            builder.add_articulation([*rod_joints, j_d6])

            # Record per-joint data for error computation.
            d6_parent_ids.append(int(body))
            d6_child_ids.append(int(rod_bodies[0]))
            d6_parent_local.append(parent_anchor_local)
            d6_child_local.append(child_anchor_local)
            d6_parent_frame_q.append(parent_frame_q)
            d6_child_frame_q.append(child_frame_q)

            locked_lin = cfg["locked_lin"]
            locked_ang = cfg["locked_ang"]
            d6_lin_axis_local.append(wp.vec3(*locked_lin) if locked_lin else wp.vec3(0.0, 0.0, 0.0))
            d6_ang_axis_local.append(wp.vec3(*locked_ang) if locked_ang else wp.vec3(0.0, 0.0, 0.0))
            d6_has_free_lin.append(1 if locked_lin else 0)
            d6_has_free_ang.append(1 if locked_ang else 0)

            if locked_lin and locked_lin != (1, 0, 0):
                self._free_lin_indices.append(joint_idx)
            if locked_ang and locked_ang != (0, 1, 0):
                self._free_ang_indices.append(joint_idx)

            self.anchor_bodies.append(body)
            anchor_base_pos.append(wp.vec3(x, y, z))
            anchor_phase.append(0.0)
            anchor_motion_type.append(cfg["motion"])

        builder.add_ground_plane()
        builder.color()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        # Device-side kinematic anchor buffers.
        self.device = self.solver.device
        self.anchor_bodies_wp = wp.array(self.anchor_bodies, dtype=wp.int32, device=self.device)
        self.anchor_base_pos_wp = wp.array(anchor_base_pos, dtype=wp.vec3, device=self.device)
        self.anchor_phase_wp = wp.array(anchor_phase, dtype=float, device=self.device)
        self.anchor_motion_type_wp = wp.array(anchor_motion_type, dtype=wp.int32, device=self.device)
        self.sim_time_array = wp.zeros(1, dtype=float, device=self.device)

        # Device-side test buffers.
        self._d6_parent_ids = wp.array(d6_parent_ids, dtype=wp.int32, device=self.device)
        self._d6_child_ids = wp.array(d6_child_ids, dtype=wp.int32, device=self.device)
        self._d6_parent_local = wp.array(d6_parent_local, dtype=wp.vec3, device=self.device)
        self._d6_child_local = wp.array(d6_child_local, dtype=wp.vec3, device=self.device)
        self._d6_parent_frame_q = wp.array(d6_parent_frame_q, dtype=wp.quat, device=self.device)
        self._d6_child_frame_q = wp.array(d6_child_frame_q, dtype=wp.quat, device=self.device)
        self._d6_lin_axis_local = wp.array(d6_lin_axis_local, dtype=wp.vec3, device=self.device)
        self._d6_ang_axis_local = wp.array(d6_ang_axis_local, dtype=wp.vec3, device=self.device)
        self._d6_has_free_lin = wp.array(d6_has_free_lin, dtype=wp.int32, device=self.device)
        self._d6_has_free_ang = wp.array(d6_has_free_ang, dtype=wp.int32, device=self.device)

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
                    1.0,  # ramp_time [s]
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
        """Launch the error kernel and return (err_pos, err_ang, d_along, rot_along)."""
        n = self._d6_parent_ids.shape[0]
        err_pos = wp.zeros(n, dtype=float, device=self.device)
        err_ang = wp.zeros(n, dtype=float, device=self.device)
        d_along = wp.zeros(n, dtype=float, device=self.device)
        rot_along_free = wp.zeros(n, dtype=float, device=self.device)
        wp.launch(
            kernel=compute_d6_joint_error,
            dim=n,
            inputs=[
                self._d6_parent_ids,
                self._d6_child_ids,
                self._d6_parent_local,
                self._d6_child_local,
                self._d6_parent_frame_q,
                self._d6_child_frame_q,
                self._d6_lin_axis_local,
                self._d6_ang_axis_local,
                self._d6_has_free_lin,
                self._d6_has_free_ang,
                self.state_0.body_q,
                err_pos,
                err_ang,
                d_along,
                rot_along_free,
            ],
            device=self.device,
        )
        return err_pos.numpy(), err_ang.numpy(), d_along.numpy(), rot_along_free.numpy()

    def test_final(self):
        if self._d6_parent_ids.shape[0] == 0:
            return

        err_pos_np, err_ang_np, d_along_np, rot_along_np = self._compute_errors()

        # 1. Position error: along the locked axis for joints with free linear DOFs,
        #    full coincidence for fully-locked joints.
        tol_pos = 2.0e-3
        err_pos_max = float(np.max(err_pos_np))
        if err_pos_max > tol_pos:
            raise AssertionError(f"D6 joint position error too large: max={err_pos_max:.6g} (tol={tol_pos})")

        # 2. Angular error: about the locked axis for joints with free angular DOFs,
        #    full coincidence for fully-locked joints.
        tol_ang = 1.0e-2
        err_ang_max = float(np.max(err_ang_np))
        if err_ang_max > tol_ang:
            raise AssertionError(f"D6 joint angular error too large: max={err_ang_max:.6g} (tol={tol_ang})")

        # 3. Free linear DOF exercised (configs where driven X is a free axis).
        if self._free_lin_indices:
            free_d = np.abs(d_along_np[self._free_lin_indices])
            max_d = float(np.max(free_d))
            if max_d < 0.1:
                raise AssertionError(f"D6 free linear displacement too small: max(|d_along|)={max_d:.6g} < 0.1 m")

        # 4. Free angular DOF exercised (configs where driven Y rotation is free).
        if self._free_ang_indices:
            free_rot = np.abs(rot_along_np[self._free_ang_indices])
            max_rot = float(np.max(free_rot))
            if max_rot < 0.1:
                raise AssertionError(f"D6 free angular rotation too small: max(|rot_along|)={max_rot:.6g} < 0.1 rad")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
