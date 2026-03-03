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
# Example Cable Prismatic Joints
#
# Visual test for VBD PRISMATIC joints with kinematic anchors:
# - Create multiple kinematic anchor bodies (sphere, capsule, box, bear mesh)
# - Attach a cable (rod) to each anchor via a PRISMATIC joint (axis = X)
# - Two sets of drivers side by side:
#     Left column  (mode 0): translate in X (along free slider axis) — cable slides freely
#     Right column (mode 1): translate in Y (perpendicular, locked) — cable follows
# - Both sets use translation to showcase the prismatic's unique translational DOF.
#   (Rotation-based driving is not used because the VBD prismatic solver
#    requires small parent rotations for accurate perpendicular constraint.)
#
###########################################################################

import math

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd


@wp.kernel
def compute_prismatic_joint_error(
    parent_ids: wp.array(dtype=wp.int32),
    child_ids: wp.array(dtype=wp.int32),
    parent_local: wp.array(dtype=wp.vec3),
    child_local: wp.array(dtype=wp.vec3),
    parent_frame_q: wp.array(dtype=wp.quat),
    child_frame_q: wp.array(dtype=wp.quat),
    joint_axis_local: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    out_err_pos_perp: wp.array(dtype=float),
    out_err_ang: wp.array(dtype=float),
    out_c_along_axis: wp.array(dtype=float),
):
    """Test-only: compute PRISMATIC joint error (perpendicular position + full angle).

    For each i:
      - Perpendicular position error: ||C_perp|| where C_perp is the component of
        the anchor difference perpendicular to the joint axis in world space.
      - Angular error: full rotation angle between joint frames.
      - Along-axis displacement: |C_along| — expected to be nonzero for set 0
        drivers that slide the cable along the free axis.
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

    # Project out the free-axis (sliding) component
    a_local = wp.normalize(joint_axis_local[i])
    a_world = wp.normalize(wp.quat_rotate(wp.mul(qp, parent_frame_q[i]), a_local))
    c_along = wp.dot(C, a_world)
    C_perp = C - c_along * a_world
    out_err_pos_perp[i] = wp.length(C_perp)
    out_c_along_axis[i] = c_along

    # Full angular error between joint frames
    qpf = wp.mul(qp, parent_frame_q[i])
    qcf = wp.mul(qc, child_frame_q[i])
    dq = wp.mul(wp.quat_inverse(qpf), qcf)
    dq = wp.normalize(dq)
    w = wp.clamp(wp.abs(dq[3]), 0.0, 1.0)
    out_err_ang[i] = 2.0 * wp.acos(w)


@wp.kernel
def advance_time(t: wp.array(dtype=float), dt: float):
    t[0] = t[0] + dt


@wp.kernel
def move_kinematic_anchors(
    t: wp.array(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
    base_pos: wp.array(dtype=wp.vec3),
    phase: wp.array(dtype=float),
    mode: wp.array(dtype=wp.int32),
    ramp_time: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Kinematically animate anchor poses.  Each anchor has a mode:

    mode 0 -- translate in X (along free slider axis): cable slides freely,
              doesn't follow. Showcases the prismatic's unique translational DOF.
    mode 1 -- translate in Y (perpendicular, locked): cable must follow sideways.
              Shows the constraint locks perpendicular directions.
    """
    i = wp.tid()
    b = anchor_ids[i]

    t0 = t[0]
    u = wp.clamp(t0 / ramp_time, 0.0, 1.0)
    ramp = u * u * (3.0 - 2.0 * u)  # smoothstep

    ti = t0 + phase[i]
    p0 = base_pos[i]
    m = mode[i]

    w = 1.5

    if m == 0:
        # Translate in X (along free slider axis) — cable doesn't follow
        dx = (ramp * 0.5) * wp.sin(w * ti)
        pos = wp.vec3(p0[0] + dx, p0[1], p0[2])
    else:
        # Translate in Y (perpendicular, locked) — cable follows sideways
        dy = (ramp * 0.5) * wp.sin(w * ti)
        pos = wp.vec3(p0[0], p0[1] + dy, p0[2])

    body_q[b] = wp.transform(pos, wp.quat_identity())
    body_qd[b] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _auto_scale(mesh, target_diameter: float) -> float:
    verts = mesh.vertices
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    max_dim = float(np.max(bb_max - bb_min))
    return (target_diameter / max_dim) if max_dim > 1.0e-8 else 1.0


class Example:
    """Visual test for VBD PRISMATIC joints with kinematic anchors.

    High-level structure:
    - Build several kinematic "driver" bodies (sphere/capsule/box/bear mesh).
    - For each driver, create a straight cable (rod) hanging down in world -Z.
    - Attach the first rod segment to the driver using a PRISMATIC joint (slider axis X).
    - Two side-by-side sets driven by pure translation:
        Left  (mode 0) — translate X (FREE axis):  cable slides, doesn't follow.
        Right (mode 1) — translate Y (LOCKED):     cable follows sideways.
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

        # Default gravity (Z-up). Anchors are kinematic and moved explicitly.
        builder = newton.ModelBuilder()

        # Contacts.
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e1
        builder.default_shape_cfg.mu = 0.8

        # Load meshes for variety.
        bear_stage = Usd.Stage.Open(newton.examples.get_asset("bear.usd"))
        bear_mesh = newton.usd.get_mesh(bear_stage.GetPrimAtPath("/root/bear/bear"))

        target_d = 0.35
        bear_scale = _auto_scale(bear_mesh, target_d)

        # For the bear, shift the mesh so the body origin is approximately at mid-height.
        bear_verts = bear_mesh.vertices
        bear_bb_min = bear_verts.min(axis=0)
        bear_bb_max = bear_verts.max(axis=0)
        bear_center_z = 0.5 * float(bear_bb_min[2] + bear_bb_max[2])
        bear_mesh_pz = -bear_center_z * bear_scale

        # Driver bodies (kinematic) + attached cables.
        driver_specs = [
            ("sphere", None),
            ("capsule", None),
            ("box", None),
            ("bear", (bear_mesh, bear_scale)),
        ]

        # Anchors are kinematic and moved via a kernel.
        self.anchor_bodies = []
        anchor_base_pos: list[wp.vec3] = []
        anchor_phase: list[float] = []
        anchor_mode: list[int] = []

        # Test mode: record PRISMATIC joint anchor definitions so we can verify constraint satisfaction.
        prismatic_parent_ids: list[int] = []
        prismatic_child_ids: list[int] = []
        prismatic_parent_local: list[wp.vec3] = []
        prismatic_child_local: list[wp.vec3] = []
        prismatic_parent_frame_q: list[wp.quat] = []
        prismatic_child_frame_q: list[wp.quat] = []
        prismatic_axis_local: list[wp.vec3] = []

        # Spread the anchor+cable sets along Y (not X).
        y0 = -1.2
        dy = 0.6

        # Anchor height (raise this to give the cables more room to hang)
        z0 = 4.0

        # Two sets: free-slide (translate X) and constrained-follow (translate Z)
        sets = [
            ("free_slide", -1.0, 0),  # (label, x_offset, mode)
            ("locked_y", 1.0, 1),
        ]

        # Track which joint indices belong to set 0 (free slide) for the extra test.
        self._set0_joint_indices: list[int] = []
        joint_counter = 0

        for set_idx, (_label, x_offset, m) in enumerate(sets):
            for i, (kind, mesh_info) in enumerate(driver_specs):
                x = x_offset
                y = y0 + dy * i
                z = z0
                # Small per-shape vertical offsets (world -Z) for visual variety / clearance.
                if kind == "sphere":
                    z = z0 - 0.05
                elif kind == "bear":
                    z = z0 - 0.10

                # Anchor body: kinematic (will be driven by kernel).
                body = builder.add_link(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()), mass=0.0)

                # Add a visible collision shape + choose an anchor offset in local frame
                # (initialize per-shape parameters for clarity/type-checkers)
                r = 0.0
                hh = 0.0
                hx = 0.0
                hy = 0.0
                hz = 0.0
                attach_offset = 0.18
                key_prefix = f"{_label}_{kind}_{i}"
                if kind == "sphere":
                    r = 0.18
                    builder.add_shape_sphere(body=body, radius=r, label=f"drv_{key_prefix}")
                    attach_offset = r
                elif kind == "capsule":
                    r = 0.12
                    hh = 0.18
                    # Lay the capsule down horizontally (capsule axis is +Z in shape-local frame).
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
                else:
                    mesh, scale = mesh_info
                    builder.add_shape_mesh(
                        body=body,
                        mesh=mesh,
                        scale=(scale, scale, scale),
                        xform=wp.transform(p=wp.vec3(0.0, 0.0, bear_mesh_pz), q=wp.quat_identity()),
                        label=f"drv_{key_prefix}",
                    )

                # Make the driver strictly kinematic (override any mass/inertia contributed by shapes).
                builder.body_mass[body] = 0.0
                builder.body_inv_mass[body] = 0.0
                builder.body_inertia[body] = wp.mat33(0.0)
                builder.body_inv_inertia[body] = wp.mat33(0.0)

                # Cable root anchor directly below the driver's center of mass.
                parent_anchor_local = wp.vec3(0.0, 0.0, -attach_offset)
                anchor_world = wp.vec3(x, y, z - attach_offset)

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
                    # Build one articulation including both the prismatic joint and all rod joints.
                    wrap_in_articulation=False,
                )

                # Attach cable start point to driver with a prismatic joint.
                # Use the cable's initial segment orientation in the parent frame so the
                # joint rest pose matches the rod's natural hanging direction.
                child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
                joint_axis = (1.0, 0.0, 0.0)
                parent_frame_q = rod_quats[0]
                child_frame_q = wp.quat_identity()
                j_prismatic = builder.add_joint_prismatic(
                    parent=body,
                    child=rod_bodies[0],
                    parent_xform=wp.transform(parent_anchor_local, parent_frame_q),
                    child_xform=wp.transform(child_anchor_local, child_frame_q),
                    axis=joint_axis,
                    label=f"attach_{key_prefix}",
                )
                # Put all joints (rod cable joints + the prismatic attachment) into one articulation.
                builder.add_articulation([*rod_joints, j_prismatic])

                # Record joint definitions for testing.
                prismatic_parent_ids.append(int(body))
                prismatic_child_ids.append(int(rod_bodies[0]))
                prismatic_parent_local.append(parent_anchor_local)
                prismatic_child_local.append(child_anchor_local)
                prismatic_parent_frame_q.append(parent_frame_q)
                prismatic_child_frame_q.append(child_frame_q)
                prismatic_axis_local.append(wp.vec3(*joint_axis))

                if set_idx == 0:
                    self._set0_joint_indices.append(joint_counter)
                joint_counter += 1

                self.anchor_bodies.append(body)
                anchor_base_pos.append(wp.vec3(x, y, z))
                # Sync motion within each set for easy visual comparison.
                anchor_phase.append(0.0)
                anchor_mode.append(int(m))

        builder.add_ground_plane()
        builder.color()
        self.model = builder.finalize()

        # Stiffen prismatic constraint caps (non-cable joints) so the attachment behaves near-hard.
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

        # Device-side kinematic anchor motion buffers (graph-capture friendly)
        self.device = self.solver.device
        self.anchor_bodies_wp = wp.array(self.anchor_bodies, dtype=wp.int32, device=self.device)
        self.anchor_base_pos_wp = wp.array(anchor_base_pos, dtype=wp.vec3, device=self.device)
        self.anchor_phase_wp = wp.array(anchor_phase, dtype=float, device=self.device)
        self.anchor_mode_wp = wp.array(anchor_mode, dtype=wp.int32, device=self.device)
        self.sim_time_array = wp.zeros(1, dtype=float, device=self.device)

        # Test buffers (prismatic joint constraint satisfaction)
        self._prismatic_parent_ids = wp.array(prismatic_parent_ids, dtype=wp.int32, device=self.device)
        self._prismatic_child_ids = wp.array(prismatic_child_ids, dtype=wp.int32, device=self.device)
        self._prismatic_parent_local = wp.array(prismatic_parent_local, dtype=wp.vec3, device=self.device)
        self._prismatic_child_local = wp.array(prismatic_child_local, dtype=wp.vec3, device=self.device)
        self._prismatic_parent_frame_q = wp.array(prismatic_parent_frame_q, dtype=wp.quat, device=self.device)
        self._prismatic_child_frame_q = wp.array(prismatic_child_frame_q, dtype=wp.quat, device=self.device)
        self._prismatic_axis_local = wp.array(prismatic_axis_local, dtype=wp.vec3, device=self.device)

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

            # Kinematic anchor motion (translation only, identity rotation)
            wp.launch(
                kernel=move_kinematic_anchors,
                dim=self.anchor_bodies_wp.shape[0],
                inputs=[
                    self.sim_time_array,
                    self.anchor_bodies_wp,
                    self.anchor_base_pos_wp,
                    self.anchor_phase_wp,
                    self.anchor_mode_wp,
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

            # Advance time for anchor motion (on device)
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
        """Launch the error kernel and return (err_pos_np, err_ang_np, c_along_np)."""
        n = self._prismatic_parent_ids.shape[0]
        err_pos_perp = wp.zeros(n, dtype=float, device=self.device)
        err_ang = wp.zeros(n, dtype=float, device=self.device)
        c_along = wp.zeros(n, dtype=float, device=self.device)
        wp.launch(
            kernel=compute_prismatic_joint_error,
            dim=n,
            inputs=[
                self._prismatic_parent_ids,
                self._prismatic_child_ids,
                self._prismatic_parent_local,
                self._prismatic_child_local,
                self._prismatic_parent_frame_q,
                self._prismatic_child_frame_q,
                self._prismatic_axis_local,
                self.state_0.body_q,
                err_pos_perp,
                err_ang,
                c_along,
            ],
            device=self.device,
        )
        return err_pos_perp.numpy(), err_ang.numpy(), c_along.numpy()

    def test_final(self):
        if self._prismatic_parent_ids.shape[0] == 0:
            return

        err_pos_np, err_ang_np, c_along_np = self._compute_errors()
        err_pos_max = float(np.max(err_pos_np))
        err_ang_max = float(np.max(err_ang_np))

        # 1. Perpendicular position error: all joints
        tol_pos = 2.0e-3
        if err_pos_max > tol_pos:
            raise AssertionError(
                f"PRISMATIC joint perpendicular position error too large: max={err_pos_max:.6g} (tol={tol_pos})"
            )

        # 2. Angular error: all joints
        tol_ang = 1.0e-2
        if err_ang_max > tol_ang:
            raise AssertionError(f"PRISMATIC joint angular error too large: max={err_ang_max:.6g} (tol={tol_ang})")

        # 3. Free-axis sliding exercised: set 0 only (proves slider is free, not secretly fixed)
        if self._set0_joint_indices:
            set0_c_along = np.abs(c_along_np[self._set0_joint_indices])
            max_c_along = float(np.max(set0_c_along))
            min_slide_tol = 0.01  # at least 1cm of sliding
            if max_c_along < min_slide_tol:
                raise AssertionError(
                    f"PRISMATIC joint free-axis sliding too small in set 0: "
                    f"max(|C_along|)={max_c_along:.6g} < {min_slide_tol}"
                )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
