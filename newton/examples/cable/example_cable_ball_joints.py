# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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
# Example Cable Ball Joints
#
# Visual test for VBD BALL joints with kinematic anchors:
# - Create multiple kinematic anchor bodies (sphere, capsule, box, bear mesh)
# - Attach a cable (rod) to each anchor via a BALL joint
# - Drive anchors kinematically (rotation) and verify cables follow
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
def compute_ball_joint_error(
    parent_ids: wp.array(dtype=wp.int32),
    child_ids: wp.array(dtype=wp.int32),
    parent_local: wp.array(dtype=wp.vec3),
    child_local: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    out_err_pos: wp.array(dtype=float),
    out_rel_angle: wp.array(dtype=float),
):
    """Test-only: compute BALL joint error (position coincidence + relative rotation).

    For each i:
      - Position error: world-space distance between parent and child anchor points.
      - Relative rotation angle: angle between parent and child body orientations.
        Expected to be nonzero because ball joints free all rotation — the cable
        should NOT follow the parent's rotation.
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

    # Relative rotation between parent and child bodies.
    # Ball joint uses identity frame quaternions, so body quats are the joint frames.
    dq = wp.mul(wp.quat_inverse(qp), qc)
    dq = wp.normalize(dq)
    w = wp.clamp(wp.abs(dq[3]), 0.0, 1.0)
    out_rel_angle[i] = 2.0 * wp.acos(w)


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

    mode 0 -- rotate about X: cable ignores (all rotation free in ball joint).
    mode 1 -- rotate about Y: cable ignores (all rotation free in ball joint).
    Same motion as fixed/revolute examples for cross-example comparison.
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
        # Rotate about X — cable ignores (ball frees all rotation)
        ang = (ramp * 0.8) * wp.sin(w * ti)
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), ang)
    else:
        # Rotate about Y — cable ignores (ball frees all rotation)
        ang = (ramp * 1.6) * wp.sin(w * ti + 0.7)
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ang)

    body_q[b] = wp.transform(p0, q)
    body_qd[b] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _auto_scale(mesh, target_diameter: float) -> float:
    verts = mesh.vertices
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    max_dim = float(np.max(bb_max - bb_min))
    return (target_diameter / max_dim) if max_dim > 1.0e-8 else 1.0


class Example:
    """Visual test for VBD BALL joints with kinematic anchors.

    High-level structure:
    - Build several kinematic "driver" bodies (sphere/capsule/box/bear mesh).
    - For each driver, create a straight cable (rod) hanging down in world -Z.
    - Attach the first rod segment to the driver using a BALL joint (point-to-point attachment).
    - Kinematically animate the drivers and verify the BALL joint keeps the two anchor points coincident.
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

        # Test mode: record BALL joint anchor definitions so we can verify constraint satisfaction.
        ball_parent_ids: list[int] = []
        ball_child_ids: list[int] = []
        ball_parent_local: list[wp.vec3] = []
        ball_child_local: list[wp.vec3] = []

        # Spread the anchor+cable sets along Y (not X).
        y0 = -1.2
        dy = 0.6

        # Anchor height (raise this to give the cables more room to hang)
        z0 = 4.0

        # Two sets: rotate about X and rotate about Y (same motion as fixed/revolute examples)
        sets = [
            ("rotate_x", -1.0, 0),  # (label, x_offset, mode)
            ("rotate_y", 1.0, 1),
        ]

        for _set_idx, (_label, x_offset, m) in enumerate(sets):
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
                    # Build one articulation including both the ball joint and all rod joints.
                    wrap_in_articulation=False,
                )

                # Attach cable start point to driver with a ball joint.
                # `add_rod()` convention:
                # - rod body i has its *body origin* at `positions[i]` (segment start)
                # - the capsule shape (and COM) are offset by +half_height along the body's local +Z
                #
                # Therefore, the cable "start endpoint" is located at body-local z=0 for `rod_bodies[0]`.
                child_anchor_local = wp.vec3(0.0, 0.0, 0.0)
                j_ball = builder.add_joint_ball(
                    parent=body,
                    child=rod_bodies[0],
                    parent_xform=wp.transform(parent_anchor_local, wp.quat_identity()),
                    child_xform=wp.transform(child_anchor_local, wp.quat_identity()),
                    label=f"attach_{key_prefix}",
                )
                # Put all joints (rod cable joints + the ball attachment) into one articulation.
                # Builder requires joint indices be monotonically increasing and contiguous.
                # We create rod joints first, then the ball joint, so the correct order is:
                builder.add_articulation([*rod_joints, j_ball])

                # Record anchor point definitions for testing (world-space error should be near-zero).
                ball_parent_ids.append(int(body))
                ball_child_ids.append(int(rod_bodies[0]))
                ball_parent_local.append(parent_anchor_local)
                ball_child_local.append(child_anchor_local)

                self.anchor_bodies.append(body)
                anchor_base_pos.append(wp.vec3(x, y, z))
                # Sync motion within each set for easy visual comparison.
                anchor_phase.append(0.0)
                anchor_mode.append(int(m))

        builder.add_ground_plane()
        builder.color()
        self.model = builder.finalize()

        # Stiffen ball constraint caps (non-cable joints) so the attachment behaves near-hard.
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=0.1,
            rigid_joint_linear_ke=1.0e9,
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

        # Test buffers (ball joint constraint satisfaction)
        self._ball_parent_ids = wp.array(ball_parent_ids, dtype=wp.int32, device=self.device)
        self._ball_child_ids = wp.array(ball_child_ids, dtype=wp.int32, device=self.device)
        self._ball_parent_local = wp.array(ball_parent_local, dtype=wp.vec3, device=self.device)
        self._ball_child_local = wp.array(ball_child_local, dtype=wp.vec3, device=self.device)

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

            # Kinematic anchor rotation (position fixed, rotation driven)
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
        """Launch the error kernel and return (err_pos_np, rel_angle_np)."""
        n = self._ball_parent_ids.shape[0]
        err_pos = wp.zeros(n, dtype=float, device=self.device)
        rel_angle = wp.zeros(n, dtype=float, device=self.device)
        wp.launch(
            kernel=compute_ball_joint_error,
            dim=n,
            inputs=[
                self._ball_parent_ids,
                self._ball_child_ids,
                self._ball_parent_local,
                self._ball_child_local,
                self.state_0.body_q,
                err_pos,
                rel_angle,
            ],
            device=self.device,
        )
        return err_pos.numpy(), rel_angle.numpy()

    def test_final(self):
        if self._ball_parent_ids.shape[0] == 0:
            return

        err_pos_np, rel_angle_np = self._compute_errors()

        # 1. Position coincidence: all joints
        err_pos_max = float(np.max(err_pos_np))
        tol_pos = 2.0e-3
        if err_pos_max > tol_pos:
            raise AssertionError(f"BALL joint anchor error too large: max={err_pos_max:.6g} > tol={tol_pos}")

        # 2. Angular freedom: cable should NOT follow parent rotation (ball frees all rotation).
        #    At least some joints should show significant relative rotation, proving the
        #    ball joint isn't secretly constraining orientation (which would make it a fixed joint).
        max_rel_angle = float(np.max(rel_angle_np))
        min_rotation_tol = 0.1  # at least 0.1 rad of relative rotation
        if max_rel_angle < min_rotation_tol:
            raise AssertionError(
                f"BALL joint angular freedom not exercised: "
                f"max(relative_angle)={max_rel_angle:.6g} < {min_rotation_tol}"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
