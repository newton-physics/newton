# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SDF RJ45 Plug-Socket Insertion
#
# Use the translation gizmo to move the plug toward the socket.
# Click an axis arrow to slide along one axis, or a plane square
# for two-axis motion. The latch deflects on entry and locks
# the plug once fully inserted.
#
# Command: uv run -m newton.examples contacts_rj45_plug
#
###########################################################################

import dataclasses

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.usd
import newton.utils
from newton.math import quat_between_vectors_robust
from newton.solvers import SolverVBD

CONTACT_KE = 1.0e5
CONTACT_KD = 0.0

SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    mu=0.0,
    ke=CONTACT_KE,
    kd=CONTACT_KD,
    gap=0.002,
    density=1.0e6,
    mu_torsional=0.0,
    mu_rolling=0.0,
)

MESH_SDF_MAX_RESOLUTION = 128
MESH_SDF_NARROW_BAND_RANGE = (-2.0 * SHAPE_CFG.gap, 2.0 * SHAPE_CFG.gap)

PLUG_Y_OFFSET = -0.025

CABLE_RADIUS = 0.00325
CABLE_KINEMATIC_COUNT = 4  # first N rod bodies are inside the plug and follow it

# Contact parameters for cable and ground plane (tuned for VBD).
CABLE_MU = 2.0

# Latch revolute-joint tuning.
LATCH_LIMIT_LOWER = -0.2  # max inward deflection [rad]
LATCH_LIMIT_UPPER = 0.3  # max outward deflection [rad]
LATCH_SPRING_KE = 0.15  # angular return-spring stiffness [N*m/rad]
LATCH_SPRING_KD = 0.2  # dimensionless damping ratio (VBD: D = kd * ke)
LATCH_LIMIT_KD = 1.0e-4  # dimensionless limit damping (VBD: D = kd * limit_ke)


@wp.kernel
def _apply_gizmo_force(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    pick_target: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    pick_body: wp.array[int],
    plug_idx: int,
    latch_idx: int,
    gravity: wp.vec3,
):
    """Apply forces based on interaction mode.

    ``pick_body[0]`` encodes the mode:
      * ``>= 0`` -- viewer is picking that body index (damping only on others)
      * ``< 0``  -- spring toward ``pick_target[0]``

    Also cancels gravity for the plug and latch so only the cable sags.
    """
    # Cancel gravity for plug and latch unconditionally.
    anti_g0 = -gravity * body_mass[plug_idx]
    anti_g1 = -gravity * body_mass[latch_idx]
    wp.atomic_add(body_f, plug_idx, wp.spatial_vector(anti_g0, wp.vec3(0.0)))
    wp.atomic_add(body_f, latch_idx, wp.spatial_vector(anti_g1, wp.vec3(0.0)))

    target = pick_target[0]
    picked_body = pick_body[0]

    if picked_body >= 0:
        if picked_body != plug_idx:
            vel0 = wp.spatial_top(body_qd[plug_idx])
            mass0 = body_mass[plug_idx]
            f0 = -(10.0 + mass0) * damping * vel0
            wp.atomic_add(body_f, plug_idx, wp.spatial_vector(f0, wp.vec3(0.0)))
        if picked_body != latch_idx:
            vel1 = wp.spatial_top(body_qd[latch_idx])
            mass1 = body_mass[latch_idx]
            f1 = -(10.0 + mass1) * damping * vel1
            wp.atomic_add(body_f, latch_idx, wp.spatial_vector(f1, wp.vec3(0.0)))
        return

    pos0 = wp.transform_get_translation(body_q[plug_idx])
    vel0 = wp.spatial_top(body_qd[plug_idx])
    mass0 = body_mass[plug_idx]
    mult0 = 10.0 + mass0

    f0 = mult0 * (stiffness * (target - pos0) - damping * vel0)
    wp.atomic_add(body_f, plug_idx, wp.spatial_vector(f0, wp.vec3(0.0)))

    vel1 = wp.spatial_top(body_qd[latch_idx])
    mass1 = body_mass[latch_idx]
    spring_accel = (target - pos0) * (mult0 * stiffness / mass0)
    f1 = spring_accel * mass1 - vel1 * ((10.0 + mass1) * damping)
    wp.atomic_add(body_f, latch_idx, wp.spatial_vector(f1, wp.vec3(0.0)))


@wp.kernel
def _sync_cable_anchors(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    plug_idx: int,
    anchor_indices: wp.array[int],
    anchor_offsets: wp.array[wp.vec3],
    anchor_rotations: wp.array[wp.quat],
):
    """Copy the plug transform into kinematic cable bodies."""
    tid = wp.tid()
    plug_tf = body_q[plug_idx]
    plug_pos = wp.transform_get_translation(plug_tf)
    plug_rot = wp.transform_get_rotation(plug_tf)
    idx = anchor_indices[tid]
    anchor_world = plug_pos + wp.quat_rotate(plug_rot, anchor_offsets[tid])
    cable_rot = wp.normalize(wp.mul(plug_rot, anchor_rotations[tid]))
    body_q[idx] = wp.transform(anchor_world, cable_rot)
    body_qd[idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _align_cable_orientations(
    body_q: wp.array[wp.transform],
    cable_body_idx: wp.array[int],
    cable_next_idx: wp.array[int],
):
    """Swing-correct each dynamic cable capsule to its deformed segment direction.

    Keeps the body origin (capsule start point) fixed and only updates
    the rotation so the capsule's +Z axis points toward the next body's
    origin.  This preserves chain connectivity at the start-point side.
    """
    tid = wp.tid()
    bi = cable_body_idx[tid]
    bi_next = cable_next_idx[tid]

    tf = body_q[bi]
    pos = wp.transform_get_translation(tf)
    rot = wp.transform_get_rotation(tf)

    next_pos = wp.transform_get_translation(body_q[bi_next])
    seg = next_pos - pos
    seg_len = wp.length(seg)
    if seg_len < 1.0e-10:
        return
    d = seg / seg_len

    z_current = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    q_swing = quat_between_vectors_robust(z_current, d)
    rot_new = wp.normalize(wp.mul(q_swing, rot))

    body_q[bi] = wp.transform(pos, rot_new)


def _load_mesh(stage, prim_path: str) -> tuple[newton.Mesh, wp.vec3]:
    """Load a mesh from USD, center at prim origin, and build SDF.

    Returns:
        Tuple of (mesh, prim_pos) where prim_pos is the prim world-space
        translation [m], usable as the body/shape position.
    """
    prim = stage.GetPrimAtPath(prim_path)
    usd_mesh = newton.usd.get_mesh(prim, load_normals=True)

    tf = newton.usd.get_transform(prim, local=False)
    prim_pos = wp.transform_get_translation(tf)

    vertices = np.array(usd_mesh.vertices, dtype=np.float32)
    indices = np.array(usd_mesh.indices, dtype=np.int32)
    normals = np.array(usd_mesh.normals, dtype=np.float32) if usd_mesh.normals is not None else None

    mesh = newton.Mesh(vertices, indices, normals=normals)
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=SHAPE_CFG.gap,
    )
    return mesh, prim_pos


def _load_cable_centerline(stage) -> tuple[wp.vec3, ...]:
    """Load cable centerline from the ``/World/CableCurve`` BasisCurves prim.

    Returns world-space positions with :data:`PLUG_Y_OFFSET` applied.
    """
    prim = stage.GetPrimAtPath("/World/CableCurve")
    all_points = UsdGeom.BasisCurves(prim).GetPointsAttr().Get()

    tf = newton.usd.get_transform(prim, local=False)
    prim_pos = wp.transform_get_translation(tf)

    return tuple(
        wp.vec3(
            float(p[0]) + float(prim_pos[0]),
            float(p[1]) + float(prim_pos[1]) + PLUG_Y_OFFSET,
            float(p[2]) + float(prim_pos[2]),
        )
        for p in all_points
    )


class Example:
    def __init__(self, viewer, _args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.pick_stiffness = 50.0
        self.pick_damping = 10.0

        usd_path = newton.examples.get_asset("rj45_plug.usd")
        stage = Usd.Stage.Open(usd_path)

        socket_mesh, sc = _load_mesh(stage, "/World/Socket")
        plug_mesh, pc = _load_mesh(stage, "/World/Plug")
        latch_mesh, lc = _load_mesh(stage, "/World/Latch")

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.rigid_gap = 0.005

        builder.add_ground_plane()

        # Socket (static body)
        socket_shape = builder.add_shape_mesh(
            -1,
            mesh=socket_mesh,
            xform=wp.transform(sc, wp.quat_identity()),
            cfg=SHAPE_CFG,
            label="socket",
        )

        # Plug (dynamic body, offset along -Y insertion axis)
        plug_pos = wp.vec3(pc[0], pc[1] + PLUG_Y_OFFSET, pc[2])
        self._plug_body = builder.add_link(
            xform=wp.transform(plug_pos, wp.quat_identity()),
            label="plug",
        )
        plug_shape = builder.add_shape_mesh(
            self._plug_body,
            mesh=plug_mesh,
            cfg=SHAPE_CFG,
        )

        # Latch (dynamic body, same Y offset as plug)
        latch_pos = wp.vec3(lc[0], lc[1] + PLUG_Y_OFFSET, lc[2])
        self._latch_body = builder.add_link(
            xform=wp.transform(latch_pos, wp.quat_identity()),
            label="latch",
        )
        latch_shape = builder.add_shape_mesh(
            self._latch_body,
            mesh=latch_mesh,
            cfg=SHAPE_CFG,
        )
        connector_shapes = (socket_shape, plug_shape, latch_shape)

        # D6 joint: world -> plug (free translation, locked rotation)
        JointDof = newton.ModelBuilder.JointDofConfig
        d6_joint = builder.add_joint_d6(
            parent=-1,
            child=self._plug_body,
            linear_axes=(
                JointDof(axis=(1.0, 0.0, 0.0)),
                JointDof(axis=(0.0, 1.0, 0.0)),
                JointDof(axis=(0.0, 0.0, 1.0)),
            ),
            angular_axes=None,
            parent_xform=wp.transform(plug_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )

        # Revolute joint: plug -> latch (hinge along -X axis)
        rev_joint = builder.add_joint_revolute(
            parent=self._plug_body,
            child=self._latch_body,
            axis=(-1.0, 0.0, 0.0),
            parent_xform=wp.transform(lc - pc, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_ke=LATCH_SPRING_KE,
            target_kd=LATCH_SPRING_KD,
            limit_lower=LATCH_LIMIT_LOWER,
            limit_upper=LATCH_LIMIT_UPPER,
            limit_kd=LATCH_LIMIT_KD,
            collision_filter_parent=True,
        )

        builder.add_articulation([d6_joint, rev_joint])

        cable_points = _load_cable_centerline(stage)
        cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
        bend_stiffness = 1.0e1

        rod_bodies, _ = builder.add_rod(
            positions=cable_points,
            quaternions=cable_quats,
            radius=CABLE_RADIUS,
            cfg=dataclasses.replace(
                builder.default_shape_cfg,
                ke=CONTACT_KE,
                kd=CONTACT_KD,
                mu=CABLE_MU,
            ),
            bend_stiffness=bend_stiffness,
            bend_damping=1.0e-1,
            label="cable",
        )

        # Collision-filter cable segments that overlap the plug at rest.
        for body_idx in rod_bodies[:CABLE_KINEMATIC_COUNT]:
            for cable_shape in builder.body_shapes[body_idx]:
                for conn_shape in connector_shapes:
                    builder.add_shape_collision_filter_pair(cable_shape, conn_shape)

        # Lock the kinematic prefix and the far cable end.
        for idx in (*rod_bodies[:CABLE_KINEMATIC_COUNT], rod_bodies[-1]):
            builder.body_mass[idx] = 0.0
            builder.body_inv_mass[idx] = 0.0
            builder.body_inertia[idx] = wp.mat33(0.0)
            builder.body_inv_inertia[idx] = wp.mat33(0.0)

        anchor_body_ids = tuple(rod_bodies[:CABLE_KINEMATIC_COUNT])
        anchor_offsets = tuple(cable_points[i] - plug_pos for i in range(CABLE_KINEMATIC_COUNT))
        anchor_rots = tuple(cable_quats[i] for i in range(CABLE_KINEMATIC_COUNT))

        builder.color()
        self.model = builder.finalize()

        self._cable_anchor_indices = wp.array(anchor_body_ids, dtype=int, device=self.model.device)
        self._cable_anchor_offsets = wp.array(anchor_offsets, dtype=wp.vec3, device=self.model.device)
        self._cable_anchor_rotations = wp.array(anchor_rots, dtype=wp.quat, device=self.model.device)

        # Chord alignment: include the last kinematic body so it visually
        # aims toward the first dynamic body after deformation.  The solver
        # always sees the rest-relative rotation (set by _sync_cable_anchors
        # before each step); this post-solve correction is purely cosmetic.
        align_start = max(CABLE_KINEMATIC_COUNT - 1, 0)
        align_bodies = tuple(rod_bodies[align_start:-1])
        align_next = tuple(rod_bodies[align_start + 1 :])
        self._cable_align_indices = wp.array(align_bodies, dtype=int, device=self.model.device)
        self._cable_align_next = wp.array(align_next, dtype=int, device=self.model.device)
        self._cable_align_count = len(align_bodies)

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True

        self.viewer.set_camera(
            pos=wp.vec3(0.125, plug_pos[1] - 0.025, 0.03),
            pitch=-10.0,
            yaw=180.0,
        )
        if hasattr(self.viewer, "_cam_speed"):
            self.viewer._cam_speed = 0.2

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._initial_body_q = self.state_0.body_q.numpy().copy()

        self.solver = SolverVBD(
            self.model,
            iterations=12,
            rigid_body_contact_buffer_size=256,
        )
        for j in range(self.model.joint_count):
            self.solver.set_joint_constraint_mode(j, False)

        self._rest_pos = plug_pos
        self.gizmo_tf = wp.transform(plug_pos, wp.quat_identity())

        self._pick_body = wp.array([-1], dtype=int, device=self.model.device)
        self._pick_target = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self._gravity = wp.vec3(*self.model.gravity.numpy()[0])

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                kernel=_apply_gizmo_force,
                dim=1,
                inputs=(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.model.body_mass,
                    self._pick_target,
                    self.pick_stiffness,
                    self.pick_damping,
                    self._pick_body,
                    self._plug_body,
                    self._latch_body,
                    self._gravity,
                ),
                device=self.model.device,
            )
            self.viewer.apply_forces(self.state_0)

            # Teleport kinematic cable bodies to follow the plug before the
            # solver runs, so joint constraints at the boundary see the
            # correct anchor positions and rest-relative rotations.
            wp.launch(
                kernel=_sync_cable_anchors,
                dim=CABLE_KINEMATIC_COUNT,
                inputs=(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._plug_body,
                    self._cable_anchor_indices,
                    self._cable_anchor_offsets,
                    self._cable_anchor_rotations,
                ),
                device=self.model.device,
            )
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            # Snap each capsule's +Z to the chord toward the next node so the
            # rendered chain follows the deformed centerline.  Purely cosmetic;
            # the solver doesn't see these rotations until _sync_cable_anchors
            # resets the kinematic prefix on the next substep.
            wp.launch(
                kernel=_align_cable_orientations,
                dim=self._cable_align_count,
                inputs=(
                    self.state_0.body_q,
                    self._cable_align_indices,
                    self._cable_align_next,
                ),
                device=self.model.device,
            )

    def step(self):
        gp = wp.transform_get_translation(self.gizmo_tf)

        picked_body = int(self.viewer.picking.pick_body.numpy()[0])

        self._pick_body.assign([picked_body])
        self._pick_target.assign([gp])

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        counts = self.solver.body_body_contact_counts.numpy()
        buf = self.solver.body_body_contact_buffer_pre_alloc
        overflow = np.where(counts > buf)[0]
        if len(overflow):
            for i in overflow:
                label = self.model.body_label[i] if hasattr(self.model, "body_label") else i
                print(f"[contact overflow] body {label} (idx={i}): {counts[i]} contacts (buffer={buf})")

        # Snap gizmo to the plug when the user isn't dragging it.
        gizmo_active = self.viewer.gizmo_is_using
        if not gizmo_active:
            plug_tf = self.state_0.body_q.numpy()[self._plug_body]
            if picked_body >= 0:
                snap = wp.vec3(*plug_tf[:3])
            else:
                snap = wp.vec3(self._rest_pos[0], plug_tf[1], self._rest_pos[2])
            self.gizmo_tf[:] = wp.transform(snap, wp.quat_identity())

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_gizmo("plug", self.gizmo_tf, rotate=())
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        initial_q = self._initial_body_q
        for i in range(len(body_q)):
            assert np.all(np.isfinite(body_q[i])), f"Body {i} has non-finite transform"
            drift = np.linalg.norm(body_q[i] - initial_q[i])
            assert drift < 1.0, f"Body {i} drifted {drift:.4f} from initial transform"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
