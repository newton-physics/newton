# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adapter between Newton's contact/state buffers and the conveyor force pipeline.

The conveyor force pipeline (``conveyor_forces_kernels`` + ``conveyor_forces_actuators``) is
solver-agnostic: it consumes per-(body, belt) contact data laid out in flat, densely packed
buffers, plus per-body state arrays. Newton reports contacts in a single flat list instead, so
this module rebuilds the packed layout from a :class:`~newton.Contacts` buffer each step.

Responsibilities:

- Map Newton shapes/bodies to the pipeline's belt/body indices (from the ``Belt`` / ``Body`` specs
  the caller builds alongside the scene).
- Classify each Newton rigid contact as a (body, belt) contact, compute its world-space point,
  belt-facing normal, and normal-force magnitude.
- Densely pack the accepted contacts per (body, belt) pair with matching count/start-index
  arrays (``pair_contacts_count`` / ``pair_contacts_start_indices``).
- Gather Newton body state into the per-body arrays the pipeline expects (COM-to-world inputs,
  velocities, inverse mass/inertia), and apply the pipeline's summed force/torque to
  ``State.body_f``.
"""

from typing import NamedTuple

import numpy as np
import warp as wp

wp.config.enable_backward = False


# ---------------------------------------------------------------------------
# Contact classification and dense packing
# ---------------------------------------------------------------------------
@wp.kernel
def classify_contacts(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    contact_force_vec: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_conveyor: wp.array[wp.int32],
    body_pipeline_idx: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    # output
    c_valid: wp.array[wp.int32],
    c_body: wp.array[wp.int32],
    c_belt: wp.array[wp.int32],
    c_point: wp.array[wp.vec3],
    c_normal: wp.array[wp.vec3],
    c_force: wp.array[wp.float32],
    pair_count: wp.array2d[wp.uint32],
):
    """Decide whether contact ``i`` is a belt/body contact and record its pipeline inputs."""
    i = wp.tid()
    c_valid[i] = 0
    if i >= rigid_contact_count[0]:
        return

    shape0 = rigid_contact_shape0[i]
    shape1 = rigid_contact_shape1[i]
    if shape0 < 0 or shape1 < 0:
        return

    conv0 = shape_conveyor[shape0]
    conv1 = shape_conveyor[shape1]

    normal = rigid_contact_normal[i]  # points shape0 -> shape1
    if conv0 >= 0 and conv1 < 0:
        belt = conv0
        body_shape = shape1
        body_point_local = rigid_contact_point1[i]
        normal_toward_body = normal
    elif conv1 >= 0 and conv0 < 0:
        belt = conv1
        body_shape = shape0
        body_point_local = rigid_contact_point0[i]
        normal_toward_body = -normal
    else:
        return

    body_nb = shape_body[body_shape]
    if body_nb < 0:
        return
    bp = body_pipeline_idx[body_nb]
    if bp < 0:
        return

    c_valid[i] = 1
    c_body[i] = bp
    c_belt[i] = belt
    c_point[i] = wp.transform_point(body_q[body_nb], body_point_local)
    c_normal[i] = normal_toward_body
    # Scalar compressive normal force; the pipeline filters out points with zero force.
    c_force[i] = wp.abs(wp.dot(contact_force_vec[i], normal_toward_body))

    wp.atomic_add(pair_count, bp, belt, wp.uint32(1))


@wp.kernel
def prefix_sum_starts(
    body_count: int,
    belt_count: int,
    pair_count: wp.array2d[wp.uint32],
    # output
    pair_start: wp.array2d[wp.uint32],
    fill: wp.array2d[wp.uint32],
):
    """Exclusive prefix sum over (body-major, belt-minor) pairs for dense packing."""
    run = wp.uint32(0)
    for bi in range(body_count):
        for j in range(belt_count):
            pair_start[bi, j] = run
            run += pair_count[bi, j]
            fill[bi, j] = wp.uint32(0)


@wp.kernel
def scatter_contacts(
    c_valid: wp.array[wp.int32],
    c_body: wp.array[wp.int32],
    c_belt: wp.array[wp.int32],
    c_point: wp.array[wp.vec3],
    c_normal: wp.array[wp.vec3],
    c_force: wp.array[wp.float32],
    pair_start: wp.array2d[wp.uint32],
    # output
    fill: wp.array2d[wp.uint32],
    flat_point: wp.array2d[wp.float32],
    flat_normal: wp.array2d[wp.float32],
    flat_force: wp.array2d[wp.float32],
):
    """Scatter accepted contacts into the densely packed per-pair flat buffers."""
    i = wp.tid()
    if c_valid[i] == 0:
        return
    bi = c_body[i]
    j = c_belt[i]
    slot = wp.int32(pair_start[bi, j]) + wp.int32(wp.atomic_add(fill, bi, j, wp.uint32(1)))

    p = c_point[i]
    n = c_normal[i]
    flat_point[slot, 0] = p[0]
    flat_point[slot, 1] = p[1]
    flat_point[slot, 2] = p[2]
    flat_normal[slot, 0] = n[0]
    flat_normal[slot, 1] = n[1]
    flat_normal[slot, 2] = n[2]
    flat_force[slot, 0] = c_force[i]


# ---------------------------------------------------------------------------
# Body-state gather / force apply
# ---------------------------------------------------------------------------
@wp.kernel
def gather_body_state(
    pipeline_to_newton: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
    # output (pipeline-ordered, plain arrays wrapped as indexedarray for the kernels)
    body_positions: wp.array2d[wp.float32],
    body_orientations: wp.array2d[wp.float32],
    body_com_positions: wp.array2d[wp.float32],
    body_com_orientations: wp.array2d[wp.float32],
    body_linear_velocities: wp.array2d[wp.float32],
    body_angular_velocities: wp.array2d[wp.float32],
    body_inverse_masses: wp.array2d[wp.float32],
    body_inverse_inertias: wp.array2d[wp.float32],
):
    """Copy Newton per-body state into the pipeline's per-body arrays (wxyz quaternions)."""
    p = wp.tid()
    nb = pipeline_to_newton[p]

    q = body_q[nb]
    pos = wp.transform_get_translation(q)
    rot = wp.transform_get_rotation(q)  # (x, y, z, w)

    body_positions[p, 0] = pos[0]
    body_positions[p, 1] = pos[1]
    body_positions[p, 2] = pos[2]

    body_orientations[p, 0] = rot[3]  # w
    body_orientations[p, 1] = rot[0]  # x
    body_orientations[p, 2] = rot[1]  # y
    body_orientations[p, 3] = rot[2]  # z

    com = body_com[nb]
    body_com_positions[p, 0] = com[0]
    body_com_positions[p, 1] = com[1]
    body_com_positions[p, 2] = com[2]

    body_com_orientations[p, 0] = 1.0
    body_com_orientations[p, 1] = 0.0
    body_com_orientations[p, 2] = 0.0
    body_com_orientations[p, 3] = 0.0

    qd = body_qd[nb]
    lin = wp.spatial_top(qd)
    ang = wp.spatial_bottom(qd)
    body_linear_velocities[p, 0] = lin[0]
    body_linear_velocities[p, 1] = lin[1]
    body_linear_velocities[p, 2] = lin[2]
    body_angular_velocities[p, 0] = ang[0]
    body_angular_velocities[p, 1] = ang[1]
    body_angular_velocities[p, 2] = ang[2]

    body_inverse_masses[p, 0] = body_inv_mass[nb]

    ii = body_inv_inertia[nb]
    body_inverse_inertias[p, 0] = ii[0, 0]
    body_inverse_inertias[p, 1] = ii[0, 1]
    body_inverse_inertias[p, 2] = ii[0, 2]
    body_inverse_inertias[p, 3] = ii[1, 0]
    body_inverse_inertias[p, 4] = ii[1, 1]
    body_inverse_inertias[p, 5] = ii[1, 2]
    body_inverse_inertias[p, 6] = ii[2, 0]
    body_inverse_inertias[p, 7] = ii[2, 1]
    body_inverse_inertias[p, 8] = ii[2, 2]


@wp.kernel
def extract_linear(spatial_force: wp.array[wp.spatial_vector], out_vec: wp.array[wp.vec3]):
    """Copy the linear part of a per-contact spatial wrench into a plain force vector."""
    i = wp.tid()
    out_vec[i] = wp.spatial_top(spatial_force[i])


@wp.kernel
def apply_body_forces(
    pipeline_to_newton: wp.array[wp.int32],
    force_buffer: wp.array2d[wp.float32],
    torque_buffer: wp.array2d[wp.float32],
    # output
    body_f: wp.array[wp.spatial_vector],
):
    """Add the pipeline's per-body world-space force/torque (about COM) into ``State.body_f``."""
    p = wp.tid()
    nb = pipeline_to_newton[p]
    f = wp.vec3(force_buffer[p, 0], force_buffer[p, 1], force_buffer[p, 2])
    t = wp.vec3(torque_buffer[p, 0], torque_buffer[p, 1], torque_buffer[p, 2])
    body_f[nb] = body_f[nb] + wp.spatial_vector(f, t)


class NewtonConveyorAdapter:
    """Owns the buffers that bridge a Newton :class:`~newton.Model` to the conveyor pipeline."""

    class Belt(NamedTuple):
        """A driven belt surface: which shape it is, and how the pipeline should drive it."""

        shape: int
        velocity_field_type: int
        velocity_field_id: int
        material_index: int
        surface_normal: wp.vec3
        contact_processing_threshold: float

    class Body(NamedTuple):
        """A transported rigid body and the friction-table row that applies to it."""

        body: int
        material_index: int

    def __init__(self, model, belts, bodies, friction_table, contacts):
        self.model = model
        self.device = model.device

        self.belt_count = len(belts)
        self.body_count = len(bodies)
        M = self.belt_count
        N = self.body_count

        # shape -> belt pipeline index (or -1)
        shape_conveyor = np.full(model.shape_count, -1, dtype=np.int32)
        for belt_idx, belt in enumerate(belts):
            shape_conveyor[belt.shape] = belt_idx

        # newton body -> pipeline body index (or -1), and the reverse gather order
        body_pipeline_idx = np.full(model.body_count, -1, dtype=np.int32)
        pipeline_to_newton = np.zeros(N, dtype=np.int32)
        for pi, body in enumerate(bodies):
            body_pipeline_idx[body.body] = pi
            pipeline_to_newton[pi] = body.body

        d = self.device
        self.shape_conveyor = wp.array(shape_conveyor, dtype=wp.int32, device=d)
        self.body_pipeline_idx = wp.array(body_pipeline_idx, dtype=wp.int32, device=d)
        self.pipeline_to_newton = wp.array(pipeline_to_newton, dtype=wp.int32, device=d)

        # Static per-body / per-belt metadata used by the pipeline kernels.
        self.body_material_index = wp.array([b.material_index for b in bodies], dtype=wp.uint32, device=d)
        belt_indices = np.zeros((M, 3), dtype=np.uint32)
        surface_normals = np.zeros((M, 3), dtype=np.float32)
        thresholds = np.zeros(M, dtype=np.float32)
        for j, belt in enumerate(belts):
            belt_indices[j, 0] = belt.velocity_field_type
            belt_indices[j, 1] = belt.velocity_field_id
            belt_indices[j, 2] = belt.material_index
            surface_normals[j] = belt.surface_normal
            thresholds[j] = belt.contact_processing_threshold
        self.conveyor_belt_to_indices_map = wp.array(belt_indices, dtype=wp.uint32, device=d)
        self.surface_normal_buffer = wp.array(surface_normals, dtype=wp.vec3, device=d)
        self.contact_processing_threshold_buffer = wp.array(thresholds, dtype=wp.float32, device=d)

        self.friction_table = wp.array(np.array(friction_table, dtype=np.float32), dtype=wp.float32, device=d)

        # Contact-layout buffers.
        C = contacts.rigid_contact_max
        self.max_contact_count = C
        self.contact_force_vec = wp.zeros(C, dtype=wp.vec3, device=d)
        self.c_valid = wp.zeros(C, dtype=wp.int32, device=d)
        self.c_body = wp.zeros(C, dtype=wp.int32, device=d)
        self.c_belt = wp.zeros(C, dtype=wp.int32, device=d)
        self.c_point = wp.zeros(C, dtype=wp.vec3, device=d)
        self.c_normal = wp.zeros(C, dtype=wp.vec3, device=d)
        self.c_force = wp.zeros(C, dtype=wp.float32, device=d)
        self.pair_count = wp.zeros((N, M), dtype=wp.uint32, device=d)
        self.pair_start = wp.zeros((N, M), dtype=wp.uint32, device=d)
        self.fill = wp.zeros((N, M), dtype=wp.uint32, device=d)
        self.flat_point = wp.zeros((C, 3), dtype=wp.float32, device=d)
        self.flat_normal = wp.zeros((C, 3), dtype=wp.float32, device=d)
        self.flat_force = wp.zeros((C, 1), dtype=wp.float32, device=d)

        # Per-body pipeline arrays (plain) + persistent indexed views for the kernels.
        self.body_positions = wp.zeros((N, 3), dtype=wp.float32, device=d)
        self.body_orientations = wp.zeros((N, 4), dtype=wp.float32, device=d)
        self.body_com_positions = wp.zeros((N, 3), dtype=wp.float32, device=d)
        self.body_com_orientations = wp.zeros((N, 4), dtype=wp.float32, device=d)
        self.body_linear_velocities = wp.zeros((N, 3), dtype=wp.float32, device=d)
        self.body_angular_velocities = wp.zeros((N, 3), dtype=wp.float32, device=d)
        self.body_inverse_masses = wp.zeros((N, 1), dtype=wp.float32, device=d)
        self.body_inverse_inertias = wp.zeros((N, 9), dtype=wp.float32, device=d)

        body_idx = wp.array(np.arange(N, dtype=np.int32), dtype=wp.int32, device=d)
        self._body_positions_ia = wp.indexedarray(self.body_positions, [body_idx])
        self._body_com_positions_ia = wp.indexedarray(self.body_com_positions, [body_idx])
        self._body_linear_velocities_ia = wp.indexedarray(self.body_linear_velocities, [body_idx])
        self._body_angular_velocities_ia = wp.indexedarray(self.body_angular_velocities, [body_idx])
        self._body_inverse_masses_ia = wp.indexedarray(self.body_inverse_masses, [body_idx])
        self._body_inverse_inertias_ia = wp.indexedarray(self.body_inverse_inertias, [body_idx])
        self._pair_count_ia = wp.indexedarray(self.pair_count, [body_idx])
        self._pair_start_ia = wp.indexedarray(self.pair_start, [body_idx])

    # -- indexed views the pipeline kernels take directly --
    @property
    def pair_contacts_count(self):
        return self._pair_count_ia

    @property
    def pair_contacts_start_indices(self):
        return self._pair_start_ia

    def report_contact_forces(self, solver, contacts, state_post, body_q_prev, dt, solver_type):
        """Fill ``contact_force_vec`` with the per-contact world-space force from the solver."""
        if solver_type == "vbd":
            solver.collect_rigid_contact_forces(state_post.body_q, body_q_prev, contacts, dt)
            wp.copy(self.contact_force_vec, contacts.rigid_contact_force)
        else:
            solver.update_contacts(contacts)
            wp.launch(
                extract_linear,
                dim=self.max_contact_count,
                inputs=[contacts.force, self.contact_force_vec],
                device=self.device,
            )

    def build_contact_layout(self, contacts, state_post):
        """Reformat Newton contacts into the densely packed per-(body, belt) pipeline layout."""
        self.pair_count.zero_()
        wp.launch(
            classify_contacts,
            dim=self.max_contact_count,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                self.contact_force_vec,
                self.model.shape_body,
                self.shape_conveyor,
                self.body_pipeline_idx,
                state_post.body_q,
            ],
            outputs=[
                self.c_valid,
                self.c_body,
                self.c_belt,
                self.c_point,
                self.c_normal,
                self.c_force,
                self.pair_count,
            ],
            device=self.device,
        )
        wp.launch(
            prefix_sum_starts,
            dim=1,
            inputs=[self.body_count, self.belt_count, self.pair_count],
            outputs=[self.pair_start, self.fill],
            device=self.device,
        )
        wp.launch(
            scatter_contacts,
            dim=self.max_contact_count,
            inputs=[
                self.c_valid,
                self.c_body,
                self.c_belt,
                self.c_point,
                self.c_normal,
                self.c_force,
                self.pair_start,
            ],
            outputs=[self.fill, self.flat_point, self.flat_normal, self.flat_force],
            device=self.device,
        )

    def gather_state(self, state):
        """Refresh the per-body pipeline arrays from Newton state/model."""
        wp.launch(
            gather_body_state,
            dim=self.body_count,
            inputs=[
                self.pipeline_to_newton,
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
            ],
            outputs=[
                self.body_positions,
                self.body_orientations,
                self.body_com_positions,
                self.body_com_orientations,
                self.body_linear_velocities,
                self.body_angular_velocities,
                self.body_inverse_masses,
                self.body_inverse_inertias,
            ],
            device=self.device,
        )

    def apply_forces(self, state, force_buffer, torque_buffer):
        """Add the pipeline's per-body wrench to ``state.body_f``."""
        wp.launch(
            apply_body_forces,
            dim=self.body_count,
            inputs=[self.pipeline_to_newton, force_buffer, torque_buffer],
            outputs=[state.body_f],
            device=self.device,
        )
