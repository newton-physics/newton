# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame-to-frame anchors for rigid-soft contacts.

A soft contact record is identified by its soft feature (``soft_contact_indices``), its shape, and,
for full-surface mesh records, its mesh feature pair. Unlike rigid contacts, this key is exact, so
no sort or nearest search is needed: each record looks up the previous record with the same key
through a few short lists per first corner.

Each record carries an anchor (:class:`SoftContactAnchors`) beside its fresh contact points: a soft-side
material point and a body-frame surface point. A matched record keeps its previous anchor while the two
points still touch and have slipped apart tangentially by at most the matching position threshold, and
while the body point lies within that threshold of this pass's contact plane, so a record that moves to
an opposing wall of the same mesh starts over. Opposing body-frame normals also restart a record, even
when a thin wall's two faces fit within the position threshold. Records on a moving surface (nonzero
``soft_contact_body_vel``) anchor at their fresh points, since the surface material moves away from any
body-frame point. Consumers can then measure tangential slip as the drift between the two anchor points,
which persists across collision passes, rather than as the motion since the last one. A kept anchor also
keeps the normal load its consumer left on it; every other record starts without one. The fresh contact
points are never changed.

The matcher saves a buffer's anchors as the consumers left them, so a solver may move them between
passes; VBD drags them along once a contact slides. Each buffer therefore replays its own anchors, and
only those its pipeline wrote (:attr:`SoftContactHistory.owner`) since the last reset of the record's
worlds (:attr:`SoftContactHistory.reset_stamp`). Call order (inside :class:`~newton.CollisionPipeline`)::

    matcher.save(...)  # before the pass rewrites the buffer's records
    matcher.replay(...)  # after all soft records are written
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..core.reset import WorldResetEpochs, world_reset_epoch

if TYPE_CHECKING:
    from ..sim.contacts import Contacts
    from ..sim.model import Model
    from ..sim.state import State


@wp.struct
class SoftContactAnchors:
    """Contact points each rigid-soft record keeps across collision passes; null arrays when none are kept."""

    enabled: wp.array[wp.bool]
    """Whether the current records use anchored friction; device-resident for CUDA graph replay."""
    barycentric: wp.array[wp.vec3]
    """Soft-side material point, as weights over the record's soft particles [unitless]."""
    body_pos: wp.array[wp.vec3]
    """Body-frame surface point [m]."""
    normal_force: wp.array[float]
    """Normal load [N] the record carried at the end of the previous step, -1 without one."""


class SoftContactHistory:
    """Per-buffer state of soft contact matching: the anchors and what is needed to replay them.

    Args:
        capacity: Number of soft contact records (``soft_contact_max``).
        device: Device to allocate on.
        replayable: Whether a collision pass replays the anchors into this buffer. Copies that only
            carry anchors to a solver, such as coupled-solver filtered buffers, skip the replay state.
    """

    def __init__(self, capacity: int, device=None, replayable: bool = True):
        replay_capacity = capacity if replayable else 0
        with wp.ScopedDevice(device):
            self.anchors = SoftContactAnchors()
            self.anchors.enabled = wp.zeros(1, dtype=wp.bool)
            self.anchors.barycentric = wp.zeros(capacity, dtype=wp.vec3)
            self.anchors.body_pos = wp.zeros(capacity, dtype=wp.vec3)
            self.anchors.normal_force = wp.full(capacity, -1.0, dtype=float)
            # Body-frame normals let replay tell opposing faces apart even after the body rotates.
            self.normal_body = wp.zeros(replay_capacity, dtype=wp.vec3)
            self.reset_stamp = wp.zeros(replay_capacity, dtype=wp.int32)
            """World reset epochs each record was written at."""
            self.owner = wp.full(1, -1, dtype=wp.int32)
            """History id of the pipeline that last wrote the buffer's records."""

    def use_fresh_points(self, contacts: Contacts) -> None:
        """Mirror fresh geometry for consumers when a pipeline without matching fills this buffer."""
        wp.copy(self.anchors.barycentric, contacts.soft_contact_barycentric)
        wp.copy(self.anchors.body_pos, contacts.soft_contact_body_pos)
        self.anchors.normal_force.fill_(-1.0)
        self.anchors.enabled.fill_(False)


@wp.func
def _soft_point(corners: wp.vec3i, bary: wp.vec3, particle_q: wp.array[wp.vec3]):
    x = bary[0] * particle_q[corners[0]]
    if corners[1] >= 0:
        x += bary[1] * particle_q[corners[1]]
    if corners[2] >= 0:
        x += bary[2] * particle_q[corners[2]]
    return x


@wp.func
def _soft_radius(corners: wp.vec3i, particle_radius: wp.array[float]):
    radius = particle_radius[corners[0]]
    if corners[1] >= 0:
        radius = wp.max(radius, particle_radius[corners[1]])
    if corners[2] >= 0:
        radius = wp.max(radius, particle_radius[corners[2]])
    return radius


@wp.func
def _same_key(
    corners: wp.vec3i, shape: int, feature: wp.vec3i, prev_corners: wp.vec3i, prev_shape: int, prev_feature: wp.vec3i
):
    return corners == prev_corners and shape == prev_shape and feature == prev_feature


SLOTS_PER_PARTICLE = wp.constant(4)


@wp.func
def _key_slot(corners: wp.vec3i, shape: int, feature: wp.vec3i):
    """List head of a record key: one of a few heads of its first corner, picked by hashing the rest.

    Grouping by the first corner keeps neighboring records' heads close in memory, and spreading over a few
    heads keeps lists short when one particle starts many records (faces and edges against several shapes).
    """
    h = wp.uint32(2166136261)
    h = (h ^ wp.uint32(corners[1])) * wp.uint32(16777619)
    h = (h ^ wp.uint32(corners[2])) * wp.uint32(16777619)
    h = (h ^ wp.uint32(shape)) * wp.uint32(16777619)
    h = (h ^ wp.uint32(feature[0])) * wp.uint32(16777619)
    h = (h ^ wp.uint32(feature[1])) * wp.uint32(16777619)
    h = (h ^ wp.uint32(feature[2])) * wp.uint32(16777619)
    return corners[0] * SLOTS_PER_PARTICLE + int(h % wp.uint32(SLOTS_PER_PARTICLE))


@wp.func
def _record_feature(features: wp.array[wp.vec3i], index: int):
    if features.shape[0] == 0:
        return wp.vec3i(-1, -1, -1)
    return features[index]


@wp.kernel(enable_backward=False)
def _replay_soft_contacts_kernel(
    thread_count: int,
    world_count: int,
    pos_threshold: float,
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[float],
    contact_count: wp.array[int],
    contact_max: int,
    contact_indices: wp.array[wp.vec3i],
    contact_shape: wp.array[int],
    contact_features: wp.array[wp.vec3i],
    contact_barycentric: wp.array[wp.vec3],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    reset_epochs: wp.array[wp.int32],
    prev_head: wp.array[wp.int32],
    prev_next: wp.array[wp.int32],
    prev_indices: wp.array[wp.vec3i],
    prev_shape: wp.array[wp.int32],
    prev_features: wp.array[wp.vec3i],
    prev_reset_stamp: wp.array[wp.int32],
    prev_barycentric: wp.array[wp.vec3],
    prev_body_pos: wp.array[wp.vec3],
    prev_normal_body: wp.array[wp.vec3],
    prev_normal_force: wp.array[float],
    # out: every record's reset stamp, body-frame normal, and anchor
    contact_reset_stamp: wp.array[wp.int32],
    contact_normal_body: wp.array[wp.vec3],
    anchor_barycentric: wp.array[wp.vec3],
    anchor_body_pos: wp.array[wp.vec3],
    anchor_normal_force: wp.array[float],
):
    for i in range(wp.tid(), wp.min(contact_count[0], contact_max), thread_count):
        corners = contact_indices[i]
        shape = contact_shape[i]
        stamp = world_reset_epoch(particle_world[corners[0]], reset_epochs, world_count) + world_reset_epoch(
            shape_world[shape], reset_epochs, world_count
        )
        contact_reset_stamp[i] = stamp
        X_wb = wp.transform_identity()
        body = shape_body[shape]
        if body >= 0:
            X_wb = body_q[body]
        n = contact_normal[i]
        normal_body = wp.transform_vector(wp.transform_inverse(X_wb), n)
        contact_normal_body[i] = normal_body
        anchor_barycentric[i] = contact_barycentric[i]
        anchor_body_pos[i] = contact_body_pos[i]
        anchor_normal_force[i] = -1.0
        # A moving surface (conveyor) carries its material away from any point fixed in the body frame.
        if wp.length_sq(contact_body_vel[i]) > 0.0:
            continue
        feature = _record_feature(contact_features, i)

        contact_distance = _soft_radius(corners, particle_radius)
        if shape_margin.shape[0] > 0:
            contact_distance += shape_margin[shape]

        fresh_surface = wp.transform_point(X_wb, contact_body_pos[i])
        best = int(-1)
        best_slip_sq = pos_threshold * pos_threshold
        node = prev_head[_key_slot(corners, shape, feature)]
        while node >= 0:
            prev_feature = _record_feature(prev_features, node)
            if prev_reset_stamp[node] == stamp and _same_key(
                corners, shape, feature, prev_indices[node], prev_shape[node], prev_feature
            ):
                # The previous points still describe this contact if the body point lies on the surface this
                # pass touches, they still touch, and they have not slipped apart. Only the fresh point's plane
                # is a reference: a soft edge or face lying flat on a surface has no unique closest point, so
                # detection may pick another point on it every pass.
                surface = wp.transform_point(X_wb, prev_body_pos[node])
                d = _soft_point(corners, prev_barycentric[node], particle_q) - surface
                normal_distance = wp.dot(n, d)
                slip_sq = wp.length_sq(d - normal_distance * n)
                on_surface = wp.abs(wp.dot(n, surface - fresh_surface)) <= pos_threshold
                if (
                    on_surface
                    # A distance tolerance alone can accept the opposite face of a thin wall. Compare
                    # in the body frame so rigid rotation does not invalidate an unchanged surface.
                    and wp.dot(normal_body, prev_normal_body[node]) > 0.0
                    and normal_distance < contact_distance
                    and (slip_sq < best_slip_sq or (slip_sq == best_slip_sq and (best < 0 or node < best)))
                ):
                    best = node
                    best_slip_sq = slip_sq
            node = prev_next[node]

        if best >= 0:
            anchor_barycentric[i] = prev_barycentric[best]
            anchor_body_pos[i] = prev_body_pos[best]
            anchor_normal_force[i] = prev_normal_force[best]


@wp.kernel(enable_backward=False)
def _save_soft_contacts_kernel(
    thread_count: int,
    owner: int,
    history_owner: wp.array[wp.int32],
    contact_count: wp.array[int],
    contact_max: int,
    contact_indices: wp.array[wp.vec3i],
    contact_shape: wp.array[int],
    contact_features: wp.array[wp.vec3i],
    contact_reset_stamp: wp.array[wp.int32],
    anchor_barycentric: wp.array[wp.vec3],
    anchor_body_pos: wp.array[wp.vec3],
    contact_normal_body: wp.array[wp.vec3],
    anchor_normal_force: wp.array[float],
    prev_head: wp.array[wp.int32],
    prev_next: wp.array[wp.int32],
    prev_indices: wp.array[wp.vec3i],
    prev_shape: wp.array[wp.int32],
    prev_features: wp.array[wp.vec3i],
    prev_reset_stamp: wp.array[wp.int32],
    prev_barycentric: wp.array[wp.vec3],
    prev_body_pos: wp.array[wp.vec3],
    prev_normal_body: wp.array[wp.vec3],
    prev_normal_force: wp.array[float],
):
    if history_owner[0] != owner:
        return
    for i in range(wp.tid(), wp.min(contact_count[0], contact_max), thread_count):
        corners = contact_indices[i]
        prev_indices[i] = corners
        prev_shape[i] = wp.int32(contact_shape[i])
        if prev_features.shape[0] > 0:
            prev_features[i] = _record_feature(contact_features, i)
        prev_reset_stamp[i] = contact_reset_stamp[i]
        prev_barycentric[i] = anchor_barycentric[i]
        prev_body_pos[i] = anchor_body_pos[i]
        prev_normal_body[i] = contact_normal_body[i]
        prev_normal_force[i] = anchor_normal_force[i]
        slot = _key_slot(corners, contact_shape[i], _record_feature(contact_features, i))
        prev_next[i] = wp.atomic_exch(prev_head, slot, wp.int32(i))


class SoftContactMatcher:
    """Replay of rigid-soft contact anchors across collision passes.

    Internal helper owned by :class:`~newton.CollisionPipeline` with ``soft_contact_matching=True``.
    It owns the saved records; its buffers are allocated at construction. Contact-buffer anchors
    must also be allocated before capture (automatically for the pipeline's own buffers).

    Args:
        capacity: Maximum number of soft contacts (``soft_contact_max``).
        particle_count: Number of particles in the model.
        owner: The pipeline's history id; a buffer's records are saved only if this pipeline wrote them.
        resets: The pipeline's world reset epochs; records are saved stamped with them.
        pos_threshold: Largest tangential slip [m] between a record's previous soft and body points,
            and largest distance [m] of the previous body point from this pass's contact plane, for the
            record to keep them.
        mesh_features: Whether records carry full-surface mesh feature ids, which then join the key.
        device: Device to allocate on.
    """

    def __init__(
        self,
        capacity: int,
        particle_count: int,
        owner: int,
        resets: WorldResetEpochs,
        pos_threshold: float,
        mesh_features: bool,
        device=None,
    ):
        with wp.ScopedDevice(device):
            self._capacity = capacity
            # A fixed grid that strides over the live records, so a pass does not pay for empty capacity.
            cuda = wp.get_device().is_cuda
            self._thread_count = min(capacity, wp.get_device().sm_count * 2048) if cuda else capacity
            self._owner = owner
            self._resets = resets
            self._pos_threshold = pos_threshold
            self._prev_head = wp.full(particle_count * SLOTS_PER_PARTICLE, -1, dtype=wp.int32)
            self._prev_next = wp.full(capacity, -1, dtype=wp.int32)
            self._prev_indices = wp.full(capacity, wp.vec3i(-1, -1, -1), dtype=wp.vec3i)
            self._prev_shape = wp.full(capacity, -1, dtype=wp.int32)
            self._prev_features = wp.full(capacity if mesh_features else 0, wp.vec3i(-1, -1, -1), dtype=wp.vec3i)
            self._prev_reset_stamp = wp.zeros(capacity, dtype=wp.int32)
            self._prev_barycentric = wp.zeros(capacity, dtype=wp.vec3)
            self._prev_body_pos = wp.zeros(capacity, dtype=wp.vec3)
            self._prev_normal_body = wp.zeros(capacity, dtype=wp.vec3)
            self._prev_normal_force = wp.zeros(capacity, dtype=float)
            self._empty_features = wp.empty(0, dtype=wp.vec3i)
            self._empty_body_q = wp.empty(0, dtype=wp.transform)

    def _features(self, contacts: Contacts) -> wp.array[wp.vec3i]:
        features = contacts._soft_contact_mesh_features
        if features is None or self._prev_features.shape[0] == 0:
            return self._empty_features
        return features

    def save(self, contacts: Contacts) -> None:
        """Save the buffer's anchors as its consumers left them, before a pass rewrites its records.

        Records another pipeline wrote are not this pipeline's history, so none are saved.
        """
        history = contacts._soft_contact_history
        self._prev_head.fill_(-1)
        wp.launch(
            _save_soft_contacts_kernel,
            dim=self._thread_count,
            inputs=[
                self._thread_count,
                self._owner,
                history.owner,
                contacts.soft_contact_count,
                self._capacity,
                contacts.soft_contact_indices,
                contacts.soft_contact_shape,
                self._features(contacts),
                history.reset_stamp,
                history.anchors.barycentric,
                history.anchors.body_pos,
                history.normal_body,
                history.anchors.normal_force,
            ],
            outputs=[
                self._prev_head,
                self._prev_next,
                self._prev_indices,
                self._prev_shape,
                self._prev_features,
                self._prev_reset_stamp,
                self._prev_barycentric,
                self._prev_body_pos,
                self._prev_normal_body,
                self._prev_normal_force,
            ],
            device=contacts.soft_contact_count.device,
        )

    def replay(self, model: Model, state: State, contacts: Contacts) -> None:
        """Anchor this pass's records, at their saved anchors where matched, and stamp every record."""
        history = contacts._soft_contact_history
        wp.launch(
            _replay_soft_contacts_kernel,
            dim=self._thread_count,
            inputs=[
                self._thread_count,
                self._resets.world_count,
                self._pos_threshold,
                state.particle_q,
                model.particle_radius,
                model.particle_world,
                state.body_q if state.body_q is not None else self._empty_body_q,
                model.shape_body,
                model.shape_world,
                model.shape_margin,
                contacts.soft_contact_count,
                self._capacity,
                contacts.soft_contact_indices,
                contacts.soft_contact_shape,
                self._features(contacts),
                contacts.soft_contact_barycentric,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                contacts.soft_contact_body_vel,
                self._resets.epochs,
                self._prev_head,
                self._prev_next,
                self._prev_indices,
                self._prev_shape,
                self._prev_features,
                self._prev_reset_stamp,
                self._prev_barycentric,
                self._prev_body_pos,
                self._prev_normal_body,
                self._prev_normal_force,
            ],
            outputs=[
                history.reset_stamp,
                history.normal_body,
                history.anchors.barycentric,
                history.anchors.body_pos,
                history.anchors.normal_force,
            ],
            device=contacts.soft_contact_count.device,
        )
        history.anchors.enabled.fill_(True)
