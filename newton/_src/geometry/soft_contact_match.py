# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame-to-frame sticky replay for rigid-soft contacts.

A soft contact record is identified by its soft feature (``soft_contact_indices``), its shape, and,
for full-surface mesh records, its mesh feature pair. Unlike rigid contacts, this key is exact, so
no sort or nearest search is needed: each record looks up the previous record with the same key
through a few short lists per first corner.

Sticky replay keeps a matched record's contact points where they were: the soft-side material point
(``soft_contact_barycentric``) and the body-frame surface point (``soft_contact_body_pos``). A record
keeps them while they still touch and have slipped apart tangentially by at most the matching
position threshold, and while the body point lies within that threshold of this pass's contact plane,
so a record that moves to an opposing wall of the same mesh starts over. Opposing body-frame normals
also restart a record, even when a thin wall's two faces fit within the position threshold. Records
on a moving surface (nonzero ``soft_contact_body_vel``) keep fresh points, since the surface material moves away from any
body-frame point. The fresh normal is kept, so a rotating body still pushes along its current surface
normal. Consumers can then measure tangential slip as the drift between the two replayed points, which
persists across collision passes, rather than as the motion since the last one.

The matcher saves a buffer's records as the consumers left them, so a solver may move the kept points
between passes; VBD drags them along once a contact slides. Each buffer therefore replays its own
records, and only those its pipeline wrote (``Contacts._soft_contact_history_owner``) since the last
reset of the record's worlds (``Contacts._soft_contact_reset_stamp``). Call order (inside
:class:`~newton.CollisionPipeline`)::

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
    # out: every record's reset stamp
    contact_reset_stamp: wp.array[wp.int32],
    contact_normal_body: wp.array[wp.vec3],
    # in/out: matched rows are overwritten with the previous contact points
    contact_barycentric: wp.array[wp.vec3],
    contact_body_pos: wp.array[wp.vec3],
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
            contact_barycentric[i] = prev_barycentric[best]
            contact_body_pos[i] = prev_body_pos[best]


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
    contact_barycentric: wp.array[wp.vec3],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal_body: wp.array[wp.vec3],
    prev_head: wp.array[wp.int32],
    prev_next: wp.array[wp.int32],
    prev_indices: wp.array[wp.vec3i],
    prev_shape: wp.array[wp.int32],
    prev_features: wp.array[wp.vec3i],
    prev_reset_stamp: wp.array[wp.int32],
    prev_barycentric: wp.array[wp.vec3],
    prev_body_pos: wp.array[wp.vec3],
    prev_normal_body: wp.array[wp.vec3],
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
        prev_barycentric[i] = contact_barycentric[i]
        prev_body_pos[i] = contact_body_pos[i]
        prev_normal_body[i] = contact_normal_body[i]
        slot = _key_slot(corners, contact_shape[i], _record_feature(contact_features, i))
        prev_next[i] = wp.atomic_exch(prev_head, slot, wp.int32(i))


class SoftContactMatcher:
    """Sticky replay of rigid-soft contact points across collision passes.

    Internal helper owned by :class:`~newton.CollisionPipeline` in ``contact_matching="sticky"``
    mode. It owns the saved records; all buffers are allocated at construction, so :meth:`save` and
    :meth:`replay` can be captured in a CUDA graph.

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
            self._empty_features = wp.empty(0, dtype=wp.vec3i)
            self._empty_body_q = wp.empty(0, dtype=wp.transform)

    def _features(self, contacts: Contacts) -> wp.array[wp.vec3i]:
        features = contacts._soft_contact_mesh_features
        if features is None or self._prev_features.shape[0] == 0:
            return self._empty_features
        return features

    def save(self, contacts: Contacts) -> None:
        """Save the buffer's records as its consumers left them, before a pass rewrites them.

        Records another pipeline wrote are not this pipeline's history, so none are saved.
        """
        self._prev_head.fill_(-1)
        wp.launch(
            _save_soft_contacts_kernel,
            dim=self._thread_count,
            inputs=[
                self._thread_count,
                self._owner,
                contacts._soft_contact_history_owner,
                contacts.soft_contact_count,
                self._capacity,
                contacts.soft_contact_indices,
                contacts.soft_contact_shape,
                self._features(contacts),
                contacts._soft_contact_reset_stamp,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_body_pos,
                contacts._soft_contact_normal_body,
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
            ],
            device=contacts.soft_contact_count.device,
        )

    def replay(self, model: Model, state: State, contacts: Contacts) -> None:
        """Replay the saved contact points onto this pass's matched records and stamp every record."""
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
            ],
            outputs=[
                contacts._soft_contact_reset_stamp,
                contacts._soft_contact_normal_body,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_body_pos,
            ],
            device=contacts.soft_contact_count.device,
        )
