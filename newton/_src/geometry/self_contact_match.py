# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pass-to-pass sticky friction anchors for tri-mesh self-contacts.

A vertex-triangle contact is identified by its vertex row and triangle, an edge-edge contact by its
first edge's row and second edge. Each detected pair carries an anchor that describes material
points: for a vertex-triangle pair, the vertex position in the triangle's frame; for an edge-edge
pair, the closest points of the two edge lines and their separation. Friction measures slip as the
pair's drift from its anchor, which stays meaningful once a closest point reaches a triangle boundary
or an edge end.

Anchors live at the step-start positions the solver measures motion from, not at the detection
positions, which may be mid-step. A new pair anchors at its step-start position, so its slip is this
step's motion, exactly as without anchors. A pair found again keeps its anchor, dragged along to at
most the matching position threshold behind it; that much drift can build up between steps, where
kinematic particles move. The matcher saves the rows as the solver left them, so a solver may drag
the anchors between passes; VBD drags them to its friction regularization length after each step.
The fresh normal is kept, so only friction reads the anchor.

Call order (inside :class:`~newton.CollisionPipeline`)::

    matcher.save(...)  # before detection rewrites the rows
    matcher.replay(...)  # after vertex-triangle and edge-edge detection, or
    matcher.skip()  # when another detection replaces these rows before friction reads them
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..core.reset import WorldResetEpochs, world_reset_since
from .tri_mesh_collision import (
    NO_ANCHOR,
    TriMeshCollisionInfo,
    drag_edge_edge_anchor,
    drag_vertex_triangle_anchor,
    edge_edge_anchor,
    get_edge_colliding_edges_count,
    get_vertex_colliding_triangles_count,
    vertex_triangle_anchor,
)

if TYPE_CHECKING:
    from ..sim.contacts import Contacts
    from ..sim.model import Model


@wp.func
def _pair_reset_since(
    world_a: int, world_b: int, epochs: wp.array[wp.int32], cursor: wp.array[wp.int32], world_count: int
):
    """Return whether the world of either side of a pair has been reset since ``cursor`` caught up."""
    return world_reset_since(world_a, epochs, cursor, world_count) or world_reset_since(
        world_b, epochs, cursor, world_count
    )


@wp.kernel(enable_backward=False)
def _save_vertex_triangle_rows_kernel(
    rows_current: wp.array[wp.int32],
    collision_info: TriMeshCollisionInfo,
    prev_count: wp.array[wp.int32],
    prev_triangle: wp.array[wp.int32],
    prev_anchor: wp.array[wp.vec3],
):
    v, i = wp.tid()
    if rows_current[0] == 0:
        return
    count = get_vertex_colliding_triangles_count(collision_info, v)
    if i == 0:
        prev_count[v] = count
    if i < count:
        slot = collision_info.vertex_colliding_triangles_offsets[v] + i
        prev_triangle[slot] = collision_info.vertex_colliding_triangles[2 * slot + 1]
        prev_anchor[slot] = collision_info.vertex_colliding_triangles_anchor[slot]


@wp.kernel(enable_backward=False)
def _save_edge_edge_rows_kernel(
    rows_current: wp.array[wp.int32],
    collision_info: TriMeshCollisionInfo,
    prev_count: wp.array[wp.int32],
    prev_edge: wp.array[wp.int32],
    prev_anchor: wp.array[wp.vec2],
):
    e1, i = wp.tid()
    if rows_current[0] == 0:
        return
    count = get_edge_colliding_edges_count(collision_info, e1)
    if i == 0:
        prev_count[e1] = count
    if i < count:
        slot = collision_info.edge_colliding_edges_offsets[e1] + i
        prev_edge[slot] = collision_info.edge_colliding_edges[2 * slot + 1]
        prev_anchor[slot] = collision_info.edge_colliding_edges_anchor[slot]


@wp.kernel(enable_backward=False)
def _replay_vertex_triangle_anchors_kernel(
    world_count: int,
    pos_threshold: float,
    step_start_q: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    reset_epochs: wp.array[wp.int32],
    reset_cursor: wp.array[wp.int32],
    prev_count: wp.array[wp.int32],
    prev_triangle: wp.array[wp.int32],
    prev_anchor: wp.array[wp.vec3],
    collision_info: TriMeshCollisionInfo,
):
    v, i = wp.tid()
    if i >= get_vertex_colliding_triangles_count(collision_info, v):
        return
    offset = collision_info.vertex_colliding_triangles_offsets[v]
    slot = offset + i
    tri = collision_info.vertex_colliding_triangles[2 * slot + 1]
    if tri < 0:
        return
    p = step_start_q[v]
    a = step_start_q[tri_indices[tri, 0]]
    b = step_start_q[tri_indices[tri, 1]]
    c = step_start_q[tri_indices[tri, 2]]
    anchor = vertex_triangle_anchor(p, a, b, c)
    prev_n = prev_count[v]
    vertex_world = particle_world[v]
    triangle_world = particle_world[tri_indices[tri, 0]]
    if _pair_reset_since(vertex_world, triangle_world, reset_epochs, reset_cursor, world_count):
        prev_n = 0
    for j in range(offset, offset + prev_n):
        if prev_triangle[j] == tri:
            kept = prev_anchor[j]
            if kept[0] != NO_ANCHOR:
                anchor = drag_vertex_triangle_anchor(kept, p, a, b, c, pos_threshold)
            break
    collision_info.vertex_colliding_triangles_anchor[slot] = anchor


@wp.kernel(enable_backward=False)
def _replay_edge_edge_anchors_kernel(
    world_count: int,
    pos_threshold: float,
    parallel_epsilon: float,
    step_start_q: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    reset_epochs: wp.array[wp.int32],
    reset_cursor: wp.array[wp.int32],
    prev_count: wp.array[wp.int32],
    prev_edge: wp.array[wp.int32],
    prev_anchor: wp.array[wp.vec2],
    collision_info: TriMeshCollisionInfo,
):
    e1, i = wp.tid()
    if i >= get_edge_colliding_edges_count(collision_info, e1):
        return
    offset = collision_info.edge_colliding_edges_offsets[e1]
    slot = offset + i
    e2 = collision_info.edge_colliding_edges[2 * slot + 1]
    if e2 < 0:
        return
    p0 = step_start_q[edge_indices[e1, 2]]
    p1 = step_start_q[edge_indices[e1, 3]]
    q0 = step_start_q[edge_indices[e2, 2]]
    q1 = step_start_q[edge_indices[e2, 3]]
    anchor = edge_edge_anchor(p0, p1, q0, q1, parallel_epsilon)
    prev_n = prev_count[e1]
    e1_world = particle_world[edge_indices[e1, 2]]
    e2_world = particle_world[edge_indices[e2, 2]]
    if _pair_reset_since(e1_world, e2_world, reset_epochs, reset_cursor, world_count):
        prev_n = 0
    for j in range(offset, offset + prev_n):
        if prev_edge[j] == e2:
            anchor = drag_edge_edge_anchor(prev_anchor[j], p0, p1, q0, q1, parallel_epsilon, pos_threshold)
            break
    collision_info.edge_colliding_edges_anchor[slot] = anchor


class SelfContactMatcher:
    """Sticky friction anchors for tri-mesh self-contacts across detection passes.

    Internal helper owned by :class:`~newton.CollisionPipeline` in ``contact_matching="sticky"``
    mode once soft self-contact is configured. It owns the saved rows; all buffers are allocated at
    construction, so every call can be captured in a CUDA graph.

    Args:
        particle_count: Number of mesh vertices.
        edge_count: Number of mesh edges.
        vertex_buffer_size: Self-contact capacity per vertex row.
        edge_buffer_size: Self-contact capacity per edge row.
        resets: The pipeline's world reset epochs; each buffer keeps its own cursor on them.
        pos_threshold: Largest drift [m] a pair keeps from its anchor across passes.
        edge_edge_parallel_epsilon: Threshold the detector treats edge directions as parallel below.
        device: Device to allocate on.
    """

    def __init__(
        self,
        particle_count: int,
        edge_count: int,
        vertex_buffer_size: int,
        edge_buffer_size: int,
        resets: WorldResetEpochs,
        pos_threshold: float,
        edge_edge_parallel_epsilon: float,
        device=None,
    ):
        with wp.ScopedDevice(device):
            self._vertex_rows = (particle_count, vertex_buffer_size)
            self._edge_rows = (edge_count, edge_buffer_size)
            self._resets = resets
            self._pos_threshold = pos_threshold
            self._edge_edge_parallel_epsilon = edge_edge_parallel_epsilon
            self._prev_vertex_count = wp.zeros(particle_count, dtype=wp.int32)
            self._prev_triangle = wp.full(particle_count * vertex_buffer_size, -1, dtype=wp.int32)
            self._prev_triangle_anchor = wp.zeros(particle_count * vertex_buffer_size, dtype=wp.vec3)
            self._prev_edge_count = wp.zeros(edge_count, dtype=wp.int32)
            self._prev_edge = wp.full(edge_count * edge_buffer_size, -1, dtype=wp.int32)
            self._prev_edge_anchor = wp.zeros(edge_count * edge_buffer_size, dtype=wp.vec2)
            # Whether the last pass anchored its rows, so they are worth saving.
            self._rows_current = wp.zeros(1, dtype=wp.int32)

    def allocate_anchors(self, contacts: Contacts) -> None:
        """Give a buffer one friction anchor per self-contact pair slot and its own reset cursor."""
        collision_info = contacts.soft_self_contact_data
        device = self._rows_current.device
        collision_info.vertex_colliding_triangles_anchor = wp.zeros(
            self._prev_triangle.shape[0], dtype=wp.vec3, device=device
        )
        collision_info.edge_colliding_edges_anchor = wp.zeros(self._prev_edge.shape[0], dtype=wp.vec2, device=device)
        contacts._self_contact_reset_cursor = self._resets.cursor()

    def save(self, contacts: Contacts) -> None:
        """Save the buffer's rows and anchors as the solver left them, before detection rewrites them."""
        collision_info = contacts.soft_self_contact_data
        if (
            collision_info.vertex_colliding_triangles_anchor.shape != self._prev_triangle.shape
            or collision_info.edge_colliding_edges_anchor.shape != self._prev_edge.shape
        ):
            raise ValueError(
                "This Contacts buffer has no sticky self-contact anchors; allocate it with "
                "CollisionPipeline.contacts() after init_soft_self_contact()."
            )
        device = self._rows_current.device
        wp.launch(
            _save_vertex_triangle_rows_kernel,
            dim=self._vertex_rows,
            inputs=[self._rows_current, collision_info],
            outputs=[self._prev_vertex_count, self._prev_triangle, self._prev_triangle_anchor],
            device=device,
        )
        wp.launch(
            _save_edge_edge_rows_kernel,
            dim=self._edge_rows,
            inputs=[self._rows_current, collision_info],
            outputs=[self._prev_edge_count, self._prev_edge, self._prev_edge_anchor],
            device=device,
        )

    def replay(self, model: Model, step_start_q: wp.array[wp.vec3], contacts: Contacts) -> None:
        """Anchor this pass's pairs at ``step_start_q``, keeping matched pairs' saved anchors."""
        collision_info = contacts.soft_self_contact_data
        device = self._rows_current.device
        wp.launch(
            _replay_vertex_triangle_anchors_kernel,
            dim=self._vertex_rows,
            inputs=[
                self._resets.world_count,
                self._pos_threshold,
                step_start_q,
                model.particle_world,
                model.tri_indices,
                self._resets.epochs,
                contacts._self_contact_reset_cursor,
                self._prev_vertex_count,
                self._prev_triangle,
                self._prev_triangle_anchor,
            ],
            outputs=[collision_info],
            device=device,
        )
        wp.launch(
            _replay_edge_edge_anchors_kernel,
            dim=self._edge_rows,
            inputs=[
                self._resets.world_count,
                self._pos_threshold,
                self._edge_edge_parallel_epsilon,
                step_start_q,
                model.particle_world,
                model.edge_indices,
                self._resets.epochs,
                contacts._self_contact_reset_cursor,
                self._prev_edge_count,
                self._prev_edge,
                self._prev_edge_anchor,
            ],
            outputs=[collision_info],
            device=device,
        )
        self._resets.catch_up(contacts._self_contact_reset_cursor)
        self._rows_current.fill_(1)

    def skip(self) -> None:
        """Leave this pass's rows unanchored; the next save keeps the rows saved before it."""
        self._rows_current.fill_(0)
