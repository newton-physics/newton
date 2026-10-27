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

from ..core.reset import reset_world_selected
from .contact_match import _accumulate_reset_world_mask_kernel
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
    prev_anchor: wp.array[wp.vec3],
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
    reset_world_mask: wp.array[wp.bool],
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
    if reset_world_selected(particle_world[v], reset_world_mask, world_count):
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
    step_start_q: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    reset_world_mask: wp.array[wp.bool],
    prev_count: wp.array[wp.int32],
    prev_edge: wp.array[wp.int32],
    prev_anchor: wp.array[wp.vec3],
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
    anchor = edge_edge_anchor(p0, p1, q0, q1)
    prev_n = prev_count[e1]
    if reset_world_selected(particle_world[edge_indices[e1, 2]], reset_world_mask, world_count):
        prev_n = 0
    for j in range(offset, offset + prev_n):
        if prev_edge[j] == e2:
            kept = prev_anchor[j]
            if kept[0] != NO_ANCHOR and anchor[0] != NO_ANCHOR:
                anchor = drag_edge_edge_anchor(kept, p0, p1, q0, q1, pos_threshold)
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
        world_count: Number of local worlds.
        pos_threshold: Largest drift [m] a pair keeps from its anchor across passes.
        device: Device to allocate on.
    """

    def __init__(
        self,
        particle_count: int,
        edge_count: int,
        vertex_buffer_size: int,
        edge_buffer_size: int,
        world_count: int,
        pos_threshold: float,
        device=None,
    ):
        with wp.ScopedDevice(device):
            self._vertex_rows = (particle_count, vertex_buffer_size)
            self._edge_rows = (edge_count, edge_buffer_size)
            self._world_count = int(world_count)
            self._pos_threshold = pos_threshold
            self._prev_vertex_count = wp.zeros(particle_count, dtype=wp.int32)
            self._prev_triangle = wp.full(particle_count * vertex_buffer_size, -1, dtype=wp.int32)
            self._prev_triangle_anchor = wp.zeros(particle_count * vertex_buffer_size, dtype=wp.vec3)
            self._prev_edge_count = wp.zeros(edge_count, dtype=wp.int32)
            self._prev_edge = wp.full(edge_count * edge_buffer_size, -1, dtype=wp.int32)
            self._prev_edge_anchor = wp.zeros(edge_count * edge_buffer_size, dtype=wp.vec3)
            # Whether the last pass anchored its rows, so they are worth saving.
            self._rows_current = wp.zeros(1, dtype=wp.int32)
            self._reset_world_mask = wp.zeros(self._world_count + 1, dtype=wp.bool)

    def reset(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Forget all or reset-selected saved rows; selections accumulate until the next replay."""
        if world_mask is None:
            self._reset_world_mask.fill_(True)
            return
        wp.launch(
            _accumulate_reset_world_mask_kernel,
            dim=world_mask.shape[0],
            inputs=[world_mask, self._reset_world_mask],
            device=self._reset_world_mask.device,
        )

    def allocate_anchors(self, collision_info: TriMeshCollisionInfo) -> None:
        """Give a self-contact result buffer one friction anchor per pair slot."""
        device = self._rows_current.device
        collision_info.vertex_colliding_triangles_anchor = wp.zeros(
            self._prev_triangle.shape[0], dtype=wp.vec3, device=device
        )
        collision_info.edge_colliding_edges_anchor = wp.zeros(self._prev_edge.shape[0], dtype=wp.vec3, device=device)

    def save(self, contacts: Contacts) -> None:
        """Save the buffer's rows and anchors as the solver left them, before detection rewrites them."""
        collision_info = contacts.soft_self_contact_data
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
                self._world_count,
                self._pos_threshold,
                step_start_q,
                model.particle_world,
                model.tri_indices,
                self._reset_world_mask,
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
                self._world_count,
                self._pos_threshold,
                step_start_q,
                model.particle_world,
                model.edge_indices,
                self._reset_world_mask,
                self._prev_edge_count,
                self._prev_edge,
                self._prev_edge_anchor,
            ],
            outputs=[collision_info],
            device=device,
        )
        self._reset_world_mask.fill_(False)
        self._rows_current.fill_(1)

    def skip(self) -> None:
        """Leave this pass's rows unanchored; the next save keeps the rows saved before it."""
        self._rows_current.fill_(0)
