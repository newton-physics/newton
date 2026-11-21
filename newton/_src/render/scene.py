# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The scene a camera traverses: shapes, standalone particles and deformable triangles as one set of items.

Items are numbered shapes first, then particles, then triangles (see :class:`~.types.SceneData`), so
one traversal finds the closest hit among all of them and the item range tells what was hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..geometry.bvh import compute_bvh_group_roots
from ..sim import Model, State
from .types import SceneData


@dataclass
class SceneItems:
    """The non-shape items of a camera's scene, shared by both backends.

    Built when the camera is constructed; the triangle points are synced from the state each frame.
    """

    item_particles: wp.array[wp.int32]
    """Standalone particles, which are neither on a deformable surface nor in a tetrahedron."""
    triangle_points: wp.array[wp.vec3f] | None
    """Points of the deformable triangles, or ``None`` without triangles."""
    triangle_indices: wp.array[wp.int32] | None
    """Vertex triplets of the deformable triangles."""
    triangle_point_worlds: wp.array[wp.int32] | None
    """World of each triangle point, -1 for the global world."""
    triangle_worlds: wp.array[wp.int32] | None
    """World of each triangle (that of its first point), -1 for the global world."""

    @property
    def particle_count(self) -> int:
        return self.item_particles.shape[0]

    @property
    def triangle_count(self) -> int:
        return 0 if self.triangle_indices is None else self.triangle_indices.shape[0] // 3


def empty_scene_data(device: wp.Device) -> SceneData:
    """Scene data with every array empty, to fill in per backend."""
    scene = SceneData()
    scene.id = wp.uint64(0)
    empty_int = wp.empty(0, dtype=wp.int32, device=device)
    scene.world_roots = empty_int
    scene.shape_item_count = 0
    scene.item_shapes = wp.empty(0, dtype=wp.uint32, device=device)
    scene.particle_item_count = 0
    scene.item_particles = empty_int
    scene.particle_positions = wp.empty(0, dtype=wp.vec3f, device=device)
    scene.particle_radii = wp.empty(0, dtype=wp.float32, device=device)
    scene.triangle_points = wp.empty(0, dtype=wp.vec3f, device=device)
    scene.triangle_indices = empty_int
    scene.cluster_triangle_start = empty_int
    scene.cluster_triangles = empty_int
    scene.slot_gaussians = wp.empty((0, 0), dtype=wp.int32, device=device)
    return scene


@wp.kernel(enable_backward=False)
def compute_scene_item_bounds(
    shape_count: wp.int32,
    shape_lowers: wp.array[wp.vec3f],
    shape_uppers: wp.array[wp.vec3f],
    shape_groups: wp.array[wp.int32],
    particle_count: wp.int32,
    item_particles: wp.array[wp.int32],
    particle_positions: wp.array[wp.vec3f],
    particle_radii: wp.array[wp.float32],
    particle_worlds: wp.array[wp.int32],
    triangle_points: wp.array[wp.vec3f],
    triangle_indices: wp.array[wp.int32],
    triangle_worlds: wp.array[wp.int32],
    world_count: wp.int32,
    out_lowers: wp.array[wp.vec3f],
    out_uppers: wp.array[wp.vec3f],
    out_groups: wp.array[wp.int32],
):
    item = wp.tid()
    if item < shape_count:
        out_lowers[item] = shape_lowers[item]
        out_uppers[item] = shape_uppers[item]
        out_groups[item] = shape_groups[item]
        return
    if item < shape_count + particle_count:
        particle = item_particles[item - shape_count]
        center = particle_positions[particle]
        radius = wp.vec3f(particle_radii[particle])
        out_lowers[item] = center - radius
        out_uppers[item] = center + radius
        world = particle_worlds[particle]
    else:
        face = item - shape_count - particle_count
        p0 = triangle_points[triangle_indices[face * 3 + 0]]
        p1 = triangle_points[triangle_indices[face * 3 + 1]]
        p2 = triangle_points[triangle_indices[face * 3 + 2]]
        lower = wp.min(wp.min(p0, p1), p2)
        upper = wp.max(wp.max(p0, p1), p2)
        # Flat, axis-aligned faces have zero-thickness bounds that ray-box tests can miss.
        pad = wp.vec3f(1.0e-6 * (1.0 + wp.max(wp.max(wp.abs(lower)), wp.max(wp.abs(upper)))))
        out_lowers[item] = lower - pad
        out_uppers[item] = upper + pad
        world = triangle_worlds[face]
    out_groups[item] = wp.where(world < 0, world_count, world)


class WarpScene:
    """A camera's Warp BVH over its scene items, grouped by world.

    Shape items are the leaves of :attr:`Model.bvh_shapes`, with the bounds of its last refit, so
    the model's shape BVH must be refit before :meth:`refit`. Cameras without particles or
    deformable triangles traverse the model's shape BVH directly and need no scene of their own.
    """

    def __init__(self, model: Model, items: SceneItems):
        self.model = model
        self.items = items
        self.bvh: wp.Bvh | None = None
        self.world_roots = wp.empty(model.world_count + 1, dtype=wp.int32, device=model.device)
        self._shape_bvh: wp.Bvh | None = None
        self._lowers = self._uppers = self._groups = None

    def _shape_items(self) -> tuple[int, wp.array, wp.array, wp.array]:
        model = self.model
        if model.bvh_shapes is None or model.bvh_shape_count_enabled == 0:
            empty = wp.empty(0, dtype=wp.vec3f, device=model.device)
            return 0, empty, empty, wp.empty(0, dtype=wp.int32, device=model.device)
        bvh = model.bvh_shapes
        return model.bvh_shape_count_enabled, bvh.lowers, bvh.uppers, bvh.groups

    def refit(self, state: State) -> None:
        """Recompute item bounds from the model's shape BVH and *state*, and refit the BVH.

        Builds the BVH on first use and again after the model's shape BVH was rebuilt.
        """
        model, items = self.model, self.items
        shape_count, shape_lowers, shape_uppers, shape_groups = self._shape_items()
        count = shape_count + items.particle_count + items.triangle_count
        rebuild = self.bvh is None or self._shape_bvh is not model.bvh_shapes
        if rebuild:
            self._lowers = wp.empty(count, dtype=wp.vec3f, device=model.device)
            self._uppers = wp.empty(count, dtype=wp.vec3f, device=model.device)
            self._groups = wp.empty(count, dtype=wp.int32, device=model.device)
        empty_vec = wp.empty(0, dtype=wp.vec3f, device=model.device)
        empty_int = wp.empty(0, dtype=wp.int32, device=model.device)
        has_particles = items.particle_count > 0
        wp.launch(
            compute_scene_item_bounds,
            dim=count,
            inputs=[
                shape_count,
                shape_lowers,
                shape_uppers,
                shape_groups,
                items.particle_count,
                items.item_particles,
                state.particle_q if has_particles else empty_vec,
                model.particle_radius if has_particles else wp.empty(0, dtype=wp.float32, device=model.device),
                model.particle_world if has_particles else empty_int,
                items.triangle_points if items.triangle_count else empty_vec,
                items.triangle_indices if items.triangle_count else empty_int,
                items.triangle_worlds if items.triangle_count else empty_int,
                model.world_count,
            ],
            outputs=[self._lowers, self._uppers, self._groups],
            device=model.device,
        )
        if rebuild:
            # SAH matches the quality of the deformable mesh and particle BVHs this replaces.
            self.bvh = wp.Bvh(self._lowers, self._uppers, constructor="sah", groups=self._groups)
            self._shape_bvh = model.bvh_shapes
            wp.launch(
                compute_bvh_group_roots,
                dim=model.world_count + 1,
                inputs=[self.bvh.id, self.world_roots],
                device=model.device,
            )
        else:
            self.bvh.refit()

    def scene_data(self, state: State) -> SceneData:
        model, items = self.model, self.items
        scene = empty_scene_data(model.device)
        scene.id = wp.uint64(self.bvh.id)
        scene.world_roots = self.world_roots
        scene.shape_item_count = self._shape_items()[0]
        if scene.shape_item_count:
            scene.item_shapes = model._bvh_shape_active_leaves
        fill_item_data(scene, model, items, state)
        return scene


def fill_item_data(scene: SceneData, model: Model, items: SceneItems, state: State) -> None:
    """Fill the particle and triangle fields of *scene*."""
    if items.particle_count:
        scene.particle_item_count = items.particle_count
        scene.item_particles = items.item_particles
        scene.particle_positions = state.particle_q
        scene.particle_radii = model.particle_radius
    if items.triangle_count:
        scene.triangle_points = items.triangle_points
        scene.triangle_indices = items.triangle_indices


def model_shape_scene_data(model: Model) -> SceneData:
    """Scene data tracing the model's shape BVH directly, for cameras whose scene holds only shapes."""
    scene = empty_scene_data(model.device)
    if model.bvh_shape_count_enabled > 0 and model.bvh_shapes is not None:
        scene.id = wp.uint64(model.bvh_shapes.id)
        scene.world_roots = model.bvh_shapes_group_roots
        scene.shape_item_count = model.bvh_shape_count_enabled
        scene.item_shapes = model._bvh_shape_active_leaves
    return scene


def standalone_particles(model: Model, topology_particle_mask: np.ndarray | None) -> np.ndarray:
    """Indices of the particles rendered as spheres."""
    if model.particle_count == 0:
        return np.zeros(0, dtype=np.int32)
    if topology_particle_mask is None:
        return np.arange(model.particle_count, dtype=np.int32)
    return np.flatnonzero(~topology_particle_mask).astype(np.int32)
