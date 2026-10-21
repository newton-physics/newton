# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""On-demand camera speed adjustment near selected surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..geometry.kernels import sdf_box, sdf_capsule, sdf_cone, sdf_cylinder
from ..geometry.types import GeoType

if TYPE_CHECKING:
    from newton._src.viewer.viewer_workspace import ViewerWorkspace


@wp.func
def _distance(p: wp.vec3, kind: int, scale: wp.vec3, mesh: wp.uint64, lower: wp.vec3, upper: wp.vec3) -> float:
    if kind == GeoType.SPHERE:
        return wp.length(p) - scale[0]
    if kind == GeoType.BOX:
        return sdf_box(p, scale[0], scale[1], scale[2])
    if kind == GeoType.CAPSULE:
        return sdf_capsule(p, scale[0], scale[1], 2)
    if kind == GeoType.CYLINDER:
        return sdf_cylinder(p, scale[0], scale[1], 2)
    if kind == GeoType.CONE:
        return sdf_cone(p, scale[0], scale[1], 2)
    if kind == GeoType.PLANE:
        if scale[0] > 0.0 and scale[1] > 0.0:
            return wp.length(wp.vec3(wp.max(wp.abs(p[0]) - scale[0], 0.0), wp.max(wp.abs(p[1]) - scale[1], 0.0), p[2]))
        return wp.abs(p[2])
    if kind == GeoType.ELLIPSOID:
        # A conservative distance bound, unlike the approximate ellipsoid SDF.
        radii = wp.max(wp.abs(scale), wp.vec3(1.0e-9))
        return (wp.length(wp.cw_div(p, radii)) - 1.0) * wp.min(radii)
    if kind == GeoType.MESH or kind == GeoType.CONVEX_MESH or kind == GeoType.HFIELD:
        radii = wp.max(wp.abs(scale), wp.vec3(1.0e-9))
        point = wp.cw_div(p, scale)
        query = wp.mesh_query_point_no_sign(mesh, point, 1.0e12)
        if query.result:
            nearest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            # The smallest singular value keeps navigation slow near nonuniform or
            # mirrored instances. Unsigned distance also handles open sheets.
            return wp.length(point - nearest) * wp.min(radii)
        return 1.0e12
    half = (upper - lower) * 0.5
    return sdf_box(p - (upper + lower) * 0.5, half[0], half[1], half[2])


@wp.kernel
def _selection_distance(
    indices: wp.array[int],
    types: wp.array[int],
    scales: wp.array[wp.vec3],
    meshes: wp.array[wp.uint64],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    parents: wp.array[int],
    local: wp.array[wp.transform],
    bodies: wp.array[wp.transform],
    worlds: wp.array[int],
    offsets: wp.array[wp.vec3],
    layer: wp.transform,
    start: wp.vec3,
    result: wp.array[float],
) -> None:
    shape = indices[wp.tid()]
    pose = local[shape]
    if parents[shape] >= 0:
        pose = bodies[parents[shape]] * pose
    offset = wp.vec3(0.0)
    world = worlds[shape]
    if world >= 0 and world < offsets.shape[0]:
        offset = offsets[world]
    origin = wp.transform_point(
        wp.transform_inverse(pose), wp.transform_point(wp.transform_inverse(layer), start) - offset
    )
    distance = _distance(origin, types[shape], scales[shape], meshes[shape], lower[shape], upper[shape])
    wp.atomic_min(result, 0, wp.abs(distance))


@wp.kernel
def _surface_distance(mesh: wp.uint64, start: wp.vec3, result: wp.array[float]) -> None:
    distance = _distance(start, int(GeoType.MESH), wp.vec3(1.0), mesh, wp.vec3(0.0), wp.vec3(0.0))
    wp.atomic_min(result, 0, wp.abs(distance))


@wp.kernel
def _particle_distance(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    radii: wp.array[float],
    start: wp.vec3,
    result: wp.array[float],
) -> None:
    i = wp.tid()
    wp.atomic_min(result, 0, wp.abs(wp.length(start - points[i]) - radii[indices[i]]))


class SelectionNavigation:
    """Reuse device buffers; read one reduced distance only during navigation."""

    def __init__(self, workspace: ViewerWorkspace) -> None:
        self.workspace = workspace
        self.key = None

    def __call__(self, start: wp.vec3, end: wp.vec3, *, reference: float = 0.0) -> float:
        ws = self.workspace
        strength = max(0.0, float(ws.viewer.camera_slowdown))
        model, state, shapes = ws.model, ws.viewer._last_state, ws._selection_shapes
        deformable = getattr(ws, "_deformable_selection", None)
        if strength == 0 or reference <= 0 or model is None or state is None or (not shapes and deformable is None):
            return 1.0
        key = (id(model), shapes)
        if self.key != key:
            self.indices = wp.array(shapes, dtype=int, device=model.device)
            self.result = wp.empty(1, dtype=float, device=model.device)
            self.empty_offsets = wp.empty(0, dtype=wp.vec3, device=model.device)
            self.key = key
        self.result.fill_(1.0e12)
        if shapes:
            wp.launch(
                _selection_distance,
                dim=len(shapes),
                inputs=[
                    self.indices,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_source_ptr,
                    model.shape_collision_aabb_lower,
                    model.shape_collision_aabb_upper,
                    model.shape_body,
                    model.shape_transform,
                    state.body_q,
                    model.shape_world,
                    ws.viewer.world_offsets if ws.viewer.world_offsets is not None else self.empty_offsets,
                    ws.viewer.layer.xform,
                    wp.vec3(*start),
                ],
                outputs=[self.result],
                device=model.device,
            )
        if deformable is not None:
            if deformable.mesh is not None:
                if getattr(self, "surface", None) is not deformable:
                    self.surface = deformable
                    self.surface_mesh = wp.Mesh(points=deformable.points, indices=deformable.faces)
                    self.surface_revision = deformable.revision
                elif self.surface_revision != deformable.revision:
                    self.surface_mesh.refit()
                    self.surface_revision = deformable.revision
                wp.launch(
                    _surface_distance,
                    dim=1,
                    inputs=[self.surface_mesh.id, wp.vec3(*start)],
                    outputs=[self.result],
                    device=model.device,
                )
            for source, indices, points in deformable.batches:
                wp.launch(
                    _particle_distance,
                    dim=len(indices),
                    inputs=[points, indices, source.model.particle_radius, wp.vec3(*start)],
                    outputs=[self.result],
                    device=model.device,
                )
        distance = float(self.result.numpy()[0])
        # Slow down on either side of a surface, with a nonzero floor so the
        # camera can cross it and recover speed inside or beyond the object.
        proximity = min(1.0, max(0.0, distance / reference))
        return max(0.02 / max(1.0, strength), proximity**strength)
