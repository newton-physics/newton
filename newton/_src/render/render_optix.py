# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A camera's OptiX scene: one instance structure over shapes, standalone particles and deformable triangles.

The shape instances are those of :attr:`Model.optix_shapes`. Every world's particles share one
built-in sphere structure and every world's deformable triangles share one triangle structure,
with their points moved into their world's cell, so each adds a single instance and the camera
traces everything in one traversal. Triangles are rebuilt as clusters each frame when the installed
``warp-optix`` supports them, and refit otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..geometry import Gaussian, GeoType, raycast
from ..geometry.optix import create_pipeline, ray_interval, wo, world_offset
from . import raytrace
from .scene import SceneItems, empty_scene_data, fill_item_data
from .types import MeshData, RenderConfig, RenderState, SceneData

if TYPE_CHECKING:
    from ..geometry.optix import OptixShapes
    from ..sim import Model, State

TRIANGLE_INSTANCE_ID = 0xFFFFFF
"""Instance ID of the deformable triangles."""
PARTICLE_INSTANCE_ID = 0xFFFFFE
"""Instance ID of the standalone particles."""
PARTICLE_MASK = 2
"""Visibility mask bit of the particle instance, so renders without particles skip it."""
PARTICLE_BUILD_FLAGS = wo.BuildFlags.ALLOW_UPDATE | wo.BuildFlags.PREFER_FAST_BUILD
"""Build flags of the particle spheres, which their built-in intersection is compiled for."""
CLUSTER_TRIANGLES = 128
"""Triangles per cluster; at most 3 * 128 vertices, which never exceeds the 256-vertex limit after a split."""

_RAY_MASK_ALL = 0xFF
_RAY_MASK_NO_PARTICLES = 0xFF & ~PARTICLE_MASK


def clusters_supported(device: wp.Device) -> bool:
    """Whether the installed ``warp-optix`` and the device support triangle clusters."""
    if not hasattr(wo, "TriangleClusterAccel"):
        return False
    try:
        return bool(wo.cluster_capabilities(device)["flags"])
    except Exception:
        return False


def cluster_triangles(
    indices: np.ndarray, points: np.ndarray, worlds: np.ndarray, max_triangles: int = CLUSTER_TRIANGLES
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition triangles into clusters of nearby triangles that never span worlds.

    Triangles are ordered by world, then along a Morton curve of their centroids, and cut into runs
    of ``max_triangles``. Runs referencing more than 256 points are halved until they fit.

    Returns:
        ``(clusters, local_indices, source_points, faces)``: one ``(first_point, point_count,
        first_triangle, triangle_count)`` row per cluster; cluster-local vertex triplets; the
        triangle point of each cluster point; and the face of each clustered triangle.
    """
    tris = indices.reshape(-1, 3).astype(np.int64)
    centroids = points[tris].mean(axis=1)
    lo, hi = centroids.min(axis=0), centroids.max(axis=0)
    cells = ((centroids - lo) / np.maximum(hi - lo, 1.0e-9) * 1023.0).astype(np.uint64)

    def spread(x):
        x = (x | (x << np.uint64(16))) & np.uint64(0x030000FF)
        x = (x | (x << np.uint64(8))) & np.uint64(0x0300F00F)
        x = (x | (x << np.uint64(4))) & np.uint64(0x030C30C3)
        return (x | (x << np.uint64(2))) & np.uint64(0x09249249)

    morton = spread(cells[:, 0]) | (spread(cells[:, 1]) << np.uint64(1)) | (spread(cells[:, 2]) << np.uint64(2))
    faces = np.lexsort((morton, worlds)).astype(np.int64)
    sorted_worlds = worlds[faces]
    # Runs restart at each world.
    world_start = np.flatnonzero(np.r_[True, sorted_worlds[1:] != sorted_worlds[:-1]])
    rank = np.arange(len(faces)) - np.repeat(world_start, np.diff(np.r_[world_start, len(faces)]))
    run = np.cumsum(np.r_[True, (rank[1:] % max_triangles == 0) | (sorted_worlds[1:] != sorted_worlds[:-1])]) - 1

    cluster_rows, local, sources, ordered = [], [], [], []
    point_count = triangle_count = 0
    run_start = np.flatnonzero(np.r_[True, run[1:] != run[:-1]])
    pending = [(int(a), int(b)) for a, b in zip(run_start, np.r_[run_start[1:], len(faces)], strict=True)]
    while pending:
        start, end = pending.pop()
        run_faces = faces[start:end]
        unique, inverse = np.unique(tris[run_faces], return_inverse=True)
        if len(unique) > 256:
            middle = (start + end) // 2
            pending += [(middle, end), (start, middle)]
            continue
        cluster_rows.append((point_count, len(unique), triangle_count, len(run_faces)))
        local.append(inverse.reshape(-1))
        sources.append(unique)
        ordered.append(run_faces)
        point_count += len(unique)
        triangle_count += len(run_faces)
    return (
        np.asarray(cluster_rows, dtype=np.int32),
        np.concatenate(local).astype(np.int32),
        np.concatenate(sources).astype(np.int32),
        np.concatenate(ordered).astype(np.int32),
    )


@wp.kernel(enable_backward=False)
def pack_triangle_points(
    grid: wp.int32,
    extent: wp.float32,
    world_count: wp.int32,
    sources: wp.array[wp.int32],
    points: wp.array[wp.vec3f],
    point_worlds: wp.array[wp.int32],
    out_points: wp.array[wp.vec3f],
    out_of_cell: wp.array[wp.int32],
):
    """Move triangle points into their world's cell; with ``sources``, gather them into cluster order."""
    i = wp.tid()
    source = i
    if sources.shape[0] > 0:
        source = sources[i]
    p = points[source]
    if wp.max(wp.abs(p)) > extent:
        out_of_cell[0] = 1
    world = point_worlds[source]
    out_points[i] = p + world_offset(wp.where(world < 0, world_count, world), grid, extent)


@wp.kernel(enable_backward=False)
def pack_particles(
    grid: wp.int32,
    extent: wp.float32,
    world_count: wp.int32,
    item_particles: wp.array[wp.int32],
    particle_worlds: wp.array[wp.int32],
    particle_positions: wp.array[wp.vec3f],
    particle_radii: wp.array[wp.float32],
    out_centers: wp.array[wp.vec3f],
    out_radii: wp.array[wp.float32],
    out_of_cell: wp.array[wp.int32],
):
    """Move each particle sphere into its world's cell."""
    i = wp.tid()
    particle = item_particles[i]
    p = particle_positions[particle]
    radius = particle_radii[particle]
    if wp.max(wp.abs(p)) + radius > extent:
        out_of_cell[0] = 1
    world = particle_worlds[particle]
    out_centers[i] = p + world_offset(wp.where(world < 0, world_count, world), grid, extent)
    out_radii[i] = radius


@wp.kernel(enable_backward=False)
def write_scene_instance(
    index: wp.int32,
    instance_id: wp.int32,
    traversable: wp.uint64,
    sbt_offset: wp.int32,
    mask: wp.int32,
    records: wp.array[wo.Instance],
):
    records[index] = wo.make_instance(
        wp.transform_identity(),
        traversable,
        instance_id,
        sbt_offset,
        mask,
        wp.static(int(wo.InstanceFlags.DISABLE_ANYHIT)),
    )


class OptixScene:
    """A camera's OptiX instance structure over :attr:`Model.optix_shapes`, particles and deformable triangles.

    Without particles or triangles, it traces the model's instance structure directly. Otherwise it
    owns an instance structure whose first records are copies of the model's shape instances, so
    :meth:`Model.bvh_refit_shapes` with ``backend="optix"`` must run before :meth:`refit`.
    """

    def __init__(
        self, model: Model, shapes: OptixShapes, items: SceneItems, state: State, *, use_clusters: bool | None = None
    ):
        self.model = model
        self.shapes = shapes
        self.items = items
        self.pipelines: dict = {}
        device = model.device
        self.has_particles = items.particle_count > 0
        self.has_triangles = items.triangle_count > 0
        if use_clusters is None:
            use_clusters = self.has_triangles and clusters_supported(device)
        self.clustered = bool(use_clusters and self.has_triangles)
        self.has_global_items = shapes.has_global_instances
        self.accel: wo.Accel | None = None
        self.triangle_accel = None
        self.particle_accel: wo.Accel | None = None
        self._empty_int = wp.empty(0, dtype=wp.int32, device=device)
        if not (self.has_particles or self.has_triangles):
            return

        children = list(shapes.geometry)
        extra = []
        if self.has_triangles:
            triangle_worlds = items.triangle_worlds.numpy()
            self.has_global_items |= bool(np.any(triangle_worlds < 0))
            if self.clustered:
                clusters, local, sources, faces = cluster_triangles(
                    items.triangle_indices.numpy(), items.triangle_points.numpy(), triangle_worlds
                )
                self._point_sources = wp.array(sources, dtype=wp.int32, device=device)
                self._packed_points = wp.empty(len(sources), dtype=wp.vec3f, device=device)
                self._local_indices = wp.array(local, dtype=wp.int32, device=device)
                self._clusters = wp.array(clusters, dtype=wp.vec4i, device=device)
                self.cluster_triangle_start = wp.array(clusters[:, 2], dtype=wp.int32, device=device)
                self.cluster_triangles = wp.array(faces, dtype=wp.int32, device=device)
            else:
                self._point_sources = self._empty_int
                self._packed_points = wp.empty(items.triangle_points.shape[0], dtype=wp.vec3f, device=device)
            self._pack_points()
            if self.clustered:
                self.triangle_accel = wo.TriangleClusterAccel(
                    self._packed_points, self._local_indices, self._clusters, validate=False
                )
            else:
                self.triangle_accel = wo.Accel(
                    [wo.triangles(self._packed_points, items.triangle_indices, validate=False)],
                    flags=wo.BuildFlags.ALLOW_UPDATE | wo.BuildFlags.PREFER_FAST_TRACE,
                )
            children.append(self.triangle_accel)
            extra.append((TRIANGLE_INSTANCE_ID, self.triangle_accel.id, 0, 1))
        if self.has_particles:
            particle_worlds = model.particle_world
            self.has_global_items |= bool(np.any(particle_worlds.numpy()[items.item_particles.numpy()] < 0))
            self._particle_centers = wp.empty(items.particle_count, dtype=wp.vec3f, device=device)
            self._particle_radii = wp.empty(items.particle_count, dtype=wp.float32, device=device)
            self._pack_particles(state.particle_q)
            self.particle_accel = wo.Accel(
                [wo.spheres(self._particle_centers, self._particle_radii)], flags=PARTICLE_BUILD_FLAGS
            )
            children.append(self.particle_accel)
            extra.append((PARTICLE_INSTANCE_ID, self.particle_accel.id, 2, PARTICLE_MASK))

        shape_instances = shapes.instance_shapes.shape[0]
        self.records = wp.empty(shape_instances + len(extra), dtype=wo.Instance, device=device)
        if shape_instances:
            wp.copy(self.records, shapes.instances, count=shape_instances)
        for k, (instance_id, traversable, sbt_offset, mask) in enumerate(extra):
            wp.launch(
                write_scene_instance,
                dim=1,
                inputs=[shape_instances + k, instance_id, wp.uint64(traversable), sbt_offset, mask, self.records],
                device=device,
            )
        self.accel = wo.Accel(
            [wo.instances(self.records, children)],
            flags=wo.BuildFlags.ALLOW_UPDATE | wo.BuildFlags.PREFER_FAST_BUILD,
        )

    @property
    def id(self) -> int:
        if self.accel is not None:
            return self.accel.id
        return self.shapes.id

    @property
    def has_instances(self) -> bool:
        return self.accel is not None or self.shapes.has_instances

    def _pack_points(self) -> None:
        shapes = self.shapes
        wp.launch(
            pack_triangle_points,
            dim=self._packed_points.shape[0],
            inputs=[
                shapes.grid,
                shapes.extent,
                shapes.world_count,
                self._point_sources,
                self.items.triangle_points,
                self.items.triangle_point_worlds,
            ],
            outputs=[self._packed_points, shapes.out_of_cell],
            device=self.model.device,
        )

    def _pack_particles(self, particle_q: wp.array[wp.vec3f]) -> None:
        shapes = self.shapes
        wp.launch(
            pack_particles,
            dim=self.items.particle_count,
            inputs=[
                shapes.grid,
                shapes.extent,
                shapes.world_count,
                self.items.item_particles,
                self.model.particle_world,
                particle_q,
                self.model.particle_radius,
            ],
            outputs=[self._particle_centers, self._particle_radii, shapes.out_of_cell],
            device=self.model.device,
        )

    def refit(self, state: State) -> None:
        """Update the particles and triangles from *state* and the camera's triangle points, and refit."""
        if self.accel is None:
            return
        shape_instances = self.shapes.instance_shapes.shape[0]
        if shape_instances:
            wp.copy(self.records, self.shapes.instances, count=shape_instances)
        if self.has_triangles:
            self._pack_points()
            if self.clustered:
                self.triangle_accel.rebuild()
            else:
                self.triangle_accel.refit()
        if self.has_particles:
            self._pack_particles(state.particle_q)
            self.particle_accel.refit()
        self.accel.refit()

    def scene_data(self, state: State) -> SceneData:
        scene = empty_scene_data(self.model.device)
        scene.id = wp.uint64(self.id if self.has_instances else 0)
        scene.world_roots = self.shapes.world_planes.flatten()
        fill_item_data(scene, self.model, self.items, state)
        if self.clustered:
            scene.cluster_triangle_start = self.cluster_triangle_start
            scene.cluster_triangles = self.cluster_triangles
        return scene

    def create_pipeline(self, kernel: wp.Kernel, config: RenderConfig) -> wo.Pipeline:
        return create_pipeline(
            self.shapes,
            kernel,
            enable_backface_culling=config.enable_backface_culling,
            particle_build_flags=PARTICLE_BUILD_FLAGS if self.has_particles else None,
            clustered_geometry=self.clustered,
        )

    def _traversal_count(self, config: RenderConfig) -> int:
        return 2 if config.enable_global_world and self.has_global_items else 1

    def create_closest_hit(self, config: RenderConfig, state: RenderState, with_normal: bool) -> wp.Function:
        shapes = self.shapes
        grid, extent, world_count = shapes.grid, shapes.extent, shapes.world_count
        planes_per_world = shapes.world_planes.shape[1]
        has_particles = self.has_particles
        has_triangles = self.has_triangles
        clustered = self.clustered
        ray_mask = _RAY_MASK_ALL if config.enable_particles else _RAY_MASK_NO_PARTICLES
        traversals = self._traversal_count(config)
        has_instances = self.has_instances

        @wp.func
        def plane_hit(
            hit: raytrace.ClosestHit,
            si: int,
            shape_sizes: wp.array[wp.vec3f],
            shape_transforms: wp.array[wp.transformf],
            origin: wp.vec3f,
            direction: wp.vec3f,
        ) -> raytrace.ClosestHit:
            distance, normal = raycast.ray_intersect_shape(
                shape_transforms[si],
                shape_sizes[si],
                GeoType.PLANE,
                origin,
                direction,
                wp.static(config.enable_backface_culling),
            )
            if distance >= 0.0 and distance < hit.distance:
                hit.distance = distance
                hit.normal = normal
                hit.shape_index = wp.uint32(si)
                hit.bary_u = 0.0
                hit.bary_v = 0.0
                hit.face_idx = -1
            return hit

        @wp.func
        def closest_hit(
            scene: SceneData,
            world_index: wp.int32,
            max_distance: wp.float32,
            shape_types: wp.array[wp.int32],
            shape_sizes: wp.array[wp.vec3f],
            shape_transforms: wp.array[wp.transformf],
            shape_source_ptr: wp.array[wp.uint64],
            shape_mesh_data_ids: wp.array[wp.int32],
            mesh_data: wp.array[MeshData],
            gaussians_data: wp.array[Gaussian.Data],
            origin: wp.vec3f,
            direction: wp.vec3f,
            camera_forward: wp.vec3f,
        ) -> raytrace.ClosestHit:
            hit = raytrace.ClosestHit()
            hit.distance = max_distance
            hit.shape_index = raytrace.NO_HIT_SHAPE_ID
            hit.color = wp.vec3f(0.0)
            # Planes first: a nearer hit shortens the traversals.
            if wp.static(config.enable_global_world):
                for p in range(wp.static(len(shapes.planes))):
                    hit = plane_hit(hit, wp.static(shapes.planes[p]), shape_sizes, shape_transforms, origin, direction)
            for p in range(wp.static(planes_per_world)):
                si = scene.world_roots[world_index * wp.static(planes_per_world) + p]
                if si >= 0:
                    hit = plane_hit(hit, si, shape_sizes, shape_transforms, origin, direction)
            # Unrolled: dynamic traversal loops miscompile with warp-optix 0.3.0.
            for i in range(wp.static(traversals)):
                world = wp.where(i == 1, wp.static(world_count), world_index)
                interval = ray_interval(origin, direction, wp.static(extent), hit.distance)
                if wp.static(has_instances) and interval[1] >= interval[0]:
                    packed_origin = origin + world_offset(world, wp.static(grid), wp.static(extent))
                    flags = wp.uint32(0)
                    if wp.static(config.enable_backface_culling):
                        flags = wp.uint32(wp.static(int(wo.RayFlags.CULL_BACK_FACING_TRIANGLES)))
                    query = wo.accel_query_ray(
                        scene.id, packed_origin, direction, interval[1], interval[0], wp.static(ray_mask), flags
                    )
                    if query.result and query.t < hit.distance:
                        instance = int(query.instance_id)
                        hit.distance = query.t
                        if wp.static(has_triangles) and instance == wp.static(TRIANGLE_INSTANCE_ID):
                            face = int(query.primitive_index)
                            if wp.static(clustered):
                                face = scene.cluster_triangles[
                                    scene.cluster_triangle_start[int(query.cluster_id)] + face
                                ]
                            hit.shape_index = raytrace.TRIANGLE_MESH_SHAPE_ID
                            # Albedo-only renders also need the face when textures are disabled.
                            hit.face_idx = face
                            if wp.static(with_normal):
                                hit.normal = raytrace.triangle_face_normal(scene, face, direction)
                                hit.bary_u = 1.0 - query.u - query.v
                                hit.bary_v = query.u
                        elif wp.static(has_particles) and instance == wp.static(PARTICLE_INSTANCE_ID):
                            particle = scene.item_particles[int(query.primitive_index)]
                            hit.shape_index = raytrace.PARTICLES_SHAPE_ID
                            hit.face_idx = particle
                            if wp.static(with_normal):
                                hit.normal = wp.normalize(
                                    origin + direction * query.t - scene.particle_positions[particle]
                                )
                        else:
                            si = instance
                            hit.shape_index = wp.uint32(si)
                            if shape_types[si] == GeoType.MESH:
                                mesh = shape_source_ptr[si]
                                face = int(query.primitive_index)
                                u = 1.0 - query.u - query.v
                                v = query.u
                                hit.bary_u = u
                                hit.bary_v = v
                                hit.face_idx = face
                                if wp.static(with_normal):
                                    mesh_data_id = shape_mesh_data_ids[si]
                                    smooth = False
                                    if mesh_data_id > -1:
                                        smooth = mesh_data[mesh_data_id].normals.shape[0] > 0
                                    # Vertex normals replace the face normal, so only compute it without them.
                                    normal = wp.vec3f(0.0)
                                    if not smooth:
                                        normal = raycast.safe_div_vec3(
                                            wp.mesh_eval_face_normal(mesh, face), shape_sizes[si]
                                        )
                                    hit.normal = raytrace._mesh_hit_normal(
                                        shape_transforms[si],
                                        shape_sizes[si],
                                        mesh,
                                        mesh_data_id,
                                        mesh_data,
                                        normal,
                                        face,
                                        u,
                                        v,
                                    )
                            else:
                                hit.bary_u = 0.0
                                hit.bary_v = 0.0
                                hit.face_idx = -1
                                if wp.static(with_normal):
                                    _distance, normal = raycast.ray_intersect_shape(
                                        shape_transforms[si],
                                        shape_sizes[si],
                                        shape_types[si],
                                        origin,
                                        direction,
                                        wp.static(config.enable_backface_culling),
                                    )
                                    hit.normal = normal
            return hit

        return closest_hit

    def create_first_hit(self, config: RenderConfig, state: RenderState) -> wp.Function:
        shapes = self.shapes
        grid, extent, world_count = shapes.grid, shapes.extent, shapes.world_count
        planes_per_world = shapes.world_planes.shape[1]
        ray_mask = _RAY_MASK_ALL if config.enable_particles else _RAY_MASK_NO_PARTICLES
        traversals = self._traversal_count(config)
        has_instances = self.has_instances

        @wp.func
        def plane_blocks(
            si: int,
            shape_sizes: wp.array[wp.vec3f],
            shape_transforms: wp.array[wp.transformf],
            origin: wp.vec3f,
            direction: wp.vec3f,
            max_dist: wp.float32,
        ) -> wp.bool:
            distance, _normal = raycast.ray_intersect_shape_no_normal(
                shape_transforms[si],
                shape_sizes[si],
                GeoType.PLANE,
                origin,
                direction,
                wp.static(config.enable_backface_culling),
            )
            return distance >= 0.0 and distance < max_dist

        @wp.func
        def first_hit(
            scene: SceneData,
            world_index: wp.int32,
            shape_types: wp.array[wp.int32],
            shape_sizes: wp.array[wp.vec3f],
            shape_transforms: wp.array[wp.transformf],
            shape_source_ptr: wp.array[wp.uint64],
            origin: wp.vec3f,
            direction: wp.vec3f,
            max_dist: wp.float32,
        ) -> wp.bool:
            if wp.static(config.enable_global_world):
                for p in range(wp.static(len(shapes.planes))):
                    if plane_blocks(
                        wp.static(shapes.planes[p]), shape_sizes, shape_transforms, origin, direction, max_dist
                    ):
                        return True
            for p in range(wp.static(planes_per_world)):
                si = scene.world_roots[world_index * wp.static(planes_per_world) + p]
                if si >= 0:
                    if plane_blocks(si, shape_sizes, shape_transforms, origin, direction, max_dist):
                        return True
            # Unrolled: dynamic traversal loops miscompile with warp-optix 0.3.0.
            for i in range(wp.static(traversals)):
                world = wp.where(i == 1, wp.static(world_count), world_index)
                interval = ray_interval(origin, direction, wp.static(extent), max_dist)
                if wp.static(has_instances) and interval[1] >= interval[0]:
                    packed_origin = origin + world_offset(world, wp.static(grid), wp.static(extent))
                    if wo.accel_query_ray_anyhit(
                        scene.id, packed_origin, direction, interval[1], interval[0], wp.static(ray_mask)
                    ):
                        return True
            return False

        return first_hit
