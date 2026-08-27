# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle surface extraction using anisotropic kernels and marching cubes.

Implements the method from Yu & Turk, "Reconstructing Surfaces of
Particle-Based Fluids Using Anisotropic Kernels", Eurographics/ACM SIGGRAPH
Symposium on Computer Animation, 2010.

The pipeline computes per-particle anisotropy matrices via Weighted PCA,
then evaluates a smooth scalar field on a sparse volume using oriented
ellipsoidal kernels, and extracts the isosurface with
:class:`warp.MarchingCubes`.

Typical usage::

    surface_ctx = ParticleSurface(voxel_size=0.01)
    verts, indices, normals = surface_ctx.extract(
        state.particle_q,
        model.particle_radius,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import warp as wp

from . import particle_surface_kernels as kernels
from . import particle_surface_sparse_kernels as sparse_kernels
from .hashtable import HashTable

__all__ = ["ParticleSurface", "extract_particle_surface"]

_MESH_SMOOTH_SHRINK_PER_VOXEL = 0.15
_MIN_DENSITY_MARCHING_THRESHOLD = 0.01
_REBUILDABLE_FINE_LEAF_CELL_RATIO = 16
_REBUILDABLE_LOWER_LEAF_RATIO = 64
_REBUILDABLE_UPPER_LOWER_RATIO = 64

# ---------------------------------------------------------------------------
# ParticleSurface context
# ---------------------------------------------------------------------------


class _ParticleSurfaceWorkspaceBase:
    """Common particle-bound and mesh-count launches."""

    def reset(self) -> None:
        if self.world_count > 1:
            self.hash_spacing.zero_()
        world_mesh_counts = self.mesh_counts if self.world_count == 1 else self.world_mesh_counts
        wp.launch(
            kernels.reset_bounds_and_counts,
            dim=self.world_count,
            inputs=[self.lower, self.upper, self.grid_counts, world_mesh_counts],
            device=self.device,
        )

    def reset_mesh_counts(self) -> None:
        world_mesh_counts = self.mesh_counts if self.world_count == 1 else self.world_mesh_counts
        wp.launch(
            kernels.reset_mesh_counts,
            dim=self.world_count,
            inputs=[world_mesh_counts, self.grid_counts],
            device=self.device,
        )

    def compute_mesh_world_starts(self) -> None:
        world_mesh_counts = self.mesh_counts if self.world_count == 1 else self.world_mesh_counts
        wp.launch(
            kernels.compute_mesh_world_starts,
            dim=1,
            inputs=[
                world_mesh_counts,
                self.world_count,
                self.vertex_world_offsets,
                self.index_world_offsets,
                self.mesh_counts,
            ],
            device=self.device,
        )

    def compute_particle_bounds(
        self,
        positions: wp.array[wp.vec3],
        flags: wp.array[wp.int32],
        use_flags: int,
        particle_world: wp.array[wp.int32],
        use_worlds: int,
        sentinel_distance: float,
    ) -> None:
        if positions.shape[0] > 0:
            tile_size = kernels._AABB_TILE_SIZE if self.device.is_cuda else 1
            kernel = kernels.compute_particle_bounds_worlds if use_worlds != 0 else kernels.compute_particle_bounds
            inputs = [positions, flags, use_flags]
            if use_worlds != 0:
                inputs.extend((particle_world, use_worlds, self.world_count))
            inputs.extend((self.lower, self.upper, self.grid_counts))
            wp.launch(
                kernel,
                dim=((positions.shape[0] + tile_size - 1) // tile_size, tile_size),
                inputs=inputs,
                block_dim=tile_size,
                device=self.device,
            )
        if use_worlds != 0:
            wp.launch(
                kernels.finalize_particle_bounds_worlds,
                dim=self.world_count,
                inputs=[
                    self.lower,
                    self.upper,
                    self.inactive_position,
                    self.hash_spacing,
                    use_worlds,
                    sentinel_distance,
                ],
                device=self.device,
            )
            wp.launch(
                kernels.finalize_hash_spacing,
                dim=1,
                inputs=[self.hash_spacing, sentinel_distance / 1.0e6],
                device=self.device,
            )
        else:
            wp.launch(
                kernels.finalize_particle_bounds,
                dim=1,
                inputs=[self.lower, self.upper, self.inactive_position, sentinel_distance],
                device=self.device,
            )


class _ParticleSurfaceSparseWorkspace(_ParticleSurfaceWorkspaceBase):
    """Sparse-volume storage and launches for particle surface extraction."""

    def __init__(
        self,
        max_grid_cells: int | None,
        world_count: int,
        voxel_size: float,
        padding: int,
        support_leaf_radius: int | None,
        device: wp.DeviceLike,
    ):
        self.world_count = int(world_count)
        self.voxel_size = float(voxel_size)
        self.padding = int(padding)
        self.device = wp.get_device(device)
        self.rebuildable = max_grid_cells is not None
        self.requested_max_grid_cells = max_grid_cells

        self.lower = wp.empty(self.world_count, dtype=wp.vec3, device=self.device)
        self.upper = wp.empty(self.world_count, dtype=wp.vec3, device=self.device)
        self.inactive_position = wp.empty(self.world_count, dtype=wp.vec3, device=self.device)
        self.hash_spacing = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.grid_origin = wp.empty(self.world_count, dtype=wp.vec3, device=self.device)
        self.grid_dims = wp.empty(self.world_count, dtype=wp.vec3i, device=self.device)
        self.grid_counts = wp.zeros(7 * self.world_count, dtype=wp.int32, device=self.device)
        self.per_world_status: wp.array[wp.int32] = self.grid_counts[3::7]
        self.grid_node_world_start = wp.zeros(self.world_count + 1, dtype=wp.int32, device=self.device)
        self.grid_cell_world_start = wp.zeros(self.world_count + 1, dtype=wp.int32, device=self.device)
        self.active_particle_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.world_mesh_counts = wp.zeros(3 * self.world_count, dtype=wp.int32, device=self.device)
        self.mesh_write_counts = wp.zeros(3 * self.world_count, dtype=wp.int32, device=self.device)
        self.mesh_counts = wp.zeros(3, dtype=wp.int32, device=self.device)
        self.vertex_world_offsets = wp.zeros(self.world_count + 1, dtype=wp.int32, device=self.device)
        self.index_world_offsets = wp.zeros(self.world_count + 1, dtype=wp.int32, device=self.device)
        self.env_offsets = wp.zeros(self.world_count, dtype=wp.vec3i, device=self.device)
        self.packed_lower = wp.zeros(self.world_count, dtype=wp.vec3i, device=self.device)
        self.packed_upper = wp.zeros(self.world_count, dtype=wp.vec3i, device=self.device)

        self.volume: wp.Volume | None = None
        self.voxel_ijk: wp.array[wp.vec3i] | None = None
        self.cell_world: wp.array[wp.int32] | None = None
        self.node_world: wp.array[wp.int32] | None = None
        self.field = wp.empty(0, dtype=wp.float32, device=self.device)
        self.field_temp = wp.empty_like(self.field)
        self.field_orig = wp.empty_like(self.field)
        self.edge_indices: wp.array[wp.int32] | None = None
        self.vertices: wp.array[wp.vec3] | None = None
        self.vertices_temp: wp.array[wp.vec3] | None = None
        self.indices: wp.array[wp.int32] | None = None
        self.normals: wp.array[wp.vec3] | None = None
        self.neighbor_sum: wp.array[wp.vec3] | None = None
        self.valence: wp.array[wp.int32] | None = None
        self.support_leaf_keys = wp.empty(1, dtype=wp.vec3i, device=self.device)
        self.support_leaf_mask = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.expanded_leaf_keys = wp.empty(1, dtype=wp.vec3i, device=self.device)
        self.expanded_leaf_mask = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.max_support_reach_voxels = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._configured_support_leaf_radius = support_leaf_radius
        self._support_leaf_radius = support_leaf_radius
        self.leaf_volume: wp.Volume | None = None
        self.leaf_hash: HashTable | None = None
        self.leaf_ijk = wp.empty(1, dtype=wp.vec3i, device=self.device)
        self.topology_occupancy = wp.zeros(1, dtype=wp.uint32, device=self.device)
        self.topology_occupancy_temp = wp.zeros(1, dtype=wp.uint32, device=self.device)
        self.topology_voxels = wp.empty(1, dtype=wp.vec3i, device=self.device)
        self.topology_voxel_mask = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.topology_voxel_count = wp.zeros(1, dtype=wp.uint32, device=self.device)
        self.rebuild_status = wp.zeros(5, dtype=wp.uint32, device=self.device)
        corner_offsets = wp.MarchingCubes.CUBE_CORNER_OFFSETS
        edge_offsets: list[tuple[int, int, int]] = []
        edge_axes: list[int] = []
        for first, second in wp.MarchingCubes.EDGE_TO_CORNERS:
            first_corner = corner_offsets[first]
            second_corner = corner_offsets[second]
            edge_offsets.append(tuple(min(first_corner[a], second_corner[a]) for a in range(3)))
            edge_axes.append(next(a for a in range(3) if first_corner[a] != second_corner[a]))
        self.case_ranges = wp.array(wp.MarchingCubes.CASE_TO_TRI_RANGE, dtype=wp.int32, device=self.device)
        self.local_edges = wp.array(wp.MarchingCubes.TRI_LOCAL_INDICES, dtype=wp.int32, device=self.device)
        self.corner_offsets = wp.array(corner_offsets, dtype=wp.vec3i, device=self.device)
        self.edge_offsets = wp.array(edge_offsets, dtype=wp.vec3i, device=self.device)
        self.edge_axes = wp.array(edge_axes, dtype=wp.int32, device=self.device)
        self._dummy_vertex = wp.empty(1, dtype=wp.vec3, device=self.device)
        self._dummy_index = wp.empty(1, dtype=wp.int32, device=self.device)

        self.max_grid_cells = 0
        self.max_grid_nodes = 0
        self.max_vertices = 0
        self.max_indices = 0
        self.launch_threads = 1
        if self.rebuildable:
            self._allocate_rebuildable_topology(int(max_grid_cells))

    @property
    def cell_grid(self) -> wp.Volume | None:
        """Sparse active-cell index grid."""
        return self.volume

    @property
    def node_grid(self) -> wp.Volume | None:
        """Sparse scalar-field node index grid."""
        return self.volume

    @property
    def cell_ijk(self) -> wp.array[wp.vec3i] | None:
        return self.voxel_ijk

    @property
    def node_ijk(self) -> wp.array[wp.vec3i] | None:
        return self.voxel_ijk

    def _allocate_rebuildable_topology(self, max_grid_cells: int) -> None:
        if max_grid_cells <= 0:
            raise ValueError("max_grid_cells must be positive")
        max_tiles = max((max_grid_cells + 511) // 512, 1)
        self.max_grid_cells = max_tiles * 512
        max_leaf_nodes, max_lower_nodes, max_upper_nodes = self._topology_node_capacities(
            self.max_grid_cells,
            _REBUILDABLE_FINE_LEAF_CELL_RATIO,
        )
        dummy_points = wp.zeros(1, dtype=wp.vec3i, device=self.device)
        self.leaf_hash = HashTable(max_leaf_nodes, device=self.device)
        self.volume = wp.Volume.allocate_by_voxels(
            dummy_points,
            voxel_size=self.voxel_size,
            translation=(0.5 * self.voxel_size,) * 3,
            device=self.device,
            rebuildable=True,
            max_active_voxels=self.max_grid_cells,
            max_leaf_nodes=max_leaf_nodes,
            max_lower_nodes=max_lower_nodes,
            max_upper_nodes=max_upper_nodes,
            status=self.rebuild_status[4:5],
        )
        self.leaf_ijk = wp.empty(self.leaf_hash.capacity, dtype=wp.vec3i, device=self.device)
        occupancy_word_capacity = 16 * self.leaf_hash.capacity
        self.topology_occupancy = wp.zeros(occupancy_word_capacity, dtype=wp.uint32, device=self.device)
        self.topology_occupancy_temp = wp.zeros_like(self.topology_occupancy)
        self.topology_voxels = wp.empty(self.max_grid_cells, dtype=wp.vec3i, device=self.device)
        self.topology_voxel_mask = wp.zeros(self.max_grid_cells, dtype=wp.int32, device=self.device)
        self.cell_world = wp.zeros(self.max_grid_cells, dtype=wp.int32, device=self.device)
        self.node_world = self.cell_world
        self.voxel_ijk = wp.empty(self.max_grid_cells, dtype=wp.vec3i, device=self.device)
        self.max_grid_nodes = self.max_grid_cells
        self._allocate_field_and_mesh(self.max_grid_nodes, self.max_grid_cells, allocate_mesh=True)

    def _topology_node_capacities(self, max_grid_cells: int, leaf_cell_ratio: int) -> tuple[int, int, int]:
        max_leaf_nodes = max((max_grid_cells + leaf_cell_ratio - 1) // leaf_cell_ratio, 1)
        max_lower_nodes = max(
            (max_leaf_nodes + _REBUILDABLE_LOWER_LEAF_RATIO - 1) // _REBUILDABLE_LOWER_LEAF_RATIO,
            64 * self.world_count,
        )
        max_upper_nodes = max(
            (max_lower_nodes + _REBUILDABLE_UPPER_LOWER_RATIO - 1) // _REBUILDABLE_UPPER_LOWER_RATIO,
            64 * self.world_count,
        )
        return max_leaf_nodes, max_lower_nodes, max_upper_nodes

    def _allocate_field_and_mesh(self, node_count: int, cell_count: int, *, allocate_mesh: bool) -> None:
        self.max_grid_nodes = int(node_count)
        self.max_grid_cells = int(cell_count)
        self.field = wp.empty(node_count, dtype=wp.float32, device=self.device)
        self.field_temp = wp.empty_like(self.field)
        self.field_orig = wp.empty_like(self.field)
        self.launch_threads = min(max(node_count, cell_count, 1), kernels._MAX_CAPACITY_LAUNCH_THREADS)
        if not allocate_mesh:
            return
        self.max_vertices = 3 * node_count
        self.max_indices = 15 * cell_count
        self.edge_indices = wp.empty(3 * node_count, dtype=wp.int32, device=self.device)
        self.vertices = wp.empty(self.max_vertices, dtype=wp.vec3, device=self.device)
        self.vertices_temp = wp.empty_like(self.vertices)
        self.indices = wp.empty(self.max_indices, dtype=wp.int32, device=self.device)
        self.normals = wp.empty(self.max_vertices, dtype=wp.vec3, device=self.device)
        self.neighbor_sum = wp.empty(self.max_vertices, dtype=wp.vec3, device=self.device)
        self.valence = wp.empty(self.max_vertices, dtype=wp.int32, device=self.device)

    def ensure_support_leaf_keys(self, particle_count: int) -> None:
        size = max(sparse_kernels._SUPPORT_VOXEL_COUNT * particle_count, 1)
        if self.support_leaf_keys.shape[0] == size:
            return
        self.support_leaf_keys = wp.empty(size, dtype=wp.vec3i, device=self.device)
        self.support_leaf_mask = wp.empty(size, dtype=wp.int32, device=self.device)
        if not self.rebuildable:
            self.expanded_leaf_keys = wp.empty(1, dtype=wp.vec3i, device=self.device)
            self.expanded_leaf_mask = wp.zeros(1, dtype=wp.int32, device=self.device)
            self.topology_occupancy = wp.zeros(1, dtype=wp.uint32, device=self.device)
            self.topology_occupancy_temp = wp.zeros(1, dtype=wp.uint32, device=self.device)
            self.topology_voxels = wp.empty(1, dtype=wp.vec3i, device=self.device)
            self.topology_voxel_mask = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._support_leaf_radius = self._configured_support_leaf_radius

    def compute_grid(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        flags: wp.array[wp.int32],
        use_flags: int,
        particle_world: wp.array[wp.int32],
        use_worlds: int,
        det_G: wp.array[float],
        density_reach: wp.array[wp.vec3],
        particle_sdf_radius_scale: float,
        particle_sdf_band: float,
        particle_sdf: bool,
        anisotropic: bool,
        topology_halo_voxels: int,
    ) -> None:
        wp.launch(kernels.reset_bounds, dim=self.world_count, inputs=[self.lower, self.upper], device=self.device)
        if positions.shape[0] > 0:
            tile_size = kernels._AABB_TILE_SIZE if self.device.is_cuda else 1
            kernel = kernels.compute_kernel_bounds_worlds if use_worlds != 0 else kernels.compute_kernel_bounds
            inputs = [positions, radii, flags, use_flags]
            if use_worlds != 0:
                inputs.extend((particle_world, use_worlds, self.world_count))
            inputs.extend(
                (
                    det_G,
                    density_reach,
                    particle_sdf_radius_scale,
                    particle_sdf_band,
                    int(particle_sdf),
                    int(anisotropic),
                    self.lower,
                    self.upper,
                )
            )
            wp.launch(
                kernel,
                dim=((positions.shape[0] + tile_size - 1) // tile_size, tile_size),
                inputs=inputs,
                block_dim=tile_size,
                device=self.device,
            )
        wp.launch(
            sparse_kernels.finalize_sparse_grids,
            dim=1,
            inputs=[
                self.lower,
                self.upper,
                self.grid_counts,
                self.grid_origin,
                self.grid_dims,
                self.env_offsets,
                self.packed_lower,
                self.packed_upper,
                self.active_particle_count,
                self.world_count,
                self.voxel_size,
                self.padding,
                topology_halo_voxels,
            ],
            device=self.device,
        )

    def build_topology(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        flags: wp.array[wp.int32],
        use_flags: int,
        particle_world: wp.array[wp.int32],
        use_worlds: int,
        det_G: wp.array[float],
        density_reach: wp.array[wp.vec3],
        particle_sdf_radius_scale: float,
        particle_sdf_band: float,
        particle_sdf: bool,
        anisotropic_sdf: bool,
        stencil_voxels: int,
        G: wp.array[wp.mat33],
    ) -> None:
        self.rebuild_status.zero_()
        if not self.rebuildable:
            self.max_support_reach_voxels.zero_()
            self.ensure_support_leaf_keys(positions.shape[0])
            if positions.shape[0] > 0:
                wp.launch(
                    sparse_kernels.emit_particle_support_voxels,
                    dim=sparse_kernels._SUPPORT_VOXEL_COUNT * positions.shape[0],
                    inputs=[
                        positions,
                        radii,
                        flags,
                        use_flags,
                        particle_world,
                        use_worlds,
                        self.world_count,
                        det_G,
                        density_reach,
                        particle_sdf_radius_scale,
                        particle_sdf_band,
                        int(particle_sdf),
                        int(anisotropic_sdf),
                        self.env_offsets,
                        1.0 / self.voxel_size,
                        self.support_leaf_keys,
                        self.support_leaf_mask,
                        self.max_support_reach_voxels,
                    ],
                    device=self.device,
                )
            else:
                self.support_leaf_mask.zero_()
            max_reach_voxels = float(self.max_support_reach_voxels.numpy()[0])
            # Half-reach octant samples leave at most half the support width between samples.
            voxel_radius = int(math.ceil(0.5 * max_reach_voxels)) + 1
            self._support_leaf_radius = (voxel_radius + 7) // 8

        if self.rebuildable:
            self._rebuild_topology(
                positions,
                radii,
                flags,
                use_flags,
                particle_world,
                use_worlds,
                G,
                det_G,
                density_reach,
                particle_sdf_radius_scale,
                particle_sdf_band,
                particle_sdf,
                anisotropic_sdf,
                stencil_voxels,
            )
        else:
            self._build_exact_topology(
                positions,
                radii,
                flags,
                use_flags,
                particle_world,
                use_worlds,
                G,
                det_G,
                density_reach,
                particle_sdf_radius_scale,
                particle_sdf_band,
                particle_sdf,
                anisotropic_sdf,
                stencil_voxels,
            )
        self._classify_topology()

    def _populate_topology(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        flags: wp.array[wp.int32],
        use_flags: int,
        particle_world: wp.array[wp.int32],
        use_worlds: int,
        G: wp.array[wp.mat33],
        det_G: wp.array[float],
        density_reach: wp.array[wp.vec3],
        particle_sdf_radius_scale: float,
        particle_sdf_band: float,
        particle_sdf: bool,
        anisotropic_sdf: bool,
        stencil_voxels: int,
    ) -> None:
        self.topology_occupancy.zero_()
        self.topology_voxel_mask.zero_()
        self.topology_voxel_count.zero_()
        lane_count = 64 if self.device.is_cuda else 1
        if positions.shape[0] > 0:
            wp.launch(
                sparse_kernels.mark_particle_support_voxels,
                dim=(positions.shape[0], lane_count),
                inputs=[
                    self.leaf_volume.id,
                    positions,
                    radii,
                    flags,
                    use_flags,
                    particle_world,
                    use_worlds,
                    self.world_count,
                    G,
                    det_G,
                    density_reach,
                    particle_sdf_radius_scale,
                    particle_sdf_band,
                    int(particle_sdf),
                    int(anisotropic_sdf),
                    self.env_offsets,
                    1.0 / self.voxel_size,
                    lane_count,
                    self.topology_occupancy,
                    self.rebuild_status[0:1],
                ],
                device=self.device,
            )

        source = self.topology_occupancy
        destination = self.topology_occupancy_temp
        for axis in range(3):
            destination.zero_()
            wp.launch(
                sparse_kernels.dilate_topology_axis,
                dim=self.launch_threads,
                inputs=[
                    self.leaf_volume.id,
                    self.leaf_ijk,
                    source,
                    destination,
                    stencil_voxels,
                    axis,
                    self.rebuild_status[0:1],
                    self.launch_threads,
                ],
                device=self.device,
            )
            source, destination = destination, source
        wp.launch(
            sparse_kernels.emit_topology_voxels,
            dim=self.launch_threads,
            inputs=[
                self.leaf_volume.id,
                self.leaf_ijk,
                source,
                self.topology_voxels,
                self.topology_voxel_mask,
                self.topology_voxel_count,
                self.rebuild_status[0:1],
                self.launch_threads,
            ],
            device=self.device,
        )

    def _rebuild_topology(self, *topology_args) -> None:
        positions = topology_args[0]
        leaf_hash = self.leaf_hash
        hash_clear_threads = min(65_536, leaf_hash.capacity)
        wp.launch(
            sparse_kernels.clear_topology_leaf_hash,
            dim=hash_clear_threads,
            inputs=[
                leaf_hash.keys,
                leaf_hash.active_slots,
                self.topology_occupancy,
                self.topology_occupancy_temp,
                self.topology_voxel_mask,
                self.topology_voxel_count,
                hash_clear_threads,
            ],
            device=self.device,
        )
        wp.launch(
            sparse_kernels.reset_topology_leaf_hash_counts,
            dim=1,
            inputs=[leaf_hash.keys, leaf_hash.active_slots, self.topology_voxel_count],
            device=self.device,
        )

        candidate_lane_count = 16 if self.device.is_cuda else 1
        if positions.shape[0] > 0:
            wp.launch(
                sparse_kernels.insert_particle_candidate_leaves,
                dim=(positions.shape[0], candidate_lane_count),
                inputs=[
                    positions,
                    topology_args[1],
                    topology_args[2],
                    topology_args[3],
                    topology_args[4],
                    topology_args[5],
                    self.world_count,
                    topology_args[7],
                    topology_args[8],
                    topology_args[9],
                    topology_args[10],
                    int(topology_args[11]),
                    int(topology_args[12]),
                    self.env_offsets,
                    1.0 / self.voxel_size,
                    int(topology_args[13]),
                    candidate_lane_count,
                    leaf_hash.keys,
                    leaf_hash.active_slots,
                    self.leaf_ijk,
                    self.rebuild_status[0:1],
                ],
                device=self.device,
            )

            support_lane_count = 64 if self.device.is_cuda else 1
            wp.launch(
                sparse_kernels.mark_particle_support_voxels_hash,
                dim=(positions.shape[0], support_lane_count),
                inputs=[
                    leaf_hash.keys,
                    positions,
                    topology_args[1],
                    topology_args[2],
                    topology_args[3],
                    topology_args[4],
                    topology_args[5],
                    self.world_count,
                    topology_args[6],
                    topology_args[7],
                    topology_args[8],
                    topology_args[9],
                    topology_args[10],
                    int(topology_args[11]),
                    int(topology_args[12]),
                    self.env_offsets,
                    1.0 / self.voxel_size,
                    support_lane_count,
                    self.topology_occupancy,
                    self.rebuild_status[0:1],
                ],
                device=self.device,
            )

        source = self.topology_occupancy
        destination = self.topology_occupancy_temp
        for axis in range(3):
            wp.launch(
                sparse_kernels.dilate_topology_axis_hash,
                dim=self.launch_threads,
                inputs=[
                    leaf_hash.keys,
                    leaf_hash.active_slots,
                    self.leaf_ijk,
                    source,
                    destination,
                    int(topology_args[13]),
                    axis,
                    self.launch_threads,
                ],
                device=self.device,
            )
            source, destination = destination, source
        wp.launch(
            sparse_kernels.emit_topology_voxels_hash,
            dim=self.launch_threads,
            inputs=[
                leaf_hash.keys,
                leaf_hash.active_slots,
                self.leaf_ijk,
                source,
                self.topology_voxels,
                self.topology_voxel_mask,
                self.topology_voxel_count,
                self.rebuild_status[0:1],
                self.launch_threads,
            ],
            device=self.device,
        )
        self.volume.rebuild(
            self.topology_voxels,
            status=self.rebuild_status[4:5],
            point_mask=self.topology_voxel_mask,
        )
        self.volume.get_voxels(out=self.voxel_ijk)

    def _build_exact_topology(self, *topology_args) -> None:
        initial_leaf_volume = wp.Volume.allocate_by_voxels(
            self.support_leaf_keys,
            voxel_size=1.0,
            device=self.device,
            point_mask=self.support_leaf_mask,
        )
        initial_leaf_count = initial_leaf_volume.get_active_stats().voxel_count
        initial_leaf_ijk = wp.empty(initial_leaf_count, dtype=wp.vec3i, device=self.device)
        initial_leaf_volume.get_voxels(out=initial_leaf_ijk)
        leaf_radius = self._support_leaf_radius + (int(topology_args[-1]) + 7) // 8
        neighbor_width = 2 * leaf_radius + 1
        expansion_count = initial_leaf_count * neighbor_width * neighbor_width * neighbor_width
        self.expanded_leaf_keys = wp.empty(expansion_count, dtype=wp.vec3i, device=self.device)
        self.expanded_leaf_mask = wp.empty(expansion_count, dtype=wp.int32, device=self.device)
        wp.launch(
            sparse_kernels.expand_candidate_leaf_keys,
            dim=expansion_count,
            inputs=[
                initial_leaf_volume.id,
                initial_leaf_ijk,
                leaf_radius,
                self.expanded_leaf_keys,
                self.expanded_leaf_mask,
                self.rebuild_status[2:3],
            ],
            device=self.device,
        )
        self.leaf_volume = wp.Volume.allocate_by_voxels(
            self.expanded_leaf_keys,
            voxel_size=1.0,
            device=self.device,
            point_mask=self.expanded_leaf_mask,
        )
        leaf_count = self.leaf_volume.get_active_stats().voxel_count
        self.leaf_ijk = wp.empty(leaf_count, dtype=wp.vec3i, device=self.device)
        self.leaf_volume.get_voxels(out=self.leaf_ijk)
        candidate_voxel_count = 512 * leaf_count
        occupancy_word_count = (candidate_voxel_count + 31) // 32
        self.topology_occupancy = wp.zeros(occupancy_word_count, dtype=wp.uint32, device=self.device)
        self.topology_occupancy_temp = wp.zeros_like(self.topology_occupancy)
        self.topology_voxels = wp.empty(candidate_voxel_count, dtype=wp.vec3i, device=self.device)
        self.topology_voxel_mask = wp.zeros(candidate_voxel_count, dtype=wp.int32, device=self.device)
        self.launch_threads = min(max(candidate_voxel_count, 1), kernels._MAX_CAPACITY_LAUNCH_THREADS)
        self._populate_topology(*topology_args)
        volume = wp.Volume.allocate_by_voxels(
            self.topology_voxels,
            voxel_size=self.voxel_size,
            translation=(0.5 * self.voxel_size,) * 3,
            device=self.device,
            point_mask=self.topology_voxel_mask,
        )
        self.volume = volume
        cell_count = volume.get_active_stats().voxel_count
        self.cell_world = wp.empty(cell_count, dtype=wp.int32, device=self.device)
        self.node_world = self.cell_world
        self.voxel_ijk = wp.empty(cell_count, dtype=wp.vec3i, device=self.device)
        volume.get_voxels(out=self.voxel_ijk)
        node_count = cell_count
        self._allocate_field_and_mesh(node_count, cell_count, allocate_mesh=False)

    def _classify_topology(self) -> None:
        if self.world_count == 1:
            wp.launch(
                sparse_kernels.classify_sparse_topology_single_world,
                dim=self.launch_threads,
                inputs=[self.volume.id, self.cell_world, self.grid_counts, self.launch_threads],
                device=self.device,
            )
        else:
            wp.launch(
                sparse_kernels.classify_sparse_topology,
                dim=self.launch_threads,
                inputs=[
                    self.volume.id,
                    self.voxel_ijk,
                    self.packed_lower,
                    self.packed_upper,
                    self.cell_world,
                    self.grid_counts,
                    self.world_count,
                    self.launch_threads,
                ],
                device=self.device,
            )
        wp.launch(
            sparse_kernels.finalize_sparse_topology,
            dim=1,
            inputs=[
                self.rebuild_status,
                self.grid_counts,
                self.grid_node_world_start,
                self.grid_cell_world_start,
                self.world_count,
                self.requested_max_grid_cells if self.requested_max_grid_cells is not None else self.max_grid_cells,
            ],
            device=self.device,
        )

    def evaluate_field(
        self,
        smoothed: wp.array[wp.vec3],
        radii: wp.array[float],
        flags: wp.array[wp.int32],
        use_flags: int,
        particle_world: wp.array[wp.int32],
        use_worlds: int,
        G: wp.array[wp.mat33],
        det_G: wp.array[float],
        density_reach: wp.array[wp.vec3],
        *,
        surface_method: str,
        anisotropic: bool,
        particle_sdf_radius_scale: float,
        particle_sdf_band: float,
        kernel_radius: float,
        field_mode: str,
        threshold: float,
        blur_weights: wp.array[float] | None,
        blur_radius: int,
        blur_iterations: int,
        redistance_iterations: int,
    ) -> None:
        particle_sdf = surface_method == "particle_sdf"
        outside_value = kernel_radius * particle_sdf_band if particle_sdf else 0.0
        node_grid = self.volume
        wp.launch(
            sparse_kernels.fill_field,
            dim=self.launch_threads,
            inputs=[node_grid.id, self.field, outside_value, self.launch_threads],
            device=self.device,
        )
        if smoothed.shape[0] > 0:
            common = [
                node_grid.id,
                smoothed,
                radii,
                flags,
                use_flags,
                particle_world,
                use_worlds,
                self.world_count,
            ]
            if particle_sdf and anisotropic:
                wp.launch(
                    sparse_kernels.evaluate_particle_sdf_anisotropic,
                    dim=smoothed.shape[0],
                    inputs=[
                        *common,
                        G,
                        det_G,
                        density_reach,
                        particle_sdf_radius_scale,
                        particle_sdf_band,
                        self.env_offsets,
                        1.0 / self.voxel_size,
                        self.field,
                    ],
                    device=self.device,
                )
            elif particle_sdf:
                wp.launch(
                    sparse_kernels.evaluate_particle_sdf_isotropic,
                    dim=smoothed.shape[0],
                    inputs=[
                        *common,
                        particle_sdf_radius_scale,
                        particle_sdf_band,
                        self.env_offsets,
                        1.0 / self.voxel_size,
                        self.field,
                    ],
                    device=self.device,
                )
            else:
                if self.device.is_cuda:
                    if anisotropic:
                        lane_count = sparse_kernels._DENSITY_SPLAT_LANES_ANISOTROPIC
                        block_dim = sparse_kernels._DENSITY_SPLAT_BLOCK_DIM_ANISOTROPIC
                    else:
                        lane_count = sparse_kernels._DENSITY_SPLAT_LANES_ISOTROPIC
                        block_dim = sparse_kernels._DENSITY_SPLAT_BLOCK_DIM_ISOTROPIC
                else:
                    lane_count = 1
                    block_dim = 256
                wp.launch(
                    sparse_kernels.evaluate_density,
                    dim=(smoothed.shape[0], lane_count),
                    inputs=[
                        *common,
                        G,
                        det_G,
                        density_reach,
                        self.env_offsets,
                        1.0 / self.voxel_size,
                        lane_count,
                        self.field,
                    ],
                    device=self.device,
                    block_dim=block_dim,
                )

        if blur_iterations > 0 and blur_radius > 0 and blur_weights is not None:
            source = self.field
            destination = self.field_temp
            for _ in range(blur_iterations):
                for axis in range(3):
                    wp.launch(
                        sparse_kernels.blur_field_axis,
                        dim=self.launch_threads,
                        inputs=[
                            node_grid.id,
                            self.voxel_ijk,
                            source,
                            destination,
                            blur_weights,
                            blur_radius,
                            axis,
                            outside_value,
                            self.launch_threads,
                        ],
                        device=self.device,
                    )
                    source, destination = destination, source
            if source is not self.field:
                self.field, self.field_temp = source, destination

        if field_mode == "sdf":
            if not particle_sdf:
                wp.launch(
                    sparse_kernels.density_to_sdf,
                    dim=self.launch_threads,
                    inputs=[node_grid.id, self.field, threshold, self.launch_threads],
                    device=self.device,
                )
            self.redistance(redistance_iterations, outside_value=outside_value if particle_sdf else threshold)

    def redistance(self, iterations: int, *, outside_value: float = 0.0) -> None:
        for _ in range(iterations):
            wp.launch(
                sparse_kernels.redistance_step,
                dim=self.launch_threads,
                inputs=[
                    self.volume.id,
                    self.voxel_ijk,
                    self.field,
                    self.field_temp,
                    outside_value,
                    1.0 / self.voxel_size,
                    self.launch_threads,
                ],
                device=self.device,
            )
            self.field, self.field_temp = self.field_temp, self.field

    def resize_mesh_exact(self, vertex_count: int, index_count: int) -> None:
        self.max_vertices = vertex_count
        self.max_indices = index_count
        self.edge_indices = wp.empty(3 * self.max_grid_nodes, dtype=wp.int32, device=self.device)
        self.vertices = wp.empty(vertex_count, dtype=wp.vec3, device=self.device)
        self.vertices_temp = wp.empty_like(self.vertices)
        self.indices = wp.empty(index_count, dtype=wp.int32, device=self.device)
        self.normals = wp.empty(vertex_count, dtype=wp.vec3, device=self.device)
        self.neighbor_sum = wp.empty(vertex_count, dtype=wp.vec3, device=self.device)
        self.valence = wp.empty(vertex_count, dtype=wp.int32, device=self.device)

    def _launch_mesh(self, threshold: float, output_counts: wp.array[wp.int32], write_output: int) -> None:
        vertices = self.vertices if write_output != 0 else self._dummy_vertex
        edge_indices = self.edge_indices if write_output != 0 else self._dummy_index
        indices = self.indices if write_output != 0 else self._dummy_index
        wp.launch(
            sparse_kernels.extract_mesh_vertices,
            dim=self.launch_threads,
            inputs=[
                self.volume.id,
                self.voxel_ijk,
                self.node_world,
                self.env_offsets,
                self.field,
                threshold,
                self.voxel_size,
                self.world_count,
                vertices,
                edge_indices,
                output_counts,
                self.vertex_world_offsets,
                write_output,
                self.launch_threads,
            ],
            device=self.device,
        )
        wp.launch(
            sparse_kernels.extract_mesh_indices,
            dim=self.launch_threads,
            inputs=[
                self.volume.id,
                self.volume.id,
                self.voxel_ijk,
                self.cell_world,
                self.field,
                threshold,
                self.world_count,
                self.case_ranges,
                self.local_edges,
                self.corner_offsets,
                self.edge_offsets,
                self.edge_axes,
                edge_indices,
                indices,
                output_counts,
                self.index_world_offsets,
                write_output,
                self.launch_threads,
            ],
            device=self.device,
        )

    def count_mesh(self, threshold: float) -> None:
        self.reset_mesh_counts()
        counts = self.mesh_counts if self.world_count == 1 else self.world_mesh_counts
        self._launch_mesh(threshold, counts, 0)
        self.compute_mesh_world_starts()

    def extract_mesh(
        self,
        threshold: float,
        *,
        counts_precomputed: bool,
        flip_winding: bool,
        smooth_iterations: int,
        smooth_lambda: float,
        compute_normals: bool,
    ) -> None:
        if self.edge_indices is None or self.vertices is None or self.indices is None:
            raise RuntimeError("Mesh capacity was not allocated")
        if counts_precomputed:
            self.mesh_write_counts.zero_()
            output_counts = self.mesh_write_counts
        else:
            self.reset_mesh_counts()
            output_counts = self.mesh_counts if self.world_count == 1 else self.world_mesh_counts
        wp.launch(
            sparse_kernels.reset_edge_indices,
            dim=self.launch_threads,
            inputs=[self.volume.id, self.edge_indices, self.launch_threads],
            device=self.device,
        )
        self._launch_mesh(threshold, output_counts, 1)
        if not counts_precomputed:
            self.compute_mesh_world_starts()
        if flip_winding:
            wp.launch(
                kernels.flip_mesh_winding,
                dim=self.launch_threads,
                inputs=[self.indices, self.mesh_counts, self.launch_threads],
                device=self.device,
            )
        for _ in range(smooth_iterations):
            wp.launch(
                kernels.clear_mesh_neighbors,
                dim=self.launch_threads,
                inputs=[self.neighbor_sum, self.valence, self.mesh_counts, self.launch_threads],
                device=self.device,
            )
            wp.launch(
                kernels.scatter_mesh_neighbors,
                dim=self.launch_threads,
                inputs=[
                    self.vertices,
                    self.indices,
                    self.neighbor_sum,
                    self.valence,
                    self.mesh_counts,
                    self.launch_threads,
                ],
                device=self.device,
            )
            wp.launch(
                kernels.apply_mesh_smoothing,
                dim=self.launch_threads,
                inputs=[
                    self.vertices,
                    self.neighbor_sum,
                    self.valence,
                    smooth_lambda,
                    self.vertices_temp,
                    self.mesh_counts,
                    self.launch_threads,
                ],
                device=self.device,
            )
            self.vertices, self.vertices_temp = self.vertices_temp, self.vertices
        if compute_normals:
            wp.launch(
                kernels.clear_mesh_normals,
                dim=self.launch_threads,
                inputs=[self.normals, self.mesh_counts, self.launch_threads],
                device=self.device,
            )
            wp.launch(
                kernels.accumulate_mesh_normals,
                dim=self.launch_threads,
                inputs=[self.vertices, self.indices, self.normals, self.mesh_counts, self.launch_threads],
                device=self.device,
            )
            wp.launch(
                kernels.normalize_mesh_normals,
                dim=self.launch_threads,
                inputs=[self.normals, self.mesh_counts, self.launch_threads],
                device=self.device,
            )


class _SparseSdfMetadata(NamedTuple):
    """Internal sparse SDF metadata."""

    workspace: _ParticleSurfaceSparseWorkspace
    topology_halo: float


class ParticleSurface:
    """Reusable context for extracting a triangle mesh from particle data.

    Uses the Yu & Turk (2010) anisotropic kernel method: per-particle
    Weighted PCA determines oriented ellipsoidal kernels that produce a
    smooth scalar field whose isosurface tightly wraps the particles.

    Args:
        voxel_size: Edge length of each grid voxel [m].
        max_grid_cells: Maximum active sparse-grid cell count across all worlds.
            When set, extraction uses preallocated, graph-capturable buffers.
            When ``None``, each extraction uses tight sparse field and mesh
            allocations. Rebuildable topology-node capacities assume spatially
            coherent surface bands; highly scattered fields can exhaust them
            before reaching this cell count.
        world_count: Number of independent particle worlds to extract.
        kernel_radius: Search radius for neighbor queries [m].
            Defaults to ``3 * voxel_size``.
        threshold: Isosurface level for marching cubes.  The scalar field
            is approximately 1.0 inside dense particle regions.  Defaults to
            0.25.
        smooth_lambda: Blending factor for position smoothing [0, 1].
            Higher values produce smoother surfaces.  Defaults to 0.5.
        anisotropic: Enable per-particle WPCA anisotropic kernels.
            When ``False`` (default), all particles use isotropic kernels.
        anisotropy_ratio: Maximum anisotropic kernel axis ratio.  Higher values
            allow flatter ellipsoids.
        kernel_scale: Kernel radius multiplier relative to ``kernel_radius``.
            This sets the isotropic kernel radius and the geometric-mean
            radius of anisotropic kernels.
        anisotropy_scale: Relative multiplier for anisotropic kernel radii.
            Values greater than 1 widen anisotropic kernels without changing
            the isotropic fallback scale.  Defaults to 1.
        anisotropy_min_neighbors: Minimum number of other particles required
            for anisotropic kernels. Sparser particles use isotropic kernels.
        anisotropy_binning: Share one WPCA kernel among particles in each
            spatial bin. Bins are half the kernel radius wide. This reduces
            neighbor-query cost at the expense of spatial resolution.
        anisotropy_strength: Blend from isotropic kernels to anisotropic
            kernels [0, 1].  Lower values preserve more normal support from
            boundary particles back into the interior.
        surface_method: Surface reconstruction method. ``"density"`` uses
            anisotropic density splatting. ``"particle_sdf"`` directly unions
            per-particle anisotropic ellipsoid SDFs and stores an SDF field.
        particle_sdf_radius_scale: Radius multiplier for ``surface_method="particle_sdf"``.
        particle_sdf_band: Narrow-band half-width in normalized ellipsoid
            coordinates for ``surface_method="particle_sdf"``. Must be at
            least 1 so the band contains the zero level set.
        padding: Extra voxels added around the particle bounding box.
        field_smooth_iterations: Number of separable Gaussian blur passes
            applied to the scalar field before marching cubes.  Defaults to
            0.
        field_smooth_radius: Half-width of the Gaussian blur in voxels.
            Defaults to 1.
        field_mode: Field representation retained after extraction.
            ``"density"`` keeps the scalar density field used by marching
            cubes.  ``"sdf"`` converts it to a signed distance approximation
            with negative values inside the particle surface.  Defaults to
            ``"sdf"`` for ``surface_method="particle_sdf"`` and ``"density"``
            otherwise.
        redistance_iterations: Number of Eikonal redistancing iterations
            applied when ``field_mode="sdf"``.  Set to 0 to skip.
        mesh_smooth_iterations: Number of Laplacian smoothing passes
            applied to the extracted mesh.  Set to 0 to disable.
        mesh_smooth_lambda: Laplacian step size [0, 1].
        device: Warp device for computation.
    """

    @dataclass(frozen=True)
    class SparseField:
        """Sparse scalar field stored as an index grid and per-voxel data.

        Pass :attr:`volume`, :attr:`voxel_data`, and :attr:`background` to
        :func:`warp.volume_sample_index` to sample the field. The voxel-data
        buffer may include reserved capacity; the volume maps index-space
        coordinates to the corresponding live entries.

        Multiple worlds share one packed index grid. To sample world ``i`` at
        world-space position ``position``, pass
        ``position / particle_surface.voxel_size + wp.vec3(world_index_offsets[i])``
        as the ``uvw`` argument to :func:`warp.volume_sample_index`. The offset
        is zero for a single-world surface.

        Reacquire :attr:`ParticleSurface.sparse_field` after updating or
        extracting the field because its underlying storage may be replaced.
        When using preallocated storage, inspect :attr:`per_world_status`
        before consuming results; a nonzero entry means the sparse field may
        be incomplete because its capacity was exceeded.

        """

        volume: wp.Volume
        """NanoVDB index grid defining the sparse field topology."""

        voxel_data: wp.array[float]
        """Scalar feature values indexed by :attr:`volume`."""

        background: float
        """Value used when sampling outside the indexed topology."""

        world_index_offsets: wp.array[wp.vec3i]
        """Offset of each world's coordinates in the packed index grid [voxels],
        shape ``(world_count,)``.
        """

        per_world_status: wp.array[wp.int32]
        """Extraction status per world, shape ``(world_count,)``.

        Zero indicates success; a nonzero value indicates sparse-grid
        overflow.
        """

    class ExtractionMesh:
        """Particle surface mesh and its device-resident logical counts.

        Buffers are exact-sized when ``max_grid_cells`` is ``None`` and
        preallocated otherwise.

        Vertices and indices from each world occupy contiguous ranges. Their
        offsets are stored in :attr:`vertex_world_offsets` and
        :attr:`index_world_offsets`, each with shape ``(world_count + 1,)``.
        World ``i`` occupies the half-open range ``[offsets[i], offsets[i + 1])``.
        """

        def __init__(
            self,
            vertices: wp.array[wp.vec3] | None,
            indices: wp.array[wp.int32] | None,
            normals: wp.array[wp.vec3] | None,
            counts: wp.array[wp.int32],
            active_particle_count: wp.array[wp.int32],
            vertex_world_offsets: wp.array[wp.int32],
            index_world_offsets: wp.array[wp.int32],
            *,
            exact: bool,
        ):
            self.vertices = vertices
            self.indices = indices
            self.normals = normals
            self._counts = counts
            self._active_particle_count: wp.array[wp.int32] = active_particle_count
            self._grid_overflow: wp.array[wp.int32] = counts[2:3]
            self.vertex_world_offsets: wp.array[wp.int32] = vertex_world_offsets
            self.index_world_offsets: wp.array[wp.int32] = index_world_offsets
            self.world_count = vertex_world_offsets.shape[0] - 1
            self._exact = exact

        @classmethod
        def _from_workspace(
            cls,
            workspace: _ParticleSurfaceSparseWorkspace,
            *,
            compute_normals: bool,
            exact: bool,
        ) -> ParticleSurface.ExtractionMesh:
            normals = workspace.normals if compute_normals else None
            return cls(
                workspace.vertices,
                workspace.indices,
                normals,
                workspace.mesh_counts,
                workspace.active_particle_count,
                workspace.vertex_world_offsets,
                workspace.index_world_offsets,
                exact=exact,
            )

        def to_arrays(
            self,
        ) -> tuple[wp.array[wp.vec3] | None, wp.array[wp.int32] | None, wp.array[wp.vec3] | None]:
            """Return one exact-length mesh containing every world as a disconnected component."""
            if self._exact:
                if self.vertices is None or self.indices is None or self.indices.shape[0] == 0:
                    return None, None, None
                return self.vertices, self.indices, self.normals

            counts = self._counts.numpy()
            if int(counts[2]) != 0:
                raise ValueError("Particle surface exceeds configured max_grid_cells")
            vertex_count, index_count = int(counts[0]), int(counts[1])
            if vertex_count == 0 or index_count == 0:
                return None, None, None
            normals = self.normals[:vertex_count] if self.normals is not None else None
            return self.vertices[:vertex_count], self.indices[:index_count], normals

        def __iter__(self):
            """Iterate over exact-length vertex, index, and normal arrays."""
            return iter(self.to_arrays())

    def __init__(
        self,
        voxel_size: float,
        *,
        kernel_radius: float | None = None,
        threshold: float = 0.25,
        smooth_lambda: float = 0.5,
        anisotropic: bool = False,
        anisotropy_ratio: float = 4.0,
        kernel_scale: float = 0.5,
        anisotropy_scale: float = 1.0,
        anisotropy_min_neighbors: int = 25,
        padding: int = 2,
        field_smooth_iterations: int = 0,
        field_smooth_radius: int = 1,
        field_mode: Literal["density", "sdf"] | None = None,
        redistance_iterations: int = 0,
        mesh_smooth_iterations: int = 0,
        mesh_smooth_lambda: float = 1.0,
        device: wp.DeviceLike = None,
        anisotropy_strength: float = 1.0,
        surface_method: Literal["density", "particle_sdf"] = "density",
        particle_sdf_radius_scale: float = 1.0,
        particle_sdf_band: float = 2.0,
        max_grid_cells: int | None = None,
        world_count: int = 1,
        anisotropy_binning: bool = False,
    ):
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if kernel_radius is None:
            kernel_radius = 3.0 * voxel_size
        elif not math.isfinite(kernel_radius) or kernel_radius <= 0.0:
            raise ValueError("kernel_radius must be positive")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if not 0.0 <= smooth_lambda <= 1.0:
            raise ValueError("smooth_lambda must be in [0, 1]")
        if not math.isfinite(anisotropy_ratio) or anisotropy_ratio < 1.0:
            raise ValueError("anisotropy_ratio must be at least 1")
        if not math.isfinite(kernel_scale) or kernel_scale <= 0.0:
            raise ValueError("kernel_scale must be positive")
        if not math.isfinite(anisotropy_scale) or anisotropy_scale <= 0.0:
            raise ValueError("anisotropy_scale must be positive")
        if anisotropy_min_neighbors < 0:
            raise ValueError("anisotropy_min_neighbors must be non-negative")
        if not 0.0 <= anisotropy_strength <= 1.0:
            raise ValueError("anisotropy_strength must be in [0, 1]")
        if padding < 0:
            raise ValueError("padding must be non-negative")
        if field_smooth_iterations < 0:
            raise ValueError("field_smooth_iterations must be non-negative")
        if field_smooth_radius < 0:
            raise ValueError("field_smooth_radius must be non-negative")
        if redistance_iterations < 0:
            raise ValueError("redistance_iterations must be non-negative")
        if mesh_smooth_iterations < 0:
            raise ValueError("mesh_smooth_iterations must be non-negative")
        if not 0.0 <= mesh_smooth_lambda <= 1.0:
            raise ValueError("mesh_smooth_lambda must be in [0, 1]")
        if not math.isfinite(particle_sdf_radius_scale) or particle_sdf_radius_scale <= 0.0:
            raise ValueError("particle_sdf_radius_scale must be positive")
        if not math.isfinite(particle_sdf_band) or particle_sdf_band < 1.0:
            raise ValueError("particle_sdf_band must be at least 1")
        if world_count <= 0:
            raise ValueError("world_count must be positive")
        if surface_method not in ("density", "particle_sdf"):
            raise ValueError(f"Unsupported surface_method {surface_method!r}; expected 'density' or 'particle_sdf'")
        if field_mode is None:
            field_mode = "sdf" if surface_method == "particle_sdf" else "density"
        elif field_mode not in ("density", "sdf"):
            raise ValueError(f"Unsupported field_mode {field_mode!r}; expected 'density' or 'sdf'")
        if surface_method == "particle_sdf" and field_mode != "sdf":
            raise ValueError("surface_method='particle_sdf' requires field_mode='sdf' or field_mode=None")
        if redistance_iterations > 0 and field_mode != "sdf":
            raise ValueError("redistance_iterations requires field_mode='sdf'")

        self._voxel_size = voxel_size
        self._kernel_radius = kernel_radius
        self._anisotropic = anisotropic
        self._threshold = threshold
        self._smooth_lambda = smooth_lambda
        self._anisotropy_ratio = anisotropy_ratio
        self._anisotropy_scale = anisotropy_scale
        self._kernel_scale = kernel_scale
        self._anisotropy_min_neighbors = anisotropy_min_neighbors
        self._anisotropy_binning = anisotropy_binning
        self._anisotropy_strength = anisotropy_strength
        self._surface_method = surface_method
        self._particle_sdf_radius_scale = particle_sdf_radius_scale
        self._particle_sdf_band = particle_sdf_band
        self._padding = padding
        self._field_smooth_iterations = field_smooth_iterations
        self._field_smooth_radius = field_smooth_radius
        self._field_mode = field_mode
        self._redistance_iterations = redistance_iterations
        self._mesh_smooth_iterations = mesh_smooth_iterations
        self._mesh_smooth_lambda = mesh_smooth_lambda
        self._world_count = int(world_count)

        self._device = wp.get_device() if device is None else wp.get_device(device)

        # Cached objects (allocated lazily)
        self._hash_grid: wp.HashGrid | None = None
        self._blur_weights: wp.array[float] | None = None
        self._hash_grid_dim: int = 0
        self._resource_device: wp.Device | None = None

        # Per-particle temporaries
        self._smoothed: wp.array[wp.vec3] | None = None
        self._G: wp.array[wp.mat33] | None = None
        self._det_G: wp.array[float] | None = None
        self._density_reach: wp.array[wp.vec3] | None = None
        self._isotropic_fallback: wp.array[wp.int32] | None = None
        self._hash_positions: wp.array[wp.vec3] | None = None
        self._anisotropy_bin_keys: wp.array[wp.uint64] | None = None
        self._anisotropy_bin_particles: wp.array[wp.int32] | None = None
        self._anisotropy_bin_markers: wp.array[wp.int32] | None = None
        self._anisotropy_bin_indices: wp.array[wp.int32] | None = None
        self._all_particle_flags: wp.array[wp.int32] | None = None
        self._n_particles: int = 0
        self._max_particles: int = 0
        self._max_grid_cells = max_grid_cells
        self._workspace: _ParticleSurfaceSparseWorkspace | None = None
        self._grid_dims: list[tuple[int, int, int]] | None = None
        self._has_field = False

        if max_grid_cells is not None:
            self._configure_grid_workspace(max_grid_cells, device=self._device)

    @property
    def voxel_size(self) -> float:
        """Edge length of each grid voxel [m]."""
        return self._voxel_size

    @property
    def field_mode(self) -> Literal["density", "sdf"]:
        """Field representation retained after extraction."""
        return self._field_mode

    @property
    def world_count(self) -> int:
        """Number of independent particle worlds."""
        return self._world_count

    @property
    def sparse_field(self) -> ParticleSurface.SparseField | None:
        """Current sparse scalar field, or ``None`` before field extraction."""
        if self._workspace is None or not self._has_field or self._workspace.volume is None:
            return None
        return self.SparseField(
            self._workspace.volume,
            self._workspace.field,
            self._field_background(),
            self._workspace.env_offsets,
            self._workspace.per_world_status,
        )

    def _field_background(self) -> float:
        if self._surface_method == "particle_sdf":
            return self._kernel_radius * self._particle_sdf_band
        if self._field_mode == "sdf":
            return self._threshold
        return 0.0

    def _topology_halo_voxels(self) -> int:
        return (
            self._padding + self._field_smooth_iterations * self._field_smooth_radius + self._redistance_iterations + 1
        )

    def _require_sparse_sdf_metadata(self) -> _SparseSdfMetadata:
        """Return internal metadata for the current sparse SDF."""
        if self._field_mode != "sdf":
            raise ValueError("Sparse SDF access requires ParticleSurface(field_mode='sdf')")
        if self._workspace is None or not self._has_field or self._workspace.volume is None:
            raise ValueError("Particle surface field has not been extracted")
        topology_halo = self._topology_halo_voxels() * self._voxel_size
        return _SparseSdfMetadata(self._workspace, topology_halo)

    @property
    def _grid_node_world_start(self) -> wp.array[wp.int32] | None:
        return None if self._workspace is None else self._workspace.grid_node_world_start

    @property
    def _grid_cell_world_start(self) -> wp.array[wp.int32] | None:
        return None if self._workspace is None else self._workspace.grid_cell_world_start

    @property
    def _field(self) -> wp.array[wp.float32] | None:
        if self._world_count != 1:
            raise RuntimeError("Use _field_for_world() for a multi-world surface")
        return self._field_for_world(0)

    def _field_for_world(self, world: int) -> wp.array[wp.float32] | None:
        if self._workspace is None or not self._has_field:
            return None
        if world < 0 or world >= self._world_count:
            raise IndexError(f"world index {world} is out of range for {self._world_count} worlds")
        counts = self._workspace.grid_counts.numpy().reshape(self._world_count, 7)[world]
        if int(counts[3]) != 0:
            raise ValueError("Particle surface exceeds configured max_grid_cells")
        starts = self._workspace.grid_node_world_start.numpy()
        begin = int(starts[world])
        end = int(starts[world + 1])
        if end == begin:
            return None
        return self._workspace.field[begin:end]

    @property
    def _sparse_volume(self) -> wp.Volume | None:
        return None if self._workspace is None else self._workspace.volume

    @property
    def _grid_origin_value(self) -> wp.vec3 | None:
        if self._world_count != 1:
            raise RuntimeError("Use _grid_origin_for_world() for a multi-world surface")
        return self._grid_origin_for_world(0)

    def _grid_origin_for_world(self, world: int) -> wp.vec3 | None:
        if self._workspace is None or not self._has_field:
            return None
        if world < 0 or world >= self._world_count:
            raise IndexError(f"world index {world} is out of range for {self._world_count} worlds")
        return wp.vec3(self._workspace.grid_origin.numpy()[world])

    @property
    def _grid_dims_value(self) -> tuple[int, int, int] | None:
        if self._world_count != 1:
            raise RuntimeError("Use _grid_dims_for_world() for a multi-world surface")
        return self._grid_dims_for_world(0)

    def _grid_dims_for_world(self, world: int) -> tuple[int, int, int] | None:
        if world < 0 or world >= self._world_count:
            raise IndexError(f"world index {world} is out of range for {self._world_count} worlds")
        if self._max_grid_cells is None:
            return None if self._grid_dims is None else self._grid_dims[world]
        if self._workspace is None or not self._has_field:
            return None
        counts = self._workspace.grid_counts.numpy().reshape(self._world_count, 7)[world]
        return tuple(int(value) for value in counts[4:7])

    @property
    def _smoothed_positions(self) -> wp.array[wp.vec3] | None:
        return self._smoothed

    def _configure_grid_workspace(
        self,
        max_grid_cells: int,
        device: wp.DeviceLike = None,
    ) -> ParticleSurface:
        """Preallocate the graph-capturable extraction workspace."""
        device_obj = self._device if device is None else wp.get_device(device)
        self._clear_device_resources()
        self._max_grid_cells = max_grid_cells
        self._device = device_obj
        self._resource_device = device_obj
        self._workspace = _ParticleSurfaceSparseWorkspace(
            max_grid_cells=max_grid_cells,
            world_count=self._world_count,
            voxel_size=self._voxel_size,
            padding=self._padding,
            support_leaf_radius=self._density_support_leaf_radius(),
            device=device_obj,
        )
        self._has_field = False

        hash_grid_dim = max(16, int(math.ceil(max_grid_cells ** (1.0 / 3.0))))
        self._hash_grid = wp.HashGrid(hash_grid_dim, hash_grid_dim, hash_grid_dim, device=device_obj)
        self._hash_grid_dim = hash_grid_dim
        if self._field_smooth_iterations > 0 and self._field_smooth_radius > 0:
            self._ensure_blur_weights(device_obj)
        return self

    def _density_support_leaf_radius(self) -> int | None:
        if self._surface_method != "density":
            return None

        axis_scale = 1.0
        if self._anisotropic and self._anisotropy_strength > 0.0 and self._anisotropy_ratio > 1.0:
            # Geometric-mean normalization bounds the longest WPCA axis by ratio**(2/3).
            # Apply the same inverse-radius blend as the anisotropy kernels.
            min_relative_inverse_radius = (1.0 - self._anisotropy_strength) + self._anisotropy_strength / (
                self._anisotropy_scale * self._anisotropy_ratio ** (2.0 / 3.0)
            )
            axis_scale = max(axis_scale, 1.0 / min_relative_inverse_radius)
        max_reach_voxels = (
            kernels._DENSITY_KERNEL_SUPPORT * self._kernel_scale * self._kernel_radius * axis_scale / self._voxel_size
        )
        # Half-reach octant samples leave at most half the support width between samples.
        voxel_radius = int(math.ceil(0.5 * max_reach_voxels)) + 1
        return (voxel_radius + 7) // 8

    def update_field(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        *,
        particle_flags: wp.array[wp.int32] | None = None,
        particle_world: wp.array[wp.int32] | None = None,
    ) -> ParticleSurface.SparseField | None:
        """Update the scalar field without extracting a mesh.

        Args:
            positions: Particle positions [m], shape ``(N,)``, dtype ``wp.vec3``.
            radii: Per-particle radii [m], shape ``(N,)``, dtype ``wp.float32``.
            particle_flags: Optional per-particle flags. Particles without
                :attr:`~newton.ParticleFlags.ACTIVE` are skipped.
            particle_world: Optional world index per particle. Particles with
                negative or out-of-range world indices are skipped.

        Returns:
            The sparse index grid and its per-voxel data, or ``None`` when no
            sparse field topology was produced.
        """
        self._extract(
            positions,
            radii,
            compute_normals=False,
            particle_flags=particle_flags,
            particle_world=particle_world,
            compute_mesh=False,
        )
        return self.sparse_field

    # -- Core extraction --

    def extract(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        *,
        compute_normals: bool = True,
        particle_flags: wp.array[wp.int32] | None = None,
        particle_world: wp.array[wp.int32] | None = None,
    ) -> ParticleSurface.ExtractionMesh:
        """Extract a triangle mesh from particle positions.

        When ``max_grid_cells`` is set, this method performs no host
        synchronization and can be captured in a CUDA graph. Otherwise it
        allocates exact-size field and mesh arrays.

        Args:
            positions: Particle positions [m], shape ``(N,)``, dtype ``wp.vec3``.
            radii: Per-particle radii [m], shape ``(N,)``, dtype ``wp.float32``.
            compute_normals: Whether to compute per-vertex normals.
            particle_flags: Optional per-particle flags.  Particles without
                :attr:`~newton.ParticleFlags.ACTIVE` are skipped.
            particle_world: Optional world index per particle. Particles with
                negative or out-of-range world indices are skipped.

        Returns:
            Mesh buffers and device-resident logical counts.
        """
        return self._extract(
            positions,
            radii,
            compute_normals=compute_normals,
            particle_flags=particle_flags,
            particle_world=particle_world,
            compute_mesh=True,
        )

    def _extract(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        *,
        compute_normals: bool = True,
        particle_flags: wp.array[wp.int32] | None = None,
        particle_world: wp.array[wp.int32] | None = None,
        compute_mesh: bool = True,
    ) -> ParticleSurface.ExtractionMesh:
        self._validate_positions_layout(positions)
        particle_count = positions.shape[0]
        device = positions.device
        self._validate_radii_layout(positions, radii, particle_count)
        self._validate_particle_flags_layout(particle_flags, particle_count, device)
        self._validate_particle_world_layout(particle_world, particle_count, device)
        return self._extract_impl(
            positions,
            radii,
            compute_normals=compute_normals,
            particle_flags=particle_flags,
            particle_world=particle_world,
            compute_mesh=compute_mesh,
        )

    def redistance(self, iterations: int) -> None:
        """Apply Eikonal redistancing to the current SDF field.

        Args:
            iterations: Number of redistancing iterations.
        """
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self._field_mode != "sdf":
            raise ValueError("SDF redistancing requires field_mode='sdf'")
        if self._workspace is None or not self._has_field:
            return
        if iterations == 0:
            return
        self._workspace.redistance(iterations, outside_value=self._field_background())

    def resurface(
        self,
        *,
        compute_normals: bool = True,
    ) -> ParticleSurface.ExtractionMesh:
        """Re-run marching cubes on the current field.

        Args:
            compute_normals: Whether to compute per-vertex normals.

        Returns:
            Mesh buffers and device-resident logical counts.
        """
        if self._workspace is None or not self._has_field:
            raise RuntimeError("extract() must populate the field before resurfacing")
        if self._workspace.volume is None:
            self._workspace.reset_mesh_counts()
            self._workspace.compute_mesh_world_starts()
            result = self.ExtractionMesh._from_workspace(
                self._workspace,
                compute_normals=compute_normals,
                exact=self._max_grid_cells is None,
            )
            return result
        return self._extract_current_mesh(
            self._workspace,
            compute_normals=compute_normals,
            exact=self._max_grid_cells is None,
        )

    # -- Internal helpers --

    def _extract_impl(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        *,
        compute_normals: bool,
        particle_flags: wp.array[wp.int32] | None,
        particle_world: wp.array[wp.int32] | None,
        compute_mesh: bool,
    ) -> ParticleSurface.ExtractionMesh:
        particle_count = positions.shape[0]
        device = positions.device
        device_obj = wp.get_device(device)
        exact = self._max_grid_cells is None
        if exact:
            if device_obj != self._resource_device:
                self._clear_device_resources()
                self._device = device_obj
                self._resource_device = device_obj
            hash_grid_dim = max(16, int(math.ceil(max(particle_count, 1) ** (1.0 / 3.0))))
            self._ensure_hash_grid(hash_grid_dim, device)
            self._workspace = _ParticleSurfaceSparseWorkspace(
                max_grid_cells=None,
                world_count=self._world_count,
                voxel_size=self._voxel_size,
                padding=self._padding,
                support_leaf_radius=None,
                device=device,
            )
            workspace = self._workspace
        else:
            if device_obj != self._resource_device:
                self._configure_grid_workspace(self._max_grid_cells, device=device)
            workspace = self._workspace

        self._ensure_workspace_particle_resources(particle_count)
        flags, use_flags = self._field_flag_args(particle_flags, particle_count, device)
        worlds = particle_world if particle_world is not None else flags
        use_worlds = int(particle_world is not None)
        workspace.reset()
        sentinel_distance = 1.0e6 * max(self._kernel_radius, self._voxel_size)
        workspace.compute_particle_bounds(
            positions,
            flags,
            use_flags,
            worlds,
            use_worlds,
            sentinel_distance,
        )
        self._prepare_particle_values(
            workspace,
            positions,
            particle_count,
            flags,
            use_flags,
            worlds,
            use_worlds,
            device,
        )

        isotropic_sdf = not self._anisotropic or self._anisotropy_strength <= 0.0 or self._anisotropy_ratio <= 1.0
        topology_halo_voxels = self._topology_halo_voxels()
        workspace.compute_grid(
            self._smoothed[:particle_count],
            radii,
            flags,
            use_flags,
            worlds,
            use_worlds,
            self._det_G[:particle_count],
            self._density_reach[:particle_count],
            self._particle_sdf_radius_scale,
            self._particle_sdf_band,
            self._surface_method == "particle_sdf",
            not isotropic_sdf,
            topology_halo_voxels,
        )
        if exact:
            grid_counts = workspace.grid_counts.numpy().reshape(self._world_count, 7)
            self._grid_dims = [tuple(int(value) for value in counts[4:7]) for counts in grid_counts]
            if int(np.sum(grid_counts[:, 2])) == 0:
                self._has_field = True
                workspace.reset_mesh_counts()
                workspace.compute_mesh_world_starts()
                result = self.ExtractionMesh._from_workspace(
                    workspace,
                    compute_normals=compute_normals,
                    exact=True,
                )
                return result
        else:
            self._grid_dims = None

        workspace.build_topology(
            self._smoothed[:particle_count],
            radii,
            flags,
            use_flags,
            worlds,
            use_worlds,
            self._det_G[:particle_count],
            self._density_reach[:particle_count],
            self._particle_sdf_radius_scale,
            self._particle_sdf_band,
            self._surface_method == "particle_sdf",
            not isotropic_sdf,
            topology_halo_voxels,
            self._G[:particle_count],
        )

        if self._field_smooth_iterations > 0 and self._field_smooth_radius > 0:
            self._ensure_blur_weights(workspace.device)
        workspace.evaluate_field(
            self._smoothed[:particle_count],
            radii,
            flags,
            use_flags,
            worlds,
            use_worlds,
            self._G[:particle_count],
            self._det_G[:particle_count],
            self._density_reach[:particle_count],
            surface_method=self._surface_method,
            anisotropic=not isotropic_sdf,
            particle_sdf_radius_scale=self._particle_sdf_radius_scale,
            particle_sdf_band=self._particle_sdf_band,
            kernel_radius=self._kernel_radius,
            field_mode=self._field_mode,
            threshold=self._threshold,
            blur_weights=self._blur_weights,
            blur_radius=self._field_smooth_radius,
            blur_iterations=self._field_smooth_iterations,
            redistance_iterations=self._redistance_iterations,
        )
        self._has_field = True
        if compute_mesh:
            return self._extract_current_mesh(workspace, compute_normals=compute_normals, exact=exact)

        workspace.reset_mesh_counts()
        workspace.compute_mesh_world_starts()
        result = self.ExtractionMesh._from_workspace(
            workspace,
            compute_normals=compute_normals,
            exact=exact,
        )
        return result

    def _extract_current_mesh(
        self,
        workspace: _ParticleSurfaceSparseWorkspace,
        *,
        compute_normals: bool,
        exact: bool,
    ) -> ParticleSurface.ExtractionMesh:
        threshold = self._marching_threshold()
        counts_precomputed = exact or self._world_count > 1
        if counts_precomputed:
            workspace.count_mesh(threshold)
        if exact:
            mesh_counts = workspace.mesh_counts.numpy()
            workspace.resize_mesh_exact(int(mesh_counts[0]), int(mesh_counts[1]))

        workspace.extract_mesh(
            threshold,
            counts_precomputed=counts_precomputed,
            flip_winding=self._field_mode == "density",
            smooth_iterations=self._mesh_smooth_iterations,
            smooth_lambda=self._mesh_smooth_lambda,
            compute_normals=compute_normals,
        )
        result = self.ExtractionMesh._from_workspace(
            workspace,
            compute_normals=compute_normals,
            exact=exact,
        )
        return result

    def _clear_device_resources(self):
        self._workspace = None
        self._grid_dims = None
        self._has_field = False
        self._hash_grid = None
        self._blur_weights = None
        self._hash_grid_dim = 0
        self._smoothed = None
        self._G = None
        self._det_G = None
        self._density_reach = None
        self._isotropic_fallback = None
        self._hash_positions = None
        self._anisotropy_bin_keys = None
        self._anisotropy_bin_particles = None
        self._anisotropy_bin_markers = None
        self._anisotropy_bin_indices = None
        self._all_particle_flags = None
        self._n_particles = 0
        self._max_particles = 0

    def _prepare_particle_values(
        self,
        workspace: _ParticleSurfaceSparseWorkspace,
        positions: wp.array[wp.vec3],
        particle_count: int,
        flags: wp.array[wp.int32],
        use_flags: int,
        particle_world: wp.array[wp.int32],
        use_worlds: int,
        device: wp.DeviceLike,
    ) -> None:
        smoothed = self._smoothed[:particle_count]
        G = self._G[:particle_count]
        det_G = self._det_G[:particle_count]
        density_reach = self._density_reach[:particle_count]
        isotropic_fallback = self._isotropic_fallback[:particle_count]
        hash_positions = positions
        needs_smoothing_hash = self._smooth_lambda > 1.0e-6

        if use_worlds != 0 and particle_count > 0:
            hash_positions = self._hash_positions[:particle_count]
            wp.launch(
                kernels.compute_hash_positions,
                dim=particle_count,
                inputs=[
                    positions,
                    flags,
                    use_flags,
                    particle_world,
                    use_worlds,
                    self._world_count,
                    workspace.lower,
                    workspace.hash_spacing,
                    workspace.inactive_position,
                    hash_positions,
                ],
                device=device,
            )
        elif (use_flags != 0 or needs_smoothing_hash) and particle_count > 0:
            hash_positions = self._hash_positions[:particle_count]
            wp.launch(
                kernels.copy_active_or_sentinel_positions,
                dim=particle_count,
                inputs=[
                    positions,
                    flags,
                    use_flags,
                    particle_world,
                    use_worlds,
                    self._world_count,
                    workspace.inactive_position,
                    hash_positions,
                ],
                device=device,
            )

        if self._smooth_lambda > 1.0e-6 and particle_count > 0:
            self._hash_grid.build(hash_positions, self._kernel_radius)
            if hash_positions is not positions:
                wp.launch(
                    kernels.smooth_positions_flagged,
                    dim=particle_count,
                    inputs=[
                        self._hash_grid.id,
                        positions,
                        hash_positions,
                        flags,
                        use_flags,
                        particle_world,
                        use_worlds,
                        self._world_count,
                        workspace.inactive_position,
                        self._kernel_radius,
                        self._smooth_lambda,
                        smoothed,
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    kernels._smooth_positions,
                    dim=particle_count,
                    inputs=[
                        self._hash_grid.id,
                        hash_positions,
                        self._kernel_radius,
                        self._smooth_lambda,
                        smoothed,
                    ],
                    device=device,
                )
        elif particle_count > 0:
            if use_flags != 0 or use_worlds != 0 or self._anisotropic:
                wp.launch(
                    kernels.copy_active_or_sentinel_positions,
                    dim=particle_count,
                    inputs=[
                        positions,
                        flags,
                        use_flags,
                        particle_world,
                        use_worlds,
                        self._world_count,
                        workspace.inactive_position,
                        smoothed,
                    ],
                    device=device,
                )
            else:
                wp.copy(smoothed, positions)

        if self._anisotropic and particle_count > 0:
            anisotropy_hash_positions = self._hash_positions[:particle_count]
            if use_worlds != 0:
                wp.launch(
                    kernels.compute_hash_positions,
                    dim=particle_count,
                    inputs=[
                        smoothed,
                        flags,
                        use_flags,
                        particle_world,
                        use_worlds,
                        self._world_count,
                        workspace.lower,
                        workspace.hash_spacing,
                        workspace.inactive_position,
                        anisotropy_hash_positions,
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    kernels.copy_active_or_sentinel_positions,
                    dim=particle_count,
                    inputs=[
                        smoothed,
                        flags,
                        use_flags,
                        particle_world,
                        use_worlds,
                        self._world_count,
                        workspace.inactive_position,
                        anisotropy_hash_positions,
                    ],
                    device=device,
                )
            self._hash_grid.build(anisotropy_hash_positions, self._kernel_radius)
            if self._anisotropy_binning:
                self._prepare_binned_anisotropy(
                    workspace,
                    smoothed,
                    particle_count,
                    flags,
                    use_flags,
                    particle_world,
                    use_worlds,
                    G,
                    det_G,
                    density_reach,
                    isotropic_fallback,
                    device,
                )
            elif use_worlds != 0:
                wp.launch(
                    kernels._compute_anisotropy_worlds,
                    dim=particle_count,
                    inputs=[
                        self._hash_grid.id,
                        smoothed,
                        anisotropy_hash_positions,
                        flags,
                        use_flags,
                        particle_world,
                        use_worlds,
                        self._world_count,
                        self._kernel_radius,
                        self._anisotropy_ratio,
                        self._anisotropy_scale,
                        self._kernel_scale,
                        self._anisotropy_min_neighbors,
                        self._anisotropy_strength,
                        G,
                        det_G,
                        density_reach,
                        isotropic_fallback,
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    kernels._compute_anisotropy,
                    dim=particle_count,
                    inputs=[
                        self._hash_grid.id,
                        smoothed,
                        flags,
                        use_flags,
                        self._kernel_radius,
                        self._anisotropy_ratio,
                        self._anisotropy_scale,
                        self._kernel_scale,
                        self._anisotropy_min_neighbors,
                        self._anisotropy_strength,
                        G,
                        det_G,
                        density_reach,
                        isotropic_fallback,
                    ],
                    device=device,
                )
        elif particle_count > 0:
            wp.launch(
                kernels._fill_isotropic_G,
                dim=particle_count,
                inputs=[
                    self._kernel_radius,
                    self._kernel_scale,
                    flags,
                    use_flags,
                    particle_world,
                    use_worlds,
                    self._world_count,
                    G,
                    det_G,
                    density_reach,
                    isotropic_fallback,
                ],
                device=device,
            )

    def _prepare_binned_anisotropy(
        self,
        workspace: _ParticleSurfaceSparseWorkspace,
        smoothed: wp.array[wp.vec3],
        particle_count: int,
        flags: wp.array[wp.int32],
        use_flags: int,
        particle_world: wp.array[wp.int32],
        use_worlds: int,
        G: wp.array[wp.mat33],
        det_G: wp.array[float],
        density_reach: wp.array[wp.vec3],
        isotropic_fallback: wp.array[wp.int32],
        device: wp.DeviceLike,
    ) -> None:
        keys = self._anisotropy_bin_keys
        particles = self._anisotropy_bin_particles
        bin_markers = self._anisotropy_bin_markers[:particle_count]
        bin_indices = self._anisotropy_bin_indices[:particle_count]
        bin_starts = particles[particle_count : 2 * particle_count]
        bin_size = 0.5 * self._kernel_radius

        wp.launch(
            kernels.build_anisotropy_bin_keys,
            dim=particle_count,
            inputs=[
                smoothed,
                flags,
                use_flags,
                particle_world,
                use_worlds,
                self._world_count,
                workspace.lower,
                1.0 / bin_size,
                self._kernel_radius,
                self._kernel_scale,
                keys,
                particles,
                G,
                det_G,
                density_reach,
                isotropic_fallback,
            ],
            device=device,
        )
        wp.utils.radix_sort_pairs(keys, particles, particle_count)
        wp.launch(
            kernels.mark_anisotropy_bins,
            dim=particle_count,
            inputs=[keys, bin_markers],
            device=device,
        )
        wp.utils.array_scan(bin_markers, bin_indices, inclusive=True)
        wp.launch(
            kernels.compact_anisotropy_bins,
            dim=particle_count,
            inputs=[bin_markers, bin_indices, bin_starts],
            device=device,
        )
        wp.launch(
            kernels._compute_anisotropy_bins,
            dim=particle_count,
            inputs=[
                self._hash_grid.id,
                smoothed,
                flags,
                use_flags,
                particle_world,
                use_worlds,
                self._world_count,
                keys,
                particles,
                bin_starts,
                bin_indices,
                particle_count,
                workspace.lower,
                workspace.hash_spacing,
                bin_size,
                self._kernel_radius,
                self._anisotropy_ratio,
                self._anisotropy_scale,
                self._kernel_scale,
                self._anisotropy_min_neighbors,
                self._anisotropy_strength,
                G,
                det_G,
                density_reach,
                isotropic_fallback,
            ],
            device=device,
        )

    def _ensure_workspace_particle_resources(self, particle_count: int) -> None:
        if particle_count <= self._max_particles and self._smoothed is not None:
            return
        alloc_particles = max(particle_count, 1)
        self._smoothed = wp.empty(alloc_particles, dtype=wp.vec3, device=self._device)
        self._G = wp.empty(alloc_particles, dtype=wp.mat33, device=self._device)
        self._det_G = wp.empty(alloc_particles, dtype=float, device=self._device)
        self._density_reach = wp.empty(alloc_particles, dtype=wp.vec3, device=self._device)
        self._isotropic_fallback = wp.empty(alloc_particles, dtype=wp.int32, device=self._device)
        self._hash_positions = wp.empty(alloc_particles, dtype=wp.vec3, device=self._device)
        if self._anisotropy_binning:
            self._anisotropy_bin_keys = wp.empty(2 * alloc_particles, dtype=wp.uint64, device=self._device)
            self._anisotropy_bin_particles = wp.empty(2 * alloc_particles, dtype=wp.int32, device=self._device)
            self._anisotropy_bin_markers = wp.empty(alloc_particles, dtype=wp.int32, device=self._device)
            self._anisotropy_bin_indices = wp.empty(alloc_particles, dtype=wp.int32, device=self._device)
        self._all_particle_flags = wp.empty(alloc_particles, dtype=wp.int32, device=self._device)
        self._n_particles = alloc_particles
        self._max_particles = particle_count

    def _ensure_hash_grid(self, dimension: int, device: wp.DeviceLike) -> None:
        if self._hash_grid is None or self._hash_grid_dim != dimension:
            self._hash_grid = wp.HashGrid(dimension, dimension, dimension, device=device)
            self._hash_grid_dim = dimension

    def _field_flag_args(
        self,
        particle_flags: wp.array[wp.int32] | None,
        n: int,
        device: wp.DeviceLike,
    ) -> tuple[wp.array[wp.int32], int]:
        if particle_flags is not None:
            return particle_flags, 1
        return self._ensure_all_particle_flags(n, device), 0

    def _ensure_all_particle_flags(self, n: int, device: wp.DeviceLike) -> wp.array[wp.int32]:
        alloc_particles = max(n, 1)
        if (
            self._all_particle_flags is None
            or self._all_particle_flags.shape[0] < alloc_particles
            or self._all_particle_flags.device != wp.get_device(device)
        ):
            self._all_particle_flags = wp.empty(alloc_particles, dtype=wp.int32, device=device)
        return self._all_particle_flags

    def _validate_particle_flags_layout(
        self,
        particle_flags: wp.array[wp.int32] | None,
        n: int,
        device: wp.DeviceLike,
    ):
        if particle_flags is None:
            return
        if particle_flags.ndim != 1:
            raise ValueError(f"particle_flags must be a 1-D array, got shape {particle_flags.shape}")
        if particle_flags.shape[0] != n:
            raise ValueError(f"particle_flags length ({particle_flags.shape[0]}) must match positions length ({n})")
        if particle_flags.device != wp.get_device(device):
            raise ValueError(f"particle_flags device ({particle_flags.device}) must match positions device ({device})")
        if particle_flags.dtype != wp.int32:
            raise TypeError(f"particle_flags must have dtype wp.int32, got {particle_flags.dtype}")

    def _validate_particle_world_layout(
        self,
        particle_world: wp.array[wp.int32] | None,
        n: int,
        device: wp.DeviceLike,
    ) -> None:
        if particle_world is None:
            return
        if particle_world.ndim != 1:
            raise ValueError(f"particle_world must be a 1-D array, got shape {particle_world.shape}")
        if particle_world.shape[0] != n:
            raise ValueError(f"particle_world length ({particle_world.shape[0]}) must match positions length ({n})")
        if particle_world.device != wp.get_device(device):
            raise ValueError(f"particle_world device ({particle_world.device}) must match positions device ({device})")
        if particle_world.dtype != wp.int32:
            raise TypeError(f"particle_world must have dtype wp.int32, got {particle_world.dtype}")

    def _ensure_blur_weights(self, device: wp.DeviceLike):
        hw = self._field_smooth_radius
        if hw <= 0:
            return
        device = wp.get_device(device)
        if (
            self._blur_weights is not None
            and self._blur_weights.shape[0] == hw + 1
            and self._blur_weights.device == device
        ):
            return
        sigma = max(hw / 2.0, 0.5)
        w = np.array([math.exp(-0.5 * (d / sigma) ** 2) for d in range(hw + 1)], dtype=np.float32)
        w /= w[0] + 2.0 * np.sum(w[1:])
        self._blur_weights = wp.array(w, dtype=float, device=device)

    def _marching_threshold(self) -> float:
        if self._field_mode == "sdf":
            effective_threshold = 0.0
            if self._mesh_smooth_iterations > 0:
                shrink = (
                    _MESH_SMOOTH_SHRINK_PER_VOXEL
                    * math.sqrt(float(self._mesh_smooth_iterations))
                    * self._mesh_smooth_lambda
                    * self._voxel_size
                )
                if self._surface_method == "particle_sdf" or self._redistance_iterations > 0:
                    effective_threshold = shrink
                else:
                    effective_threshold = shrink / self._kernel_radius
            return effective_threshold

        effective_threshold = self._threshold
        if self._mesh_smooth_iterations > 0:
            shrink = (
                _MESH_SMOOTH_SHRINK_PER_VOXEL
                * math.sqrt(float(self._mesh_smooth_iterations))
                * self._mesh_smooth_lambda
                * self._voxel_size
            )
            effective_threshold = max(self._threshold - shrink / self._kernel_radius, _MIN_DENSITY_MARCHING_THRESHOLD)
        return effective_threshold

    def _validate_radii_layout(self, positions: wp.array[wp.vec3], radii: wp.array[float], n: int):
        if not isinstance(radii, wp.array):
            raise TypeError(f"radii must be a Warp array, got {type(radii).__name__}")
        if radii.ndim != 1:
            raise ValueError(f"radii must be a 1-D array, got shape {radii.shape}")
        if radii.shape[0] != n:
            raise ValueError(f"radii length ({radii.shape[0]}) must match positions length ({n})")
        if radii.device != positions.device:
            raise ValueError(f"radii device ({radii.device}) must match positions device ({positions.device})")
        if radii.dtype != wp.float32:
            raise TypeError(f"radii must have dtype wp.float32, got {radii.dtype}")

    def _validate_positions_layout(self, positions: wp.array[wp.vec3]):
        if not isinstance(positions, wp.array):
            raise TypeError(f"positions must be a Warp array, got {type(positions).__name__}")
        if positions.ndim != 1:
            raise ValueError(f"positions must be a 1-D array, got shape {positions.shape}")
        if positions.dtype != wp.vec3:
            raise TypeError(f"positions must have dtype wp.vec3, got {positions.dtype}")


def extract_particle_surface(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    voxel_size: float,
    *,
    max_grid_cells: int | None = None,
    kernel_radius: float | None = None,
    threshold: float = 0.25,
    smooth_lambda: float = 0.5,
    mesh_smooth_iterations: int = 0,
    compute_normals: bool = True,
    anisotropic: bool = False,
    field_mode: Literal["density", "sdf"] | None = None,
    redistance_iterations: int = 0,
    particle_flags: wp.array[wp.int32] | None = None,
    particle_world: wp.array[wp.int32] | None = None,
    world_count: int = 1,
    anisotropy_ratio: float = 4.0,
    kernel_scale: float = 0.5,
    anisotropy_scale: float = 1.0,
    anisotropy_min_neighbors: int = 25,
    anisotropy_binning: bool = False,
    anisotropy_strength: float = 1.0,
    field_smooth_iterations: int = 0,
    field_smooth_radius: int = 1,
    surface_method: Literal["density", "particle_sdf"] = "density",
    particle_sdf_radius_scale: float = 1.0,
    particle_sdf_band: float = 2.0,
) -> ParticleSurface.ExtractionMesh:
    """Extract a triangle mesh from particle positions (one-shot convenience).

    Args:
        positions: Particle positions [m], shape ``(N,)``, dtype ``wp.vec3``.
        radii: Per-particle radii [m], shape ``(N,)``, dtype ``wp.float32``.
        voxel_size: Edge length of each grid voxel [m].
        max_grid_cells: Maximum active sparse-grid cell count across all worlds.
            When set, extraction uses graph-capturable preallocated buffers.
            When ``None``, it uses tight sparse allocations. Rebuildable
            topology-node capacities assume spatially coherent surface bands;
            highly scattered fields can exhaust them before reaching this cell
            count.
        kernel_radius: Search radius [m].  Defaults to ``3 * voxel_size``.
        threshold: Isosurface level.
        smooth_lambda: Position smoothing blend factor [0, 1].
        mesh_smooth_iterations: Laplacian mesh smoothing passes.
        compute_normals: Whether to compute per-vertex normals.
        anisotropic: Enable per-particle WPCA anisotropic kernels.
        field_mode: Field representation retained after extraction.  Defaults
            to ``"sdf"`` for ``surface_method="particle_sdf"`` and ``"density"``
            otherwise.
        redistance_iterations: Number of Eikonal redistancing iterations
            applied when ``field_mode="sdf"``.
        particle_flags: Optional per-particle flags.  Particles without
            :attr:`~newton.ParticleFlags.ACTIVE` are skipped.
        particle_world: Optional world index per particle.
        world_count: Number of independent particle worlds to extract.
        anisotropy_ratio: Maximum anisotropic kernel axis ratio.
        kernel_scale: Kernel radius multiplier relative to ``kernel_radius``.
        anisotropy_scale: Relative multiplier for anisotropic kernel radii.
        anisotropy_min_neighbors: Minimum number of other particles required
            for anisotropic kernels.
        anisotropy_binning: Share one WPCA kernel among particles in each
            spatial bin. Bins are half the kernel radius wide. This reduces
            neighbor-query cost at the expense of spatial resolution.
        anisotropy_strength: Blend from isotropic kernels to anisotropic
            kernels [0, 1].
        field_smooth_iterations: Number of separable Gaussian blur passes
            applied to the scalar field before marching cubes.
        field_smooth_radius: Half-width of the Gaussian blur in voxels.
        surface_method: Surface reconstruction method. ``"density"`` uses
            anisotropic density splatting. ``"particle_sdf"`` directly unions
            per-particle anisotropic ellipsoid SDFs.
        particle_sdf_radius_scale: Radius multiplier for ``surface_method="particle_sdf"``.
        particle_sdf_band: Narrow-band half-width in normalized ellipsoid
            coordinates for ``surface_method="particle_sdf"``. Must be at least
            1 so the band contains the zero level set.

    Returns:
        Mesh buffers and device-resident logical counts.
    """
    ctx = ParticleSurface(
        voxel_size=voxel_size,
        max_grid_cells=max_grid_cells,
        kernel_radius=kernel_radius,
        threshold=threshold,
        smooth_lambda=smooth_lambda,
        anisotropic=anisotropic,
        anisotropy_ratio=anisotropy_ratio,
        anisotropy_scale=anisotropy_scale,
        kernel_scale=kernel_scale,
        anisotropy_min_neighbors=anisotropy_min_neighbors,
        anisotropy_binning=anisotropy_binning,
        anisotropy_strength=anisotropy_strength,
        field_smooth_iterations=field_smooth_iterations,
        field_smooth_radius=field_smooth_radius,
        surface_method=surface_method,
        particle_sdf_radius_scale=particle_sdf_radius_scale,
        particle_sdf_band=particle_sdf_band,
        field_mode=field_mode,
        redistance_iterations=redistance_iterations,
        mesh_smooth_iterations=mesh_smooth_iterations,
        world_count=world_count,
        device=positions.device,
    )
    return ctx.extract(
        positions,
        radii=radii,
        compute_normals=compute_normals,
        particle_flags=particle_flags,
        particle_world=particle_world,
    )
