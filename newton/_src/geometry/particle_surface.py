# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle surface extraction using anisotropic kernels and marching cubes.

Implements the method from Yu & Turk, "Reconstructing Surfaces of
Particle-Based Fluids Using Anisotropic Kernels", Eurographics/ACM SIGGRAPH
Symposium on Computer Animation, 2010.

The pipeline computes per-particle anisotropy matrices via Weighted PCA,
then evaluates a smooth scalar field on a regular grid using oriented
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
from typing import Any, Literal

import numpy as np
import warp as wp
import warp.fem as fem

from . import particle_surface_kernels as kernels

__all__ = ["ParticleSurface", "extract_particle_surface"]

_MESH_SMOOTH_SHRINK_PER_VOXEL = 0.15
_MIN_DENSITY_MARCHING_THRESHOLD = 0.01

# ---------------------------------------------------------------------------
# ParticleSurface context
# ---------------------------------------------------------------------------


class ParticleSurfaceCapacity:
    """Shared storage and launches for particle surface extraction."""

    def __init__(
        self,
        max_grid_cells: int,
        voxel_size: float,
        padding: int,
        device: wp.DeviceLike,
        *,
        max_grid_nodes: int | None = None,
        allocate_field: bool = True,
        allocate_mesh: bool = True,
    ):
        if max_grid_cells <= 0:
            raise ValueError("max_grid_cells must be positive")

        self.max_grid_cells = int(max_grid_cells)
        self.voxel_size = float(voxel_size)
        self.padding = int(padding)
        self.device = wp.get_device(device)

        # For positive integer cell dimensions with product C,
        # (nx + 1)(ny + 1)(nz + 1) <= 4(C + 1).
        self.max_grid_nodes = 4 * (self.max_grid_cells + 1) if max_grid_nodes is None else int(max_grid_nodes)
        self.max_vertices = 3 * self.max_grid_nodes if allocate_mesh else 0
        self.max_indices = 15 * self.max_grid_cells if allocate_mesh else 0
        self.launch_threads = min(max(self.max_grid_nodes, 1), kernels._MAX_CAPACITY_LAUNCH_THREADS)

        self.lower = wp.empty(1, dtype=wp.vec3, device=self.device)
        self.upper = wp.empty(1, dtype=wp.vec3, device=self.device)
        self.inactive_position = wp.empty(1, dtype=wp.vec3, device=self.device)
        self.grid_origin = wp.empty(1, dtype=wp.vec3, device=self.device)
        self.grid_dims = wp.empty(1, dtype=wp.vec3i, device=self.device)
        self.grid_counts = wp.zeros(7, dtype=wp.int32, device=self.device)
        self.mesh_counts = wp.zeros(3, dtype=wp.int32, device=self.device)

        field_size = self.max_grid_nodes if allocate_field else 0
        self.field = wp.empty(field_size, dtype=wp.float32, device=self.device)
        self.field_temp = wp.empty_like(self.field)
        self.field_orig = wp.empty_like(self.field)
        self.edge_indices: wp.array[wp.int32] | None = None
        self.vertices: wp.array[wp.vec3] | None = None
        self.vertices_temp: wp.array[wp.vec3] | None = None
        self.indices: wp.array[wp.int32] | None = None
        self.normals: wp.array[wp.vec3] | None = None
        self.neighbor_sum: wp.array[wp.vec3] | None = None
        self.valence: wp.array[wp.int32] | None = None
        if allocate_mesh:
            self.edge_indices = wp.empty(3 * self.max_grid_nodes, dtype=wp.int32, device=self.device)
            self.vertices = wp.empty(self.max_vertices, dtype=wp.vec3, device=self.device)
            self.vertices_temp = wp.empty_like(self.vertices)
            self.indices = wp.empty(self.max_indices, dtype=wp.int32, device=self.device)
            self.normals = wp.empty(self.max_vertices, dtype=wp.vec3, device=self.device)
            self.neighbor_sum = wp.empty(self.max_vertices, dtype=wp.vec3, device=self.device)
            self.valence = wp.empty(self.max_vertices, dtype=wp.int32, device=self.device)

        corner_offsets = wp.MarchingCubes.CUBE_CORNER_OFFSETS
        edge_corners = wp.MarchingCubes.EDGE_TO_CORNERS
        edge_offsets: list[tuple[int, int, int]] = []
        edge_axes: list[int] = []
        for first, second in edge_corners:
            first_corner = corner_offsets[first]
            second_corner = corner_offsets[second]
            offset = tuple(min(first_corner[axis], second_corner[axis]) for axis in range(3))
            edge_offsets.append(offset)
            edge_axes.append(next(axis for axis in range(3) if first_corner[axis] != second_corner[axis]))

        self.case_ranges = wp.array(
            wp.MarchingCubes.CASE_TO_TRI_RANGE,
            dtype=wp.int32,
            device=self.device,
        )
        self.local_edges = wp.array(
            wp.MarchingCubes.TRI_LOCAL_INDICES,
            dtype=wp.int32,
            device=self.device,
        )
        self.corner_offsets = wp.array(corner_offsets, dtype=wp.vec3i, device=self.device)
        self.edge_offsets = wp.array(edge_offsets, dtype=wp.vec3i, device=self.device)
        self.edge_axes = wp.array(edge_axes, dtype=wp.int32, device=self.device)
        self._dummy_vertex = wp.empty(1, dtype=wp.vec3, device=self.device)
        self._dummy_index = wp.empty(1, dtype=wp.int32, device=self.device)

    def reset(self) -> None:
        wp.launch(
            kernels.reset_bounds_and_counts,
            dim=1,
            inputs=[self.lower, self.upper, self.grid_counts, self.mesh_counts],
            device=self.device,
        )

    def reset_mesh_counts(self) -> None:
        wp.launch(
            kernels.reset_mesh_counts,
            dim=1,
            inputs=[self.mesh_counts, self.grid_counts],
            device=self.device,
        )

    def resize_grid_exact(self, node_count: int, cell_count: int) -> None:
        """Replace provisional field storage with realized-size buffers."""
        self.max_grid_nodes = node_count
        self.max_grid_cells = cell_count
        self.launch_threads = min(max(node_count, 1), kernels._MAX_CAPACITY_LAUNCH_THREADS)
        if self.field.shape[0] != node_count:
            self.field = wp.empty(node_count, dtype=wp.float32, device=self.device)
            self.field_temp = wp.empty_like(self.field)
            self.field_orig = wp.empty_like(self.field)
        wp.launch(kernels.clear_grid_overflow, dim=1, inputs=[self.grid_counts], device=self.device)

    def resize_mesh_exact(self, vertex_count: int, index_count: int) -> None:
        """Allocate realized-size marching-cubes and post-processing buffers."""
        self.max_vertices = vertex_count
        self.max_indices = index_count
        edge_count = 3 * self.max_grid_nodes
        if self.edge_indices is None or self.edge_indices.shape[0] != edge_count:
            self.edge_indices = wp.empty(edge_count, dtype=wp.int32, device=self.device)
        if self.vertices is None or self.vertices.shape[0] != vertex_count:
            self.vertices = wp.empty(vertex_count, dtype=wp.vec3, device=self.device)
            self.vertices_temp = wp.empty_like(self.vertices)
            self.normals = wp.empty(vertex_count, dtype=wp.vec3, device=self.device)
            self.neighbor_sum = wp.empty(vertex_count, dtype=wp.vec3, device=self.device)
            self.valence = wp.empty(vertex_count, dtype=wp.int32, device=self.device)
        if self.indices is None or self.indices.shape[0] != index_count:
            self.indices = wp.empty(index_count, dtype=wp.int32, device=self.device)

    def compute_particle_bounds(
        self,
        positions: wp.array[wp.vec3],
        flags: wp.array[wp.int32],
        use_flags: int,
        sentinel_distance: float,
    ) -> None:
        if positions.shape[0] > 0:
            wp.launch(
                kernels.compute_particle_bounds,
                dim=positions.shape[0],
                inputs=[positions, flags, use_flags, self.lower, self.upper, self.grid_counts],
                device=self.device,
            )
        wp.launch(
            kernels.finalize_particle_bounds,
            dim=1,
            inputs=[self.lower, self.upper, self.inactive_position, sentinel_distance],
            device=self.device,
        )

    def compute_grid(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        flags: wp.array[wp.int32],
        use_flags: int,
        det_G: wp.array[float],
        density_reach: wp.array[wp.vec3],
        particle_sdf_radius_scale: float,
        particle_sdf_band: float,
        particle_sdf: bool,
        anisotropic: bool,
    ) -> None:
        wp.launch(kernels.reset_bounds, dim=1, inputs=[self.lower, self.upper], device=self.device)
        if positions.shape[0] > 0:
            wp.launch(
                kernels.compute_kernel_bounds,
                dim=positions.shape[0],
                inputs=[
                    positions,
                    radii,
                    flags,
                    use_flags,
                    det_G,
                    density_reach,
                    particle_sdf_radius_scale,
                    particle_sdf_band,
                    int(particle_sdf),
                    int(anisotropic),
                    self.lower,
                    self.upper,
                ],
                device=self.device,
            )
        wp.launch(
            kernels.finalize_grid,
            dim=1,
            inputs=[
                self.lower,
                self.upper,
                self.grid_counts,
                self.grid_origin,
                self.grid_dims,
                self.grid_counts,
                self.voxel_size,
                self.padding,
                self.max_grid_cells,
                self.max_grid_nodes,
            ],
            device=self.device,
        )

    def evaluate_field(
        self,
        smoothed: wp.array[wp.vec3],
        radii: wp.array[float],
        flags: wp.array[wp.int32],
        use_flags: int,
        G: wp.array[wp.mat33],
        det_G: wp.array[float],
        density_reach: wp.array[wp.vec3],
        *,
        surface_method: str,
        anisotropic_sdf: bool,
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
        wp.launch(
            kernels.fill_field,
            dim=self.launch_threads,
            inputs=[self.field, self.grid_counts, outside_value, self.launch_threads],
            device=self.device,
        )

        particle_count = smoothed.shape[0]
        if particle_count > 0:
            if particle_sdf:
                kernel = (
                    kernels.evaluate_particle_sdf_anisotropic
                    if anisotropic_sdf
                    else kernels.evaluate_particle_sdf_isotropic
                )
                if anisotropic_sdf:
                    inputs = [
                        smoothed,
                        radii,
                        flags,
                        use_flags,
                        G,
                        det_G,
                        density_reach,
                        particle_sdf_radius_scale,
                        particle_sdf_band,
                        self.grid_origin,
                        self.grid_dims,
                        self.grid_counts,
                        1.0 / self.voxel_size,
                        self.field,
                    ]
                else:
                    inputs = [
                        smoothed,
                        radii,
                        flags,
                        use_flags,
                        particle_sdf_radius_scale,
                        particle_sdf_band,
                        self.grid_origin,
                        self.grid_dims,
                        self.grid_counts,
                        1.0 / self.voxel_size,
                        self.field,
                    ]
                wp.launch(kernel, dim=particle_count, inputs=inputs, device=self.device)
            else:
                wp.launch(
                    kernels.evaluate_density,
                    dim=particle_count,
                    inputs=[
                        smoothed,
                        radii,
                        flags,
                        use_flags,
                        G,
                        det_G,
                        density_reach,
                        self.grid_origin,
                        self.grid_dims,
                        self.grid_counts,
                        1.0 / self.voxel_size,
                        self.field,
                    ],
                    device=self.device,
                )

        if blur_iterations > 0 and blur_radius > 0 and blur_weights is not None:
            source = self.field
            destination = self.field_temp
            for _ in range(blur_iterations):
                for axis in range(3):
                    wp.launch(
                        kernels.blur_field_axis,
                        dim=self.launch_threads,
                        inputs=[
                            source,
                            destination,
                            blur_weights,
                            blur_radius,
                            axis,
                            self.grid_dims,
                            self.grid_counts,
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
                    kernels.density_to_sdf,
                    dim=self.launch_threads,
                    inputs=[self.field, self.grid_counts, threshold, self.launch_threads],
                    device=self.device,
                )
            self.redistance(redistance_iterations)

    def redistance(self, iterations: int) -> None:
        for _ in range(iterations):
            wp.launch(
                kernels.redistance_step,
                dim=self.launch_threads,
                inputs=[
                    self.field,
                    self.field_temp,
                    self.grid_dims,
                    self.grid_counts,
                    1.0 / self.voxel_size,
                    self.launch_threads,
                ],
                device=self.device,
            )
            self.field, self.field_temp = self.field_temp, self.field

    def extract_mesh(
        self,
        threshold: float,
        *,
        flip_winding: bool,
        smooth_iterations: int,
        smooth_lambda: float,
        compute_normals: bool,
    ) -> None:
        if (
            self.edge_indices is None
            or self.vertices is None
            or self.vertices_temp is None
            or self.indices is None
            or self.normals is None
            or self.neighbor_sum is None
            or self.valence is None
        ):
            raise RuntimeError("Mesh capacity was not allocated")
        self.reset_mesh_counts()
        wp.launch(
            kernels.reset_edge_indices,
            dim=self.launch_threads,
            inputs=[self.edge_indices, self.grid_counts, self.launch_threads],
            device=self.device,
        )
        wp.launch(
            kernels.extract_mesh_vertices,
            dim=self.launch_threads,
            inputs=[
                self.field,
                threshold,
                self.grid_origin,
                self.grid_dims,
                self.grid_counts,
                self.voxel_size,
                self.vertices,
                self.edge_indices,
                self.mesh_counts,
                1,
                self.launch_threads,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.extract_mesh_indices,
            dim=self.launch_threads,
            inputs=[
                self.field,
                threshold,
                self.grid_dims,
                self.grid_counts,
                self.case_ranges,
                self.local_edges,
                self.corner_offsets,
                self.edge_offsets,
                self.edge_axes,
                self.edge_indices,
                self.indices,
                self.mesh_counts,
                1,
                self.launch_threads,
            ],
            device=self.device,
        )
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

    def count_mesh(self, threshold: float) -> None:
        """Count marching-cubes output using the production extraction kernels."""
        self.reset_mesh_counts()
        wp.launch(
            kernels.extract_mesh_vertices,
            dim=self.launch_threads,
            inputs=[
                self.field,
                threshold,
                self.grid_origin,
                self.grid_dims,
                self.grid_counts,
                self.voxel_size,
                self._dummy_vertex,
                self._dummy_index,
                self.mesh_counts,
                0,
                self.launch_threads,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.extract_mesh_indices,
            dim=self.launch_threads,
            inputs=[
                self.field,
                threshold,
                self.grid_dims,
                self.grid_counts,
                self.case_ranges,
                self.local_edges,
                self.corner_offsets,
                self.edge_offsets,
                self.edge_axes,
                self._dummy_index,
                self._dummy_index,
                self.mesh_counts,
                0,
                self.launch_threads,
            ],
            device=self.device,
        )


class ParticleSurface:
    """Reusable context for extracting a triangle mesh from particle data.

    Uses the Yu & Turk (2010) anisotropic kernel method: per-particle
    Weighted PCA determines oriented ellipsoidal kernels that produce a
    smooth scalar field whose isosurface tightly wraps the particles.

    Args:
        voxel_size: Edge length of each grid voxel [m].
        max_grid_cells: Maximum logical grid cell count. When set, extraction
            uses preallocated, graph-capturable buffers. When ``None``, each
            extraction uses tight field and mesh allocations.
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
        anisotropy_strength: Blend from isotropic kernels to anisotropic
            kernels [0, 1].  Lower values preserve more normal support from
            boundary particles back into the interior.
        surface_method: Surface reconstruction method. ``"density"`` uses
            anisotropic density splatting. ``"particle_sdf"`` directly unions
            per-particle anisotropic ellipsoid SDFs and stores an SDF field.
        particle_sdf_radius_scale: Radius multiplier for ``surface_method="particle_sdf"``.
        particle_sdf_band: Narrow-band half-width in normalized ellipsoid
            coordinates for ``surface_method="particle_sdf"``.
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

    class ExtractionMesh:
        """Particle surface mesh and its device-resident logical counts.

        The entries of :attr:`counts` are the vertex count, flattened index
        count, and grid-overflow flag. Buffers are exact-sized when
        ``max_grid_cells`` is ``None`` and preallocated otherwise.
        """

        def __init__(
            self,
            vertices: wp.array[wp.vec3] | None,
            indices: wp.array[wp.int32] | None,
            normals: wp.array[wp.vec3] | None,
            counts: wp.array[wp.int32],
            grid_counts: wp.array[wp.int32],
            *,
            exact: bool,
        ):
            self.vertices = vertices
            self.indices = indices
            self.normals = normals
            self.counts = counts
            self.active_particle_count: wp.array[wp.int32] = grid_counts[2:3]
            self.grid_overflow: wp.array[wp.int32] = counts[2:3]
            self._exact = exact

        @classmethod
        def _from_capacity(
            cls,
            workspace: ParticleSurfaceCapacity,
            grid_counts: wp.array[wp.int32],
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
                grid_counts,
                exact=exact,
            )

        def to_arrays(
            self,
        ) -> tuple[wp.array[wp.vec3] | None, wp.array[wp.int32] | None, wp.array[wp.vec3] | None]:
            """Return exact-length arrays, synchronizing only preallocated meshes."""
            if self._exact:
                if self.vertices is None or self.indices is None or self.indices.shape[0] == 0:
                    return None, None, None
                return self.vertices, self.indices, self.normals

            counts = self.counts.numpy()
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
        if not math.isfinite(particle_sdf_band) or particle_sdf_band <= 0.0:
            raise ValueError("particle_sdf_band must be positive")
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

        self.voxel_size = voxel_size
        self.kernel_radius = kernel_radius
        self.anisotropic = anisotropic
        self.threshold = threshold
        self.smooth_lambda = smooth_lambda
        self.anisotropy_ratio = anisotropy_ratio
        self.anisotropy_scale = anisotropy_scale
        self.kernel_scale = kernel_scale
        self.anisotropy_min_neighbors = anisotropy_min_neighbors
        self.anisotropy_strength = anisotropy_strength
        self.surface_method = surface_method
        self.particle_sdf_radius_scale = particle_sdf_radius_scale
        self.particle_sdf_band = particle_sdf_band
        self.padding = padding
        self.field_smooth_iterations = field_smooth_iterations
        self.field_smooth_radius = field_smooth_radius
        self.field_mode = field_mode
        self.redistance_iterations = redistance_iterations
        self.mesh_smooth_iterations = mesh_smooth_iterations
        self.mesh_smooth_lambda = mesh_smooth_lambda

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
        self._hash_positions: wp.array[wp.vec3] | None = None
        self._all_particle_flags: wp.array[wp.int32] | None = None
        self._n_particles: int = 0
        self._max_particles: int = 0
        self._max_grid_cells = max_grid_cells
        self._capacity: ParticleSurfaceCapacity | None = None
        self._grid_dims: tuple[int, int, int] | None = None
        self._has_field = False

        # Last extraction results
        self._verts: wp.array[wp.vec3] | None = None
        self._indices: wp.array[wp.int32] | None = None
        self._normals: wp.array[wp.vec3] | None = None

        if max_grid_cells is not None:
            self._configure_grid_capacity(max_grid_cells, device=self._device)

    # -- Public properties --

    @property
    def verts(self) -> wp.array[wp.vec3] | None:
        """Vertex positions from the last extraction [m]."""
        return self._verts

    @property
    def max_grid_cells(self) -> int | None:
        """Preallocated logical grid-cell capacity, or ``None`` for tight allocation."""
        return self._max_grid_cells

    @property
    def indices(self) -> wp.array[wp.int32] | None:
        """Triangle indices from the last extraction, shape ``(3 * triangle_count,)``."""
        return self._indices

    @property
    def normals(self) -> wp.array[wp.vec3] | None:
        """Unit-length per-vertex normals from the last extraction."""
        return self._normals

    @property
    def field(self) -> wp.array3d[wp.float32] | None:
        """Exact-size scalar field view.

        Accessing a preallocated field reads its logical dimensions from the device.
        """
        if self._capacity is None or not self._has_field:
            return None
        if self.max_grid_cells is None:
            dims = self._grid_dims
            node_count = dims[0] * dims[1] * dims[2]
        else:
            counts = self._capacity.grid_counts.numpy()
            if int(counts[3]) != 0:
                raise ValueError("Particle surface exceeds configured max_grid_cells")
            dims = tuple(int(value) for value in counts[4:7])
            node_count = int(counts[0])
        return self._capacity.field[:node_count].reshape(dims)

    @property
    def grid_origin(self) -> wp.vec3 | None:
        """World-space grid origin [m].

        Accessing the origin reads it from the device.
        """
        if self._capacity is None or not self._has_field:
            return None
        return wp.vec3(self._capacity.grid_origin.numpy()[0])

    @property
    def grid_dims(self) -> tuple[int, int, int] | None:
        """Logical grid node counts.

        Accessing a preallocated grid reads its dimensions from the device.
        """
        if self.max_grid_cells is None:
            return self._grid_dims
        if self._capacity is None or not self._has_field:
            return None
        return tuple(int(value) for value in self._capacity.grid_counts.numpy()[4:7])

    @property
    def smoothed_positions(self) -> wp.array[wp.vec3] | None:
        """Smoothed particle positions from the last extraction [m]."""
        return self._smoothed

    def _configure_grid_capacity(
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
        self._capacity = ParticleSurfaceCapacity(
            max_grid_cells=max_grid_cells,
            voxel_size=self.voxel_size,
            padding=self.padding,
            device=device_obj,
        )
        self._has_field = False

        hash_grid_dim = max(16, int(math.ceil(max_grid_cells ** (1.0 / 3.0))))
        self._hash_grid = wp.HashGrid(hash_grid_dim, hash_grid_dim, hash_grid_dim, device=device_obj)
        self._hash_grid_dim = hash_grid_dim
        if self.field_smooth_iterations > 0 and self.field_smooth_radius > 0:
            self._ensure_blur_weights(device_obj)
        return self

    def update_field(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        particle_flags: wp.array[wp.int32] | None = None,
    ) -> wp.array[Any]:
        """Update the scalar field without extracting a mesh.

        Args:
            positions: Particle positions [m], shape ``(N,)``, dtype ``wp.vec3``.
            radii: Per-particle radii [m], shape ``(N,)``, dtype ``wp.float32``.
            particle_flags: Optional per-particle flags.  Particles without
                :attr:`~newton.ParticleFlags.ACTIVE` are skipped without
                compacting the particle arrays.

        Returns:
            The exact field when ``max_grid_cells`` is ``None``, otherwise the
            preallocated flat field buffer.
        """
        self.extract(
            positions,
            radii,
            compute_normals=False,
            particle_flags=particle_flags,
            compute_mesh=False,
        )
        return self.field if self.max_grid_cells is None else self._capacity.field

    def fem_field(self) -> fem.DiscreteField:
        """Return the scalar field as a :class:`warp.fem.DiscreteField`.

        The field lives on a Q1 (trilinear) function space over a
        :class:`warp.fem.Grid3D` matching the extraction grid.  It can be
        used directly with :func:`warp.fem.interpolate` or
        :func:`warp.fem.integrate` to evaluate smooth values, gradients,
        and curvature at arbitrary positions.  With ``field_mode="density"``,
        the values are the scalar density field.  With ``field_mode="sdf"``
        or ``surface_method="particle_sdf"``, negative values are inside the
        particle surface and positive values are outside.

        Must be called after :meth:`extract` or :meth:`update_field`.

        Returns:
            A :class:`warp.fem.DiscreteField` with scalar ``float`` DOFs.
        """
        if not self._has_field:
            raise RuntimeError("extract() or update_field() must populate the field before fem_field()")
        field = self.field
        origin = self.grid_origin
        nx, ny, nz = field.shape
        grid = fem.Grid3D(
            bounds_lo=origin,
            bounds_hi=wp.vec3(
                origin[0] + (nx - 1) * self.voxel_size,
                origin[1] + (ny - 1) * self.voxel_size,
                origin[2] + (nz - 1) * self.voxel_size,
            ),
            res=wp.vec3i(nx - 1, ny - 1, nz - 1),
        )
        space = fem.make_polynomial_space(grid, degree=1, dtype=float)
        discrete_field = fem.make_discrete_field(space)
        discrete_field.dof_values = field.flatten()
        return discrete_field

    # -- Core extraction --

    def extract(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        compute_normals: bool = True,
        particle_flags: wp.array[wp.int32] | None = None,
        compute_mesh: bool = True,
    ) -> ParticleSurface.ExtractionMesh:
        """Extract a triangle mesh from particle positions.

        When :attr:`max_grid_cells` is set, this method performs no host
        synchronization and can be captured in a CUDA graph. Otherwise it
        allocates exact-size field and mesh arrays.

        Args:
            positions: Particle positions [m], shape ``(N,)``, dtype ``wp.vec3``.
            radii: Per-particle radii [m], shape ``(N,)``, dtype ``wp.float32``.
            compute_normals: Whether to compute per-vertex normals.
            particle_flags: Optional per-particle flags.  Particles without
                :attr:`~newton.ParticleFlags.ACTIVE` are skipped.
            compute_mesh: Whether to run Marching Cubes and mesh post-processing.
                Set to ``False`` to update only :attr:`field`, for example before
                modifying the field and calling :meth:`resurface`.

        Returns:
            Mesh buffers and device-resident logical counts.
        """
        self._validate_positions_layout(positions)
        particle_count = positions.shape[0]
        device = positions.device
        self._validate_radii_layout(positions, radii, particle_count)
        self._validate_particle_flags_layout(particle_flags, particle_count, device)
        return self._extract_impl(
            positions,
            radii,
            compute_normals=compute_normals,
            particle_flags=particle_flags,
            compute_mesh=compute_mesh,
        )

    def redistance(self, iterations: int | None = None) -> None:
        """Apply Eikonal redistancing to the current SDF field.

        Args:
            iterations: Number of redistancing iterations.  Defaults to
                :attr:`redistance_iterations`.
        """
        if self.field_mode != "sdf":
            raise ValueError("redistance() requires field_mode='sdf'")
        if self._capacity is None or not self._has_field:
            return
        if iterations is None:
            iterations = self.redistance_iterations
        if iterations <= 0:
            return
        self._capacity.redistance(iterations)

    def resurface(
        self,
        compute_normals: bool = True,
    ) -> ParticleSurface.ExtractionMesh:
        """Re-run marching cubes on the current field.

        Use after externally modifying :attr:`field`, for example to
        extrapolate an SDF into collider regions before extracting the mesh.
        """
        if self._capacity is None or not self._has_field:
            raise RuntimeError("extract() or update_field() must populate the field before resurface()")
        return self._extract_current_mesh(
            self._capacity,
            compute_normals=compute_normals,
            exact=self.max_grid_cells is None,
        )

    # -- Internal helpers --

    def _extract_impl(
        self,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        *,
        compute_normals: bool,
        particle_flags: wp.array[wp.int32] | None,
        compute_mesh: bool,
    ) -> ParticleSurface.ExtractionMesh:
        particle_count = positions.shape[0]
        device = positions.device
        device_obj = wp.get_device(device)
        exact = self.max_grid_cells is None
        if exact:
            if device_obj != self._resource_device:
                self._clear_device_resources()
                self._device = device_obj
                self._resource_device = device_obj
            hash_grid_dim = max(16, int(math.ceil(max(particle_count, 1) ** (1.0 / 3.0))))
            self._ensure_hash_grid(hash_grid_dim, device)
            if self._capacity is None:
                self._capacity = ParticleSurfaceCapacity(
                    max_grid_cells=1,
                    max_grid_nodes=1,
                    voxel_size=self.voxel_size,
                    padding=self.padding,
                    device=device,
                    allocate_field=False,
                    allocate_mesh=False,
                )
            workspace = self._capacity
        else:
            if device_obj != self._resource_device:
                self._configure_grid_capacity(self.max_grid_cells, device=device)
            workspace = self._capacity

        self._ensure_capacity_particle_resources(particle_count)
        flags, use_flags = self._field_flag_args(particle_flags, particle_count, device)
        workspace.reset()
        sentinel_distance = 1.0e6 * max(self.kernel_radius, self.voxel_size)
        workspace.compute_particle_bounds(positions, flags, use_flags, sentinel_distance)
        self._prepare_particle_values(workspace, positions, particle_count, flags, use_flags, device)

        isotropic_sdf = not self.anisotropic or self.anisotropy_strength <= 0.0 or self.anisotropy_ratio <= 1.0
        workspace.compute_grid(
            self._smoothed[:particle_count],
            radii,
            flags,
            use_flags,
            self._det_G[:particle_count],
            self._density_reach[:particle_count],
            self.particle_sdf_radius_scale,
            self.particle_sdf_band,
            self.surface_method == "particle_sdf",
            not isotropic_sdf,
        )
        if exact:
            grid_counts = workspace.grid_counts.numpy()
            node_count = int(grid_counts[0])
            cell_count = int(grid_counts[1])
            self._grid_dims = tuple(int(value) for value in grid_counts[4:7])
            workspace.resize_grid_exact(node_count, cell_count)
        else:
            self._grid_dims = None

        if self.field_smooth_iterations > 0 and self.field_smooth_radius > 0:
            self._ensure_blur_weights(workspace.device)
        workspace.evaluate_field(
            self._smoothed[:particle_count],
            radii,
            flags,
            use_flags,
            self._G[:particle_count],
            self._det_G[:particle_count],
            self._density_reach[:particle_count],
            surface_method=self.surface_method,
            anisotropic_sdf=not isotropic_sdf,
            particle_sdf_radius_scale=self.particle_sdf_radius_scale,
            particle_sdf_band=self.particle_sdf_band,
            kernel_radius=self.kernel_radius,
            field_mode=self.field_mode,
            threshold=self.threshold,
            blur_weights=self._blur_weights,
            blur_radius=self.field_smooth_radius,
            blur_iterations=self.field_smooth_iterations,
            redistance_iterations=self.redistance_iterations,
        )
        self._has_field = True
        if compute_mesh:
            return self._extract_current_mesh(workspace, compute_normals=compute_normals, exact=exact)

        workspace.reset_mesh_counts()
        result = self.ExtractionMesh._from_capacity(
            workspace,
            workspace.grid_counts,
            compute_normals=compute_normals,
            exact=exact,
        )
        self._verts, self._indices, self._normals = result.vertices, result.indices, result.normals
        return result

    def _extract_current_mesh(
        self,
        workspace: ParticleSurfaceCapacity,
        *,
        compute_normals: bool,
        exact: bool,
    ) -> ParticleSurface.ExtractionMesh:
        threshold = self._marching_threshold()
        if exact:
            workspace.count_mesh(threshold)
            mesh_counts = workspace.mesh_counts.numpy()
            workspace.resize_mesh_exact(int(mesh_counts[0]), int(mesh_counts[1]))

        workspace.extract_mesh(
            threshold,
            flip_winding=self.field_mode == "density",
            smooth_iterations=self.mesh_smooth_iterations,
            smooth_lambda=self.mesh_smooth_lambda,
            compute_normals=compute_normals,
        )
        result = self.ExtractionMesh._from_capacity(
            workspace,
            workspace.grid_counts,
            compute_normals=compute_normals,
            exact=exact,
        )
        self._verts, self._indices, self._normals = result.vertices, result.indices, result.normals
        return result

    def _clear_results(self):
        self._verts = None
        self._indices = None
        self._normals = None

    def _clear_device_resources(self):
        self._clear_results()
        self._capacity = None
        self._grid_dims = None
        self._has_field = False
        self._hash_grid = None
        self._blur_weights = None
        self._hash_grid_dim = 0
        self._smoothed = None
        self._G = None
        self._det_G = None
        self._density_reach = None
        self._hash_positions = None
        self._all_particle_flags = None
        self._n_particles = 0
        self._max_particles = 0

    def _prepare_particle_values(
        self,
        workspace: ParticleSurfaceCapacity,
        positions: wp.array[wp.vec3],
        particle_count: int,
        flags: wp.array[wp.int32],
        use_flags: int,
        device: wp.DeviceLike,
    ) -> None:
        smoothed = self._smoothed[:particle_count]
        G = self._G[:particle_count]
        det_G = self._det_G[:particle_count]
        density_reach = self._density_reach[:particle_count]
        hash_positions = positions

        if use_flags != 0 and particle_count > 0:
            hash_positions = self._hash_positions[:particle_count]
            wp.launch(
                kernels.copy_active_or_sentinel_positions,
                dim=particle_count,
                inputs=[positions, flags, use_flags, workspace.inactive_position, hash_positions],
                device=device,
            )

        if self.smooth_lambda > 1.0e-6 and particle_count > 0:
            self._hash_grid.build(hash_positions, self.kernel_radius)
            if use_flags != 0:
                wp.launch(
                    kernels.smooth_positions_flagged,
                    dim=particle_count,
                    inputs=[
                        self._hash_grid.id,
                        hash_positions,
                        flags,
                        workspace.inactive_position,
                        self.kernel_radius,
                        self.smooth_lambda,
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
                        self.kernel_radius,
                        self.smooth_lambda,
                        smoothed,
                    ],
                    device=device,
                )
        elif particle_count > 0:
            if use_flags != 0:
                wp.launch(
                    kernels.copy_active_or_sentinel_positions,
                    dim=particle_count,
                    inputs=[positions, flags, use_flags, workspace.inactive_position, smoothed],
                    device=device,
                )
            else:
                wp.copy(smoothed, positions)

        if self.anisotropic and particle_count > 0:
            self._hash_grid.build(smoothed, self.kernel_radius)
            wp.launch(
                kernels._compute_anisotropy,
                dim=particle_count,
                inputs=[
                    self._hash_grid.id,
                    smoothed,
                    flags,
                    use_flags,
                    self.kernel_radius,
                    self.anisotropy_ratio,
                    self.anisotropy_scale,
                    self.kernel_scale,
                    self.anisotropy_min_neighbors,
                    self.anisotropy_strength,
                    G,
                    det_G,
                    density_reach,
                ],
                device=device,
            )
        elif particle_count > 0:
            wp.launch(
                kernels._fill_isotropic_G,
                dim=particle_count,
                inputs=[self.kernel_radius, self.kernel_scale, flags, use_flags, G, det_G, density_reach],
                device=device,
            )

    def _ensure_capacity_particle_resources(self, particle_count: int) -> None:
        if particle_count <= self._max_particles and self._smoothed is not None:
            return
        alloc_particles = max(particle_count, 1)
        self._smoothed = wp.empty(alloc_particles, dtype=wp.vec3, device=self._device)
        self._G = wp.empty(alloc_particles, dtype=wp.mat33, device=self._device)
        self._det_G = wp.empty(alloc_particles, dtype=float, device=self._device)
        self._density_reach = wp.empty(alloc_particles, dtype=wp.vec3, device=self._device)
        self._hash_positions = wp.empty(alloc_particles, dtype=wp.vec3, device=self._device)
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

    def _ensure_blur_weights(self, device: wp.DeviceLike):
        hw = self.field_smooth_radius
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
        if self.field_mode == "sdf":
            effective_threshold = 0.0
            if self.mesh_smooth_iterations > 0:
                shrink = (
                    _MESH_SMOOTH_SHRINK_PER_VOXEL
                    * math.sqrt(float(self.mesh_smooth_iterations))
                    * self.mesh_smooth_lambda
                    * self.voxel_size
                )
                if self.surface_method == "particle_sdf" or self.redistance_iterations > 0:
                    effective_threshold = shrink
                else:
                    effective_threshold = shrink / self.kernel_radius
            return effective_threshold

        effective_threshold = self.threshold
        if self.mesh_smooth_iterations > 0:
            shrink = (
                _MESH_SMOOTH_SHRINK_PER_VOXEL
                * math.sqrt(float(self.mesh_smooth_iterations))
                * self.mesh_smooth_lambda
                * self.voxel_size
            )
            effective_threshold = max(self.threshold - shrink / self.kernel_radius, _MIN_DENSITY_MARCHING_THRESHOLD)
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
    anisotropy_ratio: float = 4.0,
    kernel_scale: float = 0.5,
    anisotropy_scale: float = 1.0,
    anisotropy_min_neighbors: int = 25,
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
        max_grid_cells: Maximum logical grid cell count. When set, extraction
            uses graph-capturable preallocated buffers. When ``None``, it uses
            tight allocations.
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
        anisotropy_ratio: Maximum anisotropic kernel axis ratio.
        kernel_scale: Kernel radius multiplier relative to ``kernel_radius``.
        anisotropy_scale: Relative multiplier for anisotropic kernel radii.
        anisotropy_min_neighbors: Minimum number of other particles required
            for anisotropic kernels.
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
            coordinates for ``surface_method="particle_sdf"``.

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
        anisotropy_strength=anisotropy_strength,
        field_smooth_iterations=field_smooth_iterations,
        field_smooth_radius=field_smooth_radius,
        surface_method=surface_method,
        particle_sdf_radius_scale=particle_sdf_radius_scale,
        particle_sdf_band=particle_sdf_band,
        field_mode=field_mode,
        redistance_iterations=redistance_iterations,
        mesh_smooth_iterations=mesh_smooth_iterations,
        device=positions.device,
    )
    return ctx.extract(positions, radii=radii, compute_normals=compute_normals, particle_flags=particle_flags)
