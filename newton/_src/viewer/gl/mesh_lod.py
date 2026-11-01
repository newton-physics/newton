# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Screen-space display detail for densely sampled SDF surfaces."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING
from weakref import proxy

import numpy as np

if TYPE_CHECKING:
    from newton._src.viewer.gl.opengl import MeshGL, MeshInstancerGL, RendererGL
    from newton.viewer import ViewerBase, ViewerGL


class SdfMeshLOD:
    """Choose display-only SDF mesh detail for one instanced batch and owning layer.

    Level zero is the original mesh; coarser levels are extracted lazily at
    strides dividing the SDF subgrid size. The source SDF and collision queries
    remain unchanged. Cached meshes are registered in the owning layer, even if
    another layer is active when rendering requests a new level.

    Camera and shadow passes use their own frusta and resolution limits. Draw
    ranges preserve instance order so transforms, visibility, and picking indices
    keep referring to the original batch.
    """

    def __init__(self, viewer: ViewerGL, batch: ViewerBase.ShapeInstances) -> None:
        """Read one SDF descriptor and initialize lazily generated mesh levels for a shape batch.

        Retain the original mesh as level zero and select valid coarsening
        strides from the subgrid size. The viewer reference is weak to avoid
        extending its lifetime through the mesh cache.
        """
        self.viewer = proxy(viewer)
        self.model = viewer.model
        self.layer = viewer.layer
        self.index = int(viewer._shape_sdf_index_host[batch.model_shapes[0]])
        entry = viewer.model._texture_sdf_data[self.index : self.index + 1].numpy()[0]
        self.center = (entry["sdf_box_lower"] + entry["sdf_box_upper"]) * 0.5
        self.half = (entry["sdf_box_upper"] - entry["sdf_box_lower"]) * 0.5
        self.radius = float(np.linalg.norm(entry["sdf_box_upper"] - entry["sdf_box_lower"]) * 0.5)
        self.cell = float(np.linalg.norm(entry["voxel_size"]))
        size = int(entry["subgrid_size"])
        self.strides = np.array([s for s in (1, 2, 4, 8) if size % s == 0])
        self.meshes = {0: viewer.objects[batch.mesh]}

    def mesh(self, level: int) -> MeshGL:
        """Return a cached display mesh, falling back to level zero for empty extraction."""
        if level not in self.meshes:
            from ...geometry.sdf_texture import compute_isomesh_from_texture_sdf  # noqa: PLC0415
            from ...geometry.types import GeoType  # noqa: PLC0415

            viewer, model = self.viewer, self.model
            texture = model._texture_sdf_coarse_textures[self.index]
            source = compute_isomesh_from_texture_sdf(
                model._texture_sdf_data,
                self.index,
                model._texture_sdf_subgrid_start_slots[self.index],
                (texture.width - 1, texture.height - 1, texture.depth - 1),
                device=model.device,
                cell_stride=int(self.strides[level]),
            )
            # Thin features can disappear on a coarse sampling grid. An empty
            # level must fall back to the exact surface, never hide the shape.
            if source is None:
                self.meshes[level] = self.meshes[0]
            else:
                original_layer = viewer.layer
                viewer._save_active_layer_state()
                try:
                    viewer._active_layer_id = self.layer.layer_id
                    viewer._load_layer_state(self.layer)
                    name = viewer._populate_geometry(int(GeoType.MESH), (1.0, 1.0, 1.0), 0.0, True, source)
                    self.meshes[level] = viewer.objects[name]
                finally:
                    viewer._save_active_layer_state()
                    viewer._active_layer_id = original_layer.layer_id
                    viewer._load_layer_state(original_layer)
        return self.meshes[level]

    def _visible(self, matrix: np.ndarray, centers: np.ndarray, axes: np.ndarray) -> np.ndarray:
        """Conservatively test transformed SDF boxes against all six clip planes."""
        planes = np.concatenate((matrix[3] + matrix[:3], matrix[3] - matrix[:3]))
        distance = centers @ planes[:, :3].T + planes[:, 3]
        extent = np.abs(np.einsum("nij,pj->npi", axes, planes[:, :3])) @ self.half
        # Keep touching bounds, including mirrored and nonuniform instances.
        allowance = 1e-6 * np.linalg.norm(planes[:, :3], axis=1)
        return np.all(distance + extent >= -allowance, axis=1)

    def _draws(self, levels: np.ndarray, visible: np.ndarray) -> tuple[tuple[MeshGL, int, int], ...]:
        """Group contiguous instances as (mesh, first, count), without sorting the batch."""
        levels = np.where(visible, levels, -1)
        boundaries = np.r_[0, np.flatnonzero(np.diff(levels)) + 1, len(levels)]
        return tuple(
            (self.mesh(int(levels[start])), int(start), int(end - start))
            for start, end in pairwise(boundaries)
            if end > start and levels[start] >= 0
        )

    def shadow_draws(self, renderer: RendererGL, obj: MeshInstancerGL) -> tuple[tuple[MeshGL, int, int], ...]:
        """Retain offscreen casters that intersect the light's own frustum."""
        centers, axes = obj._lod_bounds
        matrix = renderer._light_space_matrix.reshape(4, 4).T
        # A close camera does not increase the shadow map's resolution. Sample
        # only detail its texels can resolve, retaining a half-texel cell limit.
        pixels = matrix[:2, :3] * np.array((renderer._shadow_width, renderer._shadow_height))[:, None] * 0.5
        magnification = np.linalg.norm(pixels, ord=2) * np.linalg.norm(axes, axis=2).max(axis=1)
        levels = obj._lod_levels.copy()
        for level, stride in enumerate(self.strides[1:], 1):
            resolved = self.cell * magnification * stride < 0.5
            levels[resolved] = np.maximum(levels[resolved], level)
        return self._draws(levels, self._visible(matrix, centers, axes))

    def prepare(self, renderer: RendererGL, obj: MeshInstancerGL) -> None:
        """Prepare camera-relative bounds and pass-specific draws for this frame.

        Projected voxel size selects detail per instance. Hysteresis preserves a
        previous coarse level near a threshold to avoid oscillation while moving
        the camera. Shadow draws subsequently reuse these transformed bounds.
        """
        # Reuse the matrices already staged for GL. This adds no per-frame
        # device readback on the host-upload path, and only small transforms
        # when explicit CUDA/GL interoperability is enabled.
        transforms = obj._host_transforms
        if transforms is None:
            transforms = obj.world_xforms[: obj.active_instances].numpy()
        transforms = transforms[: obj.active_instances]
        scale = np.linalg.norm(transforms[:, :3, :3], axis=2).max(axis=1)
        centers = np.einsum("i,nij->nj", self.center, transforms[:, :3, :3]) + transforms[:, 3, :3]
        clearance = np.maximum(
            np.linalg.norm(centers - renderer.camera.pos, axis=1) - self.radius * scale, renderer.camera.near
        )
        focal = abs(renderer._projection_matrix[5]) * renderer._get_scene_viewport()[3] * 0.5
        pixels = self.cell * scale * focal / clearance
        levels = np.zeros(len(transforms), dtype=np.int32)
        previous = getattr(obj, "_lod_levels", None)
        for level, stride in enumerate(self.strides[1:], 1):
            threshold = np.full(len(transforms), 0.5)
            if previous is not None and len(previous) == len(levels):
                # Enter a coarse level below half a pixel; leave it only after
                # its projected cells grow beyond three quarters of a pixel.
                threshold[previous >= level] = 0.75
            levels[pixels * stride < threshold] = level
        obj._lod_levels = levels
        # Use the cropped viewport's projection, in camera-relative coordinates
        # just like the shaders. Do not submit millions of offscreen triangles
        # to every X-ray pass simply because their instance batch is visible.
        centers = centers - renderer.camera.pos
        axes = transforms[:, :3, :3]
        obj._lod_bounds = centers, axes
        view = renderer._view_matrix.reshape(4, 4).T.copy()
        view[:3, 3] = 0.0
        matrix = renderer._projection_matrix.reshape(4, 4).T @ view
        visible = self._visible(matrix, centers, axes)
        obj._lod_draws = self._draws(levels, visible)
        obj._lod_opaque_draws = obj._lod_transparent_draws = obj._lod_wire_draws = None
        if getattr(renderer, "_physics_xray", False):
            self._prepare_xray(renderer, obj, transforms, levels, visible, view)

    def _prepare_xray(
        self,
        renderer: RendererGL,
        obj: MeshInstancerGL,
        transforms: np.ndarray,
        levels: np.ndarray,
        visible: np.ndarray,
        view: np.ndarray,
    ) -> None:
        """Skip entire instance ranges whose fragments the inspection shaders reject."""
        from .physics_view import _mesh_metrics  # noqa: PLC0415

        center, half = _mesh_metrics(obj.mesh)
        axes = transforms[:, :3, :3]
        scale = np.maximum(np.linalg.norm(axes, axis=2), 1e-12)
        # Subtract translations first, matching the shader at large coordinates.
        delta = np.asarray(renderer.camera.pos, dtype=np.float32) - transforms[:, 3, :3]
        delta -= np.einsum("i,nij->nj", center, axes)
        local_camera = np.einsum("nj,nij->ni", delta, axes) / scale
        extent = half * scale
        clearance = np.linalg.norm(np.maximum(np.abs(local_camera) - extent, 0.0), axis=1)
        radius = np.maximum(np.linalg.norm(extent, axis=1), max(renderer._xray_inspection_radius, 1e-9))
        reach = radius * renderer._projection_matrix[5] * max(renderer._xray_proximity_range, 1e-6)
        # Match the shader's proximity cutoff. Let both passes handle the tiny
        # boundary band so float rounding cannot leave a hole in the surface.
        fading = renderer._xray_proximity_range > 0.0
        transparent = fading & (clearance <= 5.001 * reach)
        opaque = ~transparent | (clearance >= 4.999 * reach)
        obj._lod_opaque_draws = self._draws(levels, visible & opaque)
        obj._lod_transparent_draws = self._draws(levels, visible & transparent)
        wire_visible = visible & transparent
        if renderer._xray_adaptive:
            # Every marching-cubes triangle fits inside one sampled cell. Bound
            # its projected longest edge using the largest perspective Jacobian
            # over the instance box; inradius <= longest_edge / (2 * sqrt(3)).
            # This rejects only wire triangles already discarded by the geometry
            # shader, without reading vertices or reducing surface detail.
            centers, axes = obj._lod_bounds
            camera_centers = centers @ view[:3, :3].T
            camera_extent = np.einsum("nij,i->nj", np.abs(axes @ view[:3, :3].T), self.half)
            depth = -camera_centers[:, 2] - camera_extent[:, 2]
            safe_depth = np.maximum(depth, 1e-12)
            slope = (np.abs(camera_centers[:, :2]) + camera_extent[:, :2]) / safe_depth[:, None]
            _, _, width, height = renderer._get_scene_viewport()
            focal = max(abs(renderer._projection_matrix[0]) * width, abs(renderer._projection_matrix[5]) * height) / 2
            edge = self.cell * scale.max(axis=1) * self.strides[levels]
            projected_edge = edge * focal * np.sqrt(1 + np.sum(slope * slope, axis=1)) / safe_depth
            wire_visible &= (depth <= 0) | (projected_edge > 2 * np.sqrt(3))
        obj._lod_wire_draws = self._draws(levels, wire_visible)
