# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..core import Axis
from ..geometry import GeoType, Mesh
from ..sim import Model, State
from ..utils import load_texture, normalize_texture
from .render import create_kernel
from .types import ClearData, LightType, MeshData, RenderConfig, RenderOrder, TextureData


class RenderContext:
    @dataclass(unsafe_hash=True)
    class RenderState:
        """Mutable flags tracking which render outputs are active."""

        num_gaussians: int = 0
        has_particles: bool = False
        render_color: bool = False
        render_depth: bool = False
        render_forward_depth: bool = False
        render_shape_index: bool = False
        render_normal: bool = False
        render_albedo: bool = False
        render_hdr_color: bool = False

    DEFAULT_CLEAR_DATA = ClearData()
    DEFAULT_RENDER_CONFIG = RenderConfig()

    def __init__(self, model: Model, load_textures: bool = True):
        """Create a render context for a Newton simulation model.

        Populates shape, triangle, and texture data from *model*. BVH
        acceleration structures for shapes and particles live on
        :class:`~newton.Model` and are built for the initial state by
        :meth:`~newton.ModelBuilder.finalize`; refit them via
        :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles` before later frames that
        change geometry.

        Args:
            model: Newton simulation model providing shapes and particles.
            load_textures: Load mesh textures from disk. Set False for
                checkerboard or custom texture workflows.
        """
        self._model = model
        self._render_state = RenderContext.RenderState()

        self._kernel_cache: dict[int, wp.Kernel] = {}

        self._triangle_mesh: wp.Mesh | None = None
        self._triangle_mesh_group_roots: wp.array[wp.int32] = wp.full(
            model.world_count + 1, value=-1, dtype=wp.int32, device=self.device
        )

        self._triangle_points: wp.array[wp.vec3f] | None = None
        self._triangle_indices: wp.array[wp.int32] | None = None
        self._topology_particle_mask: wp.array[wp.bool] | None = None

        self._has_particles: bool = False

        self._shape_texture_ids: wp.array[wp.int32] | None = None
        self._shape_mesh_data_ids: wp.array[wp.int32] | None = None
        self._shape_render_type: wp.array[wp.int32] | None = None

        self._mesh_data: wp.array[MeshData] | None = None
        self._texture_data: wp.array[TextureData] | None = None
        self._texture_data_source: list[TextureData] = []
        self._mesh_data_source: list[MeshData] = []

        self._lights_active: wp.array[wp.bool] | None = None
        self._lights_type: wp.array[wp.int32] | None = None
        self._lights_cast_shadow: wp.array[wp.bool] | None = None
        self._lights_position: wp.array[wp.vec3f] | None = None
        self._lights_orientation: wp.array[wp.vec3f] | None = None

        # Heightfields are triangulated meshes (their wp.Mesh lives in
        # shape_source_ptr), so the renderer treats them as meshes: it reuses
        # the MESH ray-intersection path, which keeps heightfield handling out
        # of the render kernels entirely (no extra shape-type branch, so no
        # register/occupancy cost). The remapped type array is what the render
        # kernel dispatches on; model.shape_type (HFIELD) is left untouched for
        # collision and BVH bounds.
        if model.shape_type is not None:
            shape_type_np = model.shape_type.numpy()
            if np.any(shape_type_np == int(GeoType.HFIELD)):
                shape_type_np = shape_type_np.copy()
                shape_type_np[shape_type_np == int(GeoType.HFIELD)] = int(GeoType.MESH)
                self._shape_render_type = wp.array(shape_type_np, dtype=wp.int32, device=model.shape_type.device)

        if model.particle_q is not None and model.particle_q.shape[0]:
            self._has_particles = True
            self._render_state.has_particles = True
            topology_particle_mask = np.zeros(model.particle_q.shape[0], dtype=bool)

            def mask_topology_particles(indices: wp.array[wp.int32] | None):
                if indices is not None and indices.shape[0]:
                    topology_particle_mask[indices.numpy().reshape(-1)] = True

            if model.tri_indices is not None and model.tri_indices.shape[0]:
                self._set_triangle_points(model.particle_q)
                self._set_triangle_indices(model.tri_indices.flatten())
                # Deformable-owned vertices render through the triangle mesh; tet indices catch
                # interior volume particles that are not referenced by boundary triangles.
                mask_topology_particles(model.tri_indices)
                mask_topology_particles(model.tet_indices)
            self._topology_particle_mask = wp.array(
                topology_particle_mask, dtype=wp.bool, device=model.particle_q.device
            )

        if model.gaussians_data is not None:
            self._render_state.num_gaussians = model.gaussians_data.shape[0]

        self._load_texture_and_mesh_data(model, load_textures)

    @property
    def model(self) -> Model:
        return self._model

    @property
    def device(self) -> wp.Device:
        return self.model.device

    @property
    def world_count(self) -> int:
        return self.model.world_count

    @property
    def up_axis(self) -> Axis:
        return Axis.from_any(self.model.up_axis)

    def _get_shape_render_type(self) -> wp.array[wp.int32] | None:
        if self._shape_render_type is not None:
            return self._shape_render_type
        return self.model.shape_type

    def update(self, state: State):
        """Synchronize triangle-mesh points from the current simulation state.

        Shape and particle BVHs are built by :meth:`~newton.ModelBuilder.finalize`
        and refit separately via :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles`.

        Args:
            state: Current simulation state with particle positions.
        """

        if self._has_triangle_mesh:
            self._set_triangle_points(state.particle_q)
            self._sync_triangle_mesh()

    def create_default_light(
        self,
        enable_shadows: bool = True,
        direction: wp.vec3f | None = None,
    ) -> None:
        """Create a default directional light oriented at ``(-1, 1, -1)``.

        Args:
            enable_shadows: Enable shadow casting for this light.
            direction: Normalized light direction. If ``None``, defaults to
                (normalized ``(-1, 1, -1)``).
        """
        self._lights_active = wp.array([True], dtype=wp.bool, device=self.device)
        self._lights_type = wp.array([LightType.DIRECTIONAL], dtype=wp.int32, device=self.device)
        self._lights_cast_shadow = wp.array([enable_shadows], dtype=wp.bool, device=self.device)
        self._lights_position = wp.array([wp.vec3f(0.0)], dtype=wp.vec3f, device=self.device)
        self._lights_orientation = wp.array(
            [direction if direction is not None else wp.vec3f(-0.57735026, 0.57735026, -0.57735026)],
            dtype=wp.vec3f,
            device=self.device,
        )

    def assign_checkerboard_material(
        self,
        *,
        shape_indices: Sequence[int] | np.ndarray,
        resolution: int = 64,
        checker_size: int = 32,
    ) -> None:
        """Assign a gray checkerboard texture material to selected shapes.

        Args:
            shape_indices: Shape indices that should use the checkerboard texture.
            resolution: Texture resolution in pixels (square texture).
            checker_size: Size of each checkerboard square in pixels.
        """
        shape_indices = np.asarray(shape_indices, dtype=np.int64).reshape(-1)
        invalid = (shape_indices < 0) | (shape_indices >= self.model.shape_count)
        if invalid.any():
            raise ValueError("shape_indices contains an out-of-range shape index")

        checkerboard = (
            (np.arange(resolution) // checker_size)[:, None] + (np.arange(resolution) // checker_size)
        ) % 2 == 0

        pixels = np.where(checkerboard, 0xFF808080, 0xFFBFBFBF).astype(np.uint32)

        texture_ids = np.full(self.model.shape_count, fill_value=-1, dtype=np.int32)
        texture_ids[shape_indices] = 0

        checkerboard_data = TextureData()
        checkerboard_data.texture = wp.Texture2D(
            pixels.view(np.uint8).reshape(resolution, resolution, 4),
            filter_mode=wp.TextureFilterMode.CLOSEST,
            address_mode=wp.TextureAddressMode.WRAP,
            normalized_coords=True,
            dtype=wp.uint8,
            num_channels=4,
            device=self.device,
        )

        checkerboard_data.repeat = wp.vec2f(1.0, 1.0)

        self._texture_data_source = [checkerboard_data]
        self._texture_data = wp.array(self._texture_data_source, dtype=TextureData, device=self.device)
        self._shape_texture_ids = wp.array(texture_ids, dtype=wp.int32, device=self.device)

    def render(
        self,
        state: State,
        *,
        camera_transforms: wp.array[wp.transformf],
        camera_rays: wp.array3d[wp.vec3f],
        world_indices: wp.array[wp.int32],
        color_image: wp.array3d[wp.uint32] | None = None,
        hdr_color_image: wp.array3d[wp.vec3f] | None = None,
        depth_image: wp.array3d[wp.float32] | None = None,
        forward_depth_image: wp.array3d[wp.float32] | None = None,
        shape_index_image: wp.array3d[wp.uint32] | None = None,
        normal_image: wp.array3d[wp.vec3f] | None = None,
        albedo_image: wp.array3d[wp.uint32] | None = None,
        clear_data: ClearData | None = DEFAULT_CLEAR_DATA,
        config: RenderConfig | None = DEFAULT_RENDER_CONFIG,
        kernel_block_dim: int = 64,
    ):
        """Raytrace the scene into the provided output images.

        Renders one view per camera transform. The number of views is
        ``view_count = camera_transforms.shape[0]`` and is independent of the
        model's world count. At least one output image must be supplied; all
        non-``None`` output arrays must have shape ``(view_count, height, width)``.

        Shape and particle BVHs on the model are built for the initial state by
        :meth:`~newton.ModelBuilder.finalize`. Before later frames that change
        geometry, refit them via
        :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles` before calling this
        method.

        Args:
            state: Current simulation state (for particle positions).
            camera_transforms: Per-view camera transforms, shape
                ``(view_count,)``.
            camera_rays: Ray origins and directions, shape
                ``(height, width, 2)``.
            world_indices: Per-view world selector, shape ``(view_count,)``,
                dtype ``int32``. A non-negative entry is the model world index
                rendered for that view; a negative entry disables the view using
                a :class:`~newton.WorldRenderFlag` sentinel
                (``DISABLE_PRESERVE`` / ``DISABLE_CLEAR``). World indices must be
                in ``[0, world_count)``.
            color_image: Output RGBA color buffer (packed ``uint32``).
            depth_image: Output depth buffer [m].
            forward_depth_image: Output forward-depth buffer [m].
            shape_index_image: Output shape-index buffer.
            normal_image: Output world-space surface normals.
            albedo_image: Output albedo buffer (packed ``uint32``).
            clear_data: Values used to clear output images before
                rendering. Pass ``None`` to use :attr:`DEFAULT_CLEAR_DATA`.
            hdr_color_image: Output linear HDR color buffer.
            config: Render settings for this render call. If ``None``, uses
                default :class:`Config` settings.
            kernel_block_dim: Thread block dimension forwarded to ``wp.launch``
                for the render megakernel.
        """
        model = self.model
        if config is None:
            config = RenderContext.DEFAULT_RENDER_CONFIG

        if model.shape_count > 0 and model.bvh_shape_enabled is None:
            raise RuntimeError(
                "Shape BVH is missing. ModelBuilder.finalize() builds it for finalized models; "
                "call model.bvh_build_shapes(state) for manually populated models."
            )

        has_shapes = model.bvh_shape_count_enabled > 0
        if has_shapes and (model.bvh_shapes is None or model.bvh_shapes_group_roots is None):
            raise RuntimeError("Shape BVH is incomplete; rebuild it with model.bvh_build_shapes(state).")

        has_particles = (
            config.enable_particles
            and self._render_state.has_particles
            and self._has_particles
            and state.particle_q is not None
            and state.particle_q.shape[0] > 0
        )
        if has_particles and (model.bvh_particles is None or model.bvh_particles_group_roots is None):
            raise RuntimeError(
                "Particle BVH is missing. ModelBuilder.finalize() builds it for finalized models; "
                "call model.bvh_build_particles(state) for manually populated models."
            )

        if has_shapes or has_particles or self._has_triangle_mesh or self._has_gaussians:
            height = camera_rays.shape[0]
            width = camera_rays.shape[1]

            if clear_data is None:
                clear_data = RenderContext.DEFAULT_CLEAR_DATA

            self._render_state.render_color = color_image is not None
            self._render_state.render_depth = depth_image is not None
            self._render_state.render_forward_depth = forward_depth_image is not None
            self._render_state.render_shape_index = shape_index_image is not None
            self._render_state.render_normal = normal_image is not None
            self._render_state.render_albedo = albedo_image is not None
            self._render_state.render_hdr_color = hdr_color_image is not None

            # One view per camera transform; independent of the model world count.
            view_count = camera_transforms.shape[0]

            assert camera_rays.shape == (height, width, 2), f"camera_rays size must match {height} x {width} x 2"

            assert world_indices.shape == (view_count,), f"world_indices size must match view count {view_count}"
            assert world_indices.dtype == wp.int32, f"world_indices dtype must be int32, got {world_indices.dtype}"
            assert world_indices.device == wp.get_device(self.device), (
                f"world_indices device must match {wp.get_device(self.device)}"
            )

            if color_image is not None:
                assert color_image.shape == (view_count, height, width), (
                    f"color_image size must match {view_count} x {height} x {width}"
                )

            if depth_image is not None:
                assert depth_image.shape == (view_count, height, width), (
                    f"depth_image size must match {view_count} x {height} x {width}"
                )

            if forward_depth_image is not None:
                assert forward_depth_image.shape == (view_count, height, width), (
                    f"forward_depth_image size must match {view_count} x {height} x {width}"
                )

            if shape_index_image is not None:
                assert shape_index_image.shape == (view_count, height, width), (
                    f"shape_index_image size must match {view_count} x {height} x {width}"
                )

            if normal_image is not None:
                assert normal_image.shape == (view_count, height, width), (
                    f"normal_image size must match {view_count} x {height} x {width}"
                )

            if albedo_image is not None:
                assert albedo_image.shape == (view_count, height, width), (
                    f"albedo_image size must match {view_count} x {height} x {width}"
                )
            if hdr_color_image is not None:
                assert hdr_color_image.shape == (view_count, height, width), (
                    f"hdr_color_image size must match {view_count} x {height} x {width}"
                )

            total_pixels = view_count * width * height

            # Reshaping output images to one dimension, slightly improves performance in the Kernel.
            if color_image is not None:
                color_image = color_image.reshape(total_pixels)
            if depth_image is not None:
                depth_image = depth_image.reshape(total_pixels)
            if forward_depth_image is not None:
                forward_depth_image = forward_depth_image.reshape(total_pixels)
            if shape_index_image is not None:
                shape_index_image = shape_index_image.reshape(total_pixels)
            if normal_image is not None:
                normal_image = normal_image.reshape(total_pixels)
            if albedo_image is not None:
                albedo_image = albedo_image.reshape(total_pixels)
            if hdr_color_image is not None:
                hdr_color_image = hdr_color_image.reshape(total_pixels)

            kernel_cache_key = hash((config, self._render_state, clear_data))
            render_kernel = self._kernel_cache.get(kernel_cache_key)
            if render_kernel is None:
                render_kernel = create_kernel(config, self._render_state, clear_data)
                self._kernel_cache[kernel_cache_key] = render_kernel

            particle_count = state.particle_q.shape[0] if has_particles else 0

            pixels_per_view = width * height
            if config.render_order == RenderOrder.TILED:
                tiles_x = (width + config.tile_width - 1) // config.tile_width
                tiles_y = (height + config.tile_height - 1) // config.tile_height
                pixels_per_view = tiles_x * tiles_y * config.tile_width * config.tile_height

            wp.launch(
                kernel=render_kernel,
                dim=(view_count * pixels_per_view),
                inputs=[
                    # Model and config
                    view_count,
                    self.light_count,
                    width,
                    height,
                    # Camera
                    camera_rays,
                    camera_transforms,
                    world_indices,
                    # Shape BVH
                    model.bvh_shape_count_enabled,
                    model.bvh_shapes.id if model.bvh_shapes is not None else 0,
                    model.bvh_shapes_group_roots,
                    # Shapes
                    model.bvh_shape_enabled,
                    self._get_shape_render_type(),  # HFIELD remapped to MESH; renderer treats heightfields as meshes
                    model.shape_scale,
                    model.shape_color,
                    model.bvh_shape_world_transforms,
                    model.shape_source_ptr,
                    self._shape_texture_ids,
                    self._shape_mesh_data_ids,
                    # Particle BVH
                    particle_count,
                    model.bvh_particles.id if model.bvh_particles is not None else 0,
                    model.bvh_particles_group_roots,
                    # Particles
                    state.particle_q if has_particles else None,
                    model.particle_radius if has_particles else None,
                    self._topology_particle_mask if has_particles else None,
                    # Triangle Mesh
                    self._triangle_mesh.id if self._triangle_mesh is not None else 0,
                    self._triangle_mesh_group_roots,
                    # Meshes
                    self._mesh_data,
                    # Gaussians
                    model.gaussians_data,
                    # Textures
                    self._texture_data,
                    # Lights
                    self._lights_active,
                    self._lights_type,
                    self._lights_cast_shadow,
                    self._lights_position,
                    self._lights_orientation,
                    # Outputs
                    color_image,
                    depth_image,
                    forward_depth_image,
                    shape_index_image,
                    normal_image,
                    albedo_image,
                    hdr_color_image,
                ],
                device=self.device,
                block_dim=kernel_block_dim,
            )

    @property
    def light_count(self) -> int:
        if self._lights_active is not None:
            return self._lights_active.shape[0]
        return 0

    @property
    def _has_triangle_mesh(self) -> bool:
        return self._triangle_points is not None and self._triangle_indices is not None

    @property
    def _has_gaussians(self) -> bool:
        return self.model.gaussians_data is not None

    def _set_triangle_points(self, triangle_points: wp.array[wp.vec3f]) -> None:
        if self._triangle_points is None or self._triangle_points.ptr != triangle_points.ptr:
            self._triangle_mesh = None
        self._triangle_points = triangle_points

    def _set_triangle_indices(self, triangle_indices: wp.array[wp.int32]) -> None:
        if self._triangle_indices is None or self._triangle_indices.ptr != triangle_indices.ptr:
            self._triangle_mesh = None
        self._triangle_indices = triangle_indices

    def _sync_triangle_mesh(self):
        if self._triangle_mesh is None:
            triangle_indices_np = self._triangle_indices.reshape((-1, 3)).numpy()
            particle_world_np = self.model.particle_world.numpy()
            triangle_world_np = particle_world_np[triangle_indices_np[:, 0]]
            triangle_groups_np = np.where(triangle_world_np < 0, self.world_count, triangle_world_np).astype(np.int32)
            triangle_groups = wp.array(triangle_groups_np, dtype=wp.int32, device=self.device)

            self._triangle_mesh = wp.Mesh(
                self._triangle_points, self._triangle_indices, groups=triangle_groups, bvh_constructor="sah"
            )

            wp.launch(
                kernel=RenderContext._compute_mesh_group_roots,
                dim=self.world_count + 1,
                inputs=[self._triangle_mesh.id, self._triangle_mesh_group_roots],
                device=self.device,
            )
        else:
            self._triangle_mesh.refit()

    @wp.kernel(enable_backward=False)
    def _compute_mesh_group_roots(mesh_id: wp.uint64, out_group_roots: wp.array[wp.int32]):
        group = wp.tid()
        out_group_roots[group] = wp.mesh_get_group_root(mesh_id, group)

    def _load_texture_and_mesh_data(self, model: Model, load_textures: bool):
        """Load mesh UV/normal data and textures from *model*.

        Populates mesh data, texture data, and the
        per-shape texture/mesh-data index arrays. Textures and mesh
        data are deduplicated by hash/identity.

        Args:
            model: Newton simulation model containing shape sources.
            load_textures: If ``True``, load image textures from disk;
                otherwise assign ``-1`` texture IDs to all shapes.
        """
        self._mesh_data_source = []
        self._texture_data_source = []

        texture_hashes = {}
        mesh_hashes = {}

        mesh_data_ids = []
        texture_data_ids = []

        for shape in model.shape_source:
            if isinstance(shape, Mesh):
                if shape.texture is not None and load_textures:
                    if shape.texture_hash not in texture_hashes:
                        pixels = load_texture(shape.texture)
                        if pixels is None:
                            raise ValueError(f"Failed to load texture: {shape.texture}")

                        # Normalize texture to ensure a consistent channel layout and dtype
                        pixels = normalize_texture(pixels, require_channels=True)
                        if pixels.dtype != np.uint8:
                            pixels = pixels.astype(np.uint8, copy=False)

                        texture_hashes[shape.texture_hash] = len(self._texture_data_source)

                        data = TextureData()
                        data.texture = wp.Texture2D(
                            pixels,
                            filter_mode=wp.TextureFilterMode.LINEAR,
                            address_mode=wp.TextureAddressMode.WRAP,
                            normalized_coords=True,
                            dtype=wp.uint8,
                            num_channels=4,
                            device=self.device,
                        )
                        data.repeat = wp.vec2f(1.0, 1.0)
                        self._texture_data_source.append(data)

                    texture_data_ids.append(texture_hashes[shape.texture_hash])
                else:
                    texture_data_ids.append(-1)

                if shape.uvs is not None or shape.normals is not None:
                    if shape not in mesh_hashes:
                        mesh_hashes[shape] = len(self._mesh_data_source)

                        data = MeshData()
                        if shape.uvs is not None:
                            data.uvs = wp.array(shape.uvs, dtype=wp.vec2f, device=self.device)
                        if shape.normals is not None:
                            data.normals = wp.array(shape.normals, dtype=wp.vec3f, device=self.device)
                        self._mesh_data_source.append(data)

                    mesh_data_ids.append(mesh_hashes[shape])
                else:
                    mesh_data_ids.append(-1)
            else:
                texture_data_ids.append(-1)
                mesh_data_ids.append(-1)

        self._texture_data = wp.array(self._texture_data_source, dtype=TextureData, device=self.device)
        self._shape_texture_ids = wp.array(texture_data_ids, dtype=wp.int32, device=self.device)

        self._mesh_data = wp.array(self._mesh_data_source, dtype=MeshData, device=self.device)
        self._shape_mesh_data_ids = wp.array(mesh_data_ids, dtype=wp.int32, device=self.device)
