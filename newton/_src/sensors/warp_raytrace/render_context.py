# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...core import Axis
from ...geometry import Gaussian, GeoType, Mesh
from ...sim import Model, State
from ...utils import load_texture, normalize_texture
from .render import create_kernel
from .types import ClearData, MeshData, RenderConfig, RenderOrder, TextureData


class RenderContext:
    Config = RenderConfig
    ClearData = ClearData

    @dataclass(unsafe_hash=True)
    class State:
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
    DEFAULT_RENDER_CONFIG = Config()

    def __init__(self, world_count: int = 1, device: str | None = None):
        """Create a new render context.

        Args:
            world_count: Number of simulation worlds to render.
            device: Warp device string (e.g. ``"cuda:0"``). If ``None``,
                the default Warp device is used.
        """
        self.device: str | None = device
        self.state = RenderContext.State()

        self.kernel_cache: dict[int, wp.Kernel] = {}

        self.world_count: int = world_count
        self.up_axis: Axis = Axis.Z

        self.triangle_mesh: wp.Mesh | None = None
        self.triangle_mesh_group_roots: wp.array[wp.int32] = wp.full(
            self.world_count + 1, value=-1, dtype=wp.int32, device=self.device
        )

        self.__triangle_points: wp.array[wp.vec3f] | None = None
        self.__triangle_indices: wp.array[wp.int32] | None = None
        self.__topology_particle_mask: wp.array[wp.bool] | None = None

        self.__gaussians_data: wp.array[Gaussian.Data] | None = None
        self.__has_particles: bool = False

        self.shape_count_total: int = 0
        self.shape_world_index: wp.array[wp.int32] | None = None
        self.shape_colors: wp.array[wp.vec3f] | None = None
        self.shape_source_ptr: wp.array[wp.uint64] | None = None
        self.shape_texture_ids: wp.array[wp.int32] | None = None
        self.shape_mesh_data_ids: wp.array[wp.int32] | None = None
        self.shape_render_type: wp.array[wp.int32] | None = None

        self.mesh_data: wp.array[MeshData] | None = None
        self.texture_data: wp.array[TextureData] | None = None

        self.particle_world: wp.array[wp.int32] | None = None

        self.lights_active: wp.array[wp.bool] | None = None
        self.lights_type: wp.array[wp.int32] | None = None
        self.lights_cast_shadow: wp.array[wp.bool] | None = None
        self.lights_position: wp.array[wp.vec3f] | None = None
        self.lights_orientation: wp.array[wp.vec3f] | None = None

    def init_from_model(self, model: Model, load_textures: bool = True):
        """Initialize render context state from a Newton simulation model.

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

        self.world_count = model.world_count
        self.up_axis = Axis.from_any(model.up_axis)
        self.triangle_mesh = None
        self.triangle_mesh_group_roots = wp.full(self.world_count + 1, value=-1, dtype=wp.int32, device=self.device)
        self.__triangle_points = None
        self.__triangle_indices = None
        self.__topology_particle_mask = None
        self.__has_particles = False
        self.state.has_particles = False

        self.shape_count_total = model.shape_count
        self.shape_world_index = model.shape_world
        self.shape_source_ptr = model.shape_source_ptr

        # Heightfields are triangulated meshes (their wp.Mesh lives in
        # shape_source_ptr), so the renderer treats them as meshes: it reuses
        # the MESH ray-intersection path, which keeps heightfield handling out
        # of the render kernels entirely (no extra shape-type branch, so no
        # register/occupancy cost). The remapped type array is what the render
        # kernel dispatches on; model.shape_type (HFIELD) is left untouched for
        # collision and BVH bounds.
        self.shape_render_type = model.shape_type
        if model.shape_type is not None:
            shape_type_np = model.shape_type.numpy()
            if np.any(shape_type_np == int(GeoType.HFIELD)):
                shape_type_np = shape_type_np.copy()
                shape_type_np[shape_type_np == int(GeoType.HFIELD)] = int(GeoType.MESH)
                self.shape_render_type = wp.array(shape_type_np, dtype=wp.int32, device=model.shape_type.device)

        if model.particle_q is not None and model.particle_q.shape[0]:
            self.__has_particles = True
            self.state.has_particles = True
            topology_particle_mask = np.zeros(model.particle_q.shape[0], dtype=bool)

            def mask_topology_particles(indices: wp.array[wp.int32] | None):
                if indices is not None and indices.shape[0]:
                    topology_particle_mask[indices.numpy().reshape(-1)] = True

            if model.tri_indices is not None and model.tri_indices.shape[0]:
                self.triangle_points = model.particle_q
                self.triangle_indices = model.tri_indices.flatten()
                self.particle_world = model.particle_world
                # Deformable-owned vertices render through the triangle mesh; tet indices catch
                # interior volume particles that are not referenced by boundary triangles.
                mask_topology_particles(model.tri_indices)
                mask_topology_particles(model.tet_indices)
            self.__topology_particle_mask = wp.array(
                topology_particle_mask, dtype=wp.bool, device=model.particle_q.device
            )

        self.shape_colors = model.shape_color
        self.gaussians_data = model.gaussians_data

        self.__load_texture_and_mesh_data(model, load_textures)

    def update(self, model: Model, state: State):
        """Synchronize triangle-mesh points from the current simulation state.

        Shape and particle BVHs are built by :meth:`~newton.ModelBuilder.finalize`
        and refit separately via :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles`.

        Args:
            model: Newton simulation model (for shape metadata).
            state: Current simulation state with particle positions.
        """

        if self.has_triangle_mesh:
            self.triangle_points = state.particle_q
            self._sync_triangle_mesh()

    def render(
        self,
        model: Model,
        state: State,
        *,
        camera_transforms: wp.array2d[wp.transformf],
        camera_rays: wp.array4d[wp.vec3f],
        color_image: wp.array4d[wp.uint32] | None = None,
        hdr_color_image: wp.array4d[wp.vec3f] | None = None,
        depth_image: wp.array4d[wp.float32] | None = None,
        forward_depth_image: wp.array4d[wp.float32] | None = None,
        shape_index_image: wp.array4d[wp.uint32] | None = None,
        normal_image: wp.array4d[wp.vec3f] | None = None,
        albedo_image: wp.array4d[wp.uint32] | None = None,
        clear_data: RenderContext.ClearData | None = DEFAULT_CLEAR_DATA,
        config: RenderContext.Config | None = DEFAULT_RENDER_CONFIG,
        kernel_block_dim: int = 64,
    ):
        """Raytrace the scene into the provided output images.

        At least one output image must be supplied. All non-``None``
        output arrays must have shape
        ``(world_count, camera_count, height, width)``.

        Shape and particle BVHs on *model* are built for the initial state by
        :meth:`~newton.ModelBuilder.finalize`. Before later frames that change
        geometry, refit them via
        :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles` before calling this
        method.

        Args:
            model: Simulation model providing shape metadata and BVHs.
            state: Current simulation state (for particle positions).
            camera_transforms: Per-camera transforms, shape
                ``(camera_count, world_count)``.
            camera_rays: Ray origins and directions, shape
                ``(camera_count, height, width, 2)``.
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
            and self.state.has_particles
            and self.__has_particles
            and state.particle_q is not None
            and state.particle_q.shape[0] > 0
        )
        if has_particles and (model.bvh_particles is None or model.bvh_particles_group_roots is None):
            raise RuntimeError(
                "Particle BVH is missing. ModelBuilder.finalize() builds it for finalized models; "
                "call model.bvh_build_particles(state) for manually populated models."
            )

        if has_shapes or has_particles or self.has_triangle_mesh or self.has_gaussians:
            width = camera_rays.shape[2]
            height = camera_rays.shape[1]
            camera_count = camera_rays.shape[0]

            if clear_data is None:
                clear_data = RenderContext.DEFAULT_CLEAR_DATA

            self.state.render_color = color_image is not None
            self.state.render_depth = depth_image is not None
            self.state.render_forward_depth = forward_depth_image is not None
            self.state.render_shape_index = shape_index_image is not None
            self.state.render_normal = normal_image is not None
            self.state.render_albedo = albedo_image is not None
            self.state.render_hdr_color = hdr_color_image is not None

            assert camera_transforms.shape == (camera_count, self.world_count), (
                f"camera_transforms size must match {camera_count} x {self.world_count}"
            )

            assert camera_rays.shape == (camera_count, height, width, 2), (
                f"camera_rays size must match {camera_count} x {height} x {width} x 2"
            )

            if color_image is not None:
                assert color_image.shape == (self.world_count, camera_count, height, width), (
                    f"color_image size must match {self.world_count} x {camera_count} x {height} x {width}"
                )

            if depth_image is not None:
                assert depth_image.shape == (self.world_count, camera_count, height, width), (
                    f"depth_image size must match {self.world_count} x {camera_count} x {height} x {width}"
                )

            if forward_depth_image is not None:
                assert forward_depth_image.shape == (self.world_count, camera_count, height, width), (
                    f"forward_depth_image size must match {self.world_count} x {camera_count} x {height} x {width}"
                )

            if shape_index_image is not None:
                assert shape_index_image.shape == (self.world_count, camera_count, height, width), (
                    f"shape_index_image size must match {self.world_count} x {camera_count} x {height} x {width}"
                )

            if normal_image is not None:
                assert normal_image.shape == (self.world_count, camera_count, height, width), (
                    f"normal_image size must match {self.world_count} x {camera_count} x {height} x {width}"
                )

            if albedo_image is not None:
                assert albedo_image.shape == (self.world_count, camera_count, height, width), (
                    f"albedo_image size must match {self.world_count} x {camera_count} x {height} x {width}"
                )
            if hdr_color_image is not None:
                assert hdr_color_image.shape == (self.world_count, camera_count, height, width), (
                    f"hdr_color_image size must match {self.world_count} x {camera_count} x {height} x {width}"
                )

            if config.render_order == RenderOrder.TILED:
                assert width % config.tile_width == 0, "render width must be a multiple of tile_width"
                assert height % config.tile_height == 0, "render height must be a multiple of tile_height"

            # Reshaping output images to one dimension, slightly improves performance in the Kernel.
            if color_image is not None:
                color_image = color_image.reshape(self.world_count * camera_count * width * height)
            if depth_image is not None:
                depth_image = depth_image.reshape(self.world_count * camera_count * width * height)
            if forward_depth_image is not None:
                forward_depth_image = forward_depth_image.reshape(self.world_count * camera_count * width * height)
            if shape_index_image is not None:
                shape_index_image = shape_index_image.reshape(self.world_count * camera_count * width * height)
            if normal_image is not None:
                normal_image = normal_image.reshape(self.world_count * camera_count * width * height)
            if albedo_image is not None:
                albedo_image = albedo_image.reshape(self.world_count * camera_count * width * height)
            if hdr_color_image is not None:
                hdr_color_image = hdr_color_image.reshape(self.world_count * camera_count * width * height)

            kernel_cache_key = hash((config, self.state, clear_data))
            render_kernel = self.kernel_cache.get(kernel_cache_key)
            if render_kernel is None:
                render_kernel = create_kernel(config, self.state, clear_data)
                self.kernel_cache[kernel_cache_key] = render_kernel

            particle_count = state.particle_q.shape[0] if has_particles else 0

            wp.launch(
                kernel=render_kernel,
                dim=(self.world_count * camera_count * width * height),
                inputs=[
                    # Model and config
                    self.world_count,
                    camera_count,
                    self.light_count,
                    width,
                    height,
                    # Camera
                    camera_rays,
                    camera_transforms,
                    # Shape BVH
                    model.bvh_shape_count_enabled,
                    model.bvh_shapes.id if model.bvh_shapes is not None else 0,
                    model.bvh_shapes_group_roots,
                    # Shapes
                    model.bvh_shape_enabled,
                    self.shape_render_type,  # HFIELD remapped to MESH; renderer treats heightfields as meshes
                    model.shape_scale,
                    self.shape_colors,
                    model.bvh_shape_world_transforms,
                    self.shape_source_ptr,
                    self.shape_texture_ids,
                    self.shape_mesh_data_ids,
                    # Particle BVH
                    particle_count,
                    model.bvh_particles.id if model.bvh_particles is not None else 0,
                    model.bvh_particles_group_roots,
                    # Particles
                    state.particle_q if has_particles else None,
                    model.particle_radius if has_particles else None,
                    self.__topology_particle_mask if has_particles else None,
                    # Triangle Mesh
                    self.triangle_mesh.id if self.triangle_mesh is not None else 0,
                    self.triangle_mesh_group_roots,
                    # Meshes
                    self.mesh_data,
                    # Gaussians
                    self.gaussians_data,
                    # Textures
                    self.texture_data,
                    # Lights
                    self.lights_active,
                    self.lights_type,
                    self.lights_cast_shadow,
                    self.lights_position,
                    self.lights_orientation,
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
        if self.lights_active is not None:
            return self.lights_active.shape[0]
        return 0

    @property
    def gaussians_count_total(self) -> int:
        if self.gaussians_data is not None:
            return self.gaussians_data.shape[0]
        return 0

    @property
    def has_particles(self) -> bool:
        return self.__has_particles

    @property
    def has_triangle_mesh(self) -> bool:
        return self.__triangle_points is not None and self.__triangle_indices is not None

    @property
    def has_gaussians(self) -> bool:
        return self.gaussians_data is not None

    @property
    def triangle_points(self) -> wp.array[wp.vec3f]:
        return self.__triangle_points

    @triangle_points.setter
    def triangle_points(self, triangle_points: wp.array[wp.vec3f]):
        if self.__triangle_points is None or self.__triangle_points.ptr != triangle_points.ptr:
            self.triangle_mesh = None
        self.__triangle_points = triangle_points

    @property
    def triangle_indices(self) -> wp.array[wp.int32]:
        return self.__triangle_indices

    @triangle_indices.setter
    def triangle_indices(self, triangle_indices: wp.array[wp.int32]):
        if self.__triangle_indices is None or self.__triangle_indices.ptr != triangle_indices.ptr:
            self.triangle_mesh = None
        self.__triangle_indices = triangle_indices

    def _sync_triangle_mesh(self):
        if self.triangle_mesh is None:
            triangle_indices_np = self.triangle_indices.reshape((-1, 3)).numpy()
            particle_world_np = self.particle_world.numpy()
            triangle_world_np = particle_world_np[triangle_indices_np[:, 0]]
            triangle_groups_np = np.where(triangle_world_np < 0, self.world_count, triangle_world_np).astype(np.int32)
            triangle_groups = wp.array(triangle_groups_np, dtype=wp.int32, device=self.device)

            self.triangle_mesh = wp.Mesh(
                self.triangle_points, self.triangle_indices, groups=triangle_groups, bvh_constructor="sah"
            )

            wp.launch(
                kernel=RenderContext._compute_mesh_group_roots,
                dim=self.world_count + 1,
                inputs=[self.triangle_mesh.id, self.triangle_mesh_group_roots],
                device=self.device,
            )
        else:
            self.triangle_mesh.refit()

    @wp.kernel(enable_backward=False)
    def _compute_mesh_group_roots(mesh_id: wp.uint64, out_group_roots: wp.array[wp.int32]):
        group = wp.tid()
        out_group_roots[group] = wp.mesh_get_group_root(mesh_id, group)

    @property
    def gaussians_data(self) -> wp.array[Gaussian.Data]:
        return self.__gaussians_data

    @gaussians_data.setter
    def gaussians_data(self, gaussians_data: wp.array[Gaussian.Data]):
        self.__gaussians_data = gaussians_data
        if gaussians_data is None:
            self.state.num_gaussians = 0
        else:
            self.state.num_gaussians = gaussians_data.shape[0]

    def __load_texture_and_mesh_data(self, model: Model, load_textures: bool):
        """Load mesh UV/normal data and textures from *model*.

        Populates :attr:`mesh_data`, :attr:`texture_data`, and the
        per-shape texture/mesh-data index arrays. Textures and mesh
        data are deduplicated by hash/identity.

        Args:
            model: Newton simulation model containing shape sources.
            load_textures: If ``True``, load image textures from disk;
                otherwise assign ``-1`` texture IDs to all shapes.
        """
        self.__mesh_data = []
        self.__texture_data = []

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

                        texture_hashes[shape.texture_hash] = len(self.__texture_data)

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
                        self.__texture_data.append(data)

                    texture_data_ids.append(texture_hashes[shape.texture_hash])
                else:
                    texture_data_ids.append(-1)

                if shape.uvs is not None or shape.normals is not None:
                    if shape not in mesh_hashes:
                        mesh_hashes[shape] = len(self.__mesh_data)

                        data = MeshData()
                        if shape.uvs is not None:
                            data.uvs = wp.array(shape.uvs, dtype=wp.vec2f, device=self.device)
                        if shape.normals is not None:
                            data.normals = wp.array(shape.normals, dtype=wp.vec3f, device=self.device)
                        self.__mesh_data.append(data)

                    mesh_data_ids.append(mesh_hashes[shape])
                else:
                    mesh_data_ids.append(-1)
            else:
                texture_data_ids.append(-1)
                mesh_data_ids.append(-1)

        self.texture_data = wp.array(self.__texture_data, dtype=TextureData, device=self.device)
        self.shape_texture_ids = wp.array(texture_data_ids, dtype=wp.int32, device=self.device)

        self.mesh_data = wp.array(self.__mesh_data, dtype=MeshData, device=self.device)
        self.shape_mesh_data_ids = wp.array(mesh_data_ids, dtype=wp.int32, device=self.device)
