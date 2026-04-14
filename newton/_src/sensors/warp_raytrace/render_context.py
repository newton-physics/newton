# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...core import MAXVAL
from ...geometry import Gaussian, GeoType, Mesh, ShapeFlags
from ...sim import Model, State
from ...utils import load_texture, normalize_texture
from .bvh import (
    compute_bvh_group_roots,
    compute_particle_bvh_bounds,
    compute_shape_bvh_bounds,
)
from .gaussians import compute_gaussian_bounds
from .render import create_kernel
from .types import ClearData, MeshData, RenderConfig, RenderOrder, TextureData
from .utils import Utils


@wp.kernel(enable_backward=False)
def compute_shape_bounds(
    in_shape_type: wp.array[wp.int32],
    in_shape_ptr: wp.array[wp.uint64],
    in_gaussians: wp.array[Gaussian.Data],
    out_bounds: wp.array2d[wp.vec3f],
):
    tid = wp.tid()

    min_point = wp.vec3(MAXVAL)
    max_point = wp.vec3(-MAXVAL)

    if in_shape_type[tid] == GeoType.MESH:
        mesh = wp.mesh_get(in_shape_ptr[tid])
        for i in range(mesh.points.shape[0]):
            min_point = wp.min(min_point, mesh.points[i])
            max_point = wp.max(max_point, mesh.points[i])

    elif in_shape_type[tid] == GeoType.GAUSSIAN:
        gaussian_id = in_shape_ptr[tid]
        for i in range(in_gaussians[gaussian_id].num_points):
            lower, upper = compute_gaussian_bounds(in_gaussians[gaussian_id], i)
            min_point = wp.min(min_point, lower)
            max_point = wp.max(max_point, upper)

    out_bounds[tid, 0] = min_point
    out_bounds[tid, 1] = max_point


@wp.kernel(enable_backward=False)
def convert_newton_transform(
    in_body_transforms: wp.array[wp.transform],
    in_shape_body: wp.array[wp.int32],
    in_transform: wp.array[wp.transformf],
    in_scale: wp.array[wp.vec3f],
    out_transforms: wp.array[wp.transformf],
    out_sizes: wp.array[wp.vec3f],
):
    tid = wp.tid()

    body = in_shape_body[tid]
    body_transform = wp.transform_identity()
    if body >= 0:
        body_transform = in_body_transforms[body]

    out_transforms[tid] = wp.mul(body_transform, in_transform[tid])
    out_sizes[tid] = in_scale[tid]


@wp.func
def is_supported_shape_type(shape_type: wp.int32) -> wp.bool:
    if shape_type == GeoType.BOX:
        return True
    if shape_type == GeoType.CAPSULE:
        return True
    if shape_type == GeoType.CYLINDER:
        return True
    if shape_type == GeoType.ELLIPSOID:
        return True
    if shape_type == GeoType.PLANE:
        return True
    if shape_type == GeoType.SPHERE:
        return True
    if shape_type == GeoType.CONE:
        return True
    if shape_type == GeoType.MESH:
        return True
    if shape_type == GeoType.GAUSSIAN:
        return True
    wp.printf("Unsupported shape geom type: %d\n", shape_type)
    return False


@wp.kernel(enable_backward=False)
def compute_enabled_shapes(
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    out_shape_enabled: wp.array[wp.uint32],
    out_shape_enabled_count: wp.array[wp.int32],
):
    tid = wp.tid()

    if not bool(shape_flags[tid] & ShapeFlags.VISIBLE):
        return

    if not is_supported_shape_type(shape_type[tid]):
        return

    index = wp.atomic_add(out_shape_enabled_count, 0, 1)
    out_shape_enabled[index] = wp.uint32(tid)


class RenderContext:
    Config = RenderConfig
    ClearData = ClearData

    @dataclass(unsafe_hash=True)
    class State:
        """Mutable flags tracking which render outputs are active."""

        num_gaussians: int = 0
        render_color: bool = False
        render_depth: bool = False
        render_shape_index: bool = False
        render_normal: bool = False
        render_albedo: bool = False

    DEFAULT_CLEAR_DATA = ClearData()

    def __init__(self, world_count: int = 1, config: Config | None = None, device: str | None = None):
        """Create a new render context.

        Args:
            world_count: Number of simulation worlds to render.
            config: Render configuration. If ``None``, uses default
                :class:`Config` settings.
            device: Warp device string (e.g. ``"cuda:0"``). If ``None``,
                the default Warp device is used.
        """
        self.device: str | None = device
        self.utils = Utils(self)
        self.config = config if config else RenderContext.Config()
        self.state = RenderContext.State()

        self.kernel_cache: dict[int, wp.Kernel] = {}

        self.world_count: int = world_count

        self.bvh_shapes: wp.Bvh | None = None
        self.bvh_shapes_group_roots: wp.array[wp.int32] | None = None

        self.bvh_particles: wp.Bvh | None = None
        self.bvh_particles_group_roots: wp.array[wp.int32] | None = None

        self.triangle_mesh: wp.Mesh | None = None
        self.shape_count_enabled: int = 0
        self.shape_count_total: int = 0

        self.__triangle_points: wp.array[wp.vec3f] | None = None
        self.__triangle_indices: wp.array[wp.int32] | None = None

        self.__particles_position: wp.array[wp.vec3f] | None = None
        self.__particles_radius: wp.array[wp.float32] | None = None
        self.__particles_world_index: wp.array[wp.int32] | None = None

        self.__gaussians_data: wp.array[Gaussian.Data] | None = None

        self.shape_enabled: wp.array[wp.uint32] | None = None
        self.shape_types: wp.array[wp.int32] | None = None
        self.shape_sizes: wp.array[wp.vec3f] | None = None
        self.shape_transforms: wp.array[wp.transformf] | None = None
        self.shape_colors: wp.array[wp.vec3f] | None = None
        self.shape_world_index: wp.array[wp.int32] | None = None
        self.shape_source_ptr: wp.array[wp.uint64] | None = None
        self.shape_bounds: wp.array2d[wp.vec3f] | None = None
        self.shape_texture_ids: wp.array[wp.int32] | None = None
        self.shape_mesh_data_ids: wp.array[wp.int32] | None = None

        self.mesh_data: wp.array[MeshData] | None = None
        self.texture_data: wp.array[TextureData] | None = None

        self.lights_active: wp.array[wp.bool] | None = None
        self.lights_type: wp.array[wp.int32] | None = None
        self.lights_cast_shadow: wp.array[wp.bool] | None = None
        self.lights_position: wp.array[wp.vec3f] | None = None
        self.lights_orientation: wp.array[wp.vec3f] | None = None

    def init_from_model(self, model: Model, load_textures: bool = True):
        """Initialize render context state from a Newton simulation model.

        Populates shape, particle, triangle, and texture data from *model*.
        Call once after construction or when the model topology changes.

        Args:
            model: Newton simulation model providing shapes and particles.
            load_textures: Load mesh textures from disk. Set False for
                checkerboard or custom texture workflows.
        """

        self.world_count = model.world_count
        self.bvh_shapes = None
        self.bvh_shapes_group_roots = None
        self.bvh_particles = None
        self.bvh_particles_group_roots = None
        self.triangle_mesh = None
        self.__triangle_points = None
        self.__triangle_indices = None
        self.__particles_position = None
        self.__particles_radius = None
        self.__particles_world_index = None

        self.shape_source_ptr = model.shape_source_ptr
        self.shape_bounds = wp.empty((model.shape_count, 2), dtype=wp.vec3f, ndim=2, device=self.device)

        if model.particle_q is not None and model.particle_q.shape[0]:
            self.particles_position = model.particle_q
            self.particles_radius = model.particle_radius
            self.particles_world_index = model.particle_world
            if model.tri_indices is not None and model.tri_indices.shape[0]:
                self.triangle_points = model.particle_q
                self.triangle_indices = model.tri_indices.flatten()
                self.config.enable_particles = False

        self.shape_enabled = wp.empty(model.shape_count, dtype=wp.uint32, device=self.device)
        self.shape_types = model.shape_type
        self.shape_sizes = wp.empty(model.shape_count, dtype=wp.vec3f, device=self.device)
        self.shape_transforms = wp.empty(model.shape_count, dtype=wp.transformf, device=self.device)

        self.shape_colors = model.shape_color
        self.shape_world_index = model.shape_world
        self.gaussians_data = model.gaussians_data

        self.__load_texture_and_mesh_data(model, load_textures)

        num_enabled_shapes = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=compute_enabled_shapes,
            dim=model.shape_count,
            inputs=[
                model.shape_type,
                model.shape_flags,
                self.shape_enabled,
                num_enabled_shapes,
            ],
            device=self.device,
        )
        self.shape_count_total = model.shape_count
        self.shape_count_enabled = int(num_enabled_shapes.numpy()[0])
        self.__compute_shape_bounds()

    def update(self, model: Model, state: State):
        """Synchronize transforms and particle positions from simulation state.

        Args:
            model: Newton simulation model (for shape metadata).
            state: Current simulation state with body transforms and
                particle positions.
        """

        if self.has_shapes:
            wp.launch(
                kernel=convert_newton_transform,
                dim=model.shape_count,
                inputs=[
                    state.body_q,
                    model.shape_body,
                    model.shape_transform,
                    model.shape_scale,
                    self.shape_transforms,
                    self.shape_sizes,
                ],
                device=self.device,
            )

        if self.has_triangle_mesh:
            self.triangle_points = state.particle_q

        if self.has_particles:
            self.particles_position = state.particle_q

    def refit_bvh(self):
        """Rebuild or refit the BVH acceleration structures.

        Updates shape, particle, and triangle-mesh BVHs so that
        subsequent :meth:`render` calls use current geometry positions.
        """
        self.bvh_shapes, self.bvh_shapes_group_roots = self.__update_bvh(
            self.bvh_shapes, self.bvh_shapes_group_roots, self.shape_count_enabled, self.__compute_bvh_bounds_shapes
        )
        self.bvh_particles, self.bvh_particles_group_roots = self.__update_bvh(
            self.bvh_particles,
            self.bvh_particles_group_roots,
            self.particle_count_total,
            self.__compute_bvh_bounds_particles,
        )

        if self.has_triangle_mesh:
            if self.triangle_mesh is None:
                self.triangle_mesh = wp.Mesh(self.triangle_points, self.triangle_indices, device=self.device)
            else:
                self.triangle_mesh.refit()

    def render(
        self,
        camera_transforms: wp.array2d[wp.transformf],
        camera_rays: wp.array4d[wp.vec3f],
        color_image: wp.array4d[wp.uint32] | None = None,
        depth_image: wp.array4d[wp.float32] | None = None,
        shape_index_image: wp.array4d[wp.uint32] | None = None,
        normal_image: wp.array4d[wp.vec3f] | None = None,
        albedo_image: wp.array4d[wp.uint32] | None = None,
        refit_bvh: bool = True,
        clear_data: RenderContext.ClearData | None = DEFAULT_CLEAR_DATA,
    ):
        """Raytrace the scene into the provided output images.

        At least one output image must be supplied. All non-``None``
        output arrays must have shape
        ``(world_count, camera_count, height, width)``.

        Args:
            camera_transforms: Per-camera transforms, shape
                ``(camera_count, world_count)``.
            camera_rays: Ray origins and directions, shape
                ``(camera_count, height, width, 2)``.
            color_image: Output RGBA color buffer (packed ``uint32``).
            depth_image: Output depth buffer [m].
            shape_index_image: Output shape-index buffer.
            normal_image: Output world-space surface normals.
            albedo_image: Output albedo buffer (packed ``uint32``).
            refit_bvh: If ``True``, call :meth:`refit_bvh` before
                rendering.
            clear_data: Values used to clear output images before
                rendering. Pass ``None`` to use :attr:`DEFAULT_CLEAR_DATA`.
        """
        if self.has_shapes or self.has_particles or self.has_triangle_mesh or self.has_gaussians:
            if refit_bvh:
                self.refit_bvh()

            width = camera_rays.shape[2]
            height = camera_rays.shape[1]
            camera_count = camera_rays.shape[0]

            if clear_data is None:
                clear_data = RenderContext.DEFAULT_CLEAR_DATA

            self.state.render_color = color_image is not None
            self.state.render_depth = depth_image is not None
            self.state.render_shape_index = shape_index_image is not None
            self.state.render_normal = normal_image is not None
            self.state.render_albedo = albedo_image is not None

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

            if self.config.render_order == RenderOrder.TILED:
                assert width % self.config.tile_width == 0, "render width must be a multiple of tile_width"
                assert height % self.config.tile_height == 0, "render height must be a multiple of tile_height"

            # Reshaping output images to one dimension, slightly improves performance in the Kernel.
            if color_image is not None:
                color_image = color_image.reshape(self.world_count * camera_count * width * height)
            if depth_image is not None:
                depth_image = depth_image.reshape(self.world_count * camera_count * width * height)
            if shape_index_image is not None:
                shape_index_image = shape_index_image.reshape(self.world_count * camera_count * width * height)
            if normal_image is not None:
                normal_image = normal_image.reshape(self.world_count * camera_count * width * height)
            if albedo_image is not None:
                albedo_image = albedo_image.reshape(self.world_count * camera_count * width * height)

            kernel_cache_key = hash((self.config, self.state, clear_data))
            render_kernel = self.kernel_cache.get(kernel_cache_key)
            if render_kernel is None:
                render_kernel = create_kernel(self.config, self.state, clear_data)
                self.kernel_cache[kernel_cache_key] = render_kernel

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
                    self.shape_count_enabled,
                    self.bvh_shapes.id if self.bvh_shapes else 0,
                    self.bvh_shapes_group_roots,
                    # Shapes
                    self.shape_enabled,
                    self.shape_types,
                    self.shape_sizes,
                    self.shape_colors,
                    self.shape_transforms,
                    self.shape_source_ptr,
                    self.shape_texture_ids,
                    self.shape_mesh_data_ids,
                    # Particle BVH
                    self.particle_count_total,
                    self.bvh_particles.id if self.bvh_particles else 0,
                    self.bvh_particles_group_roots,
                    # Particles
                    self.particles_position,
                    self.particles_radius,
                    # Triangle Mesh
                    self.triangle_mesh.id if self.triangle_mesh is not None else 0,
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
                    shape_index_image,
                    normal_image,
                    albedo_image,
                ],
                device=self.device,
            )

    @property
    def world_count_total(self) -> int:
        if self.config.enable_global_world:
            return self.world_count + 1
        return self.world_count

    @property
    def particle_count_total(self) -> int:
        if self.particles_position is not None:
            return self.particles_position.shape[0]
        return 0

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
    def has_shapes(self) -> bool:
        return self.shape_count_enabled > 0

    @property
    def has_particles(self) -> bool:
        return self.particles_position is not None

    @property
    def has_triangle_mesh(self) -> bool:
        return self.triangle_points is not None

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

    @property
    def particles_position(self) -> wp.array[wp.vec3f]:
        return self.__particles_position

    @particles_position.setter
    def particles_position(self, particles_position: wp.array[wp.vec3f]):
        if self.__particles_position is None or self.__particles_position.ptr != particles_position.ptr:
            self.bvh_particles = None
        self.__particles_position = particles_position

    @property
    def particles_radius(self) -> wp.array[wp.float32]:
        return self.__particles_radius

    @particles_radius.setter
    def particles_radius(self, particles_radius: wp.array[wp.float32]):
        if self.__particles_radius is None or self.__particles_radius.ptr != particles_radius.ptr:
            self.bvh_particles = None
        self.__particles_radius = particles_radius

    @property
    def particles_world_index(self) -> wp.array[wp.int32]:
        return self.__particles_world_index

    @particles_world_index.setter
    def particles_world_index(self, particles_world_index: wp.array[wp.int32]):
        if self.__particles_world_index is None or self.__particles_world_index.ptr != particles_world_index.ptr:
            self.bvh_particles = None
        self.__particles_world_index = particles_world_index

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

    def __update_bvh(
        self,
        bvh: wp.Bvh,
        group_roots: wp.array[wp.int32],
        size: int,
        bounds_callback: Callable[[wp.array[wp.vec3f], wp.array[wp.vec3f], wp.array[wp.int32]], None],
    ):
        """Build a new BVH or refit an existing one.

        If *bvh* is ``None`` a new :class:`wp.Bvh` is constructed and
        group roots are computed; otherwise the existing BVH is refit
        in-place.

        Args:
            bvh: Existing BVH to refit, or ``None`` to build a new one.
            group_roots: Existing group-root array, or ``None``.
            size: Number of primitives (shapes or particles).
            bounds_callback: Callback that fills lower/upper/group
                arrays for the BVH primitives.

        Returns:
            Tuple of ``(bvh, group_roots)``.
        """
        if size:
            lowers = bvh.lowers if bvh is not None else wp.zeros(size, dtype=wp.vec3f, device=self.device)
            uppers = bvh.uppers if bvh is not None else wp.zeros(size, dtype=wp.vec3f, device=self.device)
            groups = bvh.groups if bvh is not None else wp.zeros(size, dtype=wp.int32, device=self.device)

            bounds_callback(lowers, uppers, groups)

            if bvh is None:
                bvh = wp.Bvh(lowers, uppers, groups=groups)
                group_roots = wp.zeros((self.world_count_total), dtype=wp.int32, device=self.device)

                wp.launch(
                    kernel=compute_bvh_group_roots,
                    dim=self.world_count_total,
                    inputs=[bvh.id, group_roots],
                    device=self.device,
                )
            else:
                bvh.refit()

        return bvh, group_roots

    def __compute_bvh_bounds_shapes(
        self, lowers: wp.array[wp.vec3f], uppers: wp.array[wp.vec3f], groups: wp.array[wp.int32]
    ):
        """Compute axis-aligned bounding boxes for enabled shapes."""
        wp.launch(
            kernel=compute_shape_bvh_bounds,
            dim=self.shape_count_enabled,
            inputs=[
                self.shape_count_enabled,
                self.world_count_total,
                self.shape_world_index,
                self.shape_enabled,
                self.shape_types,
                self.shape_sizes,
                self.shape_transforms,
                self.shape_bounds,
                lowers,
                uppers,
                groups,
            ],
            device=self.device,
        )

    def __compute_bvh_bounds_particles(
        self, lowers: wp.array[wp.vec3f], uppers: wp.array[wp.vec3f], groups: wp.array[wp.int32]
    ):
        """Compute axis-aligned bounding boxes for particles."""
        wp.launch(
            kernel=compute_particle_bvh_bounds,
            dim=self.particle_count_total,
            inputs=[
                self.particle_count_total,
                self.world_count_total,
                self.particles_world_index,
                self.particles_position,
                self.particles_radius,
                lowers,
                uppers,
                groups,
            ],
            device=self.device,
        )

    def __compute_shape_bounds(self):
        """Compute per-shape local-space bounding boxes for meshes and Gaussians."""
        wp.launch(
            kernel=compute_shape_bounds,
            dim=self.shape_source_ptr.size,
            inputs=[
                self.shape_types,
                self.shape_source_ptr,
                self.gaussians_data,
                self.shape_bounds,
            ],
            device=self.device,
        )

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

    def create_color_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.uint32]:
        """Create an output array for color rendering.

        .. deprecated:: 1.1
            Use :meth:`SensorTiledCamera.utils.create_color_image_output`.
        """
        warnings.warn(
            "RenderContext.create_color_image_output is deprecated, use SensorTiledCamera.utils.create_color_image_output instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_color_image_output(width, height, camera_count)

    def create_depth_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.float32]:
        """Create an output array for depth rendering.

        .. deprecated:: 1.1
            Use :meth:`SensorTiledCamera.utils.create_depth_image_output`.
        """
        warnings.warn(
            "RenderContext.create_depth_image_output is deprecated, use SensorTiledCamera.utils.create_depth_image_output instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_depth_image_output(width, height, camera_count)

    def create_shape_index_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.uint32]:
        """Create an output array for shape-index rendering.

        .. deprecated:: 1.1
            Use :meth:`SensorTiledCamera.utils.create_shape_index_image_output`.
        """
        warnings.warn(
            "RenderContext.create_shape_index_image_output is deprecated, use SensorTiledCamera.utils.create_shape_index_image_output instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_shape_index_image_output(width, height, camera_count)

    def create_normal_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.vec3f]:
        """Create an output array for surface-normal rendering.

        .. deprecated:: 1.1
            Use :meth:`SensorTiledCamera.utils.create_normal_image_output`.
        """
        warnings.warn(
            "RenderContext.create_normal_image_output is deprecated, use SensorTiledCamera.utils.create_normal_image_output instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_normal_image_output(width, height, camera_count)

    def create_albedo_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.uint32]:
        """Create an output array for albedo rendering.

        .. deprecated:: 1.1
            Use :meth:`SensorTiledCamera.utils.create_albedo_image_output`.
        """
        warnings.warn(
            "RenderContext.create_albedo_image_output is deprecated, use SensorTiledCamera.utils.create_albedo_image_output instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_albedo_image_output(width, height, camera_count)
