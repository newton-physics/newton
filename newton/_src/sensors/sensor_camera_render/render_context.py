# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import warp as wp

from ...core import Axis
from ...geometry import GeoType, Mesh
from ...geometry.deformable_visual import DeformableVisualMesh, DeformableVisuals, _DeformableVisualSkinBatches
from ...sim import Model, State
from ...utils import load_texture, normalize_texture
from .render import create_kernel
from .types import (
    AntiAliasing,
    ClearData,
    DeformableSurfaceMode,
    LightType,
    MeshData,
    RayGenerationMode,
    RenderConfig,
    RenderOrder,
    TextureData,
)

# Default hemispheric ambient radiance (linear RGB) for surfaces facing up and down.
DEFAULT_AMBIENT_SKY_COLOR = (0.2, 0.2, 0.225)
DEFAULT_AMBIENT_GROUND_COLOR = (0.05, 0.05, 0.06)


class RenderContext:
    @dataclass(unsafe_hash=True)
    class RenderState:
        """Mutable flags tracking which render outputs are active."""

        num_gaussians: int = 0
        has_particles: bool = False
        has_world_indices: bool = False
        has_triangle_mesh_textures: bool = False
        render_color: bool = False
        render_depth: bool = False
        render_forward_depth: bool = False
        render_shape_index: bool = False
        render_normal: bool = False
        render_albedo: bool = False
        render_hdr_color: bool = False
        render_face_index: bool = False
        render_barycentric: bool = False
        has_view_clip: bool = False
        ray_mode: int = 0
        ray_function: Any = None

    DEFAULT_CLEAR_DATA = ClearData()
    DEFAULT_RENDER_CONFIG = RenderConfig()

    def __init__(
        self,
        model: Model,
        load_textures: bool = True,
        *,
        deformable_surface_mode: DeformableSurfaceMode = DeformableSurfaceMode.BOTH,
    ):
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
            load_textures: Load shape and visual mesh textures from file paths or in-memory images.
            deformable_surface_mode: Which deformable surfaces to upload: the simulation triangles,
                the visual meshes, or both.
        """
        self._model = model
        self._deformable_surface_mode = deformable_surface_mode
        self._render_state = RenderContext.RenderState()

        self._kernel_cache: dict[tuple, wp.Kernel] = {}

        self._triangle_mesh: wp.Mesh | None = None
        self._triangle_mesh_group_roots: wp.array[wp.int32] = wp.full(
            model.world_count + 1, value=-1, dtype=wp.int32, device=self.device
        )

        self._triangle_points: wp.array[wp.vec3f] | None = None
        self._triangle_indices: wp.array[wp.int32] | None = None
        self._topology_particle_mask: wp.array[wp.bool] | None = None

        # Skinned meshes need a combined point buffer: simulation particles first,
        # then visual vertices. Without skins, retain the State's particle buffer.
        self._triangle_particle_start = 0
        self._triangle_particle_count = 0
        self._skinned_vertex_count = 0
        self._skinned_points: wp.array[wp.vec3f] | None = None
        self._triangle_groups: wp.array[wp.int32] | None = None
        self._triangle_mesh_uvs = wp.empty(0, dtype=wp.vec2f, device=self.device)
        self._triangle_mesh_texture_ids = wp.empty(0, dtype=wp.int32, device=self.device)

        self._has_particles: bool = False

        self._shape_texture_ids: wp.array[wp.int32] | None = None
        self._shape_mesh_data_ids: wp.array[wp.int32] | None = None
        self._shape_render_type: wp.array[wp.int32] | None = None
        self._empty_triangle_colors: wp.array[wp.vec3f] | None = None

        self._mesh_data: wp.array[MeshData] | None = None
        self._texture_data: wp.array[TextureData] | None = None
        self._texture_data_source: list[TextureData] = []
        self._checkerboard_texture_id: int | None = None
        self._mesh_data_source: list[MeshData] = []

        self._lights_active: wp.array[wp.bool] | None = None
        self._lights_type: wp.array[wp.int32] | None = None
        self._lights_cast_shadow: wp.array[wp.bool] | None = None
        self._lights_position: wp.array[wp.vec3f] | None = None
        self._lights_orientation: wp.array[wp.vec3f] | None = None
        self._lights_color: wp.array[wp.vec3f] | None = None

        self._ambient_sky_color = wp.vec3f(*DEFAULT_AMBIENT_SKY_COLOR)
        self._ambient_ground_color = wp.vec3f(*DEFAULT_AMBIENT_GROUND_COLOR)
        self._ambient_lighting: wp.array[wp.vec3f] | None = None

        # Heightfields and convex hulls are triangle meshes (their wp.Mesh lives
        # in shape_source_ptr), so the renderer treats them as meshes: it reuses
        # the MESH ray-intersection path, which keeps them out of the render
        # kernels entirely (no extra shape-type branch, so no register/occupancy
        # cost). The remapped type array is what the render kernel dispatches on;
        # model.shape_type is left untouched for collision and BVH bounds.
        if model.shape_type is not None:
            shape_type_np = model.shape_type.numpy()
            as_mesh = np.isin(shape_type_np, (int(GeoType.HFIELD), int(GeoType.CONVEX_MESH)))
            if np.any(as_mesh):
                shape_type_np = shape_type_np.copy()
                shape_type_np[as_mesh] = int(GeoType.MESH)
                self._shape_render_type = wp.array(shape_type_np, dtype=wp.int32, device=model.shape_type.device)

        # The batches name the meshes this camera uploads and skins; they are the one source for
        # the skin topology, texture ids and per-frame skinning.
        self._skin_batches: _DeformableVisualSkinBatches = model._get_deformable_visual_skin_batches()
        deformable_visual_meshes = (
            self._skin_batches.meshes if deformable_surface_mode != DeformableSurfaceMode.SIMULATION_ONLY else ()
        )
        has_simulation_triangles = bool(
            deformable_surface_mode != DeformableSurfaceMode.VISUAL_ONLY and model.particle_count and model.tri_count
        )
        if has_simulation_triangles:
            self._triangle_particle_count = model.particle_count
            if deformable_visual_meshes:
                # Copy only the triangle span into the combined buffer. Interleaved
                # non-triangle particles inside that span are copied too.
                tri_particles = model.tri_indices.numpy()
                self._triangle_particle_start = int(tri_particles.min())
                self._triangle_particle_count = int(tri_particles.max()) + 1 - self._triangle_particle_start
        if model.particle_q is not None and model.particle_q.shape[0]:
            self._has_particles = True
            self._render_state.has_particles = True
            topology_particle_mask = np.zeros(model.particle_q.shape[0], dtype=bool)

            def mask_topology_particles(indices: wp.array[wp.int32] | None):
                if indices is not None and indices.shape[0]:
                    topology_particle_mask[indices.numpy().reshape(-1)] = True

            if model.tri_count:
                # Hiding the coarse surface must not expose its particles as spheres.
                mask_topology_particles(model.tri_indices)
                mask_topology_particles(model.tet_indices)
            self._topology_particle_mask = wp.array(
                topology_particle_mask, dtype=wp.bool, device=model.particle_q.device
            )

        if model.gaussians_data is not None:
            self._render_state.num_gaussians = model.gaussians_data.shape[0]

        skin_texture_ids = self._load_texture_and_mesh_data(model, load_textures, deformable_visual_meshes)
        if has_simulation_triangles or deformable_visual_meshes:
            self._init_triangle_mesh(model, deformable_visual_meshes, skin_texture_ids)
            if self._skinned_vertex_count:
                self._skinned_points = self._triangle_points[self._triangle_particle_count :]

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

    def _get_triangle_colors(self) -> wp.array[wp.vec3f]:
        """Per-face display colors (sRGB) of the simulation triangles; empty without ``Model.tri_color``.

        Simulation faces come first in the combined index buffer and the kernel shades faces
        beyond this array white, so visual-mesh faces need no entries. With the simulation
        surface hidden, the first faces are visual meshes and must not take its colors.
        """
        colors = self.model.tri_color if self._triangle_particle_count else None
        if colors is not None and colors.shape[0] == self.model.tri_count:
            return colors
        if self._empty_triangle_colors is None:
            self._empty_triangle_colors = wp.zeros(0, dtype=wp.vec3f, device=self.device)
        return self._empty_triangle_colors

    def _get_shape_render_type(self) -> wp.array[wp.int32] | None:
        if self._shape_render_type is not None:
            return self._shape_render_type
        return self.model.shape_type

    def update(self, state: State, deformable_visuals: DeformableVisuals | None = None):
        """Synchronize triangle-mesh points from the current simulation state.

        Shape and particle BVHs are built by :meth:`~newton.ModelBuilder.finalize`
        and refit separately via :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles`.

        Args:
            state: Current simulation state with particle positions and body transforms.
            deformable_visuals: Precomputed visual points, or ``None`` to skin from the state.

        Raises:
            ValueError: If visual points are supplied to a context that leaves the visual
                meshes out, so a caller cannot assume they were rendered.
        """
        if deformable_visuals is not None and self._deformable_surface_mode == DeformableSurfaceMode.SIMULATION_ONLY:
            raise ValueError(
                "deformable_visuals were supplied, but this camera was constructed with "
                "DeformableSurfaceMode.SIMULATION_ONLY and does not render visual meshes"
            )

        if self._has_triangle_mesh:
            model = self.model
            if not self._skinned_vertex_count:
                self._set_triangle_points(state.particle_q)
            elif self._triangle_particle_count:
                needed = self._triangle_particle_start + self._triangle_particle_count
                if state.particle_q is None or state.particle_q.shape[0] < needed:
                    raise ValueError(
                        f"State.particle_q with at least {needed} entries is required by this model's "
                        "simulation triangles"
                    )
                wp.copy(
                    self._triangle_points,
                    state.particle_q,
                    src_offset=self._triangle_particle_start,
                    count=self._triangle_particle_count,
                )
            if self._skinned_vertex_count:
                if deformable_visuals is None:
                    self._skin_batches.validate_state(state, model)
                    # The camera uses face normals, so the direct path only evaluates points.
                    self._skin_batches.skin(state, model, self._skinned_points)
                else:
                    wp.copy(
                        self._triangle_points,
                        deformable_visuals.points,
                        dest_offset=self._triangle_particle_count,
                        count=self._skinned_vertex_count,
                    )
            self._sync_triangle_mesh()

    def _up_vector(self) -> wp.vec3f:
        up = [0.0, 0.0, 0.0]
        up[int(self.up_axis)] = 1.0
        return wp.vec3f(*up)

    def _get_ambient_lighting(self) -> wp.array[wp.vec3f]:
        if self._ambient_lighting is None:
            self._ambient_lighting = wp.array(
                [self._ambient_sky_color, self._ambient_ground_color, self._up_vector()],
                dtype=wp.vec3f,
                device=self.device,
            )
        return self._ambient_lighting

    def set_ambient_light(
        self,
        sky_color: Sequence[float] | wp.vec3f,
        ground_color: Sequence[float] | wp.vec3f | None = None,
    ) -> None:
        """Set the hemispheric ambient light.

        Surfaces facing the model's up axis receive *sky_color*, surfaces facing
        away receive *ground_color*, and other orientations blend between the
        two. The colors are linear RGB radiance and are passed to the render
        kernel at runtime, so changing them does not recompile it.

        Args:
            sky_color: Linear RGB ambient radiance from above.
            ground_color: Linear RGB ambient radiance from below. If ``None``,
                *sky_color* is used, which gives uniform ambient light.
        """
        sky = wp.vec3f(*(float(c) for c in sky_color))
        ground = sky if ground_color is None else wp.vec3f(*(float(c) for c in ground_color))
        self._ambient_sky_color = sky
        self._ambient_ground_color = ground
        if self._ambient_lighting is not None:
            self._ambient_lighting.assign(
                np.array([list(sky), list(ground), list(self._up_vector())], dtype=np.float32)
            )

    def create_default_light(
        self,
        enable_shadows: bool = True,
        direction: wp.vec3f | None = None,
        color: Sequence[float] | wp.vec3f | None = None,
    ) -> None:
        """Create a default directional light.

        Args:
            enable_shadows: Enable shadow casting for this light.
            direction: Normalized light direction. If ``None``, defaults to
                normalized ``(-1, 1, -1)`` for a Z-up model, rotated to the
                model's up axis otherwise, so the light always points down.
            color: Linear RGB radiance of the light. If ``None``, defaults to
                white ``(1, 1, 1)``.
        """
        if direction is None:
            direction = wp.vec3f(-0.57735026, 0.57735026, -0.57735026)
            if self.up_axis != Axis.Z:
                rot = Axis.Z.quat_between_axes(self.up_axis)
                direction = wp.vec3f(wp.quat_rotate(rot, wp.vec3(direction)))
        light_color = wp.vec3f(1.0) if color is None else wp.vec3f(*(float(c) for c in color))
        self._lights_active = wp.array([True], dtype=wp.bool, device=self.device)
        self._lights_type = wp.array([LightType.DIRECTIONAL], dtype=wp.int32, device=self.device)
        self._lights_cast_shadow = wp.array([enable_shadows], dtype=wp.bool, device=self.device)
        self._lights_position = wp.array([wp.vec3f(0.0)], dtype=wp.vec3f, device=self.device)
        self._lights_orientation = wp.array([direction], dtype=wp.vec3f, device=self.device)
        self._lights_color = wp.array([light_color], dtype=wp.vec3f, device=self.device)

    def assign_checkerboard_material(
        self,
        *,
        shape_indices: Sequence[int] | np.ndarray,
        resolution: int = 64,
        checker_size: int = 32,
    ) -> None:
        """Assign a gray checkerboard texture material to selected shapes.

        Each call sets the whole selection: the selected shapes show the
        checkerboard and every other shape is untextured. Skinned visual
        meshes keep their textures, and the textures loaded at construction
        stay resident so every texture id stays valid.

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
        if self._checkerboard_texture_id is None:
            self._checkerboard_texture_id = len(self._texture_data_source)
        texture_ids[shape_indices] = self._checkerboard_texture_id

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

        # The checkerboard owns one stable slot, so shape and skinned-mesh texture ids stay valid.
        if self._checkerboard_texture_id == len(self._texture_data_source):
            self._texture_data_source.append(checkerboard_data)
        else:
            self._texture_data_source[self._checkerboard_texture_id] = checkerboard_data
        self._texture_data = wp.array(self._texture_data_source, dtype=TextureData, device=self.device)
        self._shape_texture_ids = wp.array(texture_ids, dtype=wp.int32, device=self.device)

    def render(
        self,
        state: State,
        *,
        camera_transforms: wp.array[wp.transformf],
        camera_rays: wp.array4d[wp.vec3f],
        world_indices: wp.array[wp.int32] | None = None,
        color_image: wp.array3d[wp.uint32] | None = None,
        hdr_color_image: wp.array3d[wp.vec3f] | None = None,
        depth_image: wp.array3d[wp.float32] | None = None,
        forward_depth_image: wp.array3d[wp.float32] | None = None,
        shape_index_image: wp.array3d[wp.uint32] | None = None,
        normal_image: wp.array3d[wp.vec3f] | None = None,
        albedo_image: wp.array3d[wp.uint32] | None = None,
        clear_data: ClearData | None = DEFAULT_CLEAR_DATA,
        config: RenderConfig | None = DEFAULT_RENDER_CONFIG,
        face_index_image: wp.array3d[wp.int32] | None = None,
        barycentric_image: wp.array3d[wp.vec2f] | None = None,
        image_size: tuple[int, int] | None = None,
        sample_count: int | None = None,
        view_ray_bundle: wp.array[wp.int32] | None = None,
        view_pinhole: wp.array[wp.vec4f] | None = None,
        view_clip: wp.array[wp.vec2f] | None = None,
        view_frame: wp.array[wp.int32] | None = None,
        ray_function: Any = None,
        ray_params: wp.array2d[wp.float32] | None = None,
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
                ``(height, width, sample_count, 2)``.
            world_indices: Optional per-view world selector, shape
                ``(view_count,)``, dtype ``int32``. A non-negative entry is the
                model world index rendered for that view; a negative entry
                disables the view using a
                :class:`~newton.sensors.SensorCamera.WorldRenderFlag` sentinel
                (``DISABLE_PRESERVE`` / ``DISABLE_CLEAR``). World indices must be
                in ``[0, world_count)``. If ``None``, each view renders its own
                world (``world_index == view_index``), which requires
                ``view_count <= world_count``.
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
                default :class:`RenderConfig` settings.
        """
        model = self.model
        if config is None:
            config = RenderContext.DEFAULT_RENDER_CONFIG
        try:
            anti_aliasing = AntiAliasing(config.anti_aliasing)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid anti_aliasing mode: {config.anti_aliasing!r}") from exc
        if ray_function is not None:
            ray_mode = RayGenerationMode.FUNCTION
        elif view_pinhole is not None:
            ray_mode = RayGenerationMode.PINHOLE
        elif view_ray_bundle is not None:
            ray_mode = RayGenerationMode.BUNDLE_INDEXED
        else:
            ray_mode = RayGenerationMode.BUNDLE
        if sample_count is None:
            if camera_rays is None:
                raise ValueError("sample_count is required when camera_rays is not given")
            sample_count = camera_rays.shape[2]
        if sample_count > 1 and anti_aliasing == AntiAliasing.NONE:
            raise ValueError("anti_aliasing must be SSAA or MSAA for camera_rays with more than one sample")

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
            if image_size is not None:
                width, height = int(image_size[0]), int(image_size[1])
            else:
                height = camera_rays.shape[0]
                width = camera_rays.shape[1]

            if clear_data is None:
                clear_data = RenderContext.DEFAULT_CLEAR_DATA

            self._render_state.has_world_indices = world_indices is not None
            self._render_state.render_color = color_image is not None
            self._render_state.render_depth = depth_image is not None
            self._render_state.render_forward_depth = forward_depth_image is not None
            self._render_state.render_shape_index = shape_index_image is not None
            self._render_state.render_normal = normal_image is not None
            self._render_state.render_albedo = albedo_image is not None
            self._render_state.render_hdr_color = hdr_color_image is not None
            self._render_state.render_face_index = face_index_image is not None
            self._render_state.render_barycentric = barycentric_image is not None
            self._render_state.has_view_clip = view_clip is not None
            self._render_state.ray_mode = int(ray_mode)
            self._render_state.ray_function = ray_function

            # One view per camera transform; independent of the model world count.
            view_count = camera_transforms.shape[0]

            # Validate caller-owned arrays explicitly rather than with ``assert``:
            # asserts are stripped under ``python -O``, which would let a bad shape,
            # dtype, or device reach the kernel as an out-of-bounds device access.
            expected = (view_count, height, width)
            render_device = wp.get_device(self.device)

            def _check_output(name: str, image):
                if image is not None and image.shape != expected:
                    raise ValueError(f"{name} shape must be {expected}, got {tuple(image.shape)}")

            if ray_mode == RayGenerationMode.BUNDLE:
                if camera_rays.ndim != 4 or camera_rays.shape[2] <= 0 or camera_rays.shape[3] != 2:
                    raise ValueError(
                        f"camera_rays shape must be ({height}, {width}, sample_count, 2), got {tuple(camera_rays.shape)}"
                    )
            elif ray_mode == RayGenerationMode.BUNDLE_INDEXED:
                if (
                    camera_rays.ndim != 4
                    or camera_rays.shape[0] % height != 0
                    or camera_rays.shape[1] != width
                    or camera_rays.shape[2] != sample_count
                    or camera_rays.shape[3] != 2
                ):
                    raise ValueError(
                        f"stacked camera_rays shape must be (bundle_count * {height}, {width}, {sample_count}, 2), "
                        f"got {tuple(camera_rays.shape)}"
                    )
            # ``world_indices`` is optional: when ``None`` each view renders its own
            # world (``world_index == view_index``); otherwise validate the mapping.
            if world_indices is not None:
                if world_indices.shape != (view_count,):
                    raise ValueError(f"world_indices shape must be ({view_count},), got {tuple(world_indices.shape)}")
                if world_indices.dtype != wp.int32:
                    raise ValueError(f"world_indices dtype must be int32, got {world_indices.dtype}")
                if world_indices.device != render_device:
                    raise ValueError(f"world_indices device must be {render_device}, got {world_indices.device}")

            _check_output("color_image", color_image)
            _check_output("depth_image", depth_image)
            _check_output("forward_depth_image", forward_depth_image)
            _check_output("shape_index_image", shape_index_image)
            _check_output("normal_image", normal_image)
            _check_output("albedo_image", albedo_image)
            _check_output("hdr_color_image", hdr_color_image)
            _check_output("face_index_image", face_index_image)
            _check_output("barycentric_image", barycentric_image)

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
            if face_index_image is not None:
                face_index_image = face_index_image.reshape(total_pixels)
            if barycentric_image is not None:
                barycentric_image = barycentric_image.reshape(total_pixels)

            # Key the cache on the value tuple itself (dict hashes AND compares by
            # equality), so two configs that merely share a hash cannot collide onto
            # one kernel. ``block_dim`` is a launch-only parameter that does not
            # affect codegen, so normalize it out of the key: configs differing only
            # in block_dim must reuse the same compiled kernel. ``replace`` also
            # snapshots the mutable config so a later mutation cannot alter a stored
            # key (state/clear_data are snapshotted the same way on insert).
            key_config = replace(config, block_dim=0)
            render_kernel = self._kernel_cache.get((key_config, self._render_state, clear_data))
            if render_kernel is None:
                render_kernel = create_kernel(config, self._render_state, clear_data)
                self._kernel_cache[(key_config, replace(self._render_state), replace(clear_data))] = render_kernel

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
                    model.world_count,
                    self.light_count,
                    width,
                    height,
                    # Camera
                    camera_rays,
                    camera_transforms,
                    world_indices,
                    int(sample_count),
                    view_ray_bundle,
                    view_pinhole,
                    view_clip,
                    view_frame,
                    ray_params,
                    # Shape BVH
                    model.bvh_shape_count_enabled,
                    model.bvh_shapes.id if model.bvh_shapes is not None else 0,
                    model.bvh_shapes_group_roots,
                    # Shapes
                    model.bvh_shape_enabled,
                    self._get_shape_render_type(),  # HFIELD and CONVEX_MESH remapped to MESH
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
                    self._get_triangle_colors(),
                    self._triangle_mesh_uvs,
                    self._triangle_mesh_texture_ids,
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
                    self._lights_color,
                    self._get_ambient_lighting(),
                    # Outputs
                    color_image,
                    depth_image,
                    forward_depth_image,
                    shape_index_image,
                    normal_image,
                    albedo_image,
                    hdr_color_image,
                    face_index_image,
                    barycentric_image,
                ],
                device=self.device,
                block_dim=config.block_dim,
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
            triangle_groups = self._triangle_groups
            if triangle_groups is None:
                triangle_indices_np = self._triangle_indices.reshape((-1, 3)).numpy()
                particle_world_np = self.model.particle_world.numpy()
                triangle_world_np = particle_world_np[triangle_indices_np[:, 0]]
                triangle_groups_np = np.where(triangle_world_np < 0, self.world_count, triangle_world_np).astype(
                    np.int32
                )
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

    def _load_texture_and_mesh_data(
        self, model: Model, load_textures: bool, deformable_visual_meshes: Sequence[DeformableVisualMesh]
    ) -> list[int]:
        """Load mesh UV/normal data and textures from *model*.

        Populates mesh data, texture data, and the
        per-shape texture/mesh-data index arrays. Textures and mesh
        data are deduplicated by hash/identity.

        Args:
            model: Newton simulation model containing shape sources.
            load_textures: Load image textures from paths or in-memory images.
                Otherwise leave shapes and visual meshes untextured.
            deformable_visual_meshes: The visual meshes this camera renders, whose
                textures are loaded like shape textures.

        Returns:
            The texture slot of each visual mesh, in the given order, or ``-1`` for an
            untextured mesh.
        """
        self._mesh_data_source = []
        self._texture_data_source = []

        texture_hashes = {}
        mesh_hashes = {}

        mesh_data_ids = []
        texture_data_ids = []

        def texture_id(texture, texture_hash):
            if texture is None or not load_textures:
                return -1
            if texture_hash not in texture_hashes:
                pixels = load_texture(texture)
                if pixels is None:
                    raise ValueError(f"Failed to load texture: {texture}")
                pixels = normalize_texture(pixels, require_channels=True)
                if pixels.shape[2] == 3:
                    # RGB and grayscale sources need an opaque alpha channel.
                    alpha = np.full((*pixels.shape[:2], 1), 255, dtype=np.uint8)
                    pixels = np.concatenate((pixels, alpha), axis=2)
                texture_hashes[texture_hash] = len(self._texture_data_source)
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
            return texture_hashes[texture_hash]

        shape_types = model.shape_type.numpy() if model.shape_type is not None else ()
        for shape, shape_type in zip(model.shape_source, shape_types, strict=True):
            if isinstance(shape, Mesh):
                texture_data_ids.append(texture_id(shape.texture, shape.texture_hash))

                # A convex hull renders its collision mesh, whose vertices may be deduplicated, so
                # the per-vertex UVs and normals of its source mesh do not match the hit faces.
                if shape_type != GeoType.CONVEX_MESH and (shape.uvs is not None or shape.normals is not None):
                    if shape not in mesh_hashes:
                        mesh_hashes[shape] = len(self._mesh_data_source)

                        data = MeshData()
                        if shape.uvs is not None:
                            # Apply the mesh's authored tiling, offset, and rotation, as the viewers do.
                            transform = np.asarray(shape.texture_transform, dtype=np.float32)
                            uvs = np.asarray(shape.uvs, dtype=np.float32) @ transform[:, :2].T + transform[:, 2]
                            data.uvs = wp.array(uvs, dtype=wp.vec2f, device=self.device)
                        if shape.normals is not None:
                            data.normals = wp.array(shape.normals, dtype=wp.vec3f, device=self.device)
                        self._mesh_data_source.append(data)

                    mesh_data_ids.append(mesh_hashes[shape])
                else:
                    mesh_data_ids.append(-1)
            else:
                texture_data_ids.append(-1)
                mesh_data_ids.append(-1)

        # Sampled image hashes can miss changed channels. Keep skin keys separate
        # from rigid-mesh hashes and share uploads only for the same source.
        skin_texture_ids = [
            texture_id(mesh.texture, ("skin", mesh.texture if isinstance(mesh.texture, str) else id(mesh.texture)))
            if mesh.uvs is not None
            else -1
            for mesh in deformable_visual_meshes
        ]
        self._render_state.has_triangle_mesh_textures = any(i >= 0 for i in skin_texture_ids)

        self._texture_data = wp.array(self._texture_data_source, dtype=TextureData, device=self.device)
        self._shape_texture_ids = wp.array(
            np.asarray(texture_data_ids, dtype=np.int32), dtype=wp.int32, device=self.device
        )

        self._mesh_data = wp.array(self._mesh_data_source, dtype=MeshData, device=self.device)
        self._shape_mesh_data_ids = wp.array(mesh_data_ids, dtype=wp.int32, device=self.device)
        return skin_texture_ids

    def _init_triangle_mesh(
        self, model: Model, deformable_visual_meshes: Sequence[DeformableVisualMesh], skin_texture_ids: list[int]
    ):
        """Initialize simulation triangles and optional skinned surfaces.

        Only skins need a camera-owned buffer. It holds the particle span the simulation
        triangles reference, then visual vertices in model order for batched skinning.
        """
        if not deformable_visual_meshes:
            self._set_triangle_points(model.particle_q)
            self._set_triangle_indices(model.tri_indices.flatten())
            return

        triangle_indices: list[np.ndarray] = []
        triangle_uvs: list[np.ndarray] = []
        triangle_texture_ids: list[np.ndarray] = []
        vertex_worlds: list[np.ndarray] = []

        # UVs are only read back and uploaded when some visual mesh has a texture to sample.
        textured = self._render_state.has_triangle_mesh_textures
        if self._triangle_particle_count:
            start, count = self._triangle_particle_start, self._triangle_particle_count
            sim_indices = model.tri_indices.numpy().astype(np.int32, copy=False) - start
            triangle_indices.append(sim_indices.reshape(-1))
            if textured:
                triangle_uvs.append(np.zeros((count, 2), dtype=np.float32))
            triangle_texture_ids.append(np.full(sim_indices.shape[0], -1, dtype=np.int32))
            vertex_worlds.append(model.particle_world.numpy()[start : start + count].astype(np.int32, copy=False))

        # The batches already hold every mesh's triangles indexed by destination vertex; those
        # vertices follow the simulation particles in the camera's buffer.
        triangle_indices.append(self._skin_batches.face_indices.numpy() + self._triangle_particle_count)
        for mesh_index, mesh in enumerate(deformable_visual_meshes):
            start, end = self._skin_batches.mesh_ranges[mesh_index]
            if textured:
                if mesh.uvs is None:
                    triangle_uvs.append(np.zeros((mesh.vertex_count, 2), dtype=np.float32))
                else:
                    triangle_uvs.append(mesh.uvs.numpy().astype(np.float32, copy=False).reshape(-1, 2))
            face_count = mesh.indices.shape[0] // 3
            triangle_texture_ids.append(np.full(face_count, skin_texture_ids[mesh_index], dtype=np.int32))
            vertex_worlds.append(np.full(end - start, mesh.world, dtype=np.int32))

        self._skinned_vertex_count = self._skin_batches.vertex_count
        self._triangle_points = wp.empty(
            self._triangle_particle_count + self._skinned_vertex_count, dtype=wp.vec3f, device=self.device
        )
        indices_np = np.concatenate(triangle_indices)
        self._triangle_indices = wp.array(indices_np, dtype=wp.int32, device=self.device)
        if textured:
            self._triangle_mesh_uvs = wp.array(np.concatenate(triangle_uvs), dtype=wp.vec2f, device=self.device)
            self._triangle_mesh_texture_ids = wp.array(
                np.concatenate(triangle_texture_ids), dtype=wp.int32, device=self.device
            )
        # Each face belongs to its first vertex's world; global faces (-1) share the extra group.
        face_world_np = np.concatenate(vertex_worlds)[indices_np.reshape(-1, 3)[:, 0]]
        self._triangle_groups = wp.array(
            np.where(face_world_np < 0, self.world_count, face_world_np).astype(np.int32),
            dtype=wp.int32,
            device=self.device,
        )
