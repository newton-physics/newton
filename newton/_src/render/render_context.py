# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import numpy as np
import warp as wp

from ..core import Axis
from ..geometry import GeoType, Mesh
from ..geometry.deformable_visual import DeformableVisualMesh, DeformableVisuals, _DeformableVisualSkinBatches
from ..sim import Model, State
from ..utils import load_texture, normalize_texture
from .render import create_kernel
from .scene import SceneItems, WarpScene, model_shape_scene_data, standalone_particles
from .types import (
    AntiAliasing,
    ClearData,
    DeformableSurfaceMode,
    LightType,
    MeshData,
    RenderConfig,
    RenderOrder,
    RenderState,
    SceneData,
    TextureData,
)

# Default hemispheric ambient radiance (linear RGB) for surfaces facing up and down.
DEFAULT_AMBIENT_SKY_COLOR = (0.2, 0.2, 0.225)
DEFAULT_AMBIENT_GROUND_COLOR = (0.05, 0.05, 0.06)


@wp.kernel(enable_backward=False)
def _gather_variant_render_data(
    row_shapes: wp.array[wp.int32],
    shape_variant_shape: wp.array[wp.int32],
    variant_render_type: wp.array[wp.int32],
    variant_texture_ids: wp.array[wp.int32],
    variant_mesh_data_ids: wp.array[wp.int32],
    shape_texture_override: wp.array[wp.int32],
    # outputs
    shape_render_type: wp.array[wp.int32],
    shape_texture_ids: wp.array[wp.int32],
    shape_mesh_data_ids: wp.array[wp.int32],
):
    shape = row_shapes[wp.tid()]
    variant_shape = shape_variant_shape[shape]
    if variant_shape >= 0:
        shape_render_type[shape] = variant_render_type[variant_shape]
        shape_mesh_data_ids[shape] = variant_mesh_data_ids[variant_shape]
        if shape_texture_override.shape[0] > 0:
            shape_texture_ids[shape] = shape_texture_override[shape]
        else:
            shape_texture_ids[shape] = variant_texture_ids[variant_shape]


def _render_types(shape_types: np.ndarray) -> np.ndarray:
    """Shape types as rendered: heightfields and convex hulls are triangle meshes."""
    as_mesh = np.isin(shape_types, (int(GeoType.HFIELD), int(GeoType.CONVEX_MESH)))
    return np.where(as_mesh, int(GeoType.MESH), shape_types).astype(np.int32)


class RenderContext:
    DEFAULT_CLEAR_DATA = ClearData()
    DEFAULT_RENDER_CONFIG = RenderConfig()

    def __init__(
        self,
        model: Model,
        load_textures: bool = True,
        *,
        backend: Literal["warp", "optix"] = "warp",
        deformable_surface_mode: DeformableSurfaceMode = DeformableSurfaceMode.BOTH,
    ):
        """Create a render context for a Newton simulation model.

        Populates shape, triangle, and texture data from *model*. The context traces
        one scene of shapes, standalone particles and deformable triangles; :meth:`update`
        refits it from the state and from
        the model's shapes, which :meth:`~newton.Model.bvh_refit_shapes` refits.

        Args:
            model: Newton simulation model providing shapes and particles.
            load_textures: Load shape and visual mesh textures from file paths or in-memory images.
            backend: ``"warp"`` or ``"optix"``.
            deformable_surface_mode: Which deformable surfaces to upload: the simulation triangles,
                the visual meshes, or both.
        """
        self._model = model
        self._deformable_surface_mode = deformable_surface_mode
        self._render_state = RenderState()

        self._kernel_cache: dict[tuple, wp.Kernel] = {}

        self._triangle_points: wp.array[wp.vec3f] | None = None
        self._triangle_indices: wp.array[wp.int32] | None = None

        # Skinned meshes need a combined point buffer: simulation particles first,
        # then visual vertices. Without skins, retain the State's particle buffer.
        self._triangle_particle_start = 0
        self._triangle_particle_count = 0
        self._skinned_vertex_count = 0
        self._skinned_points: wp.array[wp.vec3f] | None = None
        self._triangle_point_worlds: wp.array[wp.int32] | None = None
        self._triangle_mesh_uvs = wp.empty(0, dtype=wp.vec2f, device=self.device)
        self._triangle_mesh_texture_ids = wp.empty(0, dtype=wp.int32, device=self.device)

        self._scene: WarpScene | OptixScene | None = None

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
        # Variant rows render the variant shape they hold; their per-shape data is gathered before each render.
        self._variant_rows = model._variant_row_shapes if model._has_variant_rows else None
        self._shape_texture_override: wp.array[wp.int32] | None = None
        if model.shape_type is not None:
            shape_type_np = model.shape_type.numpy()
            if self._variant_rows is not None or np.any(_render_types(shape_type_np) != shape_type_np):
                self._shape_render_type = wp.array(
                    _render_types(shape_type_np), dtype=wp.int32, device=model.shape_type.device
                )

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
        topology_particle_mask = None
        if model.particle_q is not None and model.particle_q.shape[0]:
            topology_particle_mask = np.zeros(model.particle_q.shape[0], dtype=bool)

            def mask_topology_particles(indices: wp.array[wp.int32] | None):
                if indices is not None and indices.shape[0]:
                    topology_particle_mask[indices.numpy().reshape(-1)] = True

            if model.tri_count:
                # Hiding the coarse surface must not expose its particles as spheres.
                mask_topology_particles(model.tri_indices)
                mask_topology_particles(model.tet_indices)

        if model.shape_count:
            gaussian_worlds = model.shape_world.numpy()[model.shape_type.numpy() == GeoType.GAUSSIAN]
            local = np.bincount(gaussian_worlds[gaussian_worlds >= 0], minlength=1).max()
            self._render_state.gaussian_shapes_per_ray = int(local + np.count_nonzero(gaussian_worlds < 0))

        skin_texture_ids = self._load_texture_and_mesh_data(model, load_textures, deformable_visual_meshes)
        if has_simulation_triangles or deformable_visual_meshes:
            self._init_triangle_mesh(model, deformable_visual_meshes, skin_texture_ids)
            if self._skinned_vertex_count:
                self._skinned_points = self._triangle_points[self._triangle_particle_count :]

        # Every non-shape item the camera traverses alongside the shapes.
        item_particles = standalone_particles(model, topology_particle_mask)
        triangle_worlds = None
        if self._has_triangle_mesh:
            point_worlds = self._triangle_point_worlds
            if point_worlds is None:
                point_worlds = model.particle_world
            first_points = self._triangle_indices.numpy()[0::3]
            triangle_worlds = wp.array(point_worlds.numpy()[first_points], dtype=wp.int32, device=self.device)
            self._triangle_point_worlds = point_worlds
        self._scene_items = SceneItems(
            item_particles=wp.array(item_particles, dtype=wp.int32, device=self.device),
            triangle_points=self._triangle_points if self._has_triangle_mesh else None,
            triangle_indices=self._triangle_indices if self._has_triangle_mesh else None,
            triangle_point_worlds=self._triangle_point_worlds,
            triangle_worlds=triangle_worlds,
        )
        self._render_state.has_particles = self._scene_items.particle_count > 0
        self._render_state.has_triangle_mesh = self._scene_items.triangle_count > 0

        if backend not in ("warp", "optix"):
            raise ValueError(f"backend must be 'warp' or 'optix', got {backend!r}")
        if backend == "optix":
            try:
                from ..geometry import optix  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError("The OptiX camera backend requires the newton[optix] extra") from exc
            optix.check_supported(model)
        self.backend = backend

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

    def _gather_variant_render_data(self) -> None:
        """Copy the render type, texture, and mesh data of the variant shape each variant row holds."""
        if self._variant_rows is None:
            return
        override = self._shape_texture_override
        wp.launch(
            _gather_variant_render_data,
            dim=self._variant_rows.shape[0],
            inputs=[
                self._variant_rows,
                self.model._shape_variant_shape,
                self._variant_render_type,
                self._variant_texture_ids,
                self._variant_mesh_data_ids,
                wp.empty(0, dtype=wp.int32, device=self.device) if override is None else override,
            ],
            outputs=[self._shape_render_type, self._shape_texture_ids, self._shape_mesh_data_ids],
            device=self.device,
        )

    def _get_shape_render_type(self) -> wp.array[wp.int32] | None:
        if self._shape_render_type is not None:
            return self._shape_render_type
        return self.model.shape_type

    def update(self, state: State, deformable_visuals: DeformableVisuals | None = None):
        """Synchronize deformable points from the current simulation state and refit the camera's scene.

        The scene holds the model's shapes, as of their last :meth:`~newton.Model.bvh_refit_shapes`,
        with the camera's standalone particles and deformable triangles, so refit the shapes first.

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
        self._refit_scene(state)

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
        if self._variant_rows is not None:
            # The variant shapes' texture ids index the replaced texture table; variant rows keep
            # the checkerboard assignment whichever variant they hold.
            self._shape_texture_override = wp.array(texture_ids, dtype=wp.int32, device=self.device)

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
    ):
        """Raytrace the scene into the provided output images.

        Renders one view per camera transform. The number of views is
        ``view_count = camera_transforms.shape[0]`` and is independent of the
        model's world count. At least one output image must be supplied; all
        non-``None`` output arrays must have shape ``(view_count, height, width)``.

        Renders the scene as of the last :meth:`update`, building it from *state* if there
        was none. Refit the model's shapes with :meth:`~newton.Model.bvh_refit_shapes`
        before that update; the OptiX backend traces :attr:`~newton.Model.optix_shapes`, built
        with ``bvh_build_shapes(state, backend="optix")`` and refit with
        ``bvh_refit_shapes(state, backend="optix")``.

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
        if self.backend == "optix":
            if self.device.stream.is_capturing:
                raise RuntimeError("The OptiX camera backend does not support CUDA graph capture; use backend='warp'")
            self._optix_shapes()

        if config is None:
            config = RenderContext.DEFAULT_RENDER_CONFIG
        try:
            anti_aliasing = AntiAliasing(config.anti_aliasing)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid anti_aliasing mode: {config.anti_aliasing!r}") from exc
        if camera_rays.ndim == 4 and camera_rays.shape[2] > 1 and anti_aliasing == AntiAliasing.NONE:
            raise ValueError("anti_aliasing must be SSAA or MSAA for camera_rays with more than one sample")

        if self.backend == "warp" and model.shape_count > 0 and model.bvh_shape_enabled is None:
            raise RuntimeError(
                "Shape BVH is missing. ModelBuilder.finalize() builds it for finalized models; "
                "call model.bvh_build_shapes(state) for manually populated models."
            )
        if (
            self.backend == "warp"
            and model.bvh_shape_count_enabled > 0
            and (model.bvh_shapes is None or model.bvh_shapes_group_roots is None)
        ):
            raise RuntimeError("Shape BVH is incomplete; rebuild it with model.bvh_build_shapes(state).")
        scene = self._scene_data(state)
        optix = self._scene if self.backend == "optix" else None
        has_planes = optix is not None and bool(optix.shapes.planes or optix.shapes.world_planes.shape[1])

        if scene.id or has_planes or self._has_gaussians:
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

            if camera_rays.ndim != 4 or camera_rays.shape[2] <= 0 or camera_rays.shape[3] != 2:
                raise ValueError(
                    f"camera_rays shape must be ({height}, {width}, sample_count, 2), got {tuple(camera_rays.shape)}"
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

            # Key the cache on the value tuple itself (dict hashes AND compares by
            # equality), so two configs that merely share a hash cannot collide onto
            # one kernel. ``block_dim`` is a launch-only parameter that does not
            # affect codegen, so normalize it out of the key: configs differing only
            # in block_dim must reuse the same compiled kernel. ``replace`` also
            # snapshots the mutable config so a later mutation cannot alter a stored
            # key (state/clear_data are snapshotted the same way on insert).
            # OptiX pipelines are cached on the shapes they trace, so a rebuild releases them.
            key_config = replace(config, block_dim=0)
            # OptiX pipelines are cached on the scene they trace, so a rebuild releases them.
            cache = self._kernel_cache if optix is None else optix.pipelines
            render_kernel = cache.get((key_config, self._render_state, clear_data))
            if render_kernel is None:
                # Copies: the factories capture them in wp.static, and shared modules may recompile.
                render_kernel = create_kernel(replace(config), replace(self._render_state), replace(clear_data), optix)
                if optix is not None:
                    render_kernel = optix.create_pipeline(render_kernel, config)
                cache[(key_config, replace(self._render_state), replace(clear_data))] = render_kernel

            pixels_per_view = width * height
            if config.render_order == RenderOrder.TILED:
                tiles_x = (width + config.tile_width - 1) // config.tile_width
                tiles_y = (height + config.tile_height - 1) // config.tile_height
                pixels_per_view = tiles_x * tiles_y * config.tile_width * config.tile_height

            self._gather_variant_render_data()
            if optix is None:
                launch, launch_options = wp.launch, {"device": self.device, "block_dim": config.block_dim}
            else:
                from ..geometry.optix import wo  # noqa: PLC0415

                launch, launch_options = wo.launch, {}
            launch(
                render_kernel,
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
                    scene,
                    # Shapes
                    self._get_shape_render_type(),  # HFIELD and CONVEX_MESH remapped to MESH
                    model.shape_scale,
                    model.shape_color,
                    model.bvh_shape_world_transforms if optix is None else optix.shapes.shape_world_transforms,
                    model.shape_source_ptr,
                    self._shape_texture_ids,
                    self._shape_mesh_data_ids,
                    # Triangle Mesh
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
                ],
                **launch_options,
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
        self._triangle_points = triangle_points
        if getattr(self, "_scene_items", None) is not None:
            self._scene_items.triangle_points = triangle_points

    def _set_triangle_indices(self, triangle_indices: wp.array[wp.int32]) -> None:
        self._triangle_indices = triangle_indices

    @property
    def _has_own_scene(self) -> bool:
        """Whether the camera traverses its own scene, rather than only the model's shapes."""
        items = self._scene_items
        return items.particle_count > 0 or items.triangle_count > 0 or self.backend == "optix"

    def _optix_shapes(self):
        shapes = self.model.optix_shapes
        if shapes is None:
            raise RuntimeError(
                "The OptiX camera backend requires Model.bvh_build_shapes(state, backend='optix') to have been "
                "called first."
            )
        return shapes

    def _refit_scene(self, state: State) -> None:
        """Build the camera's scene on first use, or refit it to *state*."""
        if not self._has_own_scene:
            return
        if self.backend == "optix":
            from .render_optix import OptixScene  # noqa: PLC0415

            if self.device.stream.is_capturing:
                raise RuntimeError("The OptiX camera backend does not support CUDA graph capture; use backend='warp'")
            shapes = self._optix_shapes()
            if self._scene is None or self._scene.shapes is not shapes:
                # Built from the current state, so it needs no refit.
                self._scene = OptixScene(self.model, shapes, self._scene_items, state)
            else:
                self._scene.refit(state)
            return
        if self._scene is None:
            self._scene = WarpScene(self.model, self._scene_items)
        self._scene.refit(state)

    def _scene_data(self, state: State) -> SceneData:
        """The scene to render, built from *state* if the camera has not synced one yet."""
        if not self._has_own_scene:
            return model_shape_scene_data(self.model)
        stale = self._scene is None
        if self.backend == "optix":
            stale = stale or self._scene.shapes is not self._optix_shapes()
        else:
            stale = stale or self._scene._shape_bvh is not self.model.bvh_shapes
        if stale:
            self._refit_scene(state)
        return self._scene.scene_data(state)

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
        # Holding the last image stops glibc from returning the decode buffers to the OS after every texture.
        last_pixels = None

        def texture_id(texture, texture_hash):
            nonlocal last_pixels
            if texture is None or not load_textures:
                return -1
            if texture_hash not in texture_hashes:
                pixels = load_texture(texture)
                if pixels is None:
                    raise ValueError(f"Failed to load texture: {texture}")
                last_pixels = pixels
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

        def ids(sources: Sequence, shape_types: Sequence) -> tuple[list[int], list[int]]:
            """Texture and mesh-data ids of each source, loading data that is not loaded yet."""
            texture_data_ids = []
            mesh_data_ids = []
            for shape, shape_type in zip(sources, shape_types, strict=True):
                if not isinstance(shape, Mesh):
                    texture_data_ids.append(-1)
                    mesh_data_ids.append(-1)
                    continue
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
            return texture_data_ids, mesh_data_ids

        shape_types = model.shape_type.numpy() if model.shape_type is not None else ()
        texture_data_ids, mesh_data_ids = ids(model.shape_source, shape_types)
        if self._variant_rows is not None:
            variant_types = model._variant_shape_geometry["shape_type"].numpy()
            variant_texture_ids, variant_mesh_data_ids = ids(model._variant_shape_source, variant_types)
            self._variant_render_type = wp.array(_render_types(variant_types), dtype=wp.int32, device=self.device)
            self._variant_texture_ids = wp.array(variant_texture_ids, dtype=wp.int32, device=self.device)
            self._variant_mesh_data_ids = wp.array(variant_mesh_data_ids, dtype=wp.int32, device=self.device)

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
        self._triangle_points = wp.zeros(
            self._triangle_particle_count + self._skinned_vertex_count, dtype=wp.vec3f, device=self.device
        )
        indices_np = np.concatenate(triangle_indices)
        self._triangle_indices = wp.array(indices_np, dtype=wp.int32, device=self.device)
        if textured:
            self._triangle_mesh_uvs = wp.array(np.concatenate(triangle_uvs), dtype=wp.vec2f, device=self.device)
            self._triangle_mesh_texture_ids = wp.array(
                np.concatenate(triangle_texture_ids), dtype=wp.int32, device=self.device
            )
        # Each face belongs to its first vertex's world.
        self._triangle_point_worlds = wp.array(np.concatenate(vertex_worlds), dtype=wp.int32, device=self.device)
