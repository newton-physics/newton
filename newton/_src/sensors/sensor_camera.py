# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import math
import os
import re
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ..core.types import Devicelike
from ..geometry.flags import ShapeFlags
from ..sim.camera import CameraLens
from ..utils.selection import match_labels
from .sensor_camera_render import Utils
from .sensor_camera_render.camera_utils import _multisample_offset
from .sensor_camera_render.types import (
    AntiAliasing,
    ClearData,
    GaussianRenderMode,
    RayQuery,
    RenderConfig,
    RenderOrder,
    TextureProjectionMode,
    WorldRenderFlag,
)

if TYPE_CHECKING:
    from ..sim.model import Model
    from ..sim.state import State

# Enable NVTX ranges / timing around SensorCamera.update() when NEWTON_PROFILE is set.
PROFILE_ENABLED = os.environ.get("NEWTON_PROFILE", "0") != "0"


# Render contexts shared by bound sensors of the same model (scene data, textures, lights).
_SHARED_RENDER_CONTEXTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

_HIT_TRIANGLE_MESH = 0xFFFFFFFD
_HIT_PARTICLES = 0xFFFFFFFE
_HIT_NONE = 0xFFFFFFFF


@wp.kernel(enable_backward=False)
def _compute_view_transforms_kernel(
    view_frame: wp.array[wp.int32],
    frame_body: wp.array[wp.int32],
    frame_transform: wp.array[wp.transformf],
    body_q: wp.array[wp.transformf],
    out_transforms: wp.array[wp.transformf],
):
    view = wp.tid()
    frame = view_frame[view]
    body = frame_body[frame]
    transform = frame_transform[frame]
    if body >= 0:
        transform = wp.transform_multiply(body_q[body], transform)
    out_transforms[view] = transform


@wp.kernel(enable_backward=False)
def _gather_view_clip_kernel(
    view_frame: wp.array[wp.int32],
    camera_clipping_range: wp.array[wp.vec2f],
    out_clip: wp.array[wp.vec2f],
):
    view = wp.tid()
    out_clip[view] = camera_clipping_range[view_frame[view]]


@wp.kernel(enable_backward=False)
def _compute_view_pinhole_kernel(
    view_frame: wp.array[wp.int32],
    camera_focal_length: wp.array[wp.float32],
    camera_aperture: wp.array[wp.vec2f],
    camera_aperture_offset: wp.array[wp.vec2f],
    aspect: wp.float32,
    out_pinhole: wp.array[wp.vec4f],
):
    view = wp.tid()
    camera = view_frame[view]
    focal = camera_focal_length[camera]
    aperture = camera_aperture[camera]
    horizontal = aperture[0]
    vertical = aperture[1]
    # A zero aperture component follows the image aspect ratio (square pixels).
    if horizontal <= 0.0:
        horizontal = vertical * aspect
    if vertical <= 0.0:
        vertical = horizontal / aspect
    offset = camera_aperture_offset[camera]
    out_pinhole[view] = wp.vec4f(horizontal / focal, vertical / focal, offset[0] / focal, offset[1] / focal)


@wp.kernel(enable_backward=False)
def _fill_pinhole_bundles_kernel(
    bundle_pinhole: wp.array[wp.vec4f],
    width: wp.int32,
    height: wp.int32,
    sample_count: wp.int32,
    out_rays: wp.array4d[wp.vec3f],
):
    bundle, py, px, sample_index = wp.tid()
    pinhole = bundle_pinhole[bundle]
    offset = _multisample_offset(sample_index, sample_count)
    u = (float(px) + offset[0]) / float(width)
    v = (float(py) + offset[1]) / float(height)
    direction = wp.vec3f((u - 0.5) * pinhole[0] + pinhole[2], (0.5 - v) * pinhole[1] + pinhole[3], -1.0)
    row = bundle * height + py
    out_rays[row, px, sample_index, 0] = wp.vec3f(0.0)
    out_rays[row, px, sample_index, 1] = wp.normalize(direction)


@wp.kernel(enable_backward=False)
def _apply_view_mask_kernel(
    view_world: wp.array[wp.int32],
    view_mask: wp.array[wp.bool],
    out_world_indices: wp.array[wp.int32],
):
    view = wp.tid()
    if view_mask[view]:
        out_world_indices[view] = view_world[view]
    else:
        out_world_indices[view] = wp.static(int(WorldRenderFlag.DISABLE_PRESERVE))


def _validate_camera_ray_output(
    width: int,
    height: int,
    out_rays: wp.array4d[wp.vec3f] | None,
    device: Devicelike = None,
    sample_count: int = 1,
) -> tuple[int, int, int, wp.array4d[wp.vec3f], wp.Device]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    sample_count = int(sample_count)
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")

    expected_shape = (height, width, sample_count, 2)
    target_device = wp.get_device(device) if device is not None else None

    if out_rays is None:
        out_rays = wp.empty(expected_shape, dtype=wp.vec3f, device=device)
    else:
        if not isinstance(out_rays, wp.array):
            raise TypeError(f"out_rays must be a Warp array, got {type(out_rays).__name__}")
        if out_rays.dtype != wp.vec3f:
            raise ValueError(f"out_rays must have dtype vec3f, got {out_rays.dtype}")
        if out_rays.shape != expected_shape:
            raise ValueError(f"out_rays must have shape {expected_shape}, got {out_rays.shape}")
        if target_device is not None and out_rays.device != target_device:
            raise ValueError(f"out_rays is on {out_rays.device}, expected {target_device}")

    return width, height, sample_count, out_rays, out_rays.device


class SensorCamera:
    """Raytraced camera sensor that renders a Newton model.

    A camera sensor owns an internal renderer built for a model plus the default
    render settings (:attr:`default_clear_data`, :attr:`default_render_config`)
    used when :meth:`update` is not given per-call overrides. The caller supplies
    the camera-space rays, the world-space per-view camera transforms, and the
    output image buffers to :meth:`update`; the number of views is inferred from
    the leading dimension of the camera transforms.

    Camera frame convention: each camera looks along its local ``-Z`` axis, with
    ``+Y`` up and ``+X`` right (the USD/OpenGL convention). The per-view
    ``camera_transforms`` place that camera frame in world space, and the
    ``camera_rays`` bundle stores per-pixel origins in ``[..., 0]`` and
    directions in ``[..., 1]``, both expressed in camera space.

    Deformable triangle surfaces use :attr:`Model.tri_color` as sRGB albedo,
    like shape colors, and fall back to white when triangle colors are
    unavailable. Set :attr:`RenderConfig.enable_backface_culling` to ``False``
    to render both sides with camera-facing normals.

    The render configuration types are exposed as nested attributes (e.g.
    ``SensorCamera.RenderConfig``, ``SensorCamera.ClearData``,
    ``SensorCamera.WorldRenderFlag``); they are not part of the top-level
    ``newton`` namespace.

    Example::

        import warp as wp
        import newton
        from newton.sensors import SensorCamera

        camera = SensorCamera(model)
        camera.create_default_light()

        width, height = 640, 480
        camera_rays = SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=1.0, device=model.device)
        camera_transforms = wp.array([wp.transform_identity()], dtype=wp.transformf, device=model.device)
        color = camera.allocate_image_color(camera_transforms.shape[0], width, height)

        camera.update(state, camera_transforms, camera_rays, color_image=color)
    """

    class RayMode(enum.IntEnum):
        """How a bound sensor obtains the rays of built-in pinhole lenses.

        .. experimental::
        """

        PRECOMPUTED = 0
        """Cache one ray bundle per distinct lens among the selected cameras."""
        ANALYTIC = 1
        """Generate pinhole rays in the render kernel from the Model's camera intrinsics."""

    HIT_TRIANGLE_MESH = _HIT_TRIANGLE_MESH
    """``shape_index_image`` value of hits on deformable triangle surfaces; ``face_index_image``
    then holds the triangle index into :attr:`~newton.Model.tri_indices`."""
    HIT_PARTICLES = _HIT_PARTICLES
    """``shape_index_image`` value of particle hits; ``face_index_image`` then holds the particle index."""
    HIT_NONE = _HIT_NONE
    """``shape_index_image`` value of rays that hit nothing."""

    RayQuery = RayQuery
    AntiAliasing = AntiAliasing
    ClearData = ClearData
    GaussianRenderMode = GaussianRenderMode
    RenderConfig = RenderConfig
    RenderOrder = RenderOrder
    TextureProjectionMode = TextureProjectionMode
    Utils = Utils
    WorldRenderFlag = WorldRenderFlag

    def __init__(
        self,
        model: Model,
        *,
        cameras: str | list[str] | re.Pattern[str] | list[int] | None = None,
        sites: str | list[str] | re.Pattern[str] | list[int] | None = None,
        width: int | None = None,
        height: int | None = None,
        sample_count: int = 1,
        ray_mode: SensorCamera.RayMode | int = 0,
        ray_function: Any = None,
        ray_params: wp.array2d[wp.float32] | None = None,
        clipping_range: tuple[float, float] | None = None,
        default_clear_data: ClearData | None = None,
        default_render_config: RenderConfig | None = None,
        load_textures: bool = True,
    ):
        """Construct a camera sensor for a model.

        A sensor constructed with ``cameras`` or ``sites`` is *bound*: it renders one
        view per selected camera or site and world, computes view poses from the
        state, and needs no camera transforms or rays in :meth:`update`.

        .. experimental::

            ``cameras``, ``sites``, ``width``, ``height``, ``sample_count``,
            ``ray_mode``, ``ray_function``, ``ray_params`` and ``clipping_range``
            are part of the experimental model-camera API.

        Args:
            model: Newton simulation model to render. The sensor builds and owns
                its internal renderer for this model.
            cameras: Model cameras to render, by label pattern or index (see
                :ref:`label-matching`). Mutually exclusive with ``sites``.
            sites: Sites to render from, by label pattern or index. Sites only
                provide a frame, so rays come from :meth:`set_camera_rays`, a
                ``ray_function`` or :meth:`update`'s ``camera_rays``.
            width: Image width [px]; defaults to the cameras' common nominal resolution.
            height: Image height [px]; defaults to the cameras' common nominal resolution.
            sample_count: Rays per pixel of generated rays.
            ray_mode: :class:`RayMode` used for pinhole cameras.
            ray_function: Optional ``@wp.func`` ``(params: wp.array2d[float], query:
                SensorCamera.RayQuery) -> (origin, direction)`` evaluated in the render
                kernel for every view.
            ray_params: Parameter table passed to ``ray_function``.
            clipping_range: Near and far clip distance [m] for every view. Defaults
                to the cameras' clip ranges, or ``(0, 1e6)`` for sites.
            default_clear_data: Clear values used by :meth:`update` when its
                ``clear_data`` argument is ``None``. Defaults to ``ClearData()``.
            default_render_config: Render settings used by :meth:`update` when
                its ``render_config`` argument is ``None``. Defaults to
                ``RenderConfig()``.
            load_textures: Load mesh textures from disk. Set ``False`` for
                checkerboard or custom-texture workflows (see
                :meth:`assign_checkerboard_material`).
        """
        self.default_clear_data: ClearData = default_clear_data if default_clear_data is not None else ClearData()
        """Clear values used by :meth:`update` when its ``clear_data`` argument is ``None``."""
        self.default_render_config: RenderConfig = (
            default_render_config if default_render_config is not None else RenderConfig()
        )
        """Render settings used by :meth:`update` when its ``render_config`` argument is ``None``."""

        from .sensor_camera_render.render_context import RenderContext  # noqa: PLC0415

        self._bound = cameras is not None or sites is not None
        if self._bound:
            # Bound sensors share scene data (and lights) with the other bound sensors of the model.
            contexts = _SHARED_RENDER_CONTEXTS.setdefault(model, {})
            render_context = contexts.get(load_textures)
            if render_context is None:
                render_context = contexts[load_textures] = RenderContext(model, load_textures=load_textures)
            self._render_context = render_context
            self._bind(
                cameras,
                sites,
                width,
                height,
                sample_count,
                SensorCamera.RayMode(ray_mode),
                ray_function,
                ray_params,
                clipping_range,
            )
        else:
            self._render_context = RenderContext(model, load_textures=load_textures)

    # ------------------------------------------------------------------
    # Bound sensors (model cameras and sites)

    def _bind(self, cameras, sites, width, height, sample_count, ray_mode, ray_function, ray_params, clipping_range):
        model = self._render_context.model
        if cameras is not None and sites is not None:
            raise ValueError("cameras and sites are mutually exclusive")
        self._from_sites = sites is not None
        if self._from_sites:
            frames = list(match_labels(model.shape_label, sites))
            if not frames:
                raise ValueError(f"No sites match {sites!r}")
            flags = model.shape_flags.numpy()
            not_sites = [f for f in frames if not (int(flags[f]) & int(ShapeFlags.SITE))]
            if not_sites:
                raise ValueError(f"Shapes {not_sites} are not sites")
            frame_world = model.shape_world.numpy()
            self._frame_labels = model.shape_label
            frame_lens = {f: int(CameraLens.CUSTOM) for f in frames}
        else:
            frames = list(match_labels(model.camera_label, cameras))
            if not frames:
                raise ValueError(f"No cameras match {cameras!r}")
            frame_world = model.camera_world.numpy()
            self._frame_labels = model.camera_label
            lens = model.camera_lens.numpy()
            frame_lens = {f: int(lens[f]) for f in frames}

        view_frame, view_world = [], []
        for frame in frames:
            world = int(frame_world[frame])
            worlds = range(model.world_count) if world < 0 else (world,)
            for w in worlds:
                view_frame.append(int(frame))
                view_world.append(int(w))
        if not view_frame:
            raise ValueError("The selection produces no views (the model has no worlds)")

        if width is None or height is None:
            if self._from_sites:
                raise ValueError("width and height are required for site-bound sensors")
            resolution = {tuple(int(v) for v in model.camera_resolution.numpy()[f]) for f in frames}
            if len(resolution) != 1 or (0, 0) in resolution:
                raise ValueError("width and height are required unless all selected cameras share a resolution")
            (res_w, res_h) = resolution.pop()
            width = res_w if width is None else width
            height = res_h if height is None else height
        if int(width) <= 0 or int(height) <= 0 or int(sample_count) <= 0:
            raise ValueError("width, height and sample_count must be positive")

        device = self.device
        self.width = int(width)
        """Image width [px] of a bound sensor."""
        self.height = int(height)
        """Image height [px] of a bound sensor."""
        self.sample_count = int(sample_count)
        """Rays per pixel of a bound sensor."""
        self.view_count = len(view_frame)
        """Number of views rendered by a bound sensor."""
        self.view_camera = wp.array(view_frame, dtype=wp.int32, device=device)
        """Model camera (or site) index of each view, shape ``(view_count,)``."""
        self.view_world = wp.array(view_world, dtype=wp.int32, device=device)
        """World rendered by each view, shape ``(view_count,)``."""

        self._view_frame_np = np.asarray(view_frame, dtype=np.int32)
        self._view_lens_np = np.asarray([frame_lens[f] for f in view_frame], dtype=np.int32)
        self._ray_mode = ray_mode
        self._ray_function = ray_function
        self._ray_params = ray_params if ray_params is not None else wp.zeros((1, 1), dtype=wp.float32, device=device)
        self._view_transforms = wp.zeros(self.view_count, dtype=wp.transformf, device=device)
        self._world_indices = wp.zeros(self.view_count, dtype=wp.int32, device=device)
        self._view_clip = wp.zeros(self.view_count, dtype=wp.vec2f, device=device)
        self._clip_override = clipping_range
        if clipping_range is None and self._from_sites:
            self._clip_override = (0.0, 1.0e6)
        if self._clip_override is not None:
            near, far = (float(v) for v in self._clip_override)
            if not 0.0 <= near < far:
                raise ValueError(f"clipping_range must satisfy 0 <= near < far, got {self._clip_override}")
            self._view_clip.fill_(wp.vec2f(near, far))
        self._view_pinhole = wp.zeros(self.view_count, dtype=wp.vec4f, device=device)

        if ray_mode == SensorCamera.RayMode.ANALYTIC and ray_function is None:
            if np.any(self._view_lens_np != int(CameraLens.PINHOLE)):
                raise ValueError("RayMode.ANALYTIC needs PINHOLE cameras; use set_camera_rays or a ray_function")

        # Precomputed bundles: pinhole lens groups first, custom bundles appended by set_camera_rays().
        self._view_bundle_np = np.full(self.view_count, -1, dtype=np.int32)
        self._pinhole_bundle_count = 0
        self._pinhole_params = np.zeros((0, 4), dtype=np.float32)
        self._custom_bundles: list[tuple[np.ndarray, wp.array]] = []  # (views, rays)
        self._camera_rays: wp.array | None = None
        self._view_ray_bundle = wp.zeros(self.view_count, dtype=wp.int32, device=device)
        if ray_function is None and ray_mode == SensorCamera.RayMode.PRECOMPUTED:
            self.sync_camera_rays()

    @property
    def is_bound(self) -> bool:
        """Whether this sensor renders model cameras or sites selected at construction."""
        return self._bound

    def _require_bound(self) -> None:
        if not self._bound:
            raise RuntimeError("This operation requires a sensor constructed with cameras or sites")

    def _views_matching(self, cameras) -> np.ndarray:
        if cameras is None:
            return np.arange(self.view_count)
        frames = set(match_labels(self._frame_labels, cameras))
        return np.flatnonzero(np.isin(self._view_frame_np, list(frames)))

    def _compute_pinhole_params(self, out: wp.array) -> None:
        model = self._render_context.model
        wp.launch(
            _compute_view_pinhole_kernel,
            dim=self.view_count,
            inputs=[
                self.view_camera,
                model.camera_focal_length,
                model.camera_aperture,
                model.camera_aperture_offset,
                float(self.width) / float(self.height),
            ],
            outputs=[out],
            device=self.device,
        )

    def sync_camera_rays(self) -> None:
        """Rebuild the precomputed pinhole ray bundles from the Model's camera intrinsics.

        Call after changing ``model.camera_focal_length``, ``camera_aperture`` or
        ``camera_aperture_offset`` when using :attr:`RayMode.PRECOMPUTED`. Views with
        identical intrinsics share one bundle. Custom bundles set with
        :meth:`set_camera_rays` are kept.

        .. experimental::
        """
        self._require_bound()
        pinhole_views = np.flatnonzero(self._view_lens_np == int(CameraLens.PINHOLE))
        bundle_params = np.zeros((0, 4), dtype=np.float32)
        self._view_bundle_np[pinhole_views] = -1
        if len(pinhole_views) and not self._from_sites:
            self._compute_pinhole_params(self._view_pinhole)
            params = self._view_pinhole.numpy()[pinhole_views]
            bundle_params, inverse = np.unique(params.round(7), axis=0, return_inverse=True)
            self._view_bundle_np[pinhole_views] = inverse.reshape(-1)
        self._pinhole_bundle_count = len(bundle_params)
        self._pinhole_params = bundle_params
        self._rebuild_bundles()

    def _rebuild_bundles(self) -> None:
        device = self.device
        bundle_params = self._pinhole_params
        custom_offset = self._pinhole_bundle_count
        for index, (views, _rays) in enumerate(self._custom_bundles):
            self._view_bundle_np[views] = custom_offset + index
        bundle_count = self._pinhole_bundle_count + len(self._custom_bundles)
        shape = (max(bundle_count, 1) * self.height, self.width, self.sample_count, 2)
        if self._camera_rays is None or self._camera_rays.shape != shape:
            self._camera_rays = wp.zeros(shape, dtype=wp.vec3f, device=device)
        if len(bundle_params):
            wp.launch(
                _fill_pinhole_bundles_kernel,
                dim=(len(bundle_params), self.height, self.width, self.sample_count),
                inputs=[
                    wp.array(bundle_params, dtype=wp.vec4f, device=device),
                    self.width,
                    self.height,
                    self.sample_count,
                ],
                outputs=[self._camera_rays],
                device=device,
            )
        for index, (_views, rays) in enumerate(self._custom_bundles):
            row = (custom_offset + index) * self.height
            wp.copy(self._camera_rays[row : row + self.height], rays)
        self._view_ray_bundle.assign(self._view_bundle_np)

    def set_camera_rays(
        self,
        rays: wp.array4d[wp.vec3f] | Sequence[wp.array4d[wp.vec3f]],
        *,
        cameras: str | list[str] | re.Pattern[str] | list[int] | None = None,
    ) -> None:
        """Assign custom camera-space rays to views of a bound sensor.

        Rays use the sensor's frame convention: origins in ``[..., 0]`` and unit
        directions in ``[..., 1]``, in the camera (or site) frame. A zero direction
        marks an unused ray. The rays are copied into sensor-owned storage; calling
        again with the same selection and shapes is a device copy that keeps the
        storage address, so it can run inside a captured CUDA graph.

        .. experimental::

        Args:
            rays: One ``(height, width, sample_count, 2)`` bundle shared by the
                selected views, or a sequence with one bundle per selected camera
                (in index order).
            cameras: Label pattern or indices restricting the views to update. ``None``
                selects every view of a site-bound sensor and every ``CUSTOM`` camera
                otherwise.
        """
        self._require_bound()
        if cameras is None:
            views = np.arange(self.view_count)
            if not self._from_sites:
                views = np.flatnonzero(self._view_lens_np == int(CameraLens.CUSTOM))
        else:
            views = self._views_matching(cameras)
        if len(views) == 0:
            raise ValueError("No views match the camera selection")

        expected = (self.height, self.width, self.sample_count, 2)
        bundles = list(rays) if isinstance(rays, (list, tuple)) else [rays]
        for bundle in bundles:
            if bundle.dtype != wp.vec3f or tuple(bundle.shape) != expected:
                raise ValueError(f"Ray bundles must have dtype vec3f and shape {expected}, got {tuple(bundle.shape)}")
        if len(bundles) == 1:
            groups = [(views, bundles[0])]
        else:
            frames = sorted(set(self._view_frame_np[views].tolist()))
            if len(frames) != len(bundles):
                raise ValueError(f"Expected {len(frames)} bundles (one per selected camera), got {len(bundles)}")
            groups = [(views[self._view_frame_np[views] == f], b) for f, b in zip(frames, bundles, strict=True)]

        # Reuse slots with the same views so repeated updates keep the storage layout.
        layout_changed = False
        for group_views, bundle in groups:
            slot = next(
                (i for i, (v, _) in enumerate(self._custom_bundles) if np.array_equal(v, group_views)),
                None,
            )
            if slot is None:
                # Remove these views from older slots, then append.
                kept = []
                for v, b in self._custom_bundles:
                    remaining = np.setdiff1d(v, group_views)
                    if len(remaining):
                        kept.append((remaining, b))
                self._custom_bundles = [*kept, (group_views, bundle)]
                layout_changed = True
            else:
                self._custom_bundles[slot] = (group_views, bundle)
                if not layout_changed:
                    row = (self._pinhole_bundle_count + slot) * self.height
                    wp.copy(self._camera_rays[row : row + self.height], bundle)
        if layout_changed:
            self._rebuild_bundles()

    def compute_view_transforms(self, state: State, out: wp.array[wp.transformf] | None = None) -> wp.array:
        """Compute the world-space pose of every view of a bound sensor.

        .. experimental::

        Args:
            state: State providing ``body_q``.
            out: Optional output array, shape ``(view_count,)``.

        Returns:
            View poses, shape ``(view_count,)`` of ``transformf``.
        """
        self._require_bound()
        model = self._render_context.model
        if out is None:
            out = wp.empty(self.view_count, dtype=wp.transformf, device=self.device)
        if self._from_sites:
            frame_body, frame_transform = model.shape_body, model.shape_transform
        else:
            frame_body, frame_transform = model.camera_body, model.camera_transform
        body_q = state.body_q if state.body_q is not None else wp.zeros(1, dtype=wp.transformf, device=self.device)
        wp.launch(
            _compute_view_transforms_kernel,
            dim=self.view_count,
            inputs=[self.view_camera, frame_body, frame_transform, body_q],
            outputs=[out],
            device=self.device,
        )
        return out

    @property
    def device(self) -> wp.Device:
        """Device of the model this sensor renders."""
        return self._render_context.model.device

    def allocate_image(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None, dtype: Any = None
    ) -> wp.array[Any]:
        """Allocate a zeroed output image array on the model device.

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].
            dtype: Warp element type for the array (e.g. ``wp.uint32``,
                ``wp.float32``, ``wp.vec3f``).

        Returns:
            Zeroed array of shape ``(view_count, height, width)`` and dtype
            *dtype*, on the model device.
        """
        if view_count is None or width is None or height is None:
            self._require_bound()
            view_count = self.view_count if view_count is None else view_count
            width = self.width if width is None else width
            height = self.height if height is None else height
        if dtype is None:
            raise TypeError("allocate_image() requires dtype")
        return wp.zeros((int(view_count), int(height), int(width)), dtype=dtype, device=self.device)

    def allocate_image_color(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None
    ) -> wp.array3d[wp.uint32]:
        """Allocate an RGBA color output array (packed ``uint32``).

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``uint32``.
        """
        return self.allocate_image(view_count, width, height, wp.uint32)

    def allocate_image_depth(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None
    ) -> wp.array3d[wp.float32]:
        """Allocate a ray-distance depth output array [m].

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``float32``.
        """
        return self.allocate_image(view_count, width, height, wp.float32)

    def allocate_image_depth_forward(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None
    ) -> wp.array3d[wp.float32]:
        """Allocate a forward (planar) depth output array [m].

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``float32``.
        """
        return self.allocate_image_depth(view_count, width, height)

    def allocate_image_shape_index(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None
    ) -> wp.array3d[wp.uint32]:
        """Allocate a shape-index output array.

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``uint32``.
        """
        return self.allocate_image(view_count, width, height, wp.uint32)

    def allocate_image_normal(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None
    ) -> wp.array3d[wp.vec3f]:
        """Allocate a world-space surface-normal output array (``vec3f``).

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``vec3f``.
        """
        return self.allocate_image(view_count, width, height, wp.vec3f)

    def allocate_image_albedo(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None
    ) -> wp.array3d[wp.uint32]:
        """Allocate an RGBA albedo output array (packed ``uint32``).

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``uint32``.
        """
        return self.allocate_image(view_count, width, height, wp.uint32)

    def allocate_image_color_hdr(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None
    ) -> wp.array3d[wp.vec3f]:
        """Allocate a linear HDR color output array (``vec3f``).

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``vec3f``.
        """
        return self.allocate_image(view_count, width, height, wp.vec3f)

    def allocate_image_face_index(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None
    ) -> wp.array3d[wp.int32]:
        """Allocate a face-index output array (``int32``, ``-1`` where nothing is hit).

        Holds the triangle index for deformable-surface hits, the face index for mesh
        hits and the particle index for particle hits.

        .. experimental::
        """
        return self.allocate_image(view_count, width, height, wp.int32)

    def allocate_image_barycentric(
        self, view_count: int | None = None, width: int | None = None, height: int | None = None
    ) -> wp.array3d[wp.vec2f]:
        """Allocate a barycentric-coordinate output array (``vec2f``) for triangle hits.

        .. experimental::
        """
        return self.allocate_image(view_count, width, height, wp.vec2f)

    @staticmethod
    def compute_camera_rays_pinhole(
        width: int,
        height: int,
        *,
        camera_fov: float | None = None,
        focal_length: float | None = None,
        horizontal_aperture: float | None = None,
        vertical_aperture: float | None = None,
        horizontal_aperture_offset: float = 0.0,
        vertical_aperture_offset: float = 0.0,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one pinhole camera.

        Provide either ``camera_fov`` or the aperture triple (``focal_length``,
        ``horizontal_aperture``, ``vertical_aperture``), not both. The focal
        length and apertures share consistent units; only their ratios affect
        the ray directions.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_fov: Vertical field of view [rad], in ``(0, pi)``. Mutually
                exclusive with the aperture parameters.
            focal_length: Lens focal length; must be positive.
            horizontal_aperture: Horizontal sensor aperture; must be positive.
            vertical_aperture: Vertical sensor aperture; must be positive.
            horizontal_aperture_offset: Horizontal principal-point offset.
            vertical_aperture_offset: Vertical principal-point offset.
            sample_count: Number of rays to generate per pixel. Use one for a
                centered ray per pixel.
            out_rays: Optional output buffer, shape
                ``(height, width, sample_count, 2)`` of ``vec3f``.
            device: Device for the ray bundle. Defaults to the current Warp device.

        Returns:
            Ray origins and directions, shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        use_aperture = focal_length is not None or horizontal_aperture is not None or vertical_aperture is not None
        if use_aperture:
            if camera_fov is not None:
                raise ValueError("camera_fov cannot be provided with aperture parameters.")
            if focal_length is None or horizontal_aperture is None or vertical_aperture is None:
                raise ValueError("focal_length, horizontal_aperture, and vertical_aperture must be provided together.")
            if float(focal_length) <= 0.0 or float(horizontal_aperture) <= 0.0 or float(vertical_aperture) <= 0.0:
                raise ValueError("focal_length, horizontal_aperture, and vertical_aperture must be positive.")

            wp.launch(
                kernel=camera_utils.compute_camera_rays_pinhole_from_aperture_kernel,
                dim=(height, width, sample_count),
                inputs=[
                    width,
                    height,
                    float(focal_length),
                    float(horizontal_aperture),
                    float(vertical_aperture),
                    float(horizontal_aperture_offset),
                    float(vertical_aperture_offset),
                    sample_count,
                    out_rays,
                ],
                device=device,
            )

            return out_rays

        if camera_fov is None:
            raise ValueError("camera_fov must be provided when aperture parameters are not used.")
        if not 0.0 < float(camera_fov) < math.pi:
            raise ValueError(f"camera_fov must be in (0, pi) radians, got {float(camera_fov)}.")

        wp.launch(
            kernel=camera_utils.compute_camera_rays_pinhole,
            dim=(height, width, sample_count),
            inputs=[
                width,
                height,
                float(camera_fov),
                sample_count,
                out_rays,
            ],
            device=device,
        )

        return out_rays

    @staticmethod
    def compute_camera_rays_pinhole_usd(
        width: int,
        height: int,
        camera: Any,
        *,
        time: Any | None = None,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one USD pinhole camera.

        Reads the perspective intrinsics (focal length and aperture) from a USD
        camera and builds the matching pinhole ray bundle.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera: A ``UsdGeom.Camera`` or a ``Usd.Prim`` that is a camera. Its
                projection must be ``perspective``.
            time: USD time to sample the camera attributes at (a
                ``Usd.TimeCode`` or a frame number). If ``None``, the default
                time code is used.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        camera_utils.compute_camera_rays_pinhole_usd(
            width,
            height,
            camera,
            device=device,
            time=time,
            sample_count=sample_count,
            out_rays=out_rays,
        )
        return out_rays

    @staticmethod
    def compute_camera_rays_pinhole_opencv(
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        *,
        image_width: float | None = None,
        image_height: float | None = None,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        k5: float = 0.0,
        k6: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
        s1: float = 0.0,
        s2: float = 0.0,
        s3: float = 0.0,
        s4: float = 0.0,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one OpenCV pinhole camera.

        Inverts OpenCV's rational radial, tangential, and thin-prism distortion
        model with damped Newton iteration. The four- and five-coefficient
        variants are represented by leaving unused coefficients at zero. Pixels
        whose inverse cannot be verified within the solver tolerance receive a
        zero direction.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            fx: Horizontal focal length [px].
            fy: Vertical focal length [px].
            cx: Principal point x-coordinate [px].
            cy: Principal point y-coordinate [px].
            image_width: Calibration image width [px]. If ``None``, uses *width*.
            image_height: Calibration image height [px]. If ``None``, uses
                *height*.
            k1: First numerator radial distortion coefficient.
            k2: Second numerator radial distortion coefficient.
            k3: Third numerator radial distortion coefficient.
            k4: First denominator radial distortion coefficient.
            k5: Second denominator radial distortion coefficient.
            k6: Third denominator radial distortion coefficient.
            p1: First tangential distortion coefficient.
            p2: Second tangential distortion coefficient.
            s1: First thin-prism distortion coefficient.
            s2: Second thin-prism distortion coefficient.
            s3: Third thin-prism distortion coefficient.
            s4: Fourth thin-prism distortion coefficient.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.

        Raises:
            ValueError: If any focal length or calibration image dimension is
                non-positive, or if any calibration value is non-finite.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)
        if not (math.isfinite(fx) and math.isfinite(fy) and fx > 0.0 and fy > 0.0):
            raise ValueError("fx and fy must be finite and positive.")
        if not (
            math.isfinite(image_width) and math.isfinite(image_height) and image_width > 0.0 and image_height > 0.0
        ):
            raise ValueError("image_width and image_height must be finite and positive.")
        if not all(math.isfinite(value) for value in (cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4)):
            raise ValueError("cx, cy, and distortion coefficients must be finite.")

        wp.launch(
            kernel=camera_utils.compute_camera_rays_pinhole_opencv_kernel,
            dim=(height, width, sample_count),
            inputs=[
                width,
                height,
                image_width,
                image_height,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                p1,
                p2,
                s1,
                s2,
                s3,
                s4,
                sample_count,
                out_rays,
            ],
            device=device,
        )

        return out_rays

    @staticmethod
    def compute_camera_rays_fisheye_opencv(
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        *,
        image_width: float | None = None,
        image_height: float | None = None,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one OpenCV fisheye camera.

        Inverts the OpenCV fisheye radius polynomial
        ``r = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)``,
        which must be monotonic over ``[0, min(max_fov / 2, pi)]``.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            fx: Horizontal focal length [px].
            fy: Vertical focal length [px].
            cx: Principal point x-coordinate [px].
            cy: Principal point y-coordinate [px].
            image_width: Calibration image width [px]. If ``None``, uses *width*.
            image_height: Calibration image height [px]. If ``None``, uses
                *height*.
            k1: First OpenCV fisheye distortion coefficient.
            k2: Second OpenCV fisheye distortion coefficient.
            k3: Third OpenCV fisheye distortion coefficient.
            k4: Fourth OpenCV fisheye distortion coefficient.
            max_fov: Maximum field of view [rad]. Pixels whose undistorted angle
                exceeds ``max_fov / 2`` receive a zero ray.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_opencv_kernel,
            dim=(height, width, sample_count),
            inputs=[
                width,
                height,
                image_width,
                image_height,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
                max_fov,
                sample_count,
                out_rays,
            ],
            device=device,
        )

        return out_rays

    @staticmethod
    def compute_camera_rays_fisheye_ftheta(
        width: int,
        height: int,
        optical_center_x: float,
        optical_center_y: float,
        *,
        image_width: float | None = None,
        image_height: float | None = None,
        k0: float = 0.0,
        k1: float = 1.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one F-theta fisheye camera.

        Inverts the F-theta radius polynomial
        ``r = k0 + k1 theta + k2 theta^2 + k3 theta^3 + k4 theta^4``, which must
        be monotonic over ``[0, min(max_fov / 2, pi)]``.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            optical_center_x: Optical center x-coordinate [px].
            optical_center_y: Optical center y-coordinate [px].
            image_width: Calibration image width [px] (the F-theta nominal
                width). If ``None``, uses *width*.
            image_height: Calibration image height [px] (the F-theta nominal
                height). If ``None``, uses *height*.
            k0: Constant F-theta polynomial coefficient [px].
            k1: Linear F-theta polynomial coefficient [px/rad].
            k2: Quadratic F-theta polynomial coefficient [px/rad^2].
            k3: Cubic F-theta polynomial coefficient [px/rad^3].
            k4: Quartic F-theta polynomial coefficient [px/rad^4].
            max_fov: Maximum field of view [rad]. Pixels whose undistorted angle
                exceeds ``max_fov / 2`` receive a zero ray.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_ftheta_kernel,
            dim=(height, width, sample_count),
            inputs=[
                width,
                height,
                image_width,
                image_height,
                optical_center_x,
                optical_center_y,
                k0,
                k1,
                k2,
                k3,
                k4,
                max_fov,
                sample_count,
                out_rays,
            ],
            device=device,
        )

        return out_rays

    @staticmethod
    def compute_camera_rays_fisheye_kannala_brandt(
        width: int,
        height: int,
        optical_center_x: float,
        optical_center_y: float,
        *,
        image_width: float | None = None,
        image_height: float | None = None,
        k0: float = 1.0,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one Kannala-Brandt fisheye camera.

        Inverts the Kannala-Brandt radius polynomial
        ``r = k0 theta + k1 theta^3 + k2 theta^5 + k3 theta^7``, which must be
        monotonic over ``[0, min(max_fov / 2, pi)]``.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            optical_center_x: Optical center x-coordinate [px].
            optical_center_y: Optical center y-coordinate [px].
            image_width: Calibration image width [px]. If ``None``, uses *width*.
            image_height: Calibration image height [px]. If ``None``, uses
                *height*.
            k0: First Kannala-Brandt polynomial coefficient [px/rad].
            k1: Second Kannala-Brandt polynomial coefficient [px/rad^3].
            k2: Third Kannala-Brandt polynomial coefficient [px/rad^5].
            k3: Fourth Kannala-Brandt polynomial coefficient [px/rad^7].
            max_fov: Maximum field of view [rad]. Pixels whose undistorted angle
                exceeds ``max_fov / 2`` receive a zero ray.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_kannala_brandt_kernel,
            dim=(height, width, sample_count),
            inputs=[
                width,
                height,
                image_width,
                image_height,
                optical_center_x,
                optical_center_y,
                k0,
                k1,
                k2,
                k3,
                max_fov,
                sample_count,
                out_rays,
            ],
            device=device,
        )

        return out_rays

    def compute_camera_transforms_usd(
        self,
        cameras: Any,
        *,
        time: Any | None = None,
        xform: Any | None = None,
    ) -> wp.array[wp.transformf]:
        """Read world-space camera transforms from one USD camera per view.

        Converts each camera pose from its USD stage up axis to the rendered
        model's up axis, then composes an optional scene *xform* (e.g. the
        transform passed to ``ModelBuilder.add_usd``). Use the result as the
        ``camera_transforms`` argument to :meth:`update`.

        Args:
            cameras: A single ``UsdGeom.Camera``/``Usd.Prim`` (one view) or a
                sequence of them (one per view, in view order).
            time: USD time to sample the camera poses at (a ``Usd.TimeCode`` or a
                frame number). If ``None``, the default time code is used.
            xform: Optional scene transform ``(pos, quat)`` applied on top of the
                up-axis conversion, matching the pose used when importing the
                stage.

        Returns:
            World-space camera transforms, shape ``(view_count,)`` of
            ``transformf``, on the model device.
        """
        from ..core import Axis  # noqa: PLC0415
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        model = self._render_context.model
        return camera_utils.compute_camera_transforms_usd(
            cameras,
            device=model.device,
            target_up_axis=Axis(int(model.up_axis)),
            time=time,
            xform=xform,
        )

    def create_default_light(
        self,
        *,
        enable_shadows: bool = True,
        direction: wp.vec3f | None = None,
        color: Sequence[float] | wp.vec3f | None = None,
    ) -> None:
        """Create a default directional light for the rendered scene.

        Args:
            enable_shadows: Enable shadow casting for this light. Shadows are
                only rendered when the render config also enables them, i.e.
                ``enable_shadows=True`` here **and**
                ``render_config.enable_shadows=True`` (the latter defaults to
                ``False``); both switches must be set.
            direction: Normalized light direction. If ``None``, defaults to
                normalized ``(-1, 1, -1)`` for a Z-up model, rotated to the
                model's up axis otherwise, so the light always points down.
            color: Linear RGB radiance of the light. If ``None``, defaults to
                white ``(1, 1, 1)``. Changing it does not recompile the render
                kernel.
        """
        self._render_context.create_default_light(enable_shadows=enable_shadows, direction=direction, color=color)

    def set_ambient_light(
        self,
        sky_color: Sequence[float] | wp.vec3f,
        ground_color: Sequence[float] | wp.vec3f | None = None,
    ) -> None:
        """Set the hemispheric ambient light used when ambient lighting is enabled.

        Surfaces facing the model's up axis receive *sky_color*, surfaces facing
        away receive *ground_color*, and other orientations blend between the
        two. The defaults are ``(0.2, 0.2, 0.225)`` and ``(0.05, 0.05, 0.06)``.
        Changing the colors does not recompile the render kernel.

        Args:
            sky_color: Linear RGB ambient radiance from above.
            ground_color: Linear RGB ambient radiance from below. If ``None``,
                *sky_color* is used, which gives uniform ambient light.
        """
        self._render_context.set_ambient_light(sky_color, ground_color)

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
            resolution: Texture resolution [px] (square texture).
            checker_size: Size of each checkerboard square [px].
        """
        self._render_context.assign_checkerboard_material(
            shape_indices=shape_indices, resolution=resolution, checker_size=checker_size
        )

    def sync_deformable_meshes(self, state: State) -> None:
        """Synchronize render-only deformable triangle-mesh points from *state*.

        :meth:`update` calls this by default (pass ``sync_deformables=False`` to
        skip it when you already synced). Call it explicitly only when you opt
        out of the automatic sync. It syncs deformable mesh points (rigid-only
        scenes are a no-op) but does not touch transforms; shape and particle
        BVHs are refit separately via :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles`.

        Args:
            state: Current simulation state with particle positions.
        """
        self._render_context.update(state)

    @staticmethod
    def _validate_render_array(name: str, array: Any, dtype: Any, device: wp.Device) -> None:
        if not isinstance(array, wp.array):
            raise TypeError(f"{name} must be a Warp array, got {type(array).__name__}.")
        if array.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}.")
        if array.device != device:
            raise RuntimeError(f"{name} must be on the model device ({device}), got {array.device}.")

    def update(
        self,
        state: State,
        camera_transforms: wp.array[wp.transformf] | None = None,
        camera_rays: wp.array4d[wp.vec3f] | None = None,
        *,
        color_image: wp.array3d[wp.uint32] | None = None,
        depth_image: wp.array3d[wp.float32] | None = None,
        forward_depth_image: wp.array3d[wp.float32] | None = None,
        shape_index_image: wp.array3d[wp.uint32] | None = None,
        normal_image: wp.array3d[wp.vec3f] | None = None,
        albedo_image: wp.array3d[wp.uint32] | None = None,
        hdr_color_image: wp.array3d[wp.vec3f] | None = None,
        world_indices: wp.array[wp.int32] | None = None,
        clear_data: ClearData | None = None,
        render_config: RenderConfig | None = None,
        sync_deformables: bool = True,
        view_mask: wp.array[wp.bool] | None = None,
        face_index_image: wp.array3d[wp.int32] | None = None,
        barycentric_image: wp.array3d[wp.vec2f] | None = None,
    ) -> None:
        """Render this camera sensor.

        The number of views is inferred from ``camera_transforms.shape[0]``; all
        non-``None`` output arrays must have shape ``(view_count, height, width)``
        matching the ``camera_rays`` image dimensions.

        On any frame whose geometry moved, refit the model's shape and particle
        BVHs with :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles` (both are built initially by
        :meth:`~newton.ModelBuilder.finalize`) before calling this; otherwise the
        render reads stale bounds. Deformable triangle-mesh points are synced from
        *state* automatically (see ``sync_deformables``).

        The camera looks along its local ``-Z`` axis with ``+Y`` up and ``+X``
        right (the USD/OpenGL convention); ``camera_transforms`` place that frame
        in world space. ``camera_rays[..., 0]`` are per-sample ray origins and
        ``camera_rays[..., 1]`` are ray directions, both in camera space.

        Args:
            state: Simulation state with body and particle transforms.
            camera_transforms: World-space camera transform per view [m, rad],
                shape ``(view_count,)`` of ``transformf``, on the model device.
            camera_rays: Camera-space ray origins and directions, shape
                ``(height, width, sample_count, 2)`` of ``vec3f``, on the
                model device. Bundles with ``sample_count > 1`` require
                ``render_config.anti_aliasing`` to be :attr:`AntiAliasing.SSAA`
                or :attr:`AntiAliasing.MSAA`.
            color_image: Output RGBA color buffer (packed ``uint32``).
            depth_image: Output ray-distance depth from each ray origin [m].
            forward_depth_image: Output depth measured from the camera origin
                along its forward axis [m], including per-ray origin offsets.
            shape_index_image: Output shape-index buffer.
            normal_image: Output world-space surface normals.
            albedo_image: Output albedo buffer (packed ``uint32``).
            hdr_color_image: Output linear HDR color buffer.
            world_indices: Optional per-view world selector, shape
                ``(view_count,)``. Defaults to the identity mapping (view ``i``
                renders world ``i``). A valid entry is a world index in
                ``[0, model.world_count)``. A
                :class:`~newton.sensors.SensorCamera.WorldRenderFlag` sentinel
                disables the view (``DISABLE_CLEAR`` clears the outputs,
                ``DISABLE_PRESERVE`` leaves them unchanged). Any other value,
                including ``-1`` (reserved for future global-world rendering) and
                indices ``>= model.world_count``, is treated as ``DISABLE_CLEAR``.
            clear_data: Clear values for this call. Defaults to
                :attr:`default_clear_data`.
            render_config: Render settings for this call. Defaults to
                :attr:`default_render_config`.
            sync_deformables: Sync deformable triangle-mesh points from *state*
                before rendering (a no-op for rigid-only scenes). Set ``False``
                if you already called :meth:`sync_deformable_meshes` this frame.
        """
        render_context = self._render_context
        model = render_context.model

        if self._bound:
            self._update_bound(
                state,
                camera_transforms,
                camera_rays,
                color_image=color_image,
                depth_image=depth_image,
                forward_depth_image=forward_depth_image,
                shape_index_image=shape_index_image,
                normal_image=normal_image,
                albedo_image=albedo_image,
                hdr_color_image=hdr_color_image,
                face_index_image=face_index_image,
                barycentric_image=barycentric_image,
                world_indices=world_indices,
                view_mask=view_mask,
                clear_data=clear_data,
                render_config=render_config,
                sync_deformables=sync_deformables,
            )
            return
        if camera_transforms is None or camera_rays is None:
            raise ValueError("camera_transforms and camera_rays are required for sensors without cameras or sites")
        if view_mask is not None:
            raise ValueError("view_mask requires a sensor constructed with cameras or sites")

        if sync_deformables:
            render_context.update(state)

        self._validate_render_array("camera_transforms", camera_transforms, wp.transformf, model.device)
        if camera_transforms.ndim != 1 or camera_transforms.shape[0] <= 0:
            raise ValueError(f"camera_transforms must have shape (view_count,), got {tuple(camera_transforms.shape)}.")

        self._validate_render_array("camera_rays", camera_rays, wp.vec3f, model.device)
        multisample_shape = camera_rays.ndim == 4 and camera_rays.shape[2] > 0 and camera_rays.shape[3] == 2
        if not multisample_shape or camera_rays.shape[0] <= 0 or camera_rays.shape[1] <= 0:
            raise ValueError(
                f"camera_rays must have shape (height, width, sample_count, 2), got {tuple(camera_rays.shape)}."
            )

        view_count = int(camera_transforms.shape[0])
        # Without an explicit mapping the renderer uses ``world_index == view_index``
        # (no array is built), which is only valid when there are at least as many
        # worlds as views; otherwise it would index past the model's per-world data.
        if world_indices is None and view_count > int(model.world_count):
            raise ValueError(
                f"view_count ({view_count}) exceeds model.world_count ({int(model.world_count)}); "
                "pass an explicit world_indices mapping for the extra views."
            )

        with wp.ScopedTimer("Newton::SensorCamera::update", active=PROFILE_ENABLED, use_nvtx=True, synchronize=True):
            render_context.render(
                state,
                camera_transforms=camera_transforms,
                camera_rays=camera_rays,
                world_indices=world_indices,
                color_image=color_image,
                hdr_color_image=hdr_color_image,
                depth_image=depth_image,
                forward_depth_image=forward_depth_image,
                shape_index_image=shape_index_image,
                normal_image=normal_image,
                albedo_image=albedo_image,
                face_index_image=face_index_image,
                barycentric_image=barycentric_image,
                clear_data=clear_data if clear_data is not None else self.default_clear_data,
                config=render_config if render_config is not None else self.default_render_config,
            )

    def _update_bound(
        self,
        state: State,
        camera_transforms,
        camera_rays,
        *,
        world_indices,
        view_mask,
        clear_data,
        render_config,
        sync_deformables,
        **outputs,
    ) -> None:
        render_context = self._render_context
        model = render_context.model
        if sync_deformables:
            render_context.update(state)

        if camera_transforms is None:
            camera_transforms = self.compute_view_transforms(state, self._view_transforms)
        elif camera_transforms.shape != (self.view_count,):
            raise ValueError(f"camera_transforms must have shape ({self.view_count},)")

        if self._clip_override is None:
            wp.launch(
                _gather_view_clip_kernel,
                dim=self.view_count,
                inputs=[self.view_camera, model.camera_clipping_range],
                outputs=[self._view_clip],
                device=self.device,
            )

        if world_indices is None:
            world_indices = self.view_world
            if view_mask is not None:
                if view_mask.shape != (self.view_count,) or view_mask.dtype != wp.bool:
                    raise ValueError(f"view_mask must be a bool array of shape ({self.view_count},)")
                wp.launch(
                    _apply_view_mask_kernel,
                    dim=self.view_count,
                    inputs=[self.view_world, view_mask],
                    outputs=[self._world_indices],
                    device=self.device,
                )
                world_indices = self._world_indices
        elif view_mask is not None:
            raise ValueError("Pass either world_indices or view_mask, not both")

        ray_kwargs: dict[str, Any] = {"image_size": (self.width, self.height), "view_frame": self.view_camera}
        if camera_rays is not None:
            ray_kwargs["sample_count"] = int(camera_rays.shape[2])
            if tuple(camera_rays.shape[:2]) != (self.height, self.width):
                raise ValueError(f"camera_rays must have shape ({self.height}, {self.width}, sample_count, 2)")
        elif self._ray_function is not None:
            camera_rays = self._ray_params_placeholder()
            ray_kwargs.update(
                sample_count=self.sample_count, ray_function=self._ray_function, ray_params=self._ray_params
            )
        elif self._ray_mode == SensorCamera.RayMode.ANALYTIC:
            self._compute_pinhole_params(self._view_pinhole)
            camera_rays = self._ray_params_placeholder()
            ray_kwargs.update(sample_count=self.sample_count, view_pinhole=self._view_pinhole)
        else:
            missing = np.flatnonzero(self._view_bundle_np < 0)
            if len(missing):
                labels = sorted({self._frame_labels[int(self._view_frame_np[v])] for v in missing})
                raise RuntimeError(f"No rays for {labels}; call set_camera_rays() for CUSTOM cameras and sites")
            camera_rays = self._camera_rays
            ray_kwargs.update(sample_count=self.sample_count, view_ray_bundle=self._view_ray_bundle)

        with wp.ScopedTimer("Newton::SensorCamera::update", active=PROFILE_ENABLED, use_nvtx=True, synchronize=True):
            render_context.render(
                state,
                camera_transforms=camera_transforms,
                camera_rays=camera_rays,
                world_indices=world_indices,
                view_clip=self._view_clip,
                clear_data=clear_data if clear_data is not None else self.default_clear_data,
                config=render_config if render_config is not None else self.default_render_config,
                **ray_kwargs,
                **outputs,
            )

    def _ray_params_placeholder(self) -> wp.array4d[wp.vec3f]:
        placeholder = getattr(self, "_empty_rays", None)
        if placeholder is None:
            placeholder = self._empty_rays = wp.zeros((1, 1, 1, 2), dtype=wp.vec3f, device=self.device)
        return placeholder
