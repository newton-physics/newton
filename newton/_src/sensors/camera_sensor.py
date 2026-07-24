# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ..core.types import Devicelike
from .camera_sensor_renderer.types import ClearData, GaussianRenderMode, RenderConfig, RenderLightType, RenderOrder

if TYPE_CHECKING:
    from ..sim.model import Model
    from ..sim.state import State


def _resolve_fisheye_image_size(
    axis: str,
    image_size: float | None,
    nominal_size: float | None,
    default_size: int,
) -> float:
    if image_size is not None and nominal_size is not None and image_size != nominal_size:
        raise ValueError(f"image_{axis} and nominal_{axis} must match when both are provided.")
    if image_size is not None:
        return float(image_size)
    if nominal_size is not None:
        return float(nominal_size)
    return float(default_size)


def _validate_camera_ray_output(
    width: int,
    height: int,
    out_rays: wp.array3d[wp.vec3f] | None,
    device: Devicelike = None,
) -> tuple[int, int, wp.array3d[wp.vec3f], wp.Device]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    expected_shape = (height, width, 2)
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

    return width, height, out_rays, out_rays.device


@wp.kernel(enable_backward=False)
def _compute_camera_transforms(
    shape_index_by_world: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    out_camera_transforms: wp.array[wp.transformf],
):
    world_index = wp.tid()
    shape_index = shape_index_by_world[world_index]
    if shape_index < 0:
        out_camera_transforms[world_index] = wp.transform(wp.vec3f(0.0), wp.quat_identity())
        return

    body_index = shape_body[shape_index]
    camera_transform = shape_transform[shape_index]
    if body_index >= 0:
        camera_transform = wp.transform_multiply(body_q[body_index], camera_transform)

    out_camera_transforms[world_index] = camera_transform


class CameraSensor:
    """Camera ray bundle asset for shape-backed rendering.

    A camera sensor can be attached to a model as a ``GeoType.CAMERA`` shape.
    The shape supplies the camera transform, parent body, world, label, and
    custom attributes; this object supplies the image resolution and camera-space
    rays used for rendering.
    """

    RenderConfig = RenderConfig
    ClearData = ClearData
    GaussianRenderMode = GaussianRenderMode
    RenderLightType = RenderLightType
    RenderOrder = RenderOrder

    def __init__(self, rays: wp.array | np.ndarray):
        """Construct a camera sensor from a ray bundle.

        Args:
            rays: Camera-space ray origins and directions, shape
                ``(height, width, 2)`` of ``vec3f``.
        """
        self.rays = self._coerce_rays(rays)
        self.shape_indices = wp.array([], dtype=wp.int32, device=self.rays.device)
        """Indices into the owning model's ``shape_*`` arrays for this camera."""

        self.clear_data: ClearData | None = ClearData()
        """Values used to clear output images before rendering."""
        self.render_config: RenderConfig | None = RenderConfig()
        """Render settings used by :meth:`update`."""

        self._shape_index_by_world: wp.array[wp.int32] | None = None
        self._all_world_enabled: wp.array[wp.bool] | None = None
        self._camera_transforms: wp.array[wp.transformf] | None = None
        self._model_ref = None

        self.has_inertia = False
        self.mass = 0.0
        self.com = wp.vec3()
        self.I = wp.mat33()
        self.is_solid = False

    @property
    def width(self) -> int:
        """Image width [px]."""
        return int(self.rays.shape[1])

    @property
    def height(self) -> int:
        """Image height [px]."""
        return int(self.rays.shape[0])

    @property
    def utils(self):
        """Renderer utility helpers for the owning finalized model."""
        model = self._model_ref() if self._model_ref is not None else None
        if model is None:
            raise RuntimeError("CameraSensor.utils is available after the sensor has been finalized into a model.")

        from .camera_sensor_renderer import Utils  # noqa: PLC0415

        return Utils(self._get_render_context(model), self.render_config)

    def _coerce_rays(self, rays: wp.array | np.ndarray) -> wp.array3d[wp.vec3f]:
        if isinstance(rays, np.ndarray):
            rays = wp.array(np.ascontiguousarray(rays, dtype=np.float32), dtype=wp.vec3f)

        if not isinstance(rays, wp.array):
            raise TypeError(f"rays must be a Warp or NumPy array, got {type(rays).__name__}")
        if rays.dtype != wp.vec3f:
            raise ValueError(f"CameraSensor rays must have dtype vec3f, got {rays.dtype}")

        if rays.ndim != 3 or rays.shape[0] <= 0 or rays.shape[1] <= 0 or rays.shape[2] != 2:
            raise ValueError(f"CameraSensor rays must have shape (height, width, 2), got {rays.shape}")

        return rays

    @staticmethod
    def compute_camera_rays_pinhole(
        width: int,
        height: int,
        camera_fov: float | None = None,
        *,
        focal_length: float | None = None,
        horizontal_aperture: float | None = None,
        vertical_aperture: float | None = None,
        horizontal_aperture_offset: float = 0.0,
        vertical_aperture_offset: float = 0.0,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one pinhole camera."""
        from .camera_sensor_renderer import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)

        use_aperture = focal_length is not None or horizontal_aperture is not None or vertical_aperture is not None
        if use_aperture:
            if focal_length is None or horizontal_aperture is None or vertical_aperture is None:
                raise ValueError("focal_length, horizontal_aperture, and vertical_aperture must be provided together.")

            wp.launch(
                kernel=camera_utils.compute_camera_rays_pinhole_from_aperture_kernel,
                dim=(height, width),
                inputs=[
                    width,
                    height,
                    float(focal_length),
                    float(horizontal_aperture),
                    float(vertical_aperture),
                    float(horizontal_aperture_offset),
                    float(vertical_aperture_offset),
                    out_rays,
                ],
                device=device,
            )

            return out_rays

        if camera_fov is None:
            raise ValueError("camera_fov must be provided when aperture parameters are not used.")

        wp.launch(
            kernel=camera_utils.compute_camera_rays_pinhole,
            dim=(height, width),
            inputs=[
                width,
                height,
                float(camera_fov),
                out_rays,
            ],
            device=device,
        )

        return out_rays

    @staticmethod
    def compute_camera_rays_usd_pinhole(
        width: int,
        height: int,
        camera: Any,
        *,
        time: Any | None = None,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one USD pinhole camera."""
        from .camera_sensor_renderer import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)
        camera_utils.compute_camera_rays_usd_pinhole(
            width,
            height,
            camera,
            device=device,
            time=time,
            out_rays=out_rays,
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
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one OpenCV fisheye camera."""
        from .camera_sensor_renderer import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_opencv_kernel,
            dim=(height, width),
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
        nominal_width: float | None = None,
        nominal_height: float | None = None,
        k0: float = 0.0,
        k1: float = 1.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one F-theta fisheye camera."""
        from .camera_sensor_renderer import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)
        image_width = _resolve_fisheye_image_size("width", image_width, nominal_width, width)
        image_height = _resolve_fisheye_image_size("height", image_height, nominal_height, height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_ftheta_kernel,
            dim=(height, width),
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
        nominal_width: float | None = None,
        nominal_height: float | None = None,
        k0: float = 1.0,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one Kannala-Brandt fisheye camera."""
        from .camera_sensor_renderer import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)
        image_width = _resolve_fisheye_image_size("width", image_width, nominal_width, width)
        image_height = _resolve_fisheye_image_size("height", image_height, nominal_height, height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_kannala_brandt_kernel,
            dim=(height, width),
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
                out_rays,
            ],
            device=device,
        )

        return out_rays

    @property
    def view_count(self) -> int:
        """Number of model camera shapes using this sensor."""
        return int(self.shape_indices.shape[0])

    def finalize(self, device: Devicelike = None, shape_indices: Sequence[int] | None = None) -> int:
        """Move rays to *device* for model rendering.

        Args:
            device: Target model device.
            shape_indices: Indices into the owning model's ``shape_*`` arrays
                whose source is this camera.

        Returns:
            Zero, because camera sensors do not need a ``shape_source_ptr``.
        """
        if device is not None and self.rays.device != wp.get_device(device):
            self.rays = wp.clone(self.rays, device=device)
        if shape_indices is not None:
            self.shape_indices = wp.array(shape_indices, dtype=wp.int32, device=device)
        elif device is not None and self.shape_indices.device != wp.get_device(device):
            self.shape_indices = wp.clone(self.shape_indices, device=device)

        self._shape_index_by_world = None
        self._all_world_enabled = None
        self._camera_transforms = None
        return 0

    def _bind_model(self, model: Model) -> None:
        current_model = self._model_ref() if self._model_ref is not None else None
        if current_model is model:
            return

        self._model_ref = weakref.ref(model)

    def _get_render_context(self, model: Model):
        self._bind_model(model)
        return model.init_render_context()

    def _ensure_shape_index_by_world(self, model: Model) -> None:
        if (
            self._shape_index_by_world is not None
            and self._shape_index_by_world.shape == (model.world_count,)
            and self._shape_index_by_world.device == model.device
        ):
            return

        shape_indices = self.shape_indices.numpy().astype(np.int32, copy=False)
        shape_world = model.shape_world.numpy()
        shape_index_by_world = np.full(model.world_count, -1, dtype=np.int32)
        global_shape_index = -1

        for shape_index in shape_indices:
            world_index = int(shape_world[int(shape_index)])
            if world_index < 0:
                if global_shape_index >= 0:
                    raise RuntimeError(
                        "CameraSensor output has no camera axis; attach each CameraSensor to at most one "
                        "global camera shape."
                    )
                global_shape_index = int(shape_index)
            elif world_index >= model.world_count:
                raise RuntimeError(
                    f"CameraSensor shape {int(shape_index)} references world {world_index}, but model has "
                    f"{model.world_count} worlds."
                )
            else:
                if shape_index_by_world[world_index] >= 0:
                    raise RuntimeError(
                        "CameraSensor output has no camera axis; attach each CameraSensor to at most one "
                        "camera shape per world."
                    )
                shape_index_by_world[world_index] = int(shape_index)

        if global_shape_index >= 0:
            shape_index_by_world[shape_index_by_world < 0] = global_shape_index

        self._shape_index_by_world = wp.array(shape_index_by_world, dtype=wp.int32, device=model.device)

    def _ensure_render_buffers(self, model: Model) -> None:
        if self.view_count == 0:
            raise RuntimeError("CameraSensor is not attached to a finalized model.")

        self._ensure_shape_index_by_world(model)

        transforms_shape = (model.world_count,)
        if (
            self._camera_transforms is None
            or self._camera_transforms.shape != transforms_shape
            or self._camera_transforms.device != model.device
        ):
            self._camera_transforms = wp.empty(transforms_shape, dtype=wp.transformf, device=model.device)

        if (
            self._all_world_enabled is None
            or self._all_world_enabled.shape != transforms_shape
            or self._all_world_enabled.device != model.device
        ):
            self._all_world_enabled = wp.ones(transforms_shape, dtype=wp.bool, device=model.device)

    def _update_transforms(self, model: Model, state: State) -> None:
        self._ensure_render_buffers(model)

        wp.launch(
            kernel=_compute_camera_transforms,
            dim=(model.world_count,),
            inputs=[
                self._shape_index_by_world,
                model.shape_body,
                model.shape_transform,
                state.body_q,
            ],
            outputs=[self._camera_transforms],
            device=model.device,
        )

    def update(
        self,
        model: Model,
        state: State,
        *,
        color_image: wp.array3d[wp.uint32] | None = None,
        depth_image: wp.array3d[wp.float32] | None = None,
        forward_depth_image: wp.array3d[wp.float32] | None = None,
        shape_index_image: wp.array3d[wp.uint32] | None = None,
        normal_image: wp.array3d[wp.vec3f] | None = None,
        albedo_image: wp.array3d[wp.uint32] | None = None,
        hdr_color_image: wp.array3d[wp.vec3f] | None = None,
        world_enabled: wp.array[wp.bool] | None = None,
        kernel_block_dim: int = 64,
    ) -> None:
        """Render this camera sensor from its shape transforms.

        Output arrays must have shape ``(world_count, height, width)``. The
        owning model supplies camera transforms and world indices from its
        ``shape_*`` arrays.

        Args:
            model: Simulation model whose shapes will be rendered.
            state: Simulation state with body and particle transforms.
            color_image: Output RGBA color buffer (packed ``uint32``).
            depth_image: Output depth buffer [m].
            forward_depth_image: Output forward-depth buffer [m].
            shape_index_image: Output shape-index buffer.
            normal_image: Output world-space surface normals.
            albedo_image: Output albedo buffer (packed ``uint32``).
            hdr_color_image: Output linear HDR color buffer.
            world_enabled: Optional boolean mask selecting worlds to render.
                Disabled worlds are cleared.
            kernel_block_dim: Thread block dimension forwarded to ``wp.launch``.
        """
        if self.rays.device != model.device:
            raise RuntimeError("CameraSensor rays are not on the model device; finalize the model with this camera.")
        if self.shape_indices.device != model.device:
            raise RuntimeError(
                "CameraSensor shape indices are not on the model device; finalize the model with this camera."
            )

        render_context = self._get_render_context(model)
        self._update_transforms(model, state)
        if world_enabled is None:
            world_enabled = self._all_world_enabled

        render_context.render(
            model,
            state,
            camera_transforms=self._camera_transforms,
            camera_rays=self.rays,
            world_enabled=world_enabled,
            color_image=color_image,
            hdr_color_image=hdr_color_image,
            depth_image=depth_image,
            forward_depth_image=forward_depth_image,
            shape_index_image=shape_index_image,
            normal_image=normal_image,
            albedo_image=albedo_image,
            clear_data=self.clear_data,
            config=self.render_config,
            kernel_block_dim=kernel_block_dim,
        )
