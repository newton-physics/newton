# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera


def _solid_texture(rgb: tuple[int, int, int], size: int = 8) -> np.ndarray:
    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    pixels[..., 0] = rgb[0]
    pixels[..., 1] = rgb[1]
    pixels[..., 2] = rgb[2]
    pixels[..., 3] = 255
    return pixels


_RED = (255, 0, 0)
_GREEN = (0, 255, 0)
_BLUE = (0, 0, 255)
_WHITE = (255, 255, 255)


@unittest.skipUnless(wp.is_cuda_available(), "Texture sampling requires CUDA")
class TestSensorTiledCameraTextureSwap(unittest.TestCase):
    """Runtime texture swapping from a pre-registered texture pool."""

    @staticmethod
    def _build_textured_quad_model(texture: np.ndarray | None) -> tuple[newton.Model, int]:
        """One camera-facing unit quad at z=-2 with UVs; returns (model, quad shape index)."""
        vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        mesh = newton.Mesh(vertices, indices, uvs=uvs, texture=texture)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        shape_index = builder.add_shape_mesh(body, mesh=mesh, color=(1.0, 1.0, 1.0))
        return builder.finalize(), shape_index

    def _center_rgb(self, sensor: SensorTiledCamera, state) -> np.ndarray:
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device=sensor._SensorTiledCamera__render_context.device,
        )
        camera_rays = sensor.utils.compute_pinhole_camera_rays(1, 1, math.radians(30.0))
        albedo_image = sensor.utils.create_albedo_image_output(1, 1, camera_count=1)
        sensor.update(state, camera_transforms, camera_rays, albedo_image=albedo_image)
        packed = int(albedo_image.numpy()[0, 0, 0, 0])
        return np.array([packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF], dtype=np.uint8)

    def _assert_rgb(self, actual: np.ndarray, expected: tuple[int, int, int]) -> None:
        np.testing.assert_allclose(actual.astype(np.int64), np.array(expected, dtype=np.int64), atol=2)

    def test_swap_between_registered_pool_textures(self) -> None:
        """Swaps re-point a shape at pre-registered textures without reloading anything."""
        model, quad = self._build_textured_quad_model(_solid_texture(_RED))
        sensor = SensorTiledCamera(model=model, config=SensorTiledCamera.RenderConfig(enable_textures=True))
        state = model.state()

        self._assert_rgb(self._center_rgb(sensor, state), _RED)

        green_id, blue_id = sensor.utils.register_textures([_solid_texture(_GREEN), _solid_texture(_BLUE)])

        sensor.utils.set_shape_texture_ids([quad], [green_id])
        self._assert_rgb(self._center_rgb(sensor, state), _GREEN)

        # device-resident fast path: pure warp-array scatter, no host validation
        render_context = sensor._SensorTiledCamera__render_context
        device = render_context.device
        sensor.utils.set_shape_texture_ids(
            wp.array([quad], dtype=wp.int32, device=device),
            wp.array([blue_id], dtype=wp.int32, device=device),
        )
        self._assert_rgb(self._center_rgb(sensor, state), _BLUE)

        # -1 restores the untextured base shape color
        sensor.utils.set_shape_texture_ids([quad], [-1])
        self._assert_rgb(self._center_rgb(sensor, state), _WHITE)

    def test_register_textures_enables_texturing_on_untextured_model(self) -> None:
        """Registering into an untextured build flips enable_textures and recompiles once."""
        model, quad = self._build_textured_quad_model(texture=None)
        sensor = SensorTiledCamera(model=model)
        state = model.state()

        self._assert_rgb(self._center_rgb(sensor, state), _WHITE)

        (red_id,) = sensor.utils.register_textures([_solid_texture(_RED)])
        sensor.utils.set_shape_texture_ids([quad], [red_id])
        self._assert_rgb(self._center_rgb(sensor, state), _RED)

    def test_host_inputs_are_validated(self) -> None:
        model, quad = self._build_textured_quad_model(_solid_texture(_RED))
        sensor = SensorTiledCamera(model=model, config=SensorTiledCamera.RenderConfig(enable_textures=True))

        with self.assertRaises(ValueError):
            sensor.utils.set_shape_texture_ids([quad], [99])  # texture id out of range
        with self.assertRaises(ValueError):
            sensor.utils.set_shape_texture_ids([10_000], [0])  # shape index out of range
        with self.assertRaises(ValueError):
            sensor.utils.set_shape_texture_ids([quad], [0, 0])  # length mismatch


if __name__ == "__main__":
    unittest.main()
