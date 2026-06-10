# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sensor Texture Randomization
#
# Shows how to use SensorTiledCamera.set_shape_texture_ids(..., per_world=True)
# to randomize textured rendering for the same visible shape across worlds.
#
# Command: python -m newton.examples sensor_texture_randomization
#
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.sensors import SensorTiledCamera

WORLD_COUNT = 4
WORLDS_PER_ROW = 2
IMAGE_SIZE = 160
TEXTURE_SIZE = 128

PRIMARY_COLORS = np.array(
    [
        (220, 55, 70),
        (50, 135, 225),
        (70, 180, 95),
        (235, 180, 45),
    ],
    dtype=np.uint8,
)
ACCENT_COLORS = np.array(
    [
        (255, 220, 225),
        (220, 240, 255),
        (220, 255, 225),
        (255, 245, 205),
    ],
    dtype=np.uint8,
)


def _make_texture(primary: np.ndarray, accent: np.ndarray, marker_count: int) -> np.ndarray:
    texture = np.empty((TEXTURE_SIZE, TEXTURE_SIZE, 4), dtype=np.uint8)
    texture[..., :3] = primary
    texture[..., 3] = 255

    border = 10
    texture[:border, :, :3] = accent
    texture[-border:, :, :3] = accent
    texture[:, :border, :3] = accent
    texture[:, -border:, :3] = accent

    for marker in range(marker_count):
        x0 = 18 + marker * 14
        texture[22:62, x0 : x0 + 6, :3] = accent
        texture[x0 : x0 + 6, 22:62, :3] = accent

    return texture


def _quad_mesh_vertices(offset: float = 0.0) -> np.ndarray:
    return np.array(
        [
            [-1.0 + offset, -1.0, -2.0],
            [1.0 + offset, -1.0, -2.0],
            [1.0 + offset, 1.0, -2.0],
            [-1.0 + offset, 1.0, -2.0],
        ],
        dtype=np.float32,
    )


def _build_model() -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for _ in range(WORLD_COUNT):
        builder.begin_world()
        builder.end_world()

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    uvs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    textures = [
        _make_texture(PRIMARY_COLORS[i], ACCENT_COLORS[i], marker_count=i + 1) for i in range(len(PRIMARY_COLORS))
    ]

    visible_mesh = newton.Mesh(
        _quad_mesh_vertices(),
        indices,
        uvs=uvs,
        compute_inertia=False,
        texture=textures[0],
    )
    builder.add_shape_mesh(-1, mesh=visible_mesh, color=(1.0, 1.0, 1.0), label="shared_randomized_quad")

    for texture_index, texture in enumerate(textures[1:], start=1):
        donor_mesh = newton.Mesh(
            _quad_mesh_vertices(offset=100.0 + 10.0 * texture_index),
            indices,
            uvs=uvs,
            compute_inertia=False,
            texture=texture,
        )
        builder.add_shape_mesh(-1, mesh=donor_mesh, color=(1.0, 1.0, 1.0), label=f"texture_donor_{texture_index}")

    return builder.finalize()


def _camera_transforms(device) -> wp.array2d[wp.transformf]:
    transform = wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))
    return wp.array([[transform] * WORLD_COUNT], dtype=wp.transformf, device=device)


def _unpack_rgba(packed: int) -> np.ndarray:
    value = int(packed)
    return np.array(
        [
            value & 0xFF,
            (value >> 8) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF,
        ],
        dtype=np.uint8,
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.seed = args.seed
        self.output_image = Path(args.output_image) if args.output_image else None
        self.preview_saved = False

        self.model = _build_model()
        self.state = self.model.state()
        self.viewer.set_model(self.model)

        self.sensor = SensorTiledCamera(
            model=self.model,
            config=SensorTiledCamera.RenderConfig(enable_textures=True, enable_ambient_lighting=False),
        )

        rng = np.random.default_rng(self.seed)
        self.texture_ids_by_world = rng.permutation(len(PRIMARY_COLORS)).astype(np.int32)

        texture_ids = np.full((WORLD_COUNT, self.model.shape_count), -1, dtype=np.int32)
        texture_ids[:, 0] = self.texture_ids_by_world
        self.sensor.set_shape_texture_ids(texture_ids, per_world=True)

        self.camera_transforms = _camera_transforms(self.model.device)
        self.camera_rays = self.sensor.utils.compute_pinhole_camera_rays(IMAGE_SIZE, IMAGE_SIZE, math.radians(55.0))
        self.albedo_image = self.sensor.utils.create_albedo_image_output(IMAGE_SIZE, IMAGE_SIZE)
        self.shape_index_image = self.sensor.utils.create_shape_index_image_output(IMAGE_SIZE, IMAGE_SIZE)
        self.preview_rgba = None

    def step(self):
        pass

    def render(self):
        preview_rgba = self.render_sensors()

        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        self.viewer.log_image("per_world_texture_ids", preview_rgba)
        self.viewer.end_frame()

        if self.output_image is not None and not self.preview_saved:
            self._save_preview(preview_rgba, self.output_image)
            self.preview_saved = True

    def render_sensors(self) -> wp.array3d[wp.uint8]:
        self.sensor.update(
            self.state,
            self.camera_transforms,
            self.camera_rays,
            albedo_image=self.albedo_image,
            shape_index_image=self.shape_index_image,
        )
        self.preview_rgba = self.sensor.utils.flatten_color_image_to_rgba(
            self.albedo_image, worlds_per_row=WORLDS_PER_ROW
        )
        return self.preview_rgba

    def _save_preview(self, preview_rgba: wp.array3d[wp.uint8], output_image: Path):
        from PIL import Image  # noqa: PLC0415

        output_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview_rgba.numpy()).convert("RGB").save(output_image, quality=95)
        print(f"Saved texture-randomization preview to {output_image}")

    def test_final(self):
        self.render_sensors()

        shape_index = self.shape_index_image.numpy()
        center = IMAGE_SIZE // 2
        for world_index in range(WORLD_COUNT):
            assert shape_index[world_index, 0, center, center] == 0

        albedo = self.albedo_image.numpy()
        center_colors = np.array([_unpack_rgba(albedo[world, 0, center, center])[:3] for world in range(WORLD_COUNT)])
        expected_colors = PRIMARY_COLORS[self.texture_ids_by_world]
        np.testing.assert_allclose(center_colors, expected_colors, atol=1)
        assert len({tuple(color) for color in center_colors}) == WORLD_COUNT

        preview = self.preview_rgba.numpy()
        assert preview.shape == (WORLDS_PER_ROW * IMAGE_SIZE, WORLDS_PER_ROW * IMAGE_SIZE, 4)
        assert preview.dtype == np.uint8

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=7, help="Seed for the per-world texture-ID permutation.")
        parser.add_argument("--output-image", type=str, default=None, help="Optional path for a tiled preview image.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
