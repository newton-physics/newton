# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sensor Texture Randomization
#
# Shows how to use SensorTiledCamera.set_shape_texture_ids(..., per_world=True)
# to randomize table textures in a tiled-camera robot scene.
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
TABLE_HEIGHT = 0.72

FR3_READY_Q = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -math.pi / 4.0,
    "fr3_joint3": 0.0,
    "fr3_joint4": -3.0 * math.pi / 4.0,
    "fr3_joint5": 0.0,
    "fr3_joint6": math.pi / 2.0,
    "fr3_joint7": math.pi / 4.0,
    "fr3_finger_joint1": 0.02,
    "fr3_finger_joint2": 0.02,
}

PRIMARY_COLORS = np.array(
    [
        (210, 70, 55),
        (55, 135, 215),
        (70, 170, 95),
        (220, 165, 45),
    ],
    dtype=np.uint8,
)
ACCENT_COLORS = np.array(
    [
        (255, 220, 210),
        (220, 240, 255),
        (220, 255, 225),
        (255, 245, 205),
    ],
    dtype=np.uint8,
)


def _make_table_texture(primary: np.ndarray, accent: np.ndarray, marker_count: int) -> np.ndarray:
    texture = np.empty((TEXTURE_SIZE, TEXTURE_SIZE, 4), dtype=np.uint8)
    texture[..., :3] = primary
    texture[..., 3] = 255

    border = 8
    texture[:border, :, :3] = accent
    texture[-border:, :, :3] = accent
    texture[:, :border, :3] = accent
    texture[:, -border:, :3] = accent

    for marker in range(marker_count):
        x0 = 18 + marker * 14
        texture[18:56, x0 : x0 + 5, :3] = accent
        texture[x0 : x0 + 5, 18:56, :3] = accent

    return texture


def _table_top_mesh_vertices(offset: float = 0.0) -> np.ndarray:
    return np.array(
        [
            [-0.9 + offset, -0.65, TABLE_HEIGHT],
            [0.9 + offset, -0.65, TABLE_HEIGHT],
            [0.9 + offset, 0.65, TABLE_HEIGHT],
            [-0.9 + offset, 0.65, TABLE_HEIGHT],
        ],
        dtype=np.float32,
    )


def _make_table_top_mesh(texture: np.ndarray, offset: float = 0.0) -> newton.Mesh:
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
    return newton.Mesh(
        _table_top_mesh_vertices(offset),
        indices,
        uvs=uvs,
        compute_inertia=False,
        texture=texture,
    )


def _add_table_legs(builder: newton.ModelBuilder):
    for x in (-0.75, 0.75):
        for y in (-0.5, 0.5):
            builder.add_shape_box(
                -1,
                xform=wp.transform(p=wp.vec3(x, y, TABLE_HEIGHT * 0.5), q=wp.quat_identity()),
                hx=0.045,
                hy=0.045,
                hz=TABLE_HEIGHT * 0.5,
                color=(0.36, 0.34, 0.32),
            )


def _build_fr3_builder() -> newton.ModelBuilder:
    robot_asset = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
    robot_builder = newton.ModelBuilder()
    robot_builder.add_urdf(robot_asset, floating=False)
    return robot_builder


def _build_model() -> tuple[newton.Model, int]:
    robot_builder = _build_fr3_builder()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    for world_index in range(WORLD_COUNT):
        builder.begin_world()
        builder.add_builder(
            robot_builder,
            xform=wp.transform(
                p=wp.vec3(0.45, 0.18, TABLE_HEIGHT + 0.015),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(180.0)),
            ),
            label_prefix=f"world_{world_index}/fr3",
        )
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(p=wp.vec3(-0.28, 0.18, TABLE_HEIGHT + 0.09), q=wp.quat_identity()),
            radius=0.09,
            color=(0.9, 0.42, 0.12),
            label=f"world_{world_index}/object",
        )
        builder.end_world()

    textures = [
        _make_table_texture(PRIMARY_COLORS[i], ACCENT_COLORS[i], marker_count=i + 1) for i in range(len(PRIMARY_COLORS))
    ]
    table_shape_index = builder.add_shape_mesh(
        -1,
        mesh=_make_table_top_mesh(textures[0]),
        color=(1.0, 1.0, 1.0),
        label="randomized_tabletop",
    )

    _add_table_legs(builder)

    for texture_index, texture in enumerate(textures[1:], start=1):
        builder.add_shape_mesh(
            -1,
            mesh=_make_table_top_mesh(texture, offset=100.0 + 10.0 * texture_index),
            color=(1.0, 1.0, 1.0),
            label=f"texture_donor_{texture_index}",
        )

    return builder.finalize(), table_shape_index


def _look_at_camera_transform(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> wp.transformf:
    eye_np = np.asarray(eye, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    forward = target_np - eye_np
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    rotation = wp.mat33f(
        wp.vec3f(right[0], up[0], -forward[0]),
        wp.vec3f(right[1], up[1], -forward[1]),
        wp.vec3f(right[2], up[2], -forward[2]),
    )
    return wp.transformf(wp.vec3f(*eye_np), wp.quat_from_matrix(rotation))


def _camera_transforms(device) -> wp.array2d[wp.transformf]:
    camera = _look_at_camera_transform((1.7, -1.45, 1.3), (0.3, 0.05, TABLE_HEIGHT + 0.18))
    return wp.array([[camera] * WORLD_COUNT], dtype=wp.transformf, device=device)


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


def _set_fr3_ready_pose(model: newton.Model, state: newton.State):
    joint_q = state.joint_q.numpy()
    joint_q_start = model.joint_q_start.numpy()

    for joint_index, label in enumerate(model.joint_label):
        value = FR3_READY_Q.get(label.rsplit("/", 1)[-1])
        if value is None:
            continue
        joint_q[int(joint_q_start[joint_index])] = value

    state.joint_q.assign(joint_q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.seed = args.seed
        self.output_image = Path(args.output_image) if args.output_image else None
        self.preview_saved = False

        self.model, self.table_shape_index = _build_model()
        self.state = self.model.state()
        _set_fr3_ready_pose(self.model, self.state)
        self.model.bvh_refit_shapes(self.state)
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(2.25, -2.65, 1.95), -25.0, 140.0)

        self.sensor = SensorTiledCamera(
            model=self.model,
            config=SensorTiledCamera.RenderConfig(enable_textures=True),
        )
        self.sensor.utils.create_default_light(enable_shadows=True)

        rng = np.random.default_rng(self.seed)
        self.texture_ids_by_world = rng.permutation(len(PRIMARY_COLORS)).astype(np.int32)

        texture_ids = np.full((WORLD_COUNT, self.model.shape_count), -1, dtype=np.int32)
        texture_ids[:, self.table_shape_index] = self.texture_ids_by_world
        self.sensor.set_shape_texture_ids(texture_ids, per_world=True)

        self.camera_transforms = _camera_transforms(self.model.device)
        self.camera_rays = self.sensor.utils.compute_pinhole_camera_rays(IMAGE_SIZE, IMAGE_SIZE, math.radians(45.0))
        self.color_image = self.sensor.utils.create_color_image_output(IMAGE_SIZE, IMAGE_SIZE)
        self.albedo_image = self.sensor.utils.create_albedo_image_output(IMAGE_SIZE, IMAGE_SIZE)
        self.shape_index_image = self.sensor.utils.create_shape_index_image_output(IMAGE_SIZE, IMAGE_SIZE)
        self.preview_rgba = None

    def step(self):
        pass

    def render(self):
        preview_rgba = self.render_sensors()

        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        self.viewer.log_image("randomized_table_textures", preview_rgba)
        self.viewer.end_frame()

        if self.output_image is not None and not self.preview_saved:
            self._save_preview(preview_rgba, self.output_image)
            self.preview_saved = True

    def render_sensors(self) -> wp.array3d[wp.uint8]:
        self.model.bvh_refit_shapes(self.state)
        self.sensor.update(
            self.state,
            self.camera_transforms,
            self.camera_rays,
            color_image=self.color_image,
            albedo_image=self.albedo_image,
            shape_index_image=self.shape_index_image,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        self.preview_rgba = self.sensor.utils.flatten_color_image_to_rgba(
            self.color_image, worlds_per_row=WORLDS_PER_ROW
        )
        return self.preview_rgba

    def _save_preview(self, preview_rgba: wp.array3d[wp.uint8], output_image: Path):
        from PIL import Image  # noqa: PLC0415

        output_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview_rgba.numpy()).convert("RGB").save(output_image, quality=95)
        print(f"Saved table-texture randomization preview to {output_image}")

    def test_final(self):
        self.render_sensors()

        shape_index = self.shape_index_image.numpy()
        albedo = self.albedo_image.numpy()
        matched_texture_ids = []
        for world_index in range(WORLD_COUNT):
            table_mask = shape_index[world_index, 0] == self.table_shape_index
            assert table_mask.sum() > IMAGE_SIZE * IMAGE_SIZE * 0.08

            packed_table = albedo[world_index, 0][table_mask]
            table_colors = np.array([_unpack_rgba(pixel)[:3] for pixel in packed_table], dtype=np.float32)
            mean_table_color = table_colors.mean(axis=0)
            distances = np.sum((PRIMARY_COLORS.astype(np.float32) - mean_table_color) ** 2, axis=1)
            matched_texture_ids.append(int(np.argmin(distances)))

        np.testing.assert_array_equal(matched_texture_ids, self.texture_ids_by_world)
        assert len(set(matched_texture_ids)) == WORLD_COUNT

        preview = self.preview_rgba.numpy()
        assert preview.shape == (WORLDS_PER_ROW * IMAGE_SIZE, WORLDS_PER_ROW * IMAGE_SIZE, 4)
        assert preview.dtype == np.uint8

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=7, help="Seed for the per-world table texture permutation.")
        parser.add_argument("--output-image", type=str, default=None, help="Optional path for a tiled preview image.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
