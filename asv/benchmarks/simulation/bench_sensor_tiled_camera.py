# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rendering benchmarks for the tiled camera sensor.

``FastSensorTiledCamera`` measures render-order variants on a single Franka
scene. The ``TiledCamera*`` scene benchmarks cover scenes of varying visual
complexity, all rendered with the configuration in the constants below, and
are intended for hill-climbing renderer performance:

- ``franka``: a fixed-base Franka FR3 arm.
- ``quadruped``: an ANYmal D quadruped in its nominal standing pose.
- ``franka_cabinet``: Isaac Lab's Franka cabinet (open-drawer) scene.
- ``shapes_256``: a grid of 256 primitive shapes.

Each scene is described once as a :class:`ScenePreset` in :data:`SCENES` and is
shared between the ASV benchmark classes and the preview image generator, so
previews show exactly what is benchmarked.

Run directly to benchmark, or to write PNG previews of each scene (a
single-world view and a 4x4 grid of 16 worlds)::

    uv run asv/benchmarks/simulation/bench_sensor_tiled_camera.py
    uv run asv/benchmarks/simulation/bench_sensor_tiled_camera.py --preview
"""

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import math
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import newton
import newton.examples
import newton.utils
from newton import ShapeFlags
from newton.sensors import SensorTiledCamera

NICE_NAMES = {}
ASV_RUN_TILED_CAMERA_BENCHMARKS_ENV_VAR = "NEWTON_RUN_TILED_CAMERA_BENCHMARKS"
TILED_BENCHMARK_METHODS = {
    "time_rendering_tiled_color_depth",
    "time_rendering_tiled_color_only",
    "time_rendering_tiled_depth_only",
}


def run_tiled_camera_benchmarks():
    return os.environ.get(ASV_RUN_TILED_CAMERA_BENCHMARKS_ENV_VAR, "").lower() in {"1", "true", "yes", "on"}


def nice_name(value):
    def decorator(func):
        func._nice_name = value
        return func

    return decorator


def nice_name_collector():
    def decorator(instance):
        for name, attr in instance.__dict__.items():
            if nice_name := getattr(attr, "_nice_name", None):
                NICE_NAMES[name] = nice_name
        return instance

    return decorator


@nice_name_collector()
class FastSensorTiledCamera:
    param_names = ["resolution", "world_count", "iterations"]
    params = ([64], [4096], [50])

    def __dir__(self):
        names = super().__dir__()
        if run_tiled_camera_benchmarks():
            return names
        return [name for name in names if name not in TILED_BENCHMARK_METHODS]

    def setup(self, resolution: int, world_count: int, iterations: int):
        self.device = wp.get_preferred_device()

        franka = newton.ModelBuilder()
        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )
        COLLIDE = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
        franka.shape_flags = [int(f) & ~COLLIDE for f in franka.shape_flags]
        franka.shape_collision_filter_pairs = []

        scene = newton.ModelBuilder()
        scene.replicate(franka, world_count)
        scene.add_ground_plane()

        self.model = scene.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.camera_transforms = wp.array(
            [
                [
                    wp.transformf(
                        wp.vec3f(2.4, 0.0, 0.8),
                        wp.quatf(0.4187639653682709, 0.4224344491958618, 0.5708873867988586, 0.5659270882606506),
                    )
                ]
                * world_count
            ],
            dtype=wp.transformf,
        )

        self.tiled_camera_sensor = SensorTiledCamera(model=self.model)
        self.tiled_camera_sensor.utils.create_default_light(enable_shadows=False)
        self.tiled_camera_sensor.utils.assign_checkerboard_material_to_all_shapes()

        self.camera_rays = self.tiled_camera_sensor.utils.compute_camera_rays_pinhole(
            resolution, resolution, camera_fovs=math.radians(45.0)
        )
        self.color_image = self.tiled_camera_sensor.utils.create_color_image_output(resolution, resolution)
        self.depth_image = self.tiled_camera_sensor.utils.create_depth_image_output(resolution, resolution)

        self.model.bvh_build_shapes(self.state)
        self.model.bvh_build_particles(self.state)
        self.tiled_camera_sensor.sync_transforms(self.state)

        # Warmup Kernels
        if run_tiled_camera_benchmarks():
            self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.TILED
            self.tiled_camera_sensor.render_config.tile_width = 8
            self.tiled_camera_sensor.render_config.tile_height = 8
            for out_color, out_depth in [(True, True), (True, False), (False, True)]:
                for _ in range(iterations):
                    self.tiled_camera_sensor.update(
                        self.state,
                        self.camera_transforms,
                        self.camera_rays,
                        color_image=self.color_image if out_color else None,
                        depth_image=self.depth_image if out_depth else None,
                    )

        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
        for out_color, out_depth in [(True, True), (True, False), (False, True)]:
            for _ in range(iterations):
                self.tiled_camera_sensor.update(
                    self.state,
                    self.camera_transforms,
                    self.camera_rays,
                    color_image=self.color_image if out_color else None,
                    depth_image=self.depth_image if out_depth else None,
                )

    @nice_name("Rendering (Pixel)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_pixel_priority_color_depth(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                color_image=self.color_image,
                depth_image=self.depth_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Pixel) (Color Only)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_pixel_priority_color_only(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                color_image=self.color_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Pixel) (Depth Only)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_pixel_priority_depth_only(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                depth_image=self.depth_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Tiled)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or not run_tiled_camera_benchmarks())
    def time_rendering_tiled_color_depth(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.TILED
        self.tiled_camera_sensor.render_config.tile_width = 8
        self.tiled_camera_sensor.render_config.tile_height = 8
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                color_image=self.color_image,
                depth_image=self.depth_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Tiled) (Color Only)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or not run_tiled_camera_benchmarks())
    def time_rendering_tiled_color_only(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.TILED
        self.tiled_camera_sensor.render_config.tile_width = 8
        self.tiled_camera_sensor.render_config.tile_height = 8
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                color_image=self.color_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Tiled) (Depth Only)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or not run_tiled_camera_benchmarks())
    def time_rendering_tiled_depth_only(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.TILED
        self.tiled_camera_sensor.render_config.tile_width = 8
        self.tiled_camera_sensor.render_config.tile_height = 8
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                depth_image=self.depth_image,
            )
        wp.synchronize()


ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_SEKTION_CABINET_FOLDER = "assets/urdf/sektion_cabinet_model"

# Renderer configuration applied to every scene benchmark. These are benchmark
# settings, not library defaults: SAH builds a higher-quality shape BVH
# (slower one-time build, faster ray traversal), and rendering uses the tiled
# traversal order with 8x8 pixel tiles.
BVH_CONSTRUCTOR = "sah"
KERNEL_BLOCK_DIM = 64
RENDER_ORDER = SensorTiledCamera.RenderOrder.TILED
RENDER_TILE_WIDTH = 8
RENDER_TILE_HEIGHT = 8

# Arm "ready" pose facing the cabinet drawer, from Isaac Lab's
# FrankaCabinetDirectEnvCfg (panda_joint* values mapped to the FR3 URDF names).
_FRANKA_CABINET_ARM_POSE = {
    "fr3_joint1": 1.157,
    "fr3_joint2": -1.066,
    "fr3_joint3": -0.155,
    "fr3_joint4": -2.239,
    "fr3_joint5": -1.841,
    "fr3_joint6": 1.003,
    "fr3_joint7": 0.469,
    "fr3_finger_joint1": 0.035,
    "fr3_finger_joint2": 0.035,
}

# Nominal standing configuration, from Isaac Lab's ANYmal locomotion configs.
_ANYMAL_STANDING_POSE = {
    "LF_HAA": 0.0,
    "LF_HFE": 0.4,
    "LF_KFE": -0.8,
    "RF_HAA": 0.0,
    "RF_HFE": 0.4,
    "RF_KFE": -0.8,
    "LH_HAA": 0.0,
    "LH_HFE": -0.4,
    "LH_KFE": 0.8,
    "RH_HAA": 0.0,
    "RH_HFE": -0.4,
    "RH_KFE": 0.8,
}

_MANY_SHAPES_COUNT = 256
_MANY_SHAPES_COLORS = (
    (0.27, 0.47, 0.67),
    (0.40, 0.80, 0.93),
    (0.13, 0.53, 0.20),
    (0.80, 0.73, 0.27),
    (0.93, 0.40, 0.47),
)


def _set_joint_positions(builder: newton.ModelBuilder, joint_positions: dict[str, float]) -> None:
    """Set initial positions of single-DOF joints by joint name.

    Importers produce hierarchical labels like ``"fr3/fr3_joint1"``, so names
    are matched against the last path component of each joint label.
    """
    remaining = dict(joint_positions)
    for joint_index, label in enumerate(builder.joint_label):
        value = remaining.pop(label.rsplit("/", 1)[-1], None)
        if value is not None:
            builder.joint_q[builder.joint_q_start[joint_index]] = value
    if remaining:
        raise ValueError(f"Joints not found in builder: {sorted(remaining)}")


def _disable_collision_handling(builder: newton.ModelBuilder) -> None:
    """Strip collision flags so finalization skips collision-pair generation.

    Rendering does not consume collision state, and pair enumeration dominates
    model build time at benchmark world counts.
    """
    collide = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    builder.shape_flags = [int(flags) & ~collide for flags in builder.shape_flags]
    builder.shape_collision_filter_pairs = []


def _build_franka() -> newton.ModelBuilder:
    """A fixed-base Franka FR3 arm (same content as the fast tiled camera benchmark)."""
    builder = newton.ModelBuilder()
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        floating=False,
    )
    return builder


def _build_quadruped() -> newton.ModelBuilder:
    """An ANYmal D quadruped standing on the ground."""
    builder = newton.ModelBuilder()
    asset_path = newton.utils.download_asset("anybotics_anymal_d")
    builder.add_usd(
        str(asset_path / "usd" / "anymal_d.usda"),
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    builder.joint_q[:3] = [0.0, 0.0, 0.62]
    _set_joint_positions(builder, _ANYMAL_STANDING_POSE)
    return builder


def _build_franka_cabinet() -> newton.ModelBuilder:
    """Isaac Lab's Franka cabinet (open-drawer) scene.

    Layout follows FrankaCabinetDirectEnvCfg: the Sektion cabinet sits at the
    origin raised 0.4 m, with the arm 1 m away rotated to face the drawers.
    The cabinet URDF comes from IsaacGymEnvs (Isaac Lab itself references a
    Nucleus-hosted USD of the same asset).
    """
    cabinet_folder = newton.examples.download_external_git_folder(
        ISAACGYM_ENVS_REPO_URL, ISAACGYM_SEKTION_CABINET_FOLDER
    )
    builder = newton.ModelBuilder()
    builder.add_urdf(
        cabinet_folder / "urdf" / "sektion_cabinet_2.urdf",
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()),
        floating=False,
    )
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi)),
        floating=False,
    )
    _set_joint_positions(builder, _FRANKA_CABINET_ARM_POSE)
    return builder


def _build_shapes_256() -> newton.ModelBuilder:
    """A 16x16 grid of primitive shapes (spheres, boxes, capsules, cylinders)."""
    builder = newton.ModelBuilder()
    rng = random.Random(1234)
    grid = math.isqrt(_MANY_SHAPES_COUNT)
    spacing = 0.8
    for index in range(_MANY_SHAPES_COUNT):
        row, col = divmod(index, grid)
        position = wp.vec3(
            (col - (grid - 1) / 2.0) * spacing,
            (row - (grid - 1) / 2.0) * spacing,
            0.3 + 0.4 * rng.random(),
        )
        orientation = wp.quat_rpy(
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-math.pi, math.pi),
        )
        body = builder.add_body(xform=wp.transform(position, orientation))
        color = rng.choice(_MANY_SHAPES_COLORS)
        kind = index % 4
        if kind == 0:
            builder.add_shape_sphere(body, radius=0.18, color=color)
        elif kind == 1:
            builder.add_shape_box(body, hx=0.16, hy=0.16, hz=0.16, color=color)
        elif kind == 2:
            builder.add_shape_capsule(body, radius=0.1, half_height=0.15, color=color)
        else:
            builder.add_shape_cylinder(body, radius=0.14, half_height=0.15, color=color)
    return builder


@dataclass(frozen=True)
class ScenePreset:
    """A benchmark scene: one world's worth of content plus a camera pose."""

    build: Callable[[], newton.ModelBuilder]
    """Builds a ModelBuilder holding a single world of scene content."""

    camera_eye: tuple[float, float, float]
    """Camera position [m]."""

    camera_target: tuple[float, float, float]
    """Point the camera looks at [m]."""

    light_direction: tuple[float, float, float] | None = None
    """Directional-light travel direction; ``None`` uses the sensor default."""


SCENES: dict[str, ScenePreset] = {
    "franka": ScenePreset(_build_franka, camera_eye=(2.4, 0.0, 0.8), camera_target=(0.0, 0.0, 0.4)),
    "quadruped": ScenePreset(_build_quadruped, camera_eye=(2.2, 2.2, 1.1), camera_target=(0.0, 0.0, 0.45)),
    # Light comes from the camera-facing upper-front octant so it lands on the
    # cabinet drawers (which face +x) and top, both visible to the camera.
    "franka_cabinet": ScenePreset(
        _build_franka_cabinet,
        camera_eye=(2.4, 1.8, 1.3),
        camera_target=(0.4, 0.0, 0.5),
        light_direction=(-1.0, -0.4, -0.45),
    ),
    "shapes_256": ScenePreset(_build_shapes_256, camera_eye=(9.0, 9.0, 6.0), camera_target=(0.0, 0.0, 0.0)),
}


def _look_at_transform(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> wp.transformf:
    """Camera-to-world transform at *eye* looking toward *target* (camera space: -Z forward, +Y up)."""
    eye_v = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye_v
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    rotation = np.stack([right, camera_up, -forward], axis=1)
    return wp.transformf(wp.vec3f(*eye), wp.quat_from_matrix(wp.mat33f(rotation.flatten())))


class _TiledCameraSceneRig:
    """A scene replicated across worlds with a tiled camera sensor ready to render."""

    def __init__(self, preset: ScenePreset, world_count: int, resolution: int, camera_fov_deg: float = 45.0):
        world = preset.build()
        _disable_collision_handling(world)

        scene = newton.ModelBuilder()
        scene.default_bvh_cfg.mesh_constructor = "cubql"
        scene.replicate(world, world_count)
        scene.add_ground_plane(color=(0.6, 0.6, 0.6))

        self.model = scene.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        light_direction = wp.vec3f(*preset.light_direction) if preset.light_direction is not None else None
        self.sensor = SensorTiledCamera(model=self.model)
        self.sensor.render_config.render_order = RENDER_ORDER
        self.sensor.render_config.tile_width = RENDER_TILE_WIDTH
        self.sensor.render_config.tile_height = RENDER_TILE_HEIGHT
        self.sensor.utils.create_default_light(enable_shadows=True, direction=light_direction)
        self.sensor.utils.assign_checkerboard_material(shape_indices=np.arange(self.model.shape_count))

        camera = _look_at_transform(preset.camera_eye, preset.camera_target)
        self.camera_transforms = wp.array([[camera] * world_count], dtype=wp.transformf)
        self.camera_rays = self.sensor.utils.compute_camera_rays_pinhole(
            resolution, resolution, camera_fovs=math.radians(camera_fov_deg)
        )
        self.color_image = self.sensor.utils.create_color_image_output(resolution, resolution)
        self.depth_image = self.sensor.utils.create_depth_image_output(resolution, resolution)

        self.model.bvh_build_shapes(self.state, bvh_constructor=BVH_CONSTRUCTOR)
        self.model.bvh_build_particles(self.state, bvh_constructor=BVH_CONSTRUCTOR)
        self.sensor.sync_transforms(self.state)

    def render(self, color: bool = True, depth: bool = True):
        self.sensor.update(
            self.state,
            self.camera_transforms,
            self.camera_rays,
            color_image=self.color_image if color else None,
            depth_image=self.depth_image if depth else None,
            kernel_block_dim=KERNEL_BLOCK_DIM,
        )


class _SceneBenchmark:
    """Shared ASV harness; subclasses pick a scene from :data:`SCENES` and their params."""

    param_names = ["resolution", "world_count", "iterations"]
    scene: str

    def setup(self, resolution: int, world_count: int, iterations: int):
        self.rig = _TiledCameraSceneRig(SCENES[self.scene], world_count, resolution)
        # Compile and warm the render kernels for every output combination measured below.
        for color, depth in ((True, True), (True, False), (False, True)):
            self.rig.render(color=color, depth=depth)
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_render_color_depth(self, resolution: int, world_count: int, iterations: int):
        for _ in range(iterations):
            self.rig.render(color=True, depth=True)
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_render_color_only(self, resolution: int, world_count: int, iterations: int):
        for _ in range(iterations):
            self.rig.render(color=True, depth=False)
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_render_depth_only(self, resolution: int, world_count: int, iterations: int):
        for _ in range(iterations):
            self.rig.render(color=False, depth=True)
        wp.synchronize()


class TiledCameraFranka(_SceneBenchmark):
    scene = "franka"
    params = ([64], [4096], [50])


class TiledCameraQuadruped(_SceneBenchmark):
    scene = "quadruped"
    params = ([64], [4096], [50])


class TiledCameraFrankaCabinet(_SceneBenchmark):
    scene = "franka_cabinet"
    params = ([64], [4096], [50])


class TiledCameraShapes256(_SceneBenchmark):
    scene = "shapes_256"
    params = ([64], [4096], [50])


PREVIEW_WORLD_COUNTS = (1, 16)


def write_preview_images(scene_names: list[str], output_dir: Path, image_size: int = 1024) -> list[Path]:
    """Write a PNG preview of each scene for every entry in :data:`PREVIEW_WORLD_COUNTS`.

    Single-world previews render at ``image_size``; multi-world previews tile a
    square grid of per-world renders into the same overall image size.
    """
    from PIL import Image  # noqa: PLC0415  # only needed for previews (part of the examples extra)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name in scene_names:
        for world_count in PREVIEW_WORLD_COUNTS:
            worlds_per_row = math.isqrt(world_count)
            rig = _TiledCameraSceneRig(SCENES[name], world_count, image_size // worlds_per_row)
            rig.render(color=True, depth=False)
            rgba = rig.sensor.utils.flatten_color_image_to_rgba(rig.color_image, worlds_per_row=worlds_per_row)
            path = output_dir / f"{name}_{world_count}_world{'s' if world_count > 1 else ''}.png"
            # Drop alpha: background pixels have alpha 0 and would turn transparent.
            Image.fromarray(rgba.numpy()[..., :3]).save(path)
            written.append(path)
    return written


def print_fps(name: str, duration: float, resolution: int, world_count: int, iterations: int):
    camera_count = 1

    title = f"{name}"
    if iterations > 1:
        title += " average"

    average = f"{duration * 1000.0 / iterations:.2f} ms"

    fps = f"({(1.0 / (duration / iterations) * (world_count * camera_count)):,.2f} fps)"
    print(f"{title} {'.' * (50 - len(title) - len(average))} {average} {fps if iterations > 1 else ''}")


def print_fps_results(results: dict[tuple[str, tuple[int, int, int]], float]):
    print("")
    print("=== Benchmark Results (FPS) ===")
    for (method_name, params), avg in results.items():
        print_fps(NICE_NAMES.get(method_name, method_name), avg, *params)
    print("")


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastSensorTiledCamera": FastSensorTiledCamera,
        "TiledCameraFranka": TiledCameraFranka,
        "TiledCameraQuadruped": TiledCameraQuadruped,
        "TiledCameraFrankaCabinet": TiledCameraFrankaCabinet,
        "TiledCameraShapes256": TiledCameraShapes256,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    parser.add_argument(
        "--include-tiled",
        action="store_true",
        help=f"Run the tiled render-order benchmarks. For ASV, set {ASV_RUN_TILED_CAMERA_BENCHMARKS_ENV_VAR}=1.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write PNG previews of each scene (single world and 4x4 world grid) instead of benchmarking.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("tiled_camera_previews"),
        help="Directory for preview images.",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=1024,
        help="Preview image width and height [px].",
    )
    args = parser.parse_known_args()[0]

    if args.include_tiled:
        os.environ[ASV_RUN_TILED_CAMERA_BENCHMARKS_ENV_VAR] = "1"

    if args.preview:
        for path in write_preview_images(sorted(SCENES), args.preview_dir, image_size=args.preview_size):
            print(f"Wrote {path}")
    else:
        for key in args.bench or benchmark_list.keys():
            result = run_benchmark(benchmark_list[key])
            print_fps_results(result)
