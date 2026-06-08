# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate temporary PR-review artifacts for texture projection modes.

This script renders cubic and triplanar checkerboard projection comparisons for
``SensorTiledCamera``. The generated images are intended as PR review evidence
and can be removed after the projection-mode change is accepted.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
BENCHMARK_WIDTH = 1024
BENCHMARK_HEIGHT = 1024
FLOOR_Z = -0.02
REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class CameraSetup:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    vertical_fov_deg: float


@dataclass(frozen=True)
class BenchCase:
    name: str
    model: newton.Model
    state: newton.State
    camera: CameraSetup


def camera_transform(position, target, up=(0.0, 0.0, 1.0)):
    position = np.asarray(position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    direction = target - position
    direction /= np.linalg.norm(direction)
    right = np.cross(direction, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, direction)
    rotation = np.column_stack([right, camera_up, -direction]).astype(np.float32)
    return wp.transformf(wp.vec3f(*position), wp.quat_from_matrix(wp.mat33f(rotation)))


def packed_rgba_to_image(array: np.ndarray) -> Any:
    from PIL import Image

    rgba = array.view(np.uint8).reshape(*array.shape, 4)
    return Image.fromarray(rgba, mode="RGBA")


def load_bunny_mesh(scale: float = 0.82) -> newton.Mesh:
    from pxr import Usd, UsdGeom

    bunny_filename = REPO_ROOT / "newton" / "examples" / "assets" / "bunny.usd"
    stage = Usd.Stage.Open(str(bunny_filename))
    usd_geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bunny"))

    vertices = np.array(usd_geom.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)

    # The asset is Y-up. Convert to Newton's Z-up frame, center horizontally,
    # and leave it UV-less so projection mode differences are visible.
    zup = np.empty_like(vertices)
    zup[:, 0] = vertices[:, 0]
    zup[:, 1] = vertices[:, 2]
    zup[:, 2] = vertices[:, 1]
    zup[:, 0] -= 0.5 * (zup[:, 0].min() + zup[:, 0].max())
    zup[:, 1] -= 0.5 * (zup[:, 1].min() + zup[:, 1].max())
    zup[:, 2] -= zup[:, 2].min()
    zup *= scale

    # Swapping USD Y/Z axes changes handedness, so flip triangle winding.
    indices = indices[:, [0, 2, 1]].reshape(-1)
    return newton.Mesh(zup, indices, compute_inertia=False)


def finalize_builder(builder: newton.ModelBuilder) -> tuple[newton.Model, newton.State]:
    model = builder.finalize()
    state = model.state()
    model.bvh_build_shapes(state)
    model.bvh_build_particles(state)
    return model, state


def build_primitive_scene() -> tuple[newton.Model, newton.State, CameraSetup]:
    builder = newton.ModelBuilder()

    box_body = builder.add_body(xform=wp.transform(p=wp.vec3(-1.7, 0.0, 0.68), q=wp.quat_identity()))
    builder.add_shape_box(box_body, hx=0.68, hy=0.68, hz=0.68, color=(1.0, 1.0, 1.0))

    sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.73), q=wp.quat_identity()))
    builder.add_shape_sphere(sphere_body, radius=0.73, color=(1.0, 1.0, 1.0))

    cylinder_body = builder.add_body(xform=wp.transform(p=wp.vec3(1.35, 0.0, 0.73), q=wp.quat_identity()))
    builder.add_shape_cylinder(cylinder_body, radius=0.52, half_height=0.73, color=(1.0, 1.0, 1.0))

    camera = CameraSetup(position=(3.2, -4.9, 2.25), target=(-0.15, 0.0, 0.72), vertical_fov_deg=42.0)
    return (*finalize_builder(builder), camera)


def build_bunny_scene() -> tuple[newton.Model, newton.State, CameraSetup]:
    builder = newton.ModelBuilder()

    bunny_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, FLOOR_Z), q=wp.quat_identity()))
    builder.add_shape_mesh(bunny_body, mesh=load_bunny_mesh(), color=(1.0, 1.0, 1.0))

    camera = CameraSetup(position=(2.25, -3.9, 1.7), target=(0.0, 0.0, 0.68), vertical_fov_deg=36.0)
    return (*finalize_builder(builder), camera)


def render_mode(
    model: newton.Model,
    state: newton.State,
    camera: CameraSetup,
    mode: int,
    *,
    width: int,
    height: int,
) -> tuple[Any, Any]:
    render_config = SensorTiledCamera.RenderConfig(texture_projection_mode=mode)
    sensor = SensorTiledCamera(model=model, config=render_config)
    sensor.utils.assign_checkerboard_material_to_all_shapes(resolution=64, checker_size=8)
    sensor.utils.create_default_light(enable_shadows=True)

    camera_rays = sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(camera.vertical_fov_deg))
    camera_transforms = wp.array([[camera_transform(camera.position, camera.target)]], dtype=wp.transformf)

    albedo = sensor.utils.create_albedo_image_output(width, height, camera_count=1)
    color = sensor.utils.create_color_image_output(width, height, camera_count=1)

    sensor.update(state, camera_transforms, camera_rays, albedo_image=albedo, color_image=color)

    albedo_image = packed_rgba_to_image(albedo.numpy()[0, 0]).convert("RGB")
    shaded_image = packed_rgba_to_image(color.numpy()[0, 0]).convert("RGB")
    return albedo_image, shaded_image


def make_contact_sheet(images: list[tuple[str, Any]], filename: Path, width: int, height: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    label_h = 24
    cols = 2
    rows = math.ceil(len(images) / cols)
    tile_w = width
    tile_h = height + label_h
    sheet = Image.new("RGB", (tile_w * cols, tile_h * rows), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)

    for index, (label, image) in enumerate(images):
        x = (index % cols) * tile_w
        y = (index // cols) * tile_h
        draw.rectangle((x, y, x + tile_w, y + label_h), fill=(20, 20, 20))
        draw.text((x + 8, y + 6), label, fill=(235, 235, 235), font=font)
        sheet.paste(image, (x, y + label_h))

    sheet.save(filename)


def render_scene(
    scene_name: str,
    model: newton.Model,
    state: newton.State,
    camera: CameraSetup,
    *,
    width: int,
    height: int,
) -> list[tuple[str, Any]]:
    cubic_albedo, cubic_shaded = render_mode(
        model, state, camera, SensorTiledCamera.TextureProjectionMode.CUBIC, width=width, height=height
    )
    triplanar_albedo, triplanar_shaded = render_mode(
        model, state, camera, SensorTiledCamera.TextureProjectionMode.TRIPLANAR, width=width, height=height
    )

    return [
        (f"{scene_name}: cubic albedo", cubic_albedo),
        (f"{scene_name}: triplanar albedo", triplanar_albedo),
        (f"{scene_name}: cubic shaded", cubic_shaded),
        (f"{scene_name}: triplanar shaded", triplanar_shaded),
    ]


def generate_images(output_dir: Path, width: int, height: int) -> None:
    primitive_images = render_scene("primitives", *build_primitive_scene(), width=width, height=height)
    bunny_images = render_scene("bunny", *build_bunny_scene(), width=width, height=height)

    make_contact_sheet(primitive_images, output_dir / "projection_modes_primitives.png", width, height)
    make_contact_sheet(bunny_images, output_dir / "projection_modes_bunny.png", width, height)
    make_contact_sheet(
        primitive_images + bunny_images, output_dir / "projection_modes_contact_sheet.png", width, height
    )


def make_benchmark_renderer(case: BenchCase, mode: int):
    render_config = SensorTiledCamera.RenderConfig(texture_projection_mode=mode)
    sensor = SensorTiledCamera(model=case.model, config=render_config)
    sensor.utils.assign_checkerboard_material_to_all_shapes(resolution=64, checker_size=8)

    camera_rays = sensor.utils.compute_pinhole_camera_rays(
        BENCHMARK_WIDTH, BENCHMARK_HEIGHT, math.radians(case.camera.vertical_fov_deg)
    )
    camera_transforms = wp.array([[camera_transform(case.camera.position, case.camera.target)]], dtype=wp.transformf)
    albedo = sensor.utils.create_albedo_image_output(BENCHMARK_WIDTH, BENCHMARK_HEIGHT, camera_count=1)
    return sensor, camera_transforms, camera_rays, albedo


def benchmark_mode(case: BenchCase, mode: int, warmup: int, iterations: int) -> float:
    sensor, camera_transforms, camera_rays, albedo = make_benchmark_renderer(case, mode)

    for _ in range(warmup):
        sensor.update(case.state, camera_transforms, camera_rays, albedo_image=albedo)
    wp.synchronize_device()

    start = time.perf_counter()
    for _ in range(iterations):
        sensor.update(case.state, camera_transforms, camera_rays, albedo_image=albedo)
    wp.synchronize_device()
    return (time.perf_counter() - start) * 1000.0 / iterations


def run_benchmark(warmup: int, iterations: int) -> str:
    primitive_model, primitive_state, primitive_camera = build_primitive_scene()
    bunny_model, bunny_state, bunny_camera = build_bunny_scene()
    cases = [
        BenchCase("primitives", primitive_model, primitive_state, primitive_camera),
        BenchCase("uvless_bunny_mesh", bunny_model, bunny_state, bunny_camera),
    ]
    modes = [
        ("cubic", SensorTiledCamera.TextureProjectionMode.CUBIC),
        ("triplanar", SensorTiledCamera.TextureProjectionMode.TRIPLANAR),
    ]

    rows = []
    for case in cases:
        cubic_ms = None
        for mode_name, mode in modes:
            mean_ms = benchmark_mode(case, mode, warmup, iterations)
            if mode_name == "cubic":
                cubic_ms = mean_ms
            rel = mean_ms / cubic_ms if cubic_ms is not None else 1.0
            rows.append((case.name, mode_name, mean_ms, rel))

    lines = [
        f"Resolution: {BENCHMARK_WIDTH}x{BENCHMARK_HEIGHT}",
        f"Warmup/update iterations: {warmup}/{iterations}",
        "",
        f"{'scene':<20} {'mode':<10} {'mean_ms':>10} {'relative':>10}",
    ]
    for scene, mode_name, mean_ms, rel in rows:
        lines.append(f"{scene:<20} {mode_name:<10} {mean_ms:10.3f} {rel:10.3f}x")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated images and benchmark output.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Rendered image width in pixels.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Rendered image height in pixels.")
    parser.add_argument("--skip-images", action="store_true", help="Only run the benchmark.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark cubic and triplanar projection modes.")
    parser.add_argument("--benchmark-warmup", type=int, default=20, help="Warmup updates before benchmark timing.")
    parser.add_argument("--benchmark-iterations", type=int, default=500, help="Timed benchmark updates per mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_images:
        generate_images(args.output_dir, args.width, args.height)

    if args.benchmark:
        benchmark = run_benchmark(args.benchmark_warmup, args.benchmark_iterations)
        print(benchmark)
        (args.output_dir / "projection_mode_benchmark.txt").write_text(benchmark + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
