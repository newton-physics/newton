# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ViewerGL sorted transparency against weighted OIT.

The benchmark mixes procedural scenes with repo-local USD example assets so the
transparent workloads include realistic cloth and triangle-mesh topology. PNG
output is written with the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

if os.name != "nt":
    os.environ.setdefault("PYGLET_HEADLESS", "1")

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd
from newton.viewer import ViewerGL

MODE_SORTED = "sorted"
MODE_WEIGHTED_OIT = "weighted_oit"
SCENE_CLOTH_SHIRT = "cloth_shirt"
SCENE_SHAPE_CLOUD = "shape_cloud"
SCENE_MESH_SPHERES = "mesh_spheres"
SCENE_BUNNY_MESHES = "bunny_meshes"
SCENE_ALL = (SCENE_CLOTH_SHIRT, SCENE_SHAPE_CLOUD, SCENE_MESH_SPHERES, SCENE_BUNNY_MESHES)
SCENE_ALIASES = {"tri_surface": SCENE_CLOTH_SHIRT}


@dataclass
class Scene:
    name: str
    model: newton.Model
    state: newton.State
    camera_pos: wp.vec3
    camera_target: wp.vec3
    notes: str


@dataclass
class ModeResult:
    scene: str
    mode: str
    frame_times_ms: list[float]
    screenshot: str | None

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.frame_times_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.frame_times_ms)

    @property
    def min_ms(self) -> float:
        return min(self.frame_times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.frame_times_ms)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _save_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 PNG without requiring Pillow."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (height, width, 3), got {image.shape}.")

    image = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    height, width, _channels = image.shape
    rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(rows))
    png += _png_chunk(b"IEND", b"")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _save_comparison(path: Path, weighted: np.ndarray, sorted_image: np.ndarray) -> None:
    """Save weighted OIT, sorted, and amplified absolute difference panels."""
    if weighted.shape != sorted_image.shape:
        raise ValueError(f"Comparison images must have matching shape, got {weighted.shape} and {sorted_image.shape}.")

    diff = np.abs(weighted.astype(np.int16) - sorted_image.astype(np.int16)).astype(np.uint8)
    diff = np.clip(diff.astype(np.int16) * 4, 0, 255).astype(np.uint8)
    height = weighted.shape[0]
    separator = np.full((height, 6, 3), 255, dtype=np.uint8)
    canvas = np.concatenate((weighted, separator, sorted_image, separator, diff), axis=1)
    _save_png(path, canvas)


def _set_camera(viewer: ViewerGL, pos: wp.vec3, target: wp.vec3) -> None:
    viewer.set_camera(pos=pos, pitch=0.0, yaw=-180.0)
    viewer.camera.look_at(target)
    viewer.camera.fov = 45.0


def _render_frame(viewer: ViewerGL, state: newton.State, frame_index: int) -> None:
    viewer.begin_frame(frame_index / 60.0)
    viewer.log_state(state)
    viewer.end_frame()
    viewer.renderer.gl.glFinish()


def _capture_frame(viewer: ViewerGL) -> np.ndarray:
    frame = viewer.get_frame()
    return np.ascontiguousarray(frame.numpy())


def _copy_tri_opacity(model: newton.Model, opacities: np.ndarray) -> None:
    if model.tri_count == 0 or model.tri_opacity is None:
        return
    wp.copy(model.tri_opacity, wp.array(opacities.astype(np.float32), dtype=wp.float32, device=model.device))


def _load_usd_mesh(asset_name: str, prim_path: str) -> newton.Mesh:
    from pxr import Usd

    usd_stage = Usd.Stage.Open(newton.examples.get_asset(asset_name))
    return newton.usd.get_mesh(usd_stage.GetPrimAtPath(prim_path))


def _center_vertices(vertices: np.ndarray) -> np.ndarray:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return vertices - 0.5 * (lower + upper)


def _cloth_rotation(index: int) -> wp.quat:
    yaw = -0.45 + 0.30 * index
    pitch = -0.08 + 0.05 * (index % 3)
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw) * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), pitch)


def _build_cloth_shirt_scene(instance_count: int, opacity: float, scale: float, pattern: str) -> Scene:
    builder = newton.ModelBuilder()
    shirt_mesh = _load_usd_mesh("unisex_shirt.usd", "/root/shirt")
    vertices = _center_vertices(np.asarray(shirt_mesh.vertices, dtype=np.float32)).tolist()
    indices = np.asarray(shirt_mesh.indices, dtype=np.int32)

    columns = max(1, int(np.ceil(np.sqrt(instance_count))))
    rows = max(1, int(np.ceil(instance_count / columns)))
    for index in range(instance_count):
        column = index % columns
        row = index // columns
        x = (column - 0.5 * (columns - 1)) * 0.70
        y = (row - 0.5 * (rows - 1)) * 0.26
        z = 1.25 + 0.05 * (index % 2)
        builder.add_cloth_mesh(
            pos=wp.vec3(float(x), float(y), float(z)),
            rot=_cloth_rotation(index),
            scale=scale,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=0.02,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.003,
            opacity=opacity,
            label=f"transparent_unisex_shirt_{index}",
        )

    model = builder.finalize()
    if pattern == "bands" and model.tri_count:
        bands = np.array([0.26, 0.38, 0.50, 0.62], dtype=np.float32)
        _copy_tri_opacity(model, bands[np.arange(model.tri_count) % len(bands)])

    state = model.state()
    asset_tri_count = len(indices) // 3
    return Scene(
        name=f"cloth_shirt_{instance_count}",
        model=model,
        state=state,
        camera_pos=wp.vec3(1.25, -2.60, 2.75),
        camera_target=wp.vec3(0.0, 0.0, 1.25),
        notes=(
            f"{instance_count} transparent unisex_shirt.usd cloth meshes "
            f"({asset_tri_count} triangles each, {model.tri_count} rendered cloth triangles total)"
        ),
    )


def _scene_color(index: int) -> tuple[float, float, float]:
    palette = (
        (0.88, 0.25, 0.21),
        (0.12, 0.48, 0.92),
        (0.18, 0.66, 0.39),
        (0.96, 0.73, 0.18),
        (0.62, 0.34, 0.83),
        (0.94, 0.45, 0.16),
    )
    return palette[index % len(palette)]


def _grid_position(index: int, count: int, spacing: float) -> tuple[float, float, float]:
    columns = max(1, int(np.ceil(np.sqrt(count))))
    rows = max(1, int(np.ceil(count / columns)))
    x_index = index % columns
    y_index = index // columns
    x = (x_index - 0.5 * (columns - 1)) * spacing
    y = (y_index - 0.5 * (rows - 1)) * spacing * 0.7
    z = 0.45 + 0.18 * np.sin(index * 0.63) + 0.12 * (index % 5)
    return float(x), float(y), float(z)


def _shape_rotation(index: int) -> wp.quat:
    axis = np.array([0.35 + 0.11 * (index % 3), 0.7, 0.45], dtype=np.float32)
    axis /= np.linalg.norm(axis)
    return wp.quat_from_axis_angle(wp.vec3(float(axis[0]), float(axis[1]), float(axis[2])), 0.19 * index)


def _build_shape_cloud_scene(instance_count: int, opacity: float) -> Scene:
    builder = newton.ModelBuilder()
    spacing = 0.58

    for index in range(instance_count):
        x, y, z = _grid_position(index, instance_count, spacing)
        x += 0.08 * np.sin(index * 1.7)
        y += 0.16 * np.cos(index * 0.41)
        xform = wp.transform(wp.vec3(x, y, z), _shape_rotation(index))
        color = _scene_color(index)
        shape_kind = index % 5

        if shape_kind == 0:
            builder.add_shape_box(-1, xform=xform, hx=0.20, hy=0.12, hz=0.28, color=color, opacity=opacity)
        elif shape_kind == 1:
            builder.add_shape_sphere(-1, xform=xform, radius=0.20, color=color, opacity=opacity)
        elif shape_kind == 2:
            builder.add_shape_capsule(-1, xform=xform, radius=0.11, half_height=0.24, color=color, opacity=opacity)
        elif shape_kind == 3:
            builder.add_shape_cylinder(-1, xform=xform, radius=0.15, half_height=0.27, color=color, opacity=opacity)
        else:
            builder.add_shape_ellipsoid(-1, xform=xform, rx=0.24, ry=0.14, rz=0.19, color=color, opacity=opacity)

    builder.add_ground_plane(height=-0.08)
    model = builder.finalize()
    state = model.state()
    return Scene(
        name=f"shape_cloud_{instance_count}",
        model=model,
        state=state,
        camera_pos=wp.vec3(4.6, -7.2, 3.0),
        camera_target=wp.vec3(0.0, 0.0, 0.9),
        notes=f"{instance_count} generated transparent primitive shapes plus an opaque ground plane",
    )


def _build_mesh_spheres_scene(instance_count: int, opacity: float, latitude: int, longitude: int) -> Scene:
    builder = newton.ModelBuilder()
    mesh = newton.Mesh.create_sphere(
        radius=0.22,
        num_latitudes=latitude,
        num_longitudes=longitude,
        compute_inertia=False,
    )
    spacing = 0.62

    for index in range(instance_count):
        x, y, z = _grid_position(index, instance_count, spacing)
        z += 0.35
        xform = wp.transform(wp.vec3(x, y, z), _shape_rotation(index))
        builder.add_shape_mesh(-1, xform=xform, mesh=mesh, color=_scene_color(index + 2), opacity=opacity)

    builder.add_ground_plane(height=-0.05)
    model = builder.finalize()
    state = model.state()
    tri_count = len(mesh.indices) // 3
    return Scene(
        name=f"mesh_spheres_{instance_count}",
        model=model,
        state=state,
        camera_pos=wp.vec3(4.8, -7.4, 3.2),
        camera_target=wp.vec3(0.0, 0.0, 1.0),
        notes=f"{instance_count} generated transparent sphere mesh instances ({tri_count} triangles each)",
    )


def _bunny_rotation(index: int) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.37 * index) * wp.quat_from_axis_angle(
        wp.vec3(1.0, 0.0, 0.0), -0.10 + 0.04 * (index % 5)
    )


def _build_bunny_meshes_scene(instance_count: int, opacity: float, scale: float) -> Scene:
    builder = newton.ModelBuilder()
    source_mesh = _load_usd_mesh("bunny.usd", "/root/bunny")
    vertices = _center_vertices(np.asarray(source_mesh.vertices, dtype=np.float32))
    bunny_mesh = newton.Mesh(
        vertices,
        np.asarray(source_mesh.indices, dtype=np.int32),
        compute_inertia=False,
        color=(0.72, 0.52, 0.90),
        opacity=opacity,
    )

    spacing = 0.56
    for index in range(instance_count):
        x, y, z = _grid_position(index, instance_count, spacing)
        z = 0.92 + 0.12 * (index % 4) + 0.05 * np.sin(index * 0.71)
        xform = wp.transform(wp.vec3(x, y, z), _bunny_rotation(index))
        color = _scene_color(index + 4)
        builder.add_shape_mesh(
            -1, xform=xform, mesh=bunny_mesh, scale=(scale, scale, scale), color=color, opacity=opacity
        )

    builder.add_ground_plane(height=-0.08)
    model = builder.finalize()
    state = model.state()
    tri_count = len(bunny_mesh.indices) // 3
    return Scene(
        name=f"bunny_meshes_{instance_count}",
        model=model,
        state=state,
        camera_pos=wp.vec3(4.4, -6.7, 2.9),
        camera_target=wp.vec3(0.0, 0.0, 0.95),
        notes=(
            f"{instance_count} transparent bunny.usd mesh instances "
            f"({tri_count} triangles each, {tri_count * instance_count} mesh triangles total)"
        ),
    )


def _build_scene(name: str, args: argparse.Namespace) -> Scene:
    if name == SCENE_CLOTH_SHIRT:
        return _build_cloth_shirt_scene(
            args.cloth_count,
            args.opacity,
            args.cloth_scale,
            args.cloth_opacity_pattern,
        )
    if name == SCENE_SHAPE_CLOUD:
        return _build_shape_cloud_scene(args.shape_count, args.opacity)
    if name == SCENE_MESH_SPHERES:
        return _build_mesh_spheres_scene(args.mesh_count, args.opacity, args.mesh_latitude, args.mesh_longitude)
    if name == SCENE_BUNNY_MESHES:
        return _build_bunny_meshes_scene(args.bunny_count, args.opacity, args.bunny_scale)
    raise ValueError(f"Unknown scene '{name}'. Expected one of: {', '.join(SCENE_ALL)}.")


def _create_viewer(args: argparse.Namespace) -> ViewerGL:
    viewer = ViewerGL(width=args.width, height=args.height, headless=True)
    if not hasattr(viewer.renderer, "enable_weighted_transparency"):
        viewer.close()
        raise RuntimeError("This benchmark requires the ViewerGL branch with renderer.enable_weighted_transparency.")

    viewer.renderer.draw_shadows = False
    viewer.renderer.draw_edges = False
    viewer.renderer.draw_fps = False
    if not args.use_msaa:
        viewer.renderer.msaa_samples = 0
    return viewer


def _weighted_oit_available(viewer: ViewerGL) -> bool:
    can_use = getattr(viewer.renderer, "_can_use_weighted_transparency", None)
    if can_use is None:
        return False
    return bool(can_use(scene_has_transparency=True))


def _set_scene(viewer: ViewerGL, scene: Scene) -> None:
    viewer.set_model(scene.model)
    _set_camera(viewer, scene.camera_pos, scene.camera_target)


def _run_mode(
    viewer: ViewerGL,
    scene: Scene,
    mode: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ModeResult, np.ndarray | None]:
    viewer.renderer.enable_weighted_transparency = mode == MODE_WEIGHTED_OIT

    for frame_index in range(args.warmup):
        _render_frame(viewer, scene.state, frame_index)

    screenshot_image = None
    screenshot_path = None
    if args.screenshots:
        _render_frame(viewer, scene.state, args.warmup)
        screenshot_image = _capture_frame(viewer)
        screenshot_path = output_dir / f"{scene.name}_{mode}.png"
        _save_png(screenshot_path, screenshot_image)

    frame_times_ms = []
    for frame_index in range(args.frames):
        start = time.perf_counter()
        _render_frame(viewer, scene.state, args.warmup + frame_index + 1)
        frame_times_ms.append((time.perf_counter() - start) * 1000.0)

    return (
        ModeResult(
            scene=scene.name,
            mode=mode,
            frame_times_ms=frame_times_ms,
            screenshot=str(screenshot_path) if screenshot_path is not None else None,
        ),
        screenshot_image,
    )


def _summarize(results: list[ModeResult], image_metrics: dict[str, dict[str, float]], scenes: list[Scene]) -> str:
    notes = {scene.name: scene.notes for scene in scenes}
    lines = [
        "# ViewerGL Transparency Benchmark",
        "",
        "| Scene | Mode | Mean ms | Median ms | Min ms | Max ms | Screenshot |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        screenshot = Path(result.screenshot).name if result.screenshot else ""
        lines.append(
            f"| {result.scene} | {result.mode} | {result.mean_ms:.3f} | {result.median_ms:.3f} | "
            f"{result.min_ms:.3f} | {result.max_ms:.3f} | {screenshot} |"
        )

    by_scene: dict[str, dict[str, ModeResult]] = {}
    for result in results:
        by_scene.setdefault(result.scene, {})[result.mode] = result
    paired_results = {
        scene: modes for scene, modes in by_scene.items() if MODE_SORTED in modes and MODE_WEIGHTED_OIT in modes
    }
    if paired_results:
        lines.extend(
            [
                "",
                "## Relative Performance",
                "",
                "| Scene | Sorted mean ms | Weighted OIT mean ms | OIT speedup |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for scene, modes in paired_results.items():
            sorted_result = modes[MODE_SORTED]
            weighted_result = modes[MODE_WEIGHTED_OIT]
            speedup = (sorted_result.mean_ms / weighted_result.mean_ms - 1.0) * 100.0
            lines.append(f"| {scene} | {sorted_result.mean_ms:.3f} | {weighted_result.mean_ms:.3f} | {speedup:+.1f}% |")

    if image_metrics:
        lines.extend(
            [
                "",
                "## Image Differences",
                "",
                "| Scene | Mean abs RGB | RMS RGB | Max abs RGB | Comparison |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for scene, metrics in image_metrics.items():
            comparison = Path(metrics["comparison"]).name
            lines.append(
                f"| {scene} | {metrics['mean_abs']:.3f} | {metrics['rms']:.3f} | "
                f"{metrics['max_abs']:.0f} | {comparison} |"
            )

    lines.extend(["", "## Scenes", ""])
    for name, note in notes.items():
        lines.append(f"- `{name}`: {note}.")
    lines.append("")
    return "\n".join(lines)


def _system_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "newton": getattr(newton, "__version__", "unknown"),
        "warp": getattr(wp, "__version__", "unknown"),
        "device": str(wp.get_device()),
    }


def _write_outputs(
    output_dir: Path,
    results: list[ModeResult],
    image_metrics: dict[str, dict[str, float]],
    scenes: list[Scene],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        "system": _system_info(),
        "results": [
            {
                "scene": result.scene,
                "mode": result.mode,
                "mean_ms": result.mean_ms,
                "median_ms": result.median_ms,
                "min_ms": result.min_ms,
                "max_ms": result.max_ms,
                "frame_times_ms": result.frame_times_ms,
                "screenshot": result.screenshot,
            }
            for result in results
        ],
        "image_metrics": image_metrics,
        "scenes": [{"name": scene.name, "notes": scene.notes} for scene in scenes],
    }
    (output_dir / "results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_summarize(results, image_metrics, scenes), encoding="utf-8")


def _parse_csv_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def _expand_scenes(scene_names: list[str]) -> list[str]:
    if any(scene == "all" for scene in scene_names):
        return list(SCENE_ALL)
    expanded = [SCENE_ALIASES.get(scene, scene) for scene in scene_names]
    unknown = [scene for scene in expanded if scene not in SCENE_ALL]
    if unknown:
        aliases = ", ".join(sorted(SCENE_ALIASES))
        raise ValueError(
            f"Unknown scene(s): {', '.join(unknown)}. Expected: all, {', '.join(SCENE_ALL)}"
            f" or legacy alias(es): {aliases}."
        )
    return expanded


def _expand_modes(mode_names: list[str]) -> list[str]:
    valid = (MODE_SORTED, MODE_WEIGHTED_OIT)
    unknown = [mode for mode in mode_names if mode not in valid]
    if unknown:
        raise ValueError(f"Unknown mode(s): {', '.join(unknown)}. Expected: {', '.join(valid)}.")
    return mode_names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--scenes",
        default="all",
        help=(
            f"Comma-separated scenes: all, {', '.join(SCENE_ALL)}. "
            "The old tri_surface name is accepted as an alias for cloth_shirt."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/viewer_transparency"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--opacity", type=float, default=0.45)
    parser.add_argument("--cloth-count", type=int, default=4, help="Transparent unisex_shirt.usd cloth instances.")
    parser.add_argument("--cloth-scale", type=float, default=0.025, help="Uniform scale for the shirt cloth asset.")
    parser.add_argument("--cloth-opacity-pattern", choices=("uniform", "bands"), default="uniform")
    parser.add_argument("--shape-count", type=int, default=240, help="Generated primitive shapes in shape_cloud.")
    parser.add_argument("--mesh-count", type=int, default=160, help="Generated mesh instances in mesh_spheres.")
    parser.add_argument("--mesh-latitude", type=int, default=24, help="Latitude segments for generated sphere meshes.")
    parser.add_argument(
        "--mesh-longitude", type=int, default=24, help="Longitude segments for generated sphere meshes."
    )
    parser.add_argument("--bunny-count", type=int, default=24, help="Transparent bunny.usd mesh instances.")
    parser.add_argument("--bunny-scale", type=float, default=0.46, help="Uniform scale for the bunny mesh asset.")
    parser.add_argument(
        "--modes",
        default=f"{MODE_SORTED},{MODE_WEIGHTED_OIT}",
        help=f"Comma-separated modes: {MODE_SORTED}, {MODE_WEIGHTED_OIT}.",
    )
    parser.add_argument("--use-msaa", action="store_true", help="Keep ViewerGL's MSAA path enabled.")
    parser.add_argument(
        "--allow-oit-fallback",
        action="store_true",
        help="Continue even if this GL context cannot run weighted OIT.",
    )
    parser.add_argument("--no-screenshots", dest="screenshots", action="store_false")
    parser.set_defaults(screenshots=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.frames <= 0:
        raise ValueError("--frames must be greater than zero.")
    if args.warmup < 0:
        raise ValueError("--warmup must be greater than or equal to zero.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be greater than zero.")
    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError("--opacity must be in [0, 1].")
    if args.cloth_count <= 0:
        raise ValueError("--cloth-count must be greater than zero.")
    if args.cloth_scale <= 0.0:
        raise ValueError("--cloth-scale must be greater than zero.")
    if args.shape_count <= 0:
        raise ValueError("--shape-count must be greater than zero.")
    if args.mesh_count <= 0:
        raise ValueError("--mesh-count must be greater than zero.")
    if args.mesh_latitude < 3 or args.mesh_longitude < 3:
        raise ValueError("--mesh-latitude and --mesh-longitude must be at least 3.")
    if args.bunny_count <= 0:
        raise ValueError("--bunny-count must be greater than zero.")
    if args.bunny_scale <= 0.0:
        raise ValueError("--bunny-scale must be greater than zero.")

    scenes = _expand_scenes(_parse_csv_names(args.scenes))
    modes = _expand_modes(_parse_csv_names(args.modes))
    if not scenes:
        raise ValueError("At least one scene is required.")
    if not modes:
        raise ValueError("At least one mode is required.")
    return scenes, modes


def main() -> None:
    args = _parse_args()
    scene_names, modes = _validate_args(args)
    wp.init()

    scenes = [_build_scene(name, args) for name in scene_names]
    results: list[ModeResult] = []
    screenshots: dict[str, dict[str, np.ndarray]] = {}
    viewer = _create_viewer(args)

    try:
        if MODE_WEIGHTED_OIT in modes and not _weighted_oit_available(viewer):
            message = (
                "Weighted OIT is unavailable for this OpenGL context. "
                "Use --allow-oit-fallback to record the fallback path anyway."
            )
            if not args.allow_oit_fallback:
                raise RuntimeError(message)
            print(f"Warning: {message}")

        for scene in scenes:
            screenshots[scene.name] = {}
            _set_scene(viewer, scene)
            for mode in modes:
                result, image = _run_mode(viewer, scene, mode, args, args.output_dir)
                results.append(result)
                if image is not None:
                    screenshots[scene.name][mode] = image
                print(f"{scene.name:24s} {mode:12s} mean={result.mean_ms:8.3f} ms median={result.median_ms:8.3f} ms")
    finally:
        viewer.close()

    image_metrics: dict[str, dict[str, float]] = {}
    for scene_name, images in screenshots.items():
        if MODE_WEIGHTED_OIT not in images or MODE_SORTED not in images:
            continue
        weighted = images[MODE_WEIGHTED_OIT]
        sorted_image = images[MODE_SORTED]
        diff = weighted.astype(np.float32) - sorted_image.astype(np.float32)
        comparison_path = args.output_dir / f"{scene_name}_compare.png"
        _save_comparison(comparison_path, weighted, sorted_image)
        image_metrics[scene_name] = {
            "mean_abs": float(np.mean(np.abs(diff))),
            "rms": float(np.sqrt(np.mean(diff * diff))),
            "max_abs": float(np.max(np.abs(diff))),
            "comparison": str(comparison_path),
        }

    _write_outputs(args.output_dir, results, image_metrics, scenes)
    print(f"\nWrote ViewerGL transparency benchmark artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
