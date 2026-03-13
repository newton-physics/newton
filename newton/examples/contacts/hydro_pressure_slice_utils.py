# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable utilities for hydroelastic pressure slice examples."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton


@wp.kernel
def build_slice_points(
    resolution: int,
    axis: int,
    axis_position: float,
    plane_scale: float,
    sdf_center: wp.vec3,
    sdf_half_extents: wp.vec3,
    out_points: wp.array(dtype=wp.vec3),
):
    """Sample one axis-aligned section in object-local coordinates."""
    tid = wp.tid()
    total = resolution * resolution
    if tid >= total:
        return

    u_idx = tid % resolution
    v_idx = tid // resolution

    u = float(u_idx) / float(resolution - 1)
    v = float(v_idx) / float(resolution - 1)

    hx = sdf_half_extents[0] * plane_scale
    hy = sdf_half_extents[1] * plane_scale
    hz = sdf_half_extents[2] * plane_scale

    sample = wp.vec3(0.0, 0.0, 0.0)
    if axis == 0:
        sample = wp.vec3(
            sdf_center[0] + axis_position,
            sdf_center[1] + (2.0 * u - 1.0) * hy,
            sdf_center[2] + (2.0 * v - 1.0) * hz,
        )
    elif axis == 1:
        sample = wp.vec3(
            sdf_center[0] + (2.0 * u - 1.0) * hx,
            sdf_center[1] + axis_position,
            sdf_center[2] + (2.0 * v - 1.0) * hz,
        )
    else:
        sample = wp.vec3(
            sdf_center[0] + (2.0 * u - 1.0) * hx,
            sdf_center[1] + (2.0 * v - 1.0) * hy,
            sdf_center[2] + axis_position,
        )

    out_points[tid] = sample


@wp.kernel
def sample_pressure_on_slice(
    clip_to_sdf: int,
    pressure_volume_id: wp.uint64,
    sdf_sparse_volume_id: wp.uint64,
    sdf_coarse_volume_id: wp.uint64,
    sdf_background_value: float,
    sdf_center: wp.vec3,
    sdf_half_extents: wp.vec3,
    pressure_epsilon: float,
    points: wp.array(dtype=wp.vec3),
    out_inside_flag: wp.array(dtype=wp.float32),
    out_pressure: wp.array(dtype=wp.float32),
    out_inside_count: wp.array(dtype=wp.int32),
):
    """Sample immutable pressure volume and classify pressure-support interior."""
    tid = wp.tid()
    sample = points[tid]

    pressure_idx = wp.volume_world_to_index(pressure_volume_id, sample)
    pressure = wp.volume_sample_f(pressure_volume_id, pressure_idx, wp.Volume.LINEAR)
    if wp.isnan(pressure):
        pressure = 0.0
    pressure = wp.max(pressure, 0.0)

    if clip_to_sdf == 0:
        if pressure <= pressure_epsilon:
            out_inside_flag[tid] = -1.0
            out_pressure[tid] = -1.0
            return

        out_inside_flag[tid] = 1.0
        out_pressure[tid] = pressure
        wp.atomic_add(out_inside_count, 0, 1)
        return

    lower = sdf_center - sdf_half_extents
    upper = sdf_center + sdf_half_extents
    inside_extent = (
        sample[0] >= lower[0]
        and sample[0] <= upper[0]
        and sample[1] >= lower[1]
        and sample[1] <= upper[1]
        and sample[2] >= lower[2]
        and sample[2] <= upper[2]
    )
    if not inside_extent:
        out_inside_flag[tid] = -1.0
        out_pressure[tid] = -1.0
        return

    sdf_idx = wp.volume_world_to_index(sdf_sparse_volume_id, sample)
    sdf_val = wp.volume_sample_f(sdf_sparse_volume_id, sdf_idx, wp.Volume.LINEAR)
    invalid_sdf = sdf_val >= sdf_background_value * 0.99 or wp.isnan(sdf_val)
    if invalid_sdf and sdf_coarse_volume_id != wp.uint64(0):
        coarse_idx = wp.volume_world_to_index(sdf_coarse_volume_id, sample)
        sdf_val = wp.volume_sample_f(sdf_coarse_volume_id, coarse_idx, wp.Volume.LINEAR)
        invalid_sdf = wp.isnan(sdf_val)
    if invalid_sdf or sdf_val > 0.0:
        out_inside_flag[tid] = -1.0
        out_pressure[tid] = -1.0
        return

    out_inside_flag[tid] = 1.0
    out_pressure[tid] = wp.where(pressure > pressure_epsilon, pressure, 0.0)
    wp.atomic_add(out_inside_count, 0, 1)


def density_to_rgb_image(density_grid: np.ndarray) -> np.ndarray:
    """Map density [0, 1] to RGB with a monotonic perceptual colormap."""
    d = np.clip(density_grid.astype(np.float32), 0.0, 1.0)

    # Viridis-like anchor table (normalized RGB) to avoid false contour bands.
    t_knots = np.array([0.00, 0.13, 0.25, 0.38, 0.50, 0.63, 0.75, 0.88, 1.00], dtype=np.float32)
    c_knots = (
        np.array(
            [
                [68, 1, 84],
                [71, 44, 122],
                [59, 81, 139],
                [44, 113, 142],
                [33, 144, 141],
                [39, 173, 129],
                [92, 200, 99],
                [170, 220, 50],
                [253, 231, 37],
            ],
            dtype=np.float32,
        )
        / 255.0
    )

    seg = np.searchsorted(t_knots, d, side="right") - 1
    seg = np.clip(seg, 0, len(t_knots) - 2)
    t0 = t_knots[seg]
    t1 = t_knots[seg + 1]
    alpha = (d - t0) / np.maximum(t1 - t0, 1.0e-8)
    rgb = (1.0 - alpha[..., None]) * c_knots[seg] + alpha[..., None] * c_knots[seg + 1]

    outside = density_grid < 0.0
    rgb[outside] = 0.0

    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def validate_slice_field(pressure_grid: np.ndarray) -> tuple[bool, float, float, float]:
    """Return validation metrics proving smooth interior pressure continuity."""
    inside = pressure_grid >= 0.0
    if not np.any(inside):
        return False, 0.0, 0.0, 0.0

    p_inside = pressure_grid[inside]
    max_p = float(np.max(p_inside))
    if max_p <= 1.0e-8:
        return False, 0.0, 0.0, 0.0

    p_norm = np.zeros_like(pressure_grid, dtype=np.float32)
    p_norm[inside] = pressure_grid[inside] / max_p

    # One-pixel morphological erosion marks deep interior support.
    deep = inside.copy()
    deep[1:, :] &= inside[:-1, :]
    deep[:-1, :] &= inside[1:, :]
    deep[:, 1:] &= inside[:, :-1]
    deep[:, :-1] &= inside[:, 1:]
    if not np.any(deep):
        deep = inside
    deep_p05 = float(np.percentile(p_norm[deep], 5.0)) if np.any(deep) else 0.0
    zero_fraction = float(np.mean(p_norm[inside] <= 1.0e-6))

    ys, xs = np.where(inside)
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    sub_inside = inside[y0 : y1 + 1, x0 : x1 + 1]
    outside = ~sub_inside
    exterior = np.zeros_like(outside, dtype=bool)
    q: list[tuple[int, int]] = []

    h, w = outside.shape
    for x in range(w):
        if outside[0, x] and not exterior[0, x]:
            exterior[0, x] = True
            q.append((0, x))
        if outside[h - 1, x] and not exterior[h - 1, x]:
            exterior[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if outside[y, 0] and not exterior[y, 0]:
            exterior[y, 0] = True
            q.append((y, 0))
        if outside[y, w - 1] and not exterior[y, w - 1]:
            exterior[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        y, x = q.pop()
        if y > 0 and outside[y - 1, x] and not exterior[y - 1, x]:
            exterior[y - 1, x] = True
            q.append((y - 1, x))
        if y + 1 < h and outside[y + 1, x] and not exterior[y + 1, x]:
            exterior[y + 1, x] = True
            q.append((y + 1, x))
        if x > 0 and outside[y, x - 1] and not exterior[y, x - 1]:
            exterior[y, x - 1] = True
            q.append((y, x - 1))
        if x + 1 < w and outside[y, x + 1] and not exterior[y, x + 1]:
            exterior[y, x + 1] = True
            q.append((y, x + 1))

    hole_fraction = float(np.sum(outside & (~exterior))) / float(np.sum(inside))

    valid = bool(deep_p05 > 0.05 and zero_fraction < 0.10 and hole_fraction < 1.0e-3)
    return valid, deep_p05, zero_fraction, hole_fraction


def validate_slice_metrics(
    deep_p05: float,
    zero_fraction: float,
    hole_fraction: float,
    *,
    clipped_mesh_slice: bool,
) -> bool:
    """Validate slice metrics with a slightly lower interior threshold for clipped mesh slices."""
    min_deep_p05 = 0.03 if clipped_mesh_slice else 0.05
    return bool(deep_p05 > min_deep_p05 and zero_fraction < 0.10 and hole_fraction < 1.0e-3)


def build_regular_grid_indices(resolution: int) -> np.ndarray:
    """Create triangle index buffer for a regular resolution x resolution grid."""
    tris: list[int] = []
    for v in range(resolution - 1):
        row0 = v * resolution
        row1 = (v + 1) * resolution
        for u in range(resolution - 1):
            i00 = row0 + u
            i10 = row0 + (u + 1)
            i01 = row1 + u
            i11 = row1 + (u + 1)
            tris.extend([i00, i10, i11, i00, i11, i01])
    return np.asarray(tris, dtype=np.int32)


def build_regular_grid_uvs(resolution: int) -> np.ndarray:
    """Create UVs for a regular resolution x resolution grid."""
    u = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    v = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)


def build_mesh_edge_lines(mesh: newton.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Create unique edge-line buffers from a triangle mesh."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    indices = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)

    edges: set[tuple[int, int]] = set()
    for tri in indices:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add((min(i0, i1), max(i0, i1)))
        edges.add((min(i1, i2), max(i1, i2)))
        edges.add((min(i2, i0), max(i2, i0)))

    starts = np.array([vertices[i] for i, _ in edges], dtype=np.float32)
    ends = np.array([vertices[j] for _, j in edges], dtype=np.float32)
    return starts, ends


def load_trimesh_for_hydro(mesh_file: str) -> Any:
    """Load a mesh file as a single trimesh mesh for hydro preprocessing."""
    try:
        trimesh = importlib.import_module("trimesh")
    except ImportError as exc:
        raise RuntimeError(
            "Loading external mesh files requires trimesh. "
            "Install example dependencies (for example: `uv sync --extra examples`)."
        ) from exc

    tri = trimesh.load(mesh_file, force="mesh", process=False)
    if hasattr(tri, "geometry"):
        geometries = []
        for geom in tri.geometry.values():
            geom_vertices = np.asarray(getattr(geom, "vertices", []))
            geom_faces = np.asarray(getattr(geom, "faces", []))
            if geom_vertices.size == 0 or geom_faces.size == 0:
                continue
            geometries.append(geom)
        if not geometries:
            raise ValueError(f"Mesh file '{mesh_file}' did not contain any triangle geometry.")
        tri = trimesh.util.concatenate(tuple(geometries))
    return tri


def find_first_usd_mesh_prim(stage: Any, asset_name: str):
    """Return the first mesh prim from a USD stage."""
    from pxr import UsdGeom  # noqa: PLC0415

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            return prim
    raise ValueError(f"No mesh prim found in USD asset: {asset_name}")


def collect_hydro_mesh_stats(tri_mesh: Any, vertices: np.ndarray) -> dict[str, Any]:
    """Collect topology and size stats relevant for hydro readiness."""
    extents = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    components = len(tri_mesh.split(only_watertight=False))
    return {
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(np.asarray(tri_mesh.faces).shape[0]),
        "components": int(components),
        "watertight": bool(tri_mesh.is_watertight),
        "winding_consistent": bool(tri_mesh.is_winding_consistent),
        "is_volume": bool(tri_mesh.is_volume),
        "extents": extents.astype(np.float32),
    }


def validate_hydro_mesh_stats(stats: dict[str, Any], *, allow_multiple_components: bool) -> list[str]:
    """Return human-readable hydro readiness issues for a mesh stats record."""
    issues: list[str] = []
    if not stats["watertight"]:
        issues.append("mesh is not watertight")
    if not stats["winding_consistent"]:
        issues.append("mesh winding is inconsistent")
    if not stats["is_volume"]:
        issues.append("mesh does not represent a closed volume")
    if (not allow_multiple_components) and stats["components"] != 1:
        issues.append(f"mesh has {stats['components']} disconnected components (expected 1)")
    return issues


def load_external_mesh_for_hydro(
    mesh_file: str,
    *,
    scale: float = 1.0,
    center_origin: bool = False,
    allow_multiple_components: bool = False,
    skip_hydro_validation: bool = False,
) -> tuple[newton.Mesh, str]:
    """Load, validate, and convert an external triangle mesh for hydroelastic SDF use."""
    mesh_path = Path(mesh_file).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    tri_mesh = load_trimesh_for_hydro(str(mesh_path))
    vertices = np.asarray(tri_mesh.vertices, dtype=np.float32)
    faces = np.asarray(tri_mesh.faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh file '{mesh_path}' has no vertices or faces.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Mesh vertices must have shape (N, 3); got {vertices.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Mesh faces must have shape (M, 3); got {faces.shape}.")

    if center_origin:
        center = 0.5 * (np.min(vertices, axis=0) + np.max(vertices, axis=0))
        vertices = vertices - center.astype(np.float32)
    if scale != 1.0:
        vertices = vertices * float(scale)

    tri_mesh_scaled = tri_mesh.copy()
    tri_mesh_scaled.vertices = vertices
    stats = collect_hydro_mesh_stats(tri_mesh_scaled, vertices)
    extents = stats["extents"]
    print(
        f"[hydro_pressure_slice] Mesh '{mesh_path.name}': "
        f"{stats['vertex_count']} vertices, {stats['triangle_count']} triangles, "
        f"components={stats['components']}, watertight={stats['watertight']}, "
        f"winding_consistent={stats['winding_consistent']}, is_volume={stats['is_volume']}, "
        f"extents=[{extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f}] m"
    )

    issues = validate_hydro_mesh_stats(stats, allow_multiple_components=allow_multiple_components)
    if issues and (not skip_hydro_validation):
        raise ValueError(
            "External mesh is not hydro-ready:\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + "\nUse scripts/prepare_hydro_mesh.py to repair and validate the mesh."
        )

    mesh = newton.Mesh(vertices, faces.reshape(-1), compute_inertia=False)
    return mesh, mesh_path.stem


def normalize_pressure_for_display(
    pressure_grid: np.ndarray,
    *,
    inside: np.ndarray,
    mode: str,
    global_max: float,
    percentile_low: float,
    percentile_high: float,
    gamma: float,
) -> np.ndarray:
    """Normalize pressure values for display without affecting physical validation."""
    normalized = np.full_like(pressure_grid, -1.0, dtype=np.float32)
    if not np.any(inside):
        return normalized

    p_inside = pressure_grid[inside].astype(np.float32, copy=False)
    if mode == "global":
        if global_max > 1.0e-8:
            display = p_inside / float(global_max)
        else:
            display = np.zeros_like(p_inside)
    else:
        lo = float(np.percentile(p_inside, percentile_low))
        hi = float(np.percentile(p_inside, percentile_high))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo + 1.0e-8):
            lo = float(np.min(p_inside))
            hi = float(np.max(p_inside))
        if hi <= lo + 1.0e-8:
            display = np.zeros_like(p_inside)
        else:
            display = (p_inside - lo) / (hi - lo)

    display = np.clip(display, 0.0, 1.0)
    gamma = max(float(gamma), 1.0e-4)
    if abs(gamma - 1.0) > 1.0e-6:
        display = np.power(display, gamma).astype(np.float32, copy=False)

    normalized[inside] = display
    return normalized


__all__ = [
    "build_mesh_edge_lines",
    "build_regular_grid_indices",
    "build_regular_grid_uvs",
    "build_slice_points",
    "collect_hydro_mesh_stats",
    "density_to_rgb_image",
    "find_first_usd_mesh_prim",
    "load_external_mesh_for_hydro",
    "normalize_pressure_for_display",
    "sample_pressure_on_slice",
    "validate_hydro_mesh_stats",
    "validate_slice_field",
    "validate_slice_metrics",
]
