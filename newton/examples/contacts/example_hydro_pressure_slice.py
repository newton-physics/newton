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

###########################################################################
# Example Hydro Pressure Slice
#
# Visualize the immutable pressure field produced by HydroelasticSDF by
# slicing the loaded shape and rendering one continuous heat-map section.
#
# Command: uv run -m newton.examples.contacts.example_hydro_pressure_slice --shape box
#
###########################################################################

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples


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
    pressure_volume_id: wp.uint64,
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
    if pressure <= pressure_epsilon:
        out_inside_flag[tid] = -1.0
        out_pressure[tid] = -1.0
        return

    out_inside_flag[tid] = 1.0
    out_pressure[tid] = pressure
    wp.atomic_add(out_inside_count, 0, 1)


@wp.kernel
def apply_pressure_axis_sine_modulation(
    axis: int,
    axis_center: float,
    axis_half_extent: float,
    amplitude: float,
    cycles_across_extent: float,
    phase_rad: float,
    points: wp.array(dtype=wp.vec3),
    inside_flag: wp.array(dtype=wp.float32),
    pressure: wp.array(dtype=wp.float32),
):
    """Apply optional sine modulation along one axis to interior pressure samples."""
    tid = wp.tid()
    if inside_flag[tid] <= 0.0:
        return

    p = pressure[tid]
    if p <= 0.0:
        return

    coord = 0.0
    if axis == 0:
        coord = points[tid][0]
    elif axis == 1:
        coord = points[tid][1]
    else:
        coord = points[tid][2]

    coord01 = 0.5
    if axis_half_extent > 1.0e-8:
        coord_norm = (coord - axis_center) / axis_half_extent
        coord01 = 0.5 * (coord_norm + 1.0)

    two_pi = 6.283185307179586
    wave = wp.sin(two_pi * cycles_across_extent * coord01 + phase_rad)
    modulation = wp.max(1.0 + amplitude * wave, 0.0)
    pressure[tid] = p * modulation


@wp.kernel
def apply_pressure_foot_structure_modulation(
    sdf_center: wp.vec3,
    sdf_half_extents: wp.vec3,
    structure_strength: float,
    arch_depth: float,
    medial_bias: float,
    points: wp.array(dtype=wp.vec3),
    inside_flag: wp.array(dtype=wp.float32),
    pressure: wp.array(dtype=wp.float32),
):
    """Apply localized foot-like pressure modulation (bone lobes + arch trough)."""
    tid = wp.tid()
    if inside_flag[tid] <= 0.0:
        return

    p = pressure[tid]
    if p <= 0.0:
        return

    sample = points[tid]
    hx = wp.max(sdf_half_extents[0], 1.0e-8)
    hy = wp.max(sdf_half_extents[1], 1.0e-8)
    hz = wp.max(sdf_half_extents[2], 1.0e-8)
    x = (sample[0] - sdf_center[0]) / hx
    y = (sample[1] - sdf_center[1]) / hy
    z = (sample[2] - sdf_center[2]) / hz

    medial = wp.max(-1.0, wp.min(1.0, medial_bias))

    # Heel pads are modeled as two side peaks plus one broader center lobe.
    g_heel_medial = wp.exp(
        -(
            ((x + 0.62) * (x + 0.62)) / 0.12
            + ((y - (0.19 + 0.10 * medial)) * (y - (0.19 + 0.10 * medial))) / 0.11
            + ((z + 0.15) * (z + 0.15)) / 0.14
        )
    )
    g_heel_lateral = wp.exp(
        -(
            ((x + 0.64) * (x + 0.64)) / 0.12
            + ((y + 0.20 - 0.08 * medial) * (y + 0.20 - 0.08 * medial)) / 0.12
            + ((z + 0.15) * (z + 0.15)) / 0.14
        )
    )
    g_heel_center = wp.exp(-(((x + 0.56) * (x + 0.56)) / 0.16 + (y * y) / 0.22 + ((z + 0.10) * (z + 0.10)) / 0.18))

    # Midfoot scaffold that keeps a smooth anatomical transition.
    g_talus = wp.exp(
        -(
            ((x + 0.28) * (x + 0.28)) / 0.15
            + ((y - 0.05 * medial) * (y - 0.05 * medial)) / 0.20
            + ((z + 0.03) * (z + 0.03)) / 0.14
        )
    )
    g_navicular = wp.exp(
        -(
            ((x + 0.02) * (x + 0.02)) / 0.12
            + ((y - (0.26 + 0.10 * medial)) * (y - (0.26 + 0.10 * medial))) / 0.12
            + ((z + 0.08) * (z + 0.08)) / 0.12
        )
    )
    g_cuboid = wp.exp(
        -(
            ((x + 0.00) * (x + 0.00)) / 0.13
            + ((y + 0.30 - 0.08 * medial) * (y + 0.30 - 0.08 * medial)) / 0.14
            + ((z + 0.08) * (z + 0.08)) / 0.12
        )
    )

    # Metatarsal heads are tightened to create clearer forefoot definition.
    g_meta_lateral = wp.exp(
        -(((x - 0.44) * (x - 0.44)) / 0.12 + ((y + 0.36) * (y + 0.36)) / 0.11 + ((z + 0.12) * (z + 0.12)) / 0.12)
    )
    g_meta_center = wp.exp(-(((x - 0.49) * (x - 0.49)) / 0.11 + (y * y) / 0.10 + ((z + 0.12) * (z + 0.12)) / 0.12))
    g_meta_medial = wp.exp(
        -(
            ((x - 0.52) * (x - 0.52)) / 0.10
            + ((y - (0.34 + 0.14 * medial)) * (y - (0.34 + 0.14 * medial))) / 0.10
            + ((z + 0.12) * (z + 0.12)) / 0.12
        )
    )
    g_meta_shelf = wp.exp(-(((x - 0.40) * (x - 0.40)) / 0.18 + (y * y) / 0.26 + ((z + 0.14) * (z + 0.14)) / 0.18))

    # Arch and midfoot troughs split heel and forefoot into distinct regions.
    g_arch = wp.exp(
        -(
            ((x - 0.02) * (x - 0.02)) / 0.28
            + ((y - (0.26 + 0.14 * medial)) * (y - (0.26 + 0.14 * medial))) / 0.19
            + ((z + 0.28) * (z + 0.28)) / 0.18
        )
    )
    g_midfoot_split = wp.exp(
        -(
            ((x - 0.08) * (x - 0.08)) / 0.12
            + ((y - (0.04 + 0.05 * medial)) * (y - (0.04 + 0.05 * medial))) / 0.28
            + ((z + 0.18) * (z + 0.18)) / 0.16
        )
    )
    g_toe_relief = wp.exp(-(((x - 0.85) * (x - 0.85)) / 0.12 + (y * y) / 0.24 + ((z + 0.08) * (z + 0.08)) / 0.16))

    heel_field = 0.96 * g_heel_medial + 0.92 * g_heel_lateral + 0.74 * g_heel_center
    midfoot_field = 0.44 * g_talus + 0.36 * g_navicular + 0.32 * g_cuboid
    metatarsal_field = 0.86 * g_meta_lateral + 0.94 * g_meta_center + 1.06 * g_meta_medial + 0.34 * g_meta_shelf
    positive_field = 0.82 * heel_field + 0.30 * midfoot_field + 0.98 * metatarsal_field
    negative_field = 0.72 * arch_depth * g_arch + 0.46 * g_midfoot_split + 0.24 * g_toe_relief
    structure = 0.50 * positive_field - negative_field
    modulation = wp.max(0.0, wp.min(3.0, 1.0 + structure_strength * structure))
    pressure[tid] = p * modulation


def density_to_rgb_image(density_grid: np.ndarray) -> np.ndarray:
    """Map density [0, 1] to RGB with a monotonic perceptual colormap."""
    d = np.clip(density_grid.astype(np.float32), 0.0, 1.0)

    # Viridis-like anchor table (normalized RGB) to avoid false contour bands.
    t_knots = np.array([0.00, 0.13, 0.25, 0.38, 0.50, 0.63, 0.75, 0.88, 1.00], dtype=np.float32)
    c_knots = np.array(
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
    ) / 255.0

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


def _load_trimesh_for_hydro(mesh_file: str) -> Any:
    """Load a mesh file as a single trimesh mesh for hydro preprocessing."""
    try:
        import trimesh
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


def _collect_hydro_mesh_stats(tri_mesh: Any, vertices: np.ndarray) -> dict[str, Any]:
    """Collect topology and size stats relevant for hydroelastic SDF meshes."""
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


def _validate_hydro_mesh_stats(stats: dict[str, Any], *, allow_multiple_components: bool) -> list[str]:
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

    tri_mesh = _load_trimesh_for_hydro(str(mesh_path))
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
    stats = _collect_hydro_mesh_stats(tri_mesh_scaled, vertices)
    extents = stats["extents"]
    print(
        f"[hydro_pressure_slice] Mesh '{mesh_path.name}': "
        f"{stats['vertex_count']} vertices, {stats['triangle_count']} triangles, "
        f"components={stats['components']}, watertight={stats['watertight']}, "
        f"winding_consistent={stats['winding_consistent']}, is_volume={stats['is_volume']}, "
        f"extents=[{extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f}] m"
    )

    issues = _validate_hydro_mesh_stats(stats, allow_multiple_components=allow_multiple_components)
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


class Example:
    def __init__(self, viewer, args):
        if not wp.get_device().is_cuda:
            raise RuntimeError(
                "hydro_pressure_slice requires CUDA (Mesh.build_sdf uses wp.Volume, which is CUDA-only)."
            )

        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.shape_name = args.shape
        self.has_external_mesh = args.mesh_file is not None
        self.viewer_is_viser = self.viewer.__class__.__name__ == "ViewerViser"
        self.slice_axis = {"x": 0, "y": 1, "z": 2}[args.slice_axis]
        self.slice_position_normalized = float(args.slice_position)
        self.animate_slice = bool(args.animate_slice)
        self.slice_speed = float(args.slice_speed)
        self.plane_scale = float(args.plane_scale)
        self.resolution = int(args.resolution)
        self.shape_opacity = float(args.shape_opacity)
        self.pressure_x_sine_amplitude = float(args.pressure_x_sine_amplitude)
        self.pressure_x_sine_cycles = float(args.pressure_x_sine_cycles)
        self.pressure_x_sine_phase = float(args.pressure_x_sine_phase)
        self.pressure_y_sine_amplitude = float(args.pressure_y_sine_amplitude)
        self.pressure_y_sine_cycles = float(args.pressure_y_sine_cycles)
        self.pressure_y_sine_phase = float(args.pressure_y_sine_phase)
        self.pressure_z_sine_amplitude = float(args.pressure_z_sine_amplitude)
        self.pressure_z_sine_cycles = float(args.pressure_z_sine_cycles)
        self.pressure_z_sine_phase = float(args.pressure_z_sine_phase)
        self.pressure_foot_structure_strength = float(args.pressure_foot_structure_strength)
        self.pressure_foot_arch_depth = float(args.pressure_foot_arch_depth)
        self.pressure_foot_medial_bias = float(args.pressure_foot_medial_bias)

        display_mode_arg = str(args.display_mode).strip().lower()
        if display_mode_arg == "auto":
            self.display_mode = "slice_percentile" if self.has_external_mesh else "global"
        else:
            self.display_mode = display_mode_arg
        self.display_percentile_low = float(args.display_percentile_low)
        self.display_percentile_high = float(args.display_percentile_high)
        self.display_gamma = float(args.display_gamma)

        self.display_percentile_low = float(np.clip(self.display_percentile_low, 0.0, 95.0))
        self.display_percentile_high = float(np.clip(self.display_percentile_high, 5.0, 100.0))
        if self.display_percentile_high <= self.display_percentile_low + 1.0:
            self.display_percentile_high = min(100.0, self.display_percentile_low + 1.0)

        # Auto-apply a foot-pressure visualization preset for external meshes when
        # no custom foot modulation strength was provided.
        if bool(args.foot_pressure_preset) or (self.has_external_mesh and self.pressure_foot_structure_strength <= 1.0e-8):
            if self.pressure_foot_structure_strength <= 1.0e-8:
                self.pressure_foot_structure_strength = 0.80
            if abs(self.pressure_foot_arch_depth - 1.0) <= 1.0e-8:
                self.pressure_foot_arch_depth = 1.15
            if abs(self.pressure_foot_medial_bias - 0.3) <= 1.0e-8:
                self.pressure_foot_medial_bias = 0.22
            if abs(self.slice_position_normalized) <= 1.0e-8 and self.slice_axis == 2:
                self.slice_position_normalized = -0.18

        self.show_shape = self.viewer_is_viser
        self.show_shape_wireframe = not self.viewer_is_viser
        self.show_slice = True

        self.mesh = self._create_shape_mesh(args)
        self._build_hydroelastic_reference(args)

        self.capacity = self.resolution * self.resolution
        self.slice_points = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.slice_inside_flag = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.slice_pressure = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.slice_inside_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.last_slice_count = 0
        self.last_deep_p05 = 0.0
        self.last_zero_fraction = 0.0
        self.last_hole_fraction = 0.0
        self.last_validation_ok = False

        self.slice_indices = wp.array(build_regular_grid_indices(self.resolution), dtype=wp.int32, device=self.device)
        self.slice_uvs = wp.array(build_regular_grid_uvs(self.resolution), dtype=wp.vec2, device=self.device)
        self.slice_texture = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        self._slice_points_viser = np.zeros((0, 3), dtype=np.float32)
        self._slice_colors_viser = np.zeros((0, 3), dtype=np.uint8)
        self._viser_slice_handle = None

        self.shape_xforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self.shape_colors = wp.array([wp.vec3(0.72, 0.72, 0.76)], dtype=wp.vec3, device=self.device)
        self.shape_materials = wp.array([wp.vec4(0.75, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device)
        self._refresh_shape_materials()
        line_starts, line_ends = build_mesh_edge_lines(self.mesh)
        self.shape_line_starts = wp.array(line_starts, dtype=wp.vec3, device=self.device)
        self.shape_line_ends = wp.array(line_ends, dtype=wp.vec3, device=self.device)

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(2.5, -2.0, 1.6), -20.0, 145.0)

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self._viser_controls: dict[str, Any] = {}
        if self.viewer_is_viser:
            self._setup_viser_controls()

        self._slice_dirty = True

    def _refresh_shape_materials(self):
        alpha = np.clip(self.shape_opacity, 0.0, 1.0) if self.viewer_is_viser else 0.0
        self.shape_materials = wp.array([wp.vec4(0.75, 0.0, 0.0, float(alpha))], dtype=wp.vec4, device=self.device)

    def _setup_viser_controls(self):
        """Create native viser GUI controls when running with ViewerViser."""
        server = getattr(self.viewer, "_server", None)
        gui = getattr(server, "gui", None)
        if gui is None:
            return

        axis_name = {0: "x", 1: "y", 2: "z"}[self.slice_axis]
        with gui.add_folder("Hydro Pressure Slice"):
            self._viser_controls["show_shape"] = gui.add_checkbox("Show Shape", self.show_shape)
            self._viser_controls["show_shape_wireframe"] = gui.add_checkbox("Shape Wireframe", self.show_shape_wireframe)
            self._viser_controls["show_slice"] = gui.add_checkbox("Show Slice", self.show_slice)
            self._viser_controls["animate_slice"] = gui.add_checkbox("Animate Slice", self.animate_slice)
            self._viser_controls["slice_position"] = gui.add_slider(
                "Slice Position",
                min=-1.0,
                max=1.0,
                step=0.001,
                initial_value=float(self.slice_position_normalized),
            )
            self._viser_controls["slice_axis"] = gui.add_dropdown(
                "Slice Axis",
                options=("x", "y", "z"),
                initial_value=axis_name,
            )
            self._viser_controls["shape_opacity"] = gui.add_slider(
                "Shape Opacity",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=float(self.shape_opacity),
            )
            self._viser_controls["pressure_x_sine_amplitude"] = gui.add_slider(
                "Pressure X Sine Amp",
                min=-2.0,
                max=2.0,
                step=0.01,
                initial_value=float(self.pressure_x_sine_amplitude),
            )
            self._viser_controls["pressure_x_sine_cycles"] = gui.add_slider(
                "Pressure X Sine Cycles",
                min=0.0,
                max=12.0,
                step=0.1,
                initial_value=float(self.pressure_x_sine_cycles),
            )
            self._viser_controls["pressure_x_sine_phase"] = gui.add_slider(
                "Pressure X Sine Phase [rad]",
                min=-3.14159,
                max=3.14159,
                step=0.01,
                initial_value=float(self.pressure_x_sine_phase),
            )
            self._viser_controls["pressure_y_sine_amplitude"] = gui.add_slider(
                "Pressure Y Sine Amp",
                min=-2.0,
                max=2.0,
                step=0.01,
                initial_value=float(self.pressure_y_sine_amplitude),
            )
            self._viser_controls["pressure_y_sine_cycles"] = gui.add_slider(
                "Pressure Y Sine Cycles",
                min=0.0,
                max=12.0,
                step=0.1,
                initial_value=float(self.pressure_y_sine_cycles),
            )
            self._viser_controls["pressure_y_sine_phase"] = gui.add_slider(
                "Pressure Y Sine Phase [rad]",
                min=-3.14159,
                max=3.14159,
                step=0.01,
                initial_value=float(self.pressure_y_sine_phase),
            )
            self._viser_controls["pressure_z_sine_amplitude"] = gui.add_slider(
                "Pressure Z Sine Amp",
                min=-2.0,
                max=2.0,
                step=0.01,
                initial_value=float(self.pressure_z_sine_amplitude),
            )
            self._viser_controls["pressure_z_sine_cycles"] = gui.add_slider(
                "Pressure Z Sine Cycles",
                min=0.0,
                max=12.0,
                step=0.1,
                initial_value=float(self.pressure_z_sine_cycles),
            )
            self._viser_controls["pressure_z_sine_phase"] = gui.add_slider(
                "Pressure Z Sine Phase [rad]",
                min=-3.14159,
                max=3.14159,
                step=0.01,
                initial_value=float(self.pressure_z_sine_phase),
            )
            self._viser_controls["pressure_foot_structure_strength"] = gui.add_slider(
                "Pressure Foot Strength",
                min=0.0,
                max=2.0,
                step=0.01,
                initial_value=float(self.pressure_foot_structure_strength),
            )
            self._viser_controls["pressure_foot_arch_depth"] = gui.add_slider(
                "Pressure Foot Arch Depth",
                min=0.0,
                max=2.0,
                step=0.01,
                initial_value=float(self.pressure_foot_arch_depth),
            )
            self._viser_controls["pressure_foot_medial_bias"] = gui.add_slider(
                "Pressure Foot Medial Bias",
                min=-1.0,
                max=1.0,
                step=0.01,
                initial_value=float(self.pressure_foot_medial_bias),
            )
            self._viser_controls["display_mode"] = gui.add_dropdown(
                "Display Scale",
                options=("global", "slice_percentile"),
                initial_value=self.display_mode,
            )
            self._viser_controls["display_percentile_low"] = gui.add_slider(
                "Display P Low [%]",
                min=0.0,
                max=40.0,
                step=0.1,
                initial_value=float(self.display_percentile_low),
            )
            self._viser_controls["display_percentile_high"] = gui.add_slider(
                "Display P High [%]",
                min=60.0,
                max=100.0,
                step=0.1,
                initial_value=float(self.display_percentile_high),
            )
            self._viser_controls["display_gamma"] = gui.add_slider(
                "Display Gamma",
                min=0.4,
                max=3.0,
                step=0.01,
                initial_value=float(self.display_gamma),
            )

    def _sync_viser_controls(self):
        """Pull current values from viser controls into example state."""
        if not self._viser_controls:
            return

        show_shape = bool(self._viser_controls["show_shape"].value)
        show_shape_wireframe = bool(self._viser_controls["show_shape_wireframe"].value)
        show_slice = bool(self._viser_controls["show_slice"].value)
        animate_slice = bool(self._viser_controls["animate_slice"].value)
        axis_name = str(self._viser_controls["slice_axis"].value).lower()
        axis = {"x": 0, "y": 1, "z": 2}.get(axis_name, self.slice_axis)
        shape_opacity = float(self._viser_controls["shape_opacity"].value)
        pressure_x_sine_amplitude = float(self._viser_controls["pressure_x_sine_amplitude"].value)
        pressure_x_sine_cycles = float(self._viser_controls["pressure_x_sine_cycles"].value)
        pressure_x_sine_phase = float(self._viser_controls["pressure_x_sine_phase"].value)
        pressure_y_sine_amplitude = float(self._viser_controls["pressure_y_sine_amplitude"].value)
        pressure_y_sine_cycles = float(self._viser_controls["pressure_y_sine_cycles"].value)
        pressure_y_sine_phase = float(self._viser_controls["pressure_y_sine_phase"].value)
        pressure_z_sine_amplitude = float(self._viser_controls["pressure_z_sine_amplitude"].value)
        pressure_z_sine_cycles = float(self._viser_controls["pressure_z_sine_cycles"].value)
        pressure_z_sine_phase = float(self._viser_controls["pressure_z_sine_phase"].value)
        pressure_foot_structure_strength = float(self._viser_controls["pressure_foot_structure_strength"].value)
        pressure_foot_arch_depth = float(self._viser_controls["pressure_foot_arch_depth"].value)
        pressure_foot_medial_bias = float(self._viser_controls["pressure_foot_medial_bias"].value)
        display_mode = str(self._viser_controls["display_mode"].value)
        display_percentile_low = float(self._viser_controls["display_percentile_low"].value)
        display_percentile_high = float(self._viser_controls["display_percentile_high"].value)
        display_gamma = float(self._viser_controls["display_gamma"].value)

        if self.show_shape != show_shape:
            self.show_shape = show_shape
        if self.show_shape_wireframe != show_shape_wireframe:
            self.show_shape_wireframe = show_shape_wireframe
        if self.show_slice != show_slice:
            self.show_slice = show_slice
        if self.slice_axis != axis:
            self.slice_axis = axis
            self._slice_dirty = True
        if self.shape_opacity != shape_opacity:
            self.shape_opacity = shape_opacity
            self._refresh_shape_materials()
        if self.pressure_x_sine_amplitude != pressure_x_sine_amplitude:
            self.pressure_x_sine_amplitude = pressure_x_sine_amplitude
            self._slice_dirty = True
        if self.pressure_x_sine_cycles != pressure_x_sine_cycles:
            self.pressure_x_sine_cycles = pressure_x_sine_cycles
            self._slice_dirty = True
        if self.pressure_x_sine_phase != pressure_x_sine_phase:
            self.pressure_x_sine_phase = pressure_x_sine_phase
            self._slice_dirty = True
        if self.pressure_y_sine_amplitude != pressure_y_sine_amplitude:
            self.pressure_y_sine_amplitude = pressure_y_sine_amplitude
            self._slice_dirty = True
        if self.pressure_y_sine_cycles != pressure_y_sine_cycles:
            self.pressure_y_sine_cycles = pressure_y_sine_cycles
            self._slice_dirty = True
        if self.pressure_y_sine_phase != pressure_y_sine_phase:
            self.pressure_y_sine_phase = pressure_y_sine_phase
            self._slice_dirty = True
        if self.pressure_z_sine_amplitude != pressure_z_sine_amplitude:
            self.pressure_z_sine_amplitude = pressure_z_sine_amplitude
            self._slice_dirty = True
        if self.pressure_z_sine_cycles != pressure_z_sine_cycles:
            self.pressure_z_sine_cycles = pressure_z_sine_cycles
            self._slice_dirty = True
        if self.pressure_z_sine_phase != pressure_z_sine_phase:
            self.pressure_z_sine_phase = pressure_z_sine_phase
            self._slice_dirty = True
        if self.pressure_foot_structure_strength != pressure_foot_structure_strength:
            self.pressure_foot_structure_strength = pressure_foot_structure_strength
            self._slice_dirty = True
        if self.pressure_foot_arch_depth != pressure_foot_arch_depth:
            self.pressure_foot_arch_depth = pressure_foot_arch_depth
            self._slice_dirty = True
        if self.pressure_foot_medial_bias != pressure_foot_medial_bias:
            self.pressure_foot_medial_bias = pressure_foot_medial_bias
            self._slice_dirty = True
        if self.display_mode != display_mode:
            self.display_mode = display_mode
            self._slice_dirty = True

        display_percentile_low = float(np.clip(display_percentile_low, 0.0, 95.0))
        display_percentile_high = float(np.clip(display_percentile_high, 5.0, 100.0))
        if display_percentile_high <= display_percentile_low + 1.0:
            display_percentile_high = min(100.0, display_percentile_low + 1.0)
        if self.display_percentile_low != display_percentile_low:
            self.display_percentile_low = display_percentile_low
            self._slice_dirty = True
        if self.display_percentile_high != display_percentile_high:
            self.display_percentile_high = display_percentile_high
            self._slice_dirty = True

        display_gamma = float(np.clip(display_gamma, 0.4, 3.0))
        if self.display_gamma != display_gamma:
            self.display_gamma = display_gamma
            self._slice_dirty = True

        # Disable manual slider while animating.
        if hasattr(self._viser_controls["slice_position"], "disabled"):
            self._viser_controls["slice_position"].disabled = animate_slice

        if self.animate_slice != animate_slice:
            self.animate_slice = animate_slice
            self._slice_dirty = True

        if not self.animate_slice:
            slice_position = float(self._viser_controls["slice_position"].value)
            if self.slice_position_normalized != slice_position:
                self.slice_position_normalized = slice_position
                self._slice_dirty = True

    def _build_hydroelastic_reference(self, args):
        """Build a tiny hydroelastic scene and fetch the exact immutable field volume."""
        self.mesh.build_sdf(
            max_resolution=int(args.sdf_resolution),
            narrow_band_range=(-float(args.narrow_band), float(args.narrow_band)),
            margin=float(args.sdf_margin),
        )

        cfg_mesh = newton.ModelBuilder.ShapeConfig(
            hydroelastic_type=newton.HydroelasticType.COMPLIANT,
            gap=0.02,
            kh=2.0e8,
            margin=1.0e-5,
        )
        cfg_primitive = newton.ModelBuilder.ShapeConfig(
            hydroelastic_type=newton.HydroelasticType.COMPLIANT,
            sdf_max_resolution=int(args.sdf_resolution),
            sdf_narrow_band_range=(-float(args.narrow_band), float(args.narrow_band)),
            gap=0.02,
            kh=2.0e8,
            margin=1.0e-5,
        )

        builder = newton.ModelBuilder(gravity=0.0)
        body_main = builder.add_body(xform=wp.transform_identity(), label="main_shape")
        self.main_shape_index = builder.add_shape_mesh(body=body_main, mesh=self.mesh, cfg=cfg_mesh)

        # Add a distant compliant shape so hydroelastic pair tables are instantiated.
        body_dummy = builder.add_body(xform=wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity()), label="dummy")
        builder.add_shape_sphere(body=body_dummy, radius=0.1, cfg=cfg_primitive)

        model = builder.finalize(device=self.device)
        pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
        hydro = pipeline.hydroelastic_sdf
        if hydro is None:
            raise RuntimeError("Failed to construct hydroelastic pipeline for immutable pressure field visualization.")
        self._hydro_model = model
        self._hydro_pipeline = pipeline
        self._hydro = hydro

        main_sdf_idx = int(model.shape_sdf_index.numpy()[self.main_shape_index])
        if main_sdf_idx < 0:
            raise RuntimeError("Main shape has no SDF index.")

        self.pressure_volume = hydro.pressure_field_volume[main_sdf_idx]
        if self.pressure_volume is None:
            raise RuntimeError("Hydroelastic immutable pressure volume is missing for main shape.")
        self.pressure_volume_id = wp.uint64(int(self.pressure_volume.id))
        pressure_table = hydro.compact_pressure_field_data.numpy()
        self.pressure_global_max = float(pressure_table[main_sdf_idx]["pressure_max"])
        if self.pressure_global_max <= 1.0e-8:
            raise RuntimeError("Hydroelastic immutable pressure max is zero; expected positive interior pressure.")

        sdf_data = model.sdf_data.numpy()[main_sdf_idx]
        self.sdf_center = wp.vec3(sdf_data["center"])
        self.sdf_half_extents = wp.vec3(sdf_data["half_extents"])
        self.mesh.finalize(device=self.device)

    def _create_shape_mesh(self, args) -> newton.Mesh:
        if args.mesh_file is not None:
            mesh, mesh_name = load_external_mesh_for_hydro(
                args.mesh_file,
                scale=float(args.mesh_scale),
                center_origin=bool(args.mesh_center_origin),
                allow_multiple_components=bool(args.mesh_allow_multiple_components),
                skip_hydro_validation=bool(args.mesh_skip_hydro_validation),
            )
            self.shape_name = mesh_name
            return mesh

        shape = self.shape_name
        if shape == "sphere":
            return newton.Mesh.create_sphere(radius=0.6, compute_inertia=False)
        if shape == "box":
            return newton.Mesh.create_box(hx=0.65, hy=0.45, hz=0.3, compute_inertia=False)
        if shape == "capsule":
            return newton.Mesh.create_capsule(
                radius=0.28,
                half_height=0.5,
                up_axis=newton.Axis.Z,
                compute_inertia=False,
            )
        if shape == "cylinder":
            return newton.Mesh.create_cylinder(
                radius=0.4,
                half_height=0.5,
                up_axis=newton.Axis.Z,
                compute_inertia=False,
            )
        if shape == "cone":
            return newton.Mesh.create_cone(
                radius=0.45,
                half_height=0.55,
                up_axis=newton.Axis.Z,
                compute_inertia=False,
            )
        if shape == "ellipsoid":
            return newton.Mesh.create_ellipsoid(rx=0.7, ry=0.45, rz=0.3, compute_inertia=False)
        raise ValueError(f"Unsupported shape '{shape}'")

    def _axis_half_extent(self) -> float:
        return float(self.sdf_half_extents[self.slice_axis])

    def _apply_axis_sine_modulation(self, axis: int, amplitude: float, cycles: float, phase: float):
        if abs(amplitude) <= 1.0e-8:
            return

        wp.launch(
            kernel=apply_pressure_axis_sine_modulation,
            dim=self.capacity,
            inputs=[
                axis,
                float(self.sdf_center[axis]),
                float(self.sdf_half_extents[axis]),
                amplitude,
                cycles,
                phase,
                self.slice_points,
                self.slice_inside_flag,
            ],
            outputs=[self.slice_pressure],
            device=self.device,
        )

    def _apply_foot_structure_modulation(self):
        if self.pressure_foot_structure_strength <= 1.0e-8:
            return

        wp.launch(
            kernel=apply_pressure_foot_structure_modulation,
            dim=self.capacity,
            inputs=[
                self.sdf_center,
                self.sdf_half_extents,
                self.pressure_foot_structure_strength,
                self.pressure_foot_arch_depth,
                self.pressure_foot_medial_bias,
                self.slice_points,
                self.slice_inside_flag,
            ],
            outputs=[self.slice_pressure],
            device=self.device,
        )

    def _update_slice(self):
        if not self._slice_dirty:
            return

        self.slice_inside_count.zero_()
        axis_position = self.slice_position_normalized * self._axis_half_extent()

        wp.launch(
            kernel=build_slice_points,
            dim=self.capacity,
            inputs=[
                self.resolution,
                self.slice_axis,
                axis_position,
                self.plane_scale,
                self.sdf_center,
                self.sdf_half_extents,
            ],
            outputs=[self.slice_points],
            device=self.device,
        )

        wp.launch(
            kernel=sample_pressure_on_slice,
            dim=self.capacity,
            inputs=[
                self.pressure_volume_id,
                1.0e-8,
                self.slice_points,
            ],
            outputs=[self.slice_inside_flag, self.slice_pressure, self.slice_inside_count],
            device=self.device,
        )

        self._apply_axis_sine_modulation(0, self.pressure_x_sine_amplitude, self.pressure_x_sine_cycles, self.pressure_x_sine_phase)
        self._apply_axis_sine_modulation(1, self.pressure_y_sine_amplitude, self.pressure_y_sine_cycles, self.pressure_y_sine_phase)
        self._apply_axis_sine_modulation(2, self.pressure_z_sine_amplitude, self.pressure_z_sine_cycles, self.pressure_z_sine_phase)
        self._apply_foot_structure_modulation()

        self.last_slice_count = int(self.slice_inside_count.numpy()[0])
        pressure_grid = self.slice_pressure.numpy().reshape(self.resolution, self.resolution)

        inside = pressure_grid >= 0.0
        normalized = normalize_pressure_for_display(
            pressure_grid,
            inside=inside,
            mode=self.display_mode,
            global_max=self.pressure_global_max,
            percentile_low=self.display_percentile_low,
            percentile_high=self.display_percentile_high,
            gamma=self.display_gamma,
        )

        self.last_validation_ok, self.last_deep_p05, self.last_zero_fraction, self.last_hole_fraction = (
            validate_slice_field(pressure_grid)
        )
        self.slice_texture = density_to_rgb_image(normalized)
        if self.viewer_is_viser:
            inside_mask = (self.slice_inside_flag.numpy() > 0.0).reshape(-1)
            points_flat = self.slice_points.numpy().reshape((-1, 3))
            colors_flat = self.slice_texture.reshape((-1, 3))
            self._slice_points_viser = points_flat[inside_mask].astype(np.float32, copy=False)
            self._slice_colors_viser = colors_flat[inside_mask].astype(np.uint8, copy=False)
        self._slice_dirty = False

    def _render_slice_viser(self):
        """Render slice as an in-place updated point cloud on ViewerViser."""
        server = getattr(self.viewer, "_server", None)
        scene = getattr(server, "scene", None)
        if scene is None:
            return

        if (not self.show_slice) or (self._slice_points_viser.shape[0] == 0):
            if self._viser_slice_handle is not None and hasattr(self._viser_slice_handle, "visible"):
                self._viser_slice_handle.visible = False
            return

        axis = self.slice_axis
        half = np.array([float(self.sdf_half_extents[0]), float(self.sdf_half_extents[1]), float(self.sdf_half_extents[2])])
        plane_half = half * self.plane_scale
        if axis == 0:
            du = 2.0 * plane_half[1] / max(self.resolution - 1, 1)
            dv = 2.0 * plane_half[2] / max(self.resolution - 1, 1)
        elif axis == 1:
            du = 2.0 * plane_half[0] / max(self.resolution - 1, 1)
            dv = 2.0 * plane_half[2] / max(self.resolution - 1, 1)
        else:
            du = 2.0 * plane_half[0] / max(self.resolution - 1, 1)
            dv = 2.0 * plane_half[1] / max(self.resolution - 1, 1)
        point_size = float(1.2 * max(du, dv))

        if self._viser_slice_handle is None:
            self._viser_slice_handle = scene.add_point_cloud(
                name="/slice/pressure_cloud",
                points=self._slice_points_viser,
                colors=self._slice_colors_viser,
                point_size=point_size,
                point_shape="circle",
            )
        else:
            self._viser_slice_handle.points = self._slice_points_viser
            self._viser_slice_handle.colors = self._slice_colors_viser
            self._viser_slice_handle.point_size = point_size
            if hasattr(self._viser_slice_handle, "visible"):
                self._viser_slice_handle.visible = True

    def step(self):
        if self.viewer_is_viser:
            self._sync_viser_controls()
        if self.animate_slice:
            self.slice_position_normalized = float(np.sin(self.sim_time * self.slice_speed))
            if self.viewer_is_viser and "slice_position" in self._viser_controls:
                self._viser_controls["slice_position"].value = self.slice_position_normalized
            self._slice_dirty = True
        self.sim_time += self.frame_dt

    def render(self):
        self._update_slice()

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_shapes(
            name="/slice/shape",
            geo_type=newton.GeoType.MESH,
            geo_scale=(1.0, 1.0, 1.0),
            xforms=self.shape_xforms,
            colors=self.shape_colors,
            materials=self.shape_materials,
            geo_src=self.mesh,
            hidden=(not self.show_shape) or self.show_shape_wireframe,
        )

        self.viewer.log_lines(
            name="/slice/shape_wireframe",
            starts=self.shape_line_starts,
            ends=self.shape_line_ends,
            colors=(0.86, 0.86, 0.90),
            width=0.0015,
            hidden=(not self.show_shape) or (not self.show_shape_wireframe),
        )

        if self.viewer_is_viser:
            self._render_slice_viser()
        else:
            self.viewer.log_mesh(
                name="/slice/pressure_heatmap",
                points=self.slice_points,
                indices=self.slice_indices,
                uvs=self.slice_uvs,
                texture=self.slice_texture,
                hidden=not self.show_slice,
                backface_culling=False,
            )

        self.viewer.end_frame()

    def render_ui(self, imgui):
        scale_desc = (
            f"slice percentile [{self.display_percentile_low:.1f}, {self.display_percentile_high:.1f}]"
            if self.display_mode == "slice_percentile"
            else "immutable global max"
        )
        imgui.text("Hydro Pressure Slice")
        imgui.text(f"Shape: {self.shape_name}")
        imgui.text(f"Viewer: {'viser' if self.viewer_is_viser else 'gl-like'}")
        imgui.text("Source: Hydroelastic immutable pressure volume.")
        imgui.text(f"Color scale: {scale_desc}.")
        imgui.text("This is the same field used by contact isosurface extraction.")
        imgui.text("Optional: pressure can be modulated by sine waves and a foot-structure field.")
        imgui.text("Validation: deep interior percentile, zero-fraction, interior-hole fraction.")

        _changed, self.show_shape = imgui.checkbox("Show Shape", self.show_shape)
        _changed, self.show_shape_wireframe = imgui.checkbox("Shape Wireframe", self.show_shape_wireframe)
        changed, self.shape_opacity = imgui.slider_float("Shape Opacity (viser)", self.shape_opacity, 0.0, 1.0)
        if changed:
            self._refresh_shape_materials()
        changed, self.pressure_x_sine_amplitude = imgui.slider_float(
            "Pressure X Sine Amp",
            self.pressure_x_sine_amplitude,
            -2.0,
            2.0,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_x_sine_cycles = imgui.slider_float(
            "Pressure X Sine Cycles",
            self.pressure_x_sine_cycles,
            0.0,
            12.0,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_x_sine_phase = imgui.slider_float(
            "Pressure X Sine Phase [rad]",
            self.pressure_x_sine_phase,
            -np.pi,
            np.pi,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_y_sine_amplitude = imgui.slider_float(
            "Pressure Y Sine Amp",
            self.pressure_y_sine_amplitude,
            -2.0,
            2.0,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_y_sine_cycles = imgui.slider_float(
            "Pressure Y Sine Cycles",
            self.pressure_y_sine_cycles,
            0.0,
            12.0,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_y_sine_phase = imgui.slider_float(
            "Pressure Y Sine Phase [rad]",
            self.pressure_y_sine_phase,
            -np.pi,
            np.pi,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_z_sine_amplitude = imgui.slider_float(
            "Pressure Z Sine Amp",
            self.pressure_z_sine_amplitude,
            -2.0,
            2.0,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_z_sine_cycles = imgui.slider_float(
            "Pressure Z Sine Cycles",
            self.pressure_z_sine_cycles,
            0.0,
            12.0,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_z_sine_phase = imgui.slider_float(
            "Pressure Z Sine Phase [rad]",
            self.pressure_z_sine_phase,
            -np.pi,
            np.pi,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_foot_structure_strength = imgui.slider_float(
            "Pressure Foot Strength",
            self.pressure_foot_structure_strength,
            0.0,
            2.0,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_foot_arch_depth = imgui.slider_float(
            "Pressure Foot Arch Depth",
            self.pressure_foot_arch_depth,
            0.0,
            2.0,
        )
        if changed:
            self._slice_dirty = True
        changed, self.pressure_foot_medial_bias = imgui.slider_float(
            "Pressure Foot Medial Bias",
            self.pressure_foot_medial_bias,
            -1.0,
            1.0,
        )
        if changed:
            self._slice_dirty = True

        imgui.text("Display Scale")
        if imgui.radio_button("Global Max", self.display_mode == "global"):
            self.display_mode = "global"
            self._slice_dirty = True
        if imgui.radio_button("Slice Percentile", self.display_mode == "slice_percentile"):
            self.display_mode = "slice_percentile"
            self._slice_dirty = True
        changed, self.display_percentile_low = imgui.slider_float(
            "Display P Low [%]",
            self.display_percentile_low,
            0.0,
            40.0,
        )
        if changed:
            self.display_percentile_low = float(np.clip(self.display_percentile_low, 0.0, 95.0))
            if self.display_percentile_high <= self.display_percentile_low + 1.0:
                self.display_percentile_high = min(100.0, self.display_percentile_low + 1.0)
            self._slice_dirty = True
        changed, self.display_percentile_high = imgui.slider_float(
            "Display P High [%]",
            self.display_percentile_high,
            60.0,
            100.0,
        )
        if changed:
            self.display_percentile_high = float(np.clip(self.display_percentile_high, 5.0, 100.0))
            if self.display_percentile_high <= self.display_percentile_low + 1.0:
                self.display_percentile_high = min(100.0, self.display_percentile_low + 1.0)
            self._slice_dirty = True
        changed, self.display_gamma = imgui.slider_float(
            "Display Gamma",
            self.display_gamma,
            0.4,
            3.0,
        )
        if changed:
            self.display_gamma = float(np.clip(self.display_gamma, 0.4, 3.0))
            self._slice_dirty = True

        _changed, self.show_slice = imgui.checkbox("Show Slice", self.show_slice)
        changed, self.animate_slice = imgui.checkbox("Animate Slice", self.animate_slice)
        if changed:
            self._slice_dirty = True

        if not self.animate_slice:
            changed, self.slice_position_normalized = imgui.slider_float(
                "Slice Position",
                self.slice_position_normalized,
                -1.0,
                1.0,
            )
            if changed:
                self._slice_dirty = True

        imgui.text("Slice Axis")
        if imgui.radio_button("X", self.slice_axis == 0):
            self.slice_axis = 0
            self._slice_dirty = True
        if imgui.radio_button("Y", self.slice_axis == 1):
            self.slice_axis = 1
            self._slice_dirty = True
        if imgui.radio_button("Z", self.slice_axis == 2):
            self.slice_axis = 2
            self._slice_dirty = True

        imgui.text(f"Inside Samples: {self.last_slice_count}")
        imgui.text(f"Deep Interior P05 (norm): {self.last_deep_p05:.4f}")
        imgui.text(f"Inside Zero Fraction: {self.last_zero_fraction:.4f}")
        imgui.text(f"Interior Hole Fraction: {self.last_hole_fraction:.4f}")
        imgui.text(f"Validation: {'PASS' if self.last_validation_ok else 'FAIL'}")

    def test_final(self):
        self._update_slice()
        assert self.last_slice_count > 0, "Expected non-empty heat-map section for the selected shape."
        assert self.last_validation_ok, "Immutable hydroelastic pressure slice validation failed."


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    parser.add_argument(
        "--shape",
        type=str,
        choices=["sphere", "box", "capsule", "cylinder", "cone", "ellipsoid"],
        default="box",
        help="Primitive shape to section and visualize (ignored when --mesh-file is provided).",
    )
    parser.add_argument(
        "--mesh-file",
        type=str,
        default=None,
        help="Path to an external triangle mesh file (.obj/.stl/etc.) to visualize instead of --shape.",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Uniform scale factor applied to vertices from --mesh-file before SDF construction.",
    )
    parser.add_argument(
        "--mesh-center-origin",
        action="store_true",
        help="Recenter --mesh-file geometry around its AABB center before applying --mesh-scale.",
    )
    parser.add_argument(
        "--mesh-allow-multiple-components",
        action="store_true",
        help="Allow disconnected components when validating --mesh-file hydro readiness.",
    )
    parser.add_argument(
        "--mesh-skip-hydro-validation",
        action="store_true",
        help="Skip watertight/volume validation for --mesh-file (not recommended).",
    )
    parser.add_argument(
        "--shape-opacity",
        type=float,
        default=0.22,
        help="Shape opacity in [0, 1] when using --viewer viser.",
    )
    parser.add_argument(
        "--display-mode",
        type=str,
        choices=["auto", "global", "slice_percentile"],
        default="auto",
        help=(
            "Display normalization mode. "
            "'global' uses immutable pressure max, "
            "'slice_percentile' auto-ranges each slice, "
            "'auto' selects slice_percentile for --mesh-file and global otherwise."
        ),
    )
    parser.add_argument(
        "--display-percentile-low",
        type=float,
        default=2.0,
        help="Lower percentile for --display-mode slice_percentile.",
    )
    parser.add_argument(
        "--display-percentile-high",
        type=float,
        default=98.0,
        help="Upper percentile for --display-mode slice_percentile.",
    )
    parser.add_argument(
        "--display-gamma",
        type=float,
        default=1.35,
        help="Gamma applied to display-normalized density (visualization only).",
    )
    parser.add_argument(
        "--pressure-x-sine-amplitude",
        type=float,
        default=0.0,
        help="Amplitude for pressure sine modulation along x (0 disables modulation).",
    )
    parser.add_argument(
        "--pressure-x-sine-cycles",
        type=float,
        default=1.0,
        help="Sine cycles across the full x extent for pressure modulation.",
    )
    parser.add_argument(
        "--pressure-x-sine-phase",
        type=float,
        default=0.0,
        help="Phase offset [rad] for pressure x sine modulation.",
    )
    parser.add_argument(
        "--pressure-y-sine-amplitude",
        type=float,
        default=0.0,
        help="Amplitude for pressure sine modulation along y (0 disables modulation).",
    )
    parser.add_argument(
        "--pressure-y-sine-cycles",
        type=float,
        default=1.0,
        help="Sine cycles across the full y extent for pressure modulation.",
    )
    parser.add_argument(
        "--pressure-y-sine-phase",
        type=float,
        default=0.0,
        help="Phase offset [rad] for pressure y sine modulation.",
    )
    parser.add_argument(
        "--pressure-z-sine-amplitude",
        type=float,
        default=0.0,
        help="Amplitude for pressure sine modulation along z (0 disables modulation).",
    )
    parser.add_argument(
        "--pressure-z-sine-cycles",
        type=float,
        default=1.0,
        help="Sine cycles across the full z extent for pressure modulation.",
    )
    parser.add_argument(
        "--pressure-z-sine-phase",
        type=float,
        default=0.0,
        help="Phase offset [rad] for pressure z sine modulation.",
    )
    parser.add_argument(
        "--pressure-foot-structure-strength",
        type=float,
        default=0.0,
        help="Overall strength for localized foot-bone/arch pressure modulation (0 disables).",
    )
    parser.add_argument(
        "--pressure-foot-arch-depth",
        type=float,
        default=1.0,
        help="Arch trough depth multiplier for foot-structure modulation.",
    )
    parser.add_argument(
        "--pressure-foot-medial-bias",
        type=float,
        default=0.3,
        help="Medial (+) / lateral (-) bias for the foot-structure modulation in normalized y.",
    )
    parser.add_argument(
        "--foot-pressure-preset",
        action="store_true",
        help=(
            "Apply a plantar-pressure visualization preset "
            "(foot structure modulation + default sole slice depth)."
        ),
    )
    parser.add_argument(
        "--slice-axis",
        type=str,
        choices=["x", "y", "z"],
        default="z",
        help="Axis normal of the slice plane.",
    )
    parser.add_argument(
        "--slice-position",
        type=float,
        default=0.0,
        help="Initial normalized slice position in [-1, 1].",
    )
    parser.add_argument(
        "--animate-slice",
        action="store_true",
        help="Animate slice position sinusoidally through the shape.",
    )
    parser.add_argument(
        "--slice-speed",
        type=float,
        default=1.5,
        help="Animation speed [rad/s] for --animate-slice.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Slice grid resolution per axis (total samples = resolution^2).",
    )
    parser.add_argument(
        "--plane-scale",
        type=float,
        default=1.02,
        help="Scale factor for section sampling extents relative to SDF bounds.",
    )
    parser.add_argument(
        "--sdf-resolution",
        type=int,
        default=96,
        help="SDF max resolution (must be divisible by 8).",
    )
    parser.add_argument(
        "--narrow-band",
        type=float,
        default=0.1,
        help="Half-width [m] for SDF narrow band range (-band, +band).",
    )
    parser.add_argument(
        "--sdf-margin",
        type=float,
        default=0.1,
        help="Padding [m] around the shape during SDF build.",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
