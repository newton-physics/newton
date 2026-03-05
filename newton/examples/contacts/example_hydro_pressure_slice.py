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
# Visualize immutable hydroelastic pressure by slicing the loaded shape and
# render hydro contact surfaces from the actual collision pipeline.
#
# Command: uv run -m newton.examples.contacts.example_hydro_pressure_slice --shape box
#
###########################################################################

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.usd
from newton.geometry import HydroelasticSDF


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


def _load_trimesh_for_hydro(mesh_file: str) -> Any:
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


def _find_first_usd_mesh_prim(stage: Usd.Stage, asset_name: str):
    """Return the first mesh prim from a USD stage."""
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            return prim
    raise ValueError(f"No mesh prim found in USD asset: {asset_name}")


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
        self.clip_slice_to_sdf = bool(self.has_external_mesh or self.shape_name == "foot1")
        self.plane_scale = float(args.plane_scale)
        self.resolution = int(args.resolution)
        self.shape_opacity = float(args.shape_opacity)

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

        self.show_shape = self.viewer_is_viser
        self.show_shape_wireframe = not self.viewer_is_viser
        self.show_slice = True
        self.show_contact_surface = True

        self.mesh = self._create_shape_mesh(args)
        self._build_hydroelastic_reference(args)
        if hasattr(self.viewer, "set_model"):
            self.viewer.set_model(self._hydro_model)
        if hasattr(self.viewer, "show_hydro_contact_surface"):
            self.viewer.show_hydro_contact_surface = self.show_contact_surface

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
        self.last_contact_face_count = 0
        self.last_penetrating_face_count = 0

        self.slice_indices = wp.array(build_regular_grid_indices(self.resolution), dtype=wp.int32, device=self.device)
        self.slice_uvs = wp.array(build_regular_grid_uvs(self.resolution), dtype=wp.vec2, device=self.device)
        self.slice_texture = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        self._slice_points_viser = np.zeros((0, 3), dtype=np.float32)
        self._slice_colors_viser = np.zeros((0, 3), dtype=np.uint8)
        self._viser_slice_handle = None

        self.shape_xforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self.shape_colors = wp.array([wp.vec3(0.72, 0.72, 0.76)], dtype=wp.vec3, device=self.device)
        self.support_xforms = wp.array(
            [wp.transform(self.support_center, wp.quat_identity())],
            dtype=wp.transform,
            device=self.device,
        )
        self.support_colors = wp.array([wp.vec3(0.35, 0.38, 0.42)], dtype=wp.vec3, device=self.device)
        self.shape_materials = wp.array([wp.vec4(0.75, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device)
        self.support_materials = wp.array([wp.vec4(0.55, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device)
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
        self._update_hydro_contacts()

    def _refresh_shape_materials(self):
        alpha = np.clip(self.shape_opacity, 0.0, 1.0) if self.viewer_is_viser else 0.0
        self.shape_materials = wp.array([wp.vec4(0.75, 0.0, 0.0, float(alpha))], dtype=wp.vec4, device=self.device)
        support_alpha = 0.45 * alpha if self.viewer_is_viser else 0.0
        self.support_materials = wp.array(
            [wp.vec4(0.55, 0.0, 0.0, float(support_alpha))],
            dtype=wp.vec4,
            device=self.device,
        )

    def _slice_axis_options(self) -> list[tuple[str, int]]:
        if self.shape_name == "foot1":
            return [
                ("Coronal (X normal / Y-Z plane)", 0),
                ("Sagittal (Y normal / X-Z plane)", 1),
                ("Axial (Z normal / X-Y plane)", 2),
            ]
        return [("X", 0), ("Y", 1), ("Z", 2)]

    def _slice_axis_display_label(self) -> str:
        for label, axis in self._slice_axis_options():
            if axis == self.slice_axis:
                return label
        return {0: "X", 1: "Y", 2: "Z"}[self.slice_axis]

    def _slice_axis_from_label(self, label: str) -> int:
        label_norm = label.strip().lower()
        for option_label, axis in self._slice_axis_options():
            if option_label.lower() == label_norm:
                return axis
        return {"x": 0, "y": 1, "z": 2}.get(label_norm, self.slice_axis)

    def _slice_plane_basis_labels(self) -> tuple[str, str, str]:
        if self.slice_axis == 0:
            return "+Y", "+Z", "+X"
        if self.slice_axis == 1:
            return "+X", "+Z", "+Y"
        return "+X", "+Y", "+Z"

    def _slice_plane_gizmo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = np.array(
            [float(self.sdf_center[0]), float(self.sdf_center[1]), float(self.sdf_center[2])],
            dtype=np.float32,
        )
        center[self.slice_axis] += self.slice_position_normalized * self._axis_half_extent()

        half = np.array(
            [float(self.sdf_half_extents[0]), float(self.sdf_half_extents[1]), float(self.sdf_half_extents[2])],
            dtype=np.float32,
        )
        plane_half = half * self.plane_scale

        if self.slice_axis == 0:
            u_axis, v_axis = 1, 2
        elif self.slice_axis == 1:
            u_axis, v_axis = 0, 2
        else:
            u_axis, v_axis = 0, 1

        basis_u = np.zeros(3, dtype=np.float32)
        basis_v = np.zeros(3, dtype=np.float32)
        basis_n = np.zeros(3, dtype=np.float32)
        basis_u[u_axis] = 1.0
        basis_v[v_axis] = 1.0
        basis_n[self.slice_axis] = 1.0

        u_len = max(0.06, 0.18 * float(plane_half[u_axis]))
        v_len = max(0.06, 0.18 * float(plane_half[v_axis]))
        n_len = max(0.05, 0.16 * float(half[self.slice_axis]))

        starts = np.stack([center, center, center], axis=0).astype(np.float32, copy=False)
        ends = np.stack(
            [
                center + u_len * basis_u,
                center + v_len * basis_v,
                center + n_len * basis_n,
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        colors = np.array(
            [
                [0.95, 0.25, 0.25],
                [0.25, 0.90, 0.25],
                [0.25, 0.55, 0.95],
            ],
            dtype=np.float32,
        )
        return starts, ends, colors

    def _setup_viser_controls(self):
        """Create native viser GUI controls when running with ViewerViser."""
        server = getattr(self.viewer, "_server", None)
        gui = getattr(server, "gui", None)
        if gui is None:
            return

        axis_label = self._slice_axis_display_label()
        with gui.add_folder("Hydro Pressure Slice"):
            self._viser_controls["show_shape"] = gui.add_checkbox("Show Shape", self.show_shape)
            self._viser_controls["show_shape_wireframe"] = gui.add_checkbox(
                "Shape Wireframe", self.show_shape_wireframe
            )
            self._viser_controls["show_slice"] = gui.add_checkbox("Show Slice", self.show_slice)
            self._viser_controls["show_contact_surface"] = gui.add_checkbox(
                "Show Contact Surface", self.show_contact_surface
            )
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
                options=tuple(label for label, _axis in self._slice_axis_options()),
                initial_value=axis_label,
            )
            self._viser_controls["shape_opacity"] = gui.add_slider(
                "Shape Opacity",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=float(self.shape_opacity),
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
        show_contact_surface = bool(self._viser_controls["show_contact_surface"].value)
        animate_slice = bool(self._viser_controls["animate_slice"].value)
        axis_label = str(self._viser_controls["slice_axis"].value)
        axis = self._slice_axis_from_label(axis_label)
        shape_opacity = float(self._viser_controls["shape_opacity"].value)
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
        if self.show_contact_surface != show_contact_surface:
            self.show_contact_surface = show_contact_surface
            if hasattr(self.viewer, "show_hydro_contact_surface"):
                self.viewer.show_hydro_contact_surface = self.show_contact_surface
        if self.slice_axis != axis:
            self.slice_axis = axis
            self._slice_dirty = True
        if self.shape_opacity != shape_opacity:
            self.shape_opacity = shape_opacity
            self._refresh_shape_materials()
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
        """Build a hydroelastic scene and fetch immutable pressure/contact data."""
        self.mesh.build_sdf(
            max_resolution=int(args.sdf_resolution),
            narrow_band_range=(-float(args.narrow_band), float(args.narrow_band)),
            margin=float(args.sdf_margin),
        )

        v_min, v_max = self._mesh_bounds()
        extents = np.maximum(v_max - v_min, 1.0e-4)
        support_hx = max(0.20, 0.75 * float(extents[0]))
        support_hy = max(0.20, 0.75 * float(extents[1]))
        support_hz = max(0.05, 0.20 * float(extents[2]))
        overlap = max(0.003, 0.02 * float(extents[2]))
        support_center_xy = 0.5 * (v_min[:2] + v_max[:2])
        support_top = float(v_min[2]) + overlap
        support_center = wp.vec3(
            float(support_center_xy[0]),
            float(support_center_xy[1]),
            support_top - support_hz,
        )
        self.support_center = support_center
        self.support_half_extents = (support_hx, support_hy, support_hz)

        cfg_main = newton.ModelBuilder.ShapeConfig(
            hydroelastic_type=newton.HydroelasticType.COMPLIANT,
            hydroelastic_contact_workflow=newton.HydroelasticContactWorkflow.PRESSURE,
            hydro_pressure_sine_amplitude=(0.0, 0.0, 0.0),
            hydro_pressure_sine_cycles=(1.0, 1.0, 1.0),
            hydro_pressure_sine_phase=(0.0, 0.0, 0.0),
            gap=0.02,
            kh=2.0e8,
            margin=1.0e-5,
        )
        cfg_support = newton.ModelBuilder.ShapeConfig(
            hydroelastic_type=newton.HydroelasticType.RIGID,
            hydroelastic_contact_workflow=newton.HydroelasticContactWorkflow.CLASSIC,
            hydro_pressure_sine_amplitude=(0.0, 0.0, 0.0),
            hydro_pressure_sine_cycles=(1.0, 1.0, 1.0),
            hydro_pressure_sine_phase=(0.0, 0.0, 0.0),
            sdf_max_resolution=int(args.sdf_resolution),
            sdf_narrow_band_range=(-float(args.narrow_band), float(args.narrow_band)),
            gap=0.02,
            kh=2.0e8,
            margin=1.0e-5,
        )

        builder = newton.ModelBuilder(gravity=0.0)
        body_main = builder.add_body(xform=wp.transform_identity(), label="main_shape")
        self.main_shape_index = builder.add_shape_mesh(body=body_main, mesh=self.mesh, cfg=cfg_main)

        body_support = builder.add_body(xform=wp.transform(support_center, wp.quat_identity()), label="support_shape")
        self.support_shape_index = builder.add_shape_box(
            body=body_support,
            hx=support_hx,
            hy=support_hy,
            hz=support_hz,
            cfg=cfg_support,
        )

        model = builder.finalize(device=self.device)
        sdf_config = HydroelasticSDF.Config(output_contact_surface=True, buffer_fraction=1.0)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_config,
        )
        hydro = pipeline.hydroelastic_sdf
        if hydro is None:
            raise RuntimeError("Failed to construct hydroelastic pipeline for immutable pressure field visualization.")

        self._hydro_model = model
        self._hydro_pipeline = pipeline
        self._hydro = hydro
        self._hydro_state = model.state()
        self._hydro_contacts = pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self._hydro_state)
        self._hydro_pipeline.collide(self._hydro_state, self._hydro_contacts)

        main_sdf_idx = int(model.shape_sdf_index.numpy()[self.main_shape_index])
        if main_sdf_idx < 0:
            raise RuntimeError("Main shape has no SDF index.")

        main_pressure_idx = int(hydro.shape_pressure_index.numpy()[self.main_shape_index])
        if main_pressure_idx < 0:
            raise RuntimeError("Main compliant shape has no immutable pressure profile.")

        pressure_table = hydro.compact_pressure_field_data.numpy()
        pressure_ptr = int(pressure_table[main_pressure_idx]["pressure_ptr"])
        if pressure_ptr == 0:
            raise RuntimeError("Hydroelastic immutable pressure volume pointer is zero for main shape.")
        self.pressure_volume_id = wp.uint64(pressure_ptr)

        self.pressure_global_max = float(pressure_table[main_pressure_idx]["pressure_max"])
        if self.pressure_global_max <= 1.0e-8:
            raise RuntimeError("Hydroelastic immutable pressure max is zero; expected positive interior pressure.")

        sdf_data = model.sdf_data.numpy()[main_sdf_idx]
        self.sdf_sparse_volume_id = wp.uint64(int(sdf_data["sparse_sdf_ptr"]))
        self.sdf_coarse_volume_id = wp.uint64(int(sdf_data["coarse_sdf_ptr"]))
        self.sdf_background_value = float(sdf_data["background_value"])
        self.sdf_center = wp.vec3(sdf_data["center"])
        self.sdf_half_extents = wp.vec3(sdf_data["half_extents"])
        self.mesh.finalize(device=self.device)

    def _mesh_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected mesh vertices with shape [N, 3], got {vertices.shape}.")
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    def _update_hydro_contacts(self):
        self._hydro_pipeline.collide(self._hydro_state, self._hydro_contacts)
        contact_surface = self._hydro.get_contact_surface()
        if contact_surface is None:
            self.last_contact_face_count = 0
            self.last_penetrating_face_count = 0
            return None

        num_faces = int(contact_surface.face_contact_count.numpy()[0])
        self.last_contact_face_count = num_faces
        if num_faces <= 0:
            self.last_penetrating_face_count = 0
            return contact_surface

        depths = contact_surface.contact_surface_depth.numpy()[:num_faces]
        self.last_penetrating_face_count = int(np.sum(depths < 0.0))
        return contact_surface

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
        if shape == "foot1":
            usd_path = newton.examples.get_asset("foot1.usd")
            stage = Usd.Stage.Open(usd_path)
            if stage is None:
                raise RuntimeError(f"Failed to open USD asset: {usd_path}")
            prim = _find_first_usd_mesh_prim(stage, "foot1.usd")
            src_mesh = newton.usd.get_mesh(prim)
            # Scale to ~1 m (comparable to other built-in shapes) and center at origin.
            vertices = np.asarray(src_mesh.vertices, dtype=np.float32)
            center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
            vertices = (vertices - center) * 5.0
            indices = np.asarray(src_mesh.indices, dtype=np.int32)
            return newton.Mesh(vertices, indices, compute_inertia=False)
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
                int(self.clip_slice_to_sdf),
                self.pressure_volume_id,
                self.sdf_sparse_volume_id,
                self.sdf_coarse_volume_id,
                self.sdf_background_value,
                self.sdf_center,
                self.sdf_half_extents,
                1.0e-8,
                self.slice_points,
            ],
            outputs=[self.slice_inside_flag, self.slice_pressure, self.slice_inside_count],
            device=self.device,
        )

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

        _, self.last_deep_p05, self.last_zero_fraction, self.last_hole_fraction = validate_slice_field(pressure_grid)
        self.last_validation_ok = validate_slice_metrics(
            self.last_deep_p05,
            self.last_zero_fraction,
            self.last_hole_fraction,
            clipped_mesh_slice=self.clip_slice_to_sdf,
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
        half = np.array(
            [float(self.sdf_half_extents[0]), float(self.sdf_half_extents[1]), float(self.sdf_half_extents[2])]
        )
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
        contact_surface = self._update_hydro_contacts()

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
        self.viewer.log_shapes(
            name="/slice/support_shape",
            geo_type=newton.GeoType.BOX,
            geo_scale=self.support_half_extents,
            xforms=self.support_xforms,
            colors=self.support_colors,
            materials=self.support_materials,
            hidden=not self.show_shape,
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
            # Flip texture vertically: the GL/Rerun viewers flip textures on
            # upload (OpenGL convention), but slice_texture rows already match
            # the UV v-coordinate, so pre-flip to cancel the viewer's flip.
            self.viewer.log_mesh(
                name="/slice/pressure_heatmap",
                points=self.slice_points,
                indices=self.slice_indices,
                uvs=self.slice_uvs,
                texture=np.flipud(self.slice_texture),
                hidden=not self.show_slice,
                backface_culling=False,
            )
        gizmo_starts, gizmo_ends, gizmo_colors = self._slice_plane_gizmo()
        self.viewer.log_lines(
            name="/slice/plane_basis",
            starts=wp.array(gizmo_starts, dtype=wp.vec3, device=self.device),
            ends=wp.array(gizmo_ends, dtype=wp.vec3, device=self.device),
            colors=wp.array(gizmo_colors, dtype=wp.vec3, device=self.device),
            width=0.0035,
            hidden=not self.show_slice,
        )

        if hasattr(self.viewer, "show_hydro_contact_surface"):
            self.viewer.show_hydro_contact_surface = self.show_contact_surface
        if hasattr(self.viewer, "log_hydro_contact_surface"):
            self.viewer.log_hydro_contact_surface(
                contact_surface if self.show_contact_surface else None,
                penetrating_only=True,
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
        imgui.text("Support: rigid box")
        imgui.text(f"Viewer: {'viser' if self.viewer_is_viser else 'gl-like'}")
        imgui.text("Source: immutable hydro pressure field + hydro contact surface.")
        imgui.text(f"Color scale: {scale_desc}.")
        imgui.text("Pressure profile is fixed at build time in ShapeConfig.")
        imgui.text("Validation: deep interior percentile, zero-fraction, interior-hole fraction.")
        imgui.text(f"Slice Plane: {self._slice_axis_display_label()}")
        u_label, v_label, n_label = self._slice_plane_basis_labels()
        imgui.text(f"Slice Basis: red=U {u_label}, green=V {v_label}, blue=N {n_label}")
        if self.shape_name == "foot1":
            imgui.text("Foot1 planes: Coronal=Y-Z, Sagittal=X-Z, Axial=X-Y.")

        _changed, self.show_shape = imgui.checkbox("Show Shape", self.show_shape)
        _changed, self.show_shape_wireframe = imgui.checkbox("Shape Wireframe", self.show_shape_wireframe)
        _changed, self.show_slice = imgui.checkbox("Show Slice", self.show_slice)
        changed, self.show_contact_surface = imgui.checkbox("Show Contact Surface", self.show_contact_surface)
        if changed and hasattr(self.viewer, "show_hydro_contact_surface"):
            self.viewer.show_hydro_contact_surface = self.show_contact_surface
        changed, self.shape_opacity = imgui.slider_float("Shape Opacity (viser)", self.shape_opacity, 0.0, 1.0)
        if changed:
            self._refresh_shape_materials()

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
        for label, axis in self._slice_axis_options():
            if imgui.radio_button(label, self.slice_axis == axis):
                self.slice_axis = axis
                self._slice_dirty = True

        imgui.text(f"Inside Samples: {self.last_slice_count}")
        imgui.text(f"Deep Interior P05 (norm): {self.last_deep_p05:.4f}")
        imgui.text(f"Inside Zero Fraction: {self.last_zero_fraction:.4f}")
        imgui.text(f"Interior Hole Fraction: {self.last_hole_fraction:.4f}")
        imgui.text(f"Validation: {'PASS' if self.last_validation_ok else 'FAIL'}")
        imgui.text(f"Contact Faces: {self.last_contact_face_count}")
        imgui.text(f"Penetrating Faces: {self.last_penetrating_face_count}")

    def test_final(self):
        self._update_slice()
        contact_surface = self._update_hydro_contacts()
        assert self.last_slice_count > 0, "Expected non-empty heat-map section for the selected shape."
        assert validate_slice_metrics(
            self.last_deep_p05,
            self.last_zero_fraction,
            self.last_hole_fraction,
            clipped_mesh_slice=self.clip_slice_to_sdf,
        ), "Immutable hydroelastic pressure slice validation failed."
        assert contact_surface is not None, "Expected hydro contact-surface output."
        assert self.last_contact_face_count > 0, "Expected at least one hydro contact-surface face."
        assert self.last_penetrating_face_count > 0, "Expected at least one penetrating hydro contact-surface face."


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    parser.add_argument(
        "--shape",
        type=str,
        choices=["sphere", "box", "capsule", "cylinder", "cone", "ellipsoid", "foot1"],
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
