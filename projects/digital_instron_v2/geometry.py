# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometry conditioning and grid generation for Digital Instron v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MeshQCError(RuntimeError):
    """Raised when midsole geometry is not fit for calibration."""


@dataclass(frozen=True)
class CylinderGrid:
    """Uniform vertical foundation grid for a circular indenter."""

    xy_m: np.ndarray
    cell_area_m2: float
    spacing_m: float
    radius_m: float
    neighbors: np.ndarray


@dataclass(frozen=True)
class MeshFrame:
    """Detected axis frame for mesh ray probing."""

    plane_axes: tuple[int, int]
    thickness_axis: int
    center_m: np.ndarray
    extents_m: np.ndarray


@dataclass(frozen=True)
class SpringSurfaceGrid:
    """Raycast-derived spring grid over the midsole footprint."""

    xy_m: np.ndarray
    grid_uv_m: np.ndarray
    slack_length_m: np.ndarray
    bottom_m: np.ndarray
    top_m: np.ndarray
    hit_count: np.ndarray
    cell_area_m2: float
    spacing_m: float
    frame: MeshFrame
    neighbors: np.ndarray


@dataclass(frozen=True)
class PlacedCylinderGrid:
    """Circular indenter grid placed in mesh footprint coordinates."""

    local_xy_m: np.ndarray
    grid_uv_m: np.ndarray
    center_uv_m: np.ndarray
    cell_area_m2: float
    spacing_m: float
    radius_m: float
    frame: MeshFrame
    neighbors: np.ndarray


def compute_grid_neighbors(xy_m: np.ndarray, spacing_m: float) -> np.ndarray:
    """For each point in xy_m, find indices of left, right, bottom, top neighbors.

    Returns an (N, 4) int32 array. Index is -1 if no neighbor exists.
    """
    n = len(xy_m)
    neighbors = np.full((n, 4), -1, dtype=np.int32)
    h = spacing_m
    if h <= 0.0:
        return neighbors

    coord_map = {(round(x / h), round(y / h)): idx for idx, (x, y) in enumerate(xy_m)}

    for idx, (x, y) in enumerate(xy_m):
        ux, uy = round(x / h), round(y / h)
        neighbors[idx, 0] = coord_map.get((ux - 1, uy), -1)
        neighbors[idx, 1] = coord_map.get((ux + 1, uy), -1)
        neighbors[idx, 2] = coord_map.get((ux, uy - 1), -1)
        neighbors[idx, 3] = coord_map.get((ux, uy + 1), -1)

    return neighbors


def make_cylinder_grid(radius_m: float, spacing_m: float = 0.005) -> CylinderGrid:
    """Build a uniform 2D circular grid for localized punch trials."""

    if radius_m <= 0.0:
        raise ValueError("Cylinder radius must be positive")
    if spacing_m <= 0.0:
        raise ValueError("Grid spacing must be positive")

    axis = np.arange(-radius_m, radius_m + spacing_m * 0.5, spacing_m, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = np.linalg.norm(points, axis=1) <= radius_m + spacing_m * 1.0e-9
    xy = np.ascontiguousarray(points[inside], dtype=np.float64)
    if len(xy) == 0:
        raise ValueError("Cylinder grid has no cells; reduce spacing or increase radius")
    neighbors = compute_grid_neighbors(xy, spacing_m)
    return CylinderGrid(
        xy_m=xy,
        cell_area_m2=spacing_m * spacing_m,
        spacing_m=spacing_m,
        radius_m=radius_m,
        neighbors=neighbors,
    )


def detect_mesh_frame(
    vertices_m: np.ndarray,
    thickness_axis: int | None = None,
) -> MeshFrame:
    """Infer thickness axis as the shortest mesh extent.

    Args:
        vertices_m: Nx3 vertex positions.
        thickness_axis: Override the auto-detected thickness axis. If
            ``None``, the axis with the smallest extent is used.

    Returns:
        Detected :class:`MeshFrame`.
    """

    if thickness_axis is not None and thickness_axis not in (0, 1, 2):
        raise ValueError(f"thickness_axis must be 0, 1, or 2, got {thickness_axis}")
    if thickness_axis is None:
        extents = np.ptp(vertices_m, axis=0)
        thickness_axis = int(np.argmin(extents))
    else:
        extents = np.ptp(vertices_m, axis=0)
    plane_axes = tuple(axis for axis in range(3) if axis != thickness_axis)
    return MeshFrame(
        plane_axes=(int(plane_axes[0]), int(plane_axes[1])),
        thickness_axis=int(thickness_axis),
        center_m=np.mean(vertices_m, axis=0),
        extents_m=extents,
    )


def make_footprint_grid(
    vertices_m: np.ndarray,
    spacing_m: float = 0.005,
    *,
    thickness_axis: int | None = None,
) -> tuple[np.ndarray, MeshFrame]:
    """Build a rectangular grid over the detected midsole footprint bounds."""

    if spacing_m <= 0.0:
        raise ValueError("Grid spacing must be positive")
    frame = detect_mesh_frame(vertices_m, thickness_axis=thickness_axis)
    plane = vertices_m[:, frame.plane_axes]
    mins = np.min(plane, axis=0)
    maxs = np.max(plane, axis=0)
    u = np.arange(mins[0], maxs[0] + spacing_m * 0.5, spacing_m, dtype=np.float64)
    v = np.arange(mins[1], maxs[1] + spacing_m * 0.5, spacing_m, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    grid_uv = np.column_stack((uu.ravel(), vv.ravel()))
    if len(grid_uv) == 0:
        raise ValueError("Footprint grid has no cells")
    return np.ascontiguousarray(grid_uv, dtype=np.float64), frame


def rearfoot_punch_center_uv(
    vertices_m: np.ndarray,
    frame: MeshFrame | None = None,
    *,
    heel_side: str = "min",
    length_fraction: float = 0.25,
    lateral_fraction: float = 0.5,
    lateral_band_fraction: float = 0.12,
) -> np.ndarray:
    """Place rearfoot punch center from mesh footprint bounds.

    The detected longer footprint axis is treated as shoe length. ``heel_side``
    selects which end is posterior. Fractions are measured from the heel-side
    end along length and from min-to-max along width.
    """

    mesh_frame = frame if frame is not None else detect_mesh_frame(vertices_m)
    if heel_side not in {"min", "max"}:
        raise ValueError("heel_side must be 'min' or 'max'")
    if not 0.0 <= length_fraction <= 1.0:
        raise ValueError("length_fraction must be in [0, 1]")
    if not 0.0 <= lateral_fraction <= 1.0:
        raise ValueError("lateral_fraction must be in [0, 1]")
    if lateral_band_fraction < 0.0:
        raise ValueError("lateral_band_fraction must be non-negative")

    plane = vertices_m[:, mesh_frame.plane_axes]
    mins = np.min(plane, axis=0)
    maxs = np.max(plane, axis=0)
    extents = maxs - mins
    length_index = int(np.argmax(extents))
    width_index = 1 - length_index

    center = np.empty(2, dtype=np.float64)
    if heel_side == "min":
        center[length_index] = mins[length_index] + length_fraction * extents[length_index]
    else:
        center[length_index] = maxs[length_index] - length_fraction * extents[length_index]

    half_band = max(extents[length_index] * lateral_band_fraction * 0.5, 1.0e-9)
    local = plane[np.abs(plane[:, length_index] - center[length_index]) <= half_band]
    if len(local) < 3:
        local = plane
    local_min = float(np.min(local[:, width_index]))
    local_max = float(np.max(local[:, width_index]))
    center[width_index] = local_min + lateral_fraction * (local_max - local_min)
    return center


def place_rearfoot_punch_grid(
    vertices_m: np.ndarray,
    *,
    radius_m: float,
    spacing_m: float = 0.005,
    frame: MeshFrame | None = None,
    heel_side: str = "min",
    length_fraction: float = 0.25,
    lateral_fraction: float = 0.5,
    lateral_band_fraction: float = 0.12,
) -> PlacedCylinderGrid:
    """Build a rearfoot punch grid placed on the mesh footprint."""

    mesh_frame = frame if frame is not None else detect_mesh_frame(vertices_m)
    punch = make_cylinder_grid(radius_m=radius_m, spacing_m=spacing_m)
    center = rearfoot_punch_center_uv(
        vertices_m,
        mesh_frame,
        heel_side=heel_side,
        length_fraction=length_fraction,
        lateral_fraction=lateral_fraction,
        lateral_band_fraction=lateral_band_fraction,
    )
    return PlacedCylinderGrid(
        local_xy_m=punch.xy_m,
        grid_uv_m=np.ascontiguousarray(punch.xy_m + center, dtype=np.float64),
        center_uv_m=center,
        cell_area_m2=punch.cell_area_m2,
        spacing_m=punch.spacing_m,
        radius_m=punch.radius_m,
        frame=mesh_frame,
        neighbors=punch.neighbors,
    )


def _ray_triangle_z_candidates(
    point_uv: np.ndarray,
    triangles_plane: np.ndarray,
    triangles_thickness: np.ndarray,
    candidate_indices: np.ndarray,
    eps: float = 1.0e-12,
) -> list[float]:
    if len(candidate_indices) == 0:
        return []
    tri_xy = triangles_plane[candidate_indices]
    tri_z = triangles_thickness[candidate_indices]
    px, py = float(point_uv[0]), float(point_uv[1])
    x0 = tri_xy[:, 0, 0]
    y0 = tri_xy[:, 0, 1]
    x1 = tri_xy[:, 1, 0]
    y1 = tri_xy[:, 1, 1]
    x2 = tri_xy[:, 2, 0]
    y2 = tri_xy[:, 2, 1]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegenerate = np.abs(denom) >= eps
    if not np.any(nondegenerate):
        return []
    denom = denom[nondegenerate]
    tri_z = tri_z[nondegenerate]
    x0 = x0[nondegenerate]
    y0 = y0[nondegenerate]
    x1 = x1[nondegenerate]
    y1 = y1[nondegenerate]
    x2 = x2[nondegenerate]
    y2 = y2[nondegenerate]
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1.0e-9) & (w1 >= -1.0e-9) & (w2 >= -1.0e-9)
    if not np.any(inside):
        return []
    hits = w0[inside] * tri_z[inside, 0] + w1[inside] * tri_z[inside, 1] + w2[inside] * tri_z[inside, 2]
    hits.sort()
    deduped: list[float] = []
    for value in hits:
        hit = float(value)
        if not deduped or abs(hit - deduped[-1]) > 1.0e-6:
            deduped.append(hit)
    return deduped


def raycast_grid_thickness(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    grid_uv_m: np.ndarray,
    frame: MeshFrame | None = None,
) -> dict[str, np.ndarray]:
    """Probe mesh thickness by casting rays along the detected thickness axis."""

    mesh_frame = frame if frame is not None else detect_mesh_frame(vertices_m)
    plane = vertices_m[:, mesh_frame.plane_axes]
    thickness_coord = vertices_m[:, mesh_frame.thickness_axis]
    triangles_plane = plane[faces.reshape(-1, 3)]
    triangles_thickness = thickness_coord[faces.reshape(-1, 3)]
    tri_min = np.min(triangles_plane, axis=1) - 1.0e-12
    tri_max = np.max(triangles_plane, axis=1) + 1.0e-12
    grid_uv = np.asarray(grid_uv_m, dtype=np.float64)
    if grid_uv.ndim != 2 or grid_uv.shape[1] != 2:
        raise ValueError("grid_uv_m must have shape (n, 2)")

    bottom = np.full(len(grid_uv), np.nan, dtype=np.float64)
    top = np.full(len(grid_uv), np.nan, dtype=np.float64)
    hit_count = np.zeros(len(grid_uv), dtype=np.int32)
    for i, point in enumerate(grid_uv):
        candidate_indices = np.nonzero(
            (tri_min[:, 0] <= point[0])
            & (point[0] <= tri_max[:, 0])
            & (tri_min[:, 1] <= point[1])
            & (point[1] <= tri_max[:, 1])
        )[0]
        hits = _ray_triangle_z_candidates(point, triangles_plane, triangles_thickness, candidate_indices)
        hit_count[i] = len(hits)
        if len(hits) >= 2:
            bottom[i] = hits[0]
            top[i] = hits[-1]

    return {
        "grid_uv_m": grid_uv,
        "bottom_m": bottom,
        "top_m": top,
        "thickness_m": top - bottom,
        "hit_count": hit_count,
        "plane_axes": np.asarray(mesh_frame.plane_axes, dtype=np.int32),
        "thickness_axis": np.asarray([mesh_frame.thickness_axis], dtype=np.int32),
        "center_m": mesh_frame.center_m,
        "extents_m": mesh_frame.extents_m,
    }


def build_raycast_spring_grid(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    *,
    spacing_m: float = 0.005,
    min_slack_length_m: float = 0.001,
    thickness_axis: int | None = None,
) -> SpringSurfaceGrid:
    """Build a full-footprint spring grid with raycast thickness as slack length."""

    grid_uv, frame = make_footprint_grid(vertices_m, spacing_m=spacing_m, thickness_axis=thickness_axis)
    ray = raycast_grid_thickness(vertices_m, faces, grid_uv, frame=frame)
    slack = ray["thickness_m"]
    valid = np.isfinite(slack) & (slack >= min_slack_length_m)
    if not np.any(valid):
        raise MeshQCError("Raycast spring grid has no valid cells")
    center_uv = frame.center_m[list(frame.plane_axes)]
    xy = np.ascontiguousarray(ray["grid_uv_m"][valid] - center_uv, dtype=np.float64)
    neighbors = compute_grid_neighbors(xy, spacing_m)
    return SpringSurfaceGrid(
        xy_m=xy,
        grid_uv_m=np.ascontiguousarray(ray["grid_uv_m"][valid], dtype=np.float64),
        slack_length_m=np.ascontiguousarray(slack[valid], dtype=np.float64),
        bottom_m=np.ascontiguousarray(ray["bottom_m"][valid], dtype=np.float64),
        top_m=np.ascontiguousarray(ray["top_m"][valid], dtype=np.float64),
        hit_count=np.ascontiguousarray(ray["hit_count"][valid], dtype=np.int32),
        cell_area_m2=spacing_m * spacing_m,
        spacing_m=spacing_m,
        frame=frame,
        neighbors=neighbors,
    )


def _load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            fields = line.split()
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif line.startswith("f "):
            indices = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
            if len(indices) >= 3:
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])
    if not vertices or not faces:
        raise MeshQCError(f"OBJ mesh has no vertices or faces: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines: list[str] = []
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}")
    for face in faces.reshape(-1, 3):
        lines.append(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}")
    path.write_text("\n".join(lines) + "\n")


def _mesh_thickness_m(vertices: np.ndarray) -> float:
    extents = np.ptp(vertices, axis=0)
    return float(np.min(extents))


def condition_midsole_mesh(
    mesh_path: str | Path,
    cache_dir: str | Path,
    *,
    source_units: str = "mm",
    min_thickness_m: float = 0.005,
    max_thickness_m: float = 0.08,
    remesh: bool = True,
) -> dict[str, object]:
    """Load, QC, optionally Poisson-remesh, and cache a watertight midsole mesh."""

    path = Path(mesh_path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    repaired_path = cache / f"{path.stem}.v2.repaired.obj"
    report_path = cache / f"{path.stem}.v2.mesh_qc.json"
    source_stat = path.stat()
    if repaired_path.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        thickness_m = float(report.get("thickness_m", -1.0))
        if (
            report.get("source_mesh") == str(path)
            and report.get("source_size_bytes") == source_stat.st_size
            and report.get("source_mtime_ns") == source_stat.st_mtime_ns
            and bool(report.get("repaired_watertight"))
            and min_thickness_m <= thickness_m <= max_thickness_m
        ):
            return report

    vertices, faces = _load_obj_mesh(path)
    if source_units == "mm":
        vertices = vertices * 0.001
    elif source_units != "m":
        raise ValueError("source_units must be 'mm' or 'm'")

    thickness_m = _mesh_thickness_m(vertices)
    if not (min_thickness_m <= thickness_m <= max_thickness_m):
        raise MeshQCError(
            f"Midsole thickness {thickness_m:.4g} m is outside [{min_thickness_m:.4g}, {max_thickness_m:.4g}] m"
        )

    import newton  # noqa: PLC0415

    mesh = newton.Mesh(vertices, faces.reshape(-1))
    input_watertight = bool(mesh.is_watertight)
    repaired_mesh = mesh
    if remesh and not input_watertight:
        from newton.utils import remesh_mesh  # noqa: PLC0415

        repaired_mesh = remesh_mesh(
            mesh,
            method="poisson",
            edge_segments=1,
            resolution=1500,
            depth=7,
            simplify_tolerance=None,
            verbose=False,
        )

    if not repaired_mesh.is_watertight:
        raise MeshQCError("Conditioned midsole mesh is not watertight after repair")

    repaired_faces = repaired_mesh.indices.reshape(-1, 3)
    _write_obj(repaired_path, repaired_mesh.vertices, repaired_faces)
    report = {
        "source_mesh": str(path),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "repaired_mesh": str(repaired_path),
        "input_vertices": int(len(vertices)),
        "input_faces": int(len(faces)),
        "repaired_vertices": int(len(repaired_mesh.vertices)),
        "repaired_faces": int(len(repaired_faces)),
        "input_watertight": input_watertight,
        "repaired_watertight": bool(repaired_mesh.is_watertight),
        "thickness_m": thickness_m,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
