# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample vertical material columns from triangle meshes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ColumnGrid:
    """Midsole columns and their mesh frame."""

    uv_m: np.ndarray
    bottom_m: np.ndarray
    top_m: np.ndarray
    area_m2: float
    spacing_m: float
    thickness_axis: int

    @property
    def slack_m(self) -> np.ndarray:
        """Return rest lengths."""

        return self.top_m - self.bottom_m


def load_mesh(
    path: str | Path,
    scale: float = 1.0,
    rotation_deg: tuple[float, float, float] | list[float] | None = None,
    crop_height_m: float | None = None,
) -> Any:
    """Load one triangle mesh."""

    import trimesh

    mesh = trimesh.load_mesh(path, process=False)
    mesh.vertices *= scale
    extents = np.ptp(mesh.vertices, axis=0)
    thickness_axis = int(np.argmin(extents))
    length_axis = int(np.argmax(extents))
    width_axis = ({0, 1, 2} - {thickness_axis, length_axis}).pop()
    mesh.vertices = mesh.vertices[:, [length_axis, width_axis, thickness_axis]]
    mesh.vertices[:, :2] -= (np.min(mesh.vertices[:, :2], axis=0) + np.max(mesh.vertices[:, :2], axis=0)) / 2.0
    mesh.vertices[:, 2] -= np.min(mesh.vertices[:, 2])
    if rotation_deg is not None:
        roll, pitch, yaw = np.deg2rad(rotation_deg)
        rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        mesh.vertices = mesh.vertices @ (rz @ ry @ rx).T
        mesh.vertices[:, :2] -= (np.min(mesh.vertices[:, :2], axis=0) + np.max(mesh.vertices[:, :2], axis=0)) / 2.0
        mesh.vertices[:, 2] -= np.min(mesh.vertices[:, 2])
    if crop_height_m is not None:
        faces = mesh.faces[np.max(mesh.vertices[mesh.faces, 2], axis=1) <= crop_height_m]
        vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        mesh = trimesh.Trimesh(mesh.vertices[vertices], inverse.reshape(-1, 3), process=False)
        mesh.vertices[:, 2] -= np.min(mesh.vertices[:, 2])
    return mesh


def transform_mesh(
    mesh: Any,
    rotation_deg: tuple[float, float, float] | list[float],
    translation_m: tuple[float, float, float] | list[float],
) -> None:
    """Apply an XYZ rotation about the canonical origin followed by translation."""

    roll, pitch, yaw = np.deg2rad(rotation_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    mesh.vertices = mesh.vertices @ (rz @ ry @ rx).T + np.asarray(translation_m, dtype=np.float64)


def raycast(
    mesh: Any,
    uv_m: np.ndarray,
    thickness_axis: int,
    normal_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first and last mesh intersections at each footprint point."""

    plane_axes = [axis for axis in range(3) if axis != thickness_axis]
    origins = np.zeros((len(uv_m), 3))
    origins[:, plane_axes] = uv_m
    origins[:, thickness_axis] = mesh.bounds[0, thickness_axis] - 0.01
    directions = np.zeros_like(origins)
    directions[:, thickness_axis] = 1.0
    locations, rays, triangles = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    coordinate = locations[:, thickness_axis]
    bottom = np.full(len(uv_m), np.inf)
    top = np.full(len(uv_m), -np.inf)
    bottom_normal = np.zeros(len(uv_m))
    top_normal = np.zeros(len(uv_m))
    np.minimum.at(bottom, rays, coordinate)
    np.maximum.at(top, rays, coordinate)
    normals = mesh.face_normals[triangles, thickness_axis]
    for ray in np.unique(rays):
        hits = np.flatnonzero(rays == ray)
        hits = hits[np.argsort(coordinate[hits])]
        bottom_normal[ray] = normals[hits[0]]
        top_normal[ray] = normals[hits[-1]]
    valid = (
        np.isfinite(bottom)
        & np.isfinite(top)
        & (top > bottom)
        & (bottom_normal <= -normal_threshold)
        & (top_normal >= normal_threshold)
    )
    return np.where(valid, bottom, np.nan), np.where(valid, top, np.nan)


def raycast_surface(
    mesh: Any,
    uv_m: np.ndarray,
    thickness_axis: int,
    side: str,
) -> np.ndarray:
    """Return the near or far mesh surface along the positive ray direction."""

    if side not in {"near", "far"}:
        raise ValueError("side must be 'near' or 'far'")

    plane_axes = [axis for axis in range(3) if axis != thickness_axis]
    origins = np.zeros((len(uv_m), 3))
    origins[:, plane_axes] = uv_m
    origins[:, thickness_axis] = mesh.bounds[0, thickness_axis] - 0.01
    directions = np.zeros_like(origins)
    directions[:, thickness_axis] = 1.0
    locations, rays, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    surface = np.full(len(uv_m), np.inf if side == "near" else -np.inf)
    update = np.minimum.at if side == "near" else np.maximum.at
    update(surface, rays, locations[:, thickness_axis])
    return np.where(np.isfinite(surface), surface, np.nan)


def build_column_grid(mesh: Any, spacing_m: float, min_thickness_m: float = 0.005) -> ColumnGrid:
    """Sample the midsole footprint at uniform spacing."""

    thickness_axis = int(np.argmin(np.ptp(mesh.vertices, axis=0)))
    plane_axes = [axis for axis in range(3) if axis != thickness_axis]
    lower, upper = mesh.bounds[:, plane_axes]
    u = np.arange(lower[0], upper[0] + spacing_m / 2.0, spacing_m)
    v = np.arange(lower[1], upper[1] + spacing_m / 2.0, spacing_m)
    uu, vv = np.meshgrid(u, v)
    uv = np.column_stack((uu.ravel(), vv.ravel()))
    bottom, top = raycast(mesh, uv, thickness_axis, normal_threshold=0.1)
    valid = np.isfinite(bottom) & (top - bottom >= min_thickness_m)
    return ColumnGrid(uv[valid], bottom[valid], top[valid], spacing_m**2, spacing_m, thickness_axis)


def rearfoot_center(mesh: Any, grid: ColumnGrid, length_fraction: float) -> np.ndarray:
    """Place the punch along the long footprint axis."""

    plane_axes = [axis for axis in range(3) if axis != grid.thickness_axis]
    plane = mesh.vertices[:, plane_axes]
    lower, upper = np.min(plane, axis=0), np.max(plane, axis=0)
    long_axis = int(np.argmax(upper - lower))
    center = (lower + upper) / 2.0
    center[long_axis] = lower[long_axis] + length_fraction * (upper[long_axis] - lower[long_axis])
    return center
