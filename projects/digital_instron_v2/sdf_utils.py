# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SDF construction utilities for Digital Instron v2 indenters.

Builds :class:`~newton._src.geometry.sdf_texture.TextureSDFData` for various
indenter geometries (flat plate, cylinder, STL mesh) using Warp's mesh SDF
pipeline.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.geometry.sdf_texture import (
    TextureSDFData,
    create_texture_sdf_from_mesh,
)

from .geometry import _load_obj_mesh

# Module-level cache: key -> (texture_sdf, coarse_tex, subgrid_tex, block_coords[, sparse_data])
# create_texture_sdf_from_mesh returns 4 or 5 elements depending on return_sparse_data.
_SDF_CACHE: dict[str, tuple] = {}


def _build_box_verts_faces(
    half_extents: tuple[float, float, float],
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build vertices and faces for a closed box mesh.

    Returns (vertices, faces) with shape (8, 3) float64 and (12, 3) int32.
    """
    hx, hy, hz = half_extents
    cx, cy, cz = center
    vertices = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 4, 5],
            [0, 5, 1],  # front
            [1, 5, 6],
            [1, 6, 2],  # right
            [2, 6, 7],
            [2, 7, 3],  # back
            [3, 7, 4],
            [3, 4, 0],  # left
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _build_cylinder_verts_faces(
    radius_m: float,
    height_m: float,
    num_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Build vertices and faces for a closed cylinder mesh.

    Returns (vertices, faces) with shape (2*num_segments+2, 3) float64 and
    (4*num_segments, 3) int32.
    """
    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    n = num_segments

    bottom = np.column_stack(
        [
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            np.full(n, -height_m / 2),
        ]
    )
    top = np.column_stack(
        [
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            np.full(n, height_m / 2),
        ]
    )
    bottom_center = np.array([[0.0, 0.0, -height_m / 2]], dtype=np.float64)
    top_center = np.array([[0.0, 0.0, height_m / 2]], dtype=np.float64)
    vertices = np.vstack([bottom, top, bottom_center, top_center])

    # Bottom cap fan triangles
    bottom_faces = []
    for i in range(n):
        bottom_faces.append([2 * n, i, (i + 1) % n])

    # Top cap fan triangles
    top_faces = []
    for i in range(n):
        top_faces.append([2 * n + 1, ((i + 1) % n) + n, i + n])

    # Side quads split into two triangles each
    side_faces = []
    for i in range(n):
        nxt = (i + 1) % n
        side_faces.append([i, nxt, i + n])
        side_faces.append([nxt, nxt + n, i + n])

    faces = np.array(bottom_faces + top_faces + side_faces, dtype=np.int32)
    return vertices, faces


def _cache_key(
    indenter_type: str,
    plate_height: float,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    radius_m: float,
    height_m: float,
    path: str | None,
    target_voxel_size: float,
    margin: float,
    narrow_band_range: tuple[float, float],
    device: str | None,
) -> str:
    return (
        f"{indenter_type}:{plate_height}:{bounds}:{radius_m}:{height_m}:{path}:"
        f"{target_voxel_size}:{margin}:{narrow_band_range}:{device}"
    )


def _load_ascii_stl(text: str) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    current: list[int] = []
    vertex_to_index: dict[tuple[float, float, float], int] = {}
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertex = (float(fields[1]), float(fields[2]), float(fields[3]))
            index = vertex_to_index.get(vertex)
            if index is None:
                index = len(vertices)
                vertex_to_index[vertex] = index
                vertices.append(vertex)
            current.append(index)
            if len(current) == 3:
                faces.append(current)
                current = []
    if not vertices or not faces:
        raise ValueError("ASCII STL contains no triangles")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _load_binary_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(data) < 84:
        raise ValueError("Binary STL is too small")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(data) < expected_size:
        raise ValueError("Binary STL is truncated")
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    vertex_to_index: dict[tuple[float, float, float], int] = {}
    offset = 84
    for _ in range(triangle_count):
        coords = struct.unpack_from("<12f", data, offset)
        face: list[int] = []
        for start in (3, 6, 9):
            vertex = (float(coords[start]), float(coords[start + 1]), float(coords[start + 2]))
            index = vertex_to_index.get(vertex)
            if index is None:
                index = len(vertices)
                vertex_to_index[vertex] = index
                vertices.append(vertex)
            face.append(index)
        faces.append(face)
        offset += 50
    if not vertices or not faces:
        raise ValueError("Binary STL contains no triangles")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _load_stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    try:
        return _load_ascii_stl(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return _load_binary_stl(data)


def build_indenter_sdf(
    indenter_type: str = "flat_plate",
    *,
    plate_height: float = 0.0,
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.05, -0.02),
        (0.05, 0.02),
    ),
    radius_m: float = 0.0225,
    height_m: float = 0.05,
    path: str | Path | None = None,
    target_voxel_size: float = 0.001,
    margin: float = 0.01,
    narrow_band_range: tuple[float, float] = (-0.01, 0.01),
    device: str | None = None,
) -> TextureSDFData:
    """Build a texture SDF for an indenter geometry.

    The SDF is constructed once and cached for subsequent calls with the same
    parameters.

    Args:
        indenter_type: ``'flat_plate'``, ``'cylinder'``, or ``'stl'``.
        plate_height: Z-offset of the flat plate [m] (ignored for other types).
        bounds: ``((x_min, y_min), (x_max, y_max))`` bounds for the flat plate
            [m] (ignored for other types).
        radius_m: Cylinder radius [m] (ignored for non-cylinder types).
        height_m: Cylinder height [m] (ignored for non-cylinder types).
        path: Path to OBJ/STL file for ``'stl'`` type.
        target_voxel_size: SDF voxel size [m].
        margin: Extra AABB padding around the mesh [m].
        narrow_band_range: Signed narrow-band distance range [m] as
            ``(inner, outer)``.
        device: Warp device string. Defaults to ``'cuda:0'``.

    Returns:
        The :class:`~newton._src.geometry.sdf_texture.TextureSDFData` for the
        indenter.

    Raises:
        ValueError: If ``indenter_type`` is unknown or ``path`` is missing for
            ``'stl'``.
    """
    _device = device or "cuda:0"
    key = _cache_key(
        indenter_type,
        plate_height,
        bounds,
        radius_m,
        height_m,
        str(Path(path).expanduser().resolve()) if path else None,
        target_voxel_size,
        margin,
        narrow_band_range,
        _device,
    )
    if key in _SDF_CACHE:
        return _SDF_CACHE[key][0]

    if indenter_type == "flat_plate":
        (x_min, y_min), (x_max, y_max) = bounds
        plate_width = x_max - x_min
        plate_length = y_max - y_min
        half_extents = (plate_width / 2.0, plate_length / 2.0, 0.001)
        center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, plate_height)
        vertices_np, faces_np = _build_box_verts_faces(half_extents, center)
    elif indenter_type == "cylinder":
        vertices_np, faces_np = _build_cylinder_verts_faces(radius_m, height_m, num_segments=24)
    elif indenter_type == "stl":
        if path is None:
            raise ValueError("path is required for indenter_type='stl'")
        mesh_path = Path(path).expanduser()
        if not mesh_path.exists():
            raise FileNotFoundError(f"Indenter STL does not exist: {mesh_path}")
        if mesh_path.suffix.lower() == ".obj":
            vertices_np, faces_np = _load_obj_mesh(mesh_path)
        else:
            vertices_np, faces_np = _load_stl_mesh(mesh_path)
    else:
        raise ValueError(f"Unknown indenter_type: {indenter_type}")

    pos = wp.array(vertices_np, dtype=wp.vec3, device=_device)
    indices = wp.array(faces_np.flatten(), dtype=wp.int32, device=_device)

    # Parity-based SDF is cheaper per sample; requires closed manifold mesh.
    # Our synthetic meshes (box, cylinder) are closed, and OBJ meshes loaded
    # via _load_obj_mesh are expected to be manifold.
    mesh = wp.Mesh(points=pos, indices=indices)

    result = create_texture_sdf_from_mesh(
        mesh,
        margin=margin,
        narrow_band_range=narrow_band_range,
        target_voxel_size=target_voxel_size,
        use_parity=True,
        device=_device,
    )

    _SDF_CACHE[key] = result
    return result[0]
