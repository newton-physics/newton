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

# Module-level cache: key -> (texture_sdf, coarse_tex, subgrid_tex, block_coords[, sparse_data])
# create_texture_sdf_from_mesh returns 4 or 5 elements depending on return_sparse_data.
_SDF_CACHE: dict[str, tuple] = {}


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
