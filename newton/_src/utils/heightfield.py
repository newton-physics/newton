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

from __future__ import annotations

import os

import numpy as np
import warp as wp

from ..geometry.support_function import GenericShapeData, GeoTypeEx


def load_heightfield_elevation(
    filename: str,
    nrow: int,
    ncol: int,
) -> np.ndarray:
    """Load elevation data from a PNG or binary file.

    Supports two formats following MuJoCo conventions:
    - PNG: Grayscale image where white=high, black=low
      (normalized to [0, 1])
    - Binary: MuJoCo custom format with int32 header
      (nrow, ncol) followed by float32 data

    Args:
        filename: Path to the heightfield file (PNG or binary).
        nrow: Expected number of rows.
        ncol: Expected number of columns.

    Returns:
        (nrow, ncol) float32 array of elevation values.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".png":
        from PIL import Image

        img = Image.open(filename).convert("L")
        data = np.array(img, dtype=np.float32) / 255.0
        if data.shape != (nrow, ncol):
            raise ValueError(f"PNG heightfield dimensions {data.shape} don't match expected ({nrow}, {ncol})")
        return data

    # Default: MuJoCo binary format
    # Header: (int32) nrow, (int32) ncol; payload: float32[nrow*ncol]
    with open(filename, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2 or header[0] <= 0 or header[1] <= 0:
            raise ValueError(
                f"Invalid binary heightfield header in '{filename}': expected 2 positive int32 values, got {header}"
            )
        expected_count = int(header[0]) * int(header[1])
        data = np.fromfile(f, dtype=np.float32, count=expected_count)
        if data.size != expected_count:
            raise ValueError(
                f"Binary heightfield '{filename}' payload size mismatch: "
                f"expected {expected_count} float32 values for {header[0]}x{header[1]} grid, got {data.size}"
            )
    return data.reshape(header[0], header[1])


@wp.struct
class HeightfieldData:
    """Per-shape heightfield metadata for collision kernels.

    The actual elevation data is stored in a separate concatenated array
    passed to kernels. ``data_offset`` is the starting index into that array.
    """

    data_offset: wp.int32  # Offset into the concatenated elevation array
    nrow: wp.int32
    ncol: wp.int32
    hx: wp.float32  # Half-extent X
    hy: wp.float32  # Half-extent Y
    min_z: wp.float32
    max_z: wp.float32


def create_empty_heightfield_data() -> HeightfieldData:
    """Create an empty HeightfieldData for non-heightfield shapes."""
    hd = HeightfieldData()
    hd.data_offset = 0
    hd.nrow = 0
    hd.ncol = 0
    hd.hx = 0.0
    hd.hy = 0.0
    hd.min_z = 0.0
    hd.max_z = 0.0
    return hd


@wp.func
def get_triangle_from_heightfield_cell(
    hfd: HeightfieldData,
    elevation_data: wp.array(dtype=wp.float32),
    X_ws: wp.transform,
    row: int,
    col: int,
    tri_sub: int,
) -> tuple[GenericShapeData, wp.vec3]:
    """Extract a triangle from a heightfield grid cell.

    Each grid cell (row, col) produces 2 triangles (tri_sub=0 or 1).
    Returns (GenericShapeData, v0_world) in the same format as
    get_triangle_shape_from_mesh, so GJK/MPR works unchanged.

    Triangle layout for cell (row, col)::

        p01 --- p11
         |  \\ 1  |
         | 0  \\  |
        p00 --- p10

        tri_sub=0: (p00, p10, p11)
        tri_sub=1: (p00, p11, p01)
    """
    # Grid spacing
    dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
    z_range = hfd.max_z - hfd.min_z

    # Corner positions in local space
    x0 = -hfd.hx + wp.float32(col) * dx
    x1 = x0 + dx
    y0 = -hfd.hy + wp.float32(row) * dy
    y1 = y0 + dy

    # Read elevation values from concatenated array
    base = hfd.data_offset
    h00 = elevation_data[base + row * hfd.ncol + col]
    h10 = elevation_data[base + row * hfd.ncol + (col + 1)]
    h01 = elevation_data[base + (row + 1) * hfd.ncol + col]
    h11 = elevation_data[base + (row + 1) * hfd.ncol + (col + 1)]

    # Convert to world Z: min_z + h * (max_z - min_z)
    z00 = hfd.min_z + h00 * z_range
    z10 = hfd.min_z + h10 * z_range
    z01 = hfd.min_z + h01 * z_range
    z11 = hfd.min_z + h11 * z_range

    # Local-space corner positions
    p00 = wp.vec3(x0, y0, z00)
    p10 = wp.vec3(x1, y0, z10)
    p01 = wp.vec3(x0, y1, z01)
    p11 = wp.vec3(x1, y1, z11)

    # Select triangle vertices
    if tri_sub == 0:
        v0_local = p00
        v1_local = p10
        v2_local = p11
    else:
        v0_local = p00
        v1_local = p11
        v2_local = p01

    # Transform to world space
    v0_world = wp.transform_point(X_ws, v0_local)
    v1_world = wp.transform_point(X_ws, v1_local)
    v2_world = wp.transform_point(X_ws, v2_local)

    # Create triangle shape data (same convention as get_triangle_shape_from_mesh)
    shape_data = GenericShapeData()
    shape_data.shape_type = int(GeoTypeEx.TRIANGLE)
    shape_data.scale = v1_world - v0_world  # B - A
    shape_data.auxiliary = v2_world - v0_world  # C - A

    return shape_data, v0_world
