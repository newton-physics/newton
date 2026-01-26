# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

"""Host-side helpers for the GPU Cosserat example."""

from __future__ import annotations

from typing import List

import numpy as np


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    vx, vy, vz = v

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    return np.array(
        [
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        ],
        dtype=np.float32,
    )


def build_director_lines(
    positions: np.ndarray,
    orientations: np.ndarray,
    offset: np.ndarray,
    director_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_edges = positions.shape[0] - 1
    positions = positions + offset

    starts = np.zeros((num_edges * 3, 3), dtype=np.float32)
    ends = np.zeros((num_edges * 3, 3), dtype=np.float32)
    colors = np.zeros((num_edges * 3, 3), dtype=np.float32)

    for i in range(num_edges):
        midpoint = 0.5 * (positions[i] + positions[i + 1])
        q = orientations[i]

        d1 = rotate_vector_by_quaternion(np.array([1.0, 0.0, 0.0], dtype=np.float32), q)
        d2 = rotate_vector_by_quaternion(np.array([0.0, 1.0, 0.0], dtype=np.float32), q)
        d3 = rotate_vector_by_quaternion(np.array([0.0, 0.0, 1.0], dtype=np.float32), q)

        base_idx = i * 3
        starts[base_idx] = midpoint
        ends[base_idx] = midpoint + d1 * director_scale
        colors[base_idx] = [1.0, 0.0, 0.0]

        starts[base_idx + 1] = midpoint
        ends[base_idx + 1] = midpoint + d2 * director_scale
        colors[base_idx + 1] = [0.0, 1.0, 0.0]

        starts[base_idx + 2] = midpoint
        ends[base_idx + 2] = midpoint + d3 * director_scale
        colors[base_idx + 2] = [0.0, 0.0, 1.0]

    return starts, ends, colors


def compute_linear_offsets(count: int, spacing: float) -> List[float]:
    if count <= 1:
        return [0.0]
    center = 0.5 * float(count - 1)
    return [(float(i) - center) * spacing for i in range(count)]


__all__ = [
    "build_director_lines",
    "compute_linear_offsets",
    "rotate_vector_by_quaternion",
]
