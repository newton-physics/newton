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

"""Host-side math utilities for Cosserat rod simulation."""

from __future__ import annotations

import math

import numpy as np


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create a quaternion from an axis-angle representation.
    
    Args:
        axis: 3D unit axis of rotation.
        angle: Rotation angle in radians.
    
    Returns:
        Quaternion as [x, y, z, w] array.
    """
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis / norm
    half = angle * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.
    
    Args:
        q1: First quaternion as [x, y, z, w] array.
        q2: Second quaternion as [x, y, z, w] array.
    
    Returns:
        Product quaternion as [x, y, z, w] array.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion.
    
    Args:
        v: 3D vector to rotate.
        q: Quaternion as [x, y, z, w] array.
    
    Returns:
        Rotated 3D vector.
    """
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
    """Build line segment endpoints for director visualization.
    
    Creates RGB-colored direction vectors (d1=red, d2=green, d3=blue)
    at each segment midpoint for visualizing rod orientations.
    
    Args:
        positions: Particle positions as (N, 3) array.
        orientations: Particle orientations as (N, 4) quaternion array.
        offset: 3D offset to apply to all positions.
        director_scale: Scale factor for director visualization.
    
    Returns:
        Tuple of (starts, ends, colors) arrays for line rendering.
    """
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


def compute_linear_offsets(count: int, spacing: float) -> list[float]:
    """Compute evenly spaced offsets centered around zero.
    
    Args:
        count: Number of offsets to compute.
        spacing: Spacing between adjacent offsets.
    
    Returns:
        List of offset values centered around zero.
    """
    if count <= 1:
        return [0.0]
    center = 0.5 * float(count - 1)
    return [(float(i) - center) * spacing for i in range(count)]


# Backward compatibility aliases (prefixed with underscore)
_quat_from_axis_angle = quat_from_axis_angle
_quat_multiply = quat_multiply


__all__ = [
    "_quat_from_axis_angle",
    "_quat_multiply",
    "build_director_lines",
    "compute_linear_offsets",
    "quat_from_axis_angle",
    "quat_multiply",
    "rotate_vector_by_quaternion",
]
