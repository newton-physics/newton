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


# Backward compatibility aliases (prefixed with underscore)
_quat_from_axis_angle = quat_from_axis_angle
_quat_multiply = quat_multiply


__all__ = [
    "_quat_from_axis_angle",
    "_quat_multiply",
    "quat_from_axis_angle",
    "quat_multiply",
    "rotate_vector_by_quaternion",
]
