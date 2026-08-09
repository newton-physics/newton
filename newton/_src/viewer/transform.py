# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fast host-side operations on Warp transforms used by viewers."""

from __future__ import annotations

import ctypes
from functools import cache
from typing import Any

import numpy as np
import warp as wp


@cache
def _builtin(name: str):
    """Return a cached native Warp builtin without repeated overload resolution."""
    wp.init()
    return getattr(wp._src.context.runtime.core, name)


def transform_add_translation(transform: wp.transform, offset: Any) -> wp.transform:
    """Return a copy of ``transform`` with a translation offset applied."""
    result = wp.transform()
    values = np.asarray(result)
    values[:] = np.asarray(transform)
    values[:3] += offset
    return result


def transform_assign(transform: wp.transform, value: Any) -> None:
    """Assign all components while preserving the caller-owned transform."""
    np.asarray(transform)[:] = np.asarray(value, dtype=np.float32)


def transform_assign_matrix(transform: wp.transform, matrix: Any) -> None:
    """Assign a row-major 4x4 transformation matrix to ``transform``."""
    matrix = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    if not matrix.flags.c_contiguous:
        matrix = np.ascontiguousarray(matrix)

    rotation = wp.quat()
    _builtin("wp_builtin_quat_from_matrix_mat44f")(
        wp.mat44.from_buffer(matrix),
        ctypes.byref(rotation),
    )

    values = np.asarray(transform)
    values[:3] = matrix[:3, 3]
    values[3:] = np.asarray(rotation)


def transform_assign_position_wxyz(transform: wp.transform, position: Any, wxyz: Any) -> None:
    """Assign a Viser position and WXYZ quaternion to ``transform``."""
    values = np.asarray(transform)
    values[:3] = position
    values[3:6] = wxyz[1:4]
    values[6] = wxyz[0]


def transform_from_array(value: Any) -> wp.transform:
    """Return a transform view of a contiguous float32 array-like value."""
    values = np.asarray(value, dtype=np.float32).reshape(7)
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    return wp.transform.from_buffer(values)


def transform_inverse(transform: wp.transform) -> wp.transform:
    """Return the inverse without resolving a Warp overload at Python scope."""
    result = wp.transform()
    _builtin("wp_builtin_transform_inverse_transformf")(transform, ctypes.byref(result))
    return result


def transform_multiply(a: wp.transform, b: wp.transform) -> wp.transform:
    """Multiply transforms without resolving a Warp overload at Python scope."""
    result = wp.transform()
    _builtin("wp_builtin_mul_transformf_transformf")(a, b, ctypes.byref(result))
    return result


def transform_point(transform: wp.transform, point: wp.vec3) -> wp.vec3:
    """Apply ``transform`` to a point without Python-scope overload resolution."""
    result = wp.vec3()
    _builtin("wp_builtin_transform_point_transformf_vec3f")(transform, point, ctypes.byref(result))
    return result


def transform_to_matrix(transform: wp.transform) -> np.ndarray:
    """Return ``transform`` as a row-major float32 4x4 matrix."""
    px, py, pz, qx, qy, qz, qw = np.asarray(transform)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.asarray(
        (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), px),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), py),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), pz),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def transform_to_position_wxyz(transform: wp.transform) -> tuple[np.ndarray, np.ndarray]:
    """Return a transform as a copied position and WXYZ quaternion."""
    values = np.asarray(transform)
    return values[:3].copy(), values[[6, 3, 4, 5]]


def transform_vector(transform: wp.transform, vector: wp.vec3) -> wp.vec3:
    """Apply only the rotation without Python-scope overload resolution."""
    result = wp.vec3()
    _builtin("wp_builtin_transform_vector_transformf_vec3f")(transform, vector, ctypes.byref(result))
    return result
