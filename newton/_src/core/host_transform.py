# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fast host-side operations on Warp transforms and Warp value lists.

Scalar helpers call Warp's native builtins directly instead of resolving
overloads at Python scope. Row helpers transform batches stored as float32
NumPy arrays of shape ``(..., 7)``, and packing helpers convert lists of Warp
vector, matrix, and transform values to NumPy arrays in one pass.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np
import warp as wp


@cache
def _builtin(name: str):
    """Return a cached native Warp builtin without repeated overload resolution."""
    wp.init()
    return getattr(wp._src.context.runtime.core, name)


# region Scalar transforms


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


# endregion

# region Warp value packing


def join_warp_values(values: Sequence[Any] | np.ndarray, warp_type: type) -> np.ndarray | None:
    """Join a list of ``warp_type`` vector, matrix, or transform values into one array.

    ``np.asarray`` and ``wp.array`` read a list of Warp value objects one buffer at a
    time, which dominates model building at scale; joining the buffers is several
    times faster. Returns a read-only array of shape ``(N, *warp_type._shape_)``, or
    ``None`` unless ``values`` is a non-empty list holding only ``warp_type`` objects.
    """
    if isinstance(values, np.ndarray) or not len(values) or not all(type(value) is warp_type for value in values):
        return None
    scalar_type = wp.dtype_to_numpy(warp_type._wp_scalar_type_)
    return np.frombuffer(b"".join(values), dtype=scalar_type).reshape((len(values), *warp_type._shape_))


def pack_warp_values(values: Sequence[Any] | np.ndarray, warp_type: type) -> np.ndarray:
    """Pack vector, matrix, or transform values into an array of shape ``(N, *warp_type._shape_)``.

    The result may be read-only.
    """
    joined = join_warp_values(values, warp_type)
    if joined is not None:
        return joined
    scalar_type = wp.dtype_to_numpy(warp_type._wp_scalar_type_)
    return np.asarray(values, dtype=scalar_type).reshape((-1, *warp_type._shape_))


def warp_array_from_values(values: Any, dtype: type, **kwargs: Any) -> wp.array:
    """Create a ``wp.array`` of a Warp value type, joining list input with :func:`join_warp_values`."""
    joined = join_warp_values(values, dtype)
    return wp.array(values if joined is None else joined, dtype=dtype, **kwargs)


def transform_rows(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Pack transform-like values into a float32 array of shape ``(N, 7)``. The result may be read-only."""
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=np.float32).reshape(-1, 7)
    # Entries may be any transform-like value, e.g. a ``(p, q)`` pair.
    values = [value if type(value) is wp.transformf else wp.transform(*value) for value in values]
    return pack_warp_values(values, wp.transformf)


def store_transform_rows(destination: list | np.ndarray, targets: np.ndarray, rows: np.ndarray) -> None:
    """Write ``(N, 7)`` transform rows to ``targets`` of a transform list or an ``(M, 7)`` array."""
    if isinstance(destination, np.ndarray):
        destination[targets] = rows
        return
    for target, row in zip(targets.tolist(), rows, strict=True):
        destination[target] = wp.transform.from_buffer_copy(row)


# endregion

# region Batched transforms

# Batched float32 counterparts of Warp's quat_rotate, quat mul, transform_multiply, and
# transform_inverse. Each mirrors the operation order of warp/native so the results match
# the scalar builtins bit for bit on platforms without fused multiply-add contraction.
_CYCLE_1 = np.asarray((1, 2, 0))
_CYCLE_2 = np.asarray((2, 0, 1))
_QUAT_CONJUGATE_SIGNS = np.asarray((-1.0, -1.0, -1.0, 1.0), dtype=np.float32)

_NATIVE_TRANSFORM_ROW_LIMIT = 16
"""Batches up to this many transforms call Warp's scalar builtins row by row.

Below this size NumPy's fixed cost per operation outweighs the per-row native call.
"""


def quat_rotate_rows(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rotate broadcastable ``(..., 3)`` float32 vectors by ``(..., 4)`` quaternions."""
    v, w = q[..., :3], q[..., 3:]
    two = np.float32(2.0)
    c = two * w * w - np.float32(1.0)
    vx = v * x
    d = two * (vx[..., 0:1] + vx[..., 1:2] + vx[..., 2:3])
    cross = v[..., _CYCLE_1] * x[..., _CYCLE_2] - v[..., _CYCLE_2] * x[..., _CYCLE_1]
    return x * c + v * d + cross * w * two


def quat_multiply_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply broadcastable ``(..., 4)`` float32 quaternions as ``a * b``."""
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    xyz = aw * bv + bw * av + av[..., _CYCLE_1] * bv[..., _CYCLE_2] - bv[..., _CYCLE_1] * av[..., _CYCLE_2]
    ab = av * bv
    w = aw * bw - ab[..., 0:1] - ab[..., 1:2] - ab[..., 2:3]
    return np.concatenate((xyz, w), axis=-1)


def _native_transform_rows(builtin_name: str, *operands: np.ndarray) -> np.ndarray:
    """Apply a native Warp transform builtin to each row of same-shaped ``(..., 7)`` operands."""
    builtin = _builtin(builtin_name)
    shape = operands[0].shape
    rows = [np.ascontiguousarray(operand, dtype=np.float32).reshape(-1, 7) for operand in operands]
    out = np.empty((len(rows[0]), 7), dtype=np.float32)
    for i, out_row in enumerate(out):
        builtin(
            *(wp.transform.from_buffer_copy(row[i]) for row in rows), ctypes.byref(wp.transform.from_buffer(out_row))
        )
    return out.reshape(shape)


def transform_multiply_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose broadcastable ``(..., 7)`` float32 transforms as ``a * b``."""
    a, b = np.broadcast_arrays(a, b)
    if a.size <= 7 * _NATIVE_TRANSFORM_ROW_LIMIT:
        return _native_transform_rows("wp_builtin_mul_transformf_transformf", a, b)
    return np.concatenate(
        (quat_rotate_rows(a[..., 3:], b[..., :3]) + a[..., :3], quat_multiply_rows(a[..., 3:], b[..., 3:])), axis=-1
    )


def transform_inverse_rows(t: np.ndarray) -> np.ndarray:
    """Invert ``(..., 7)`` float32 transforms."""
    if t.size <= 7 * _NATIVE_TRANSFORM_ROW_LIMIT:
        return _native_transform_rows("wp_builtin_transform_inverse_transformf", t)
    q_inv = t[..., 3:] * _QUAT_CONJUGATE_SIGNS
    return np.concatenate((-quat_rotate_rows(q_inv, t[..., :3]), q_inv), axis=-1)


# endregion
