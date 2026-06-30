# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH utility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def sph_is_bool_like(value: Any) -> bool:
    """Return whether ``value`` is a Python or NumPy boolean scalar."""
    return isinstance(value, bool | np.bool_)


def sph_contains_bool_entry(value: Any) -> bool:
    """Return whether a scalar or iterable contains boolean entries."""
    if sph_is_bool_like(value):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype != object:
            return False
        values = value.flat
    else:
        if isinstance(value, str | bytes):
            return False
        try:
            values = iter(value)
        except TypeError:
            return False
    return any(sph_is_bool_like(entry) for entry in values)


def sph_finite_scalar(value: Any, finite_message: str, *, type_message: str | None = None) -> float:
    """Parse a finite scalar while rejecting booleans before ``float`` coercion."""
    if sph_is_bool_like(value):
        raise ValueError(finite_message)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(type_message or finite_message) from exc
    if not np.isfinite(result):
        raise ValueError(finite_message)
    return result


def sph_vec3_array(
    value: Any,
    shape_message: str,
    finite_message: str,
    *,
    bool_message: str | None = None,
) -> np.ndarray:
    """Parse a finite 3-vector array while rejecting boolean components."""
    if sph_contains_bool_entry(value):
        raise ValueError(bool_message or finite_message)
    try:
        coords = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(shape_message) from exc
    if coords.shape != (3,):
        raise ValueError(shape_message)
    if not np.all(np.isfinite(coords)):
        raise ValueError(finite_message)
    return coords


def sph_wp_vec3_array(values: Any) -> np.ndarray:
    """Return a Warp vec3-like array as an ``(n, 3)`` float32 NumPy array."""
    return np.asarray(values.numpy(), dtype=np.float32)
