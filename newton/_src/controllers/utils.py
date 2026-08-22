# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for :mod:`newton.controllers`."""

from __future__ import annotations

from typing import Any

import warp as wp


def _validate_array(
    *,
    array: Any,
    name: str,
    dtype: Any,
    shape: tuple[int, ...],
    device: wp.DeviceLike,
    required: bool = True,
) -> None:
    """Validate a ``wp.array`` argument's dtype, shape, and device.

    Args:
        array: Value to validate, or ``None`` for an omitted optional argument.
        name: Argument name, used in error messages.
        dtype: Warp dtype the array must have.
        shape: Shape the array must have. A ``-1`` entry accepts any size in
            that dimension, so ``(-1,)`` means "1-D, any length".
        device: Device the array must live on.
        required: Whether ``None`` is rejected.
    """
    if array is None:
        if required:
            raise ValueError(f"{name} is required, cannot be `None`.")
        return
    if not isinstance(array, wp.array):
        raise TypeError(f"{name} must be a wp.array, got {type(array).__name__}.")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}.")
    actual = tuple(array.shape)
    if len(actual) != len(shape) or any(want not in (-1, got) for got, want in zip(actual, shape, strict=True)):
        expected = "(" + ", ".join("*" if d == -1 else str(d) for d in shape) + ")"
        raise ValueError(f"{name} must have shape {expected}, got {actual}.")
    if array.device != device:
        raise ValueError(f"{name} must be on device {device}, got {array.device}.")


def _validate_flat_port(
    *,
    array: Any,
    name: str,
    min_length: int,
    device: wp.DeviceLike,
) -> None:
    """Validate a caller-bound flat float32 array before any kernel reads it.

    Unlike :func:`_validate_array`, the length is a lower bound: a port may be
    bound to a larger simulation array that the controller indexes into.

    Args:
        array: Array bound to the port by the caller.
        name: Port name, used in error messages.
        min_length: Smallest length the port's indices can be read from safely.
        device: Device the array must live on.
    """
    if not isinstance(array, wp.array):
        raise TypeError(f"{name} must be a wp.array, got {type(array).__name__}.")
    if array.dtype != wp.float32:
        raise TypeError(f"{name} must have dtype {wp.float32}, got {array.dtype}.")
    if array.device != device:
        raise ValueError(f"{name} must be on device {device}, got {array.device}.")
    if array.size < min_length:
        raise ValueError(f"{name} must have length at least {min_length}, got {array.size}.")


def _normalize_indices(
    *,
    idx: wp.array[wp.uint32] | None,
    default_idx: wp.array[wp.uint32],
) -> wp.array[wp.uint32]:
    """Return ``idx`` if supplied, otherwise ``default_idx``.

    Both are validated by :func:`_validate_array` at construction, so this only
    selects between them.
    """
    return default_idx if idx is None else idx
