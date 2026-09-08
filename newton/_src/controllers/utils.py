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
    allow_indexed: bool = False,
) -> None:
    """Validate a Warp array's dtype, shape, and device.

    ``shape`` is exact and carries no wildcards, so its length states the
    expected dimensionality. For an array whose own length defines a count
    rather than having to match one, pass ``shape=(array.size,)``: that
    equality holds only for a 1-D array, so a multi-dimensional argument is
    still rejected.

    Args:
        array: Value to validate, or ``None`` for an omitted optional argument.
        name: Argument or port name, used in error messages.
        dtype: Warp dtype the array must have.
        shape: Exact shape the array must have.
        device: Device the array must live on.
        required: Whether ``None`` is rejected.
        allow_indexed: Whether a :class:`wp.indexedarray` view is accepted.
            Set for caller-bound ports, which may be bound to a view of a
            simulation-sized array rather than to an array of its own.
    """
    if array is None:
        if required:
            raise ValueError(f"{name} is required, cannot be `None`.")
        return
    accepted = wp.array | wp.indexedarray if allow_indexed else wp.array
    if not isinstance(array, accepted):
        expected = "a wp.array or wp.indexedarray" if allow_indexed else "a wp.array"
        raise TypeError(f"{name} must be {expected}, got {type(array).__name__}.")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}.")
    if array.device != device:
        raise ValueError(f"{name} must be on device {device}, got {array.device}.")
    if tuple(array.shape) != shape:
        hint = ""
        if allow_indexed:
            # Only ports can be bound to a view, so only they get the hint.
            hint = (
                " To bind a simulation-sized array, pass a view: sim_array[ctrl.qd_start], using a "
                "model-based controller's own q_start/qd_start properties."
            )
        raise ValueError(f"{name} must have shape {shape}, got {tuple(array.shape)}.{hint}")


def _bake_optional_float_array(
    value: wp.array[wp.float32] | float | None,
    size: int,
    *,
    device: wp.DeviceLike,
    requires_grad: bool,
) -> wp.array[wp.float32] | None:
    """Broadcast a scalar, or copy an array, into a fresh buffer of the given size.

    Returns ``None`` for a live parameter, which is read from the input
    struct each step instead. A wp.array is already validated by
    :func:`_validate_array`.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return wp.full(size, float(value), dtype=wp.float32, device=device, requires_grad=requires_grad)
    baked = wp.zeros(size, dtype=wp.float32, device=device, requires_grad=requires_grad)
    wp.copy(baked, value)
    return baked
