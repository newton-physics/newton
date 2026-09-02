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
            hint = " To bind a simulation-sized array, pass a view: sim_array[selection.qd_start]."
        raise ValueError(f"{name} must have shape {shape}, got {tuple(array.shape)}.{hint}")
