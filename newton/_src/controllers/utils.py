# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for :mod:`newton.controllers`."""

from __future__ import annotations

from typing import Any

import warp as wp


def _normalize_port(
    port,
    controller_indices: wp.array[wp.uint32],
    *,
    name: str,
) -> tuple[wp.array, wp.array[wp.uint32]]:
    """Normalize a port to a ``(array, port_indices)`` tuple.

    Args:
        port: Either a bare :class:`wp.array` (in which case ``port_indices``
            defaults to ``controller_indices``) or a ``(array, port_indices)``
            tuple.
        controller_indices: Controller-level lookup, used as the default
            when ``port`` is bare.
        name: Port name used in error messages.

    Returns:
        Normalized ``(array, port_indices)`` pair.
    """
    if isinstance(port, tuple):
        if len(port) != 2:
            raise ValueError(f"Port '{name}': tuple must be (array, port_indices); got {len(port)} elements.")
        array, port_indices = port
        if not isinstance(array, wp.array):
            raise TypeError(f"Port '{name}': first tuple element must be wp.array, got {type(array).__name__}.")
        if not isinstance(port_indices, wp.array):
            raise TypeError(f"Port '{name}': second tuple element must be wp.array, got {type(port_indices).__name__}.")
        if port_indices.shape != controller_indices.shape:
            raise ValueError(
                f"Port '{name}': port_indices shape {port_indices.shape} must match "
                f"controller indices shape {controller_indices.shape}."
            )
    elif isinstance(port, wp.array):
        array = port
        port_indices = controller_indices
    else:
        raise TypeError(f"Port '{name}': expected wp.array or (wp.array, wp.array) tuple, got {type(port).__name__}.")
    return array, port_indices


def _validate_per_group(
    array: Any,
    num_robots: int,
    dtype: Any,
    name: str,
) -> wp.array:
    """Validate a per-group port: a bare wp.array of length ``num_robots``.

    Args:
        array: The user-supplied port value.
        num_robots: Expected outer length.
        dtype: Expected Warp element dtype (e.g. ``wp.vec3``, ``wp.quat``, ``wp.float32``).
        name: Port name used in error messages.

    Returns:
        The same array, unchanged.
    """
    if not isinstance(array, wp.array):
        raise TypeError(f"Port '{name}': expected wp.array of length {num_robots}, got {type(array).__name__}.")
    if array.shape != (num_robots,):
        raise ValueError(f"Port '{name}': shape {array.shape} must equal ({num_robots},).")
    if array.dtype != dtype:
        raise TypeError(f"Port '{name}': dtype must be {dtype}, got {array.dtype}.")
    return array
