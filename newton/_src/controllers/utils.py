# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for :mod:`newton.controllers`."""

from __future__ import annotations

from typing import Any

import warp as wp


def _normalize_port(
    spec: Any,
    control_law_indices: wp.array[wp.uint32],
    *,
    name: str,
) -> tuple[str, wp.array[wp.uint32]]:
    """Normalize a per-DOF port spec at :meth:`ControlLaw.__init__` time.

    Per-DOF ports name *where on the* ``input``/``output`` object the array
    lives at step time; they're never the array itself. The returned
    ``attr_name`` is later resolved against the user-supplied ``input`` or
    ``output`` via ``getattr(source, attr_name)``.

    Args:
        spec: One of:

            - ``str`` — attribute name on the source object. ``port_indices``
              defaults to ``control_law_indices``.
            - ``(str, wp.array[wp.uint32])`` — attribute name + custom
              ``port_indices``. Used when the source array's layout differs
              from the controller-level ``indices``.
        control_law_indices: ControlLaw-level lookup. Default port indices.
        name: Port name used in error messages.

    Returns:
        ``(attr_name, port_indices)`` ready to be stored on the ControlLaw.
    """
    if isinstance(spec, str):
        return spec, control_law_indices
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError(f"Port '{name}': tuple must be (attr_name, port_indices); got {len(spec)} elements.")
        attr_name, port_indices = spec
        if not isinstance(attr_name, str):
            raise TypeError(
                f"Port '{name}': first tuple element must be str (attribute name), got {type(attr_name).__name__}."
            )
        if not isinstance(port_indices, wp.array):
            raise TypeError(f"Port '{name}': second tuple element must be wp.array, got {type(port_indices).__name__}.")
        if port_indices.shape != control_law_indices.shape:
            raise ValueError(
                f"Port '{name}': port_indices shape {port_indices.shape} must match "
                f"ControlLaw indices shape {control_law_indices.shape}."
            )
        return attr_name, port_indices
    raise TypeError(f"Port '{name}': expected str or (str, wp.array) tuple, got {type(spec).__name__}.")


def _normalize_per_group_port(spec: Any, *, name: str) -> str:
    """Normalize a per-group port spec at ``__init__`` time. Returns the attr name."""
    if not isinstance(spec, str):
        raise TypeError(f"Port '{name}': expected str (attribute name), got {type(spec).__name__}.")
    return spec


def _resolve_input_array(
    source: Any,
    attr_name: str,
    *,
    name: str,
) -> wp.array:
    """Step-time: fetch a port array from ``source`` (an ``input`` or
    ``output`` object) and validate it's a :class:`wp.array`.

    Shape/dtype are not validated here — Warp will raise on a kernel-launch
    mismatch with a precise message. The cheap type check catches typos
    early (e.g. ``input.joint_q = some_list``).
    """
    try:
        arr = getattr(source, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Port '{name}': source object has no attribute '{attr_name}'.") from e
    if not isinstance(arr, wp.array):
        raise TypeError(f"Port '{name}': source.{attr_name} must be wp.array, got {type(arr).__name__}.")
    return arr


def _resolve_per_group_array(
    source: Any,
    attr_name: str,
    num_robots: int,
    dtype: Any,
    *,
    name: str,
) -> wp.array:
    """Step-time resolver for per-group ports. Checks shape + dtype since
    they're documented contract (``length == num_robots`` with a specific
    Warp dtype like ``wp.vec3`` / ``wp.quat`` / ``wp.float32``).
    """
    arr = _resolve_input_array(source, attr_name, name=name)
    if arr.shape != (num_robots,):
        raise ValueError(f"Port '{name}': source.{attr_name} shape {arr.shape} must equal ({num_robots},).")
    if arr.dtype != dtype:
        raise TypeError(f"Port '{name}': source.{attr_name} dtype must be {dtype}, got {arr.dtype}.")
    return arr
