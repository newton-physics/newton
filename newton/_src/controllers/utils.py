# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for :mod:`newton.controllers`."""

from __future__ import annotations

from typing import Any

import warp as wp


def _normalize_port(spec: Any, *, name: str) -> tuple[str, wp.array[wp.uint32]]:
    """Normalize a port spec at :meth:`ControlLaw.__init__` time.

    Every port spec is a 2-tuple ``(attr_name, port_indices)``:

    - ``attr_name``: the attribute name on the user-supplied ``input`` or
      ``output`` object where the live array lives at step time. Resolved
      via ``getattr(source, attr_name)`` inside the law's ``compute()``.
    - ``port_indices``: a ``wp.array[wp.uint32]`` giving the inner lookup
      used to index into the live array — ``arr[port_indices[i]]`` for
      per-DOF ports, ``arr[port_indices[r]]`` for per-robot ports.

    The caller is responsible for cross-checking ``port_indices.shape``
    against either ``num_outputs`` (per-DOF ports) or ``num_robots``
    (per-robot ports). This helper only validates the structural shape
    of the spec itself.

    Args:
        spec: A 2-tuple ``(str, wp.array[wp.uint32])``.
        name: Port name used in error messages.

    Returns:
        ``(attr_name, port_indices)``.
    """
    if not isinstance(spec, tuple) or len(spec) != 2:
        raise TypeError(f"Port '{name}': expected a 2-tuple (attr_name, port_indices), got {type(spec).__name__}.")
    attr_name, port_indices = spec
    if not isinstance(attr_name, str):
        raise TypeError(
            f"Port '{name}': first tuple element must be str (attribute name), got {type(attr_name).__name__}."
        )
    if not isinstance(port_indices, wp.array):
        raise TypeError(f"Port '{name}': second tuple element must be wp.array, got {type(port_indices).__name__}.")
    return attr_name, port_indices


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


def _resolve_per_robot_array(
    source: Any,
    attr_name: str,
    dtype: Any,
    *,
    name: str,
) -> wp.array:
    """Step-time resolver for per-robot ports. Adds a dtype check on top of
    :func:`_resolve_input_array` — the dtype is part of the port's
    documented contract (``wp.vec3`` / ``wp.quat`` / ``wp.float32``).

    Shape is no longer enforced here: with custom per-robot indices the
    source array may be any length as long as every index is in bounds,
    and bounds violations surface at the kernel launch.
    """
    arr = _resolve_input_array(source, attr_name, name=name)
    if arr.dtype != dtype:
        raise TypeError(f"Port '{name}': source.{attr_name} dtype must be {dtype}, got {arr.dtype}.")
    return arr
