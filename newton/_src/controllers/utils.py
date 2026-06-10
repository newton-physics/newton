# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for :mod:`newton.controllers`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import warp as wp


def _normalize_live_port(
    attr_name: Any,
    idx: Any,
    default_idx: wp.array,
    *,
    name: str,
) -> tuple[str, wp.array]:
    """Normalize a ``(attr_name, idx)`` "live" port pair.

    Live ports are looked up on the user-supplied ``input`` / ``output``
    object at step time via ``getattr(source, attr_name)``. The kernel
    reads / writes ``arr[idx[i]]``. When ``idx`` is ``None`` the
    controller's ``default_idx`` is used — natural-order indexing into a
    source array whose layout already matches the controller's view.
    """
    if not isinstance(attr_name, str):
        raise TypeError(f"Port '{name}': attr name must be str, got {type(attr_name).__name__}.")
    if idx is None:
        indices = default_idx
    elif isinstance(idx, wp.array):
        indices = idx
    else:
        raise TypeError(f"Port '{name}': idx must be wp.array[uint32] or None, got {type(idx).__name__}.")
    return attr_name, indices


def _normalize_gain_port(
    value: Any,
    expected_size: int,
    dtype: Any,
    device: Any,
    requires_grad: bool,
    *,
    name: str,
) -> tuple[str | None, wp.array | None]:
    """Normalize a ``wp.array[dtype] | str`` gain-style port.

    Gain ports are read in natural order: the kernel evaluates ``arr[i]``
    where ``i`` is the natural per-output (or per-robot) loop index. They
    accept two shapes:

    - ``str``: live — at step time the controller resolves
      ``getattr(input, value)`` and reads that array.
    - ``wp.array``: baked — the controller stores a copy of the supplied
      array and reads from that copy each step. Mutating the user's
      original after construction has no effect.

    Returns ``(attr_name_if_live, baked_array_if_baked)`` — exactly one
    of the two is non-``None``.
    """
    if isinstance(value, str):
        return value, None
    if isinstance(value, wp.array):
        if value.size != expected_size:
            raise ValueError(f"Port '{name}': baked array length {value.size} must equal {expected_size}.")
        if value.dtype != dtype:
            raise TypeError(f"Port '{name}': baked array dtype {value.dtype} must equal {dtype}.")
        baked = wp.zeros(expected_size, dtype=dtype, device=device, requires_grad=requires_grad)
        wp.copy(baked, value)
        return None, baked
    raise TypeError(f"Port '{name}': must be wp.array[{dtype}] or str (attr name); got {type(value).__name__}.")


def _resolve_input_array(
    source: Any,
    attr_name: str,
    *,
    name: str,
) -> wp.array:
    """Step-time: fetch a port array from ``source`` and validate it's a
    :class:`wp.array`.

    Shape/dtype are not validated here — Warp surfaces those at the kernel
    launch with a precise message. The cheap type check catches typos early.
    """
    try:
        arr = getattr(source, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Port '{name}': source object has no attribute '{attr_name}'.") from e
    if not isinstance(arr, wp.array):
        raise TypeError(f"Port '{name}': source.{attr_name} must be wp.array, got {type(arr).__name__}.")
    return arr


def _resolve_typed_array(
    source: Any,
    attr_name: str,
    dtype: Any,
    *,
    name: str,
) -> wp.array:
    """Step-time resolver with a dtype check on top of :func:`_resolve_input_array`."""
    arr = _resolve_input_array(source, attr_name, name=name)
    if arr.dtype != dtype:
        raise TypeError(f"Port '{name}': source.{attr_name} dtype must be {dtype}, got {arr.dtype}.")
    return arr


def _allocate_namespace(
    specs: list[tuple[str, Any, int]],
    device: Any,
    requires_grad: bool,
) -> SimpleNamespace:
    """Build a :class:`SimpleNamespace` with one zero-allocated ``wp.array``
    per ``(attr_name, dtype, size)`` spec. Duplicate attr names take the
    larger size (smaller views are a prefix of the larger).
    """
    merged: dict[str, tuple[Any, int]] = {}
    for attr, dtype, size in specs:
        if attr in merged:
            prev_dtype, prev_size = merged[attr]
            if prev_dtype != dtype:
                raise TypeError(f"Field '{attr}': two ports declared incompatible dtypes ({prev_dtype} vs {dtype}).")
            merged[attr] = (dtype, max(prev_size, size))
        else:
            merged[attr] = (dtype, size)
    return SimpleNamespace(
        **{
            attr: wp.zeros(size, dtype=dtype, device=device, requires_grad=requires_grad)
            for attr, (dtype, size) in merged.items()
        }
    )
