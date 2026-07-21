# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for :mod:`newton.controllers`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import warp as wp


def _normalize_indices(
    idx: Any,
    default_idx: wp.array[wp.uint32],
    *,
    name: str,
) -> wp.array[wp.uint32]:
    """Return ``idx`` after validation, or ``default_idx`` if ``idx`` is ``None``."""
    if idx is None:
        return default_idx

    if (not isinstance(idx, wp.array)) or (idx.dtype != wp.uint32):
        raise TypeError(f"Port '{name}': idx must be wp.array[uint32] or None, got {type(idx).__name__}.")

    if idx.size != default_idx.size:
        raise TypeError(
            f"Port '{name}': indices must be the same size as default_dof_indices: {idx.size} != {default_idx.size}."
        )

    return idx


def _normalize_parameter_port(
    value: Any,
    expected_size: int,
    dtype: Any,
    device: Any,
    requires_grad: bool,
    *,
    name: str,
) -> tuple[str | None, wp.array[Any] | None]:
    """Normalize a parameter-type port.

    - ``str``: live — at step time the controller resolves
      ``getattr(input, value)`` and reads that array.
    - ``wp.array``: baked — the controller stores a copy of the supplied
      array and reads from it each step.

    Returns ``(attr_name, baked_array)`` — exactly one is non-``None``.
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


def _allocate_namespace(
    specs: list[tuple[str, Any, int]],
    device: Any,
    requires_grad: bool,
) -> SimpleNamespace:
    """Build a :class:`SimpleNamespace` with one zero-allocated ``wp.array`` per spec.

    Duplicate attr names take the larger size (smaller views are a prefix of the larger).
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
