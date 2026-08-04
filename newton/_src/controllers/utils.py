# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for :mod:`newton.controllers`."""

from __future__ import annotations

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
        raise ValueError(
            f"Port '{name}': indices must be the same size as default_dof_indices: {idx.size} != {default_idx.size}."
        )

    return idx
