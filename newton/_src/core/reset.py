# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared reset helpers."""

from __future__ import annotations

import warnings

import warp as wp


@wp.func
def reset_world_selected(world: int, world_mask: wp.array[wp.bool], world_count: int) -> bool:
    """Return whether an entity world is selected by a canonical reset mask."""
    if world >= 0 and world < world_count:
        return world_mask[world]
    return world == -1 and world_mask[world_count]


def normalize_reset_world_mask(
    world_mask: wp.array[wp.bool] | None,
    *,
    world_count: int,
    device: wp.Device,
    allow_legacy: bool = False,
) -> wp.array[wp.bool] | None:
    """Validate a reset mask and return the canonical shape.

    By default, only ``world_count + 1`` is accepted. ``allow_legacy`` also
    accepts the deprecated ``world_count`` shape and appends an unselected
    global slot. The result is therefore always ``None`` or canonical.
    """
    if world_mask is None:
        return None
    if not isinstance(world_mask, wp.array):
        raise TypeError("'world_mask' must be a Warp array or None.")
    if world_mask.dtype != wp.bool:
        raise TypeError("'world_mask' must have dtype bool.")
    if world_mask.ndim != 1:
        raise ValueError("'world_mask' must be one-dimensional.")
    if world_mask.device != device:
        raise ValueError(f"'world_mask' device {world_mask.device} does not match expected device {device}.")
    mask_size = world_mask.shape[0]
    if mask_size == world_count + 1:
        return world_mask
    if allow_legacy and mask_size == world_count:
        warnings.warn(
            "world_mask with shape (world_count,) is deprecated; use shape (world_count + 1,), "
            "where the final entry selects global entities in world -1.",
            DeprecationWarning,
            stacklevel=4,
        )
        normalized_mask = wp.zeros(world_count + 1, dtype=wp.bool, device=device)
        if world_count > 0:
            wp.copy(normalized_mask, world_mask, count=world_count)
        return normalized_mask
    if allow_legacy:
        raise ValueError(f"world_mask has size {mask_size}, expected {world_count} or {world_count + 1}.")
    raise ValueError(f"'world_mask' length {mask_size} must equal model.world_count + 1 ({world_count + 1}).")
