# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared reset helpers."""

from __future__ import annotations

import warp as wp


@wp.func
def reset_world_selected(world: int, world_mask: wp.array[wp.bool], world_count: int) -> bool:
    """Return whether an entity world is selected by a canonical reset mask."""
    if world >= 0 and world < world_count:
        return world_mask[world]
    return world == -1 and world_mask[world_count]


@wp.func
def world_reset_since(world: int, epochs: wp.array[wp.int32], cursor: wp.array[wp.int32], world_count: int) -> bool:
    """Return whether an entity world has been reset since ``cursor`` last caught up with ``epochs``."""
    if world >= 0 and world < world_count:
        return epochs[world] != cursor[world]
    return world == -1 and epochs[world_count] != cursor[world_count]


@wp.kernel(enable_backward=False)
def _advance_world_reset_epochs_kernel(world_mask: wp.array[wp.bool], epochs: wp.array[wp.int32]):
    world = wp.tid()
    if world_mask[world]:
        epochs[world] += 1


class WorldResetEpochs:
    """Per-world reset counts that each holder of contact history catches up on by itself.

    A single consumed reset mask reaches only the first holder that reads it. Each holder instead keeps
    a :meth:`cursor` and treats a world as reset while its epoch differs from the cursor.

    Args:
        world_count: Number of local worlds.
        device: Device to allocate on.
    """

    def __init__(self, world_count: int, device=None):
        self.world_count = int(world_count)
        self.epochs = wp.zeros(self.world_count + 1, dtype=wp.int32, device=device)
        self._all_worlds = wp.full(self.world_count + 1, True, dtype=wp.bool, device=device)

    def advance(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Count a reset of the selected worlds, or of all worlds."""
        wp.launch(
            _advance_world_reset_epochs_kernel,
            dim=self.epochs.shape[0],
            inputs=[self._all_worlds if world_mask is None else world_mask, self.epochs],
            device=self.epochs.device,
        )

    def cursor(self) -> wp.array[wp.int32]:
        """Return a new cursor that has seen every reset so far."""
        return wp.clone(self.epochs)

    def catch_up(self, cursor: wp.array[wp.int32]) -> None:
        """Mark every reset so far as seen by ``cursor``."""
        wp.copy(cursor, self.epochs)


def normalize_reset_world_mask(
    world_mask: wp.array[wp.bool] | None,
    *,
    world_count: int,
    device: wp.Device,
) -> wp.array[wp.bool] | None:
    """Validate a reset mask with shape ``(world_count + 1,)``.

    The final entry selects global entities in world ``-1``. Returns the mask
    unchanged, or ``None`` if no mask was given.
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
    if mask_size != world_count + 1:
        raise ValueError(f"'world_mask' length {mask_size} must equal model.world_count + 1 ({world_count + 1}).")
    return world_mask
