# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit effort mode: control law at the current state, then clamps."""

from __future__ import annotations

from typing import Any

import warp as wp

from .clamping.base import Clamping
from .controllers.base import Controller


class _EffortModeExplicit:
    """Explicit effort mode: control law at the current state, then clamps."""

    def __init__(self, controller: Controller, clamping: list[Clamping], device: wp.Device):
        self._controller = controller
        self._clamping = clamping
        self._device = device

    def is_graphable(self) -> bool:
        return self._controller.is_graphable()

    def compute_force(
        self,
        sim_state: Any,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        computed_forces: wp.array[float],
        applied_forces: wp.array[float] | None,
        ctrl_state: Controller.State | None,
        dt: float | None,
    ) -> wp.array[float]:
        """Compute raw effort into *computed_forces*, clamp into *applied_forces*.

        Returns the buffer holding the final (clamped) effort.
        """
        self._controller.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            feedforward,
            pos_indices,
            vel_indices,
            target_pos_indices,
            target_vel_indices,
            computed_forces,
            ctrl_state,
            dt,
            device=self._device,
        )
        forces = computed_forces
        for clamp in self._clamping:
            clamp.modify_forces(
                forces, applied_forces, positions, velocities, pos_indices, vel_indices, device=self._device
            )
            forces = applied_forces
        return forces
