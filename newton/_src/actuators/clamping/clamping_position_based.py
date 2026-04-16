# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import warp as wp

from .base import Clamping


@wp.func
def _interp_1d(
    x: float,
    xs: wp.array[float],
    ys: wp.array[float],
    n: int,
) -> float:
    """Linearly interpolate (x -> y) from sorted sample arrays, clamping at boundaries."""
    if n <= 0:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    for k in range(n - 1):
        if xs[k + 1] >= x:
            dx = xs[k + 1] - xs[k]
            if dx == 0.0:
                return ys[k]
            t = (x - xs[k]) / dx
            return ys[k] + t * (ys[k + 1] - ys[k])
    return ys[n - 1]


@wp.kernel
def _position_based_clamp_kernel(
    current_pos: wp.array[float],
    state_indices: wp.array[wp.uint32],
    lookup_angles: wp.array[float],
    lookup_torques: wp.array[float],
    lookup_size: int,
    src: wp.array[float],
    dst: wp.array[float],
):
    """Angle-dependent clamping via interpolated lookup table: read src, write dst."""
    i = wp.tid()
    state_idx = state_indices[i]
    limit = _interp_1d(current_pos[state_idx], lookup_angles, lookup_torques, lookup_size)
    dst[i] = wp.clamp(src[i], -limit, limit)


class ClampingPositionBased(Clamping):
    """Angle-dependent torque clamping via lookup table.

    Replaces a fixed ±max_force box clamp with angle-dependent torque
    limits interpolated from a lookup table. Models actuators where
    the transmission ratio and thus maximum output torque vary with
    joint angle (e.g., linkage-driven joints).
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"lookup_angles", "lookup_torques"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "lookup_angles" not in args or "lookup_torques" not in args:
            raise ValueError("ClampingPositionBased requires 'lookup_angles' and 'lookup_torques' arguments")
        return {
            "lookup_angles": tuple(args["lookup_angles"]),
            "lookup_torques": tuple(args["lookup_torques"]),
        }

    def __init__(
        self,
        lookup_angles: wp.array[float] | tuple[float, ...] | list[float],
        lookup_torques: wp.array[float] | tuple[float, ...] | list[float],
    ):
        """Initialize position-based clamp.

        Args:
            lookup_angles: Sorted joint angles [rad] for the torque lookup table. Shape (K,).
            lookup_torques: Max output torques [N·m] corresponding to lookup_angles. Shape (K,).
        """
        if len(lookup_angles) != len(lookup_torques):
            raise ValueError(
                f"lookup_angles length ({len(lookup_angles)}) must match lookup_torques length ({len(lookup_torques)})"
            )
        if not isinstance(lookup_torques, wp.array) and any(t < 0 for t in lookup_torques):
            raise ValueError("lookup_torques must contain non-negative values for symmetric clamping")
        self.lookup_size = len(lookup_angles)
        if not isinstance(lookup_angles, wp.array):
            lookup_angles = wp.array(np.array(lookup_angles, dtype=np.float32))
        if not isinstance(lookup_torques, wp.array):
            lookup_torques = wp.array(np.array(lookup_torques, dtype=np.float32))
        self.lookup_angles = lookup_angles
        self.lookup_torques = lookup_torques

    def modify_forces(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        input_indices: wp.array[wp.uint32],
        num_actuators: int,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_position_based_clamp_kernel,
            dim=num_actuators,
            inputs=[
                positions,
                input_indices,
                self.lookup_angles,
                self.lookup_torques,
                self.lookup_size,
                src_forces,
            ],
            outputs=[dst_forces],
            device=device,
        )
