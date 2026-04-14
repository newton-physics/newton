# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import warp as wp

from .base import Clamping


@wp.func
def _interp_1d(
    x: float,
    xs: wp.array(dtype=float),
    ys: wp.array(dtype=float),
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
def _remotized_clamp_kernel(
    current_pos: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    lookup_angles: wp.array(dtype=float),
    lookup_torques: wp.array(dtype=float),
    lookup_size: int,
    src: wp.array(dtype=float),
    dst: wp.array(dtype=float),
):
    """Angle-dependent clamping via interpolated lookup table: read src, write dst."""
    i = wp.tid()
    state_idx = state_indices[i]
    limit = _interp_1d(current_pos[state_idx], lookup_angles, lookup_torques, lookup_size)
    dst[i] = wp.clamp(src[i], -limit, limit)


class ClampingPositionBased(Clamping):
    """Angle-dependent torque clamping via lookup table.

    Replaces a fixed ±max_force box clamp with angle-dependent torque
    limits interpolated from a lookup table. Models remotized actuators
    (e.g., linkage-driven joints) where the transmission ratio and thus
    maximum output torque vary with joint angle.

    This is a post-controller dynamic.
    """

    SHARED_PARAMS = {"lookup_angles", "lookup_torques"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "lookup_angles" not in args or "lookup_torques" not in args:
            raise ValueError("ClampPositionBased requires 'lookup_angles' and 'lookup_torques' arguments")
        return {
            "lookup_angles": tuple(args["lookup_angles"]),
            "lookup_torques": tuple(args["lookup_torques"]),
        }

    def __init__(
        self,
        lookup_angles: wp.array | tuple[float, ...] | list[float],
        lookup_torques: wp.array | tuple[float, ...] | list[float],
    ):
        """Initialize remotized clamp dynamic.

        Args:
            lookup_angles: Sorted joint angles for the torque lookup table. Shape (K,).
            lookup_torques: Max output torques corresponding to lookup_angles. Shape (K,).
        """
        if len(lookup_angles) != len(lookup_torques):
            raise ValueError(
                f"lookup_angles length ({len(lookup_angles)}) must match "
                f"lookup_torques length ({len(lookup_torques)})"
            )
        self.lookup_size = len(lookup_angles)
        self._raw_angles = lookup_angles
        self._raw_torques = lookup_torques
        self.lookup_angles: wp.array | None = None
        self.lookup_torques: wp.array | None = None

        if isinstance(lookup_angles, wp.array):
            self.lookup_angles = lookup_angles
        if isinstance(lookup_torques, wp.array):
            self.lookup_torques = lookup_torques

    def set_device(self, device: wp.Device) -> None:
        if self.lookup_angles is None:
            self.lookup_angles = wp.array(
                np.array(self._raw_angles, dtype=np.float32), device=device
            )
        if self.lookup_torques is None:
            self.lookup_torques = wp.array(
                np.array(self._raw_torques, dtype=np.float32), device=device
            )

    def modify_forces(
        self,
        src_forces: wp.array,
        dst_forces: wp.array,
        positions: wp.array,
        velocities: wp.array,
        input_indices: wp.array,
        num_actuators: int,
    ) -> None:
        wp.launch(
            kernel=_remotized_clamp_kernel,
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
        )
