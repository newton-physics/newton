# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import warp as wp

from .base import Clamping


@wp.kernel
def _box_clamp_kernel(
    max_force: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
):
    """Clamp src forces to ±max_force, write to dst."""
    i = wp.tid()
    dst[i] = wp.clamp(src[i], -max_force[i], max_force[i])


class ClampingMaxForce(Clamping):
    """Box-clamp forces to ±max_force per actuator."""

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        max_force = args.get("max_force", math.inf)
        if max_force < 0:
            raise ValueError(f"max_force must be non-negative, got {max_force}")
        return {"max_force": max_force}

    def __init__(self, max_force: wp.array):
        """Initialize clamp dynamic.

        Args:
            max_force: Per-actuator force limits. Shape (N,).
        """
        self.max_force = max_force

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
            kernel=_box_clamp_kernel,
            dim=num_actuators,
            inputs=[self.max_force, src_forces],
            outputs=[dst_forces],
        )
