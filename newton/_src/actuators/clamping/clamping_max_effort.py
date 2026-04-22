# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import warp as wp

from .base import Clamping


@wp.kernel
def _box_clamp_kernel(
    max_effort: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
):
    """Clamp src efforts to ±max_effort, write to dst."""
    i = wp.tid()
    dst[i] = wp.clamp(src[i], -max_effort[i], max_effort[i])


class ClampingMaxEffort(Clamping):
    """Symmetric clamp on actuator output effort.

    Clamps the actuator output to ``[-max_effort, +max_effort]``.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        max_effort = args.get("max_effort", math.inf)
        if max_effort < 0:
            raise ValueError(f"max_effort must be non-negative, got {max_effort}")
        return {"max_effort": max_effort}

    def __init__(self, max_effort: wp.array[float]):
        """Initialize max-effort clamp.

        Args:
            max_effort: Per-actuator effort limits [N or N·m]. Shape ``(N,)``.
        """
        self.max_effort = max_effort

    def modify_forces(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_box_clamp_kernel,
            dim=len(src_forces),
            inputs=[self.max_effort, src_forces],
            outputs=[dst_forces],
            device=device,
        )
