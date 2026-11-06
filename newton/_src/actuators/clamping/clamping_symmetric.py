# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any, ClassVar

import warp as wp

from ..space import DriveOutputSpace
from .base import ClampingBase


@wp.func
def _evaluate_symmetric_clamp(
    value: wp.float64,
    q: wp.float64,
    qd: wp.float64,
    params: wp.array2d[float],
    i: int,
    base: int,
) -> wp.float64:
    """Implicit-solve entry point; params row is ``[limit]``."""
    limit = wp.float64(params[i, base])
    return wp.clamp(value, -limit, limit)


@wp.kernel
def _box_clamp_kernel(
    limit: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
):
    """Clamp src efforts to ±limit, write to dst."""
    i = wp.tid()
    dst[i] = wp.clamp(src[i], -limit[i], limit[i])


class ClampingSymmetric(ClampingBase):
    """Clamp a drive output to [-limit, +limit] in its declared input space.

    EFFORT limits are forces [N] or torques [N·m]. DUTY limits are dimensionless.
    This clamp does not convert between the two spaces.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"input_space"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        limit = args.get("limit", math.inf)
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        return {"limit": limit, "input_space": DriveOutputSpace(args.get("input_space", DriveOutputSpace.EFFORT))}

    def __init__(self, limit: wp.array[float], *, input_space: DriveOutputSpace = DriveOutputSpace.EFFORT):
        """Initialize max-effort clamp.

        Args:
            limit: Non-negative bounds in the selected input units. Shape ``(N,)``.
            input_space: EFFORT [N or N·m] or dimensionless DUTY.
        """
        self.input_space = DriveOutputSpace(input_space)
        self.limit = limit

    evaluate_clamp = _evaluate_symmetric_clamp

    def param_width(self) -> int:
        return 1

    def bind_params(self, block: wp.array2d[float]) -> None:
        block[:, 0].assign(self.limit)
        self.limit = block[:, 0]

    def modify_forces(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        device: wp.Device | None = None,
        *,
        supply_voltage: wp.array[float] | None = None,
    ) -> None:
        wp.launch(
            kernel=_box_clamp_kernel,
            dim=len(src_forces),
            inputs=[self.limit, src_forces],
            outputs=[dst_forces],
            device=device,
        )
