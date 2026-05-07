# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from .base import Controller


@wp.kernel
def _pd_effort_kernel(
    current_pos: wp.array[float],
    current_vel: wp.array[float],
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    feedforward: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    kp: wp.array[float],
    kd: wp.array[float],
    const_effort: wp.array[float],
    efforts: wp.array[float],
):
    """effort = const_effort + feedforward + kp*(target_pos - current_pos) + kd*(target_vel - current_vel)."""
    i = wp.tid()
    pos_idx = pos_indices[i]
    vel_idx = vel_indices[i]
    tgt_pos_idx = target_pos_indices[i]
    tgt_vel_idx = target_vel_indices[i]

    position_error = target_pos[tgt_pos_idx] - current_pos[pos_idx]
    velocity_error = target_vel[tgt_vel_idx] - current_vel[vel_idx]

    const_e = float(0.0)
    if const_effort:
        const_e = const_effort[i]

    ff = float(0.0)
    if feedforward:
        ff = feedforward[tgt_vel_idx]

    effort = const_e + ff + kp[i] * position_error + kd[i] * velocity_error
    efforts[i] = effort


class ControllerPD(Controller):
    """Stateless PD (Proportional-Derivative) controller.

    Effort law::

        effort = const_effort + feedforward + kp * (target_pos - current_pos) + kd * (target_vel - current_vel)
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        kp = args.get("kp", 0.0)
        if kp < 0:
            raise ValueError(f"kp must be non-negative, got {kp}")
        kd = args.get("kd", 0.0)
        if kd < 0:
            raise ValueError(f"kd must be non-negative, got {kd}")
        return {"kp": kp, "kd": kd, "const_effort": args.get("const_effort", 0.0)}

    def __init__(
        self,
        kp: wp.array[float],
        kd: wp.array[float],
        const_effort: wp.array[float] | None = None,
    ):
        """Initialize PD controller.

        Args:
            kp: Proportional gains [N/m or N·m/rad]. Shape ``(N,)``.
            kd: Derivative gains [N·s/m or N·m·s/rad]. Shape ``(N,)``.
            const_effort: Constant bias effort [N or N·m]. Shape ``(N,)``. ``None`` to skip.
        """
        if kp.shape != kd.shape:
            raise ValueError(f"kp shape {kp.shape} must match kd shape {kd.shape}")
        if const_effort is not None and const_effort.shape != kp.shape:
            raise ValueError(f"const_effort shape {const_effort.shape} must match kp shape {kp.shape}")
        self.kp = kp
        self.kd = kd
        self.const_effort = const_effort

    def is_stateful(self) -> bool:
        return False

    def is_graphable(self) -> bool:
        return True

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: Controller.State | None,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_pd_effort_kernel,
            dim=len(forces),
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                feedforward,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                self.kp,
                self.kd,
                self.const_effort,
            ],
            outputs=[forces],
            device=device,
        )
