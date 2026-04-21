# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from .base import Controller


@wp.kernel
def _pd_force_kernel(
    current_pos: wp.array[float],
    current_vel: wp.array[float],
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    control_input: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    kp: wp.array[float],
    kd: wp.array[float],
    constant_force: wp.array[float],
    forces: wp.array[float],
):
    """PD force: f = constant + act + kp*(target_pos - q) + kd*(target_vel - v)."""
    i = wp.tid()
    pos_idx = pos_indices[i]
    vel_idx = vel_indices[i]
    tgt_pos_idx = target_pos_indices[i]
    tgt_vel_idx = target_vel_indices[i]

    position_error = target_pos[tgt_pos_idx] - current_pos[pos_idx]
    velocity_error = target_vel[tgt_vel_idx] - current_vel[vel_idx]

    const_f = float(0.0)
    if constant_force:
        const_f = constant_force[i]

    act = float(0.0)
    if control_input:
        act = control_input[tgt_vel_idx]

    force = const_f + act + kp[i] * position_error + kd[i] * velocity_error
    forces[i] = force


class ControllerPD(Controller):
    """Stateless PD controller.

    Force law: f = constant + act + Kp*(target_pos - q) + Kd*(target_vel - v)

    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "kp": args.get("kp", 0.0),
            "kd": args.get("kd", 0.0),
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        kp: wp.array[float],
        kd: wp.array[float],
        constant_force: wp.array[float] | None = None,
    ):
        """Initialize PD controller.

        Args:
            kp: Proportional gains. Shape (N,).
            kd: Derivative gains. Shape (N,).
            constant_force: Constant force offsets [N or N·m]. Shape (N,). None to skip.
        """
        if kp.shape != kd.shape:
            raise ValueError(f"kp shape {kp.shape} must match kd shape {kd.shape}")
        if constant_force is not None and constant_force.shape != kp.shape:
            raise ValueError(f"constant_force shape {constant_force.shape} must match kp shape {kp.shape}")
        self.kp = kp
        self.kd = kd
        self.constant_force = constant_force

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
            kernel=_pd_force_kernel,
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
                self.constant_force,
            ],
            outputs=[forces],
            device=device,
        )
