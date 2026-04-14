# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp

from .base import Controller


@wp.kernel
def _pd_force_kernel(
    current_pos: wp.array(dtype=float),
    current_vel: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    target_vel: wp.array(dtype=float),
    control_input: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    force_indices: wp.array(dtype=wp.uint32),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    constant_force: wp.array(dtype=float),
    forces: wp.array(dtype=float),
):
    """PD force: f = constant + act + kp*(target_pos - q) + kd*(target_vel - v)."""
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]
    force_idx = force_indices[i]

    position_error = target_pos[target_idx] - current_pos[state_idx]
    velocity_error = target_vel[target_idx] - current_vel[state_idx]

    const_f = float(0.0)
    if constant_force:
        const_f = constant_force[i]

    act = float(0.0)
    if control_input:
        act = control_input[target_idx]

    force = const_f + act + kp[i] * position_error + kd[i] * velocity_error
    forces[force_idx] = force


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
        kp: wp.array,
        kd: wp.array,
        constant_force: wp.array | None = None,
    ):
        """Initialize PD controller.

        Args:
            kp: Proportional gains. Shape (N,).
            kd: Derivative gains. Shape (N,).
            constant_force: Constant force offsets. Shape (N,). None to skip.
        """
        self.kp = kp
        self.kd = kd
        self.constant_force = constant_force

    def compute(
        self,
        positions: wp.array,
        velocities: wp.array,
        target_pos: wp.array,
        target_vel: wp.array,
        act_input: wp.array | None,
        input_indices: wp.array,
        target_indices: wp.array,
        forces: wp.array,
        force_indices: wp.array,
        num_actuators: int,
        state: Any,
        dt: float,
    ) -> None:
        wp.launch(
            kernel=_pd_force_kernel,
            dim=num_actuators,
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                act_input,
                input_indices,
                target_indices,
                force_indices,
                self.kp,
                self.kd,
                self.constant_force,
            ],
            outputs=[forces],
        )
