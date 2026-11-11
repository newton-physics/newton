# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from .base import StateEncoderBase


@wp.kernel
def _state_encoder_fixed_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    pos_indices: wp.array2d[wp.uint32],
    vel_indices: wp.array2d[wp.uint32],
    position_gear: wp.array2d[float],
    velocity_gear: wp.array2d[float],
    output_positions: wp.array[float],
    output_velocities: wp.array[float],
):
    i = wp.tid()
    q = float(0.0)
    qd = float(0.0)
    for j in range(pos_indices.shape[0]):
        q += position_gear[j, i] * positions[pos_indices[j, i]]
        qd += velocity_gear[j, i] * velocities[vel_indices[j, i]]
    output_positions[i] = q
    output_velocities[i] = qd


class StateEncoderFixed(StateEncoderBase):
    """Encoder measuring weighted sums of the input joint positions and velocities.

    Actuator ``i`` reads

    .. math::

        q_i = \\sum_j g^{q}_{j, i} \\, q_{\\text{input}[j, i]}, \\qquad
        \\dot q_i = \\sum_j g^{\\dot q}_{j, i} \\, \\dot q_{\\text{input}[j, i]}

    For example, a sensor mounted after a passive backlash joint between the
    actuated joint and the link reads both joints with
    ``position_gear=[1, 1]``:

    .. math::

        q = q_{\\text{actuated}} + q_{\\text{backlash}}

    This captures gearbox backlash when the backlash joint has a small motion
    range: the solver handles the dead zone, where the actuator effort does
    not reach the link. A drive that reads the position from a link-side
    sensor and the velocity from the motor uses ``velocity_gear=[1, 0]``.

    With :meth:`~newton.ModelBuilder.add_actuator`, pass ``position_gear`` and
    ``velocity_gear``, one weight per input joint or one for all of them.
    ``position_gear`` defaults to 1.0 and ``velocity_gear`` to
    ``position_gear``.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        position_gear = args.get("position_gear", 1.0)
        return {"position_gear": position_gear, "velocity_gear": args.get("velocity_gear", position_gear)}

    def __init__(self, position_gear: wp.array2d[float], velocity_gear: wp.array2d[float] | None = None):
        """Initialize a state encoder.

        Args:
            position_gear: Weight of each input joint position of each
                actuator, shape ``(joints, N)``, matching the actuator's input
                indices.
            velocity_gear: Weight of each input joint velocity of each
                actuator, shape ``(joints, N)``. ``None`` uses
                *position_gear*.
        """
        if velocity_gear is None:
            velocity_gear = position_gear
        for name, gear in (("position_gear", position_gear), ("velocity_gear", velocity_gear)):
            if gear.ndim != 2:
                raise ValueError(f"{name} must be a 2-D array of shape (joints, N), got {gear.ndim} dimensions")
        self.position_gear = position_gear
        """Weight of each input joint position of each actuator, shape ``(joints, N)``."""
        self.velocity_gear = velocity_gear
        """Weight of each input joint velocity of each actuator, shape ``(joints, N)``."""

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int) -> None:
        for name, gear in (("position_gear", self.position_gear), ("velocity_gear", self.velocity_gear)):
            if tuple(gear.shape) != (num_joints, num_actuators):
                raise ValueError(f"{name} shape {tuple(gear.shape)} must be ({num_joints}, {num_actuators})")

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        vel_indices: wp.array2d[wp.uint32],
        output_positions: wp.array[float],
        output_velocities: wp.array[float],
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_state_encoder_fixed_kernel,
            dim=output_positions.shape[0],
            inputs=[positions, velocities, pos_indices, vel_indices, self.position_gear, self.velocity_gear],
            outputs=[output_positions, output_velocities],
            device=device,
        )
