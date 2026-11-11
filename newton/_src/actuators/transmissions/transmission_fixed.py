# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .base import TransmissionBase


@wp.kernel
def _transmission_fixed_kernel(
    forces: wp.array[float],
    computed_forces: wp.array[float],
    dof_indices: wp.array2d[wp.uint32],
    gear: wp.array2d[float],
    output: wp.array[float],
    computed_output: wp.array[float],
):
    j, i = wp.tid()
    dof = dof_indices[j, i]
    ratio = gear[j, i]
    wp.atomic_add(output, dof, ratio * forces[i])
    if computed_output:
        wp.atomic_add(computed_output, dof, ratio * computed_forces[i])


@wp.kernel
def _transmission_fixed_state_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    pos_indices: wp.array2d[wp.uint32],
    dof_indices: wp.array2d[wp.uint32],
    gear: wp.array2d[float],
    output_positions: wp.array[float],
    output_velocities: wp.array[float],
):
    i = wp.tid()
    q = float(0.0)
    qd = float(0.0)
    for j in range(dof_indices.shape[0]):
        q += gear[j, i] * positions[pos_indices[j, i]]
        qd += gear[j, i] * velocities[dof_indices[j, i]]
    output_positions[i] = q
    output_velocities[i] = qd


class TransmissionFixed(TransmissionBase):
    """Transmission with constant gear ratios.

    Actuator ``i`` is at

    .. math::

        s_i = \\sum_j g_{j, i} \\, q_{\\text{index}[j, i]}, \\qquad
        \\dot s_i = \\sum_j g_{j, i} \\, \\dot q_{\\text{index}[j, i]}

    and joint ``j`` receives :math:`g_{j, i} \\, \\tau_i`, where
    :math:`\\tau_i` is the actuator effort. For example, a gripper with
    ``gear=[0.5, 0.5]`` reads its opening ``0.5 qL + 0.5 qR`` and gives
    each finger half the effort.

    Covers gearboxes, belt and cable reductions, differentials and MuJoCo's ``gear`` and fixed tendons.

    With :meth:`~newton.ModelBuilder.add_actuator`, pass ``gear``, one ratio
    per joint or one for all of them (default: 1.0).
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {"gear": args.get("gear", 1.0)}

    def __init__(self, gear: wp.array2d[float]):
        """Initialize a fixed transmission.

        Args:
            gear: Ratio of actuator position to joint position, and of
                joint effort to actuator effort, shape ``(joints, N)``,
                matching the actuator's indices.
        """
        if gear.ndim != 2:
            raise ValueError(f"gear must be a 2-D array of shape (joints, N), got {gear.ndim} dimensions")
        self.gear = gear
        """Ratio of actuator position to joint position, shape ``(joints, N)``."""

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int, requires_grad: bool = False) -> None:
        if tuple(self.gear.shape) != (num_joints, num_actuators):
            raise ValueError(f"gear shape {tuple(self.gear.shape)} must be ({num_joints}, {num_actuators})")
        self._positions = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._velocities = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._indices = wp.array(np.arange(num_actuators, dtype=np.uint32), device=device)

    def compute(
        self,
        forces: wp.array[float],
        computed_forces: wp.array[float],
        dof_indices: wp.array2d[wp.uint32],
        output: wp.array[float],
        computed_output: wp.array[float] | None,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_transmission_fixed_kernel,
            dim=dof_indices.shape,
            inputs=[forces, computed_forces, dof_indices, self.gear],
            outputs=[output, computed_output],
            device=device,
        )

    def compute_state(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        dof_indices: wp.array2d[wp.uint32],
        device: wp.Device | None = None,
    ) -> tuple[wp.array[float], wp.array[wp.uint32], wp.array[float], wp.array[wp.uint32]]:
        wp.launch(
            kernel=_transmission_fixed_state_kernel,
            dim=self._positions.shape[0],
            inputs=[positions, velocities, pos_indices, dof_indices, self.gear],
            outputs=[self._positions, self._velocities],
            device=device,
        )
        return self._positions, self._indices, self._velocities, self._indices
