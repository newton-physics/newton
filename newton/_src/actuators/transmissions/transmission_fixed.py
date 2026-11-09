# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

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
def _transmission_fixed_velocity_kernel(
    velocities: wp.array[float],
    dof_indices: wp.array2d[wp.uint32],
    gear: wp.array2d[float],
    output: wp.array[float],
):
    i = wp.tid()
    v = float(0.0)
    for j in range(dof_indices.shape[0]):
        v += gear[j, i] * velocities[dof_indices[j, i]]
    output[i] = v


class TransmissionFixed(TransmissionBase):
    """Transmission with constant gear ratios.

    Output joint ``j`` of actuator ``i`` receives

    .. math::

        \\tau_{\\text{effort}[j, i]} \\mathrel{+}= g_{j, i} \\, \\tau_i

    where :math:`\\tau_i` is the actuator effort. The motor turns at

    .. math::

        \\dot q_i = \\sum_j g_{j, i} \\, \\dot q_{\\text{effort}[j, i]}

    Covers gearboxes, belt and cable reductions, differentials and MuJoCo's ``gear`` and fixed tendons.

    With :meth:`~newton.ModelBuilder.add_actuator`, pass ``gear``, one ratio
    per output joint or one for all of them (default: 1.0).
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {"gear": args.get("gear", 1.0)}

    def __init__(self, gear: wp.array2d[float]):
        """Initialize a fixed transmission.

        Args:
            gear: Ratio of joint effort to actuator effort, shape
                ``(joints, N)``, matching the actuator's output indices.
        """
        if gear.ndim != 2:
            raise ValueError(f"gear must be a 2-D array of shape (joints, N), got {gear.ndim} dimensions")
        self.gear = gear
        """Ratio of joint effort to actuator effort, shape ``(joints, N)``."""

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int) -> None:
        if tuple(self.gear.shape) != (num_joints, num_actuators):
            raise ValueError(f"gear shape {tuple(self.gear.shape)} must be ({num_joints}, {num_actuators})")

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

    def compute_velocity(
        self,
        velocities: wp.array[float],
        dof_indices: wp.array2d[wp.uint32],
        output: wp.array[float],
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_transmission_fixed_velocity_kernel,
            dim=output.shape[0],
            inputs=[velocities, dof_indices, self.gear],
            outputs=[output],
            device=device,
        )
