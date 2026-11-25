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
    efficiency: wp.array2d[float],
    output: wp.array[float],
    computed_output: wp.array[float],
):
    j, i = wp.tid()
    dof = dof_indices[j, i]
    ratio = efficiency[j, i] * gear[j, i]
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


@wp.kernel
def _transmission_fixed_velocity_kernel(
    velocities: wp.array[float],
    dof_indices: wp.array2d[wp.uint32],
    gear: wp.array2d[float],
    output_velocities: wp.array[float],
):
    i = wp.tid()
    qd = float(0.0)
    for j in range(dof_indices.shape[0]):
        qd += gear[j, i] * velocities[dof_indices[j, i]]
    output_velocities[i] = qd


class TransmissionFixed(TransmissionBase):
    """Transmission with constant gear ratios.

    Actuator ``i`` is at

    .. math::

        s_i = \\sum_j g_{j, i} \\, q_{\\text{pos\\_indices}[j, i]}, \\qquad
        \\dot s_i = \\sum_j g_{j, i} \\, \\dot q_{\\text{indices}[j, i]}

    and DOF ``effort_indices[j, i]`` receives
    :math:`\\eta_{j, i} \\, g_{j, i} \\, \\tau_i`, where :math:`\\tau_i` is
    the actuator effort and :math:`\\eta_{j, i}` the efficiency. For example,
    a gripper with ``gear=[0.5, 0.5]`` reads the mean finger position
    ``0.5 qL + 0.5 qR``, half its opening, and gives each finger half the
    effort.

    The efficiency models the losses of the transmission, e.g. 0.8 for a
    transmission that loses 20 % of the power: it scales only the effort,
    not the positions and velocities. It is constant and does not depend on
    the direction of the power flow, so a joint that back-drives the
    actuator also receives :math:`\\eta \\, g \\, \\tau`: back-driving
    losses are not modeled, and the joint returns less power than the
    actuator absorbs. With efficiency 1, the default, the transmission
    conserves power.

    Covers gearboxes, belt and cable reductions, differentials and MuJoCo's ``gear`` and fixed tendons.

    With :meth:`~newton.ModelBuilder.add_actuator`, pass ``gear`` and
    ``efficiency``, each one value per joint or one for all of them
    (defaults: 1.0).
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        efficiency = args.get("efficiency")
        if efficiency is None:
            efficiency = 1.0
        if not np.all((np.asarray(efficiency) > 0.0) & (np.asarray(efficiency) <= 1.0)):
            raise ValueError(f"TransmissionFixed efficiency must be in (0, 1], got {efficiency}")
        return {"gear": args.get("gear", 1.0), "efficiency": efficiency}

    def __init__(self, gear: wp.array2d[float], efficiency: wp.array2d[float] | None = None):
        """Initialize a fixed transmission.

        Args:
            gear: Ratio of actuator position to joint position, and of
                joint effort to actuator effort, shape ``(joints, N)``,
                matching the actuator's indices. Dimensionless when the
                actuator and joints are both rotational or both linear.
            efficiency: Fraction of the geared actuator effort each joint
                receives, in ``(0, 1]``, shape ``(joints, N)``. ``None`` uses
                1 for every joint.

        Raises:
            ValueError: *gear* or *efficiency* is not 2-D.
        """
        if efficiency is None:
            efficiency = wp.ones(gear.shape, dtype=wp.float32, device=gear.device)
        for name, array in (("gear", gear), ("efficiency", efficiency)):
            if array.ndim != 2:
                raise ValueError(f"{name} must be a 2-D array of shape (joints, N), got {array.ndim} dimensions")
        self.gear = gear
        """Ratio of actuator position to joint position, and of joint effort to actuator effort, shape ``(joints, N)``."""
        self.efficiency = efficiency
        """Fraction of the geared actuator effort each joint receives, shape ``(joints, N)``."""

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int, requires_grad: bool = False) -> None:
        for name, array in (("gear", self.gear), ("efficiency", self.efficiency)):
            if tuple(array.shape) != (num_joints, num_actuators):
                raise ValueError(f"{name} shape {tuple(array.shape)} must be ({num_joints}, {num_actuators})")
        self._positions = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._velocities = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._motor_velocities = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._indices = wp.array(np.arange(num_actuators, dtype=np.uint32), device=device)

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        dof_indices: wp.array2d[wp.uint32],
        forces: wp.array[float],
        computed_forces: wp.array[float],
        effort_indices: wp.array2d[wp.uint32],
        output: wp.array[float],
        computed_output: wp.array[float] | None,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_transmission_fixed_kernel,
            dim=effort_indices.shape,
            inputs=[forces, computed_forces, effort_indices, self.gear, self.efficiency],
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
        out_positions, out_velocities = self._positions, self._velocities
        if self._positions.requires_grad:
            out_positions, out_velocities = wp.empty_like(out_positions), wp.empty_like(out_velocities)
        wp.launch(
            kernel=_transmission_fixed_state_kernel,
            dim=self._positions.shape[0],
            inputs=[positions, velocities, pos_indices, dof_indices, self.gear],
            outputs=[out_positions, out_velocities],
            device=device,
        )
        return out_positions, self._indices, out_velocities, self._indices

    def compute_velocity(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        dof_indices: wp.array2d[wp.uint32],
        device: wp.Device | None = None,
    ) -> tuple[wp.array[float], wp.array[wp.uint32]]:
        out_velocities = self._motor_velocities
        if out_velocities.requires_grad:
            out_velocities = wp.empty_like(out_velocities)
        wp.launch(
            kernel=_transmission_fixed_velocity_kernel,
            dim=out_velocities.shape[0],
            inputs=[velocities, dof_indices, self.gear],
            outputs=[out_velocities],
            device=device,
        )
        return out_velocities, self._indices
