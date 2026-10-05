# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp


@wp.kernel
def _battery_refresh_kernel(
    nominal_voltage: wp.array[float],
    sag_gain: wp.array[float],
    min_voltage: wp.array[float],
    torque: wp.array[float],
    voltage: wp.array[float],
):
    b = wp.tid()
    voltage[b] = wp.max(nominal_voltage[b] - sag_gain[b] * torque[b], min_voltage[b])
    torque[b] = 0.0


@wp.kernel
def _battery_reset_kernel(
    mask: wp.array[wp.bool],
    nominal_voltage: wp.array[float],
    torque: wp.array[float],
    voltage: wp.array[float],
):
    b = wp.tid()
    if mask:
        if not mask[b]:
            return
    torque[b] = 0.0
    voltage[b] = nominal_voltage[b]


class Battery:
    """Supply shared by the actuators of one or more DOFs.

    The supply voltage sags with the total motor torque drawn from it:

    .. math::

        V = \\max\\big(V_{\\text{nominal}} - g_{\\text{sag}} \\sum |\\tau_{\\text{motor}}|,\\;
        V_{\\min}\\big)

    Pass the battery to :meth:`Actuator.step <newton.actuators.Actuator.step>`.
    A drive that uses a battery (e.g. :class:`~newton.actuators.DriveBAM`)
    reads :attr:`voltage` and adds the magnitude of its motor torque to
    :attr:`torque`. Call :meth:`refresh` once per step, before stepping the
    actuators: it computes :attr:`voltage` from the torque drawn on the
    previous step by every actuator on the same battery, also across
    actuator groups, and clears :attr:`torque`.

    Usage::

        battery = Battery(dof_battery, nominal_voltage, sag_gain, min_voltage)
        for step in range(steps):
            battery.refresh()
            control.joint_f.zero_()
            for actuator in model.actuators:
                actuator.step(state, control, dt=dt, battery=battery)
    """

    def __init__(
        self,
        dof_battery: wp.array[wp.int32],
        nominal_voltage: wp.array[float],
        sag_gain: wp.array[float],
        min_voltage: wp.array[float],
    ):
        """Initialize batteries at their nominal voltage.

        Args:
            dof_battery: Battery index of each DOF, ``-1`` for DOFs without a
                battery. Shape ``(joint_dof_count,)``, indexed like
                ``joint_qd``.
            nominal_voltage: Voltage without load [V]. Shape ``(B,)``.
            sag_gain: Voltage drop per unit of total motor torque [V/(N·m)].
                Shape ``(B,)``.
            min_voltage: Lower bound of the sagged voltage [V]. Shape ``(B,)``.
        """
        shape = nominal_voltage.shape
        for name, array in (("sag_gain", sag_gain), ("min_voltage", min_voltage)):
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} must match nominal_voltage shape {shape}")
        device = nominal_voltage.device
        self.dof_battery = dof_battery
        """Battery index of each DOF, ``-1`` for none, shape (joint_dof_count,)."""
        self.nominal_voltage = nominal_voltage
        """Voltage without load [V], shape (B,)."""
        self.sag_gain = sag_gain
        """Voltage drop per unit of total motor torque [V/(N·m)], shape (B,)."""
        self.min_voltage = min_voltage
        """Lower bound of the sagged voltage [V], shape (B,)."""
        self.torque = wp.zeros(shape, dtype=float, device=device)
        """Total motor torque magnitude drawn since the last :meth:`refresh` [N·m], shape (B,)."""
        self.voltage = wp.zeros(shape, dtype=float, device=device)
        """Supply voltage read by the drives [V], shape (B,)."""
        self.reset()

    def refresh(self) -> None:
        """Compute the voltage from the drawn torque and clear :attr:`torque`."""
        wp.launch(
            _battery_refresh_kernel,
            dim=len(self.nominal_voltage),
            inputs=[self.nominal_voltage, self.sag_gain, self.min_voltage],
            outputs=[self.torque, self.voltage],
            device=self.nominal_voltage.device,
        )

    def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
        """Restore the nominal voltage and clear the drawn torque.

        Args:
            mask: Boolean mask of length B. ``True`` entries are reset.
                ``None`` resets all.
        """
        wp.launch(
            _battery_reset_kernel,
            dim=len(self.nominal_voltage),
            inputs=[mask, self.nominal_voltage],
            outputs=[self.torque, self.voltage],
            device=self.nominal_voltage.device,
        )
