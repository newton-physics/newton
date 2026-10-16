# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

SUPPLY_VOLTAGE_ATTR = "supply_voltage"


@wp.kernel
def _battery_draw_kernel(
    dof_battery: wp.array[wp.int32],
    joint_f: wp.array[float],
    hold: wp.array[wp.bool],
    torque: wp.array[float],
):
    d = wp.tid()
    b = dof_battery[d]
    if b < 0:
        return
    if hold[b]:
        return
    wp.atomic_add(torque, b, wp.abs(joint_f[d]))


@wp.kernel
def _battery_voltage_kernel(
    nominal_voltage: wp.array[float],
    sag_gain: wp.array[float],
    min_voltage: wp.array[float],
    torque: wp.array[float],
    hold: wp.array[wp.bool],
    voltage: wp.array[float],
):
    b = wp.tid()
    voltage[b] = wp.max(nominal_voltage[b] - sag_gain[b] * torque[b], min_voltage[b])
    hold[b] = False


@wp.kernel
def _battery_supply_kernel(
    dof_battery: wp.array[wp.int32],
    voltage: wp.array[float],
    supply_voltage: wp.array[float],
):
    d = wp.tid()
    b = dof_battery[d]
    if b >= 0:
        supply_voltage[d] = voltage[b]


@wp.kernel
def _battery_reset_kernel(
    mask: wp.array[wp.bool],
    nominal_voltage: wp.array[float],
    torque: wp.array[float],
    voltage: wp.array[float],
    hold: wp.array[wp.bool],
):
    b = wp.tid()
    if mask:
        if not mask[b]:
            return
    torque[b] = 0.0
    voltage[b] = nominal_voltage[b]
    hold[b] = True


class Battery:
    """Supply shared by the actuators of one or more DOFs.

    Holds B batteries; each per-battery array has one entry per battery, and
    :attr:`dof_battery` maps each DOF to the battery it draws from.

    The supply voltage sags with the total effort on the DOFs of a battery:

    .. math::

        V = \\max\\big(V_{\\text{nominal}} - g_{\\text{sag}} \\sum |\\tau|,\\;
        V_{\\min}\\big)

    :meth:`refresh` reads the effort from ``control.joint_f`` and writes the
    voltage of each DOF's battery into ``control.supply_voltage``. A drive
    that reads the supply voltage from there, e.g.
    :class:`~newton.actuators.DriveBAM` with ``supply_from_control=True``,
    declares ``supply_voltage`` as a custom input, so
    :class:`~newton.ModelBuilder` registers it on :class:`~newton.Control`.
    DOFs with battery ``-1`` are neither counted nor written.

    Call :meth:`refresh` once per step, before ``control.joint_f`` is
    cleared for the next actuator step: the voltage of one step is then
    computed from the effort of the previous one. Everything accumulated
    into ``joint_f`` on a battery's DOFs counts toward its draw.

    The supply and the drives form an explicit feedback loop. For
    :class:`~newton.actuators.DriveBAM` it settles only while
    ``sag_gain * sum(kt * max_pwm / resistance) < 1`` over the DOFs sharing
    the battery; above that the voltage oscillates between
    :attr:`min_voltage` and :attr:`nominal_voltage` instead of converging.
    The clamp bounds that oscillation but does not damp it, so an unstable
    setting still produces plausible-looking voltages.

    Usage::

        battery = Battery(dof_battery, nominal_voltage, sag_gain, min_voltage)
        for step in range(steps):
            battery.refresh(control)
            control.joint_f.zero_()
            for actuator in model.actuators:
                actuator.step(state, control, dt=dt)
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
            sag_gain: Voltage drop per unit of total effort [V/(N·m)].
                Shape ``(B,)``.
            min_voltage: Lower bound of the sagged voltage [V]. Shape ``(B,)``.

        Raises:
            ValueError: An array shape does not match *nominal_voltage*, or a
                *dof_battery* entry is outside ``[-1, B)``.
        """
        shape = nominal_voltage.shape
        for name, array in (("sag_gain", sag_gain), ("min_voltage", min_voltage)):
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} must match nominal_voltage shape {shape}")
        indices = dof_battery.numpy()
        if indices.size and (indices.min() < -1 or indices.max() >= shape[0]):
            raise ValueError(
                f"dof_battery entries must be -1 or in [0, {shape[0]}), got [{indices.min()}, {indices.max()}]"
            )
        device = nominal_voltage.device
        self.dof_battery = dof_battery
        """Battery index of each DOF, ``-1`` for none, shape (joint_dof_count,)."""
        self.nominal_voltage = nominal_voltage
        """Voltage without load [V], shape (B,)."""
        self.sag_gain = sag_gain
        """Voltage drop per unit of total effort [V/(N·m)], shape (B,)."""
        self.min_voltage = min_voltage
        """Lower bound of the sagged voltage [V], shape (B,)."""
        self.torque = wp.zeros(shape, dtype=float, device=device)
        """Total effort magnitude drawn at the last :meth:`refresh` [N·m], shape (B,)."""
        self.voltage = wp.zeros(shape, dtype=float, device=device)
        """Supply voltage [V], shape (B,)."""
        self._hold = wp.zeros(shape, dtype=wp.bool, device=device)
        self.reset()

    def refresh(self, control: Any) -> None:
        """Compute the voltage from ``control.joint_f`` and write ``control.supply_voltage``.

        Call once per step, before clearing ``joint_f`` for the next actuator
        step. A battery reset since the last call keeps its nominal voltage
        and ignores the effort in ``joint_f``.

        Args:
            control: Object carrying ``joint_f`` and ``supply_voltage``, both
                shaped like :attr:`dof_battery`, e.g. :class:`~newton.Control`.

        Raises:
            ValueError: ``joint_f`` or ``supply_voltage`` does not match the
                shape of :attr:`dof_battery`.
        """
        joint_f = control.joint_f
        supply_voltage = getattr(control, SUPPLY_VOLTAGE_ATTR)
        for name, array in (("joint_f", joint_f), (SUPPLY_VOLTAGE_ATTR, supply_voltage)):
            if array is None or array.shape != self.dof_battery.shape:
                shape = None if array is None else array.shape
                raise ValueError(f"control.{name} shape {shape} must match dof_battery shape {self.dof_battery.shape}")
        device = self.nominal_voltage.device
        self.torque.zero_()
        wp.launch(
            _battery_draw_kernel,
            dim=len(self.dof_battery),
            inputs=[self.dof_battery, joint_f, self._hold],
            outputs=[self.torque],
            device=device,
        )
        wp.launch(
            _battery_voltage_kernel,
            dim=len(self.nominal_voltage),
            inputs=[self.nominal_voltage, self.sag_gain, self.min_voltage, self.torque],
            outputs=[self._hold, self.voltage],
            device=device,
        )
        wp.launch(
            _battery_supply_kernel,
            dim=len(self.dof_battery),
            inputs=[self.dof_battery, self.voltage],
            outputs=[supply_voltage],
            device=device,
        )

    def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
        """Restore the nominal voltage and clear the drawn torque.

        The next :meth:`refresh` keeps the nominal voltage for the reset
        batteries.

        Args:
            mask: Boolean mask of length B. ``True`` entries are reset.
                ``None`` resets all.
        """
        wp.launch(
            _battery_reset_kernel,
            dim=len(self.nominal_voltage),
            inputs=[mask, self.nominal_voltage],
            outputs=[self.torque, self.voltage, self._hold],
            device=self.nominal_voltage.device,
        )
