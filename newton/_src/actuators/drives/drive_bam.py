# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from ...sim.model import Model
from .base import DriveBase


@wp.kernel
def _bam_effort_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    target_pos: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    kp: wp.array[float],
    max_pwm: wp.array[float],
    max_current: wp.array[float],
    kt: wp.array[float],
    resistance: wp.array[float],
    vin: wp.array[float],
    supply_from_battery: wp.array[float],
    supply_voltage: wp.array[float],
    efforts: wp.array[float],
    supply_current_per_effort: wp.array[float],
):
    i = wp.tid()
    vel = velocities[vel_indices[i]]

    supply = vin[i]
    if supply_from_battery[i] != 0.0:
        supply = supply_voltage[i]

    duty = (target_pos[target_pos_indices[i]] - positions[pos_indices[i]]) * kp[i]
    if max_current[i] > 0.0 and supply > 0.0:
        center = kt[i] * vel / supply
        span = resistance[i] * max_current[i] / supply
        duty = wp.clamp(duty, center - span, center + span)
    duty = wp.clamp(duty, -max_pwm[i], max_pwm[i])

    current = (supply * duty - kt[i] * vel) / resistance[i]
    efforts[i] = kt[i] * current
    if supply_current_per_effort:
        supply_current_per_effort[i] = duty / kt[i]


class DriveBAM(DriveBase):
    """Servo drive from BAM (Better Actuator Models), without friction.

    Models the servo firmware and the DC motor of an identified BAM fit:

    .. math::

        d_0 &= k_p\\, (q_{\\text{target}} - q) \\\\
        d_1 &= \\operatorname{clamp}\\Big(d_0,\\;
               \\frac{k_t\\, \\dot q - R\\, I_{\\max}}{V},\\;
               \\frac{k_t\\, \\dot q + R\\, I_{\\max}}{V}\\Big) \\\\
        d &= \\operatorname{clamp}(d_1,\\; -d_{\\max},\\; d_{\\max}) \\\\
        I &= \\frac{V\\, d - k_t\\, \\dot q}{R} \\\\
        \\tau &= k_t\\, I

    where :math:`d` is the PWM duty cycle, :math:`I` the motor current
    [A] and :math:`\\tau` the effort [N·m or N]. The symbols map to the
    parameters as follows:

    * :math:`k_p`: ``kp``, the firmware gain times the error gain of a BAM fit;
    * :math:`k_t`: ``kt``;
    * :math:`R`: ``resistance``;
    * :math:`V`: ``vin``, or ``actuator_supply_voltage`` with
      ``supply_from_battery``;
    * :math:`I_{\\max}`: ``max_current``; :math:`d_1 = d_0` when it is 0;
    * :math:`d_{\\max}`: ``max_pwm``.

    The current limit narrows the duty cycle so that :math:`|I| \\le
    I_{\\max}`. The ``max_pwm`` clamp is applied last, so at high velocity
    the back-EMF term can still drive :math:`|I|` above ``max_current``; this
    matches the BAM reference firmware. Target velocity and feedforward are
    ignored: the modelled firmware has no torque input.

    The supply voltage is either fixed or varying. A fixed supply, e.g. a
    bench or regulated power supply as used to identify a BAM fit, is set by
    ``vin``. A varying supply, e.g. a battery whose voltage sags under load,
    is enabled with ``supply_from_battery=True``: the drive then reads the
    voltage of each actuator from the ``actuator_supply_voltage`` input on
    ``sim_control`` (see :attr:`DriveBase.custom_inputs
    <newton.actuators.DriveBase.custom_inputs>`), which the battery's
    :meth:`~newton.actuators.BatteryBase.step` writes every step, and
    ignores ``vin``. It then also writes :math:`d / k_t` into the
    ``actuator_supply_current_per_effort`` output (see :attr:`DriveBase.custom_outputs
    <newton.actuators.DriveBase.custom_outputs>`): times the effort, it gives
    the current the actuator draws from the battery, since the battery
    supplies the motor current only during the PWM on-time.

    .. hint::

        This drive covers the firmware and the DC motor of a BAM servo. To
        model the complete servo, combine it with:

        * a :class:`~newton.actuators.ClampingMaxEffort` for the stall-torque
          bound, since the torque is not limited beyond the duty cycle and
          current limits;
        * joint friction in the solver (``Model.joint_friction``) for BAM's
          gearbox friction, which depends on the joint load;
        * optionally an :class:`~newton.actuators.InputProcessorDelay` for
          command latency, an :class:`~newton.actuators.InputProcessorBacklash`
          for gearbox backlash, and a battery
          (:meth:`~newton.ModelBuilder.add_battery`) for a sagging supply.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        supply_from_battery = float(bool(args.get("supply_from_battery", False)))
        required = {"kt", "resistance"} | (set() if supply_from_battery else {"vin"})
        missing = required - set(args)
        if missing:
            raise ValueError(f"DriveBAM is missing parameter(s): {', '.join(sorted(missing))}")
        resolved = {
            "kp": args.get("kp", 0.0),
            "kt": args["kt"],
            "resistance": args["resistance"],
            "vin": args.get("vin", 0.0),
            "max_pwm": args.get("max_pwm", 1.0),
            "max_current": args.get("max_current", 0.0),
            "supply_from_battery": supply_from_battery,
        }
        for name in ("kt", "resistance"):
            if resolved[name] <= 0.0:
                raise ValueError(f"{name} must be positive, got {resolved[name]}")
        for name in ("kp", "vin", "max_pwm", "max_current"):
            if resolved[name] < 0.0:
                raise ValueError(f"{name} must be non-negative, got {resolved[name]}")
        return resolved

    def __init__(
        self,
        kp: wp.array[float],
        kt: wp.array[float],
        resistance: wp.array[float],
        vin: wp.array[float],
        max_pwm: wp.array[float],
        max_current: wp.array[float] | None = None,
        supply_from_battery: wp.array[float] | None = None,
    ):
        """Initialize the BAM drive.

        Args:
            kp: Duty cycle per position error [1/rad or 1/m]. Shape ``(N,)``.
            kt: Torque constant [N·m/A]. Shape ``(N,)``.
            resistance: Motor resistance [Ω]. Shape ``(N,)``.
            vin: Supply voltage [V]. Ignored when *supply_from_battery* is
                set. Shape ``(N,)``.
            max_pwm: Largest duty cycle magnitude. Shape ``(N,)``.
            max_current: Firmware current limit [A]; 0 disables it. Shape
                ``(N,)``. ``None`` disables it.
            supply_from_battery: Per actuator, nonzero to read the supply
                voltage from the ``actuator_supply_voltage`` input, as written
                by the battery's :meth:`~newton.actuators.BatteryBase.step`,
                instead of *vin*. Shape ``(N,)``. ``None`` uses *vin* everywhere.

        Raises:
            ValueError: An array shape does not match *kp*, or *kt* or
                *resistance* is not positive.
        """
        n = kp.shape
        self.kp = kp
        """Duty cycle per position error [1/rad or 1/m], shape (N,)."""
        self.kt = kt
        """Torque constant [N·m/A], shape (N,)."""
        self.resistance = resistance
        """Motor resistance [Ω], shape (N,)."""
        self.vin = vin
        """Supply voltage [V], used unless :attr:`supply_from_battery`, shape (N,)."""
        self.max_pwm = max_pwm
        """Largest duty cycle magnitude, shape (N,)."""
        self.max_current = max_current if max_current is not None else wp.zeros(n, dtype=float, device=kp.device)
        """Firmware current limit [A], 0 when disabled, shape (N,)."""
        self.supply_from_battery = (
            supply_from_battery if supply_from_battery is not None else wp.zeros(n, dtype=float, device=kp.device)
        )
        """Per actuator, nonzero when the supply voltage is read from the battery's ``actuator_supply_voltage`` input,
        shape (N,)."""
        uses_battery = bool(self.supply_from_battery.numpy().any())
        self.custom_inputs = {"actuator_supply_voltage": Model.AttributeFrequency.ACTUATOR} if uses_battery else {}
        self.supply_current_per_effort = (
            wp.zeros(n, dtype=float, device=kp.device, requires_grad=kp.requires_grad) if uses_battery else None
        )
        """Battery current per unit of effort [A/(N·m) or A/N] of the last :meth:`compute`, or ``None`` without a
        battery, shape (N,)."""
        self.custom_outputs = (
            {"actuator_supply_current_per_effort": self.supply_current_per_effort} if uses_battery else {}
        )
        for name in ("kt", "resistance", "vin", "max_pwm", "max_current", "supply_from_battery"):
            shape = getattr(self, name).shape
            if shape != n:
                raise ValueError(f"{name} shape {shape} must match kp shape {n}")
        for name in ("kt", "resistance"):
            values = getattr(self, name).numpy()
            if values.size and values.min() <= 0.0:
                raise ValueError(f"{name} must be positive, got {values.min()}")

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
        state: DriveBase.State | None,
        dt: float,
        device: wp.Device | None = None,
        *,
        custom_inputs: dict[str, Any] | None = None,
    ) -> None:
        """Compute the servo torque.

        Args:
            custom_inputs: Holds the ``actuator_supply_voltage`` array [V],
                one entry per slot, when an actuator has
                :attr:`supply_from_battery` set.

        Raises:
            ValueError: An actuator has :attr:`supply_from_battery` set and
                ``actuator_supply_voltage`` is missing.

        See :meth:`DriveBase.compute` for the other arguments.
        """
        supply_voltage = None
        if self.custom_inputs:
            supply_voltage = (custom_inputs or {}).get("actuator_supply_voltage")
            if supply_voltage is None:
                raise ValueError("DriveBAM with supply_from_battery requires 'actuator_supply_voltage' on sim_control.")
        wp.launch(
            kernel=_bam_effort_kernel,
            dim=len(forces),
            inputs=[
                positions,
                velocities,
                target_pos,
                pos_indices,
                vel_indices,
                target_pos_indices,
                self.kp,
                self.max_pwm,
                self.max_current,
                self.kt,
                self.resistance,
                self.vin,
                self.supply_from_battery,
                supply_voltage,
            ],
            outputs=[forces, self.supply_current_per_effort],
            device=device,
        )
