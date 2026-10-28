# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from .base import MotorBase


@wp.kernel
def _motor_dc_kernel(
    velocities: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    actuator_indices: wp.array[wp.uint32],
    kt: wp.array[float],
    resistance: wp.array[float],
    vin: wp.array[float],
    max_pwm: wp.array[float],
    max_current: wp.array[float],
    supply_from_battery: wp.array[float],
    supply_voltage: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
    supply_current: wp.array[float],
):
    i = wp.tid()
    vel = velocities[vel_indices[i]]

    voltage = vin[i]
    if supply_from_battery[i] != 0.0:
        voltage = supply_voltage[actuator_indices[i]]

    duty = (src[i] * resistance[i] / kt[i] + kt[i] * vel) / vin[i]
    if max_current[i] > 0.0 and voltage > 0.0:
        center = kt[i] * vel / voltage
        span = resistance[i] * max_current[i] / voltage
        duty = wp.clamp(duty, center - span, center + span)
    duty = wp.clamp(duty, -max_pwm[i], max_pwm[i])

    current = (voltage * duty - kt[i] * vel) / resistance[i]
    dst[i] = kt[i] * current
    if supply_current:
        supply_current[actuator_indices[i]] = duty * current


class MotorDC(MotorBase):
    """PWM-driven DC motor, as in BAM (Better Actuator Models).

    Turns the requested effort :math:`\\tau_{\\text{req}}` into the PWM duty
    cycle that produces it at the supply voltage ``vin``, applies the
    firmware limits, and computes the effort at the actual supply voltage:

    .. math::

        d_0 &= \\frac{1}{V_{\\text{in}}} \\Big(\\frac{R}{k_t}\\, \\tau_{\\text{req}} + k_t\\, \\dot q\\Big) \\\\
        d_1 &= \\operatorname{clamp}\\Big(d_0,\\;
               \\frac{k_t\\, \\dot q - R\\, I_{\\max}}{V},\\;
               \\frac{k_t\\, \\dot q + R\\, I_{\\max}}{V}\\Big) \\\\
        d &= \\operatorname{clamp}(d_1,\\; -d_{\\max},\\; d_{\\max}) \\\\
        I &= \\frac{V\\, d - k_t\\, \\dot q}{R} \\\\
        \\tau &= k_t\\, I

    where :math:`d` is the duty cycle, :math:`I` the motor current [A] and
    :math:`\\tau` the effort [N·m or N]. The symbols map to the parameters as
    follows:

    * :math:`k_t`: ``kt``;
    * :math:`R`: ``resistance``;
    * :math:`V_{\\text{in}}`: ``vin``;
    * :math:`V`: ``vin``, or the battery voltage with ``supply_from_battery``;
    * :math:`I_{\\max}`: ``max_current``; :math:`d_1 = d_0` when it is 0;
    * :math:`d_{\\max}`: ``max_pwm``.

    At the supply voltage ``vin`` and within the limits, the motor produces
    the requested effort. The current limit narrows the duty cycle so that
    :math:`|I| \\le I_{\\max}`. The ``max_pwm`` clamp is applied last, so at
    high velocity the back-EMF term can still drive :math:`|I|` above
    ``max_current``, and the motor brakes harder the faster it overspeeds.

    With ``supply_from_battery=True``, the motor reads its supply voltage
    from the battery and reports the current :math:`d\\, I` it draws, since
    the battery supplies the motor current only during the PWM on-time (see
    :class:`~newton.actuators.BatteryBase`).

    .. hint::

        A BAM servo with firmware gain :math:`k_{p,\\text{fw}}` and error
        gain :math:`g` computes the duty cycle :math:`d_0 = g\\, k_{p,\\text{fw}}
        (q_{\\text{target}} - q)`. This motor reproduces it exactly behind a
        :class:`~newton.actuators.DrivePD` with

        .. math::

            k_p = \\frac{k_t\\, V_{\\text{in}}\\, g\\, k_{p,\\text{fw}}}{R}, \\qquad
            k_d = \\frac{k_t^2}{R}

        where :math:`k_d` cancels the back-EMF term of :math:`d_0`. The rest
        of a BAM fit is modeled outside the motor: its armature on the joint
        and its command delay with an
        :class:`~newton.actuators.InputProcessorDelay`. Its friction is not
        modeled.

    .. note::

        This motor models the back-EMF itself, so do not combine it with a
        :class:`~newton.actuators.ClampingDCMotor`, which would apply the
        effort-speed limit a second time.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        missing = {"kt", "resistance", "vin"} - set(args)
        if missing:
            raise ValueError(f"MotorDC is missing parameter(s): {', '.join(sorted(missing))}")
        resolved = {
            "kt": args["kt"],
            "resistance": args["resistance"],
            "vin": args["vin"],
            "max_pwm": args.get("max_pwm", 1.0),
            "max_current": args.get("max_current", 0.0),
            "supply_from_battery": float(bool(args.get("supply_from_battery", False))),
        }
        for name in ("kt", "resistance", "vin"):
            if resolved[name] <= 0.0:
                raise ValueError(f"{name} must be positive, got {resolved[name]}")
        if not 0.0 <= resolved["max_pwm"] <= 1.0:
            raise ValueError(f"max_pwm must be in [0, 1], got {resolved['max_pwm']}")
        if resolved["max_current"] < 0.0:
            raise ValueError(f"max_current must be non-negative, got {resolved['max_current']}")
        return resolved

    def __init__(
        self,
        kt: wp.array[float],
        resistance: wp.array[float],
        vin: wp.array[float],
        max_pwm: wp.array[float],
        max_current: wp.array[float] | None = None,
        supply_from_battery: wp.array[float] | None = None,
    ):
        """Initialize the DC motor.

        Args:
            kt: Torque constant [N·m/A or N/A]. Shape ``(N,)``.
            resistance: Motor resistance [Ω]. Shape ``(N,)``.
            vin: Supply voltage [V] at which the requested effort is
                converted to a duty cycle, and the supply voltage without a
                battery. Shape ``(N,)``.
            max_pwm: Largest duty cycle magnitude. Shape ``(N,)``.
            max_current: Firmware current limit [A]; 0 disables it. Shape
                ``(N,)``. ``None`` disables it.
            supply_from_battery: Per actuator, nonzero to read the supply
                voltage from the battery instead of *vin*. Shape ``(N,)``.
                ``None`` uses *vin* everywhere.

        Raises:
            ValueError: An array shape does not match *kt*.
        """
        n = kt.shape
        self.kt = kt
        """Torque constant [N·m/A or N/A], shape (N,)."""
        self.resistance = resistance
        """Motor resistance [Ω], shape (N,)."""
        self.vin = vin
        """Supply voltage [V] of the duty cycle conversion, and without a battery, shape (N,)."""
        self.max_pwm = max_pwm
        """Largest duty cycle magnitude, shape (N,)."""
        self.max_current = max_current if max_current is not None else wp.zeros(n, dtype=float, device=kt.device)
        """Firmware current limit [A], 0 when disabled, shape (N,)."""
        self.supply_from_battery = (
            supply_from_battery if supply_from_battery is not None else wp.zeros(n, dtype=float, device=kt.device)
        )
        """Per actuator, nonzero when the supply voltage is read from the battery, shape (N,)."""
        for name in ("resistance", "vin", "max_pwm", "max_current", "supply_from_battery"):
            shape = getattr(self, name).shape
            if shape != n:
                raise ValueError(f"{name} shape {shape} must match kt shape {n}")
        self.uses_battery = bool(self.supply_from_battery.numpy().any())

    def compute(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        velocities: wp.array[float],
        vel_indices: wp.array[wp.uint32],
        actuator_indices: wp.array[wp.uint32] | None,
        supply_voltage: wp.array[float] | None,
        supply_current: wp.array[float] | None,
        device: wp.Device | None = None,
    ) -> None:
        """Compute the motor effort.

        Raises:
            ValueError: An actuator has :attr:`supply_from_battery` set and
                *actuator_indices* or *supply_voltage* is ``None``.

        See :meth:`MotorBase.compute` for the arguments.
        """
        if self.uses_battery and (actuator_indices is None or supply_voltage is None):
            raise ValueError("MotorDC with supply_from_battery requires actuator_indices and the supply voltage.")
        wp.launch(
            kernel=_motor_dc_kernel,
            dim=len(src_forces),
            inputs=[
                velocities,
                vel_indices,
                actuator_indices,
                self.kt,
                self.resistance,
                self.vin,
                self.max_pwm,
                self.max_current,
                self.supply_from_battery,
                supply_voltage,
                src_forces,
            ],
            outputs=[dst_forces, supply_current],
            device=device,
        )
