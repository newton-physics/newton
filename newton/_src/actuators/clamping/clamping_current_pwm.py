# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp

from .base import ClampingBase


@wp.kernel
def _current_pwm_clamp_kernel(
    velocities: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    supply_voltage: wp.array[float],
    max_current: wp.array[float],
    max_pwm: wp.array[float],
    kt: wp.array[float],
    resistance: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
):
    i = wp.tid()
    voltage = src[i]
    if max_current[i] > 0.0:
        back_emf = kt[i] * velocities[vel_indices[i]]
        span = resistance[i] * max_current[i]
        voltage = wp.clamp(voltage, back_emf - span, back_emf + span)
    limit = max_pwm[i] * wp.max(supply_voltage[i], 0.0)
    dst[i] = wp.clamp(voltage, -limit, limit)


class ClampingCurrentPWM(ClampingBase):
    """Current and PWM limits of the firmware of a servo, as in BAM (Better Actuator Models).

    Bounds the voltage command :math:`u_0` of a motor, e.g. from a
    :class:`~newton.actuators.DriveVoltageControlled`, first by the current
    limit and then by the PWM limit:

    .. math::

        u_1 &= \\operatorname{clamp}\\big(u_0,\\;
               k_t\\, \\dot q - R\\, I_{\\max},\\;
               k_t\\, \\dot q + R\\, I_{\\max}\\big) \\\\
        u &= \\operatorname{clamp}(u_1,\\; -d_{\\max} V,\\; d_{\\max} V)

    where :math:`V` is the supply voltage. The symbols map to the parameters
    as follows:

    * :math:`I_{\\max}`: ``max_current``; :math:`u_1 = u_0` when it is 0;
    * :math:`d_{\\max}`: ``max_pwm``;
    * :math:`k_t`: ``kt`` and :math:`R`: ``resistance``, taken from the
      :class:`~newton.actuators.MotorDC` of the actuator (see
      :attr:`~newton.actuators.ClampingBase.MOTOR_PARAMS`).

    The current limit keeps the motor current :math:`(u - k_t \\dot q) / R`
    of the :class:`~newton.actuators.MotorDC` within :math:`\\pm I_{\\max}`.
    The PWM limit is applied last, so at high velocity the back-EMF can
    still drive the current above ``max_current``, and the motor brakes
    harder the faster it overspeeds.
    """

    MOTOR_PARAMS: ClassVar[set[str]] = {"kt", "resistance"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        resolved = {"max_current": args.get("max_current", 0.0), "max_pwm": args.get("max_pwm", 1.0)}
        if resolved["max_current"] < 0.0:
            raise ValueError(f"max_current must be non-negative, got {resolved['max_current']}")
        if not 0.0 <= resolved["max_pwm"] <= 1.0:
            raise ValueError(f"max_pwm must be in [0, 1], got {resolved['max_pwm']}")
        return resolved

    def __init__(
        self,
        max_current: wp.array[float],
        max_pwm: wp.array[float],
        kt: wp.array[float],
        resistance: wp.array[float],
    ):
        """Initialize the clamp.

        Args:
            max_current: Firmware current limit [A]; 0 disables it. Shape ``(N,)``.
            max_pwm: Largest duty cycle magnitude. Shape ``(N,)``.
            kt: Torque constant of the motor [N·m/A or N/A]. Shape ``(N,)``.
            resistance: Resistance of the motor [Ω]. Shape ``(N,)``.

        Raises:
            ValueError: An array shape does not match *max_current*.
        """
        n = max_current.shape
        self.max_current = max_current
        """Firmware current limit [A], 0 when disabled, shape (N,)."""
        self.max_pwm = max_pwm
        """Largest duty cycle magnitude, shape (N,)."""
        self.kt = kt
        """Torque constant of the motor [N·m/A or N/A], shape (N,)."""
        self.resistance = resistance
        """Resistance of the motor [Ω], shape (N,)."""
        for name in ("max_pwm", "kt", "resistance"):
            shape = getattr(self, name).shape
            if shape != n:
                raise ValueError(f"{name} shape {shape} must match max_current shape {n}")

    def modify_forces(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        device: wp.Device | None = None,
        *,
        supply_voltage: wp.array[float] | None = None,
    ) -> None:
        """Bound the voltage command by the current and PWM limits.

        Raises:
            ValueError: *supply_voltage* is ``None``, i.e. the actuator has
                no motor.

        See :meth:`ClampingBase.modify_forces` for the arguments.
        """
        if supply_voltage is None:
            raise ValueError("ClampingCurrentPWM requires a motor, such as MotorDC, that provides the supply voltage.")
        wp.launch(
            kernel=_current_pwm_clamp_kernel,
            dim=len(src_forces),
            inputs=[
                velocities,
                vel_indices,
                supply_voltage,
                self.max_current,
                self.max_pwm,
                self.kt,
                self.resistance,
                src_forces,
            ],
            outputs=[dst_forces],
            device=device,
        )
