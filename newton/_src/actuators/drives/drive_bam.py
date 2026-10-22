# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp

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
    supply_voltage: wp.array[float],
    efforts: wp.array[float],
):
    i = wp.tid()
    vel = velocities[vel_indices[i]]

    supply = vin[i]
    if supply_voltage:
        supply = supply_voltage[i]

    duty = (target_pos[target_pos_indices[i]] - positions[pos_indices[i]]) * kp[i]
    if max_current[i] > 0.0 and supply > 0.0:
        center = kt[i] * vel / supply
        span = resistance[i] * max_current[i] / supply
        duty = wp.clamp(duty, center - span, center + span)
    duty = wp.clamp(duty, -max_pwm[i], max_pwm[i])

    current = (supply * duty - kt[i] * vel) / resistance[i]
    efforts[i] = kt[i] * current


class DriveBAM(DriveBase):
    """Servo drive from BAM (Better Actuator Models), without friction.

    Models the servo firmware and the DC motor of an identified BAM fit:

    .. math::

        d &= \\operatorname{clamp}\\big((q_{\\text{target}} - q)\\, k_p,\\;
             -d_{\\max},\\; d_{\\max}\\big) \\\\
        I &= \\frac{V\\, d - k_t\\, \\dot q}{R} \\\\
        \\tau &= k_t\\, I

    where :math:`d` is the PWM duty cycle and :math:`V` the supply voltage.
    For a BAM fit, :math:`k_p` is the firmware gain times the fit's error
    gain. When ``max_current`` is positive, the firmware narrows the duty cycle
    towards the current limit before the ``max_pwm`` clamp. The ``max_pwm``
    clamp is applied last, so at high velocity the back-EMF term can still
    drive :math:`|I|` above ``max_current``; this matches the BAM reference
    firmware. Target velocity and
    feedforward are ignored: the modelled firmware has no torque input.

    The supply voltage is either fixed or varying. A fixed supply, e.g. a
    bench or regulated power supply as used to identify a BAM fit, is set by
    ``vin``. A varying supply, e.g. a battery whose voltage sags under load,
    is enabled with ``supply_from_battery=True``: the drive then reads the
    voltage of each actuator from the ``actuator_supply_voltage`` input on
    ``sim_control`` (see :attr:`DriveBase.custom_inputs
    <newton.actuators.DriveBase.custom_inputs>`), which
    :func:`~newton.actuators.update_battery_supply` writes every step, and
    ignores ``vin``.

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

    SHARED_PARAMS: ClassVar[set[str]] = {"supply_from_battery"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        supply_from_battery = bool(args.get("supply_from_battery", False))
        required = {"kp", "kt", "resistance", "max_pwm"} | (set() if supply_from_battery else {"vin"})
        missing = required - set(args)
        if missing:
            raise ValueError(f"DriveBAM is missing parameter(s): {', '.join(sorted(missing))}")
        resolved = {
            "kp": args["kp"],
            "kt": args["kt"],
            "resistance": args["resistance"],
            "vin": args.get("vin", 0.0),
            "max_pwm": args["max_pwm"],
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
        supply_from_battery: bool = False,
    ):
        """Initialize the BAM drive.

        Args:
            kp: Duty cycle per position error [1/rad or 1/m]. Shape ``(N,)``.
            kt: Torque constant [N·m/A]. Shape ``(N,)``.
            resistance: Motor resistance [Ω]. Shape ``(N,)``.
            vin: Supply voltage [V]. Ignored when *supply_from_battery* is
                set; :meth:`~newton.ModelBuilder.add_actuator` then does not
                require it.
                Shape ``(N,)``.
            max_pwm: Largest duty cycle magnitude. Shape ``(N,)``.
            max_current: Firmware current limit [A]; 0 disables it. Shape
                ``(N,)``. ``None`` disables it.
            supply_from_battery: Read the supply voltage from the
                ``actuator_supply_voltage`` input, as written by
                :func:`~newton.actuators.update_battery_supply`, instead of
                *vin*. Shared by all actuators of a group.

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
        self.supply_from_battery = supply_from_battery
        """Whether the supply voltage is read from the ``actuator_supply_voltage`` input written by the battery."""
        from ..actuator import ACTUATOR_FREQUENCY  # noqa: PLC0415

        # ACTUATOR_FREQUENCY is the per-actuator frequency: one entry per actuator of the model, not per DOF.
        self.custom_inputs = {"actuator_supply_voltage": ACTUATOR_FREQUENCY} if supply_from_battery else {}
        for name in ("kt", "resistance", "vin", "max_pwm", "max_current"):
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
                one entry per slot, when :attr:`supply_from_battery` is set.

        Raises:
            ValueError: :attr:`supply_from_battery` is set and
                ``actuator_supply_voltage`` is missing.

        See :meth:`DriveBase.compute` for the other arguments.
        """
        supply_voltage = None
        if self.supply_from_battery:
            supply_voltage = (custom_inputs or {}).get("actuator_supply_voltage")
            if supply_voltage is None:
                raise ValueError(
                    "DriveBAM(supply_from_battery=True) requires 'actuator_supply_voltage' on sim_control."
                )
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
                supply_voltage,
            ],
            outputs=[forces],
            device=device,
        )
