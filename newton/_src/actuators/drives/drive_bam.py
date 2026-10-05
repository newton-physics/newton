# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp

from .base import DriveBase

if TYPE_CHECKING:
    from ..battery import Battery


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
    dof_indices: wp.array[wp.uint32],
    dof_battery: wp.array[wp.int32],
    battery_voltage: wp.array[float],
    battery_torque: wp.array[float],
    efforts: wp.array[float],
):
    i = wp.tid()
    vel = velocities[vel_indices[i]]

    supply = vin[i]
    b = int(-1)
    if dof_battery:
        b = dof_battery[dof_indices[i]]
    if b >= 0:
        supply = battery_voltage[b]

    duty = (target_pos[target_pos_indices[i]] - positions[pos_indices[i]]) * kp[i]
    if max_current[i] > 0.0 and supply > 0.0:
        center = kt[i] * vel / supply
        span = resistance[i] * max_current[i] / supply
        duty = wp.clamp(duty, center - span, center + span)
    duty = wp.clamp(duty, -max_pwm[i], max_pwm[i])

    current = (supply * duty - kt[i] * vel) / resistance[i]
    torque = kt[i] * current
    efforts[i] = torque
    if b >= 0:
        wp.atomic_add(battery_torque, b, wp.abs(torque))


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

    The supply voltage is ``vin``. When a :class:`~newton.actuators.Battery`
    is passed to :meth:`Actuator.step <newton.actuators.Actuator.step>` and
    the DOF has a battery, the drive reads the battery's voltage instead and
    adds :math:`|\\tau|` to the battery's drawn torque. This is the torque
    before any clamping, as in the BAM reference.

    The torque is not limited beyond the duty cycle and current limits; add
    a :class:`~newton.actuators.ClampingMaxEffort` for a stall-torque bound.
    BAM's gearbox friction needs the solver and is not part of this drive.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        missing = {"kp", "kt", "resistance", "vin", "max_pwm"} - set(args)
        if missing:
            raise ValueError(f"DriveBAM is missing parameter(s): {', '.join(sorted(missing))}")
        resolved = {
            "kp": args["kp"],
            "kt": args["kt"],
            "resistance": args["resistance"],
            "vin": args["vin"],
            "max_pwm": args["max_pwm"],
            "max_current": args.get("max_current", 0.0),
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
    ):
        """Initialize the BAM drive.

        Args:
            kp: Duty cycle per position error [1/rad or 1/m]. Shape ``(N,)``.
            kt: Torque constant [N·m/A]. Shape ``(N,)``.
            resistance: Motor resistance [Ω]. Shape ``(N,)``.
            vin: Supply voltage without a battery [V]. Shape ``(N,)``.
            max_pwm: Largest duty cycle magnitude. Shape ``(N,)``.
            max_current: Firmware current limit [A]; 0 disables it. Shape
                ``(N,)``. ``None`` disables it.
        """
        n = kp.shape
        self.kp = kp
        """Duty cycle per position error [1/rad or 1/m], shape (N,)."""
        self.kt = kt
        """Torque constant [N·m/A], shape (N,)."""
        self.resistance = resistance
        """Motor resistance [Ω], shape (N,)."""
        self.vin = vin
        """Supply voltage without a battery [V], shape (N,)."""
        self.max_pwm = max_pwm
        """Largest duty cycle magnitude, shape (N,)."""
        self.max_current = max_current if max_current is not None else wp.zeros(n, dtype=float, device=kp.device)
        """Firmware current limit [A], 0 when disabled, shape (N,)."""
        for name in ("kt", "resistance", "vin", "max_pwm", "max_current"):
            shape = getattr(self, name).shape
            if shape != n:
                raise ValueError(f"{name} shape {shape} must match kp shape {n}")

    def is_stateful(self) -> bool:
        return False

    def is_graphable(self) -> bool:
        return True

    def uses_battery(self) -> bool:
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
        battery: Battery | None = None,
        dof_indices: wp.array[wp.uint32] | None = None,
    ) -> None:
        """Compute the servo torque.

        Args:
            battery: Battery to read the supply voltage from and add the
                drawn torque to, or ``None`` to use :attr:`vin`.
            dof_indices: DOF index of each actuator slot into
                ``joint_qd``-shaped arrays, used to look up
                ``battery.dof_battery``. Required with *battery*.

        See :meth:`DriveBase.compute` for the other arguments.
        """
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
                dof_indices if battery is not None else None,
                battery.dof_battery if battery is not None else None,
                battery.voltage if battery is not None else None,
                battery.torque if battery is not None else None,
            ],
            outputs=[forces],
            device=device,
        )
