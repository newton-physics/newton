# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


@wp.kernel
def _motor_dc_supply_voltage_kernel(
    vin: wp.array[float],
    supply_from_battery: wp.array[float],
    battery_voltage: wp.array[float],
    actuator_indices: wp.array[wp.uint32],
    supply_voltage: wp.array[float],
):
    i = wp.tid()
    voltage = vin[i]
    if supply_from_battery[i] != 0.0:
        voltage = battery_voltage[actuator_indices[i]]
    supply_voltage[i] = voltage


@wp.kernel
def _motor_dc_kernel(
    velocities: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    supply_voltage: wp.array[float],
    actuator_indices: wp.array[wp.uint32],
    kt: wp.array[float],
    resistance: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
    supply_current: wp.array[float],
):
    i = wp.tid()
    limit = wp.max(supply_voltage[i], 0.0)
    voltage = wp.clamp(src[i], -limit, limit)
    current = (voltage - kt[i] * velocities[vel_indices[i]]) / resistance[i]
    dst[i] = kt[i] * current
    if supply_current:
        duty = float(0.0)
        if supply_voltage[i] > 0.0:
            duty = voltage / supply_voltage[i]
        supply_current[actuator_indices[i]] = duty * current


class MotorDC:
    """DC motor driven by a PWM voltage, as in BAM (Better Actuator Models).

    Computes the effort from the commanded voltage :math:`u`, bounded by the
    supply voltage :math:`V`, since the motor cannot receive more:

    .. math::

        u &\\leftarrow \\operatorname{clamp}(u,\\; -V,\\; V) \\\\
        I &= \\frac{u - k_t\\, \\dot q}{R} \\\\
        \\tau &= k_t\\, I

    where :math:`I` is the motor current [A], :math:`\\tau` the effort
    [N·m or N], :math:`k_t` is ``kt`` and :math:`R` is ``resistance``.

    The command :math:`u` is the output of the drive, e.g. a
    :class:`~newton.actuators.DriveVoltageControlled`, bounded by the
    command clamping, e.g. a :class:`~newton.actuators.ClampingCurrentPWM`,
    which takes ``kt`` and ``resistance`` from this motor. The other clamping
    bounds the effort of the motor.

    The motor runs after the drive and the command clamping. It provides the supply voltage :math:`V` of the actuator: ``vin``,
    or the battery voltage with ``supply_from_battery=True``. It then
    reports the current :math:`d\\, I` it draws, with the duty cycle
    :math:`d = u / V`, since the battery supplies the motor current only
    during the PWM on-time (see :class:`~newton.actuators.BatteryBase`).

    .. note::

        This motor models the back-EMF itself, so do not combine it with a
        :class:`~newton.actuators.ClampingDCMotor`, which would apply the
        effort-speed limit a second time.
    """

    SHARED_PARAMS: ClassVar[set[str]] = set()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.

        Raises:
            ValueError: ``kt`` or ``resistance`` is missing or not positive,
                or ``vin`` is without ``supply_from_battery``.
        """
        supply_from_battery = bool(args.get("supply_from_battery", False))
        required = {"kt", "resistance"} if supply_from_battery else {"kt", "resistance", "vin"}
        missing = required - set(args)
        if missing:
            raise ValueError(f"MotorDC is missing parameter(s): {', '.join(sorted(missing))}")
        resolved = {
            "kt": args["kt"],
            "resistance": args["resistance"],
            "vin": args.get("vin", 0.0),
            "supply_from_battery": float(supply_from_battery),
        }
        for name in required:
            if resolved[name] <= 0.0:
                raise ValueError(f"{name} must be positive, got {resolved[name]}")
        return resolved

    def __init__(
        self,
        kt: wp.array[float],
        resistance: wp.array[float],
        vin: wp.array[float],
        supply_from_battery: wp.array[float] | None = None,
    ):
        """Initialize the DC motor.

        Args:
            kt: Torque constant [N·m/A or N/A]. Shape ``(N,)``.
            resistance: Motor resistance [Ω]. Shape ``(N,)``.
            vin: Supply voltage [V] without a battery; unused with
                ``supply_from_battery``. Shape ``(N,)``.
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
        """Supply voltage [V] without a battery, shape (N,)."""
        self.supply_from_battery = (
            supply_from_battery if supply_from_battery is not None else wp.zeros(n, dtype=float, device=kt.device)
        )
        """Per actuator, nonzero when the supply voltage is read from the battery, shape (N,)."""
        for name in ("resistance", "vin", "supply_from_battery"):
            shape = getattr(self, name).shape
            if shape != n:
                raise ValueError(f"{name} shape {shape} must match kt shape {n}")
        self.uses_battery = bool(self.supply_from_battery.numpy().any())
        """Whether an actuator slot draws from a battery."""

    def finalize(self, device: wp.Device, num_actuators: int, requires_grad: bool = False) -> None:
        """Called by :class:`~newton.actuators.Actuator` after construction to set up device-specific resources.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators (DOFs) this motor manages.
            requires_grad: Whether to allocate internal buffers with gradient support.
        """

    def is_graphable(self) -> bool:
        """Return True if :meth:`compute` can be captured in a CUDA graph."""
        return True

    def compute_supply_voltage(
        self,
        battery_voltage: wp.array[float] | None,
        actuator_indices: wp.array[wp.uint32] | None,
        supply_voltage: wp.array[float],
        device: wp.Device | None = None,
    ) -> None:
        """Write ``vin``, or the battery voltage with ``supply_from_battery``, to *supply_voltage*.

        Args:
            battery_voltage: Supply voltage of each actuator [V], one entry
                per actuator of the model (``actuator_supply_voltage``),
                written by :meth:`BatteryBase.step <newton.actuators.BatteryBase.step>`,
                or ``None`` without a battery.
            actuator_indices: Index of each slot into *battery_voltage*, or
                ``None`` without a battery.
            supply_voltage: Supply voltage of each slot [V]. Shape ``(N,)``.
            device: Warp device for kernel launches.

        Raises:
            ValueError: An actuator has :attr:`supply_from_battery` set and
                *actuator_indices* or *battery_voltage* is ``None``.
        """
        if self.uses_battery and (actuator_indices is None or battery_voltage is None):
            raise ValueError("MotorDC with supply_from_battery requires actuator_indices and the battery voltage.")
        wp.launch(
            kernel=_motor_dc_supply_voltage_kernel,
            dim=len(supply_voltage),
            inputs=[self.vin, self.supply_from_battery, battery_voltage, actuator_indices],
            outputs=[supply_voltage],
            device=device,
        )

    def compute(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        velocities: wp.array[float],
        vel_indices: wp.array[wp.uint32],
        supply_voltage: wp.array[float],
        actuator_indices: wp.array[wp.uint32] | None,
        supply_current: wp.array[float] | None,
        device: wp.Device | None = None,
    ) -> None:
        """Read the voltage command from *src*, write the motor effort to *dst*.

        When src and dst are the same array, this is an in-place update.

        Args:
            src_forces: Voltage command [V]. Shape ``(N,)``.
            dst_forces: Motor effort [N or N·m]. Shape ``(N,)``.
            velocities: Joint velocities [m/s or rad/s].
            vel_indices: Indices into *velocities* for each slot.
            supply_voltage: Supply voltage of each slot [V], written by
                :meth:`compute_supply_voltage`. Shape ``(N,)``.
            actuator_indices: Index of each slot into *supply_current*, or
                ``None`` without a battery.
            supply_current: Current each actuator draws from its battery [A],
                one entry per actuator of the model (``actuator_supply_current``),
                written here and read by
                :meth:`BatteryBase.step <newton.actuators.BatteryBase.step>`,
                or ``None`` to skip.
            device: Warp device for kernel launches.
        """
        wp.launch(
            kernel=_motor_dc_kernel,
            dim=len(src_forces),
            inputs=[
                velocities,
                vel_indices,
                supply_voltage,
                actuator_indices,
                self.kt,
                self.resistance,
                src_forces,
            ],
            outputs=[dst_forces, supply_current],
            device=device,
        )
