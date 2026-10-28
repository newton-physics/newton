# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


class MotorBase:
    """Base class for actuator motors.

    A motor is the last stage of an actuator, after the drive and the
    clamping: it turns the requested effort into the effort the motor
    produces at the joint velocity and its supply voltage. A motor that draws
    from a battery reads its supply voltage from the battery and reports the
    current it draws.

    **Validation contract:**  :meth:`resolve_arguments` validates scalar
    parameter values before they are batched into Warp arrays.
    ``__init__`` receives pre-built arrays and validates shapes only —
    reading back array contents would force a synchronous device-to-host
    copy on every construction.
    """

    SHARED_PARAMS: ClassVar[set[str]] = set()

    uses_battery: bool = False
    """Whether an actuator slot draws from a battery. Set in ``__init__``."""

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

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
        """Read the requested effort from *src*, write the motor effort to *dst*.

        When src and dst are the same array, this is an in-place update.

        Args:
            src_forces: Requested effort [N or N·m]. Shape ``(N,)``.
            dst_forces: Motor effort [N or N·m]. Shape ``(N,)``.
            velocities: Joint velocities [m/s or rad/s].
            vel_indices: Indices into *velocities* for each slot.
            actuator_indices: Index of each slot into *supply_voltage* and
                *supply_current*, or ``None`` without a battery.
            supply_voltage: Supply voltage of each actuator [V], one entry
                per actuator of the model (``actuator_supply_voltage``),
                written by :meth:`BatteryBase.step <newton.actuators.BatteryBase.step>`,
                or ``None`` without a battery.
            supply_current: Current each actuator draws from its battery [A],
                one entry per actuator of the model (``actuator_supply_current``),
                written here and read by
                :meth:`BatteryBase.step <newton.actuators.BatteryBase.step>`,
                or ``None`` to skip.
            device: Warp device for kernel launches.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute")
