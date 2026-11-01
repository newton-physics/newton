# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, ClassVar

import warp as wp

from ...utils.deprecation import warn_deprecation

_COMPUTE_FORCES_DEPRECATION_MSG = (
    "The forces argument of DriveBase.compute() is deprecated in Newton 1.8; use output instead."
)


def _rename_compute_output(compute: Any, legacy: bool) -> Any:
    """Wrap a drive's ``compute`` so ``forces=`` and ``output=`` reach the name it declares."""

    @wraps(compute)
    def wrapper(self: DriveBase, *args: Any, **kwargs: Any) -> None:
        if "forces" in kwargs:
            warn_deprecation(_COMPUTE_FORCES_DEPRECATION_MSG)
            if "output" in kwargs:
                raise TypeError("compute() got values for both 'output' and its deprecated name 'forces'")
            kwargs["output"] = kwargs.pop("forces")
        if legacy and "output" in kwargs:
            kwargs["forces"] = kwargs.pop("output")
        return compute(self, *args, **kwargs)

    return wrapper


@wp.kernel
def _masked_zero_1d(data: wp.array[float], mask: wp.array[wp.bool]):
    i = wp.tid()
    if mask[i]:
        data[i] = 0.0


class DriveBase:
    """Base class for actuator drives.

    Drives compute actuator output effort from authored drive
    parameters, commanded inputs (targets, feedforward), and simulation
    state. The output may still be constrained by one or more
    :class:`~newton.actuators.ClampingBase` objects. Only drives that
    output a motor command instead (see :attr:`OUTPUTS_MOTOR_COMMAND`) can be
    combined with a motor (e.g. :class:`~newton.actuators.MotorDC`).

    Subclasses must override ``compute`` and ``resolve_arguments``.

    **Validation contract:**  :meth:`resolve_arguments` validates scalar
    parameter values (e.g. ``kp >= 0``) before they are batched into Warp
    arrays.  ``__init__`` receives pre-built arrays and validates shapes
    only — reading back array contents for value checks would force a
    synchronous device-to-host copy on every construction.
    """

    @dataclass
    class State:
        """Base state for drives.

        Subclass this in concrete drives that maintain internal
        state (e.g. integral accumulators, history buffers).
        """

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset state to initial values.

            Args:
                mask: Boolean mask of length N. ``True`` entries are reset.
                    ``None`` resets all.
            """

    SHARED_PARAMS: ClassVar[set[str]] = set()

    OUTPUTS_MOTOR_COMMAND: ClassVar[bool] = False
    """Whether :meth:`compute` outputs a motor command (e.g. a voltage) instead of an effort.

    A drive that outputs a motor command requires a motor, such as
    :class:`~newton.actuators.MotorDC`, and receives its supply voltage; a
    drive that outputs an effort cannot be combined with one.
    """

    custom_inputs: tuple[str, ...] = ()
    """Names of extra caller-provided Control arrays this drive reads.

    Set in ``__init__`` when the declaration depends on drive configuration.
    :class:`~newton.actuators.Actuator` reads the same-named values from
    ``sim_control`` and passes them to :meth:`compute` as ``custom_inputs``.
    The actuator routes values without validating them; the drive owns
    requiredness, fallback behavior, and validation.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        compute = cls.__dict__.get("compute")
        if compute is None:
            return
        legacy = "forces" in inspect.signature(compute).parameters
        if legacy:
            warn_deprecation(
                f"{cls.__name__}.compute() names its output argument 'forces', which is deprecated "
                "in Newton 1.8; rename it to 'output'."
            )
        cls.compute = _rename_compute_output(compute, legacy)

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        """Called by :class:`Actuator` after construction to set up device-specific resources.

        Override in subclasses that need to place tensors or networks
        on a specific device, or pre-compute index tensors.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators (DOFs) this drive manages.
        """
        pass

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
        output: wp.array[float],
        state: DriveBase.State | None,
        dt: float,
        device: wp.Device | None = None,
        *,
        custom_inputs: dict[str, Any] | None = None,
        supply_voltage: wp.array[float] | None = None,
    ) -> None:
        """Compute the drive output and write it to ``output[i]``.

        Args:
            positions: Joint positions [m or rad].
            velocities: Joint velocities [m/s or rad/s].
            target_pos: Target positions [m or rad].
            target_vel: Target velocities [m/s or rad/s].
            feedforward: Feedforward effort [N or N·m] (may be ``None``).
            pos_indices: Indices into *positions* for each DOF.
            vel_indices: Indices into *velocities* for each DOF.
            target_pos_indices: Indices into *target_pos*.
            target_vel_indices: Indices into *target_vel* and *feedforward*.
            output: Buffer to write the drive output to: the effort
                [N or N·m], or the motor command for drives that output one
                (e.g. a voltage [V] for
                :class:`~newton.actuators.DriveVoltageControlled`). Shape
                ``(N,)``.
            state: Drive state (``None`` if stateless).
            dt: Timestep [s].
            device: Warp device for kernel launches.
            custom_inputs: Arrays for the names this drive lists in
                :attr:`custom_inputs`, keyed by attribute name. Values are
                passed without validation and may be ``None`` when absent.
                Empty when the drive names none.
            supply_voltage: Supply voltage of each slot [V], from the
                actuator's motor (see :meth:`MotorDC.compute_supply_voltage
                <newton.actuators.MotorDC.compute_supply_voltage>`). Shape
                ``(N,)``. Only passed when the actuator has a motor.

        .. deprecated:: 1.8
            The argument name ``forces``; use ``output``. Subclasses that
            still name it ``forces`` keep working.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute")

    evaluate_force: ClassVar[wp.Function | None] = None
    """``@wp.func`` form of this drive's force law, called inside the
    implicit solve kernel (see :meth:`Actuator.set_effort_mode_implicit`).
    ``None`` means the drive does not support implicit actuation.

    Required signature (``float64``)::

        evaluate_force(q, qd, target_q, target_qd, feedforward,
                       params: wp.array2d[float], i: int) -> wp.float64

    ``params[i]`` holds the drive's parameters for actuator slot ``i``;
    see :meth:`bind_params`.
    """

    def bind_params(self) -> wp.array2d[float] | None:
        """Build the per-actuator parameter pack and wire attributes to it.

        Called once when the implicit effort mode is installed. Override to:

        1. Pack the drive's parameters into a contiguous
           ``(num_actuators, P)`` array — row ``i`` for actuator slot ``i``,
           layout matching :attr:`evaluate_force`.
        2. Re-point the drive's parameter arrays (e.g. ``kp``) at columns
           of the pack so later writes stay visible.

        ``None`` (the default) means the drive does not support implicit
        actuation.
        """
        return None

    def prepare_implicit(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        drive_state: DriveBase.State | None,
        dt: float,
        inv_mass: wp.array[float] | None = None,
        device: wp.Device | None = None,
        *,
        custom_inputs: dict[str, Any] | None = None,
    ) -> None:
        """Refresh the parameter pack before an implicit solve step.

        Called by the implicit effort mode once per step, before the solve
        kernel. Drives whose :attr:`evaluate_force` law needs per-step
        preparation (e.g. a neural network linearized about the current
        state) override this to rewrite the pack built by :meth:`bind_params`
        in place. The default is a no-op — parameter-static laws like PD need
        nothing here.

        ``custom_inputs`` carries the same arrays as the matching argument of
        :meth:`compute`.
        """

    def is_stateful(self) -> bool:
        """Return True if this drive maintains internal state."""
        raise NotImplementedError(f"{type(self).__name__} must implement is_stateful")

    def is_graphable(self) -> bool:
        """Return True if compute() can be captured in a CUDA graph."""
        raise NotImplementedError(f"{type(self).__name__} must implement is_graphable")

    def state(self, num_actuators: int, device: wp.Device) -> DriveBase.State | None:
        """Create and return a new state object, or None if stateless."""
        return None

    def update_state(
        self,
        current_state: DriveBase.State,
        next_state: DriveBase.State,
    ) -> None:
        """Advance internal state after a compute step.

        Args:
            current_state: Current drive state.
            next_state: Next drive state to write.
        """
        pass
