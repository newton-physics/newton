# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp

if TYPE_CHECKING:
    from ..sim.builder import ModelBuilder


class ActuatorBase:
    """Base class for actuators.

    An actuator reads simulation state and control arrays and **accumulates**
    effort into an output array (scatter-add) in :meth:`step`. The caller must
    zero the output array before stepping actuators.

    :class:`~newton.actuators.Actuator` builds the effort from components:
    input processors, a drive, clamping, a motor and a transmission.
    """

    num_actuators: int
    """Number of actuators held by this instance."""

    device: wp.Device
    """Warp device of the actuator arrays."""

    requires_grad: bool
    """Whether intermediate arrays support gradients."""

    def is_stateful(self) -> bool:
        """Return True if the actuator maintains internal state."""
        raise NotImplementedError(f"{type(self).__name__} must implement is_stateful")

    def is_graphable(self) -> bool:
        """Return True if :meth:`step` can be captured in a CUDA graph."""
        raise NotImplementedError(f"{type(self).__name__} must implement is_graphable")

    def state(self) -> Any:
        """Return a new actuator state, or ``None`` if the actuator is stateless."""
        raise NotImplementedError(f"{type(self).__name__} must implement state")

    def sim_state(self) -> Any:
        """Return an empty container with the fields this actuator reads from ``sim_state``."""
        raise NotImplementedError(f"{type(self).__name__} must implement sim_state")

    def sim_control(self) -> Any:
        """Return an empty container with the fields this actuator reads from ``sim_control``."""
        raise NotImplementedError(f"{type(self).__name__} must implement sim_control")

    def register_custom_attributes(self, builder: ModelBuilder) -> None:
        """Register the custom Control attributes this actuator reads or writes.

        Args:
            builder: Newton model builder on which to register the attributes.
        """

    def step(
        self,
        sim_state: Any,
        sim_control: Any,
        current_act_state: Any = None,
        next_act_state: Any = None,
        dt: float | None = None,
    ) -> None:
        """Accumulate the actuator effort into the output array of *sim_control*.

        Args:
            sim_state: Object or mapping carrying the state arrays.
            sim_control: Object or mapping carrying the control and output arrays.
            current_act_state: Current actuator state (``None`` if stateless).
            next_act_state: Next actuator state to write (``None`` if stateless).
            dt: Timestep [s].
        """
        raise NotImplementedError(f"{type(self).__name__} must implement step")
