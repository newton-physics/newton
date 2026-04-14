# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated actuator models for physics simulations.

This module provides a modular library of actuator components — controllers,
clamping, and delay — that compute joint forces from simulation state and
control targets. Components are composed into an :class:`Actuator` instance
and registered with :meth:`~newton.ModelBuilder.add_actuator` during model
construction.
"""

from ._src.actuators import (
    Actuator,
    Clamping,
    ClampingDCMotor,
    ClampingMaxForce,
    ClampingPositionBased,
    Controller,
    ControllerNetLSTM,
    ControllerNetMLP,
    ControllerPD,
    ControllerPID,
    Delay,
    ParsedActuator,
    StateActuator,
    parse_actuator_prim,
)

__all__ = [
    # Composed actuator
    "Actuator",
    "StateActuator",
    # Controllers
    "Controller",
    "ControllerPD",
    "ControllerPID",
    "ControllerNetMLP",
    "ControllerNetLSTM",
    # Delay
    "Delay",
    # Clamping
    "Clamping",
    "ClampingMaxForce",
    "ClampingPositionBased",
    "ClampingDCMotor",
    # USD
    "ParsedActuator",
    "parse_actuator_prim",
]
