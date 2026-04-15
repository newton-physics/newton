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
    ActuatorParsed,
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
    parse_actuator_prim,
)

__all__ = [
    "Actuator",
    "ActuatorParsed",
    "Clamping",
    "ClampingDCMotor",
    "ClampingMaxForce",
    "ClampingPositionBased",
    "Controller",
    "ControllerNetLSTM",
    "ControllerNetMLP",
    "ControllerPD",
    "ControllerPID",
    "Delay",
    "parse_actuator_prim",
]
