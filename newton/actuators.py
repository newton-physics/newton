# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated actuator models for physics simulations.

This module provides a modular library of actuator components — controllers,
clamping, and delay — that compute joint effort from simulation state and
control targets. Components are composed into an :class:`Actuator` instance
and registered with :meth:`~newton.ModelBuilder.add_actuator` during model
construction.
"""

from ._src.actuators import (
    Actuator,
    ActuatorParsed,
    Clamping,
    ClampingDCMotor,
    ClampingMaxEffort,
    ClampingPositionBased,
    ComponentKind,
    Controller,
    ControllerNeuralLSTM,
    ControllerNeuralMLP,
    ControllerPD,
    ControllerPID,
    Delay,
    parse_actuator_prim,
    register_actuator_component,
)

__all__ = [
    "Actuator",
    "ActuatorParsed",
    "Clamping",
    "ClampingDCMotor",
    "ClampingMaxEffort",
    "ClampingPositionBased",
    "ComponentKind",
    "Controller",
    "ControllerNeuralLSTM",
    "ControllerNeuralMLP",
    "ControllerPD",
    "ControllerPID",
    "Delay",
    "parse_actuator_prim",
    "register_actuator_component",
]
