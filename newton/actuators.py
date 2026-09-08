# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated actuator models for physics simulations.

This module provides a modular library of actuator components — drives,
clamping, and delay — that compute joint effort from simulation state and
control targets. Components are composed into an :class:`Actuator` instance
and registered with :meth:`~newton.ModelBuilder.add_actuator` during model
construction.

.. experimental::

    The actuator API may change without prior notice. Feedback is welcome —
    please file issues or discussion threads.
"""

import warnings
from typing import TYPE_CHECKING

from ._src.actuators import (
    Actuator,
    ActuatorParsed,
    ClampingBase,
    ClampingDCMotor,
    ClampingMaxEffort,
    ClampingPositionBased,
    ComponentKind,
    Delay,
    DriveBase,
    DriveNeuralLSTM,
    DriveNeuralMLP,
    DrivePD,
    DrivePID,
    ResponseOracle,
    SchemaNames,
    parse_actuator_prim,
    register_actuator_component,
)

__all__ = [
    "Actuator",
    "ActuatorParsed",
    "ClampingBase",
    "ClampingDCMotor",
    "ClampingMaxEffort",
    "ClampingPositionBased",
    "ComponentKind",
    "Delay",
    "DriveBase",
    "DriveNeuralLSTM",
    "DriveNeuralMLP",
    "DrivePD",
    "DrivePID",
    "ResponseOracle",
    "SchemaNames",
    "parse_actuator_prim",
    "register_actuator_component",
]

if TYPE_CHECKING:
    Clamping = ClampingBase
    Controller = DriveBase
    ControllerNeuralLSTM = DriveNeuralLSTM
    ControllerNeuralMLP = DriveNeuralMLP
    ControllerPD = DrivePD
    ControllerPID = DrivePID

_DEPRECATED_SYMBOLS = {
    "Clamping": ClampingBase,
    "Controller": DriveBase,
    "ControllerNeuralLSTM": DriveNeuralLSTM,
    "ControllerNeuralMLP": DriveNeuralMLP,
    "ControllerPD": DrivePD,
    "ControllerPID": DrivePID,
}

__deprecated_symbols__ = {
    "Clamping": "Deprecated in 1.6; use ClampingBase instead.",
    "Controller": "Deprecated in 1.6; use DriveBase instead.",
    "ControllerNeuralLSTM": "Deprecated in 1.6; use DriveNeuralLSTM instead.",
    "ControllerNeuralMLP": "Deprecated in 1.6; use DriveNeuralMLP instead.",
    "ControllerPD": "Deprecated in 1.6; use DrivePD instead.",
    "ControllerPID": "Deprecated in 1.6; use DrivePID instead.",
}


def __getattr__(name: str):
    try:
        value = _DEPRECATED_SYMBOLS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    replacement = value.__name__
    warnings.warn(
        f"newton.actuators.{name} is deprecated in Newton 1.6; use newton.actuators.{replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DEPRECATED_SYMBOLS))
