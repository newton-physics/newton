# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated actuator models for physics simulations.

This module provides a library of actuator controllers that compute joint
forces from simulation state and control targets.  Each actuator is
registered with :meth:`~newton.ModelBuilder.add_actuator` during model
construction and executed via :meth:`~newton.actuators.Actuator.step`
during simulation.
"""

from ._src.actuators import (
    Actuator,
    ActuatorDCMotor,
    ActuatorDelayedPD,
    ActuatorNetLSTM,
    ActuatorNetMLP,
    ActuatorPD,
    ActuatorPID,
    ActuatorRemotizedPD,
    ParsedActuator,
    parse_actuator_prim,
)

__all__ = [
    "Actuator",
    "ActuatorDCMotor",
    "ActuatorDelayedPD",
    "ActuatorNetLSTM",
    "ActuatorNetMLP",
    "ActuatorPD",
    "ActuatorPID",
    "ActuatorRemotizedPD",
    "ParsedActuator",
    "parse_actuator_prim",
]
