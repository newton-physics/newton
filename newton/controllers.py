# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Composable control blocks for Newton.

A :class:`ControlLaw` is a single control law (PID, differential IK, gravity
comp, …); a :class:`Controller` composes one or more of them under a
:class:`HardwareInterface` that wires :class:`ControlSignal`s to attribute
names on a deployment's runtime input / output objects. Controllers typically
run *before* actuators in a simulation step: a controller produces a desired
joint position, velocity, or force; downstream actuators turn that target into
effort.
"""

from ._src.controllers import (
    JOINT_F,
    JOINT_Q,
    JOINT_QD,
    JOINT_TARGET_Q,
    JOINT_TARGET_QD,
    ControlLaw,
    ControlLawDifferentialIK,
    ControlLawPID,
    Controller,
    ControlSignal,
    HardwareInterface,
)

__all__ = [
    "JOINT_F",
    "JOINT_Q",
    "JOINT_QD",
    "JOINT_TARGET_Q",
    "JOINT_TARGET_QD",
    "ControlLaw",
    "ControlLawDifferentialIK",
    "ControlLawPID",
    "ControlSignal",
    "Controller",
    "HardwareInterface",
]
