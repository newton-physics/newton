# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Composable control blocks for Newton.

A :class:`ControlLaw` is a single control law (PID, differential IK, gravity
comp, …); a :class:`Controller` composes one or more of them and orchestrates
the per-step zero / compute sequence. Controllers typically run *before*
actuators in a simulation step: a controller produces a desired joint
position, velocity, or force; downstream actuators turn that target into effort.
"""

from ._src.controllers import (
    ControlLaw,
    ControlLawDifferentialIK,
    ControlLawPID,
    Controller,
)

__all__ = [
    "ControlLaw",
    "ControlLawDifferentialIK",
    "ControlLawPID",
    "Controller",
]
