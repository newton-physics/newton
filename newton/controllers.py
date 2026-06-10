# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Control blocks for Newton.

A :class:`Controller` is a single self-contained control law (PID,
differential IK, gravity compensation, …). There is no framework-level
composition: callers wanting to combine multiple control laws invoke each
one's :meth:`Controller.compute` in sequence themselves. Controllers
typically run *before* actuators in a simulation step — a controller
produces a desired joint position, velocity, or force; downstream
actuators turn that target into effort.
"""

from ._src.controllers import (
    Controller,
    ControllerDifferentialKinematics,
    ControllerPID,
)

__all__ = [
    "Controller",
    "ControllerDifferentialKinematics",
    "ControllerPID",
]
