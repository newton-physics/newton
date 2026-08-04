# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated, vectorized control laws for Newton physics simulations.

This module provides standalone controllers that compute joint torques or
other actuation signals from simulation state. Each controller is a concrete
subclass of :class:`ControllerBase` and operates on flat 1D arrays matching the
layout of :class:`~newton.State` fields, making them composable with any
Newton solver.

.. experimental::
"""

from ._src.controllers import (
    ControllerBase,
    ControllerJointImpedance,
    ControllerJointImpedanceModelFree,
)

__all__ = [
    "ControllerBase",
    "ControllerJointImpedance",
    "ControllerJointImpedanceModelFree",
]
