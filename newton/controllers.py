# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated, vectorized control laws.

This module provides standalone controllers that compute signals
for the robot to track. Each controller is a concrete
subclass of :class:`ControllerBase`.

.. experimental::
"""

from ._src.controllers import (
    ControllerAdmittance,
    ControllerBase,
    ControllerDifferentialIK,
    ControllerDifferentialIKModelFree,
    ControllerJointImpedance,
    ControllerJointImpedanceModelFree,
    ControllerOperationalSpace,
    ControllerOperationalSpaceModelFree,
    DifferentialIKMethod,
)

__all__ = [
    "ControllerAdmittance",
    "ControllerBase",
    "ControllerDifferentialIK",
    "ControllerDifferentialIKModelFree",
    "ControllerJointImpedance",
    "ControllerJointImpedanceModelFree",
    "ControllerOperationalSpace",
    "ControllerOperationalSpaceModelFree",
    "DifferentialIKMethod",
]
