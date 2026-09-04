# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated, vectorized control laws.

This module provides standalone controllers that compute signals
for the robot to track. Each controller is a concrete
subclass of :class:`ControllerBase`.

.. experimental::
"""

from ._src.controllers import (
    ControllerBase,
    ControllerJointImpedance,
    ControllerJointImpedanceModelFree,
    ControllerOperationalSpace,
    ControllerOperationalSpaceModelFree,
)

__all__ = [
    "ControllerBase",
    "ControllerJointImpedance",
    "ControllerJointImpedanceModelFree",
    "ControllerOperationalSpace",
    "ControllerOperationalSpaceModelFree",
]
