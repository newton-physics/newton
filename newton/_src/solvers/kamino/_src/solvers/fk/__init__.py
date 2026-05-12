# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a solver for forward kinematics, i.e. computing body poses given
joint coordinates  and base pose, by solving the kinematic constraints with
a Gauss-Newton method. This is used as a building block in the main Kamino
solver, but can also be used standalone (e.g., for visualization purposes).
"""

from .solver import ForwardKinematicsSolver

###
# Module interface
###

__all__ = ["ForwardKinematicsSolver"]
