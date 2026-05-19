# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines data types used by the Forward Kinematics solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import warp as wp

###
# Module interface
###

__all__ = ["FKJointDoFType", "ForwardKinematicsPreconditionerType", "ForwardKinematicsStatus"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class FKJointDoFType(IntEnum):
    """
    Joint dof types for the FK solver, which currently differs from Kamino's main joint types
    by the addition of the axis joint, used to regularize tie rods between two spherical joints
    (taking out the rotation dof about their own axis).

    Importantly, the integer value is the same for all joints not specific to FK, allowing seamless
    conversions.
    """

    FREE = 0
    REVOLUTE = 1
    PRISMATIC = 2
    CYLINDRICAL = 3
    UNIVERSAL = 4
    SPHERICAL = 5
    GIMBAL = 6
    CARTESIAN = 7
    FIXED = 8
    AXIS = 9


class ForwardKinematicsPreconditionerType(IntEnum):
    """Conjugate gradient preconditioning options of the FK solver, if sparsity is enabled."""

    NONE = 0
    """No preconditioning"""

    JACOBI_DIAGONAL = 1
    """Diagonal Jacobi preconditioner"""

    JACOBI_BLOCK_DIAGONAL = 2
    """Blockwise-diagonal Jacobi preconditioner, alternating blocks of size 3 and 4 along the diagonal,
    corresponding to the position and orientation (quaternion) of individual rigid bodies."""

    @classmethod
    def from_string(cls, s: str) -> ForwardKinematicsPreconditionerType:
        """Converts a string to a ForwardKinematicsPreconditionerType enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(
                f"Invalid ForwardKinematicsPreconditionerType: {s}. Valid options are: {[e.name for e in cls]}"
            ) from e


@dataclass
class ForwardKinematicsStatus:
    """
    Container holding detailed information on the success/failure status of a forward kinematics solve.
    """

    success: np.ndarray(dtype=np.int32)
    """
    Solver success flag per world, as an integer array (0 = failure, 1 = success).\n
    Shape `(num_worlds,)` and type :class:`np.int32`.

    Note that in some cases the solver may fail to converge within the maximum number
    of iterations, but still produce a solution with a reasonable residual.
    In such cases, the success flag will be set to 0, but the `max_residual` field
    can be inspected to check the actual residuals and determine if the solution is acceptable
    for the intended application.
    """

    iterations: np.ndarray(dtype=np.int32)
    """
    Number of Gauss-Newton iterations executed per world.\n
    Shape `(num_worlds,)` and type :class:`np.int32`.
    """

    max_residual: np.ndarray(dtype=np.float32)
    """
    Maximal absolute residual at the final solution, per world. In the general case, the residual vector
    is the kinematic constraints vector; if regularization is enabled, it is the penalty gradient.

    Shape `(num_worlds,)` and type :class:`np.float32`.
    """
