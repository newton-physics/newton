# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Solvers integrate the dynamics of a :class:`~newton.Model` through the common
:class:`~newton.solvers.SolverBase` interface. Newton provides backends for
rigid articulated systems, maximal-coordinate constraints, particles, and
deformable simulation.

See ``docs/solvers/index.rst`` for solver-selection guidance and the feature,
contact-material, joint-support, and differentiability comparisons. Installed
wheel users can read the rendered guide at
https://newton-physics.github.io/newton/latest/solvers/index.html or through
the :doc:`Solvers guide </solvers/index>` in this documentation.
"""

# solver types
from ._src.solvers import (
    SolverBase,
    SolverFeatherstone,
    SolverImplicitMPM,
    SolverKamino,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverStyle3D,
    SolverVBD,
    SolverXPBD,
    style3d,
)

# solver flags
from ._src.solvers.flags import SolverNotifyFlags

__all__ = [
    "SolverBase",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverKamino",
    "SolverMuJoCo",
    "SolverNotifyFlags",
    "SolverSemiImplicit",
    "SolverStyle3D",
    "SolverVBD",
    "SolverXPBD",
    "style3d",
]
