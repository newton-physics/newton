# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Source for the detailed solver guide: docs/solvers/index.rst
"""
Solvers integrate the dynamics of a :class:`~newton.Model` through the common
:class:`~newton.solvers.SolverBase` interface. Newton provides backends for
rigid articulated systems, maximal-coordinate constraints, particles, and
deformable simulation.

For solver-selection guidance and the feature, contact-material, joint-support,
and differentiability comparisons, see the :doc:`Solvers guide </solvers/index>`.
Installed-wheel users can use the stable hosted guide at
https://newton-physics.github.io/newton/stable/solvers/index.html.
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
