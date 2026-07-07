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
import sys
from types import ModuleType

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
from ._src.solvers import coupled as _coupled

# solver flags
from ._src.solvers.flags import SolverNotifyFlags

experimental = ModuleType(f"{__name__}.experimental")
experimental.__doc__ = """Experimental solver namespaces.

.. experimental::
"""
experimental.__all__ = ["coupled"]
experimental.__path__ = []
experimental.coupled = _coupled

sys.modules[f"{__name__}.experimental"] = experimental
sys.modules[f"{__name__}.experimental.coupled"] = _coupled

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
    "experimental",
    "style3d",
]
