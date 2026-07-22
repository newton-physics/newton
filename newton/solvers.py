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

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING

from ._src.solvers import coupled as _coupled

if TYPE_CHECKING:
    from ._src.solvers import (
        SolverBase,
        SolverFeatherstone,
        SolverImplicitMPM,
        SolverKamino,
        SolverMuJoCo,
        SolverNotifyFlags,
        SolverSemiImplicit,
        SolverStyle3D,
        SolverVBD,
        SolverXPBD,
        style3d,
    )

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

# Maps each public symbol to the module that provides it and the attribute to
# fetch from that module (None returns the module itself). Symbols are
# resolved on first attribute access (PEP 562) so that ``import newton`` does
# not pay the import cost of every solver backend.
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "SolverBase": ("._src.solvers", "SolverBase"),
    "SolverFeatherstone": ("._src.solvers", "SolverFeatherstone"),
    "SolverImplicitMPM": ("._src.solvers", "SolverImplicitMPM"),
    "SolverKamino": ("._src.solvers", "SolverKamino"),
    "SolverMuJoCo": ("._src.solvers", "SolverMuJoCo"),
    "SolverNotifyFlags": ("._src.solvers", "SolverNotifyFlags"),
    "SolverSemiImplicit": ("._src.solvers", "SolverSemiImplicit"),
    "SolverStyle3D": ("._src.solvers", "SolverStyle3D"),
    "SolverVBD": ("._src.solvers", "SolverVBD"),
    "SolverXPBD": ("._src.solvers", "SolverXPBD"),
    "style3d": ("._src.solvers.style3d", None),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = importlib.import_module(module_name, __package__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


experimental = ModuleType(f"{__name__}.experimental")
experimental.__doc__ = """Experimental solver namespaces.

.. experimental::
"""
experimental.__all__ = ["coupled"]
experimental.__path__ = []
experimental.coupled = _coupled

sys.modules[f"{__name__}.experimental"] = experimental
sys.modules[f"{__name__}.experimental.coupled"] = _coupled
