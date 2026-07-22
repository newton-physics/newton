# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .featherstone import SolverFeatherstone
    from .flags import SolverNotifyFlags
    from .implicit_mpm import SolverImplicitMPM
    from .kamino import SolverKamino
    from .mujoco import SolverMuJoCo
    from .semi_implicit import SolverSemiImplicit
    from .solver import SolverBase
    from .style3d.solver_style3d import SolverStyle3D
    from .vbd import SolverVBD
    from .xpbd import SolverXPBD

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
]

# Maps each public symbol to the submodule that defines it. Symbols are
# resolved on first attribute access (PEP 562) so that importing Newton does
# not pay the import cost of every solver backend.
_LAZY_IMPORTS = {
    "SolverBase": ".solver",
    "SolverFeatherstone": ".featherstone",
    "SolverImplicitMPM": ".implicit_mpm",
    "SolverKamino": ".kamino",
    "SolverMuJoCo": ".mujoco",
    "SolverNotifyFlags": ".flags",
    "SolverSemiImplicit": ".semi_implicit",
    "SolverStyle3D": ".style3d.solver_style3d",
    "SolverVBD": ".vbd",
    "SolverXPBD": ".xpbd",
}


def __getattr__(name: str):
    try:
        module_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
