# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import style3d
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
    "style3d",
]

# Maps each public symbol to the module that provides it and the attribute to
# fetch from that module (None returns the module itself). Symbols are
# resolved on first attribute access (PEP 562) so that importing Newton does
# not pay the import cost of every solver backend.
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "SolverBase": (".solver", "SolverBase"),
    "SolverFeatherstone": (".featherstone", "SolverFeatherstone"),
    "SolverImplicitMPM": (".implicit_mpm", "SolverImplicitMPM"),
    "SolverKamino": (".kamino", "SolverKamino"),
    "SolverMuJoCo": (".mujoco", "SolverMuJoCo"),
    "SolverNotifyFlags": (".flags", "SolverNotifyFlags"),
    "SolverSemiImplicit": (".semi_implicit", "SolverSemiImplicit"),
    "SolverStyle3D": (".style3d.solver_style3d", "SolverStyle3D"),
    "SolverVBD": (".vbd", "SolverVBD"),
    "SolverXPBD": (".xpbd", "SolverXPBD"),
    "style3d": (".style3d", None),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = importlib.import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
