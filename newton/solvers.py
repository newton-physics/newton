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

from ._src import solvers as _solvers

if TYPE_CHECKING:
    from ._src.solvers import *  # noqa: F403

__all__ = [*_solvers.__all__, "experimental"]  # noqa: PLE0604


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(_solvers, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


class _LazyCoupledModule(ModuleType):
    def _load(self) -> ModuleType:
        module = importlib.import_module("._src.solvers.coupled", __package__)
        experimental.coupled = module
        sys.modules[self.__name__] = module
        return module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __dir__(self) -> list[str]:
        return dir(self._load())


experimental = ModuleType(f"{__name__}.experimental")
experimental.__doc__ = """Experimental solver namespaces.

.. experimental::
"""
experimental.__all__ = ["coupled"]
experimental.__path__ = []
experimental.coupled = _LazyCoupledModule(f"{__name__}.experimental.coupled")

sys.modules[f"{__name__}.experimental"] = experimental
sys.modules[f"{__name__}.experimental.coupled"] = experimental.coupled
