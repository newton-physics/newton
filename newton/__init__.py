# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib as _importlib
import typing as _typing

# ==================================================================================
# core
# ==================================================================================
from ._src.core import (
    MAXVAL,
    Axis,
    AxisType,
)
from ._src.exceptions import NewtonDeprecationWarning, NewtonWarning
from ._version import __version__

__all__ = [
    "MAXVAL",
    "Axis",
    "AxisType",
    "NewtonDeprecationWarning",
    "NewtonWarning",
    "__version__",
]

# ==================================================================================
# geometry
# ==================================================================================
from ._src.geometry import (
    SDF,
    Gaussian,
    GeoType,
    Heightfield,
    Mesh,
    ParticleFlags,
    ShapeFlags,
    TetMesh,
    intersect_ray,
)

__all__ += [
    "SDF",
    "Gaussian",
    "GeoType",
    "Heightfield",
    "Mesh",
    "ParticleFlags",
    "ShapeFlags",
    "TetMesh",
    "intersect_ray",
]

# ==================================================================================
# sim
# ==================================================================================
from ._src.sim import (  # noqa: E402
    BodyFlags,
    CollisionPipeline,
    CollisionSubstepScheduler,
    Contacts,
    Control,
    JointTargetMode,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    Rod,
    State,
    StateFlags,
    TendonGuide,
    TendonGuideFlags,
    TendonGuideType,
    eval_fk,
    eval_ik,
    eval_inverse_dynamics_force,
    eval_inverse_dynamics_passive,
    eval_jacobian,
    eval_mass_matrix,
    eval_mimic,
    eval_rigid_contact_kinematics,
)

__all__ += [
    "BodyFlags",
    "CollisionPipeline",
    "CollisionSubstepScheduler",
    "Contacts",
    "Control",
    "JointTargetMode",
    "JointType",
    "Model",
    "ModelBuilder",
    "ModelFlags",
    "Rod",
    "State",
    "StateFlags",
    "TendonGuide",
    "TendonGuideFlags",
    "TendonGuideType",
    "eval_fk",
    "eval_ik",
    "eval_inverse_dynamics_force",
    "eval_inverse_dynamics_passive",
    "eval_jacobian",
    "eval_mass_matrix",
    "eval_mimic",
    "eval_rigid_contact_kinematics",
]

# ==================================================================================
# submodule APIs
# ==================================================================================
from . import (  # noqa: E402
    config,
    geometry,
    math,
    solvers,
    usd,
    utils,
    viewer,
)

# Imported on first access (PEP 562): each declares Warp kernels that most programs never use.
_LAZY_SUBMODULES = frozenset({"actuators", "controllers", "ik", "selection", "sensors"})

if _typing.TYPE_CHECKING:
    from . import actuators, controllers, ik, selection, sensors
else:

    def __getattr__(name: str):
        if name not in _LAZY_SUBMODULES:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return _importlib.import_module(f".{name}", __name__)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)


__all__ += [
    "actuators",
    "config",
    "controllers",
    "geometry",
    "ik",
    "math",
    "selection",
    "sensors",
    "solvers",
    "usd",
    "utils",
    "viewer",
]

from ._src.diagnostics import install_console_fallback as _install_console_fallback  # noqa: E402

_install_console_fallback()
del _install_console_fallback
