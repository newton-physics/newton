# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
import warnings as _warnings

# ==================================================================================
# core
# ==================================================================================
from ._src.core import (
    MAXVAL,
    Axis,
    AxisType,
)
from ._version import __version__

__all__ = [
    "MAXVAL",
    "Axis",
    "AxisType",
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
    CameraLens,
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
    "CameraLens",
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
    actuators,
    config,
    controllers,
    exceptions,
    geometry,
    ik,
    math,
    selection,
    sensors,
    solvers,
    usd,
    utils,
    viewer,
)

__all__ += [
    "actuators",
    "config",
    "controllers",
    "exceptions",
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

if _sys.version_info < (3, 11):
    # Show the interpreter migration notice even under Python's default filters.
    with _warnings.catch_warnings():
        if not _sys.warnoptions:
            _warnings.simplefilter("default", exceptions.NewtonDeprecationWarning)
        _warnings.warn(
            "Support for Python 3.10 is deprecated and will be removed in Newton 1.9. "
            "Please upgrade to Python 3.11 or newer.",
            exceptions.NewtonDeprecationWarning,
            stacklevel=2,
        )
