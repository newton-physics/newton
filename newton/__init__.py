# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

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
]

# ==================================================================================
# sim
# ==================================================================================
from ._src.sim import (  # noqa: E402
    BodyFlags,
    CollisionPipeline,
    Contacts,
    Control,
    EqType,
    JointTargetMode,
    JointType,
    Model,
    ModelBuilder,
    State,
    eval_fk,
    eval_ik,
    eval_jacobian,
    eval_mass_matrix,
)

__all__ += [
    "BodyFlags",
    "CollisionPipeline",
    "Contacts",
    "Control",
    "EqType",
    "JointTargetMode",
    "JointType",
    "Model",
    "ModelBuilder",
    "State",
    "eval_fk",
    "eval_ik",
    "eval_jacobian",
    "eval_mass_matrix",
]

# ==================================================================================
# submodule APIs
# ==================================================================================
from . import geometry, ik, math, selection, sensors, solvers, usd, utils, viewer  # noqa: E402

__all__ += [
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
