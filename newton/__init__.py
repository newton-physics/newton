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
from . import actuators, controllers, geometry, ik, math, selection, sensors, solvers, usd, utils, viewer  # noqa: E402

__all__ += [
    "actuators",
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
