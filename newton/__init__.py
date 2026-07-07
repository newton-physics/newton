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

use_coord_layout_targets: bool = False
"""Use :attr:`joint_q`-aligned layout for joint position targets.

When ``False`` (the default in 1.3), :class:`~newton.Model` and
:class:`~newton.Control` expose :attr:`joint_target_pos` and
:attr:`joint_target_vel`, both shaped ``(joint_dof_count,)``. Accessing these
attributes emits a :class:`DeprecationWarning` since the position-target layout
is misaligned with :attr:`~newton.State.joint_q` whenever an articulation
contains a free or ball joint upstream of a position-controlled DOF.

When ``True``, :class:`~newton.Model` and :class:`~newton.Control` instead
expose:

- :attr:`joint_target_q` with shape ``(joint_coord_count,)``, matching
  :attr:`~newton.State.joint_q`.
- :attr:`joint_target_qd` with shape ``(joint_dof_count,)``, matching
  :attr:`~newton.State.joint_qd` (same layout as the legacy
  :attr:`joint_target_vel`).

Solvers, the actuator library, importers, and viewers honor this flag and read
whichever attributes are active. Toggle the flag before constructing a
:class:`~newton.ModelBuilder`; a subsequent release will flip the default to
``True``, then remove the flag and the legacy attributes.
"""

__all__ = [
    "MAXVAL",
    "Axis",
    "AxisType",
    "__version__",
    "use_coord_layout_targets",
]

# ==================================================================================
# geometry
# ==================================================================================
from ._src.geometry import (  # noqa: E402
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
    Contacts,
    Control,
    EqType,
    InverseDynamics,
    JointTargetMode,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    StateFlags,
    eval_fk,
    eval_ik,
    eval_inverse_dynamics,
    eval_inverse_dynamics_force,
    eval_jacobian,
    eval_mass_matrix,
)

__all__ += [
    "BodyFlags",
    "CollisionPipeline",
    "Contacts",
    "Control",
    "EqType",
    "InverseDynamics",
    "JointTargetMode",
    "JointType",
    "Model",
    "ModelBuilder",
    "ModelFlags",
    "State",
    "StateFlags",
    "eval_fk",
    "eval_ik",
    "eval_inverse_dynamics",
    "eval_inverse_dynamics_force",
    "eval_jacobian",
    "eval_mass_matrix",
]

# ==================================================================================
# submodule APIs
# ==================================================================================
from . import actuators, geometry, ik, math, selection, sensors, solvers, usd, utils, viewer  # noqa: E402

__all__ += [
    "actuators",
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
