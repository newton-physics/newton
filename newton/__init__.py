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

Controls the shape of :attr:`~newton.Model.joint_target_q` and
:attr:`~newton.Control.joint_target_q`:

- ``True``: shape ``(joint_coord_count,)``, matching
  :attr:`~newton.State.joint_q`.
- ``False`` (the default): legacy shape ``(joint_dof_count,)``, which is
  misaligned with :attr:`~newton.State.joint_q` whenever an articulation
  contains a free or ball joint upstream of a position-controlled DOF.

:attr:`joint_target_qd` is shaped ``(joint_dof_count,)`` in both layouts,
matching :attr:`~newton.State.joint_qd`.

Solvers, the actuator library, importers, and viewers honor this flag. Toggle
it before constructing a :class:`~newton.ModelBuilder`.

.. deprecated:: 1.5
    The legacy DOF-shaped layout is deprecated. In a future release the
    coordinate layout becomes the only layout and this flag is removed;
    ``finalize()`` warns when building an affected model (one whose joint
    coordinate and DOF counts differ) under ``False``. Set the flag to
    ``True`` now to migrate.
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
    JointTargetMode,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    StateFlags,
    eval_fk,
    eval_ik,
    eval_inverse_dynamics_force,
    eval_inverse_dynamics_passive,
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
    "ModelFlags",
    "State",
    "StateFlags",
    "eval_fk",
    "eval_ik",
    "eval_inverse_dynamics_force",
    "eval_inverse_dynamics_passive",
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
