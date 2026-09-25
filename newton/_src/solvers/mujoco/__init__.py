# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo solver and programmatic model-authoring helpers.

Use :class:`~newton.solvers.SolverMuJoCo` as the solver class. The module-level
helpers create MuJoCo-specific actuators, tendons, contact pairs, and equality
constraints on a :class:`~newton.ModelBuilder` without exposing custom-frequency
storage details.

Example::

    from newton.solvers import mujoco

    actuator = mujoco.add_actuator_dcmotor(
        builder,
        target=mujoco.ActuatorTarget.joint(joint),
        motorconst=(0.05, 0.05),
        resistance=2.0,
    )
"""

from .actuators import (
    ActuatorTarget,
    add_actuator_dcmotor,
    add_actuator_general,
    add_actuator_motor,
    add_actuator_position,
    add_actuator_velocity,
)
from .contacts import add_contact_pair
from .equality import add_equality_connect, add_equality_joint, add_equality_weld
from .solver_mujoco import SolverMuJoCo
from .tendons import (
    TendonWrapGeom,
    TendonWrapPulley,
    TendonWrapSite,
    add_tendon_fixed,
    add_tendon_spatial,
)

__all__ = [
    "ActuatorTarget",
    "SolverMuJoCo",
    "TendonWrapGeom",
    "TendonWrapPulley",
    "TendonWrapSite",
    "add_actuator_dcmotor",
    "add_actuator_general",
    "add_actuator_motor",
    "add_actuator_position",
    "add_actuator_velocity",
    "add_contact_pair",
    "add_equality_connect",
    "add_equality_joint",
    "add_equality_weld",
    "add_tendon_fixed",
    "add_tendon_spatial",
]
