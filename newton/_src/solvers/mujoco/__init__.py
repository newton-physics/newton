# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Programmatic model-authoring helpers for :class:`~newton.solvers.SolverMuJoCo`.

The module-level helpers create MuJoCo-specific actuators, tendons, contact
pairs, and equality constraints on a :class:`~newton.ModelBuilder` without
exposing custom-frequency storage details. Use
:class:`~newton.solvers.SolverMuJoCo` as the canonical public solver class.

Example::

    from newton.solvers import mujoco

    actuator = mujoco.add_actuator_position(
        builder,
        target=mujoco.ActuatorTarget.joint(joint),
        kp=10.0,
        kv=2.0,
    )
"""

from .actuators import (
    ActuatorTarget,
    add_actuator_general,
    add_actuator_motor,
    add_actuator_position,
    add_actuator_velocity,
)
from .contacts import add_contact_pair
from .equality import add_equality_connect, add_equality_joint, add_equality_weld
from .tendons import (
    TendonWrapGeom,
    TendonWrapPulley,
    TendonWrapSite,
    add_tendon_fixed,
    add_tendon_spatial,
)

__all__ = [
    "ActuatorTarget",
    "TendonWrapGeom",
    "TendonWrapPulley",
    "TendonWrapSite",
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
