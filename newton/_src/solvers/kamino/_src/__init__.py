# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Kamino: A physics back-end for Newton for constrained multi-body body simulation.
"""

from .core.bodies import (
    convert_base_origin_to_com,
    convert_body_com_to_origin,
    convert_body_origin_to_com,
    convert_geom_offset_origin_to_com,
)
from .core.control import ControlKamino
from .core.conversions import (
    compute_material_first_shape,
    convert_model_joint_actuation,
    convert_model_joint_transforms,
    convert_model_materials,
    validate_model_joint_updates,
)
from .core.gravity import convert_model_gravity
from .core.joints import JOINT_QMAX, JOINT_QMIN, JointActuationType
from .core.model import ModelKamino
from .core.state import StateKamino
from .geometry.contacts import (
    ContactsKamino,
    convert_contacts_kamino_to_newton,
    convert_contacts_newton_to_kamino,
)
from .geometry.detector import CollisionDetector
from .solver_kamino_impl import SolverKaminoImpl
from .utils import logger as msg

###
# Kamino API
###

__all__ = [
    "JOINT_QMAX",
    "JOINT_QMIN",
    "CollisionDetector",
    "ContactsKamino",
    "ControlKamino",
    "JointActuationType",
    "ModelKamino",
    "SolverKaminoImpl",
    "StateKamino",
    "compute_material_first_shape",
    "convert_base_origin_to_com",
    "convert_body_com_to_origin",
    "convert_body_origin_to_com",
    "convert_contacts_kamino_to_newton",
    "convert_contacts_newton_to_kamino",
    "convert_geom_offset_origin_to_com",
    "convert_model_gravity",
    "convert_model_joint_actuation",
    "convert_model_joint_transforms",
    "convert_model_materials",
    "msg",
    "validate_model_joint_updates",
]
