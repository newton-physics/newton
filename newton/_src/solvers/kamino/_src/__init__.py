# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kamino: A physics back-end for Newton for constrained multi-body body simulation.
"""

from .core.bodies import (
    convert_base_origin_to_com,
    convert_body_com_to_origin,
    convert_body_origin_to_com,
)
from .core.control import ControlKamino
from .core.conversions import convert_model_joint_transforms
from .core.gravity import convert_model_gravity
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
    "CollisionDetector",
    "ContactsKamino",
    "ControlKamino",
    "ModelKamino",
    "SolverKaminoImpl",
    "StateKamino",
    "convert_base_origin_to_com",
    "convert_body_com_to_origin",
    "convert_body_origin_to_com",
    "convert_contacts_kamino_to_newton",
    "convert_contacts_newton_to_kamino",
    "convert_model_gravity",
    "convert_model_joint_transforms",
    "msg",
]
