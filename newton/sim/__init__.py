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

from .articulation import eval_fk, eval_ik
from .builder import ModelBuilder
from .collide import CollisionPipeline, count_rigid_contact_points
from .contacts import Contacts
from .control import Control
from .flags import (
    NOTIFY_FLAG_BODY_INERTIAL_PROPERTIES,
    NOTIFY_FLAG_BODY_PROPERTIES,
    NOTIFY_FLAG_DOF_PROPERTIES,
    NOTIFY_FLAG_JOINT_AXIS_PROPERTIES,
    NOTIFY_FLAG_JOINT_PROPERTIES,
    NOTIFY_FLAG_SHAPE_PROPERTIES,
)
from .graph_coloring import color_graph, plot_graph
from .joints import (
    JOINT_BALL,
    JOINT_D6,
    JOINT_DISTANCE,
    JOINT_FIXED,
    JOINT_FREE,
    JOINT_MODE_NONE,
    JOINT_MODE_TARGET_POSITION,
    JOINT_MODE_TARGET_VELOCITY,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    get_joint_dof_count,
)
from .model import Model
from .state import State
from .style3d import Style3DModel, Style3DModelBuilder
from .types import ShapeGeometry, ShapeMaterials

__all__ = [
    "JOINT_BALL",
    "JOINT_D6",
    "JOINT_DISTANCE",
    "JOINT_FIXED",
    "JOINT_FREE",
    "JOINT_MODE_NONE",
    "JOINT_MODE_TARGET_POSITION",
    "JOINT_MODE_TARGET_VELOCITY",
    "JOINT_PRISMATIC",
    "JOINT_REVOLUTE",
    "NOTIFY_FLAG_BODY_INERTIAL_PROPERTIES",
    "NOTIFY_FLAG_BODY_PROPERTIES",
    "NOTIFY_FLAG_DOF_PROPERTIES",
    "NOTIFY_FLAG_JOINT_AXIS_PROPERTIES",
    "NOTIFY_FLAG_JOINT_PROPERTIES",
    "NOTIFY_FLAG_SHAPE_PROPERTIES",
    "CollisionPipeline",
    "Contacts",
    "Control",
    "Model",
    "ModelBuilder",
    "ShapeGeometry",
    "ShapeMaterials",
    "State",
    "Style3DModel",
    "Style3DModelBuilder",
    "color_graph",
    "count_rigid_contact_points",
    "eval_fk",
    "eval_ik",
    "get_joint_dof_count",
    "plot_graph",
]
