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

from . import solvers
from ._version import __version__

# Core functionality
from .core import (
    Axis,
    AxisType,
)

# Geometry functionality
from .geometry import (
    SDF,
    GeoType,
    Mesh,
    create_box,
    create_capsule,
    create_cone,
    create_cylinder,
    create_none,
    create_plane,
    create_sphere,
)

# Simulation functionality
from .sim import (
    EQ_CONNECT,
    EQ_JOINT,
    EQ_WELD,
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
    Contacts,
    Control,
    Model,
    ModelBuilder,
    State,
    eval_fk,
    eval_ik,
)

__all__ = [
    "EQ_CONNECT",
    "EQ_JOINT",
    "EQ_WELD",
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
    "SDF",
    "Axis",
    "AxisType",
    "Contacts",
    "Control",
    "GeoType",
    "Mesh",
    "Model",
    "ModelBuilder",
    "State",
    "__version__",
    "create_box",
    "create_capsule",
    "create_cone",
    "create_cylinder",
    "create_none",
    "create_plane",
    "create_sphere",
    "eval_fk",
    "eval_ik",
    "solvers",
]
