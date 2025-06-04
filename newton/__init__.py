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
    JOINT_BALL,
    JOINT_COMPOUND,
    JOINT_D6,
    JOINT_DISTANCE,
    JOINT_FIXED,
    JOINT_FREE,
    JOINT_MODE_FORCE,
    JOINT_MODE_TARGET_POSITION,
    JOINT_MODE_TARGET_VELOCITY,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    JOINT_UNIVERSAL,
    Axis,
    AxisType,
    Control,
    Model,
    ModelBuilder,
    State,
    eval_fk,
    eval_ik,
)

# Geometry functionality
from .geometry import (
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CONE,
    GEO_CYLINDER,
    GEO_MESH,
    GEO_NONE,
    GEO_PLANE,
    GEO_SDF,
    GEO_SPHERE,
    SDF,
    Contacts,
    Mesh,
    create_box,
    create_capsule,
    create_cone,
    create_cylinder,
    create_none,
    create_plane,
    create_sphere,
)

__all__ = [
    "GEO_BOX",
    "GEO_CAPSULE",
    "GEO_CONE",
    "GEO_CYLINDER",
    "GEO_MESH",
    "GEO_NONE",
    "GEO_PLANE",
    "GEO_SDF",
    # Geometry constants
    "GEO_SPHERE",
    # Joint constants
    "JOINT_BALL",
    "JOINT_COMPOUND",
    "JOINT_D6",
    "JOINT_DISTANCE",
    "JOINT_FIXED",
    "JOINT_FREE",
    "JOINT_MODE_FORCE",
    "JOINT_MODE_TARGET_POSITION",
    "JOINT_MODE_TARGET_VELOCITY",
    "JOINT_PRISMATIC",
    "JOINT_REVOLUTE",
    "JOINT_UNIVERSAL",
    "SDF",
    # Core types and classes
    "Axis",
    "AxisType",
    "Contacts",
    "Control",
    # Geometry types and classes
    "Mesh",
    "Model",
    "ModelBuilder",
    "ShapeMaterials",
    "State",
    # Version
    "__version__",
    "create_box",
    "create_capsule",
    "create_cone",
    "create_cylinder",
    "create_none",
    "create_plane",
    # Geometry creation functions (the main new API)
    "create_sphere",
    # Core functions
    "eval_fk",
    "eval_ik",
    # Submodules (for those who want to use them directly)
    "solvers",
]
