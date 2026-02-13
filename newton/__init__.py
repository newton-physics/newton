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
    GeoType,
    Mesh,
    ParticleFlags,
    SAPSortType,
    ShapeFlags,
)

__all__ += [
    "SDF",
    "GeoType",
    "Mesh",
    "ParticleFlags",
    "SAPSortType",
    "ShapeFlags",
]

# ==================================================================================
# sim
# ==================================================================================
from ._src.sim import (  # noqa: E402
    ActuatorMode,
    BroadPhaseMode,
    CollisionPipeline,
    Contacts,
    Control,
    EqType,
    JointType,
    Model,
    ModelBuilder,
    State,
    eval_fk,
    eval_ik,
    eval_jacobian,
    eval_mass_matrix,
)

__all__ += [
    "ActuatorMode",
    "BroadPhaseMode",
    "CollisionPipeline",
    "Contacts",
    "Control",
    "EqType",
    "JointType",
    "Model",
    "ModelBuilder",
    "State",
    "eval_fk",
    "eval_ik",
    "eval_jacobian",
    "eval_mass_matrix",
]

# ==================================================================================
# submodule APIs
# ==================================================================================
from . import geometry, ik, math, selection, sensors, solvers, utils, viewer  # noqa: E402

__all__ += [
    "geometry",
    "ik",
    "math",
    "selection",
    "sensors",
    "solvers",
    "utils",
    "viewer",
]
