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

from .broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from .broad_phase_sap import BroadPhaseSAP
from .collision_primitive import (
    box_box,
    capsule_box,
    capsule_capsule,
    mat23f,
    mat43f,
    mat83f,
    plane_box,
    plane_capsule,
    plane_cylinder,
    plane_ellipsoid,
    plane_sphere,
    sphere_box,
    sphere_capsule,
    sphere_cylinder,
    sphere_sphere,
    vec8f,
)
from .flags import ParticleFlags, ShapeFlags
from .gjk import build_ccd_generic
from .inertia import compute_shape_inertia, compute_sphere_inertia, transform_inertia
from .types import (
    MESH_MAXHULLVERT,
    SDF,
    GeoType,
    Mesh,
)
from .utils import compute_shape_radius

__all__ = [
    "MESH_MAXHULLVERT",
    "SDF",
    "BroadPhaseAllPairs",
    "BroadPhaseExplicit",
    "BroadPhaseSAP",
    "GeoType",
    "Mesh",
    "ParticleFlags",
    "ShapeFlags",
    "box_box",
    "build_ccd_generic",
    "capsule_box",
    "capsule_capsule",
    "compute_shape_inertia",
    "compute_shape_radius",
    "compute_sphere_inertia",
    "mat23f",
    "mat43f",
    "mat83f",
    "plane_box",
    "plane_capsule",
    "plane_cylinder",
    "plane_ellipsoid",
    "plane_sphere",
    "sphere_box",
    "sphere_capsule",
    "sphere_cylinder",
    "sphere_sphere",
    "transform_inertia",
    "vec8f",
]
