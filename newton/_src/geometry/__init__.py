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

from .broad_phase_common import test_group_pair, test_world_and_group_pair
from .broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from .broad_phase_sap import BroadPhaseSAP, SAPSortType
from .collision_primitive import (
    collide_box_box,
    collide_capsule_box,
    collide_capsule_capsule,
    collide_plane_box,
    collide_plane_capsule,
    collide_plane_cylinder,
    collide_plane_ellipsoid,
    collide_plane_sphere,
    collide_sphere_box,
    collide_sphere_capsule,
    collide_sphere_cylinder,
    collide_sphere_sphere,
)
from .flags import ParticleFlags, ShapeFlags
from .inertia import compute_shape_inertia, compute_sphere_inertia, transform_inertia
from .terrain_generator import generate_terrain_grid
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
    "SAPSortType",
    "ShapeFlags",
    "collide_box_box",
    "collide_capsule_box",
    "collide_capsule_capsule",
    "collide_plane_box",
    "collide_plane_capsule",
    "collide_plane_cylinder",
    "collide_plane_ellipsoid",
    "collide_plane_sphere",
    "collide_sphere_box",
    "collide_sphere_capsule",
    "collide_sphere_cylinder",
    "collide_sphere_sphere",
    "compute_shape_inertia",
    "compute_shape_radius",
    "compute_sphere_inertia",
    "generate_terrain_grid",
    "test_group_pair",
    "test_world_and_group_pair",
    "transform_inertia",
]
