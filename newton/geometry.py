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

from ._src.geometry import (
    BroadPhaseAllPairs,
    BroadPhaseExplicit,
    BroadPhaseSAP,
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
    generate_terrain_grid,
    heightfield_to_mesh,
)
from ._src.geometry.inertia import compute_shape_inertia, transform_inertia
from ._src.geometry.utils import remesh_mesh

__all__ = [
    "BroadPhaseAllPairs",
    "BroadPhaseExplicit",
    "BroadPhaseSAP",
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
    "generate_terrain_grid",
    "heightfield_to_mesh",
    "remesh_mesh",
    "transform_inertia",
]
