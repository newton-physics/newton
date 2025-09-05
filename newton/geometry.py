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

from ._src.geometry import BroadPhaseAllPairs, BroadPhaseExplicit, BroadPhaseSAP
from ._src.geometry.collision_primitive import (
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
from ._src.geometry.inertia import compute_shape_inertia, transform_inertia
from ._src.geometry.utils import remesh_mesh

__all__ = [
    "BroadPhaseAllPairs",
    "BroadPhaseExplicit",
    "BroadPhaseSAP",
    "box_box",
    "capsule_box",
    "capsule_capsule",
    "compute_shape_inertia",
    "mat23f",
    "mat43f",
    "mat83f",
    "plane_box",
    "plane_capsule",
    "plane_cylinder",
    "plane_ellipsoid",
    "plane_sphere",
    "remesh_mesh",
    "sphere_box",
    "sphere_capsule",
    "sphere_cylinder",
    "sphere_sphere",
    "transform_inertia",
    "vec8f",
]
