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

import warp as wp

from .broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from .broad_phase_sap import BroadPhaseSAP
from .flags import (
    PARTICLE_FLAG_ACTIVE,
    SHAPE_FLAG_COLLIDE_PARTICLES,
    SHAPE_FLAG_COLLIDE_SHAPES,
    SHAPE_FLAG_VISIBLE,
)
from .gjk import build_ccd_generic
from .inertia import compute_shape_inertia, transform_inertia
from .types import (
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CONE,
    GEO_CYLINDER,
    GEO_MESH,
    GEO_NONE,
    GEO_PLANE,
    GEO_SDF,
    GEO_SPHERE,
    MESH_MAXHULLVERT,
    SDF,
    Mesh,
)
from .utils import compute_shape_radius


@wp.func
def create_sphere(radius: float):
    return (GEO_SPHERE, wp.vec3(radius, 0.0, 0.0))


@wp.func
def create_box(width: float, height: float, depth: float):
    return (GEO_BOX, wp.vec3(width, height, depth))


@wp.func
def create_capsule(radius: float, height: float):
    return (GEO_CAPSULE, wp.vec3(radius, height, 0.0))


@wp.func
def create_cylinder(radius: float, height: float):
    return (GEO_CYLINDER, wp.vec3(radius, height, 0.0))


@wp.func
def create_cone(radius: float, height: float):
    return (GEO_CONE, wp.vec3(radius, height, 0.0))


@wp.func
def create_plane(width: float = 0.0, height: float = 0.0):
    """Create a plane. If width and height are 0.0, creates an infinite
    plane."""
    return (GEO_PLANE, wp.vec3(width, height, 0.0))


@wp.func
def create_none():
    """Create an empty/null geometry."""
    return (GEO_NONE, wp.vec3(0.0, 0.0, 0.0))


__all__ = [
    "GEO_BOX",
    "GEO_CAPSULE",
    "GEO_CONE",
    "GEO_CYLINDER",
    "GEO_MESH",
    "GEO_NONE",
    "GEO_PLANE",
    "GEO_SDF",
    "GEO_SPHERE",
    "MESH_MAXHULLVERT",
    "PARTICLE_FLAG_ACTIVE",
    "SDF",
    "SHAPE_FLAG_COLLIDE_PARTICLES",
    "SHAPE_FLAG_COLLIDE_SHAPES",
    "SHAPE_FLAG_VISIBLE",
    "BroadPhaseAllPairs",
    "BroadPhaseExplicit",
    "BroadPhaseSAP",
    "Mesh",
    "build_ccd_generic",
    "compute_shape_inertia",
    "compute_shape_radius",
    "create_box",
    "create_capsule",
    "create_cone",
    "create_cylinder",
    "create_none",
    "create_plane",
    "create_sphere",
    "get_shape_radius",
    "transform_inertia",
]
