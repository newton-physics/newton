# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from .broad_phase_common import test_group_pair, test_world_and_group_pair
from .broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from .broad_phase_sap import BroadPhaseSAP
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
from .inertia import compute_inertia_shape, compute_inertia_sphere, transform_inertia
from .sdf_utils import SDF
from .terrain_generator import create_mesh_heightfield, create_mesh_terrain
from .types import (
    Gaussian,
    GeoType,
    Heightfield,
    Mesh,
    TetMesh,
)
from .utils import compute_shape_radius

__all__ = [
    "SDF",
    "BroadPhaseAllPairs",
    "BroadPhaseExplicit",
    "BroadPhaseSAP",
    "Gaussian",
    "GeoType",
    "Heightfield",
    "Mesh",
    "ParticleFlags",
    "ShapeFlags",
    "TetMesh",
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
    "compute_inertia_shape",
    "compute_inertia_sphere",
    "compute_shape_radius",
    "create_mesh_heightfield",
    "create_mesh_terrain",
    "test_group_pair",
    "test_world_and_group_pair",
    "transform_inertia",
]
