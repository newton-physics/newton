# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fallbacks for USD schemas without public schema packages."""

from __future__ import annotations

import struct
from typing import Any


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


_PHYSX_LIMIT_AXES = ("linear", "angular", "transX", "transY", "transZ", "rotX", "rotY", "rotZ")
_JOINT_STATE_AXES = ("linear", "angular", "rotX", "rotY", "rotZ")


# Subset of physx-usd-schemas 25.11.1 used by SchemaResolverPhysx.
_PHYSX_SCHEMA_FALLBACKS: dict[str, dict[str, Any]] = {
    "PhysxSceneAPI": {
        "physxScene:maxVelocityIterationCount": 255,
        "physxScene:timeStepsPerSecond": 60,
    },
    "PhysxRigidBodyAPI": {
        "physxRigidBody:disableGravity": False,
        "physxRigidBody:linearDamping": 0.0,
        "physxRigidBody:angularDamping": _f32(0.05),
    },
    "PhysxJointAPI": {
        "physxJoint:armature": 0.0,
        "physxJoint:maxJointVelocity": float("inf"),
    },
    "PhysxConvexHullCollisionAPI": {
        "physxConvexHullCollision:hullVertexLimit": 64,
    },
    "PhysxCollisionAPI": {
        "physxCollision:contactOffset": float("-inf"),
        "physxCollision:restOffset": float("-inf"),
    },
    "PhysxMaterialAPI": {
        "physxMaterial:compliantContactStiffness": 0.0,
        "physxMaterial:compliantContactDamping": 0.0,
    },
    "PhysxArticulationAPI": {
        "physxArticulation:enabledSelfCollisions": True,
    },
    **{
        f"PhysxLimitAPI:{axis}": {
            f"physxLimit:{axis}:stiffness": 0.0,
            f"physxLimit:{axis}:damping": 0.0,
        }
        for axis in _PHYSX_LIMIT_AXES
    },
    **{
        f"PhysicsJointStateAPI:{axis}": {
            f"state:{axis}:physics:position": 0.0,
            f"state:{axis}:physics:velocity": 0.0,
        }
        for axis in _JOINT_STATE_AXES
    },
}


# MJC does not yet publish its USD schema resources.
_MJC_SCHEMA_FALLBACKS: dict[str, dict[str, Any]] = {
    "MjcSceneAPI": {
        "mjc:option:iterations": 100,
        "mjc:option:timestep": 0.002,
        "mjc:flag:gravity": True,
    },
    "MjcJointAPI": {
        "mjc:armature": 0.0,
        "mjc:frictionloss": 0.0,
        "mjc:solreflimit": [0.02, 1.0],
    },
    "MjcMeshCollisionAPI": {
        "mjc:maxhullvert": -1,
    },
    "MjcCollisionAPI": {
        "mjc:margin": 0.0,
        "mjc:gap": 0.0,
        "mjc:shellinertia": False,
        "mjc:solref": [0.02, 1.0],
    },
    "MjcMaterialAPI": {
        "mjc:torsionalfriction": 0.005,
        "mjc:rollingfriction": 0.0001,
    },
    "MjcActuator": {
        "mjc:ctrlRange:min": 0.0,
        "mjc:ctrlRange:max": 0.0,
        "mjc:forceRange:min": 0.0,
        "mjc:forceRange:max": 0.0,
        "mjc:actRange:min": 0.0,
        "mjc:actRange:max": 0.0,
        "mjc:lengthRange:min": 0.0,
        "mjc:lengthRange:max": 0.0,
        "mjc:gainPrm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mjc:gainType": "fixed",
        "mjc:biasPrm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mjc:biasType": "none",
        "mjc:dynPrm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mjc:dynType": "none",
        "mjc:gear": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
}
