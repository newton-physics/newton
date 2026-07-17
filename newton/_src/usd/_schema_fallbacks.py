# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fallbacks for schemas that may not be registered with PXR."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_PHYSX_LIMIT_AXES = ("linear", "angular", "transX", "transY", "transZ", "rotX", "rotY", "rotZ")
_JOINT_STATE_AXES = ("linear", "angular", "rotX", "rotY", "rotZ")


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


_SCHEMA_FALLBACKS: dict[str, dict[str, Any]] = {
    "NewtonSceneAPI": {
        "newton:maxSolverIterations": -1,
        "newton:timeStepsPerSecond": 1000,
        "newton:gravityEnabled": True,
    },
    "NewtonJointAPI": {
        "newton:armature": 0.0,
        "newton:damping": 0.0,
        "newton:friction": 0.0,
        "newton:velocityLimit": float("inf"),
        "newton:limitStiffness": float("-inf"),
        "newton:limitDamping": float("-inf"),
    },
    "NewtonCollisionAPI": {
        "newton:contactMargin": 0.0,
        "newton:contactGap": float("-inf"),
    },
    "NewtonMeshCollisionAPI": {
        "newton:maxHullVertices": -1,
    },
    "NewtonSDFCollisionAPI": {
        "newton:sdfMaxResolution": 64,
        "newton:sdfTargetVoxelSize": float("-inf"),
        "newton:sdfNarrowBandInner": _f32(-0.1),
        "newton:sdfNarrowBandOuter": _f32(0.1),
        "newton:sdfTextureFormat": "uint16",
        "newton:sdfPadding": float("-inf"),
        "newton:hydroelasticEnabled": False,
        "newton:hydroelasticStiffness": 1.0e10,
    },
    "NewtonMassAPI": {
        "newton:massModel": "solid",
        "newton:shellThickness": float("-inf"),
    },
    "NewtonArticulationRootAPI": {
        "newton:selfCollisionEnabled": True,
    },
    "NewtonMaterialAPI": {
        "newton:torsionalFriction": _f32(0.005),
        "newton:rollingFriction": _f32(0.0001),
        "newton:contactStiffness": float("-inf"),
        "newton:contactDamping": float("-inf"),
        "newton:contactFrictionGain": float("-inf"),
        "newton:contactAdhesion": float("-inf"),
    },
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


def _schema_fallbacks(
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    fallbacks = deepcopy(_SCHEMA_FALLBACKS)
    if overrides is not None:
        for schema_name, values in overrides.items():
            fallbacks.setdefault(schema_name, {}).update(values)
    return fallbacks
