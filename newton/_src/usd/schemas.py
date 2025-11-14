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

"""
USD schema resolvers. Currently not used.
"""

from typing import ClassVar

from ..geometry import MESH_MAXHULLVERT
from ..usd.schema_resolver import PrimType, SchemaAttribute, SchemaResolver


class SchemaResolverNewton(SchemaResolver):
    """
    Resolver for the Newton USD schema.

    .. note::
        The Newton USD schema is under development and may change in the future.
    """

    name: ClassVar[str] = "newton"
    mapping: ClassVar[dict[PrimType, dict[str, SchemaAttribute]]] = {
        PrimType.SCENE: {
            "time_step": SchemaAttribute("newton:timeStep", 0.002),
            "max_solver_iterations": SchemaAttribute("newton:maxSolverIterations", 5),
            "enable_gravity": SchemaAttribute("newton:enableGravity", True),
            "rigid_contact_margin": SchemaAttribute("newton:rigidContactMargin", 0.0),
        },
        PrimType.JOINT: {
            "armature": SchemaAttribute("newton:armature", 1.0e-2),
            "friction": SchemaAttribute("newton:friction", 0.0),
            "limit_linear_ke": SchemaAttribute("newton:linear:limitStiffness", 1.0e4),
            "limit_angular_ke": SchemaAttribute("newton:angular:limitStiffness", 1.0e4),
            "limit_rotX_ke": SchemaAttribute("newton:rotX:limitStiffness", 1.0e4),
            "limit_rotY_ke": SchemaAttribute("newton:rotY:limitStiffness", 1.0e4),
            "limit_rotZ_ke": SchemaAttribute("newton:rotZ:limitStiffness", 1.0e4),
            "limit_linear_kd": SchemaAttribute("newton:linear:limitDamping", 1.0e1),
            "limit_angular_kd": SchemaAttribute("newton:angular:limitDamping", 1.0e1),
            "limit_rotX_kd": SchemaAttribute("newton:rotX:limitDamping", 1.0e1),
            "limit_rotY_kd": SchemaAttribute("newton:rotY:limitDamping", 1.0e1),
            "limit_rotZ_kd": SchemaAttribute("newton:rotZ:limitDamping", 1.0e1),
            "angular_position": SchemaAttribute("newton:angular:position", 0.0),
            "linear_position": SchemaAttribute("newton:linear:position", 0.0),
            "rotX_position": SchemaAttribute("newton:rotX:position", 0.0),
            "rotY_position": SchemaAttribute("newton:rotY:position", 0.0),
            "rotZ_position": SchemaAttribute("newton:rotZ:position", 0.0),
            "angular_velocity": SchemaAttribute("newton:angular:velocity", 0.0),
            "linear_velocity": SchemaAttribute("newton:linear:velocity", 0.0),
            "rotX_velocity": SchemaAttribute("newton:rotX:velocity", 0.0),
            "rotY_velocity": SchemaAttribute("newton:rotY:velocity", 0.0),
            "rotZ_velocity": SchemaAttribute("newton:rotZ:velocity", 0.0),
        },
        PrimType.SHAPE: {
            "mesh_hull_vertex_limit": SchemaAttribute("newton:hullVertexLimit", -1),
            # Use ShapeConfig.thickness default for contact margin
            "rigid_contact_margin": SchemaAttribute("newton:rigidContactMargin", 1.0e-5),
        },
        PrimType.BODY: {
            "rigid_body_linear_damping": SchemaAttribute("newton:damping", 0.0),
        },
        PrimType.MATERIAL: {
            "priority": SchemaAttribute("newton:priority", 0),
            "weight": SchemaAttribute("newton:weight", 1.0),
            "stiffness": SchemaAttribute("newton:stiffness", 1.0e5),
            "damping": SchemaAttribute("newton:damping", 1000.0),
        },
        PrimType.ACTUATOR: {
            # Mirror MuJoCo actuator defaults when applicable
            "ctrl_low": SchemaAttribute("newton:ctrlRange:low", 0.0),
            "ctrl_high": SchemaAttribute("newton:ctrlRange:high", 0.0),
            "force_low": SchemaAttribute("newton:forceRange:low", 0.0),
            "force_high": SchemaAttribute("newton:forceRange:high", 0.0),
            "act_low": SchemaAttribute("newton:actRange:low", 0.0),
            "act_high": SchemaAttribute("newton:actRange:high", 0.0),
            "length_low": SchemaAttribute("newton:lengthRange:low", 0.0),
            "length_high": SchemaAttribute("newton:lengthRange:high", 0.0),
            "gainPrm": SchemaAttribute("newton:gainPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "gainType": SchemaAttribute("newton:gainType", "fixed"),
            "biasPrm": SchemaAttribute("newton:biasPrm", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "biasType": SchemaAttribute("newton:biasType", "none"),
            "dynPrm": SchemaAttribute("newton:dynPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "dynType": SchemaAttribute("newton:dynType", "none"),
            # The following have no MuJoCo counterpart; keep unspecified defaults
            "speedTorqueGradient": SchemaAttribute("newton:speedTorqueGradient", None),
            "torqueSpeedGradient": SchemaAttribute("newton:torqueSpeedGradient", None),
            "maxVelocity": SchemaAttribute("newton:maxVelocity", None),
            "gear": SchemaAttribute("newton:gear", [1, 0, 0, 0, 0, 0]),
        },
    }


class SchemaResolverPhysx(SchemaResolver):
    """
    Resolver for the PhysX USD schema.
    """

    name: ClassVar[str] = "physx"
    extra_attr_namespaces: ClassVar[list[str]] = [
        # Scene and rigid body
        "physxScene",
        "physxRigidBody",
        # Collisions and meshes
        "physxCollision",
        "physxConvexHullCollision",
        "physxConvexDecompositionCollision",
        "physxTriangleMeshCollision",
        "physxTriangleMeshSimplificationCollision",
        "physxSDFMeshCollision",
        # Materials
        "physxMaterial",
        # Joints and limits
        "physxJoint",
        "physxLimit",
        # Articulations
        "physxArticulation",
        # State attributes (for joint position/velocity initialization)
        "state",
        # Drive attributes
        "drive",
    ]

    mapping: ClassVar[dict[PrimType, dict[str, SchemaAttribute]]] = {
        PrimType.SCENE: {
            "time_step": SchemaAttribute(
                "physxScene:timeStepsPerSecond", 60, lambda hz: (1.0 / hz) if (hz and hz > 0) else None
            ),
            "max_solver_iterations": SchemaAttribute("physxScene:maxVelocityIterationCount", 255),
            "enable_gravity": SchemaAttribute("physxRigidBody:disableGravity", False, lambda value: not value),
            "rigid_contact_margin": SchemaAttribute("physxScene:contactOffset", 0.0),
        },
        PrimType.JOINT: {
            "armature": SchemaAttribute("physxJoint:armature", 0.0),
            # Per-axis linear limit aliases
            "limit_transX_ke": SchemaAttribute("physxLimit:linear:stiffness", 0.0),
            "limit_transY_ke": SchemaAttribute("physxLimit:linear:stiffness", 0.0),
            "limit_transZ_ke": SchemaAttribute("physxLimit:linear:stiffness", 0.0),
            "limit_transX_kd": SchemaAttribute("physxLimit:linear:damping", 0.0),
            "limit_transY_kd": SchemaAttribute("physxLimit:linear:damping", 0.0),
            "limit_transZ_kd": SchemaAttribute("physxLimit:linear:damping", 0.0),
            "limit_linear_ke": SchemaAttribute("physxLimit:linear:stiffness", 0.0),
            "limit_angular_ke": SchemaAttribute("physxLimit:angular:stiffness", 0.0),
            "limit_rotX_ke": SchemaAttribute("physxLimit:rotX:stiffness", 0.0),
            "limit_rotY_ke": SchemaAttribute("physxLimit:rotY:stiffness", 0.0),
            "limit_rotZ_ke": SchemaAttribute("physxLimit:rotZ:stiffness", 0.0),
            "limit_linear_kd": SchemaAttribute("physxLimit:linear:damping", 0.0),
            "limit_angular_kd": SchemaAttribute("physxLimit:angular:damping", 0.0),
            "limit_rotX_kd": SchemaAttribute("physxLimit:rotX:damping", 0.0),
            "limit_rotY_kd": SchemaAttribute("physxLimit:rotY:damping", 0.0),
            "limit_rotZ_kd": SchemaAttribute("physxLimit:rotZ:damping", 0.0),
            "angular_position": SchemaAttribute("state:angular:physics:position", 0.0),
            "linear_position": SchemaAttribute("state:linear:physics:position", 0.0),
            "rotX_position": SchemaAttribute("state:rotX:physics:position", 0.0),
            "rotY_position": SchemaAttribute("state:rotY:physics:position", 0.0),
            "rotZ_position": SchemaAttribute("state:rotZ:physics:position", 0.0),
            "angular_velocity": SchemaAttribute("state:angular:physics:velocity", 0.0),
            "linear_velocity": SchemaAttribute("state:linear:physics:velocity", 0.0),
            "rotX_velocity": SchemaAttribute("state:rotX:physics:velocity", 0.0),
            "rotY_velocity": SchemaAttribute("state:rotY:physics:velocity", 0.0),
            "rotZ_velocity": SchemaAttribute("state:rotZ:physics:velocity", 0.0),
        },
        PrimType.SHAPE: {
            # Mesh hull vertex limit
            "mesh_hull_vertex_limit": SchemaAttribute("physxConvexHullCollision:hullVertexLimit", 64),
            # Collision contact offset
            "rigid_contact_margin": SchemaAttribute("physxCollision:contactOffset", float("-inf")),
        },
        PrimType.MATERIAL: {
            "stiffness": SchemaAttribute("physxMaterial:compliantContactStiffness", 0.0),
            "damping": SchemaAttribute("physxMaterial:compliantContactDamping", 0.0),
        },
        PrimType.BODY: {
            # Rigid body damping
            "rigid_body_linear_damping": SchemaAttribute("physxRigidBody:linearDamping", 0.0),
            "rigid_body_angular_damping": SchemaAttribute("physxRigidBody:angularDamping", 0.05),
        },
    }


def solref_to_stiffness_damping(solref):
    """Convert MuJoCo solref (timeconst, dampratio) to internal stiffness and damping.

    Returns a tuple (stiffness, damping).

    Standard mode (timeconst > 0):
        k = 1 / (timeconst^2 * dampratio^2)
        b = 2 / timeconst
    Direct mode (both negative):
        solref encodes (-stiffness, -damping) directly
        k = -timeconst
        b = -dampratio
    """
    try:
        timeconst = float(solref[0])
        dampratio = float(solref[1])
    except (TypeError, ValueError, IndexError):
        return None, None

    # Direct mode: both negative → solref encodes (-stiffness, -damping)
    if timeconst < 0.0 and dampratio < 0.0:
        return -timeconst, -dampratio

    # Standard mode: compute stiffness and damping
    if timeconst <= 0.0:
        return None, None

    stiffness = 1.0 / (timeconst * timeconst * dampratio * dampratio)

    if dampratio <= 0.0:
        return stiffness, None

    damping = 2.0 / timeconst
    return stiffness, damping


def solref_to_stiffness(solref):
    """Convert MuJoCo solref (timeconst, dampratio) to internal stiffness.

    Standard mode (timeconst > 0): k = 1 / (timeconst^2 * dampratio^2)
    Direct mode (both negative): k = -timeconst (encodes -stiffness directly)
    """
    stiffness, _ = solref_to_stiffness_damping(solref)
    return stiffness


def solref_to_damping(solref):
    """Convert MuJoCo solref (timeconst, dampratio) to internal damping.

    Standard mode (both positive): b = 2 / timeconst
    Direct mode (both negative): b = -dampratio (encodes -damping directly)
    """
    _, damping = solref_to_stiffness_damping(solref)
    return damping


class SchemaResolverMjc(SchemaResolver):
    """
    Resolver for the MuJoCo USD schema.
    """

    name: ClassVar[str] = "mjc"

    mapping: ClassVar[dict[PrimType, dict[str, SchemaAttribute]]] = {
        PrimType.SCENE: {
            "time_step": SchemaAttribute("mjc:option:timestep", 0.002),
            "max_solver_iterations": SchemaAttribute("mjc:option:iterations", 100),
            "enable_gravity": SchemaAttribute("mjc:flag:gravity", True),
            "rigid_contact_margin": SchemaAttribute("mjc:option:o_margin", 0.0),
        },
        PrimType.JOINT: {
            "armature": SchemaAttribute("mjc:armature", 0.0),
            "friction": SchemaAttribute("mjc:frictionloss", 0.0),
            # Per-axis linear aliases mapped to solref
            "limit_transX_ke": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_stiffness),
            "limit_transY_ke": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_stiffness),
            "limit_transZ_ke": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_stiffness),
            "limit_transX_kd": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_damping),
            "limit_transY_kd": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_damping),
            "limit_transZ_kd": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_damping),
            "limit_linear_ke": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_stiffness),
            "limit_angular_ke": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_stiffness),
            "limit_rotX_ke": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_stiffness),
            "limit_rotY_ke": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_stiffness),
            "limit_rotZ_ke": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_stiffness),
            "limit_linear_kd": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_damping),
            "limit_angular_kd": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_damping),
            "limit_rotX_kd": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_damping),
            "limit_rotY_kd": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_damping),
            "limit_rotZ_kd": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_damping),
        },
        PrimType.SHAPE: {
            # Mesh
            "mesh_hull_vertex_limit": SchemaAttribute("mjc:maxhullvert", MESH_MAXHULLVERT),
            # Collisions
            "rigid_contact_margin": SchemaAttribute("mjc:margin", 0.0),
        },
        PrimType.MATERIAL: {
            # Materials and contact models
            "priority": SchemaAttribute("mjc:priority", 0),
            "weight": SchemaAttribute("mjc:solmix", 1.0),
            "stiffness": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_stiffness),
            "damping": SchemaAttribute("mjc:solref", [0.02, 1.0], solref_to_damping),
        },
        PrimType.BODY: {
            # Rigid body / joint domain
            "rigid_body_linear_damping": SchemaAttribute("mjc:damping", 0.0),
        },
        PrimType.ACTUATOR: {
            # Actuators
            "ctrl_low": SchemaAttribute("mjc:ctrlRange:min", 0.0),
            "ctrl_high": SchemaAttribute("mjc:ctrlRange:max", 0.0),
            "force_low": SchemaAttribute("mjc:forceRange:min", 0.0),
            "force_high": SchemaAttribute("mjc:forceRange:max", 0.0),
            "act_low": SchemaAttribute("mjc:actRange:min", 0.0),
            "act_high": SchemaAttribute("mjc:actRange:max", 0.0),
            "length_low": SchemaAttribute("mjc:lengthRange:min", 0.0),
            "length_high": SchemaAttribute("mjc:lengthRange:max", 0.0),
            "gainPrm": SchemaAttribute("mjc:gainPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "gainType": SchemaAttribute("mjc:gainType", "fixed"),
            "biasPrm": SchemaAttribute("mjc:biasPrm", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "biasType": SchemaAttribute("mjc:biasType", "none"),
            "dynPrm": SchemaAttribute("mjc:dynPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "dynType": SchemaAttribute("mjc:dynType", "none"),
            "gear": SchemaAttribute("mjc:gear", [1, 0, 0, 0, 0, 0]),
        },
    }
