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
USD schema resolvers.
"""

from typing import ClassVar

from ..sim.builder import ModelBuilder
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
            "max_solver_iterations": SchemaAttribute("newton:maxSolverIterations", -1),
            "time_steps_per_second": SchemaAttribute("newton:timeStepsPerSecond", 1000),
            "gravity_enabled": SchemaAttribute("newton:gravityEnabled", True),
        },
        PrimType.JOINT: {
            # warning: there is no NewtonJointAPI, none of these are schema attributes
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
            # Mesh
            "max_hull_vertices": SchemaAttribute("newton:maxHullVertices", -1),
            # Collisions
            "contact_margin": SchemaAttribute("newton:contactMargin", float("-inf")),
            # SDF (Signed Distance Field) configuration
            "sdf_max_resolution": SchemaAttribute("newton:sdfMaxResolution", None),
            "sdf_narrow_band_inner": SchemaAttribute("newton:sdfNarrowBandInner", None),
            "sdf_narrow_band_outer": SchemaAttribute("newton:sdfNarrowBandOuter", None),
            "sdf_target_voxel_size": SchemaAttribute("newton:sdfTargetVoxelSize", None),
            # Hydroelastic contacts
            "is_hydroelastic": SchemaAttribute("newton:isHydroelastic", False),
            "k_hydro": SchemaAttribute("newton:kHydro", 1.0e10),
        },
        PrimType.BODY: {},
        PrimType.MATERIAL: {
            "torsional_friction": SchemaAttribute("newton:torsionalFriction", 0.25),
            "rolling_friction": SchemaAttribute("newton:rollingFriction", 0.0005),
        },
        PrimType.ACTUATOR: {},
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
            "max_solver_iterations": SchemaAttribute("physxScene:maxVelocityIterationCount", 255),
            "time_steps_per_second": SchemaAttribute("physxScene:timeStepsPerSecond", 60),
            "gravity_enabled": SchemaAttribute("physxRigidBody:disableGravity", False, lambda value: not value),
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
            # Mesh
            "max_hull_vertices": SchemaAttribute("physxConvexHullCollision:hullVertexLimit", 64),
            # Collisions
            "contact_margin": SchemaAttribute("physxCollision:contactOffset", float("-inf")),
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
    if timeconst <= 0.0 or dampratio <= 0.0:
        return None, None

    stiffness = 1.0 / (timeconst * timeconst * dampratio * dampratio)
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
            "max_solver_iterations": SchemaAttribute("mjc:option:iterations", 100),
            "time_steps_per_second": SchemaAttribute(
                "mjc:option:timestep", 0.002, lambda s: int(1.0 / s) if (s and s > 0) else None
            ),
            "gravity_enabled": SchemaAttribute("mjc:flag:gravity", True),
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
            "max_hull_vertices": SchemaAttribute("mjc:maxhullvert", -1),
        },
        PrimType.MATERIAL: {
            # Materials
            "torsional_friction": SchemaAttribute("mjc:torsionalfriction", 0.005),
            "rolling_friction": SchemaAttribute("mjc:rollingfriction", 0.0001),
            # Contact models
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

    def validate_custom_attributes(self, builder: ModelBuilder) -> None:
        """
        Validate that MuJoCo custom attributes have been registered on the builder.

        Users must call :meth:`SolverMuJoCo.register_custom_attributes` before parsing
        USD files with this resolver.

        Raises:
            RuntimeError: If required MuJoCo custom attributes are not registered.
        """
        has_mujoco_attrs = any(attr.namespace == "mujoco" for attr in builder.custom_attributes.values())
        if not has_mujoco_attrs:
            raise RuntimeError(
                "MuJoCo custom attributes not registered. "
                "Call SolverMuJoCo.register_custom_attributes(builder) before parsing USD with SchemaResolverMjc."
            )
