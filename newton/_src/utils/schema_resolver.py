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

from __future__ import annotations

import warnings as _w
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar

import warp as wp

from ..sim.model import CustomAttribute, ModelAttributeAssignment, ModelAttributeFrequency

try:
    from pxr import Usd
except ImportError:
    Usd = None


class PrimType(IntEnum):
    """Enumeration of USD prim types that can be resolved by schema resolvers."""

    SCENE = 0
    JOINT = 1
    SHAPE = 2
    BODY = 3
    MATERIAL = 4
    ACTUATOR = 5


@dataclass
class Attribute:
    """
    Specifies a USD attribute and its transformation function.

    Args:
        usd_name (str): The name of the USD attribute.
        default (Any | None): Default USD-authored value from schema, if any.
        transformer (Callable[[Any], Any] | None): A function to transform the raw USD attribute value
            into the format expected by Newton. Takes the USD value as input and returns the transformed
            value. For example, converting PhysX timeStepsPerSecond to Newton's timestep by computing 1/hz.
    """

    usd_name: str
    default: Any | None = None
    transformer: Callable[[Any], Any] | None = None


class SchemaResolver:
    # mapping is a dictionary for known variables in Newton. Its purpose is to map usd attributes to existing Newton data.
    # PrimType -> variable -> list[Attribute]
    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {}

    # Name of the schema resolver
    name: ClassVar[str] = ""

    # extra_attr_namespaces is a list of namespaces for extra attributes that are not in the mapping.
    extra_attr_namespaces: ClassVar[list[str]] = []

    def __init__(self) -> None:
        # Precompute the full set of USD attribute names referenced by this resolver's mapping.
        names: set[str] = set()
        try:
            mapping_items = self.mapping.items()
        except AttributeError:
            mapping_items = []
        for _prim_type, var_map in mapping_items:
            try:
                var_items = var_map.items()
            except AttributeError:
                continue
            for _var, specs in var_items:
                for spec in specs:
                    names.add(spec.usd_name)
        self._solver_attributes: list[str] = list(names)

    def get_value(self, prim, prim_type: PrimType, key: str) -> tuple[Any, str] | None:
        """
        Get attribute value for a given prim type and key.

        Args:
            prim: USD prim to query
            prim_type: Prim type (PrimType enum)
            key: Attribute key within the prim type

        Returns:
            Tuple of (value, usd_attr_name) if found, None otherwise
        """
        if prim is None:
            return None
        for spec in self.mapping.get(prim_type, {}).get(key, []):
            v = _get_attr(prim, spec.usd_name)
            if v is not None:
                return (spec.transformer(v) if spec.transformer is not None else v), spec.usd_name
        return None

    def collect_prim_solver_attrs(self, prim) -> dict[str, Any]:
        """
        Collect solver-specific attributes for a single prim.
        Returns dictionary of solver-specific attributes for this prim.
        """
        if prim is None:
            return {}

        # Collect explicit attribute names defined in the resolver mapping (precomputed)
        prim_solver_attrs = (
            _collect_solver_mapped_attrs(prim, self._solver_attributes) if self._solver_attributes else {}
        )

        # Collect attributes by known solver-specific prefixes
        # USD expects namespace tokens without ':' (e.g., 'newton', 'mjc', 'physxArticulation')
        main_prefix = self.name
        all_prefixes = [main_prefix]
        if self.extra_attr_namespaces:
            all_prefixes.extend(self.extra_attr_namespaces)
        prefixed_attrs = _collect_solver_specific_attrs(prim, all_prefixes)

        # Merge and return (explicit names take precedence)
        merged: dict[str, Any] = {}
        merged.update(prefixed_attrs)
        merged.update(prim_solver_attrs)
        return merged


def _get_attr(prim, name: str):
    if prim is None:
        return None
    attr = prim.GetAttribute(name)
    if not (attr and attr.IsValid() and attr.HasAuthoredValue()):
        return None
    return attr.Get()


def _collect_solver_mapped_attrs(prim, names: list[str]) -> dict[str, Any]:
    """Collect solver-specific attributes authored on the prim that have direct mappings in the resolver mapping"""
    out = {}
    for n in names:
        v = _get_attr(prim, n)
        if v is not None:
            out[n] = v
    return out


def _collect_solver_specific_attrs(prim, namespaces: list[str]) -> dict[str, Any]:
    """Collect solver-specific authored attributes using USD namespace queries."""

    out: dict[str, Any] = {}
    if prim is None or Usd is None:
        return out

    for ns in namespaces:
        for prop in prim.GetAuthoredPropertiesInNamespace(ns):
            if Usd is not None and isinstance(prop, Usd.Attribute) and prop.IsValid() and prop.HasAuthoredValue():
                out[prop.GetName()] = prop.Get()

    return out


class SchemaResolverNewton(SchemaResolver):
    name: ClassVar[str] = "newton"
    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {
            "time_step": [Attribute("newton:timeStep", 0.002)],
            "max_solver_iterations": [Attribute("newton:maxSolverIterations", 5)],
            "enable_gravity": [Attribute("newton:enableGravity", True)],
            "rigid_contact_margin": [Attribute("newton:rigidContactMargin", 0.0)],
        },
        PrimType.JOINT: {
            "armature": [Attribute("newton:armature", 1.0e-2)],
            "friction": [Attribute("newton:friction", 0.0)],
            "limit_linear_ke": [Attribute("newton:linear:limitStiffness", 1.0e4)],
            "limit_angular_ke": [Attribute("newton:angular:limitStiffness", 1.0e4)],
            "limit_rotX_ke": [Attribute("newton:rotX:limitStiffness", 1.0e4)],
            "limit_rotY_ke": [Attribute("newton:rotY:limitStiffness", 1.0e4)],
            "limit_rotZ_ke": [Attribute("newton:rotZ:limitStiffness", 1.0e4)],
            "limit_linear_kd": [Attribute("newton:linear:limitDamping", 1.0e1)],
            "limit_angular_kd": [Attribute("newton:angular:limitDamping", 1.0e1)],
            "limit_rotX_kd": [Attribute("newton:rotX:limitDamping", 1.0e1)],
            "limit_rotY_kd": [Attribute("newton:rotY:limitDamping", 1.0e1)],
            "limit_rotZ_kd": [Attribute("newton:rotZ:limitDamping", 1.0e1)],
            "angular_position": [Attribute("newton:angular:position", 0.0)],
            "linear_position": [Attribute("newton:linear:position", 0.0)],
            "rotX_position": [Attribute("newton:rotX:position", 0.0)],
            "rotY_position": [Attribute("newton:rotY:position", 0.0)],
            "rotZ_position": [Attribute("newton:rotZ:position", 0.0)],
            "angular_velocity": [Attribute("newton:angular:velocity", 0.0)],
            "linear_velocity": [Attribute("newton:linear:velocity", 0.0)],
            "rotX_velocity": [Attribute("newton:rotX:velocity", 0.0)],
            "rotY_velocity": [Attribute("newton:rotY:velocity", 0.0)],
            "rotZ_velocity": [Attribute("newton:rotZ:velocity", 0.0)],
        },
        PrimType.SHAPE: {
            "mesh_hull_vertex_limit": [Attribute("newton:hullVertexLimit", -1)],
            # Use ShapeConfig.thickness default for contact margin
            "rigid_contact_margin": [Attribute("newton:rigidContactMargin", 1.0e-5)],
        },
        PrimType.BODY: {
            "rigid_body_linear_damping": [Attribute("newton:damping", 0.0)],
        },
        PrimType.MATERIAL: {
            "priority": [Attribute("newton:priority", 0)],
            "weight": [Attribute("newton:weight", 1.0)],
            "stiffness": [Attribute("newton:stiffness", 1.0e5)],
            "damping": [Attribute("newton:damping", 1000.0)],
        },
        PrimType.ACTUATOR: {
            # Mirror MuJoCo actuator defaults when applicable
            "ctrl_low": [Attribute("newton:ctrlRange:low", 0.0)],
            "ctrl_high": [Attribute("newton:ctrlRange:high", 0.0)],
            "force_low": [Attribute("newton:forceRange:low", 0.0)],
            "force_high": [Attribute("newton:forceRange:high", 0.0)],
            "act_low": [Attribute("newton:actRange:low", 0.0)],
            "act_high": [Attribute("newton:actRange:high", 0.0)],
            "length_low": [Attribute("newton:lengthRange:low", 0.0)],
            "length_high": [Attribute("newton:lengthRange:high", 0.0)],
            "gainPrm": [Attribute("newton:gainPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "gainType": [Attribute("newton:gainType", "fixed")],
            "biasPrm": [Attribute("newton:biasPrm", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "biasType": [Attribute("newton:biasType", "none")],
            "dynPrm": [Attribute("newton:dynPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "dynType": [Attribute("newton:dynType", "none")],
            # The following have no MuJoCo counterpart; keep unspecified defaults
            "speedTorqueGradient": [Attribute("newton:speedTorqueGradient", None)],
            "torqueSpeedGradient": [Attribute("newton:torqueSpeedGradient", None)],
            "maxVelocity": [Attribute("newton:maxVelocity", None)],
            "gear": [Attribute("newton:gear", [1, 0, 0, 0, 0, 0])],
        },
    }


class SchemaResolverPhysx(SchemaResolver):
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

    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {
            "time_step": [
                Attribute("physxScene:timeStepsPerSecond", 60, lambda hz: (1.0 / hz) if (hz and hz > 0) else None)
            ],
            "max_solver_iterations": [Attribute("physxScene:maxVelocityIterationCount", 255)],
            "enable_gravity": [Attribute("physxRigidBody:disableGravity", False, lambda value: not value)],
            "rigid_contact_margin": [Attribute("physxScene:contactOffset", 0.0)],
        },
        PrimType.JOINT: {
            "armature": [Attribute("physxJoint:armature", 0.0)],
            # Per-axis linear limit aliases
            "limit_transX_ke": [Attribute("physxLimit:linear:stiffness", 0.0)],
            "limit_transY_ke": [Attribute("physxLimit:linear:stiffness", 0.0)],
            "limit_transZ_ke": [Attribute("physxLimit:linear:stiffness", 0.0)],
            "limit_transX_kd": [Attribute("physxLimit:linear:damping", 0.0)],
            "limit_transY_kd": [Attribute("physxLimit:linear:damping", 0.0)],
            "limit_transZ_kd": [Attribute("physxLimit:linear:damping", 0.0)],
            "limit_linear_ke": [Attribute("physxLimit:linear:stiffness", 0.0)],
            "limit_angular_ke": [Attribute("physxLimit:angular:stiffness", 0.0)],
            "limit_rotX_ke": [Attribute("physxLimit:rotX:stiffness", 0.0)],
            "limit_rotY_ke": [Attribute("physxLimit:rotY:stiffness", 0.0)],
            "limit_rotZ_ke": [Attribute("physxLimit:rotZ:stiffness", 0.0)],
            "limit_linear_kd": [Attribute("physxLimit:linear:damping", 0.0)],
            "limit_angular_kd": [Attribute("physxLimit:angular:damping", 0.0)],
            "limit_rotX_kd": [Attribute("physxLimit:rotX:damping", 0.0)],
            "limit_rotY_kd": [Attribute("physxLimit:rotY:damping", 0.0)],
            "limit_rotZ_kd": [Attribute("physxLimit:rotZ:damping", 0.0)],
            "angular_position": [Attribute("state:angular:physics:position", 0.0)],
            "linear_position": [Attribute("state:linear:physics:position", 0.0)],
            "rotX_position": [Attribute("state:rotX:physics:position", 0.0)],
            "rotY_position": [Attribute("state:rotY:physics:position", 0.0)],
            "rotZ_position": [Attribute("state:rotZ:physics:position", 0.0)],
            "angular_velocity": [Attribute("state:angular:physics:velocity", 0.0)],
            "linear_velocity": [Attribute("state:linear:physics:velocity", 0.0)],
            "rotX_velocity": [Attribute("state:rotX:physics:velocity", 0.0)],
            "rotY_velocity": [Attribute("state:rotY:physics:velocity", 0.0)],
            "rotZ_velocity": [Attribute("state:rotZ:physics:velocity", 0.0)],
        },
        PrimType.SHAPE: {
            # Mesh hull vertex limit
            "mesh_hull_vertex_limit": [Attribute("physxConvexHullCollision:hullVertexLimit", 64)],
            # Collision contact offset
            "rigid_contact_margin": [Attribute("physxCollision:contactOffset", float("-inf"))],
        },
        PrimType.MATERIAL: {
            "stiffness": [Attribute("physxMaterial:compliantContactStiffness", 0.0)],
            "damping": [Attribute("physxMaterial:compliantContactDamping", 0.0)],
        },
        PrimType.BODY: {
            # Rigid body damping
            "rigid_body_linear_damping": [Attribute("physxRigidBody:linearDamping", 0.0)],
            "rigid_body_angular_damping": [Attribute("physxRigidBody:angularDamping", 0.05)],
        },
    }


def _solref_to_stiffness(solref):
    """Convert MuJoCo solref (timeconst, dampratio) to internal stiffness.

    Standard mode (timeconst > 0): k = 1 / (timeconst^2)
    Direct mode (both negative): k = -timeconst (encodes -stiffness directly)
    """
    try:
        timeconst = float(solref[0])
        dampratio = float(solref[1])
    except (TypeError, ValueError, IndexError):
        return None

    # Direct mode: both negative → solref encodes (-stiffness, -damping)
    if timeconst < 0.0 and dampratio < 0.0:
        return -timeconst

    # Standard mode: k = 1 / (timeconst^2)
    if timeconst <= 0.0:
        return None

    return 1.0 / (timeconst * timeconst)


def _solref_to_damping(solref):
    """Convert MuJoCo solref (timeconst, dampratio) to internal damping.

    Standard mode (both positive): b = 2 * dampratio / timeconst
    Direct mode (both negative): b = -dampratio (encodes -damping directly)
    """
    try:
        timeconst = float(solref[0])
        dampratio = float(solref[1])
    except (TypeError, ValueError, IndexError):
        return None

    # Direct mode: both negative → solref encodes (-stiffness, -damping)
    if timeconst < 0.0 and dampratio < 0.0:
        return -dampratio

    # Standard mode: b = 2 * dampratio / timeconst
    if timeconst <= 0.0 or dampratio <= 0.0:
        return None

    return (2.0 * dampratio) / timeconst


class SchemaResolverMjc(SchemaResolver):
    name: ClassVar[str] = "mjc"

    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {
            "time_step": [Attribute("mjc:option:timestep", 0.002)],
            "max_solver_iterations": [Attribute("mjc:option:iterations", 100)],
            "enable_gravity": [Attribute("mjc:flag:gravity", True)],
            "rigid_contact_margin": [Attribute("mjc:option:o_margin", 0.0)],
        },
        PrimType.JOINT: {
            "armature": [Attribute("mjc:armature", 0.0)],
            "friction": [Attribute("mjc:frictionloss", 0.0)],
            # Per-axis linear aliases mapped to solref
            "limit_transX_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_transY_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_transZ_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_transX_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_transY_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_transZ_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_linear_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_angular_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_rotX_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_rotY_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_rotZ_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_linear_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_angular_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_rotX_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_rotY_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_rotZ_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
        },
        PrimType.SHAPE: {
            # Mesh
            "mesh_hull_vertex_limit": [Attribute("mjc:maxhullvert", -1)],
            # Collisions
            "rigid_contact_margin": [Attribute("mjc:margin", 0.0)],
        },
        PrimType.MATERIAL: {
            # Materials and contact models
            "priority": [Attribute("mjc:priority", 0)],
            "weight": [Attribute("mjc:solmix", 1.0)],
            "stiffness": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "damping": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
        },
        PrimType.BODY: {
            # Rigid body / joint domain
            "rigid_body_linear_damping": [Attribute("mjc:damping", 0.0)],
        },
        PrimType.ACTUATOR: {
            # Actuators
            "ctrl_low": [Attribute("mjc:ctrlRange:min", 0.0)],
            "ctrl_high": [Attribute("mjc:ctrlRange:max", 0.0)],
            "force_low": [Attribute("mjc:forceRange:min", 0.0)],
            "force_high": [Attribute("mjc:forceRange:max", 0.0)],
            "act_low": [Attribute("mjc:actRange:min", 0.0)],
            "act_high": [Attribute("mjc:actRange:max", 0.0)],
            "length_low": [Attribute("mjc:lengthRange:min", 0.0)],
            "length_high": [Attribute("mjc:lengthRange:max", 0.0)],
            "gainPrm": [Attribute("mjc:gainPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "gainType": [Attribute("mjc:gainType", "fixed")],
            "biasPrm": [Attribute("mjc:biasPrm", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "biasType": [Attribute("mjc:biasType", "none")],
            "dynPrm": [Attribute("mjc:dynPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "dynType": [Attribute("mjc:dynType", "none")],
            "gear": [Attribute("mjc:gear", [1, 0, 0, 0, 0, 0])],
        },
    }


class _ResolverManager:
    def __init__(self, resolvers: list[SchemaResolver], collect_solver_attrs: bool = True):
        """
        Initialize resolver with resolver instances in priority order.

        Args:
            resolvers: List of instantiated resolvers in priority order
        """
        # Use provided resolver instances directly
        self.resolvers = list(resolvers) if resolvers is not None else []
        self._collect_solver_attrs = bool(collect_solver_attrs)

        # Dictionary to accumulate solver-specific attributes as prims are encountered
        # Pre-initialize maps for each configured resolver
        self.solver_specific_attrs: dict[str, dict[str, dict[str, Any]]] = {r.name: {} for r in self.resolvers}

        # accumulator for declared custom attributes (declaration-first pattern)
        # Key format: "namespace:attr_name" or "attr_name" for default namespace
        self._custom_attributes: dict[str, CustomAttribute] = {}

        # Flag to track if declarations have been parsed from PhysicsScene
        self._declarations_parsed = False

    def _collect_on_first_use(self, resolver: SchemaResolver, prim) -> None:
        """Collect and store solver-specific attributes for this resolver/prim on first use."""
        if prim is None:
            return
        if not self._collect_solver_attrs:
            return
        prim_path = str(prim.GetPath())
        if prim_path in self.solver_specific_attrs[resolver.name]:
            return
        attrs = resolver.collect_prim_solver_attrs(prim)
        if attrs:
            self.solver_specific_attrs[resolver.name][prim_path] = attrs

    def _parse_custom_attr_name(self, name: str) -> tuple[str | None, str] | None:
        """
        Parse custom attribute names in the format 'newton:namespace:attr_name' or 'newton:attr_name'.

        Returns:
            Tuple of (namespace, attr_name) where namespace can be None for default namespace,
            or None if the name doesn't match the expected format.
        """
        parts = name.split(":")
        if len(parts) < 2 or parts[0] != "newton":
            return None

        if len(parts) == 2:
            # newton:attr_name (default namespace)
            return None, parts[1]
        elif len(parts) == 3:
            # newton:namespace:attr_name
            return parts[1], parts[2]
        else:
            # Invalid format
            return None

    def parse_custom_attribute_declarations(self, physics_scene_prim) -> None:
        """
        Parse custom attribute declarations from PhysicsScene prim.

        Supports metadata format with assignment and frequency specified as customData:
           custom float newton:namespace:attr_name = 150.0 (
               customData = {
                   string assignment = "control"
                   string frequency = "joint_dof"
               }
           )
        Args:
            physics_scene_prim: USD PhysicsScene prim to parse declarations from
        """
        if physics_scene_prim is None or self._declarations_parsed:
            return

        self._declarations_parsed = True

        for attr in physics_scene_prim.GetAttributes():
            attr_name = attr.GetName()
            parsed = self._parse_custom_attr_name(attr_name)
            if not parsed:
                continue

            namespace, local_name = parsed
            default_value = attr.Get()

            # Try to read customData for assignment and frequency
            assignment_meta = attr.GetCustomDataByKey("assignment")
            frequency_meta = attr.GetCustomDataByKey("frequency")

            if assignment_meta and frequency_meta:
                # Metadata format
                try:
                    assignment_val = ModelAttributeAssignment[assignment_meta.upper()]
                    frequency_val = ModelAttributeFrequency[frequency_meta.upper()]
                except KeyError:
                    print(
                        f"Warning: Attribute '{attr_name}' has invalid assignment or frequency in customData. Skipping."
                    )
                    continue
            else:
                # No metadata found - skip with warning
                print(
                    f"Warning: Attribute '{attr_name}' is missing required customData (assignment and frequency). Skipping."
                )
                continue

            # Infer dtype from default value
            converted_value = self._usd_to_wp(default_value)
            dtype = self._infer_wp_dtype(converted_value)

            # Create full key: "namespace:attr_name" or "attr_name"
            full_key = f"{namespace}:{local_name}" if namespace else local_name

            # Create custom attribute specification
            # Note: name should be the local name, namespace is stored separately
            custom_attr = CustomAttribute(
                assignment=assignment_val,
                frequency=frequency_val,
                name=local_name,
                dtype=dtype,
                default=converted_value,
                namespace=namespace,
            )

            self._custom_attributes[full_key] = custom_attr

    def _usd_to_wp(self, v: Any):
        """Convert USD types to Warp-friendly representations."""
        try:
            # Handle Gf.Quat[f/d] → wp.quat(x, y, z, w) normalized
            if hasattr(v, "real") and hasattr(v, "imaginary"):
                try:
                    return wp.normalize(wp.quat(*v.imaginary, v.real))
                except (TypeError, ValueError, AttributeError):
                    _w.warn("Failed converting USD quat to wp.quat; returning original value", stacklevel=2)
        except (AttributeError, RuntimeError):
            _w.warn("Unexpected attribute access while converting USD value; returning original value", stacklevel=2)
        return v

    def _infer_wp_dtype(self, v: Any):
        """Heuristic mapping from USD value to Warp dtype."""
        try:
            # Check for quat first (before generic length checks)
            if hasattr(v, "real") and hasattr(v, "imaginary"):
                return wp.quat
            # wp.quat-like (object with x,y,z,w after conversion)
            if all(hasattr(v, c) for c in ("x", "y", "z", "w")):
                return wp.quat
            # Vector3-like
            if hasattr(v, "__len__") and len(v) == 3:
                return wp.vec3
            # Vector2-like
            if hasattr(v, "__len__") and len(v) == 2:
                return wp.vec2
            # Vector4-like (but not quat)
            if hasattr(v, "__len__") and len(v) == 4:
                return wp.vec4
        except (TypeError, AttributeError):
            # fallthrough to scalar checks
            pass
        if isinstance(v, bool):
            return wp.bool
        if isinstance(v, int):
            return wp.int32
        # default to float32 for scalars
        return wp.float32

    def get_custom_attributes_for_prim(self, prim, expected_frequency: ModelAttributeFrequency) -> dict[str, Any]:
        """
        Extract custom attributes from a prim that match the expected frequency.
        Providing the frequency is an additional check to ensure the attributes are used with the correct prim type.

        This method reads custom attributes from a USD prim, validates they are
        declared and match the expected frequency, then returns them as a
        dictionary.

        Args:
            prim: USD prim to extract attributes from
            expected_frequency: Expected frequency for the prim type

        Returns:
            Dictionary of custom attribute values keyed by full name
            (namespace:attr_name or attr_name)
        """
        if prim is None:
            return {}

        result = {}

        for attr in prim.GetAttributes():
            attr_name = attr.GetName()
            parsed = self._parse_custom_attr_name(attr_name)
            if not parsed:
                continue

            namespace, local_name = parsed
            full_key = f"{namespace}:{local_name}" if namespace else local_name

            # Check if this attribute was declared
            custom_attr = self._custom_attributes.get(full_key)
            if custom_attr is None:
                # Attribute not declared - skip it since we only collect attributes that are declared
                continue

            # Verify frequency matches (use frequency from declaration)
            if custom_attr.frequency != expected_frequency:
                continue

            # Get the value and convert it
            value = attr.Get()
            converted_value = self._usd_to_wp(value)

            result[full_key] = converted_value

        return result

    def get_value(self, prim, prim_type: PrimType, key: str, default: Any = None) -> Any:
        """
        Resolve value using engine priority, with layered fallbacks:

        1) First authored value found in resolver order (highest priority first)
        2) If none authored, use the provided 'default' argument if not None
        3) If still None, use the first non-None mapping default from resolvers in priority order

        Args:
            prim: USD prim to query (for scene prim_type, this should be scene_prim)
            prim_type: Prim type (PrimType enum)
            key: Attribute key within the prim type
            default: Default value if not found

        Returns:
            Resolved value according to the precedence above.
        """
        # 1) Authored value by engine priority
        for r in self.resolvers:
            got = r.get_value(prim, prim_type, key)
            if got is not None:
                val, _usd_attr = got
                if val is not None:
                    if self._collect_solver_attrs:
                        self._collect_on_first_use(r, prim)
                    return val

        # 2) Caller-provided default, if any
        if default is not None:
            return default

        # 3) Solver mapping defaults in priority order
        for resolver in self.resolvers:
            specs = resolver.mapping.get(prim_type, {}).get(key, []) if hasattr(resolver, "mapping") else []
            for spec in specs:
                d = getattr(spec, "default", None)
                if d is not None:
                    return d

        # Nothing found
        try:
            prim_path = str(prim.GetPath()) if prim is not None else "<None>"
        except (AttributeError, RuntimeError):
            prim_path = "<invalid>"
        print(
            f"Error: Cannot resolve value for '{prim_type.name.lower()}:{key}' on prim '{prim_path}'; "
            f"no authored value, no explicit default, and no solver mapping default."
        )
        return None

    def collect_prim_solver_attrs(self, prim) -> None:
        """
        Collect and accumulate solver-specific attributes for a single prim.

        Args:
            prim: USD prim to collect solver attributes from
        """
        if prim is None:
            return

        prim_path = str(prim.GetPath())

        if not self._collect_solver_attrs:
            return
        for resolver in self.resolvers:
            # only collect if we haven't seen this prim for this resolver
            if prim_path not in self.solver_specific_attrs[resolver.name]:
                attrs = resolver.collect_prim_solver_attrs(prim)
                if attrs:
                    self.solver_specific_attrs[resolver.name][prim_path] = attrs

    def get_solver_specific_attrs(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Get the accumulated solver-specific attributes.

        Returns:
            Dictionary with structure: solver_name -> prim_path -> {attr_name: attr_value}
            e.g., {"mjc": {"/World/Cube": {"mjc:option:timestep": 0.01}}}
        """
        return self.solver_specific_attrs.copy()

    def get_custom_attribute_declarations(
        self,
    ) -> dict[str, CustomAttribute]:
        """
        Get custom attribute declarations parsed from PhysicsScene.

        Returns:
            Dictionary keyed by full attribute name with CustomAttribute values.
        """
        return self._custom_attributes.copy()
