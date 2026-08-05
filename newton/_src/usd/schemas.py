# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Concrete USD schema resolvers used by :mod:`newton.usd`."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from ..core.types import override
from ..usd.schema_resolver import PrimType, SchemaResolver
from . import utils as usd

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim.builder import ModelBuilder


SchemaAttribute = SchemaResolver.SchemaAttribute


def _physx_gap_from_prim(prim: Usd.Prim) -> float | None:
    """Compute Newton gap from PhysX: contactOffset - restOffset [m].

    Returns None if either attribute is missing or -inf (PhysX uses -inf for "engine default").
    Only when both are finite do we compute a concrete gap.
    """
    contact_offset = usd.get_attribute(prim, "physxCollision:contactOffset")
    rest_offset = usd.get_attribute(prim, "physxCollision:restOffset")
    if contact_offset is None or rest_offset is None:
        return None
    inf = float("-inf")
    if contact_offset == inf or rest_offset == inf:
        return None
    return float(contact_offset) - float(rest_offset)


def _newton_legacy_contact_attr(legacy_name: str, material_attr: str):
    """Return a getter that reads a legacy contact custom attr with a deprecation warning."""

    def _getter(prim: Usd.Prim) -> float | None:
        value = usd.get_attribute(prim, legacy_name)
        if value is not None:
            warnings.warn(
                f"'{legacy_name}' on shape prim is deprecated; "
                f"author '{material_attr}' on the bound NewtonMaterialAPI material instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            return float(value)
        return None

    return _getter


def _newton_non_schema_joint_state_attr(attr_name: str):
    """Return a getter that reads a non-schema Newton joint state attr with a UserWarning."""

    def _getter(prim: Usd.Prim) -> float | None:
        value = usd.get_attribute(prim, attr_name)
        if value is not None:
            warnings.warn(
                f"'{attr_name}' on joint prim is a non-schema attribute. "
                f"Please file an issue at https://github.com/newton-physics/newton/issues "
                f"describing your use case so we can provide a supported alternative.",
                UserWarning,
                stacklevel=4,
            )
            return float(value)
        return None

    return _getter


def _newton_legacy_joint_limit_attr(legacy_name: str, schema_attr: str):
    """Return a getter that reads a legacy per-DOF limit attr with a deprecation warning."""

    def _getter(prim: Usd.Prim) -> float | None:
        value = usd.get_attribute(prim, legacy_name)
        if value is not None:
            warnings.warn(
                f"'{legacy_name}' on joint prim is deprecated; use '{schema_attr}' instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            return float(value)
        return None

    return _getter


def _mjc_legacy_material_solref(converter, material_attr: str):
    """Return a getter that reads legacy mjc:solref on a material prim with a deprecation warning."""

    def _getter(prim: Usd.Prim) -> float | None:
        value = usd.get_attribute(prim, "mjc:solref")
        if value is not None:
            warnings.warn(
                f"'mjc:solref' on material prim is deprecated; author '{material_attr}' on the "
                f"bound NewtonMaterialAPI material, or use per-shape 'mjc:solref' (MjcGeomAPI) "
                f"instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            return converter(value)
        return None

    return _getter


class SchemaResolverNewton(SchemaResolver):
    """Schema resolver for Newton-authored USD attributes.

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
            "armature": SchemaAttribute("newton:armature", 0.0),
            "damping": SchemaAttribute("newton:damping", None),
            "friction": SchemaAttribute("newton:friction", 0.0),
            "limit_ke": SchemaAttribute("newton:limitStiffness", None),
            "limit_kd": SchemaAttribute("newton:limitDamping", None),
            "velocity_limit": SchemaAttribute("newton:velocityLimit", float("inf")),
            # Non-schema per-DOF limit attrs (deprecated; use newton:limitStiffness / newton:limitDamping)
            "limit_linear_ke": SchemaAttribute(
                "newton:linear:limitStiffness",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr(
                    "newton:linear:limitStiffness", "newton:limitStiffness"
                ),
            ),
            "limit_angular_ke": SchemaAttribute(
                "newton:angular:limitStiffness",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr(
                    "newton:angular:limitStiffness", "newton:limitStiffness"
                ),
            ),
            "limit_rotX_ke": SchemaAttribute(
                "newton:rotX:limitStiffness",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr("newton:rotX:limitStiffness", "newton:limitStiffness"),
            ),
            "limit_rotY_ke": SchemaAttribute(
                "newton:rotY:limitStiffness",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr("newton:rotY:limitStiffness", "newton:limitStiffness"),
            ),
            "limit_rotZ_ke": SchemaAttribute(
                "newton:rotZ:limitStiffness",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr("newton:rotZ:limitStiffness", "newton:limitStiffness"),
            ),
            "limit_linear_kd": SchemaAttribute(
                "newton:linear:limitDamping",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr("newton:linear:limitDamping", "newton:limitDamping"),
            ),
            "limit_angular_kd": SchemaAttribute(
                "newton:angular:limitDamping",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr("newton:angular:limitDamping", "newton:limitDamping"),
            ),
            "limit_rotX_kd": SchemaAttribute(
                "newton:rotX:limitDamping",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr("newton:rotX:limitDamping", "newton:limitDamping"),
            ),
            "limit_rotY_kd": SchemaAttribute(
                "newton:rotY:limitDamping",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr("newton:rotY:limitDamping", "newton:limitDamping"),
            ),
            "limit_rotZ_kd": SchemaAttribute(
                "newton:rotZ:limitDamping",
                None,
                usd_value_getter=_newton_legacy_joint_limit_attr("newton:rotZ:limitDamping", "newton:limitDamping"),
            ),
            # Non-schema per-DOF initial state attrs
            "angular_position": SchemaAttribute(
                "newton:angular:position",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:angular:position"),
            ),
            "linear_position": SchemaAttribute(
                "newton:linear:position",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:linear:position"),
            ),
            "rotX_position": SchemaAttribute(
                "newton:rotX:position",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:rotX:position"),
            ),
            "rotY_position": SchemaAttribute(
                "newton:rotY:position",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:rotY:position"),
            ),
            "rotZ_position": SchemaAttribute(
                "newton:rotZ:position",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:rotZ:position"),
            ),
            "angular_velocity": SchemaAttribute(
                "newton:angular:velocity",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:angular:velocity"),
            ),
            "linear_velocity": SchemaAttribute(
                "newton:linear:velocity",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:linear:velocity"),
            ),
            "rotX_velocity": SchemaAttribute(
                "newton:rotX:velocity",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:rotX:velocity"),
            ),
            "rotY_velocity": SchemaAttribute(
                "newton:rotY:velocity",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:rotY:velocity"),
            ),
            "rotZ_velocity": SchemaAttribute(
                "newton:rotZ:velocity",
                0.0,
                usd_value_getter=_newton_non_schema_joint_state_attr("newton:rotZ:velocity"),
            ),
        },
        PrimType.SHAPE: {
            # Mesh
            "max_hull_vertices": SchemaAttribute("newton:maxHullVertices", -1),
            # Collisions: newton margin == newton:contactMargin, newton gap == newton:contactGap
            "margin": SchemaAttribute("newton:contactMargin", 0.0),
            "gap": SchemaAttribute("newton:contactGap", float("-inf")),
            # Legacy per-shape contact attrs (deprecated; use NewtonMaterialAPI instead)
            "ke": SchemaAttribute(
                "newton:contact_ke",
                None,
                usd_value_getter=_newton_legacy_contact_attr("newton:contact_ke", "newton:contactStiffness"),
            ),
            "kd": SchemaAttribute(
                "newton:contact_kd",
                None,
                usd_value_getter=_newton_legacy_contact_attr("newton:contact_kd", "newton:contactDamping"),
            ),
            "kf": SchemaAttribute(
                "newton:contact_kf",
                None,
                usd_value_getter=_newton_legacy_contact_attr("newton:contact_kf", "newton:contactFrictionGain"),
            ),
            "ka": SchemaAttribute(
                "newton:contact_ka",
                None,
                usd_value_getter=_newton_legacy_contact_attr("newton:contact_ka", "newton:contactAdhesion"),
            ),
            # SDF configuration — from NewtonSDFCollisionAPI. `-inf` is the
            # "unset" sentinel (same convention as gap / shell_thickness above).
            "sdf_max_resolution": SchemaAttribute("newton:sdfMaxResolution", float("-inf")),
            "sdf_narrow_band_inner": SchemaAttribute("newton:sdfNarrowBandInner", float("-inf")),
            "sdf_narrow_band_outer": SchemaAttribute("newton:sdfNarrowBandOuter", float("-inf")),
            "sdf_target_voxel_size": SchemaAttribute("newton:sdfTargetVoxelSize", float("-inf")),
            "sdf_texture_format": SchemaAttribute("newton:sdfTextureFormat", None),
            "sdf_padding": SchemaAttribute("newton:sdfPadding", float("-inf")),
            # Hydroelastic contacts — folded into NewtonSDFCollisionAPI
            "hydroelastic_enabled": SchemaAttribute("newton:hydroelasticEnabled", None),
            "kh": SchemaAttribute("newton:hydroelasticStiffness", float("-inf")),
            # Mass model
            "mass_model": SchemaAttribute("newton:massModel", "solid"),
            "shell_thickness": SchemaAttribute("newton:shellThickness", float("-inf")),
        },
        PrimType.BODY: {},
        PrimType.ARTICULATION: {
            "self_collision_enabled": SchemaAttribute("newton:selfCollisionEnabled", True),
        },
        PrimType.MATERIAL: {
            "mu_torsional": SchemaAttribute("newton:torsionalFriction", 0.25),
            "mu_rolling": SchemaAttribute("newton:rollingFriction", 0.0005),
            "ke": SchemaAttribute("newton:contactStiffness", None),
            "kd": SchemaAttribute("newton:contactDamping", None),
            "kf": SchemaAttribute("newton:contactFrictionGain", None),
            "ka": SchemaAttribute("newton:contactAdhesion", None),
        },
        PrimType.ACTUATOR: {},
    }


class SchemaResolverPhysx(SchemaResolver):
    """Schema resolver for PhysX USD attributes.

    For deformables this resolver only enables reading the proposal-shaped material
    attributes under the ``omniphysics:`` / ``physxDeformableBody:`` namespaces off bound
    materials. It does not translate PhysX/OmniPhysics applied schemas or asset structure,
    so native OmniPhysics deformable assets are not imported as deformables.
    """

    name: ClassVar[str] = "physx"
    # Deformable material/geometry vendor namespaces (AOUSD proposal). The public
    # schema authors under ``physics:``; these carry the same parameters in existing
    # content. Kept separate from extra_attr_namespaces so they are only consulted
    # for deformable attributes, not generic rigid-body parsing.
    deformable_attr_namespaces: ClassVar[list[str]] = ["omniphysics", "physxDeformableBody"]
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
            "velocity_limit": SchemaAttribute("physxJoint:maxJointVelocity", None),
            "limit_transX_ke": SchemaAttribute("physxLimit:transX:stiffness", 0.0),
            "limit_transY_ke": SchemaAttribute("physxLimit:transY:stiffness", 0.0),
            "limit_transZ_ke": SchemaAttribute("physxLimit:transZ:stiffness", 0.0),
            "limit_transX_kd": SchemaAttribute("physxLimit:transX:damping", 0.0),
            "limit_transY_kd": SchemaAttribute("physxLimit:transY:damping", 0.0),
            "limit_transZ_kd": SchemaAttribute("physxLimit:transZ:damping", 0.0),
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
            # Collisions: newton margin == physx restOffset, newton gap == physx contactOffset - restOffset.
            # PhysX uses -inf to mean "engine default"; treat as unset (None).
            "margin": SchemaAttribute(
                "physxCollision:restOffset", 0.0, lambda v: None if v == float("-inf") else float(v)
            ),
            "gap": SchemaAttribute(
                "physxCollision:contactOffset",
                float("-inf"),
                usd_value_getter=_physx_gap_from_prim,
                attribute_names=("physxCollision:contactOffset", "physxCollision:restOffset"),
            ),
        },
        PrimType.MATERIAL: {
            "ke": SchemaAttribute("physxMaterial:compliantContactStiffness", None),
            "kd": SchemaAttribute("physxMaterial:compliantContactDamping", None),
        },
        PrimType.BODY: {
            # Rigid body damping
            "rigid_body_linear_damping": SchemaAttribute("physxRigidBody:linearDamping", 0.0),
            "rigid_body_angular_damping": SchemaAttribute("physxRigidBody:angularDamping", 0.05),
        },
        PrimType.ARTICULATION: {
            "self_collision_enabled": SchemaAttribute("physxArticulation:enabledSelfCollisions", True),
        },
    }


def solref_to_stiffness_damping(solref: Sequence[float] | None) -> tuple[float | None, float | None]:
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
    if solref is None:
        return None, None

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


def solref_to_stiffness(solref: Sequence[float] | None) -> float | None:
    """Convert MuJoCo solref (timeconst, dampratio) to internal stiffness.

    Standard mode (timeconst > 0): k = 1 / (timeconst^2 * dampratio^2)
    Direct mode (both negative): k = -timeconst (encodes -stiffness directly)
    """
    stiffness, _ = solref_to_stiffness_damping(solref)
    return stiffness


def solref_to_damping(solref: Sequence[float] | None) -> float | None:
    """Convert MuJoCo solref (timeconst, dampratio) to internal damping.

    Standard mode (both positive): b = 2 / timeconst
    Direct mode (both negative): b = -dampratio (encodes -damping directly)
    """
    _, damping = solref_to_stiffness_damping(solref)
    return damping


# `parse_usd` divides revolute and D6-angular `limit_ke` / `limit_kd` by
# DegreesToRadian (= pi/180) on the assumption that resolver-supplied gains are
# authored in per-degree units (UsdPhysics convention). MuJoCo's `mjc:solreflimit`
# always produces per-radian stiffness/damping (mjModel never expresses stiffness
# per-degree). Pre-multiplying here cancels the importer's later division so the
# per-radian value survives. Linear axes are unaffected and use the un-scaled
# helpers above.
_RAD_PER_DEG = math.pi / 180.0


def _solref_to_stiffness_per_rad(solref: Sequence[float] | None) -> float | None:
    s = solref_to_stiffness(solref)
    return s * _RAD_PER_DEG if s is not None else None


def _solref_to_damping_per_rad(solref: Sequence[float] | None) -> float | None:
    d = solref_to_damping(solref)
    return d * _RAD_PER_DEG if d is not None else None


class SchemaResolverMjc(SchemaResolver):
    """Schema resolver for MuJoCo USD attributes."""

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
            # Per-axis aliases mapped to solreflimit (MjcJointAPI authors joint limit solref here)
            "limit_transX_ke": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], solref_to_stiffness),
            "limit_transY_ke": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], solref_to_stiffness),
            "limit_transZ_ke": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], solref_to_stiffness),
            "limit_transX_kd": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], solref_to_damping),
            "limit_transY_kd": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], solref_to_damping),
            "limit_transZ_kd": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], solref_to_damping),
            "limit_linear_ke": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], solref_to_stiffness),
            "limit_angular_ke": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], _solref_to_stiffness_per_rad),
            "limit_rotX_ke": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], _solref_to_stiffness_per_rad),
            "limit_rotY_ke": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], _solref_to_stiffness_per_rad),
            "limit_rotZ_ke": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], _solref_to_stiffness_per_rad),
            "limit_linear_kd": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], solref_to_damping),
            "limit_angular_kd": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], _solref_to_damping_per_rad),
            "limit_rotX_kd": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], _solref_to_damping_per_rad),
            "limit_rotY_kd": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], _solref_to_damping_per_rad),
            "limit_rotZ_kd": SchemaAttribute("mjc:solreflimit", [0.02, 1.0], _solref_to_damping_per_rad),
        },
        PrimType.SHAPE: {
            # Mesh
            "max_hull_vertices": SchemaAttribute("mjc:maxhullvert", -1),
            # Collision margin/gap: identity mapping to shape_margin/shape_gap
            # under MuJoCo 3.9 (parse_usd handles legacy_margin_gap).
            "margin": SchemaAttribute("mjc:margin", 0.0),
            "gap": SchemaAttribute("mjc:gap", 0.0),
            # Mass model: mjc:shellinertia (bool) → "shell" / "solid"
            "mass_model": SchemaAttribute("mjc:shellinertia", False, lambda v: "shell" if v else "solid"),
            # mjc:solref also fills shape_material_ke/kd via the legacy lossy
            # conversion for back-compat with the convert_solref(ke, kd, 1, 1)
            # round-trip; raw solref is preserved in mujoco.solref. See
            # docs/solvers/mujoco.rst > "Shape-material contact stiffness
            # and damping".
            "ke": SchemaAttribute("mjc:solref", None, solref_to_stiffness),
            "kd": SchemaAttribute("mjc:solref", None, solref_to_damping),
        },
        PrimType.MATERIAL: {
            # Materials
            "mu_torsional": SchemaAttribute("mjc:torsionalfriction", 0.005),
            "mu_rolling": SchemaAttribute("mjc:rollingfriction", 0.0001),
            # Contact models
            "priority": SchemaAttribute("mjc:priority", 0),
            "weight": SchemaAttribute("mjc:solmix", 1.0),
            # See PrimType.SHAPE above for the mjc:solref → stiffness/damping
            # back-compat mirror.
            "ke": SchemaAttribute(
                "mjc:solref",
                None,
                usd_value_getter=_mjc_legacy_material_solref(solref_to_stiffness, "newton:contactStiffness"),
            ),
            "kd": SchemaAttribute(
                "mjc:solref",
                None,
                usd_value_getter=_mjc_legacy_material_solref(solref_to_damping, "newton:contactDamping"),
            ),
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

    @override
    def validate_custom_attributes(self, builder: ModelBuilder) -> None:
        """
        Validate that MuJoCo custom attributes have been registered on the builder.

        Users must call :meth:`newton.solvers.SolverMuJoCo.register_custom_attributes` before parsing
        USD files with this resolver.

        Raises:
            RuntimeError: If required MuJoCo custom attributes are not registered.
        """
        has_mujoco_attrs = any(attr.namespace == "mujoco" for attr in builder.custom_attributes.values())
        if not has_mujoco_attrs:
            raise RuntimeError(
                "MuJoCo custom attributes not registered. Call "
                + "SolverMuJoCo.register_custom_attributes(builder) before parsing "
                + "USD with SchemaResolverMjc."
            )
