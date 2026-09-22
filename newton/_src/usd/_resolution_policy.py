# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interpret USD properties for the importer."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..sim.enums import JointTargetMode
from ..solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_MJCF_DEFAULT, SOLREF_MODE_RAW
from . import utils as usd
from .schema_resolver import PrimType, SchemaResolver, SchemaResolverManager

if TYPE_CHECKING:
    from pxr import Usd, UsdPhysics

    from ..sim.builder import ModelBuilder


# Stiffness used for a hard joint limit (NewtonJointAPI newton:limitStiffness == +inf).
_HARD_LIMIT_KE = 1.0e8


def _resolve_newton_limit_ke(
    limit_ke: float | None,
    fallback: float,
    fallback_source: str,
    builder_default: float,
) -> tuple[float, str]:
    """Resolve a NewtonJointAPI ``newton:limitStiffness`` value.

    ``limit_ke`` is ``None`` when the attribute is not authored, ``-inf`` when
    authored as the engine-default sentinel, ``+inf`` for a hard limit, or a
    finite stiffness value.

    ``fallback`` is the per-DOF stiffness resolved from lower-priority schemas
    (PhysX/MuJoCo).  ``builder_default`` is the ModelBuilder engine default.

    An explicit ``-inf`` takes precedence over the per-DOF fallback and selects
    the builder default so that a lower-priority schema cannot override an
    authored Newton sentinel.

    Returns (resolved_value, source) where source is ``"force"`` when Newton
    broadcast values are used, or the original ``fallback_source`` otherwise.
    """
    if limit_ke is None:
        return fallback, fallback_source
    if limit_ke == float("-inf"):
        return builder_default, "force"
    if limit_ke == float("inf"):
        return _HARD_LIMIT_KE, "force"
    return limit_ke, "force"


def _resolve_newton_limit_kd(
    limit_ke: float | None,
    limit_kd: float | None,
    fallback: float,
    fallback_source: str,
    builder_default: float,
) -> tuple[float, str]:
    """Resolve a NewtonJointAPI ``newton:limitDamping`` value.

    Hard limits (``limit_ke`` or ``limit_kd`` == ``+inf``) have no damping.
    An authored ``-inf`` selects the builder default (engine default), taking
    precedence over per-DOF fallbacks from lower-priority schemas.
    When neither Newton attribute is authored (``None``), the per-DOF ``fallback``
    from other resolvers is used.

    Returns (resolved_value, source) where source is ``"force"`` when Newton
    broadcast values are used, or the original ``fallback_source`` otherwise.
    """
    # Hard (rigid) limit: infinite ke or kd means no dissipation is needed.
    if limit_ke is not None and limit_ke == float("inf"):
        return 0.0, "force"
    if limit_kd is not None and limit_kd == float("inf"):
        return 0.0, "force"
    # Not authored → lower-priority per-DOF fallback.
    if limit_kd is None:
        return fallback, fallback_source
    # Authored -inf → builder default.
    if limit_kd == float("-inf"):
        return builder_default, "force"
    return limit_kd, "force"


@dataclass
class _DofParams:
    """Resolved limits, drive, and initial state for one revolute/prismatic DOF, in Newton units."""

    armature: float
    friction: float
    damping: float
    velocity_limit: float | None
    limit_lower: float
    limit_upper: float
    limit_ke: float
    limit_kd: float
    has_drive: bool
    target_pos: float
    target_vel: float
    target_ke: float
    target_kd: float
    effort_limit: float
    actuator_mode: JointTargetMode
    initial_position: float | None
    initial_velocity: float | None
    limit_solref_mode: int


def _shift_joint_limits_for_reference(dof: _DofParams, joint_custom_attrs: dict[str, Any]) -> None:
    """Convert absolute MuJoCo joint limits to Newton joint coordinates."""
    ref_key = "mujoco:dof_ref"
    if ref_key not in joint_custom_attrs:
        return
    ref = float(joint_custom_attrs[ref_key])
    dof.limit_lower -= ref
    dof.limit_upper -= ref


@dataclass
class _UsdJointProperties:
    """Resolve joint properties using defaults sampled at the start of one import."""

    resolver: SchemaResolverManager
    degrees_to_radian: float
    default_armature: float
    default_friction: float
    default_damping: float
    default_limit_ke: float
    default_limit_kd: float
    limit_gains_configured: bool
    """Whether the sampled limit gains differ from the builder's standard defaults."""
    mjc_resolver: SchemaResolver | None
    verbose: bool

    # Keep source tracking local until schema applicability and provenance are modeled globally (#3307).
    def _mjc_joint_limit_source(self, prim: Usd.Prim) -> Literal["mjc_authored", "mjc_default"] | None:
        if self.mjc_resolver is None:
            return None
        solreflimit_attr = prim.GetAttribute("mjc:solreflimit")
        if solreflimit_attr is not None and solreflimit_attr.HasAuthoredValue():
            return "mjc_authored"
        if prim and prim.IsValid() and usd.has_applied_api_schema(prim, "MjcJointAPI"):
            return "mjc_default"
        return None

    def resolve_joint_limit_gain(
        self, prim: Usd.Prim, key: str, builder_default: float
    ) -> tuple[float, Literal["force", "builder_default"]]:
        """Resolve a limit gain and report the semantics of its source."""
        for resolver in self.resolver.resolvers:
            if resolver.name == "mjc":
                continue

            spec = resolver.mapping.get(PrimType.JOINT, {}).get(key)
            if spec is None:
                continue

            authored_value = resolver.get_value(prim, PrimType.JOINT, key)
            if authored_value is not None:
                self.resolver._collect_on_first_use(resolver, prim)
                return authored_value, "force"

        return builder_default, "builder_default"

    def joint_limit_solref_mode(self, prim: Usd.Prim, ke_source: str, kd_source: str) -> int:
        """Choose MuJoCo limit-solref semantics from the resolved gain sources."""
        mjc_source = self._mjc_joint_limit_source(prim)
        if mjc_source is not None and self.mjc_resolver is not None:
            self.resolver._collect_on_first_use(self.mjc_resolver, prim)
        if mjc_source == "mjc_authored":
            return SOLREF_MODE_RAW
        if (
            mjc_source == "mjc_default"
            and ke_source == kd_source == "builder_default"
            and not self.limit_gains_configured
        ):
            return SOLREF_MODE_MJCF_DEFAULT
        return SOLREF_MODE_FORCE_SPACE

    def resolve_joint_damping(self, jp_prim: Usd.Prim) -> tuple[float, float]:
        """Resolve passive damping for linear and angular DOFs.

        MuJoCo authors SI damping per radian for angular DOFs, while Newton's
        regular USD damping mapping follows USD's per-degree convention.

        Returns:
            The linear and angular damping values in Newton units.
        """
        for resolver in self.resolver.resolvers:
            for key, angular_scale in (("damping", 1.0 / self.degrees_to_radian), ("damping_per_rad", 1.0)):
                damping = resolver.get_value(jp_prim, PrimType.JOINT, key)
                if damping is not None:
                    self.resolver._collect_on_first_use(resolver, jp_prim)
                    damping = float(damping)
                    return damping, damping * angular_scale
        return self.default_damping, self.default_damping

    def resolve_dof_params(
        self,
        jp_prim: Usd.Prim,
        jd: UsdPhysics.JointDesc,
        is_revolute: bool,
        *,
        joint_drive_gains_scaling: float,
        force_position_velocity_actuation: bool,
    ) -> _DofParams:
        """Resolve limits, drive, and initial state for one revolute/prismatic DOF.

        Returns values in Newton units (radians for revolute DOFs). ``velocity_limit``
        and the initial state stay ``None`` when unauthored so callers can apply their
        own fallbacks; drive targets/gains are zero when ``has_drive`` is False.
        """
        limit_gains_scaling = self.degrees_to_radian if is_revolute else 1.0
        armature = self.resolver.get_value(
            jp_prim, prim_type=PrimType.JOINT, key="armature", default=self.default_armature, verbose=self.verbose
        )
        friction = self.resolver.get_value(
            jp_prim, prim_type=PrimType.JOINT, key="friction", default=self.default_friction, verbose=self.verbose
        )
        linear_damping, angular_damping = self.resolve_joint_damping(jp_prim)
        damping = angular_damping if is_revolute else linear_damping
        velocity_limit = self.resolver.get_value(
            jp_prim, prim_type=PrimType.JOINT, key="velocity_limit", default=None, verbose=self.verbose
        )
        # NewtonJointAPI uses +inf for "unlimited"; treat it as the builder default below.
        if velocity_limit == float("inf"):
            velocity_limit = None
        newton_limit_ke = self.resolver.get_value(
            jp_prim, prim_type=PrimType.JOINT, key="limit_ke", default=None, verbose=self.verbose
        )
        newton_limit_kd = self.resolver.get_value(
            jp_prim, prim_type=PrimType.JOINT, key="limit_kd", default=None, verbose=self.verbose
        )
        limit_key = "limit_angular" if is_revolute else "limit_linear"
        fallback_limit_ke, limit_ke_source = self.resolve_joint_limit_gain(
            jp_prim,
            f"{limit_key}_ke",
            self.default_limit_ke * limit_gains_scaling,
        )
        fallback_limit_kd, limit_kd_source = self.resolve_joint_limit_gain(
            jp_prim,
            f"{limit_key}_kd",
            self.default_limit_kd * limit_gains_scaling,
        )
        limit_ke, limit_ke_source = _resolve_newton_limit_ke(
            newton_limit_ke, fallback_limit_ke, limit_ke_source, self.default_limit_ke * limit_gains_scaling
        )
        limit_kd, limit_kd_source = _resolve_newton_limit_kd(
            newton_limit_ke,
            newton_limit_kd,
            fallback_limit_kd,
            limit_kd_source,
            self.default_limit_kd * limit_gains_scaling,
        )
        limit_lower = jd.limit.lower
        limit_upper = jd.limit.upper

        has_drive = jd.drive.enabled
        target_pos = jd.drive.targetPosition if has_drive else 0.0
        target_vel = jd.drive.targetVelocity if has_drive else 0.0
        target_ke = jd.drive.stiffness if has_drive else 0.0
        target_kd = jd.drive.damping if has_drive else 0.0
        effort_limit = jd.drive.forceLimit if has_drive else np.inf
        if has_drive:
            actuator_mode = JointTargetMode.from_gains(
                target_ke, target_kd, force_position_velocity_actuation, has_drive=True
            )
        else:
            actuator_mode = JointTargetMode.NONE

        state_prefix = "angular" if is_revolute else "linear"
        initial_position = self.resolver.get_value(
            jp_prim, PrimType.JOINT, f"{state_prefix}_position", default=None, verbose=self.verbose
        )
        initial_velocity = self.resolver.get_value(
            jp_prim, PrimType.JOINT, f"{state_prefix}_velocity", default=None, verbose=self.verbose
        )

        if is_revolute:
            limit_lower *= self.degrees_to_radian
            limit_upper *= self.degrees_to_radian
            limit_ke /= self.degrees_to_radian
            limit_kd /= self.degrees_to_radian
            if has_drive:
                target_pos *= self.degrees_to_radian
                target_vel *= self.degrees_to_radian
                target_ke /= self.degrees_to_radian / joint_drive_gains_scaling
                target_kd /= self.degrees_to_radian / joint_drive_gains_scaling
            if velocity_limit is not None:
                velocity_limit *= self.degrees_to_radian
            if initial_position is not None:
                initial_position *= self.degrees_to_radian

        return _DofParams(
            armature=armature,
            friction=friction,
            damping=damping,
            velocity_limit=velocity_limit,
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            limit_ke=limit_ke,
            limit_kd=limit_kd,
            has_drive=has_drive,
            target_pos=target_pos,
            target_vel=target_vel,
            target_ke=target_ke,
            target_kd=target_kd,
            effort_limit=effort_limit,
            actuator_mode=actuator_mode,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            limit_solref_mode=self.joint_limit_solref_mode(jp_prim, limit_ke_source, limit_kd_source),
        )


@dataclass
class _PhysicsMaterial:
    """Physics-material values used by the USD shape importer."""

    staticFriction: float
    dynamicFriction: float
    torsionalFriction: float
    rollingFriction: float
    restitution: float
    density: float
    ke: float | None = None
    kd: float | None = None
    kf: float | None = None
    ka: float | None = None


def _resolve_physics_material(
    prim: Usd.Prim,
    desc: UsdPhysics.RigidBodyMaterialDesc,
    resolver: SchemaResolverManager,
    defaults: ModelBuilder.ShapeConfig,
    *,
    default_shape_density: float,
    verbose: bool,
) -> _PhysicsMaterial:
    """Read material values, retaining the importer's sampled density default."""

    def _resolve_contact_attr(key, _prim=prim):
        val = resolver.get_value(_prim, prim_type=PrimType.MATERIAL, key=key, verbose=verbose)
        if val is None:
            return None
        return float(val)

    if not math.isfinite(desc.density):
        warnings.warn(
            f"{prim.GetPath()}: authored material density must be finite; treating it as unspecified.",
            stacklevel=3,
        )

    return _PhysicsMaterial(
        staticFriction=desc.staticFriction,
        dynamicFriction=desc.dynamicFriction,
        restitution=desc.restitution,
        torsionalFriction=resolver.get_value(
            prim,
            prim_type=PrimType.MATERIAL,
            key="mu_torsional",
            default=defaults.mu_torsional,
            verbose=verbose,
        ),
        rollingFriction=resolver.get_value(
            prim,
            prim_type=PrimType.MATERIAL,
            key="mu_rolling",
            default=defaults.mu_rolling,
            verbose=verbose,
        ),
        # Treat non-positive, non-finite, or unauthored material density as "use importer default".
        # Effective collider/body MassAPI mass+inertia is handled later.
        density=desc.density if math.isfinite(desc.density) and desc.density > 0.0 else default_shape_density,
        ke=_resolve_contact_attr("ke"),
        kd=_resolve_contact_attr("kd"),
        kf=_resolve_contact_attr("kf"),
        ka=_resolve_contact_attr("ka"),
    )


def _resolve_shape_offsets(
    prim: Usd.Prim,
    resolver: SchemaResolverManager,
    defaults: ModelBuilder.ShapeConfig,
    *,
    legacy_margin_gap: bool,
    verbose: bool,
) -> tuple[float, float | None]:
    """Resolve collision margin and gap, including the legacy MuJoCo translation."""
    margin_val, margin_resolver = resolver.get_value_with_resolver(
        prim,
        prim_type=PrimType.SHAPE,
        key="margin",
        default=defaults.margin,
        verbose=verbose,
    )
    gap_val = resolver.get_value(
        prim,
        prim_type=PrimType.SHAPE,
        key="gap",
        verbose=verbose,
    )
    if gap_val == float("-inf"):
        gap_val = defaults.gap
    if legacy_margin_gap and margin_resolver is not None and margin_resolver.name == "mjc":
        # Legacy pre-3.9 import: newton_margin = mjc_margin - mjc_gap.
        mjc_gap = usd.get_attribute(prim, "mjc:gap")
        mjc_gap = 0.0 if mjc_gap is None else float(mjc_gap)
        newton_margin = float(margin_val) - mjc_gap
        if newton_margin < 0.0:
            warnings.warn(
                f"Prim '{prim.GetPath()}': legacy translation yields "
                f"negative margin (mjc_margin={margin_val}, mjc_gap={mjc_gap}).",
                stacklevel=3,
            )
        margin_val = newton_margin
    return margin_val, gap_val


def _resolve_shape_contact(
    prim: Usd.Prim,
    resolver: SchemaResolverManager,
    material: _PhysicsMaterial,
    defaults: ModelBuilder.ShapeConfig,
    *,
    verbose: bool,
) -> dict[str, float]:
    """Select contact response values from shape, material, and builder settings."""
    # Contact response precedence:
    #   per-shape mjc:solref (non-legacy) > material > legacy per-shape > default
    mjc_has_priority = False
    for _r in resolver.resolvers:
        if _r.name == "mjc":
            mjc_has_priority = True
            break
        if _r.name == "newton":
            break
    has_solref = mjc_has_priority and usd.get_attribute(prim, "mjc:solref") is not None
    shape_contact = {}
    for _ck in ("ke", "kd", "kf", "ka"):
        per_shape_val = resolver.get_value(prim, prim_type=PrimType.SHAPE, key=_ck, verbose=verbose)
        has_shape = per_shape_val is not None and math.isfinite(float(per_shape_val))
        mat_val = getattr(material, _ck)
        has_mat = mat_val is not None and math.isfinite(mat_val)

        if has_solref and _ck in ("ke", "kd") and has_shape:
            shape_contact[_ck] = float(per_shape_val)
        elif has_mat:
            shape_contact[_ck] = mat_val
        elif has_shape:
            shape_contact[_ck] = float(per_shape_val)
        else:
            shape_contact[_ck] = getattr(defaults, _ck)

    return shape_contact


@dataclass
class _ShapeSdfProperties:
    """Resolved SDF settings, including whether the shape applies the SDF schema."""

    has_api: bool
    max_resolution: int | None
    narrow_band_range: tuple[float, float]
    target_voxel_size: float | None
    texture_format: str
    padding: float | None


def _resolve_shape_sdf(
    prim: Usd.Prim,
    resolver: SchemaResolverManager,
    defaults: ModelBuilder.ShapeConfig,
    *,
    verbose: bool,
) -> _ShapeSdfProperties:
    """Resolve SDF settings and validate authored values before shape construction."""
    # SDF parameters. Applying NewtonSDFCollisionAPI is the canonical
    # signal that SDF generation is configured for this shape.
    has_sdf_api = prim.HasAPI("NewtonSDFCollisionAPI")
    # NewtonSDFCollisionAPI and NewtonMeshCollisionAPI are independent
    # collision representations and should not be co-applied. SDF wins
    # when both are present.
    if has_sdf_api and prim.HasAPI("NewtonMeshCollisionAPI"):
        warnings.warn(
            f"{prim.GetPath()}: NewtonSDFCollisionAPI and NewtonMeshCollisionAPI are "
            f"independent collision representations and should not be co-applied; "
            f"SDF configuration will be used.",
            stacklevel=3,
        )

    # Resolve target_voxel_size first because it overrides
    # sdf_max_resolution and the two are mutually exclusive in
    # ShapeConfig.validate().
    sdf_target_voxel_size = resolver.get_value(
        prim, prim_type=PrimType.SHAPE, key="sdf_target_voxel_size", verbose=verbose
    )
    if sdf_target_voxel_size == float("-inf"):
        sdf_target_voxel_size = None
    elif sdf_target_voxel_size is not None and sdf_target_voxel_size <= 0:
        warnings.warn(
            f"{prim.GetPath()}: newton:sdfTargetVoxelSize={sdf_target_voxel_size!r} is invalid "
            f"(must be > 0); falling back to default.",
            stacklevel=3,
        )
        sdf_target_voxel_size = None
    if sdf_target_voxel_size is None:
        sdf_target_voxel_size = defaults.sdf_target_voxel_size

    sdf_max_resolution = resolver.get_value(prim, prim_type=PrimType.SHAPE, key="sdf_max_resolution", verbose=verbose)
    if sdf_max_resolution == float("-inf"):
        sdf_max_resolution = None
    elif sdf_max_resolution is not None and sdf_max_resolution <= 0:
        warnings.warn(
            f"{prim.GetPath()}: newton:sdfMaxResolution={sdf_max_resolution!r} is invalid "
            f"(must be > 0); falling back to default.",
            stacklevel=3,
        )
        sdf_max_resolution = None
    elif sdf_max_resolution is not None and sdf_max_resolution % 8 != 0:
        warnings.warn(
            f"{prim.GetPath()}: newton:sdfMaxResolution={sdf_max_resolution!r} must be "
            f"divisible by 8 (SDF volumes are allocated in 8x8x8 tiles); falling back to default.",
            stacklevel=3,
        )
        sdf_max_resolution = None
    if sdf_target_voxel_size is not None and sdf_max_resolution is not None:
        warnings.warn(
            f"{prim.GetPath()}: both newton:sdfTargetVoxelSize and newton:sdfMaxResolution "
            f"are set; sdfTargetVoxelSize takes precedence.",
            stacklevel=3,
        )
        sdf_max_resolution = None
    if sdf_max_resolution is None:
        # When the API is applied but neither attribute is authored,
        # fall back to the schema default (64). When target voxel
        # size already drives the resolution, leave max_resolution
        # unset so the two don't conflict in ShapeConfig.validate().
        if has_sdf_api and sdf_target_voxel_size is None:
            sdf_max_resolution = 64
        else:
            sdf_max_resolution = defaults.sdf_max_resolution

    sdf_narrow_band_inner = resolver.get_value(
        prim, prim_type=PrimType.SHAPE, key="sdf_narrow_band_inner", verbose=verbose
    )
    if sdf_narrow_band_inner == float("-inf"):
        sdf_narrow_band_inner = None
    sdf_narrow_band_outer = resolver.get_value(
        prim, prim_type=PrimType.SHAPE, key="sdf_narrow_band_outer", verbose=verbose
    )
    if sdf_narrow_band_outer == float("-inf"):
        sdf_narrow_band_outer = None
    default_nb = defaults.sdf_narrow_band_range
    sdf_narrow_band_range = (
        sdf_narrow_band_inner if sdf_narrow_band_inner is not None else default_nb[0],
        sdf_narrow_band_outer if sdf_narrow_band_outer is not None else default_nb[1],
    )

    sdf_texture_format = resolver.get_value(prim, prim_type=PrimType.SHAPE, key="sdf_texture_format", verbose=verbose)
    _valid_sdf_tex_fmts = ("float32", "uint16", "uint8")
    if sdf_texture_format is not None and sdf_texture_format not in _valid_sdf_tex_fmts:
        warnings.warn(
            f"{prim.GetPath()}: newton:sdfTextureFormat={sdf_texture_format!r} is invalid "
            f"(expected one of {list(_valid_sdf_tex_fmts)}); falling back to default.",
            stacklevel=3,
        )
        sdf_texture_format = None
    if sdf_texture_format is None:
        sdf_texture_format = defaults.sdf_texture_format

    sdf_padding = resolver.get_value(prim, prim_type=PrimType.SHAPE, key="sdf_padding", verbose=verbose)
    if sdf_padding == float("-inf"):
        sdf_padding = None
    elif sdf_padding is not None and sdf_padding < 0:
        warnings.warn(
            f"{prim.GetPath()}: newton:sdfPadding={sdf_padding!r} is invalid (must be >= 0); falling back to default.",
            stacklevel=3,
        )
        sdf_padding = None

    return _ShapeSdfProperties(
        has_api=has_sdf_api,
        max_resolution=sdf_max_resolution,
        narrow_band_range=sdf_narrow_band_range,
        target_voxel_size=sdf_target_voxel_size,
        texture_format=sdf_texture_format,
        padding=sdf_padding,
    )


def _resolve_shape_hydroelastic(
    prim: Usd.Prim,
    resolver: SchemaResolverManager,
    defaults: ModelBuilder.ShapeConfig,
    sdf: _ShapeSdfProperties,
    *,
    is_mesh: bool,
    verbose: bool,
) -> tuple[bool, float]:
    """Resolve hydroelastic settings and require an SDF source for mesh shapes."""
    hydroelastic_enabled = resolver.get_value(
        prim, prim_type=PrimType.SHAPE, key="hydroelastic_enabled", verbose=verbose
    )
    kh = resolver.get_value(prim, prim_type=PrimType.SHAPE, key="kh", verbose=verbose)
    if kh == float("-inf"):
        kh = None
    elif kh is not None and kh <= 0:
        warnings.warn(
            f"{prim.GetPath()}: newton:hydroelasticStiffness={kh!r} is invalid (must be > 0); falling back to default.",
            stacklevel=3,
        )
        kh = None
    if hydroelastic_enabled is True:
        is_hydroelastic = True
    elif hydroelastic_enabled is False:
        is_hydroelastic = False
    elif sdf.has_api:
        # API applied but hydroelasticEnabled unauthored -> schema default False, not builder default.
        is_hydroelastic = False
    else:
        is_hydroelastic = defaults.is_hydroelastic
    if kh is None:
        kh = defaults.kh

    # Hydroelastic meshes need an SDF source. For primitives, a texture
    # SDF is generated from a synthesized watertight mesh at finalize(),
    # but meshes require either an attached mesh.sdf or a
    # resolution/voxel_size so one can be built deferred. Warn and
    # disable hydroelastic on this shape rather than aborting the whole
    # import — typically reached when newton:hydroelasticEnabled=true
    # is authored without applying NewtonSDFCollisionAPI.
    if is_hydroelastic and is_mesh and sdf.max_resolution is None and sdf.target_voxel_size is None:
        warnings.warn(
            f"{prim.GetPath()}: hydroelastic mesh requires newton:sdfMaxResolution "
            f"or newton:sdfTargetVoxelSize so an SDF can be generated; "
            f"disabling hydroelastic for this shape.",
            stacklevel=3,
        )
        is_hydroelastic = False

    return is_hydroelastic, kh


def _resolve_shape_shell(
    prim: Usd.Prim, resolver: SchemaResolverManager, margin_val: float
) -> tuple[bool, float, float | None]:
    """Return solidity, the inertia margin, and the raw thickness for margin restoration."""
    # Mass model and shell thickness (resolved across Newton / MuJoCo schemas)
    mass_model = resolver.get_value(prim, PrimType.SHAPE, "mass_model", default="solid")
    shape_is_solid = mass_model != "shell"
    shell_thickness_val = resolver.get_value(prim, PrimType.SHAPE, "shell_thickness")
    # When shell thickness is authored, pass it as margin so compute_inertia_shape
    # uses the correct thickness. The real collision margin is restored after add_shape.
    if shell_thickness_val is not None and math.isfinite(float(shell_thickness_val)):
        if float(shell_thickness_val) >= 0.0:
            inertia_margin = float(shell_thickness_val)
        else:
            warnings.warn(
                f"Shape {prim.GetPath()}: negative shell thickness {shell_thickness_val}; falling back to margin.",
                stacklevel=3,
            )
            inertia_margin = margin_val
    else:
        inertia_margin = margin_val

    return shape_is_solid, inertia_margin, shell_thickness_val
