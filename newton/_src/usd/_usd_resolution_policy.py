# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Importer-specific USD property resolution."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from ..solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_MJCF_DEFAULT, SOLREF_MODE_RAW
from .schema_resolver import (
    PrimType,
    SchemaResolver,
    SchemaResolverManager,
    _ImporterDefault,
    _PolicySelection,
    _ResolvedValue,
    _ResolverValue,
    _values_equal,
    _ValueSource,
)

_HARD_LIMIT_KE = 1.0e8
_VALID_SDF_TEXTURE_FORMATS = ("float32", "uint16", "uint8")


def _interpret_usd_joint_velocity_limit(value: Any) -> float | None:
    """Interpret an unlimited USD joint velocity as absent."""
    return None if value == float("inf") else value


def _interpret_usd_joint_state(value: Any, _resolver: SchemaResolver | None = None) -> float:
    """Interpret an absent USD joint state as the builder's zero state."""
    return 0.0 if value is None else value


def _interpret_usd_contact_parameter(value: Any) -> float | None:
    """Interpret an unspecified USD contact response parameter as absent."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _interpret_usd_contact_result(resolved: _ResolvedValue) -> float | None:
    """Interpret a resolved USD contact response parameter."""
    return _interpret_usd_contact_parameter(resolved.value)


def _resolve_newton_limit_ke(
    limit_ke: float | None,
    fallback: float,
    fallback_source: str,
    builder_default: float,
) -> tuple[float, str]:
    """Resolve a Newton limit-stiffness value and its consumer source."""
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
    """Resolve a Newton limit-damping value and its consumer source."""
    if limit_ke == float("inf") or limit_kd == float("inf"):
        return 0.0, "force"
    if limit_kd is None:
        return fallback, fallback_source
    if limit_kd == float("-inf"):
        return builder_default, "force"
    return limit_kd, "force"


class _UsdResolutionPolicy:
    """Resolve and interpret importer properties without traversing a USD stage."""

    @dataclass(frozen=True)
    class SceneProperties:
        """Hold interpreted PhysicsScene properties."""

        physics_dt: float
        gravity_enabled: bool
        max_solver_iterations: int

    @dataclass(frozen=True)
    class ContactResponse:
        """Group the four contact-response fields used by the importer."""

        names: ClassVar[tuple[str, ...]] = ("ke", "kd", "kf", "ka")

        ke: Any
        kd: Any
        kf: Any
        ka: Any

        @classmethod
        def from_getter(cls, getter: Callable[[str], Any]) -> _UsdResolutionPolicy.ContactResponse:
            return cls(*(getter(name) for name in cls.names))

        def get(self, name: str, default: Any = None) -> Any:
            return getattr(self, name, default)

        def items(self) -> tuple[tuple[str, Any], ...]:
            return tuple(zip(self.names, self.values(), strict=True))

        def values(self) -> tuple[Any, Any, Any, Any]:
            return (self.ke, self.kd, self.kf, self.ka)

    @dataclass(frozen=True)
    class _ContactResponseSelection:
        """Keep selected contact values and their owning input group."""

        values: _UsdResolutionPolicy.ContactResponse
        owners: _UsdResolutionPolicy.ContactResponse

    @dataclass
    class PhysicsMaterial:
        """Keep one parsed physics material and its resolution policies."""

        static_friction: float
        dynamic_friction: float
        torsional_friction: float
        rolling_friction: float
        restitution: float
        density: float
        ke: float | None = None
        kd: float | None = None
        kf: float | None = None
        ka: float | None = None
        prim: Any = None
        policies: dict[str, SchemaResolverManager._InterpretedPolicyValues] = field(default_factory=dict)

        @classmethod
        def from_shape_config(cls, config: Any) -> _UsdResolutionPolicy.PhysicsMaterial:
            """Create the importer material from the builder shape defaults."""
            return cls(
                static_friction=config.mu,
                dynamic_friction=config.mu,
                torsional_friction=config.mu_torsional,
                rolling_friction=config.mu_rolling,
                restitution=config.restitution,
                density=config.density,
            )

    @dataclass(frozen=True)
    class ShapeProperties:
        """Hold interpreted properties shared by rigid collision shapes."""

        margin: float
        gap: float
        sdf_max_resolution: int | None
        sdf_narrow_band_range: tuple[float, float]
        sdf_target_voxel_size: float | None
        sdf_texture_format: str
        sdf_padding: float | None
        is_hydroelastic: bool
        kh: float
        is_solid: bool
        shell_thickness: float | None
        inertia_margin: float

    @dataclass(frozen=True)
    class _ShapeOffsets:
        """Keep resolved contact offsets and their policy values."""

        margin: float
        gap: float
        margin_policies: SchemaResolverManager._InterpretedPolicyValues
        gap_policies: SchemaResolverManager._InterpretedPolicyValues

    @dataclass(frozen=True)
    class _SdfResolutionSettings:
        """Keep the coupled SDF resolution settings for one policy."""

        target_voxel_size: float | None
        max_resolution: int | None

    @dataclass(frozen=True)
    class _ShapeSdfProperties:
        """Keep resolved SDF settings used by shape validation."""

        narrow_band_range: tuple[float, float]
        texture_format: str
        padding: float | None
        active_settings: _UsdResolutionPolicy._SdfResolutionSettings
        legacy_settings: _UsdResolutionPolicy._SdfResolutionSettings | None
        composed_settings: _UsdResolutionPolicy._SdfResolutionSettings | None

    @dataclass(frozen=True)
    class _ShapeHydroelasticProperties:
        """Keep validated hydroelastic shape properties."""

        enabled: bool
        stiffness: float

    @dataclass(frozen=True)
    class _ShapeMassProperties:
        """Keep interpreted shell mass properties."""

        is_solid: bool
        shell_thickness: float | None
        inertia_margin: float

    @dataclass(frozen=True)
    class _JointDampingValue:
        """Keep the angular unit selected by a damping mapping key."""

        value: float
        angular_unit: Literal["degrees", "radians"] | None

    @dataclass(frozen=True)
    class JointLimitDefaults:
        """Hold builder defaults for one joint limit."""

        ke: float
        kd: float

    @dataclass(frozen=True)
    class JointLimitResult:
        """Hold one assembled joint-limit result and its source semantics."""

        ke: float
        kd: float
        ke_source: Literal["force", "mjc_authored", "mjc_default"]
        kd_source: Literal["force", "mjc_authored", "mjc_default"]

        @property
        def solref_mode(self) -> int:
            if self.ke_source == self.kd_source == "mjc_authored":
                return SOLREF_MODE_RAW
            if self.ke_source == self.kd_source == "mjc_default":
                return SOLREF_MODE_MJCF_DEFAULT
            return SOLREF_MODE_FORCE_SPACE

        @property
        def comparison(self) -> tuple[float, float, int]:
            return self.ke, self.kd, self.solref_mode

    @dataclass(frozen=True)
    class _JointLimitAudit:
        """Describe one legacy-to-composed joint-limit comparison."""

        legacy: _UsdResolutionPolicy.JointLimitResult
        composed: _UsdResolutionPolicy.JointLimitResult
        legacy_owners: tuple[str, str]
        composed_owners: tuple[str, str]

    def __init__(
        self,
        resolver: SchemaResolverManager,
        *,
        degrees_to_radian: float,
        default_joint_damping: float,
        default_joint_velocity_limit: float,
        mjc_resolver: SchemaResolver | None,
        mjc_schema_is_applied: Callable[[Any, str], bool] | None,
        verbose: bool,
    ) -> None:
        self._resolver = resolver
        self._degrees_to_radian = degrees_to_radian
        self._default_joint_damping = default_joint_damping
        self._default_joint_velocity_limit = default_joint_velocity_limit
        self._mjc_resolver = mjc_resolver
        self._mjc_schema_is_applied = mjc_schema_is_applied
        self._mjc_has_priority = False
        for candidate in resolver.resolvers:
            if candidate.name == "mjc":
                self._mjc_has_priority = True
                break
            if candidate.name == "newton":
                break
        self._verbose = verbose

    def resolve_scene(self, prim: Any) -> SceneProperties:
        """Resolve the PhysicsScene properties consumed by the importer."""

        def interpret_time_steps_per_second(result: _ResolvedValue) -> float:
            value = result.value
            return (1.0 / value) if value is not None and value > 0 else 0.001

        physics_dt = self._resolver._get_interpreted_value(
            prim,
            prim_type=PrimType.SCENE,
            key="time_steps_per_second",
            default=1000,
            verbose=self._verbose,
            interpreter=interpret_time_steps_per_second,
        ).value
        gravity_enabled = self.resolve_gravity_enabled(prim)
        max_solver_iterations = self._resolver.get_value(
            prim,
            prim_type=PrimType.SCENE,
            key="max_solver_iterations",
            default=-1,
            legacy_default=None,
            verbose=self._verbose,
        )
        return self.SceneProperties(physics_dt, gravity_enabled, max_solver_iterations)

    def resolve_gravity_enabled(self, prim: Any) -> bool:
        """Resolve gravity enablement with the shared boolean interpretation."""
        return self._resolver._get_interpreted_value(
            prim,
            prim_type=PrimType.SCENE,
            key="gravity_enabled",
            default=True,
            verbose=self._verbose,
            interpreter=lambda result: bool(result.value),
        ).value

    def resolve_contact_response(
        self,
        shape_prim: Any,
        material: PhysicsMaterial,
        shape_defaults: Any,
        *,
        has_mjc_solref: bool,
    ) -> ContactResponse:
        """Resolve and audit the final per-shape contact response."""
        has_solref = self._mjc_has_priority and has_mjc_solref
        shape_policies = self.ContactResponse.from_getter(
            lambda key: self._resolver._resolve_interpreted_policies(
                shape_prim,
                PrimType.SHAPE,
                key,
                None,
                interpreter=_interpret_usd_contact_result,
            )
        )
        material_contact_policies = self.ContactResponse.from_getter(material.policies.get)

        def select_field(
            key: str,
            policy: _PolicySelection,
        ) -> tuple[float, Literal["shape", "material", "default"]]:
            shape_result = shape_policies.get(key).select(policy)
            shape_value = None if shape_result is None else shape_result.value
            has_shape_value = shape_value is not None

            material_policy = material_contact_policies.get(key)
            material_result = None if material_policy is None else material_policy.select(policy)
            material_value = None if material_result is None else material_result.value
            has_material_value = material_value is not None

            if has_solref and key in ("ke", "kd") and has_shape_value:
                return float(shape_value), "shape"
            if has_material_value:
                return material_value, "material"
            if has_shape_value:
                return float(shape_value), "shape"
            return getattr(shape_defaults, key), "default"

        def select(policy: _PolicySelection) -> _UsdResolutionPolicy._ContactResponseSelection:
            selected = self.ContactResponse.from_getter(lambda key: select_field(key, policy))
            return self._ContactResponseSelection(
                self.ContactResponse.from_getter(lambda key: selected.get(key)[0]),
                self.ContactResponse.from_getter(lambda key: selected.get(key)[1]),
            )

        active = select("active")
        policy_inputs = [*shape_policies.values(), *material.policies.values()]
        can_audit = not self._resolver._uses_composed_fallbacks and all(
            policies.legacy is not None and policies.composed is not None for policies in policy_inputs
        )
        if not can_audit:
            return active.values

        legacy = select("legacy")
        composed = select("composed")

        def candidate(
            key: str,
            policies: SchemaResolverManager._InterpretedPolicyValues,
            owner: Literal["shape", "material"],
        ) -> SchemaResolverManager._PolicyChangeCandidate | None:
            if owner not in {legacy.owners.get(key), composed.owners.get(key)}:
                return None
            return policies.contribution(
                legacy_comparison=policies.legacy.value,
                composed_comparison=policies.composed.value,
            )

        shape_candidates = tuple(
            change
            for key, policies in shape_policies.items()
            if (change := candidate(key, policies, "shape")) is not None
        )
        self._resolver._audit_assembled_property(
            shape_prim,
            PrimType.SHAPE,
            legacy.values.values(),
            composed.values.values(),
            shape_candidates,
        )

        if material.prim is None or not material.policies:
            return active.values

        material_contact_candidates = tuple(
            change
            for key, policies in material_contact_policies.items()
            if policies is not None and (change := candidate(key, policies, "material")) is not None
        )
        self._resolver._audit_assembled_property(
            material.prim,
            PrimType.MATERIAL,
            legacy.values.values(),
            composed.values.values(),
            material_contact_candidates,
        )

        def material_friction(key: str, policy: Literal["legacy", "composed"]) -> float:
            policies = material.policies.get(key)
            resolved = None if policies is None else policies.select(policy)
            if resolved is not None:
                return resolved.value
            if key == "mu_torsional":
                return material.torsional_friction
            return material.rolling_friction

        friction_keys = ("mu_torsional", "mu_rolling")
        legacy_friction = tuple(material_friction(key, "legacy") for key in friction_keys)
        composed_friction = tuple(material_friction(key, "composed") for key in friction_keys)
        friction_candidates = tuple(
            policies.contribution(
                legacy_comparison=material_friction(key, "legacy"),
                composed_comparison=material_friction(key, "composed"),
            )
            for key in friction_keys
            if (policies := material.policies.get(key)) is not None
        )
        self._resolver._audit_assembled_property(
            material.prim,
            PrimType.MATERIAL,
            legacy_friction,
            composed_friction,
            friction_candidates,
        )
        return active.values

    def resolve_material(
        self,
        prim: Any,
        *,
        static_friction: float,
        dynamic_friction: float,
        restitution: float,
        density: float,
        default_shape: Any,
    ) -> PhysicsMaterial:
        """Resolve the material properties consumed by rigid shapes."""
        value_policies = {}

        def resolve_property(key: str, default: Any = None, *, interpret_contact: bool = False):
            policies = self._resolver._resolve_interpreted_policies(
                prim,
                PrimType.MATERIAL,
                key,
                default,
                interpreter=_interpret_usd_contact_result if interpret_contact else None,
            )
            value_policies[key] = policies
            return policies.active.value

        return self.PhysicsMaterial(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
            torsional_friction=resolve_property("mu_torsional", default_shape.mu_torsional),
            rolling_friction=resolve_property("mu_rolling", default_shape.mu_rolling),
            density=density,
            ke=resolve_property("ke", interpret_contact=True),
            kd=resolve_property("kd", interpret_contact=True),
            kf=resolve_property("kf", interpret_contact=True),
            ka=resolve_property("ka", interpret_contact=True),
            prim=prim,
            policies=value_policies,
        )

    def resolve_shape(
        self,
        prim: Any,
        *,
        prim_path: str,
        defaults: Any,
        has_sdf_api: bool,
        is_mesh: bool,
        is_plane: bool,
        legacy_margin_gap: bool,
        read_legacy_mjc_gap: Callable[[], float],
    ) -> ShapeProperties:
        """Resolve and interpret properties shared by collision shapes."""
        offsets = self._resolve_shape_offsets(
            prim,
            prim_path=prim_path,
            defaults=defaults,
            legacy_margin_gap=legacy_margin_gap,
            read_legacy_mjc_gap=read_legacy_mjc_gap,
        )

        sdf = self._resolve_shape_sdf(
            prim,
            prim_path=prim_path,
            defaults=defaults,
            has_sdf_api=has_sdf_api,
            gap_policies=offsets.gap_policies,
        )

        hydroelastic = self._resolve_shape_hydroelastic(
            prim,
            prim_path=prim_path,
            defaults=defaults,
            has_sdf_api=has_sdf_api,
            is_mesh=is_mesh,
            is_plane=is_plane,
            sdf=sdf,
        )
        mass = self._resolve_shape_mass(
            prim,
            prim_path=prim_path,
            margin=offsets.margin,
            margin_policies=offsets.margin_policies,
        )

        return self.ShapeProperties(
            margin=offsets.margin,
            gap=offsets.gap,
            sdf_max_resolution=sdf.active_settings.max_resolution,
            sdf_narrow_band_range=sdf.narrow_band_range,
            sdf_target_voxel_size=sdf.active_settings.target_voxel_size,
            sdf_texture_format=sdf.texture_format,
            sdf_padding=sdf.padding,
            is_hydroelastic=hydroelastic.enabled,
            kh=hydroelastic.stiffness,
            is_solid=mass.is_solid,
            shell_thickness=mass.shell_thickness,
            inertia_margin=mass.inertia_margin,
        )

    def _resolve_shape_offsets(
        self,
        prim: Any,
        *,
        prim_path: str,
        defaults: Any,
        legacy_margin_gap: bool,
        read_legacy_mjc_gap: Callable[[], float],
    ) -> _ShapeOffsets:
        """Resolve contact margin and gap values."""

        def interpret_margin(result: _ResolvedValue) -> float:
            value = defaults.margin if result.value is None else result.value
            if legacy_margin_gap and result.resolver is not None and result.resolver.name == "mjc":
                value = float(value) - read_legacy_mjc_gap()
            return value

        def interpret_gap(result: _ResolvedValue) -> float:
            if result.value is None or result.value == float("-inf"):
                return defaults.gap
            return result.value

        margin_policies = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "margin",
            defaults.margin,
            interpreter=interpret_margin,
        )
        gap_policies = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "gap",
            _ImporterDefault(defaults.gap),
            legacy_default=None,
            interpreter=interpret_gap,
        )
        self._audit_policy_value(prim, PrimType.SHAPE, gap_policies)
        self._audit_policy_value(prim, PrimType.SHAPE, margin_policies)

        margin = margin_policies.active.value
        raw_margin = margin_policies.active.raw_value
        margin_resolver = margin_policies.active.resolver
        if legacy_margin_gap and margin_resolver is not None and margin_resolver.name == "mjc" and margin < 0.0:
            warnings.warn(
                f"Prim '{prim_path}': legacy translation yields negative margin "
                f"(mjc_margin={raw_margin}, mjc_gap={read_legacy_mjc_gap()}).",
                stacklevel=4,
            )
        return self._ShapeOffsets(margin, gap_policies.active.value, margin_policies, gap_policies)

    def _resolve_shape_sdf(
        self,
        prim: Any,
        *,
        prim_path: str,
        defaults: Any,
        has_sdf_api: bool,
        gap_policies: SchemaResolverManager._InterpretedPolicyValues,
    ) -> _ShapeSdfProperties:
        """Resolve and validate SDF generation settings."""

        def interpret_target_voxel_size(result: _ResolvedValue) -> float | None:
            value = result.value
            if value == float("-inf") or (value is not None and value <= 0):
                value = None
            return defaults.sdf_target_voxel_size if value is None else value

        target_policies = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "sdf_target_voxel_size",
            None,
            interpreter=interpret_target_voxel_size,
        )
        raw_target = target_policies.active.raw_value
        if raw_target is not None and raw_target != float("-inf") and raw_target <= 0:
            warnings.warn(
                f"{prim_path}: newton:sdfTargetVoxelSize={raw_target!r} is invalid "
                f"(must be > 0); falling back to default.",
                stacklevel=4,
            )

        def interpret_max_resolution(result: _ResolvedValue, target: float | None) -> int | None:
            value = result.value
            if value == float("-inf") or (value is not None and (value <= 0 or value % 8 != 0)):
                value = None
            if target is not None and value is not None:
                value = None
            if value is None:
                if has_sdf_api and target is None:
                    return 64
                return defaults.sdf_max_resolution
            return value

        max_resolution_policies = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "sdf_max_resolution",
            None,
        )
        target_voxel_size = target_policies.active.value
        raw_max_resolution = max_resolution_policies.active.raw_value
        if raw_max_resolution is not None and raw_max_resolution != float("-inf") and raw_max_resolution <= 0:
            warnings.warn(
                f"{prim_path}: newton:sdfMaxResolution={raw_max_resolution!r} is invalid "
                f"(must be > 0); falling back to default.",
                stacklevel=4,
            )
        elif raw_max_resolution is not None and raw_max_resolution != float("-inf") and raw_max_resolution % 8 != 0:
            warnings.warn(
                f"{prim_path}: newton:sdfMaxResolution={raw_max_resolution!r} must be divisible by 8 "
                f"(SDF volumes are allocated in 8x8x8 tiles); falling back to default.",
                stacklevel=4,
            )
        elif target_voxel_size is not None and raw_max_resolution not in (None, float("-inf")):
            warnings.warn(
                f"{prim_path}: both newton:sdfTargetVoxelSize and newton:sdfMaxResolution are set; "
                f"sdfTargetVoxelSize takes precedence.",
                stacklevel=4,
            )

        def resolution_settings(
            policy: _PolicySelection,
        ) -> _UsdResolutionPolicy._SdfResolutionSettings:
            target = target_policies.select(policy).value
            max_result = max_resolution_policies.select(policy).resolved
            return self._SdfResolutionSettings(
                target_voxel_size=target,
                max_resolution=interpret_max_resolution(max_result, target),
            )

        sdf_max_resolution = interpret_max_resolution(max_resolution_policies.active.resolved, target_voxel_size)
        legacy_sdf_settings = None
        composed_sdf_settings = None
        if all(
            result is not None
            for result in (
                target_policies.legacy,
                target_policies.composed,
                max_resolution_policies.legacy,
                max_resolution_policies.composed,
            )
        ):
            legacy_sdf_settings = resolution_settings("legacy")
            composed_sdf_settings = resolution_settings("composed")
            self._resolver._audit_assembled_property(
                prim,
                PrimType.SHAPE,
                legacy_sdf_settings.target_voxel_size,
                composed_sdf_settings.target_voxel_size,
                (
                    target_policies.contribution(
                        legacy_comparison=legacy_sdf_settings.target_voxel_size,
                        composed_comparison=composed_sdf_settings.target_voxel_size,
                    ),
                ),
            )
            self._resolver._audit_assembled_property(
                prim,
                PrimType.SHAPE,
                legacy_sdf_settings.max_resolution,
                composed_sdf_settings.max_resolution,
                (
                    max_resolution_policies.contribution(
                        legacy_comparison=interpret_max_resolution(max_resolution_policies.legacy.resolved, None),
                        composed_comparison=interpret_max_resolution(max_resolution_policies.composed.resolved, None),
                    ),
                ),
            )

        def interpret_narrow_band(result: _ResolvedValue, default: float) -> float:
            if result.value is None or result.value == float("-inf"):
                return default
            return result.value

        default_narrow_band = defaults.sdf_narrow_band_range
        sdf_narrow_band_range = (
            self._resolver._get_interpreted_value(
                prim,
                PrimType.SHAPE,
                "sdf_narrow_band_inner",
                interpreter=lambda result: interpret_narrow_band(result, default_narrow_band[0]),
                verbose=self._verbose,
            ).value,
            self._resolver._get_interpreted_value(
                prim,
                PrimType.SHAPE,
                "sdf_narrow_band_outer",
                interpreter=lambda result: interpret_narrow_band(result, default_narrow_band[1]),
                verbose=self._verbose,
            ).value,
        )

        def interpret_texture_format(result: _ResolvedValue) -> str:
            if result.value is None or result.value not in _VALID_SDF_TEXTURE_FORMATS:
                return defaults.sdf_texture_format
            return result.value

        texture_format_result = self._resolver._get_interpreted_value(
            prim,
            PrimType.SHAPE,
            "sdf_texture_format",
            interpreter=interpret_texture_format,
        )
        raw_texture_format = texture_format_result.raw_value
        if raw_texture_format is not None and raw_texture_format not in _VALID_SDF_TEXTURE_FORMATS:
            warnings.warn(
                f"{prim_path}: newton:sdfTextureFormat={raw_texture_format!r} is invalid "
                f"(expected one of {list(_VALID_SDF_TEXTURE_FORMATS)}); falling back to default.",
                stacklevel=4,
            )

        def interpret_padding(result: _ResolvedValue) -> float | None:
            value = result.value
            if value == float("-inf") or (value is not None and value < 0):
                return None
            return value

        padding_policies = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "sdf_padding",
            None,
            interpreter=interpret_padding,
        )
        raw_padding = padding_policies.active.raw_value
        if raw_padding is not None and raw_padding != float("-inf") and raw_padding < 0:
            warnings.warn(
                f"{prim_path}: newton:sdfPadding={raw_padding!r} is invalid (must be >= 0); falling back to default.",
                stacklevel=4,
            )
        if all(
            result is not None
            for result in (
                gap_policies.legacy,
                gap_policies.composed,
                padding_policies.legacy,
                padding_policies.composed,
            )
        ):
            legacy_padding = padding_policies.legacy.value
            composed_padding = padding_policies.composed.value
            self._resolver._audit_assembled_property(
                prim,
                PrimType.SHAPE,
                gap_policies.legacy.value if legacy_padding is None else legacy_padding,
                gap_policies.composed.value if composed_padding is None else composed_padding,
                (
                    padding_policies.contribution(
                        legacy_comparison=legacy_padding,
                        composed_comparison=composed_padding,
                    ),
                ),
            )

        return self._ShapeSdfProperties(
            narrow_band_range=sdf_narrow_band_range,
            texture_format=texture_format_result.value,
            padding=padding_policies.active.value,
            active_settings=self._SdfResolutionSettings(
                target_voxel_size=target_voxel_size,
                max_resolution=sdf_max_resolution,
            ),
            legacy_settings=legacy_sdf_settings,
            composed_settings=composed_sdf_settings,
        )

    def _resolve_shape_hydroelastic(
        self,
        prim: Any,
        *,
        prim_path: str,
        defaults: Any,
        has_sdf_api: bool,
        is_mesh: bool,
        is_plane: bool,
        sdf: _ShapeSdfProperties,
    ) -> _ShapeHydroelasticProperties:
        """Resolve and validate hydroelastic shape settings."""

        def interpret_enabled(result: _ResolvedValue) -> bool:
            if result.value is True or result.value is False:
                return result.value
            if has_sdf_api:
                return False
            return defaults.is_hydroelastic

        enabled_policies = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "hydroelastic_enabled",
            None,
            interpreter=interpret_enabled,
        )

        def validate_enabled(
            enabled: bool,
            sdf_settings: _UsdResolutionPolicy._SdfResolutionSettings,
        ) -> bool:
            if is_plane:
                return False
            if enabled and is_mesh and sdf_settings.target_voxel_size is None and sdf_settings.max_resolution is None:
                return False
            return enabled

        requested = enabled_policies.active.value
        enabled = validate_enabled(requested, sdf.active_settings)
        if (
            enabled_policies.legacy is not None
            and enabled_policies.composed is not None
            and sdf.legacy_settings is not None
            and sdf.composed_settings is not None
        ):
            self._resolver._audit_assembled_property(
                prim,
                PrimType.SHAPE,
                validate_enabled(enabled_policies.legacy.value, sdf.legacy_settings),
                validate_enabled(enabled_policies.composed.value, sdf.composed_settings),
                (
                    enabled_policies.contribution(
                        legacy_comparison=enabled_policies.legacy.value,
                        composed_comparison=enabled_policies.composed.value,
                    ),
                ),
            )

        def interpret_stiffness(result: _ResolvedValue) -> float:
            if result.value == float("-inf") or result.value is None or result.value <= 0:
                return defaults.kh
            return result.value

        stiffness = self._resolver._get_interpreted_value(
            prim,
            PrimType.SHAPE,
            "kh",
            interpreter=interpret_stiffness,
            verbose=self._verbose,
        )
        raw_stiffness = stiffness.raw_value
        if raw_stiffness is not None and raw_stiffness != float("-inf") and raw_stiffness <= 0:
            warnings.warn(
                f"{prim_path}: newton:hydroelasticStiffness={raw_stiffness!r} is invalid "
                f"(must be > 0); falling back to default.",
                stacklevel=4,
            )
        if (
            requested
            and is_mesh
            and sdf.active_settings.max_resolution is None
            and sdf.active_settings.target_voxel_size is None
        ):
            warnings.warn(
                f"{prim_path}: hydroelastic mesh requires newton:sdfMaxResolution or "
                f"newton:sdfTargetVoxelSize so an SDF can be generated; disabling "
                f"hydroelastic for this shape.",
                stacklevel=4,
            )
        return self._ShapeHydroelasticProperties(enabled, stiffness.value)

    def _resolve_shape_mass(
        self,
        prim: Any,
        *,
        prim_path: str,
        margin: float,
        margin_policies: SchemaResolverManager._InterpretedPolicyValues,
    ) -> _ShapeMassProperties:
        """Resolve mass model and shell thickness settings."""
        mass_model_policies = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "mass_model",
            "solid",
            interpreter=lambda result: result.value != "shell",
        )
        self._audit_policy_value(prim, PrimType.SHAPE, mass_model_policies)

        def usable_shell_thickness(result: _ResolvedValue) -> float | None:
            if result.value is None:
                return None
            value = float(result.value)
            return value if math.isfinite(value) and value >= 0.0 else None

        shell_policies = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "shell_thickness",
            None,
            interpreter=usable_shell_thickness,
        )
        raw_shell_thickness = shell_policies.active.raw_value
        inertia_margin = margin if shell_policies.active.value is None else shell_policies.active.value
        if all(
            result is not None
            for result in (
                margin_policies.legacy,
                margin_policies.composed,
                shell_policies.legacy,
                shell_policies.composed,
            )
        ):
            legacy_shell = (
                margin_policies.legacy.value if shell_policies.legacy.value is None else shell_policies.legacy.value
            )
            composed_shell = (
                margin_policies.composed.value
                if shell_policies.composed.value is None
                else shell_policies.composed.value
            )
            self._resolver._audit_assembled_property(
                prim,
                PrimType.SHAPE,
                legacy_shell,
                composed_shell,
                (
                    shell_policies.contribution(
                        legacy_comparison=shell_policies.legacy.value,
                        composed_comparison=shell_policies.composed.value,
                    ),
                ),
            )
        if (
            raw_shell_thickness is not None
            and math.isfinite(float(raw_shell_thickness))
            and float(raw_shell_thickness) < 0.0
        ):
            warnings.warn(
                f"Shape {prim_path}: negative shell thickness {raw_shell_thickness}; falling back to margin.",
                stacklevel=4,
            )
        return self._ShapeMassProperties(
            mass_model_policies.active.value,
            raw_shell_thickness,
            inertia_margin,
        )

    def resolve_cloth_shell_thickness(self, prim: Any) -> float | None:
        """Resolve the optional shell thickness shared with cloth import."""
        mass_model = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "mass_model",
            "solid",
            interpreter=lambda result: result.value == "shell",
        )

        def interpret_thickness(result: _ResolvedValue) -> float | None:
            if result.value is None:
                return None
            value = float(result.value)
            return value if math.isfinite(value) and value > 0.0 else None

        shell_thickness = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.SHAPE,
            "shell_thickness",
            None,
            interpreter=interpret_thickness,
        )

        def effective_thickness(policy: _PolicySelection) -> float | None:
            model = mass_model.select(policy)
            thickness = shell_thickness.select(policy)
            if model is None or thickness is None or not model.value:
                return None
            return thickness.value

        if mass_model.legacy is not None and shell_thickness.legacy is not None:
            self._resolver._audit_assembled_property(
                prim,
                PrimType.SHAPE,
                effective_thickness("legacy"),
                effective_thickness("composed"),
                (mass_model.contribution(), shell_thickness.contribution()),
            )
        return effective_thickness("active")

    def resolve_max_hull_vertices(self, prim: Any, *, default: Any, override: Any) -> int:
        """Resolve the convex-hull vertex limit for a mesh."""
        return self._resolver.get_value(
            prim,
            PrimType.SHAPE,
            "max_hull_vertices",
            default=default,
            override=override,
            verbose=self._verbose,
        )

    def _audit_policy_value(
        self,
        prim: Any,
        prim_type: PrimType,
        policies: SchemaResolverManager._InterpretedPolicyValues,
    ) -> None:
        """Audit one interpreted value when both migration policies are available."""
        if policies.legacy is None or policies.composed is None:
            return
        self._resolver._audit_assembled_property(
            prim,
            prim_type,
            policies.legacy.value,
            policies.composed.value,
            (policies.contribution(),),
        )

    def resolve_joint_velocity_limits(
        self,
        prim: Any,
        *,
        revolute: tuple[bool, ...],
    ) -> tuple[float, ...]:
        """Resolve joint velocity limits in builder units for the active axes."""
        default = _ImporterDefault(self._default_joint_velocity_limit)
        return self._resolve_joint_dof_property(
            prim,
            "velocity_limit",
            revolute=revolute,
            default=default,
            legacy_default=None,
            interpreter=self._interpret_joint_velocity_limit,
        )

    def resolve_joint_passive_properties(
        self,
        prim: Any,
        *,
        default_armature: float,
        default_friction: float,
    ) -> tuple[float, float]:
        """Resolve joint armature and friction in builder units."""
        armature = self._resolver.get_value(
            prim,
            PrimType.JOINT,
            "armature",
            default=default_armature,
            verbose=self._verbose,
        )
        friction = self._resolver.get_value(
            prim,
            PrimType.JOINT,
            "friction",
            default=default_friction,
            verbose=self._verbose,
        )
        return armature, friction

    def resolve_joint_damping(
        self,
        prim: Any,
        *,
        revolute: tuple[bool, ...],
    ) -> tuple[float, ...]:
        """Resolve passive joint damping in builder units for the active axes."""

        read_value = self._resolver._cached_value_reader(prim, PrimType.JOINT)

        def read_damping_value(resolver, key):
            state = read_value(resolver, key)
            if key == "damping_per_rad" and state.usable:
                return _ResolverValue(
                    self._JointDampingValue(state.value, "radians"),
                    state.authored,
                )
            return state

        def resolve_legacy(resolvers, read_value):
            for resolver in resolvers:
                for key in ("damping", "damping_per_rad"):
                    mapping = resolver.mapping.get(PrimType.JOINT, {})
                    if key not in mapping:
                        continue
                    state = read_value(resolver, key)
                    if state.authored and state.usable:
                        return _ResolvedValue(
                            state.value,
                            resolver,
                            _ValueSource.AUTHORED,
                            mapping_key=key,
                        )
            return _ResolvedValue(self._default_joint_damping, None, _ValueSource.IMPORTER_DEFAULT)

        def audit_key(policies):
            for policy in (policies.legacy, policies.composed):
                if policy is not None and isinstance(policy.raw_value, self._JointDampingValue):
                    return "damping_per_rad"
            return "damping"

        return self._resolve_joint_dof_property(
            prim,
            "damping",
            revolute=revolute,
            default=_ImporterDefault(self._default_joint_damping),
            legacy_default=self._default_joint_damping,
            interpreter=self._interpret_joint_damping,
            resolve_legacy=resolve_legacy,
            read_value=read_damping_value,
            authored_aliases=("damping_per_rad",),
            audit_key=audit_key,
        )

    def resolve_optional_joint_state(self, prim: Any, key: str) -> float | None:
        """Resolve optional joint state without reporting normal absence as an error."""
        return self._resolver.get_value(
            prim,
            PrimType.JOINT,
            key,
            comparison_key=_interpret_usd_joint_state,
        )

    def resolve_joint_generic_limit_policies(
        self,
        prim: Any,
        read_value,
    ) -> tuple[
        SchemaResolverManager._InterpretedPolicyValues,
        SchemaResolverManager._InterpretedPolicyValues,
    ]:
        """Resolve the generic Newton joint-limit gain policies."""
        return (
            self._resolver._resolve_interpreted_policies(
                prim,
                PrimType.JOINT,
                "limit_ke",
                None,
                read_value=read_value,
            ),
            self._resolver._resolve_interpreted_policies(
                prim,
                PrimType.JOINT,
                "limit_kd",
                None,
                read_value=read_value,
            ),
        )

    def resolve_articulation_self_collision(
        self,
        prim: Any,
        *,
        default: Any,
        override: Any,
    ) -> bool:
        """Resolve whether an articulation permits self-collision."""
        return self._resolver._get_interpreted_value(
            prim,
            PrimType.ARTICULATION,
            "self_collision_enabled",
            default=default,
            override=override,
            interpreter=lambda result: bool(result.value),
            verbose=self._verbose,
        ).value

    def resolve_joint_limit_gain_policies(
        self,
        prim: Any,
        key: str,
        builder_default: float,
        read_value,
    ) -> SchemaResolverManager._InterpretedPolicyValues:
        """Resolve a limit gain under both migration policies."""

        def resolve_legacy(resolvers, read_value):
            for resolver in resolvers:
                if key not in resolver.mapping.get(PrimType.JOINT, {}):
                    continue
                state = read_value(resolver, key)
                if not state.authored:
                    continue
                if resolver.name != "mjc":
                    if not state.usable:
                        continue
                    return _ResolvedValue(state.value, resolver, _ValueSource.AUTHORED, mapping_key=key)
                value = state.value
                if value is None:
                    value = self._get_mjc_joint_limit_default(prim, key)
                return _ResolvedValue(
                    builder_default if value is None else value,
                    resolver,
                    _ValueSource.AUTHORED,
                    mapping_key=key,
                )

            if self._mjc_resolver is not None:
                value = self._get_mjc_joint_limit_default(prim, key)
                if value is not None:
                    return _ResolvedValue(
                        value,
                        self._mjc_resolver,
                        _ValueSource.COMPATIBILITY_DEFAULT,
                        mapping_key=key,
                    )
            return _ResolvedValue(builder_default, None, _ValueSource.IMPORTER_DEFAULT)

        return self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.JOINT,
            key,
            builder_default,
            resolve_legacy=resolve_legacy,
            read_value=read_value,
        )

    def resolve_joint_limits(
        self,
        prim: Any,
        defaults: Mapping[str, JointLimitDefaults],
    ) -> dict[str, JointLimitResult]:
        """Resolve and audit generic and per-axis joint-limit gains."""
        read_value = self._resolver._cached_value_reader(prim, PrimType.JOINT)
        generic_ke, generic_kd = self.resolve_joint_generic_limit_policies(prim, read_value)
        candidates = {"limit_ke": generic_ke, "limit_kd": generic_kd}
        fallback_policies = {}

        for key, builder_defaults in defaults.items():
            fallback_ke = self.resolve_joint_limit_gain_policies(
                prim,
                f"{key}_ke",
                builder_defaults.ke,
                read_value,
            )
            fallback_kd = self.resolve_joint_limit_gain_policies(
                prim,
                f"{key}_kd",
                builder_defaults.kd,
                read_value,
            )
            fallback_policies[key] = (fallback_ke, fallback_kd)
            candidates[f"{key}_ke"] = fallback_ke
            candidates[f"{key}_kd"] = fallback_kd

        def assemble(key: str, policy: _PolicySelection) -> _UsdResolutionPolicy.JointLimitResult:
            builder_defaults = defaults[key]
            fallback_ke, fallback_kd = fallback_policies[key]
            return self._resolve_joint_limit_policy_result(
                generic_ke.select(policy).resolved,
                generic_kd.select(policy).resolved,
                fallback_ke.select(policy).resolved,
                fallback_kd.select(policy).resolved,
                builder_defaults.ke,
                builder_defaults.kd,
            )

        active = {key: assemble(key, "active") for key in defaults}
        can_audit = generic_ke.legacy is not None and all(
            policies.composed is not None for policies in candidates.values()
        )
        if not can_audit:
            return active

        changes = []
        for key in defaults:
            legacy = assemble(key, "legacy")
            composed = assemble(key, "composed")
            legacy_owners = self._joint_limit_policy_owners(
                generic_ke.legacy.resolved,
                generic_kd.legacy.resolved,
                f"{key}_ke",
                f"{key}_kd",
            )
            composed_owners = self._joint_limit_policy_owners(
                generic_ke.composed.resolved,
                generic_kd.composed.resolved,
                f"{key}_ke",
                f"{key}_kd",
            )
            changes.append(self._JointLimitAudit(legacy, composed, legacy_owners, composed_owners))

        self._audit_joint_limit_changes(prim, changes, candidates)
        return active

    def _resolve_joint_limit_policy_result(
        self,
        limit_ke: _ResolvedValue,
        limit_kd: _ResolvedValue,
        fallback_ke: _ResolvedValue,
        fallback_kd: _ResolvedValue,
        builder_ke: float,
        builder_kd: float,
    ) -> JointLimitResult:
        """Assemble generic and per-axis joint limit gains."""
        fallback_ke_value, fallback_ke_source = self._interpret_joint_limit_gain(fallback_ke, builder_ke)
        fallback_kd_value, fallback_kd_source = self._interpret_joint_limit_gain(fallback_kd, builder_kd)
        resolved_ke, ke_source = _resolve_newton_limit_ke(
            limit_ke.value,
            fallback_ke_value,
            fallback_ke_source,
            builder_ke,
        )
        resolved_kd, kd_source = _resolve_newton_limit_kd(
            limit_ke.value,
            limit_kd.value,
            fallback_kd_value,
            fallback_kd_source,
            builder_kd,
        )
        return self.JointLimitResult(resolved_ke, resolved_kd, ke_source, kd_source)

    @staticmethod
    def _joint_limit_policy_owners(
        limit_ke: _ResolvedValue,
        limit_kd: _ResolvedValue,
        fallback_ke_key: str,
        fallback_kd_key: str,
    ) -> tuple[str, str]:
        """Return the inputs that own the assembled stiffness and damping."""
        ke_owner = fallback_ke_key if limit_ke.value is None else "limit_ke"
        if limit_ke.value == float("inf"):
            kd_owner = "limit_ke"
        elif limit_kd.value is None:
            kd_owner = fallback_kd_key
        else:
            kd_owner = "limit_kd"
        return ke_owner, kd_owner

    def _audit_joint_limit_changes(
        self,
        prim: Any,
        changes: Sequence[_JointLimitAudit],
        candidates: Mapping[str, SchemaResolverManager._InterpretedPolicyValues],
    ) -> None:
        """Audit the inputs that contribute to assembled joint-limit changes."""
        changed = set()
        for change in changes:
            if not _values_equal(change.legacy.ke, change.composed.ke):
                changed.update((change.legacy_owners[0], change.composed_owners[0]))
            if not _values_equal(change.legacy.kd, change.composed.kd):
                changed.update((change.legacy_owners[1], change.composed_owners[1]))
            if change.legacy.solref_mode != change.composed.solref_mode:
                changed.update((*change.legacy_owners, *change.composed_owners))

        self._resolver._audit_assembled_property(
            prim,
            PrimType.JOINT,
            tuple(change.legacy.comparison for change in changes),
            tuple(change.composed.comparison for change in changes),
            tuple(
                policies.contribution(key=owner, compare_source=True)
                for owner, policies in candidates.items()
                if owner in changed
            ),
        )

    def _get_mjc_joint_limit_default(self, prim: Any, key: str) -> float | None:
        resolver = self._mjc_resolver
        if resolver is None or self._mjc_schema_is_applied is None or not self._mjc_schema_is_applied(prim, key):
            return None
        spec = resolver.mapping.get(PrimType.JOINT, {}).get(key)
        if spec is None or spec.default is None:
            return None
        if spec.usd_value_transformer is not None:
            return spec.usd_value_transformer(spec.default)
        return spec.default

    @staticmethod
    def _interpret_joint_limit_gain(
        resolved: _ResolvedValue,
        builder_default: float,
    ) -> tuple[float, Literal["force", "mjc_authored", "mjc_default"]]:
        value = builder_default if resolved.value is None else resolved.value
        if resolved.resolver is None or resolved.resolver.name != "mjc":
            return value, "force"
        return value, "mjc_authored" if resolved.authored else "mjc_default"

    def _interpret_joint_velocity_limit(self, resolved: _ResolvedValue, *, is_revolute: bool) -> float:
        value = _interpret_usd_joint_velocity_limit(resolved.value)
        if value is None:
            return self._default_joint_velocity_limit
        if is_revolute and resolved.source != _ValueSource.IMPORTER_DEFAULT:
            value *= self._degrees_to_radian
        return value

    def _interpret_joint_damping(self, resolved: _ResolvedValue, *, is_revolute: bool) -> float:
        raw_value = resolved.value
        legacy_angular_unit = None
        if isinstance(raw_value, self._JointDampingValue):
            value = raw_value.value
            legacy_angular_unit = raw_value.angular_unit
        else:
            value = self._default_joint_damping if raw_value is None else raw_value
        if is_revolute and resolved.source not in (_ValueSource.IMPORTER_DEFAULT, _ValueSource.UNRESOLVED):
            resolver = resolved.winning_resolver
            spec = resolver.mapping.get(PrimType.JOINT, {}).get("damping") if resolver is not None else None
            angular_unit = legacy_angular_unit or (spec.angular_unit if spec is not None else "degrees")
            if angular_unit == "degrees":
                value /= self._degrees_to_radian
        return value

    def _resolve_joint_dof_property(
        self,
        prim: Any,
        key: str,
        *,
        revolute: tuple[bool, ...],
        default,
        legacy_default,
        interpreter,
        resolve_legacy=None,
        read_value=None,
        authored_aliases=(),
        audit_key=None,
    ) -> tuple[float, ...]:
        def interpret(resolved: _ResolvedValue) -> tuple[float, ...]:
            return tuple(interpreter(resolved, is_revolute=value) for value in revolute)

        policies = self._resolver._resolve_interpreted_policies(
            prim,
            PrimType.JOINT,
            key,
            default,
            legacy_default=legacy_default,
            interpreter=interpret,
            resolve_legacy=resolve_legacy,
            read_value=read_value,
            authored_aliases=authored_aliases,
        )

        active = policies.active.value
        if policies.legacy is not None and policies.composed is not None:
            self._resolver._audit_assembled_property(
                prim,
                PrimType.JOINT,
                policies.legacy.value,
                policies.composed.value,
                (policies.contribution(key=key if audit_key is None else audit_key(policies)),),
            )
        return active
