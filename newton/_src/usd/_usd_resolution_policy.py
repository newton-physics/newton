# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interpret USD properties for the importer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..sim.enums import JointTargetMode
from ..solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_MJCF_DEFAULT, SOLREF_MODE_RAW
from . import utils as usd
from .schema_resolver import PrimType, SchemaResolver, SchemaResolverManager

if TYPE_CHECKING:
    from pxr import Usd, UsdPhysics


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
