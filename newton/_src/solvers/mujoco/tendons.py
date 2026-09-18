# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Programmatic authoring helpers for MuJoCo tendons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...geometry import ShapeFlags
from ...sim import ModelBuilder
from ._authoring import _ensure_mujoco_attributes, _prepare_custom_frequency_row


@dataclass(frozen=True)
class TendonWrapSite:
    """A site entry in a spatial tendon path."""

    site: int
    """Newton site-shape index."""


@dataclass(frozen=True)
class TendonWrapGeom:
    """A geometry wrapping entry in a spatial tendon path."""

    geom: int
    """Newton non-site shape index."""

    sidesite: int | None = None
    """Optional Newton site-shape index selecting the wrapping side."""


@dataclass(frozen=True)
class TendonWrapPulley:
    """A pulley entry in a spatial tendon path."""

    divisor: float
    """Positive pulley divisor."""


def _vector(builder: ModelBuilder, key: str, values: Sequence[float], length: int) -> Any:
    components = [float(value) for value in values]
    if len(components) != length:
        raise ValueError(f"{key} requires exactly {length} values, got {len(components)}.")
    return builder.custom_attributes[key].dtype(*components)


def _tristate(value: bool | int | str) -> int:
    if isinstance(value, str):
        try:
            return {"false": 0, "true": 1, "auto": 2}[value.lower().strip()]
        except KeyError as error:
            raise ValueError(f"Expected false, true, or auto, got {value!r}.") from error
    result = int(value)
    if result not in (0, 1, 2):
        raise ValueError(f"Expected a MuJoCo tri-state value in {{0, 1, 2}}, got {value!r}.")
    return result


def _validate_shape(builder: ModelBuilder, shape: int, *, site: bool, name: str) -> None:
    shape_count = len(builder.shape_flags)
    if shape < 0 or shape >= shape_count:
        raise IndexError(f"{name} index {shape} is outside [0, {shape_count}).")
    is_site = bool(int(builder.shape_flags[shape]) & int(ShapeFlags.SITE))
    if is_site != site:
        expected = "site" if site else "non-site shape"
        raise ValueError(f"{name} index {shape} does not identify a Newton {expected}.")


def _tendon_values(
    builder: ModelBuilder,
    *,
    tendon_type: int,
    joint_adr: int,
    joint_num: int,
    wrap_adr: int,
    wrap_num: int,
    label: str | None,
    stiffness: float,
    damping: float,
    frictionloss: float,
    limited: bool | int | str,
    limit_range: Sequence[float],
    margin: float,
    solref_limit: Sequence[float],
    solimp_limit: Sequence[float],
    solref_friction: Sequence[float],
    solimp_friction: Sequence[float],
    armature: float,
    springlength: Sequence[float],
) -> dict[str, Any]:
    return {
        "mujoco:tendon_world": builder.current_world,
        "mujoco:tendon_stiffness": float(stiffness),
        "mujoco:tendon_damping": float(damping),
        "mujoco:tendon_frictionloss": float(frictionloss),
        "mujoco:tendon_limited": _tristate(limited),
        "mujoco:tendon_range": _vector(builder, "mujoco:tendon_range", limit_range, 2),
        "mujoco:tendon_margin": float(margin),
        "mujoco:tendon_solref_limit": _vector(builder, "mujoco:tendon_solref_limit", solref_limit, 2),
        "mujoco:tendon_solimp_limit": _vector(builder, "mujoco:tendon_solimp_limit", solimp_limit, 5),
        "mujoco:tendon_solref_friction": _vector(builder, "mujoco:tendon_solref_friction", solref_friction, 2),
        "mujoco:tendon_solimp_friction": _vector(builder, "mujoco:tendon_solimp_friction", solimp_friction, 5),
        "mujoco:tendon_armature": float(armature),
        "mujoco:tendon_springlength": _vector(builder, "mujoco:tendon_springlength", springlength, 2),
        "mujoco:tendon_joint_adr": int(joint_adr),
        "mujoco:tendon_joint_num": int(joint_num),
        "mujoco:tendon_label": label or "",
        "mujoco:tendon_type": int(tendon_type),
        "mujoco:tendon_wrap_adr": int(wrap_adr),
        "mujoco:tendon_wrap_num": int(wrap_num),
    }


def add_tendon_fixed(
    builder: ModelBuilder,
    joints: Sequence[tuple[int, float]],
    *,
    label: str | None = None,
    stiffness: float = 0.0,
    damping: float = 0.0,
    frictionloss: float = 0.0,
    limited: bool | int | str = "auto",
    limit_range: Sequence[float] = (0.0, 0.0),
    margin: float = 0.0,
    solref_limit: Sequence[float] = (0.02, 1.0),
    solimp_limit: Sequence[float] = (0.9, 0.95, 0.001, 0.5, 2.0),
    solref_friction: Sequence[float] = (0.02, 1.0),
    solimp_friction: Sequence[float] = (0.9, 0.95, 0.001, 0.5, 2.0),
    armature: float = 0.0,
    springlength: Sequence[float] = (-1.0, -1.0),
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a fixed tendon as a linear combination of Newton joints.

    Args:
        builder: Model builder receiving the tendon.
        joints: Ordered ``(joint_index, coefficient)`` entries.
        label: Optional tendon label.
        stiffness: Tendon stiffness [N/m].
        damping: Tendon damping [N·s/m].
        frictionloss: Tendon friction loss [N].
        limited: Tendon-limit tri-state (false, true, or auto).
        limit_range: Tendon length range [m].
        margin: Tendon limit margin [m].
        solref_limit: Limit constraint reference parameters.
        solimp_limit: Limit constraint impedance parameters.
        solref_friction: Friction constraint reference parameters.
        solimp_friction: Friction constraint impedance parameters.
        armature: Tendon armature [kg].
        springlength: Tendon spring length range [m].
        custom_attributes: Additional registered ``mujoco:tendon`` attributes.

    Returns:
        The ``mujoco:tendon`` row index.
    """
    _ensure_mujoco_attributes(builder, "mujoco:tendon_world")
    entries = [(int(joint), float(coef)) for joint, coef in joints]
    if not entries:
        raise ValueError("A fixed tendon requires at least one joint entry.")
    joint_count = len(builder.joint_type)
    for joint, _ in entries:
        if joint < 0 or joint >= joint_count:
            raise IndexError(f"joint index {joint} is outside [0, {joint_count}).")

    joint_start = builder._custom_frequency_counts.get("mujoco:tendon_joint", 0)
    values = _tendon_values(
        builder,
        tendon_type=0,
        joint_adr=joint_start,
        joint_num=len(entries),
        wrap_adr=0,
        wrap_num=0,
        label=label,
        stiffness=stiffness,
        damping=damping,
        frictionloss=frictionloss,
        limited=limited,
        limit_range=limit_range,
        margin=margin,
        solref_limit=solref_limit,
        solimp_limit=solimp_limit,
        solref_friction=solref_friction,
        solimp_friction=solimp_friction,
        armature=armature,
        springlength=springlength,
    )
    row = _prepare_custom_frequency_row(builder, "mujoco:tendon", values, custom_attributes)

    for joint, coef in entries:
        builder.add_custom_values(**{"mujoco:tendon_joint": joint, "mujoco:tendon_coef": coef})
    return builder.add_custom_values(**row)["mujoco:tendon_world"]


def add_tendon_spatial(
    builder: ModelBuilder,
    path: Sequence[TendonWrapSite | TendonWrapGeom | TendonWrapPulley],
    *,
    label: str | None = None,
    stiffness: float = 0.0,
    damping: float = 0.0,
    frictionloss: float = 0.0,
    limited: bool | int | str = "auto",
    limit_range: Sequence[float] = (0.0, 0.0),
    margin: float = 0.0,
    solref_limit: Sequence[float] = (0.02, 1.0),
    solimp_limit: Sequence[float] = (0.9, 0.95, 0.001, 0.5, 2.0),
    solref_friction: Sequence[float] = (0.02, 1.0),
    solimp_friction: Sequence[float] = (0.9, 0.95, 0.001, 0.5, 2.0),
    armature: float = 0.0,
    springlength: Sequence[float] = (-1.0, -1.0),
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a spatial tendon from typed site, geometry, and pulley entries.

    Args:
        builder: Model builder receiving the tendon.
        path: Ordered spatial tendon path entries.
        label: Optional tendon label.
        stiffness: Tendon stiffness [N/m].
        damping: Tendon damping [N·s/m].
        frictionloss: Tendon friction loss [N].
        limited: Tendon-limit tri-state (false, true, or auto).
        limit_range: Tendon length range [m].
        margin: Tendon limit margin [m].
        solref_limit: Limit constraint reference parameters.
        solimp_limit: Limit constraint impedance parameters.
        solref_friction: Friction constraint reference parameters.
        solimp_friction: Friction constraint impedance parameters.
        armature: Tendon armature [kg].
        springlength: Tendon spring length range [m].
        custom_attributes: Additional registered ``mujoco:tendon`` attributes.

    Returns:
        The ``mujoco:tendon`` row index.
    """
    _ensure_mujoco_attributes(builder, "mujoco:tendon_world")
    entries: list[tuple[int, int, int, float]] = []
    for entry in path:
        if isinstance(entry, TendonWrapSite):
            _validate_shape(builder, entry.site, site=True, name="site")
            entries.append((0, int(entry.site), -1, 0.0))
        elif isinstance(entry, TendonWrapGeom):
            _validate_shape(builder, entry.geom, site=False, name="geom")
            sidesite = -1 if entry.sidesite is None else int(entry.sidesite)
            if sidesite >= 0:
                _validate_shape(builder, sidesite, site=True, name="sidesite")
            entries.append((1, int(entry.geom), sidesite, 0.0))
        elif isinstance(entry, TendonWrapPulley):
            if entry.divisor <= 0.0:
                raise ValueError("A tendon pulley divisor must be positive.")
            entries.append((2, -1, -1, float(entry.divisor)))
        else:
            raise TypeError(f"Unsupported spatial tendon path entry {type(entry).__name__}.")
    if not entries:
        raise ValueError("A spatial tendon requires at least one path entry.")

    wrap_start = builder._custom_frequency_counts.get("mujoco:tendon_wrap", 0)
    values = _tendon_values(
        builder,
        tendon_type=1,
        joint_adr=0,
        joint_num=0,
        wrap_adr=wrap_start,
        wrap_num=len(entries),
        label=label,
        stiffness=stiffness,
        damping=damping,
        frictionloss=frictionloss,
        limited=limited,
        limit_range=limit_range,
        margin=margin,
        solref_limit=solref_limit,
        solimp_limit=solimp_limit,
        solref_friction=solref_friction,
        solimp_friction=solimp_friction,
        armature=armature,
        springlength=springlength,
    )
    row = _prepare_custom_frequency_row(builder, "mujoco:tendon", values, custom_attributes)

    for wrap_type, shape, sidesite, parameter in entries:
        builder.add_custom_values(
            **{
                "mujoco:tendon_wrap_type": wrap_type,
                "mujoco:tendon_wrap_shape": shape,
                "mujoco:tendon_wrap_sidesite": sidesite,
                "mujoco:tendon_wrap_prm": parameter,
            }
        )
    return builder.add_custom_values(**row)["mujoco:tendon_world"]


__all__ = [
    "TendonWrapGeom",
    "TendonWrapPulley",
    "TendonWrapSite",
    "add_tendon_fixed",
    "add_tendon_spatial",
]
