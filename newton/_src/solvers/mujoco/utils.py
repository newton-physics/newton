# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import warp as wp

from ...core.types import vec5
from .enums import EqType
from .equality import (
    MJC_OBJ_BODY,
    MJC_OBJ_JOINT,
    MJC_OBJ_UNKNOWN,
    MjcEqualityTargetKind,
    _add_equality_constraint,
)


def mjc_eq_solref(custom_attrs: dict[str, Any]) -> wp.vec2:
    """Return MuJoCo equality solref from parsed custom attributes or the MuJoCo default."""
    return custom_attrs.get("mujoco:eq_solref", wp.vec2(0.02, 1.0))


def mjc_eq_solimp(custom_attrs: dict[str, Any]) -> vec5:
    """Return MuJoCo equality solimp from parsed custom attributes or the MuJoCo default."""
    return custom_attrs.get("mujoco:eq_solimp", vec5(0.9, 0.95, 0.001, 0.5, 2.0))


def mjc_eq_custom_attrs(
    custom_attrs: dict[str, Any],
    target_kind: MjcEqualityTargetKind = MjcEqualityTargetKind.NONE,
    target: int = -1,
    objtype: int = MJC_OBJ_UNKNOWN,
) -> dict[str, Any]:
    """Build MuJoCo equality-row custom attributes."""
    return {
        "mujoco:eq_solref": mjc_eq_solref(custom_attrs),
        "mujoco:eq_solimp": mjc_eq_solimp(custom_attrs),
        "mujoco:equality_constraint_target_kind": int(target_kind),
        "mujoco:equality_constraint_target": target,
        "mujoco:equality_constraint_objtype": objtype,
    }


def mjc_parse_polycoef(polycoef: str | Sequence[float]) -> list[float]:
    """Parse a MuJoCo five-term equality polynomial, padding omitted terms with zeros."""
    if isinstance(polycoef, str):
        values = [float(x) for x in polycoef.split()]
    else:
        values = [float(x) for x in polycoef]
    if len(values) < 5:
        values.extend([0.0] * (5 - len(values)))
    return values[:5]


def mjc_polycoef_has_higher_order(polycoef: Sequence[float]) -> bool:
    """Return True when a MuJoCo JOINT equality uses quadratic or higher-order terms."""
    return any(float(value) != 0.0 for value in polycoef[2:5])


def mjc_loop_joint_xforms(
    builder: Any,
    body1: int,
    body2: int,
    anchor: wp.vec3,
) -> tuple[int, int, wp.transform, wp.transform]:
    """Compute Newton loop-joint endpoints and local anchors for a MuJoCo body equality."""
    if body2 >= 0:
        parent = body1
        child = body2
    elif body1 >= 0:
        parent = -1
        child = body1
    else:
        raise ValueError("At least one body is required for converted MuJoCo equality constraints.")

    body1_xform = builder.body_q[body1] if body1 >= 0 else wp.transform_identity()
    child_xform_world = builder.body_q[child]
    world_anchor = wp.transform_point(body1_xform, anchor) if body1 >= 0 else anchor
    if parent >= 0:
        parent_anchor = wp.transform_point(wp.transform_inverse(builder.body_q[parent]), world_anchor)
    else:
        parent_anchor = world_anchor
    child_anchor = wp.transform_point(wp.transform_inverse(child_xform_world), world_anchor)
    return (
        parent,
        child,
        wp.transform(parent_anchor, wp.quat_identity()),
        wp.transform(child_anchor, wp.quat_identity()),
    )


def mjc_add_equality_loop_joint(
    builder: Any,
    eq_type: EqType,
    body1: int,
    body2: int,
    anchor: wp.vec3,
    relpose: wp.transform | None,
    torquescale: float,
    label: str | None,
    enabled: bool,
    custom_attrs: dict[str, Any],
) -> tuple[int, int]:
    """Add a Newton loop joint and its authoritative MuJoCo equality row."""
    parent, child, parent_xform, child_xform = mjc_loop_joint_xforms(builder, body1, body2, anchor)
    add_joint = builder.add_joint_ball if eq_type == EqType.CONNECT else builder.add_joint_fixed
    joint_idx = add_joint(
        parent=parent,
        child=child,
        parent_xform=parent_xform,
        child_xform=child_xform,
        label=label,
        enabled=enabled,
    )
    # For WELD, relpose/torquescale remain on the equality row; the loop joint only
    # gives Newton a projected simulation object.
    eq_idx = _add_equality_constraint(
        builder,
        constraint_type=eq_type,
        body1=body1,
        body2=body2,
        anchor=anchor,
        relpose=relpose,
        torquescale=torquescale,
        label=label,
        enabled=enabled,
        custom_attributes=mjc_eq_custom_attrs(
            custom_attrs,
            target_kind=MjcEqualityTargetKind.JOINT,
            target=joint_idx,
            objtype=MJC_OBJ_BODY,
        ),
    )
    return eq_idx, joint_idx


def mjc_add_equality_mimic(
    builder: Any,
    joint1: int,
    joint2: int,
    polycoef: Sequence[float],
    label: str | None,
    enabled: bool,
    custom_attrs: dict[str, Any],
) -> tuple[int, int]:
    """Add a Newton mimic constraint and its authoritative MuJoCo equality row."""
    mimic_idx = builder.add_constraint_mimic(
        joint0=joint1,
        joint1=joint2,
        coef0=polycoef[0],
        coef1=polycoef[1],
        label=label,
        enabled=enabled,
    )
    eq_idx = _add_equality_constraint(
        builder,
        constraint_type=EqType.JOINT,
        joint1=joint1,
        joint2=joint2,
        polycoef=list(polycoef),
        label=label,
        enabled=enabled,
        custom_attributes=mjc_eq_custom_attrs(
            custom_attrs,
            target_kind=MjcEqualityTargetKind.MIMIC,
            target=mimic_idx,
            objtype=MJC_OBJ_JOINT,
        ),
    )
    return eq_idx, mimic_idx
