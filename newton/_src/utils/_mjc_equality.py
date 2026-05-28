# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import warp as wp

from ..core.types import vec5
from ..sim.enums import EqType


def mjc_eq_solref(custom_attrs: dict[str, Any]) -> wp.vec2:
    """Return MuJoCo equality solref from parsed custom attributes or the MuJoCo default."""
    return custom_attrs.get("mujoco:eq_solref", wp.vec2(0.02, 1.0))


def mjc_eq_solimp(custom_attrs: dict[str, Any]) -> vec5:
    """Return MuJoCo equality solimp from parsed custom attributes or the MuJoCo default."""
    return custom_attrs.get("mujoco:eq_solimp", vec5(0.9, 0.95, 0.001, 0.5, 2.0))


def mjc_joint_eq_custom_attrs(
    eq_type: EqType,
    body1: int,
    body2: int,
    anchor: wp.vec3,
    relpose: wp.transform | None,
    torquescale: float,
    custom_attrs: dict[str, Any],
) -> dict[str, Any]:
    """Build custom attributes that preserve a converted MuJoCo CONNECT/WELD equality."""
    return {
        "mujoco:joint_eq_type": int(eq_type),
        "mujoco:joint_eq_body1": body1,
        "mujoco:joint_eq_body2": body2,
        "mujoco:joint_eq_anchor": anchor,
        "mujoco:joint_eq_relpose": relpose or wp.transform_identity(),
        "mujoco:joint_eq_torquescale": torquescale,
        "mujoco:joint_eq_solref": mjc_eq_solref(custom_attrs),
        "mujoco:joint_eq_solimp": mjc_eq_solimp(custom_attrs),
    }


def mjc_mimic_eq_custom_attrs(polycoef: Sequence[float], custom_attrs: dict[str, Any]) -> dict[str, Any]:
    """Build custom attributes that preserve a converted MuJoCo JOINT equality."""
    return {
        "mujoco:mimic_eq_preserve": True,
        "mujoco:mimic_eq_polycoef": vec5(*polycoef),
        "mujoco:mimic_eq_solref": mjc_eq_solref(custom_attrs),
        "mujoco:mimic_eq_solimp": mjc_eq_solimp(custom_attrs),
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
) -> int:
    """Add a Newton loop joint that preserves a MuJoCo CONNECT or WELD equality."""
    parent, child, parent_xform, child_xform = mjc_loop_joint_xforms(builder, body1, body2, anchor)
    add_joint = builder.add_joint_ball if eq_type == EqType.CONNECT else builder.add_joint_fixed
    return add_joint(
        parent=parent,
        child=child,
        parent_xform=parent_xform,
        child_xform=child_xform,
        label=label,
        enabled=enabled,
        custom_attributes=mjc_joint_eq_custom_attrs(
            eq_type,
            body1,
            body2,
            anchor,
            relpose,
            torquescale,
            custom_attrs,
        ),
    )
