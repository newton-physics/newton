# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
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

logger = logging.getLogger(__name__)


def solref_invalid_mask(solref: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a mask selecting MuJoCo solref pairs with invalid zero or sign components.

    MuJoCo's all-zero inherit-default sentinel is valid.
    """
    solref = np.asarray(solref)
    timeconst = solref[..., 0]
    dampratio = solref[..., 1]
    both_zero = (timeconst == 0.0) & (dampratio == 0.0)
    invalid = (timeconst == 0.0) | (dampratio == 0.0) | (np.sign(timeconst) != np.sign(dampratio))
    return np.asarray(invalid & ~both_zero)


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
    """Parse a MuJoCo five-term equality polynomial, filling omitted terms from MuJoCo's ``0 1 0 0 0`` default."""
    if isinstance(polycoef, str):
        values = [float(x) for x in polycoef.split()]
    else:
        values = [float(x) for x in polycoef]
    return (values + [0.0, 1.0, 0.0, 0.0, 0.0][len(values) :])[:5]


def mjc_polycoef_has_higher_order(polycoef: Sequence[float]) -> bool:
    """Return True when a MuJoCo JOINT equality uses quadratic or higher-order terms."""
    return any(float(value) != 0.0 for value in polycoef[2:5])


def mjc_loop_joint_xforms(
    builder: Any,
    body1: int,
    body2: int,
    anchor: wp.vec3,
    relpose: wp.transform | None = None,
    weld: bool = False,
    world_xform: wp.transform | None = None,
) -> tuple[int, int, wp.transform, wp.transform]:
    """Compute Newton loop-joint endpoints and local frames for a MuJoCo body equality.

    A CONNECT anchor is a point in ``body1``'s frame. A WELD follows MuJoCo: ``anchor`` is a point in ``body2``'s frame
    (the world when ``body2`` is -1) that is held at ``relpose``'s position in ``body1``'s frame, and ``relpose``'s
    rotation is ``body2``'s orientation relative to ``body1``. A ``relpose`` whose quaternion is all zeros (MuJoCo's
    default) keeps the relative pose of the builder's current body poses. ``world_xform`` maps a weld's world-side
    ``anchor`` or ``relpose``, given in asset coordinates, into the builder's world.
    """
    if body2 >= 0:
        parent = body1
        child = body2
    elif body1 >= 0:
        parent = -1
        child = body1
    else:
        raise ValueError("At least one body is required for converted MuJoCo equality constraints.")

    if weld:
        world = world_xform if world_xform is not None else wp.transform_identity()

        def pose(body: int) -> wp.transform:
            return builder.body_q[body] if body >= 0 else world

        def local(body: int, frame: wp.transform) -> wp.transform:
            return wp.transform_multiply(wp.transform_inverse(builder.body_q[body]), frame) if body >= 0 else frame

        anchor_frame = wp.transform(anchor, wp.quat_identity())
        if relpose is None or not any(float(x) != 0.0 for x in relpose.q):
            # Both frames sit on body2's anchor frame at the current poses, so the weld holds them where they are.
            world_frame = wp.transform_multiply(pose(body2), anchor_frame)
            return parent, child, local(parent, world_frame), local(child, world_frame)
        # MuJoCo's weld residual: body1 * relpose.p == body2 * anchor and body1.q * relpose.q == body2.q
        frame1 = relpose if body1 >= 0 else wp.transform_multiply(world, relpose)
        frame2 = anchor_frame if body2 >= 0 else wp.transform_multiply(world, anchor_frame)
        if body2 >= 0:
            return parent, child, frame1, frame2
        return parent, child, frame2, frame1

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
    world_xform: wp.transform | None = None,
) -> tuple[int, int]:
    """Add a Newton loop joint and its authoritative MuJoCo equality row.

    A CONNECT becomes a BALL loop joint. A WELD becomes a FIXED loop joint, except that ``torquescale == 0`` (which
    MuJoCo enforces as a point constraint with rotation free) becomes a BALL loop joint on the weld's anchor. Solvers
    that enforce Newton loop joints, such as :class:`~newton.solvers.SolverSteel`, hold them as hard constraints:
    ``torquescale`` only selects BALL or FIXED, and ``solref``/``solimp`` are not used. SolverMuJoCo enforces the
    original equality row, so it keeps all of them. ``world_xform`` places a weld's world-side frame, as in
    :func:`mjc_loop_joint_xforms`; the equality row keeps the values as given.
    """
    weld = eq_type == EqType.WELD
    parent, child, parent_xform, child_xform = mjc_loop_joint_xforms(
        builder, body1, body2, anchor, relpose, weld=weld, world_xform=world_xform
    )
    add_joint = builder.add_joint_fixed if weld else builder.add_joint_ball
    if weld and torquescale == 0.0:
        # The weld frames put the anchor on body2 and its held position (relpose) on body1, which is the point
        # MuJoCo's weld residual pins; a BALL joint on those frames pins it the same way and leaves rotation free.
        add_joint = builder.add_joint_ball
        logger.warning(
            "MuJoCo weld %r has torquescale=0, which MuJoCo enforces as a point constraint with rotation free; it is "
            "converted to a BALL loop joint (rotation free) instead of a FIXED one. SolverMuJoCo enforces the "
            "original weld row.",
            label,
        )
    elif weld and torquescale != 1.0:
        logger.warning(
            "MuJoCo weld %r has torquescale=%g; it is converted to a FIXED loop joint, which solvers that enforce loop "
            "joints (such as SolverSteel) hold as a hard weld: torquescale, solref and solimp are not used. "
            "SolverMuJoCo enforces the original weld row.",
            label,
            torquescale,
        )
    joint_idx = add_joint(
        parent=parent,
        child=child,
        parent_xform=parent_xform,
        child_xform=child_xform,
        label=label,
        enabled=enabled,
    )
    # For WELD, torquescale (beyond choosing BALL or FIXED) and the solref/solimp softness remain on the
    # equality row only; the loop joint's frames carry the anchor and relpose for Newton's other solvers.
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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"ModelBuilder\.add_constraint_mimic\(\) is deprecated",
            category=DeprecationWarning,
        )
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
