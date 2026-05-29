# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo-specific helpers for constructing equality constraints on a
:class:`~newton.ModelBuilder` via the ``mujoco:equality_constraint_*`` custom
attributes.

Equality constraints are MuJoCo-specific concepts that live on the model under
the ``mujoco`` namespace. The public lower-level path for users is
:meth:`ModelBuilder.add_custom_values` with ``mujoco:equality_constraint_*``
keys; this module provides convenience used internally by the MJCF/USD
importers and by tests during the deprecation window for
``ModelBuilder.add_equality_constraint*``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp

from ...core.types import Transform, Vec3, axis_to_vec3
from ...sim.enums import EqObjType, EqType

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder


def add_equality_constraint(
    builder: ModelBuilder,
    constraint_type: EqType,
    body1: int = -1,
    body2: int = -1,
    anchor: Vec3 | None = None,
    torquescale: float | None = None,
    relpose: Transform | None = None,
    joint1: int = -1,
    joint2: int = -1,
    polycoef: list[float] | None = None,
    label: str | None = None,
    enabled: bool = True,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Append a row to the ``mujoco:equality_constraint`` custom-attribute frequency on ``builder``.

    Args:
        builder: Target :class:`~newton.ModelBuilder`.
        constraint_type: Equality constraint type (``EqType.CONNECT``,
            ``EqType.WELD``, or ``EqType.JOINT``).
        body1: Index of the first body (-1 for world).
        body2: Index of the second body (-1 for world).
        anchor: Anchor point on body1. Defaults to the origin.
        torquescale: Angular residual scale for weld. Defaults to ``1.0`` for
            ``EqType.WELD`` and ``0.0`` otherwise.
        relpose: Relative pose of body2 for weld. Defaults to the identity transform.
        joint1: Index of the first joint for joint coupling.
        joint2: Index of the second joint for joint coupling.
        polycoef: Five polynomial coefficients for ``EqType.JOINT`` coupling.
            Defaults to ``[0, 0, 0, 0, 0]``.
        label: Optional constraint label.
        enabled: Whether the constraint is active.
        custom_attributes: Additional ``mujoco:equality_constraint``-frequency
            custom attributes to assign at the new index.

    Returns:
        Index of the new constraint row.
    """
    anchor_vec = axis_to_vec3(anchor) if anchor is not None else wp.vec3()
    relpose_tf = wp.transform(*relpose) if relpose is not None else wp.transform_identity()
    if torquescale is None:
        torquescale_value = 1.0 if constraint_type == EqType.WELD else 0.0
    else:
        torquescale_value = float(torquescale)
    objtype = EqObjType.JOINT if constraint_type == EqType.JOINT else EqObjType.BODY

    indices = builder.add_custom_values(
        **{
            "mujoco:equality_constraint_type": int(constraint_type),
            "mujoco:equality_constraint_objtype": int(objtype),
            "mujoco:equality_constraint_body1": body1,
            "mujoco:equality_constraint_body2": body2,
            "mujoco:equality_constraint_anchor": anchor_vec,
            "mujoco:equality_constraint_torquescale": torquescale_value,
            "mujoco:equality_constraint_relpose": relpose_tf,
            "mujoco:equality_constraint_joint1": joint1,
            "mujoco:equality_constraint_joint2": joint2,
            "mujoco:equality_constraint_polycoef": list(polycoef) if polycoef else [0.0, 0.0, 0.0, 0.0, 0.0],
            "mujoco:equality_constraint_label": label or "",
            "mujoco:equality_constraint_enabled": enabled,
            "mujoco:equality_constraint_world": builder.current_world,
        }
    )
    constraint_idx = indices["mujoco:equality_constraint_type"]

    if custom_attributes:
        builder._process_custom_attributes(
            entity_index=constraint_idx,
            custom_attrs=custom_attributes,
            expected_frequency="mujoco:equality_constraint",
        )

    return constraint_idx
