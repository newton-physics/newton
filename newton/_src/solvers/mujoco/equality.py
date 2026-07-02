# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo-specific helpers for constructing equality constraints on a
:class:`~newton.ModelBuilder` via the ``mujoco:equality_constraint_*`` custom
attributes.

Equality constraints are MuJoCo-specific concepts that live on the model under
the ``mujoco`` namespace. The public lower-level path for users is
:meth:`ModelBuilder.add_custom_values` with ``mujoco:equality_constraint_*``
keys; this module provides convenience used internally by the MJCF/USD
importers and by tests.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

import warp as wp

from ...core.types import Transform, Vec3, axis_to_vec3, vec5
from ...sim.model import Model
from .enums import EqType

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder


class MjcEqualityTargetKind(IntEnum):
    """How a MuJoCo equality row is projected into Newton."""

    NONE = 0  # Pure equality row; no projected Newton object, target is -1.
    JOINT = 1  # Target is a Newton joint, used by converted CONNECT/WELD loop joints.
    MIMIC = 2  # Target is a Newton mimic constraint, used by converted JOINT equalities.


# Object kinds an equality row references (mjtObj-aligned).
MJC_OBJ_UNKNOWN = -1
MJC_OBJ_BODY = 1
MJC_OBJ_JOINT = 3


def _register_equality_constraint_attributes(builder: ModelBuilder) -> None:
    """Declare the ``model.mujoco.equality_constraint_*`` custom-attribute rows on ``builder``.

    Registers the per-equality-constraint custom attributes (the ``mujoco:equality_constraint``
    frequency) that back :func:`_add_equality_constraint` and surface on :class:`~newton.Model`
    under the ``mujoco`` namespace after :meth:`~newton.ModelBuilder.finalize`.

    Idempotent: re-registration with the same spec is a no-op. Called from
    :meth:`ModelBuilder.__init__` so the namespaced equality fields exist independently of
    :meth:`SolverMuJoCo.register_custom_attributes`, which also calls it when registering the
    other MuJoCo custom attributes.
    """
    ca = type(builder).CustomAttribute
    eq_freq = "mujoco:equality_constraint"
    model_assignment = Model.AttributeAssignment.MODEL

    # Register the custom frequency before any custom attributes that use it.
    builder.add_custom_frequency(type(builder).CustomFrequency(name="equality_constraint", namespace="mujoco"))

    builder.add_custom_attribute(
        ca(
            name="equality_constraint_type",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=int(EqType.CONNECT),
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_body1",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=-1,
            references="body",
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_body2",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=-1,
            references="body",
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_anchor",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.vec3,
            default=wp.vec3(),
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_torquescale",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.float32,
            default=0.0,
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_relpose",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.transform,
            default=wp.transform_identity(),
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_joint1",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=-1,
            references="joint",
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_joint2",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=-1,
            references="joint",
            namespace="mujoco",
        )
    )
    # polycoef is materialized as ``wp.array2d[wp.float32]`` with shape
    # ``[equality_constraint_count, 5]``; the value stored per entry is a
    # 5-element list of floats so the standard pipeline yields the expected 2D layout.
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_polycoef",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.float32,
            default=[0.0, 0.0, 0.0, 0.0, 0.0],
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_label",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=str,
            default="",
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_enabled",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.bool,
            default=True,
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_world",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=0,
            references="world",
            namespace="mujoco",
        )
    )
    # ``objtype`` disambiguates the body*/joint* references (see :data:`MJC_OBJ_BODY`) so the
    # table can grow to site- and tendon-anchored equalities without a layout change.
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_objtype",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=MJC_OBJ_UNKNOWN,
            namespace="mujoco",
        )
    )
    # ``target_kind`` / ``target`` link a row to the native entity it was projected onto (loop
    # joint or mimic) for solver portability; converted MJCF/USD equalities set them, while pure
    # equality rows keep ``MjcEqualityTargetKind.NONE`` / ``-1``.
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_target_kind",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=int(MjcEqualityTargetKind.NONE),
            namespace="mujoco",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="equality_constraint_target",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=-1,
            namespace="mujoco",
        )
    )
    # MuJoCo solver reference parameters, parsed from USD/MJCF and read back by SolverMuJoCo.
    builder.add_custom_attribute(
        ca(
            name="eq_solref",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.vec2,
            default=wp.vec2(0.02, 1.0),
            namespace="mujoco",
            usd_attribute_name="mjc:solref",
            mjcf_attribute_name="solref",
        )
    )
    builder.add_custom_attribute(
        ca(
            name="eq_solimp",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=vec5,
            default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
            namespace="mujoco",
            usd_attribute_name="mjc:solimp",
            mjcf_attribute_name="solimp",
        )
    )


def _add_equality_constraint(
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
    objtype = MJC_OBJ_JOINT if constraint_type == EqType.JOINT else MJC_OBJ_BODY

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
