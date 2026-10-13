# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo-specific helpers for constructing equality constraints on a
:class:`~newton.ModelBuilder` via the ``mujoco:equality_constraint_*`` custom
attributes.

Equality constraints are MuJoCo-specific concepts that live on the model under
the ``mujoco`` namespace. Public typed helpers append complete constraint rows;
the lower-level :meth:`ModelBuilder.add_custom_values` path remains available
for custom schemas.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
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


def _get_equality_constraint_count(model: Model) -> int:
    """Return zero for models that did not register MuJoCo equality attributes."""
    return getattr(getattr(model, "mujoco", None), "equality_constraint_count", 0)


def _eq_attr(builder: ModelBuilder, name: str) -> ModelBuilder.CustomAttribute:
    """Read an equality attribute by its unqualified name."""
    return builder.custom_attributes[f"mujoco:{name}"]


def _remap_equality_targets(values: list[Any], context: dict[str, Any]) -> list[Any]:
    """Offset each equality target according to its sibling target kind."""
    kind_attr = _eq_attr(context["builder"], "equality_constraint_target_kind")
    kinds = kind_attr.values or []
    offsets = {
        MjcEqualityTargetKind.JOINT: context["entity_offsets"]["joint"],
        MjcEqualityTargetKind.MIMIC: context["entity_offsets"]["constraint_mimic"],
    }
    result = []
    for row, value in zip(context["row_indices"], values, strict=True):
        kind = kinds[row] if row < len(kinds) and kinds[row] is not None else kind_attr.default
        try:
            target = int(value)
            offset = offsets.get(int(kind), 0)
        except (TypeError, ValueError):
            result.append(value)
            continue
        result.append(target + offset if target >= 0 and offset else value)
    return result


def _validate_equality_constraints(builder: ModelBuilder) -> None:
    """Validate body and joint references, including default-filled rows."""
    count = builder._custom_frequency_counts["mujoco:equality_constraint"]
    if count == 0:
        return
    labels = _eq_attr(builder, "equality_constraint_label").values or []
    for name, entity, entity_count in (
        ("equality_constraint_body1", "body", builder.body_count),
        ("equality_constraint_body2", "body", builder.body_count),
        ("equality_constraint_joint1", "joint", builder.joint_count),
        ("equality_constraint_joint2", "joint", builder.joint_count),
    ):
        attr = _eq_attr(builder, name)
        values = attr.values or []
        indices = np.fromiter(
            (values[i] if i < len(values) and values[i] is not None else attr.default for i in range(count)),
            dtype=np.int32,
            count=count,
        )
        invalid = np.flatnonzero((indices < -1) | (indices >= entity_count))
        if len(invalid):
            row = int(invalid[0])
            label = labels[row] if row < len(labels) else None
            label = label or f"equality_constraint_{row}"
            raise ValueError(
                f"Invalid {entity} reference in {name}: constraint {row} ('{label}') references {entity} {indices[row]}, "
                f"but valid range is [-1, {entity_count - 1}] ({entity}_count={entity_count})."
            )


def _finalize_equality_attribute(builder: ModelBuilder, model: Model, attribute: ModelBuilder.CustomAttribute) -> None:
    """Keep registered equality arrays available even when there are no rows."""
    count = model.custom_frequency_counts[attribute.frequency]
    model.add_attribute(
        attribute.name,
        attribute.build_array(count, device=model.device, requires_grad=model.requires_grad),
        attribute.frequency,
        attribute.assignment,
        attribute.namespace,
        attribute.references,
    )


def _collapse_equality_constraints(builder: ModelBuilder, context: dict[str, Any]) -> None:
    """Transform equality frames and projected targets before reference remapping."""
    body_merged_parent = context["body_merged_parent"]
    body_merged_transform = context["body_merged_transform"]
    joint_remap = context["joint_remap"]
    verbose = context["verbose"]
    # Each ``*_values`` is the ``list`` backing the string-frequency CustomAttribute. These
    # lists are sparse: ``add_custom_values`` only populates the fields that were supplied,
    # so an omitted optional field can be ``None``, shorter than the row count, or absent
    # entirely. Reads go through ``_at`` (default fallback), and writes pad the list to the
    # row index before assignment.
    body1_attr = _eq_attr(builder, "equality_constraint_body1")
    body2_attr = _eq_attr(builder, "equality_constraint_body2")
    type_attr = _eq_attr(builder, "equality_constraint_type")
    anchor_attr = _eq_attr(builder, "equality_constraint_anchor")
    relpose_attr = _eq_attr(builder, "equality_constraint_relpose")
    joint1_attr = _eq_attr(builder, "equality_constraint_joint1")
    joint2_attr = _eq_attr(builder, "equality_constraint_joint2")
    enabled_attr = _eq_attr(builder, "equality_constraint_enabled")
    body1_values = body1_attr.values or []
    body2_values = body2_attr.values or []
    type_values = type_attr.values or []
    if anchor_attr.values is None:
        anchor_attr.values = []
    anchor_values = anchor_attr.values
    if relpose_attr.values is None:
        relpose_attr.values = []
    relpose_values = relpose_attr.values
    joint1_values = joint1_attr.values or []
    joint2_values = joint2_attr.values or []
    if enabled_attr.values is None:
        enabled_attr.values = []
    enabled_values = enabled_attr.values

    def _at(values: list, idx: int, default: Any) -> Any:
        if idx >= len(values):
            return default
        return default if values[idx] is None else values[idx]

    # The builder remaps body1/body2/joint1/joint2 through the ``references`` field.
    # Here we only apply the MuJoCo-specific fixups that depend
    # on the original indices: anchor/relpose frame transforms when a referenced body was
    # merged into its parent, and disabling rows whose joint reference was removed.
    for i in range(builder._custom_frequency_counts["mujoco:equality_constraint"]):
        old_body1 = _at(body1_values, i, body1_attr.default)
        old_body2 = _at(body2_values, i, body2_attr.default)
        body1_was_merged = old_body1 in body_merged_parent
        body2_was_merged = old_body2 in body_merged_parent

        constraint_type = _at(type_values, i, type_attr.default)

        # Transform anchor/relpose from merged body's frame to parent body's frame
        if body1_was_merged:
            merge_xform = body_merged_transform[old_body1]
            if constraint_type == EqType.CONNECT:
                anchor = axis_to_vec3(_at(anchor_values, i, anchor_attr.default))
                while len(anchor_values) <= i:
                    anchor_values.append(None)
                anchor_values[i] = wp.transform_point(merge_xform, anchor)
            if constraint_type == EqType.WELD:
                relpose = _at(relpose_values, i, relpose_attr.default)
                while len(relpose_values) <= i:
                    relpose_values.append(None)
                relpose_values[i] = merge_xform * relpose

        if body2_was_merged and constraint_type == EqType.WELD:
            merge_xform = body_merged_transform[old_body2]
            anchor = axis_to_vec3(_at(anchor_values, i, anchor_attr.default))
            relpose = _at(relpose_values, i, relpose_attr.default)
            while len(anchor_values) <= i:
                anchor_values.append(None)
            while len(relpose_values) <= i:
                relpose_values.append(None)
            anchor_values[i] = wp.transform_point(merge_xform, anchor)
            relpose_values[i] = relpose * wp.transform_inverse(merge_xform)

        old_joint1 = _at(joint1_values, i, joint1_attr.default)
        old_joint2 = _at(joint2_values, i, joint2_attr.default)

        if old_joint1 != -1 and old_joint1 not in joint_remap:
            if verbose:
                print(f"Warning: Equality constraint references removed joint {old_joint1}, disabling constraint")
            while len(enabled_values) <= i:
                enabled_values.append(None)
            enabled_values[i] = False

        if old_joint2 != -1 and old_joint2 not in joint_remap:
            if verbose:
                print(f"Warning: Equality constraint references removed joint {old_joint2}, disabling constraint")
            while len(enabled_values) <= i:
                enabled_values.append(None)
            enabled_values[i] = False

    target_kind_attr = builder.custom_attributes.get("mujoco:equality_constraint_target_kind")
    target_attr = builder.custom_attributes.get("mujoco:equality_constraint_target")
    if target_kind_attr is not None and target_attr is not None and target_attr.values:
        for eq_idx in range(len(target_attr.values)):
            target = target_attr.values[eq_idx]
            if target is None:
                continue
            target_kind_value = (
                target_kind_attr.values[eq_idx]
                if target_kind_attr.values and eq_idx < len(target_kind_attr.values)
                else None
            )
            try:
                target_kind = int(target_kind_value) if target_kind_value is not None else 0
                old_target = int(target)
            except (TypeError, ValueError):
                continue
            if old_target < 0:
                continue
            if target_kind == MjcEqualityTargetKind.JOINT:
                if old_target in joint_remap:
                    target_attr.values[eq_idx] = joint_remap[old_target]
                else:
                    target_attr.values[eq_idx] = -1
                    target_kind_attr.values[eq_idx] = int(MjcEqualityTargetKind.NONE)
            elif target_kind == MjcEqualityTargetKind.MIMIC and old_target >= len(builder.constraint_mimic_joint0):
                target_attr.values[eq_idx] = -1
                target_kind_attr.values[eq_idx] = int(MjcEqualityTargetKind.NONE)


def _register_equality_constraint_attributes(builder: ModelBuilder) -> None:
    """Declare the ``model.mujoco.equality_constraint_*`` custom-attribute rows on ``builder``.

    Registers the per-equality-constraint custom attributes (the ``mujoco:equality_constraint``
    frequency) that back :func:`_add_equality_constraint` and surface on :class:`~newton.Model`
    under the ``mujoco`` namespace after :meth:`~newton.ModelBuilder.finalize`.

    Idempotent: re-registration with the same spec is a no-op. The equality
    authoring helpers and importers register these fields on demand, independently
    of :meth:`SolverMuJoCo.register_custom_attributes`.
    """
    ca = type(builder).CustomAttribute

    def add_attribute(attribute: ModelBuilder.CustomAttribute) -> None:
        builder.add_custom_attribute(attribute)
        builder._add_custom_attribute_model_finalizer(attribute.key, _finalize_equality_attribute)

    eq_freq = "mujoco:equality_constraint"
    model_assignment = Model.AttributeAssignment.MODEL

    # Register the custom frequency before any custom attributes that use it.
    builder.add_custom_frequency(
        type(builder).CustomFrequency(
            name="equality_constraint",
            namespace="mujoco",
            label_attribute="mujoco:equality_constraint_label",
            world_attribute="mujoco:equality_constraint_world",
            builder_validator=_validate_equality_constraints,
            collapse_transformer=_collapse_equality_constraints,
        )
    )

    add_attribute(
        ca(
            name="equality_constraint_type",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=int(EqType.CONNECT),
            namespace="mujoco",
        )
    )
    add_attribute(
        ca(
            name="equality_constraint_body1",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=-1,
            references="body",
            preserve_referenced_bodies=True,
            namespace="mujoco",
        )
    )
    add_attribute(
        ca(
            name="equality_constraint_body2",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=-1,
            references="body",
            preserve_referenced_bodies=True,
            namespace="mujoco",
        )
    )
    add_attribute(
        ca(
            name="equality_constraint_anchor",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.vec3,
            default=wp.vec3(),
            namespace="mujoco",
        )
    )
    add_attribute(
        ca(
            name="equality_constraint_torquescale",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.float32,
            default=0.0,
            namespace="mujoco",
        )
    )
    add_attribute(
        ca(
            name="equality_constraint_relpose",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.transform,
            default=wp.transform_identity(),
            namespace="mujoco",
        )
    )
    add_attribute(
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
    add_attribute(
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
    add_attribute(
        ca(
            name="equality_constraint_polycoef",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.float32,
            default=[0.0, 0.0, 0.0, 0.0, 0.0],
            namespace="mujoco",
        )
    )
    add_attribute(
        ca(
            name="equality_constraint_label",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=str,
            default="",
            namespace="mujoco",
        )
    )
    add_attribute(
        ca(
            name="equality_constraint_enabled",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.bool,
            default=True,
            namespace="mujoco",
        )
    )
    add_attribute(
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
    add_attribute(
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
    add_attribute(
        ca(
            name="equality_constraint_target_kind",
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=int(MjcEqualityTargetKind.NONE),
            namespace="mujoco",
        )
    )
    add_attribute(
        ca(
            name="equality_constraint_target",
            reference_value_transformer=_remap_equality_targets,
            frequency=eq_freq,
            assignment=model_assignment,
            dtype=wp.int32,
            default=-1,
            namespace="mujoco",
        )
    )
    # MuJoCo solver reference parameters, parsed from USD/MJCF and read back by SolverMuJoCo.
    add_attribute(
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
    add_attribute(
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
    *,
    validate_custom_attributes: bool = False,
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
        validate_custom_attributes: Reject managed-column overrides and validate extras before appending.

    Returns:
        Index of the new constraint row.
    """
    _ensure_equality_attributes(builder)
    anchor_vec = axis_to_vec3(anchor) if anchor is not None else wp.vec3()
    relpose_tf = wp.transform(*relpose) if relpose is not None else wp.transform_identity()
    if torquescale is None:
        torquescale_value = 1.0 if constraint_type == EqType.WELD else 0.0
    else:
        torquescale_value = float(torquescale)
    objtype = MJC_OBJ_JOINT if constraint_type == EqType.JOINT else MJC_OBJ_BODY

    values = {
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
    if validate_custom_attributes:
        # Importers may override columns when projecting constraints onto native
        # entities; public helpers accept only additional attributes.
        from ._authoring import _prepare_custom_frequency_row  # noqa: PLC0415

        values = _prepare_custom_frequency_row(builder, "mujoco:equality_constraint", values, custom_attributes)
        custom_attributes = None
    indices = builder.add_custom_values(**values)
    constraint_idx = indices["mujoco:equality_constraint_type"]

    if custom_attributes:
        builder._process_custom_attributes(
            entity_index=constraint_idx,
            custom_attrs=custom_attributes,
            expected_frequency="mujoco:equality_constraint",
        )

    return constraint_idx


def _validate_body(builder: ModelBuilder, body: int, name: str) -> None:
    if body < -1 or body >= len(builder.body_mass):
        raise IndexError(f"{name} index {body} is outside [-1, {len(builder.body_mass)}).")
    if body >= 0:
        _validate_world(builder, name, body, int(builder.body_world[body]))


def _validate_world(builder: ModelBuilder, name: str, index: int, world: int) -> None:
    if world not in (-1, builder.current_world):
        raise ValueError(
            f"{name} index {index} belongs to world {world}, "
            f"but the equality is added to world {builder.current_world}."
        )


def _validate_joint(builder: ModelBuilder, joint: int, name: str) -> None:
    if joint < 0 or joint >= len(builder.joint_type):
        raise IndexError(f"{name} index {joint} is outside [0, {len(builder.joint_type)}).")
    linear_dofs, angular_dofs = builder.joint_dof_dim[joint]
    if linear_dofs + angular_dofs != 1:
        raise ValueError(f"{name} index {joint} must identify a scalar joint.")
    _validate_world(builder, name, joint, int(builder.joint_world[joint]))


def _ensure_equality_attributes(builder: ModelBuilder) -> None:
    if not builder.has_custom_attribute("mujoco:equality_constraint_type"):
        _register_equality_constraint_attributes(builder)


def _equality_custom_attributes(
    custom_attributes: dict[str, Any] | None,
    solref: Sequence[float] | None,
    solimp: Sequence[float] | None,
) -> dict[str, Any] | None:
    attributes = dict(custom_attributes or {})
    if solref is not None:
        if "mujoco:eq_solref" in attributes:
            raise ValueError("solref and custom_attributes['mujoco:eq_solref'] cannot both be specified.")
        values = [float(value) for value in solref]
        if len(values) != 2:
            raise ValueError(f"solref requires exactly 2 values, got {len(values)}.")
        attributes["mujoco:eq_solref"] = wp.vec2(*values)
    if solimp is not None:
        if "mujoco:eq_solimp" in attributes:
            raise ValueError("solimp and custom_attributes['mujoco:eq_solimp'] cannot both be specified.")
        values = [float(value) for value in solimp]
        if len(values) != 5:
            raise ValueError(f"solimp requires exactly 5 values, got {len(values)}.")
        attributes["mujoco:eq_solimp"] = vec5(*values)
    return attributes or None


def add_equality_connect(
    builder: ModelBuilder,
    body1: int,
    body2: int = -1,
    *,
    anchor: Vec3 | None = None,
    label: str | None = None,
    enabled: bool = True,
    solref: Sequence[float] | None = None,
    solimp: Sequence[float] | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo connect equality between two bodies.

    Both operands must belong to the current world or the global world (``-1``).

    Args:
        builder: Model builder receiving the constraint.
        body1: First body index, or ``-1`` for the world.
        body2: Second body index, or ``-1`` for the world.
        anchor: Anchor point in the first body's frame [m].
        label: Optional constraint label.
        enabled: Whether the constraint is active.
        solref: Optional MuJoCo constraint reference parameters.
        solimp: Optional MuJoCo constraint impedance parameters.
        custom_attributes: Additional registered equality-frequency attributes.

    Returns:
        The ``mujoco:equality_constraint`` row index.
    """
    _ensure_equality_attributes(builder)
    _validate_body(builder, body1, "body1")
    _validate_body(builder, body2, "body2")
    if body1 < 0 and body2 < 0:
        raise ValueError("A connect equality must reference at least one body.")
    if body1 == body2:
        raise ValueError("A connect equality requires two distinct bodies.")
    return _add_equality_constraint(
        builder,
        EqType.CONNECT,
        body1=body1,
        body2=body2,
        anchor=anchor,
        label=label,
        enabled=enabled,
        custom_attributes=_equality_custom_attributes(custom_attributes, solref, solimp),
        validate_custom_attributes=True,
    )


def add_equality_weld(
    builder: ModelBuilder,
    body1: int,
    body2: int = -1,
    *,
    relpose: Transform | None = None,
    torquescale: float = 1.0,
    label: str | None = None,
    enabled: bool = True,
    solref: Sequence[float] | None = None,
    solimp: Sequence[float] | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo weld equality between two bodies.

    Both operands must belong to the current world or the global world (``-1``).

    Args:
        builder: Model builder receiving the constraint.
        body1: First body index, or ``-1`` for the world.
        body2: Second body index, or ``-1`` for the world.
        relpose: Relative pose of the second body in the first body's frame.
            If omitted, preserve the bodies' initial relative pose.
        torquescale: Angular residual length scale [m].
        label: Optional constraint label.
        enabled: Whether the constraint is active.
        solref: Optional MuJoCo constraint reference parameters.
        solimp: Optional MuJoCo constraint impedance parameters.
        custom_attributes: Additional registered equality-frequency attributes.

    Returns:
        The ``mujoco:equality_constraint`` row index.
    """
    _ensure_equality_attributes(builder)
    _validate_body(builder, body1, "body1")
    _validate_body(builder, body2, "body2")
    if body1 < 0 and body2 < 0:
        raise ValueError("A weld equality must reference at least one body.")
    if body1 == body2:
        raise ValueError("A weld equality requires two distinct bodies.")
    if relpose is None:
        # MuJoCo's zero-quaternion sentinel asks the compiler to infer relpose from qpos0.
        relpose = wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat(0.0, 0.0, 0.0, 0.0))
    return _add_equality_constraint(
        builder,
        EqType.WELD,
        body1=body1,
        body2=body2,
        relpose=relpose,
        torquescale=torquescale,
        label=label,
        enabled=enabled,
        custom_attributes=_equality_custom_attributes(custom_attributes, solref, solimp),
        validate_custom_attributes=True,
    )


def add_equality_joint(
    builder: ModelBuilder,
    joint1: int,
    joint2: int,
    *,
    polycoef: Sequence[float] = (0.0, 1.0),
    label: str | None = None,
    enabled: bool = True,
    solref: Sequence[float] | None = None,
    solimp: Sequence[float] | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo polynomial equality between two scalar joints.

    Both joints must belong to the current world or the global world (``-1``).

    Args:
        builder: Model builder receiving the constraint.
        joint1: Dependent Newton joint index.
        joint2: Reference Newton joint index.
        polycoef: Up to five polynomial coefficients.
        label: Optional constraint label.
        enabled: Whether the constraint is active.
        solref: Optional MuJoCo constraint reference parameters.
        solimp: Optional MuJoCo constraint impedance parameters.
        custom_attributes: Additional registered equality-frequency attributes.

    Returns:
        The ``mujoco:equality_constraint`` row index.
    """
    _ensure_equality_attributes(builder)
    _validate_joint(builder, joint1, "joint1")
    _validate_joint(builder, joint2, "joint2")
    if joint1 == joint2:
        raise ValueError("A joint equality requires two distinct joints.")
    coefficients = [float(value) for value in polycoef]
    if len(coefficients) > 5:
        raise ValueError(f"polycoef accepts at most 5 values, got {len(coefficients)}.")
    coefficients.extend([0.0] * (5 - len(coefficients)))
    return _add_equality_constraint(
        builder,
        EqType.JOINT,
        joint1=joint1,
        joint2=joint2,
        polycoef=coefficients,
        label=label,
        enabled=enabled,
        custom_attributes=_equality_custom_attributes(custom_attributes, solref, solimp),
        validate_custom_attributes=True,
    )


__all__ = [
    "add_equality_connect",
    "add_equality_joint",
    "add_equality_weld",
]
