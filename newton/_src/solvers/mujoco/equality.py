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

import logging
from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import warp as wp

from ...core.types import Transform, Vec3, axis_to_vec3, vec5
from ...sim.model import Model
from .enums import EqType

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder

logger = logging.getLogger(__name__)


class MjcEqualityTargetKind(IntEnum):
    """How a MuJoCo equality row is projected into Newton."""

    NONE = 0  # Pure equality row; no projected Newton object, target is -1.
    JOINT = 1  # Target is a Newton joint, used by converted CONNECT/WELD loop joints.
    MIMIC = 2  # Target is a Newton mimic constraint, used by converted JOINT equalities.


# Object kinds an equality row references (mjtObj-aligned).
MJC_OBJ_UNKNOWN = -1
MJC_OBJ_BODY = 1
MJC_OBJ_JOINT = 3
MJC_OBJ_TENDON = 18


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
    builder.add_custom_frequency(
        type(builder).CustomFrequency(
            name="equality_constraint",
            namespace="mujoco",
            collapse_fixed_joints_handler=_collapse_equality_constraints,
        )
    )

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
            reference_value_transformer=_remap_equality_targets,
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


def _equality_values(builder: ModelBuilder, name: str, *, writable: bool = False) -> tuple[Any, list[Any]]:
    """Return the ``mujoco:<name>`` attribute and its sparse backing values list.

    With ``writable``, an unset list is created on the attribute so writes persist.
    """
    attr = builder.custom_attributes[f"mujoco:{name}"]
    if attr.values is None and writable:
        attr.values = []
    return attr, attr.values if attr.values is not None else []


def _value_at(values: list[Any], index: int, default: Any) -> Any:
    """Read a sparse custom-frequency value, falling back to ``default`` for missing or ``None`` rows."""
    if index >= len(values) or values[index] is None:
        return default
    return values[index]


def _set_value(values: list[Any], index: int, value: Any) -> None:
    """Write a sparse custom-frequency value, padding omitted rows with ``None``."""
    if len(values) <= index:
        values.extend([None] * (index + 1 - len(values)))
    values[index] = value


def _target_kind(value: Any) -> int:
    try:
        return int(value) if value is not None else int(MjcEqualityTargetKind.NONE)
    except (TypeError, ValueError):
        return int(MjcEqualityTargetKind.NONE)


def _remap_equality_targets(values: list[Any], context: dict[str, Any]) -> list[Any]:
    """Offset ``equality_constraint_target`` values copied by :meth:`ModelBuilder.add_builder`.

    The target indexes a joint or a mimic constraint depending on the row's
    ``equality_constraint_target_kind``, so a single ``references`` offset cannot remap it.
    """
    kind_attr = context["builder"].custom_attributes.get("mujoco:equality_constraint_target_kind")
    kind_values = (kind_attr.values or []) if kind_attr is not None else []
    entity_offsets = context["entity_offsets"]
    offsets = {
        int(MjcEqualityTargetKind.JOINT): entity_offsets["joint"],
        int(MjcEqualityTargetKind.MIMIC): entity_offsets["constraint_mimic"],
    }
    remapped = []
    for row, value in zip(context["row_indices"], values, strict=True):
        try:
            target = int(value) if value is not None else -1
        except (TypeError, ValueError):
            target = -1
        offset = offsets.get(_target_kind(_value_at(kind_values, row, None)))
        remapped.append(target + offset if target >= 0 and offset is not None else value)
    return remapped


def _collapse_equality_constraints(builder: ModelBuilder, context: dict[str, Any]) -> None:
    """Update equality rows for :meth:`ModelBuilder.collapse_fixed_joints`.

    Re-expresses CONNECT anchors and WELD anchors/relative poses in the frame of the body a
    referenced body was merged into, disables rows whose joints were removed, and remaps or
    clears projected targets. Body and joint references are remapped generically afterwards.
    """
    count = builder._custom_frequency_counts.get("mujoco:equality_constraint", 0)
    joint_remap = context["joint_remap"]
    body_merged_parent = context["body_merged_parent"]
    body_merged_transform = context["body_merged_transform"]
    verbose = context["verbose"]

    if count > 0:
        body1_attr, body1_values = _equality_values(builder, "equality_constraint_body1")
        body2_attr, body2_values = _equality_values(builder, "equality_constraint_body2")
        type_attr, type_values = _equality_values(builder, "equality_constraint_type")
        anchor_attr, anchor_values = _equality_values(builder, "equality_constraint_anchor", writable=True)
        relpose_attr, relpose_values = _equality_values(builder, "equality_constraint_relpose", writable=True)
        joint1_attr, joint1_values = _equality_values(builder, "equality_constraint_joint1")
        joint2_attr, joint2_values = _equality_values(builder, "equality_constraint_joint2")
        _enabled_attr, enabled_values = _equality_values(builder, "equality_constraint_enabled", writable=True)

        for i in range(count):
            old_body1 = _value_at(body1_values, i, body1_attr.default)
            old_body2 = _value_at(body2_values, i, body2_attr.default)
            constraint_type = _value_at(type_values, i, type_attr.default)

            # Express anchor/relpose in the frame of the body a referenced body was merged into.
            if old_body1 in body_merged_parent:
                merge_xform = body_merged_transform[old_body1]
                if constraint_type == EqType.CONNECT:
                    anchor = axis_to_vec3(_value_at(anchor_values, i, anchor_attr.default))
                    _set_value(anchor_values, i, wp.transform_point(merge_xform, anchor))
                if constraint_type == EqType.WELD:
                    relpose = _value_at(relpose_values, i, relpose_attr.default)
                    _set_value(relpose_values, i, merge_xform * relpose)

            if old_body2 in body_merged_parent and constraint_type == EqType.WELD:
                merge_xform = body_merged_transform[old_body2]
                anchor = axis_to_vec3(_value_at(anchor_values, i, anchor_attr.default))
                relpose = _value_at(relpose_values, i, relpose_attr.default)
                _set_value(anchor_values, i, wp.transform_point(merge_xform, anchor))
                _set_value(relpose_values, i, relpose * wp.transform_inverse(merge_xform))

            for old_joint in (
                _value_at(joint1_values, i, joint1_attr.default),
                _value_at(joint2_values, i, joint2_attr.default),
            ):
                if old_joint != -1 and old_joint not in joint_remap:
                    if verbose:
                        logger.warning(
                            "Equality constraint references removed joint %s, disabling constraint", old_joint
                        )
                    _set_value(enabled_values, i, False)

    target_kind_attr = builder.custom_attributes.get("mujoco:equality_constraint_target_kind")
    target_attr = builder.custom_attributes.get("mujoco:equality_constraint_target")
    if target_kind_attr is None or target_attr is None or not target_attr.values:
        return
    if target_kind_attr.values is None:
        target_kind_attr.values = []
    target_values = target_attr.values
    kind_values = target_kind_attr.values
    mimic_count = len(builder.constraint_mimic_joint0)
    for row, value in enumerate(target_values):
        if value is None:
            continue
        try:
            old_target = int(value)
        except (TypeError, ValueError):
            continue
        if old_target < 0:
            continue
        kind = _target_kind(_value_at(kind_values, row, None))
        if kind == MjcEqualityTargetKind.JOINT:
            if old_target in joint_remap:
                target_values[row] = joint_remap[old_target]
            else:
                target_values[row] = -1
                _set_value(kind_values, row, int(MjcEqualityTargetKind.NONE))
        elif kind == MjcEqualityTargetKind.MIMIC and old_target >= mimic_count:
            target_values[row] = -1
            _set_value(kind_values, row, int(MjcEqualityTargetKind.NONE))


def _register_equality_tendon_attributes(builder: ModelBuilder) -> None:
    """Declare the tendon-equality columns, which reference the ``mujoco:tendon`` frequency."""
    for name in ("equality_constraint_tendon1", "equality_constraint_tendon2"):
        builder.add_custom_attribute(
            type(builder).CustomAttribute(
                name=name,
                frequency="mujoco:equality_constraint",
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                references="mujoco:tendon",
                namespace="mujoco",
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
    tendon1: int = -1,
    tendon2: int = -1,
    validate_custom_attributes: bool = False,
) -> int:
    """Append a row to the ``mujoco:equality_constraint`` custom-attribute frequency on ``builder``.

    Args:
        builder: Target :class:`~newton.ModelBuilder`.
        constraint_type: Equality constraint type (``EqType.CONNECT``,
            ``EqType.WELD``, ``EqType.JOINT``, or ``EqType.TENDON``).
        body1: Index of the first body (-1 for world).
        body2: Index of the second body (-1 for world).
        anchor: Anchor point on body1. Defaults to the origin.
        torquescale: Angular residual scale for weld. Defaults to ``1.0`` for
            ``EqType.WELD`` and ``0.0`` otherwise.
        relpose: Relative pose of body2 for weld. Defaults to the identity transform.
        joint1: Index of the first joint for joint coupling.
        joint2: Index of the second joint for joint coupling.
        polycoef: Five polynomial coefficients for ``EqType.JOINT`` and
            ``EqType.TENDON`` coupling. Defaults to ``[0, 0, 0, 0, 0]``.
        label: Optional constraint label.
        enabled: Whether the constraint is active.
        custom_attributes: Additional ``mujoco:equality_constraint``-frequency
            custom attributes to assign at the new index.
        tendon1: Index of the first tendon for tendon coupling.
        tendon2: Index of the second tendon for tendon coupling (-1 for none).
        validate_custom_attributes: Reject managed-column overrides and validate extras before appending.

    Returns:
        Index of the new constraint row.
    """
    anchor_vec = axis_to_vec3(anchor) if anchor is not None else wp.vec3()
    relpose_tf = wp.transform(*relpose) if relpose is not None else wp.transform_identity()
    if torquescale is None:
        torquescale_value = 1.0 if constraint_type == EqType.WELD else 0.0
    else:
        torquescale_value = float(torquescale)
    objtype = {EqType.JOINT: MJC_OBJ_JOINT, EqType.TENDON: MJC_OBJ_TENDON}.get(constraint_type, MJC_OBJ_BODY)

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
    if constraint_type == EqType.TENDON:
        # Tendon columns exist only with the tendon schema and need tendons to resolve on merge.
        values["mujoco:equality_constraint_tendon1"] = tendon1
        values["mujoco:equality_constraint_tendon2"] = tendon2
    if validate_custom_attributes:
        # Importers may override columns when projecting constraints onto native
        # entities; public helpers accept only additional attributes.
        from ._authoring import _prepare_custom_frequency_row  # noqa: PLC0415

        values = _prepare_custom_frequency_row(builder, "mujoco:equality_constraint", values, custom_attributes)
        custom_attributes = None
    # Pad omitted columns with None (default) so an optional attribute authored only on
    # earlier rows, e.g. MJCF solref, isn't left short and flagged at finalize.
    for attribute in builder.get_custom_attributes_by_frequency(["mujoco:equality_constraint"]):
        values.setdefault(attribute.key, None)
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


def _validate_tendon(builder: ModelBuilder, tendon: int, name: str) -> None:
    tendon_count = builder._custom_frequency_counts.get("mujoco:tendon", 0)
    if tendon < 0 or tendon >= tendon_count:
        raise IndexError(f"{name} index {tendon} is outside [0, {tendon_count}).")
    _validate_world(builder, name, tendon, int(builder.custom_attributes["mujoco:tendon_world"].values[tendon]))


def _polycoef(polycoef: Sequence[float]) -> list[float]:
    coefficients = [float(value) for value in polycoef]
    if len(coefficients) > 5:
        raise ValueError(f"polycoef accepts at most 5 values, got {len(coefficients)}.")
    return coefficients + [0.0] * (5 - len(coefficients))


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
    return _add_equality_constraint(
        builder,
        EqType.JOINT,
        joint1=joint1,
        joint2=joint2,
        polycoef=_polycoef(polycoef),
        label=label,
        enabled=enabled,
        custom_attributes=_equality_custom_attributes(custom_attributes, solref, solimp),
        validate_custom_attributes=True,
    )


def add_equality_tendon(
    builder: ModelBuilder,
    tendon1: int,
    tendon2: int = -1,
    *,
    polycoef: Sequence[float] = (0.0, 1.0),
    label: str | None = None,
    enabled: bool = True,
    solref: Sequence[float] | None = None,
    solimp: Sequence[float] | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo polynomial equality between two tendon lengths.

    Lengths are measured relative to their values at the reference
    configuration. Without ``tendon2``, only ``polycoef[0]`` is used and the
    length of ``tendon1`` is held at that offset. Both tendons must belong to
    the current world or the global world (``-1``).

    Args:
        builder: Model builder receiving the constraint.
        tendon1: Dependent ``mujoco:tendon`` index.
        tendon2: Reference ``mujoco:tendon`` index, or ``-1`` for none.
        polycoef: Up to five polynomial coefficients.
        label: Optional constraint label.
        enabled: Whether the constraint is active.
        solref: Optional MuJoCo constraint reference parameters.
        solimp: Optional MuJoCo constraint impedance parameters.
        custom_attributes: Additional registered equality-frequency attributes.

    Returns:
        The ``mujoco:equality_constraint`` row index.
    """
    from ._authoring import _ensure_mujoco_attributes  # noqa: PLC0415

    _ensure_mujoco_attributes(builder, "mujoco:equality_constraint_tendon1")
    _validate_tendon(builder, tendon1, "tendon1")
    if tendon2 != -1:
        _validate_tendon(builder, tendon2, "tendon2")
    if tendon1 == tendon2:
        raise ValueError("A tendon equality requires two distinct tendons.")
    return _add_equality_constraint(
        builder,
        EqType.TENDON,
        tendon1=tendon1,
        tendon2=tendon2,
        polycoef=_polycoef(polycoef),
        label=label,
        enabled=enabled,
        custom_attributes=_equality_custom_attributes(custom_attributes, solref, solimp),
        validate_custom_attributes=True,
    )


__all__ = [
    "add_equality_connect",
    "add_equality_joint",
    "add_equality_tendon",
    "add_equality_weld",
]
