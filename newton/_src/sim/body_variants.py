# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Builder-side storage and finalization of body variants (:meth:`ModelBuilder.add_body_variants`)."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ..geometry.flags import ShapeFlags
from ..geometry.inertia import verify_and_correct_inertia
from ..geometry.types import GeoType
from .model import _SHAPE_VARIANT_ATTRIBUTES, Model

if TYPE_CHECKING:
    from .builder import ModelBuilder


_VARIANT_ROW_ROLES = ("collider", "visual", "site")
"""Roles of body variant shape rows, in row order. Static structures such as contact pairs, joint
filters, collision groups, and the MuJoCo geom/site layout depend on a shape's role, so each row
holds shapes of a single role."""


@dataclass(frozen=True, eq=False)
class _VariantShape:
    """One shape of a body variant: its builder attributes, finalized once for every body that selects it."""

    label: str
    """Shape label in the variant builder."""
    values: dict[str, Any]
    """Generic SHAPE builder attribute values: geometry, flags, materials, and appearance."""
    custom: tuple[tuple[str, Any, Any], ...]
    """Custom SHAPE attributes as ``(key, declaration, value)``."""


def _variant_shape_role(shape: _VariantShape) -> int:
    """Index in :data:`_VARIANT_ROW_ROLES` of a variant shape's role."""
    flags = int(shape.values["shape_flags"])
    if flags & ShapeFlags.SITE:
        return 2
    return 0 if flags & ShapeFlags.COLLIDE_SHAPES else 1


@dataclass(frozen=True)
class _Variant:
    """A body variant: the shapes and mass properties of a single-body builder."""

    label: str
    """Variant label."""
    shapes: tuple[_VariantShape, ...]
    """Shapes of the variant, in authored order."""
    mass: float
    """Mass [kg] of the variant body."""
    com: tuple[float, float, float]
    """Center of mass [m] of the variant body, in its frame."""
    inertia: tuple[float, ...]
    """Flattened inertia [kg·m²] of the variant body about its center of mass."""


@dataclass(frozen=True, eq=False)
class _VariantSet:
    """Variants declared together by :meth:`ModelBuilder.add_body_variants`.

    Bodies copied by :meth:`ModelBuilder.add_builder` or :meth:`ModelBuilder.replicate` share the
    set by identity, so its shapes are finalized and stored once.
    """

    variants: tuple[_Variant, ...]
    """The variants, indexed by variant id."""

    @functools.cached_property
    def role_counts(self) -> tuple[int, ...]:
        """Rows per role in :data:`_VARIANT_ROW_ROLES`: the most shapes of that role in any variant."""
        counts = [0] * len(_VARIANT_ROW_ROLES)
        for variant in self.variants:
            roles = [_variant_shape_role(shape) for shape in variant.shapes]
            counts = [max(count, roles.count(role)) for role, count in enumerate(counts)]
        return tuple(counts)

    @property
    def row_count(self) -> int:
        """Shape rows a body needs to hold any of the variants."""
        return sum(self.role_counts)

    @functools.cached_property
    def row_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Per variant, the index of the shape each row holds, or ``-1`` for an empty row.

        Rows form one block per role, so a row's role is the same for every variant. Within a
        block, a variant's shapes of that role keep their order.
        """
        block_starts = np.concatenate(([0], np.cumsum(self.role_counts)[:-1])).tolist()
        tables = []
        for variant in self.variants:
            table = [-1] * self.row_count
            next_row = list(block_starts)
            for index, shape in enumerate(variant.shapes):
                role = _variant_shape_role(shape)
                table[next_row[role]] = index
                next_row[role] += 1
            tables.append(tuple(table))
        return tuple(tables)


@dataclass(frozen=True)
class _BodyVariants:
    """Variants declared on one body by :meth:`ModelBuilder.add_body_variants`."""

    variant_set: _VariantSet
    """The variants the body can select."""
    row_start: int
    """First of the body's contiguous variant shape rows."""
    initial: int
    """Variant selected in the finalized model."""

    def shifted(self, shape_offset: int) -> _BodyVariants:
        return replace(self, row_start=self.row_start + shape_offset)


@dataclass(frozen=True)
class _VariantRow:
    """A shape row :meth:`ModelBuilder.add_body_variants` appends to a body."""

    seed: _VariantShape
    """Variant shape the row is built from until the first selection."""
    candidates: tuple[_VariantShape, ...]
    """Variant shapes the row can hold."""
    label: str
    """Label of the row."""


def _custom_shape_value(attribute: Any, shape: int) -> Any:
    """Value of a SHAPE custom attribute for ``shape``, falling back to its default."""
    value = attribute.values.get(shape) if attribute.values else None
    return attribute.default if value is None else value


def _combine_inertial_properties(
    a: tuple[float, np.ndarray, np.ndarray], b: tuple[float, np.ndarray, np.ndarray]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Combine two ``(mass, com, inertia about com)`` triples expressed in the same frame."""
    mass = a[0] + b[0]
    if mass <= 0.0:
        return mass, a[1], a[2] + b[2]
    com = (a[0] * a[1] + b[0] * b[1]) / mass
    inertia = np.zeros((3, 3))
    for part_mass, part_com, part_inertia in (a, b):
        offset = part_com - com
        inertia += part_inertia + part_mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    return mass, com, inertia


def _validate_variant_builder(builder: ModelBuilder, variant: ModelBuilder) -> None:
    """Raise if ``variant`` cannot be a variant of a body of ``builder``."""
    if variant is builder or variant._body_variants:
        raise ValueError("Variant builders must not be this builder or declare variants themselves.")
    if variant.up_axis != builder.up_axis:
        raise ValueError("Variant builders must use the same up axis as this builder.")
    if variant.body_count != 1:
        raise ValueError(f"Each variant builder must contain exactly one body, got {variant.body_count}.")
    if variant.body_shapes.get(-1):
        raise ValueError("Variant builders must not contain shapes that are not attached to their body.")
    for shape in variant.body_shapes.get(0, ()):
        if variant.shape_type[shape] in (GeoType.PLANE, GeoType.HFIELD, GeoType.GAUSSIAN):
            raise ValueError("Variant shapes must not be planes, heightfields, or Gaussians.")
        if variant.shape_flags[shape] & ShapeFlags.HYDROELASTIC:
            # The hydroelastic pipeline caches per-shape SDF descriptors at construction.
            raise ValueError("Variant shapes must not be hydroelastic.")
    for key, attribute in variant.custom_attributes.items():
        # Selection copies model arrays; state, control, and contact arrays are created later.
        if (
            attribute.frequency == Model.AttributeFrequency.SHAPE
            and attribute.assignment != Model.AttributeAssignment.MODEL
            and attribute.values
        ):
            raise ValueError(
                f"Variant shape custom attribute '{key}' is assigned to {attribute.assignment.name}; only "
                "model-assigned shape custom attributes can follow a variant."
            )


def _snapshot_variants(
    variants: Sequence[ModelBuilder], labels: Sequence[str], shape_attributes: Sequence[str]
) -> _VariantSet:
    """Copy the shapes and mass properties of single-body variant builders into a variant set."""
    return _VariantSet(
        tuple(
            _Variant(
                label,
                tuple(
                    _VariantShape(
                        variant.shape_label[shape],
                        {name: getattr(variant, name)[shape] for name in shape_attributes},
                        tuple(
                            (key, replace(attribute, values=None), value)
                            for key, attribute in variant.custom_attributes.items()
                            if attribute.frequency == Model.AttributeFrequency.SHAPE
                            and attribute.references is None
                            and (value := _custom_shape_value(attribute, shape)) is not None
                        ),
                    )
                    for shape in variant.body_shapes.get(0, ())
                ),
                float(variant.body_mass[0]),
                tuple(float(x) for x in variant.body_com[0]),
                tuple(float(x) for x in np.asarray(variant.body_inertia[0]).flatten()),
            )
            for variant, label in zip(variants, labels, strict=True)
        )
    )


def _plan_variant_rows(variant_set: _VariantSet, initial: int, body_label: str) -> list[_VariantRow]:
    """The shape rows a body needs to hold any variant of ``variant_set``.

    A row is built from the shape the initial variant places there, or from the first variant that
    places one, until the first selection. A collider row keeps its collision group, so its shapes
    must agree on one.
    """
    rows = []
    role_rows = [0] * len(_VARIANT_ROW_ROLES)
    for r in range(variant_set.row_count):
        candidates = tuple(
            variant.shapes[table[r]]
            for variant, table in zip(variant_set.variants, variant_set.row_shapes, strict=True)
            if table[r] >= 0
        )
        role = _variant_shape_role(candidates[0])
        label = f"{body_label}/variant_{_VARIANT_ROW_ROLES[role]}_{role_rows[role]}"
        role_rows[role] += 1
        groups = {int(candidate.values["shape_collision_group"]) for candidate in candidates}
        if role == 0 and len(groups) > 1:
            raise ValueError(
                f"Variant shapes held by row '{label}' have different collision groups {sorted(groups)}; "
                "the i-th colliders of all variants must share a collision group."
            )
        initial_shape = variant_set.row_shapes[initial][r]
        seed = variant_set.variants[initial].shapes[initial_shape] if initial_shape >= 0 else candidates[0]
        rows.append(_VariantRow(seed, candidates, label))
    return rows


class _VariantShapeRows:
    """Variant shapes appended to a builder as detached shape rows while it finalizes.

    :meth:`ModelBuilder.finalize` derives their collision data (SDFs, convex support, edges, bounds)
    like a shape's. :meth:`split` then moves them out of the model's shape arrays, and
    :meth:`finalize` builds the model's variant tables from them.
    """

    def __init__(self, builder: ModelBuilder):
        self.start = builder.shape_count
        """Index of the first variant shape row."""
        self.variant_sets: dict[int, _VariantSet] = {}
        """Distinct variant sets, by identity, in body order."""
        self._rows: dict[str, Any] = {}
        for body in sorted(builder._body_variants):
            variant_set = builder._body_variants[body].variant_set
            self.variant_sets.setdefault(id(variant_set), variant_set)
        for variant_set in self.variant_sets.values():
            for variant in variant_set.variants:
                for shape in variant.shapes:
                    builder._append_shape_row(shape, -1, -1, f"{variant.label}/{shape.label}")

    def remove(self, builder: ModelBuilder) -> None:
        """Remove the rows from ``builder``."""
        if builder.shape_count == self.start:
            return
        for name in builder._shape_row_attributes():
            del getattr(builder, name)[self.start :]
        for attribute in builder.custom_attributes.values():
            if attribute.frequency == Model.AttributeFrequency.SHAPE and isinstance(attribute.values, dict):
                for key in [k for k in attribute.values if k >= self.start]:
                    del attribute.values[key]

    def split(self, builder: ModelBuilder, m: Model) -> int:
        """Move the rows out of the per-shape arrays of ``m``.

        Returns:
            The number of shapes.
        """
        start = self.start
        total = len(builder.shape_type)
        if start < total:
            for name, spec in Model._CORE_ATTRIBUTE_SPECS.items():
                if spec.frequency != Model.AttributeFrequency.SHAPE or spec.alias_of is not None:
                    continue
                value = m.__dict__.get(name)
                if isinstance(value, wp.array) and value.shape[0] == total:
                    self._rows[name] = value[start:]
                    setattr(m, name, value[: max(start, 1) if spec.requires_empty_sentinel else start])
            self._rows["shape_source"] = list(builder.shape_source[start:])
            m.shape_label = list(builder.shape_label[:start])
            m.shape_source = list(builder.shape_source[:start])
        return start

    def finalize(self, builder: ModelBuilder, m: Model) -> None:
        """Build the variant tables of ``m`` and select the initial variants."""
        device = m.device
        shape_count = m.shape_count
        rows = self._rows
        variant_sets = list(self.variant_sets.values())
        set_index = {id(variant_set): k for k, variant_set in enumerate(variant_sets)}

        # Variant shapes of all sets, finalized once: geometry, materials, and custom attributes.
        variants = [variant for variant_set in variant_sets for variant in variant_set.variants]
        variant_shape_start = np.concatenate(([0], np.cumsum([len(v.shapes) for v in variants]))).astype(np.int32)
        set_variant_start = np.concatenate(([0], np.cumsum([len(s.variants) for s in variant_sets]))).astype(np.int32)
        m._variant_shape_count = int(variant_shape_start[-1])
        m._variant_shape_geometry = {name: rows[name] for name in _SHAPE_VARIANT_ATTRIBUTES if name in rows}
        m._variant_shape_source = rows.get("shape_source", [])
        # Per variant, the variant shape each row of its set holds, or -1 for an empty row.
        tables = [table for variant_set in variant_sets for table in variant_set.row_shapes]
        variant_row_shape_start = np.concatenate(([0], np.cumsum([len(table) for table in tables]))).astype(np.int32)
        variant_row_shape = np.array(
            [
                first + index if index >= 0 else -1
                for first, table in zip(variant_shape_start[:-1], tables, strict=True)
                for index in table
            ],
            dtype=np.int32,
        )
        m._variant_row_shape_start = wp.array(variant_row_shape_start, dtype=wp.int32, device=device)
        m._variant_row_shape = wp.array(variant_row_shape, dtype=wp.int32, device=device)
        m._variant_set_labels = [[variant.label for variant in variant_set.variants] for variant_set in variant_sets]
        m._variant_shape_custom = {}
        for key, attribute in builder.custom_attributes.items():
            if (
                attribute.frequency != Model.AttributeFrequency.SHAPE
                or attribute.assignment != Model.AttributeAssignment.MODEL
                or attribute.references is not None
                or m._variant_shape_count == 0
            ):
                continue
            owner = getattr(m, attribute.namespace, None) if attribute.namespace else m
            values = getattr(owner, attribute.name, None)
            if not isinstance(values, wp.array) or values.ndim != 1:
                continue
            variant_values = {
                q: attribute.values[self.start + q]
                for q in range(m._variant_shape_count)
                if attribute.values and self.start + q in attribute.values
            }
            m._variant_shape_custom[key] = (
                replace(attribute, values=variant_values).build_array(m._variant_shape_count, device=device),
                values,
            )

        # Per body: its variant set, rows, and validated mass properties of the body as built plus
        # each variant. Bodies with the same set and the same base share the validated values.
        body_set = np.full(builder.body_count, -1, dtype=np.int32)
        body_count = np.zeros(builder.body_count, dtype=np.int32)
        body_first = np.zeros(builder.body_count, dtype=np.int32)
        body_rows = np.zeros((builder.body_count, 2), dtype=np.int32)
        body_inertia_start = np.zeros(builder.body_count, dtype=np.int32)
        initial = np.full(builder.body_count, -1, dtype=np.int32)
        shape_variant_row = np.zeros(shape_count, dtype=bool)
        masses, coms, inertias = [], [], []
        inertia_start = {}
        for body in sorted(builder._body_variants):
            body_variants = builder._body_variants[body]
            k = set_index[id(body_variants.variant_set)]
            body_set[body] = k
            body_count[body] = len(body_variants.variant_set.variants)
            body_first[body] = set_variant_start[k]
            body_rows[body] = (body_variants.row_start, body_variants.variant_set.row_count)
            shape_variant_row[
                body_variants.row_start : body_variants.row_start + body_variants.variant_set.row_count
            ] = True
            initial[body] = body_variants.initial
            base = (
                float(builder.body_mass[body]),
                np.asarray(builder.body_com[body], dtype=np.float64),
                np.asarray(builder.body_inertia[body], dtype=np.float64).reshape(3, 3),
            )
            key = (k, base[0], base[1].tobytes(), base[2].tobytes())
            if key not in inertia_start:
                inertia_start[key] = len(masses)
                for variant in body_variants.variant_set.variants:
                    mass, com, inertia = _combine_inertial_properties(
                        base,
                        (
                            variant.mass,
                            np.asarray(variant.com, dtype=np.float64),
                            np.asarray(variant.inertia, dtype=np.float64).reshape(3, 3),
                        ),
                    )
                    mass, inertia, _ = verify_and_correct_inertia(
                        mass,
                        wp.mat33(*inertia.flatten()),
                        builder.balance_inertia,
                        builder.bound_mass,
                        builder.bound_inertia,
                        f"{builder.body_label[body]}/{variant.label}",
                    )
                    masses.append(mass)
                    coms.append(com)
                    inertias.append(np.asarray(inertia, dtype=np.float32).reshape(3, 3))
            body_inertia_start[body] = inertia_start[key]

        m._body_variant_set = body_set
        m._body_variant_count = wp.array(body_count, dtype=wp.int32, device=device)
        m._body_variant_first = wp.array(body_first, dtype=wp.int32, device=device)
        m._body_variant_rows = wp.array(body_rows, dtype=wp.vec2i, device=device)
        m._body_variant_inertia_start = wp.array(body_inertia_start, dtype=wp.int32, device=device)
        m._variant_mass = wp.array(masses, dtype=wp.float32, device=device)
        m._variant_com = wp.array(np.asarray(coms, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)
        m._variant_inertia = wp.array(
            np.asarray(inertias, dtype=np.float32).reshape(-1, 3, 3), dtype=wp.mat33, device=device
        )
        m._variant_row_max = max((variant_set.row_count for variant_set in variant_sets), default=0)
        row_shapes = np.flatnonzero(shape_variant_row)
        m._variant_row_shapes = wp.array(row_shapes, dtype=wp.int32, device=device)
        # The builder flags of a row, the union written by add_body_variants, before the initial
        # selection overwrites them.
        m._variant_row_flags = wp.array(m.shape_flags.numpy()[row_shapes], dtype=wp.int32, device=device)
        m._shape_variant_row = wp.array(shape_variant_row, dtype=wp.bool, device=device)
        m._shape_variant_shape = wp.full(shape_count, -1, dtype=wp.int32, device=device)
        m.body_variant = wp.full(builder.body_count, -1, dtype=wp.int32, device=device)
        if builder._body_variants:
            m.select_body_variants(wp.array(initial, dtype=wp.int32, device=device))
