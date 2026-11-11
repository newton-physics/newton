# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .flags import ShapeFlags

if TYPE_CHECKING:
    from ..sim import Model

ShapeSelector = str | list[str] | re.Pattern[str] | list[int]
"""Label glob, list of label globs, compiled regular expression, or list of shape indices."""

SHAPE_GROUP_MAX_COUNT = 32
"""Number of shape groups a builder or model can hold, one bit of :attr:`Model.shape_groups` each."""


@dataclass(frozen=True, kw_only=True)
class ShapeFilter:
    """Selects a subset of a model's shapes for a query such as a ray cast or camera render.

    A shape matches when it satisfies every criterion that is set; a criterion set to ``None``
    does not constrain the selection. Within one criterion, any listed value suffices:

    - ``flags``: the shape has any of these :class:`~newton.ShapeFlags`.
    - ``groups``: the shape belongs to any of these shape groups (see
      :meth:`ModelBuilder.add_shape_group() <newton.ModelBuilder.add_shape_group>`).
    - ``shapes``: the shape's label or index matches this selector.
    - ``exclude_groups``: the shape belongs to none of these shape groups.
    - ``exclude_shapes``: the shape's label or index does not match this selector.

    Label selectors follow the syntax described in :ref:`label-matching`.

    A filter is a description that does not depend on a model. Register it with
    :meth:`ModelBuilder.add_shape_filter() <newton.ModelBuilder.add_shape_filter>` before
    finalizing to accelerate queries such as :func:`~newton.intersect_ray` and
    :class:`~newton.sensors.SensorCamera` with it, or resolve it on the host with :meth:`select`.

    Example:
        ::

            builder.add_shape_group("terrain", shapes="/World/ground*")
            terrain = builder.add_shape_filter(newton.ShapeFilter(groups=["terrain"]))
            model = builder.finalize()
            newton.intersect_ray(model, ..., shape_filter=terrain)
    """

    flags: ShapeFlags | int | None = None
    """Any-of mask of :class:`~newton.ShapeFlags`, or ``None`` to accept any flags."""

    groups: Sequence[str] | None = None
    """Names of the shape groups to accept, or ``None`` to accept shapes regardless of group."""

    shapes: ShapeSelector | None = None
    """Shapes to accept by label or index, or ``None`` to accept any shape. A list is stored as a tuple."""

    exclude_groups: Sequence[str] = ()
    """Names of the shape groups to reject."""

    exclude_shapes: ShapeSelector | None = None
    """Shapes to reject by label or index, or ``None`` to reject none. A list is stored as a tuple."""

    def __post_init__(self) -> None:
        # Store sequences as tuples so that filters are immutable and hashable.
        for name in ("groups", "exclude_groups"):
            value = getattr(self, name)
            if isinstance(value, str):
                raise TypeError(f"ShapeFilter.{name} expects a sequence of group names, got the string {value!r}.")
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        for name in ("shapes", "exclude_shapes"):
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))

    def select(self, model: Model) -> np.ndarray:
        """Return the indices of the shapes of ``model`` that this filter selects.

        A body variant row (see :meth:`ModelBuilder.add_body_variants()
        <newton.ModelBuilder.add_body_variants>`) is matched by the flags of all the variant
        shapes it can hold.

        Args:
            model: Model whose shapes to select.

        Returns:
            Sorted shape indices, dtype ``int32``.

        Raises:
            ValueError: If the filter names a shape group that ``model`` does not define.
        """
        mask = _shape_filter_mask(
            self,
            flags=model._shape_option_flags(),
            groups=model.shape_groups.numpy() if model.shape_groups is not None else None,
            labels=model.shape_label,
            group_names=model.shape_group_names,
        )
        return np.flatnonzero(mask).astype(np.int32)


def _shape_group_bits(names: Sequence[str], group_names: Sequence[str]) -> int:
    """Bit mask of the shape groups ``names`` within ``group_names``."""
    bits = 0
    for name in names:
        try:
            bits |= 1 << group_names.index(name)
        except ValueError:
            known = ", ".join(repr(group) for group in group_names) or "none"
            raise ValueError(f"Unknown shape group {name!r}; known shape groups: {known}.") from None
    return bits


def _shape_selector_mask(selector: ShapeSelector, labels: Sequence[str]) -> np.ndarray:
    """Boolean mask of the shapes matching ``selector`` by label or index."""
    from ..utils.selection import match_labels  # noqa: PLC0415

    mask = np.zeros(len(labels), dtype=bool)
    if isinstance(selector, tuple):
        selector = list(selector)
    indices = np.asarray(match_labels(labels if isinstance(labels, list) else list(labels), selector), dtype=np.int64)
    if len(indices) and (indices.min() < 0 or indices.max() >= len(labels)):
        raise IndexError(f"Shape index out of range [0, {len(labels)}) in shape selector.")
    mask[indices] = True
    return mask


def _shape_filter_mask(
    shape_filter: ShapeFilter,
    *,
    flags: np.ndarray,
    groups: np.ndarray | None,
    labels: Sequence[str],
    group_names: Sequence[str],
) -> np.ndarray:
    """Boolean mask of the shapes ``shape_filter`` selects, evaluated on host arrays."""
    flags = np.asarray(flags, dtype=np.int64)
    groups = np.zeros(len(flags), dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
    mask = np.ones(len(flags), dtype=bool)
    if shape_filter.flags is not None:
        mask &= (flags & int(shape_filter.flags)) != 0
    if shape_filter.groups is not None:
        mask &= (groups & _shape_group_bits(shape_filter.groups, group_names)) != 0
    if shape_filter.exclude_groups:
        mask &= (groups & _shape_group_bits(shape_filter.exclude_groups, group_names)) == 0
    if shape_filter.shapes is not None:
        mask &= _shape_selector_mask(shape_filter.shapes, labels)
    if shape_filter.exclude_shapes is not None:
        mask &= ~_shape_selector_mask(shape_filter.exclude_shapes, labels)
    return mask
