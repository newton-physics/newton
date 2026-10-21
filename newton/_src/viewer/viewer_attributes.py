# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent attribute discovery and bounded property domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..sim.model import Model

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any


Frequency = Model.AttributeFrequency
Assignment = Model.AttributeAssignment


def resolve_attribute(owner: object, name: str) -> tuple[object | None, str, Any]:
    """Resolve the registry's namespace:name notation without flattening namespaces."""
    if ":" in name:
        namespace, name = name.split(":", 1)
        owner = getattr(owner, namespace, None)
    return owner, name, getattr(owner, name, None)


def component_attributes(
    model: Model, owner: object, component: Model.AttributeAssignment
) -> Iterator[tuple[str, Model.AttributeSpec, object, str, Any]]:
    """Discover populated inspector fields without reading device array contents.

    Custom assignment metadata determines the component; built-ins are resolved
    against its actual storage. Internal compaction data is omitted from the
    inspector. Namespace containers remain intact so edits reach the original
    owner rather than a flattened copy.

    Yields:
        Qualified name, attribute specification, storage owner, unqualified name,
        and value. The value is a reference to existing storage, not a snapshot.
    """
    for name, spec in model._iter_attribute_specs():
        if name.startswith("_") or spec.compaction_policy in ("world_start", "color_groups", "passthrough"):
            continue
        # Built-ins can exist on several components (e.g. Model's initial q).
        # Custom assignments are authoritative, even though their seed lives on Model.
        if spec.assignment is not None and spec.assignment != component:
            continue
        parent, attribute, value = resolve_attribute(owner, name)
        if value is None:
            continue
        if isinstance(value, (wp.array, np.ndarray)) and value.size == 0:
            continue
        if isinstance(value, (list, tuple)) and not value:
            continue
        yield name, spec, parent, attribute, value


def attribute_summary(name: str, spec: Model.AttributeSpec, array: Any, component: Model.AttributeAssignment) -> str:
    """Describe full storage using host metadata only, without reading array values."""
    frequency = spec.frequency.name if isinstance(spec.frequency, Frequency) else str(spec.frequency)
    lines = [f"{component.name.title()}.{name.replace(':', '.')}"]
    if isinstance(array, wp.array):
        lines.extend((f"Array shape: {array.shape}", f"Dtype: {wp.types.type_repr(array.dtype)}"))
        element_shape = getattr(array.dtype, "_shape_", ())
        if element_shape:
            scalar = wp.types.type_repr(array.dtype._wp_scalar_type_)
            lines.append(f"Element: {element_shape} of {scalar}")
        lines.append(f"Device: {array.device}")
    elif isinstance(array, np.ndarray):
        lines.extend((f"Array shape: {array.shape}", f"Dtype: {array.dtype}"))
    elif isinstance(array, (list, tuple)):
        lines.append(f"{type(array).__name__.title()}: {len(array):,} entries")
    else:
        lines.append(f"Type: {type(array).__name__}")
    lines.append(f"Frequency: {frequency}")
    if spec.row_width != 1:
        lines.append(f"Entries per row: {spec.row_width}")
    if spec.references is not None:
        references = spec.references.name if isinstance(spec.references, Frequency) else str(spec.references)
        lines.append(f"References: {references}")
    return "\n".join(lines)


def frequency_row_label(frequency: Model.AttributeFrequency | str, index: int) -> str:
    """Name a storage row by its registered domain and zero-based array index."""
    name = frequency.name if isinstance(frequency, Frequency) else frequency or "element"
    return f"{name.replace('_', ' ').lower()} {index}"


@dataclass
class PropertyDomain:
    """Describe the rows belonging to an inspector selection, independently of widgets.

    Keep ranges lazy for large worlds and meshes. The inspector expands joint
    selections to coordinate/DOF spans only when their fields are displayed.
    """

    model: Model
    """Model whose attribute registry and topology define the rows."""
    frequency: Model.AttributeFrequency | str
    """Built-in or solver-defined attribute frequency to inspect."""
    indices: Sequence[int] | np.ndarray
    """Global row indices, or joint indices when ``joints`` is True."""
    joints: bool = False
    """Resolve JOINT_COORD/JOINT_DOF spans through the Model's joint start arrays."""
