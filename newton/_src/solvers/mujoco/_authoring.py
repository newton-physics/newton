# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared implementation helpers for MuJoCo model authoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...sim import ModelBuilder


def _vector(
    builder: ModelBuilder,
    key: str,
    values: Sequence[float],
    length: int,
    *,
    exact: bool = True,
) -> Any:
    """Convert ``values`` to the vector dtype of the custom attribute ``key``.

    With ``exact=False``, shorter inputs are zero-padded to ``length``.
    """
    components = [float(value) for value in values]
    if (exact and len(components) != length) or len(components) > length:
        requirement = "exactly" if exact else "at most"
        raise ValueError(f"{key} requires {requirement} {length} values, got {len(components)}.")
    components.extend([0.0] * (length - len(components)))
    return builder.custom_attributes[key].dtype(*components)


def _tristate(value: bool | int | str) -> int:
    """Convert a MuJoCo ``false``/``true``/``auto`` value to its integer code."""
    if isinstance(value, str):
        try:
            return {"false": 0, "true": 1, "auto": 2}[value.lower().strip()]
        except KeyError as error:
            raise ValueError(f"Expected false, true, or auto, got {value!r}.") from error
    result = int(value)
    if result not in (0, 1, 2):
        raise ValueError(f"Expected a MuJoCo tri-state value in {{0, 1, 2}}, got {value!r}.")
    return result


def _ensure_mujoco_attributes(builder: ModelBuilder, sentinel: str) -> None:
    """Register the MuJoCo schema when the requested domain is unavailable."""
    if builder.has_custom_attribute(sentinel):
        return

    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    SolverMuJoCo.register_custom_attributes(builder)


def _prepare_custom_frequency_row(
    builder: ModelBuilder,
    frequency: str,
    values: dict[str, Any],
    custom_attributes: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and combine one solver-owned custom-frequency row."""
    extras = custom_attributes or {}
    overlap = values.keys() & extras.keys()
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"custom_attributes cannot override MuJoCo-managed values: {names}")

    row = {**values, **extras}
    for key in row:
        attribute = builder.custom_attributes.get(key)
        if attribute is None:
            raise AttributeError(
                f"Custom attribute '{key}' is not registered. Register it before adding a {frequency} row."
            )
        if attribute.frequency != frequency:
            raise ValueError(
                f"Custom attribute '{key}' uses frequency {attribute.frequency!r}, expected {frequency!r}."
            )
    return row


def _add_custom_frequency_row(
    builder: ModelBuilder,
    frequency: str,
    values: dict[str, Any],
    *,
    index_key: str,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Append one validated custom-frequency row and return its index."""
    row = _prepare_custom_frequency_row(builder, frequency, values, custom_attributes)
    return builder.add_custom_values(**row)[index_key]


__all__ = []
