# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Programmatic authoring helpers for MuJoCo contact metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...sim import ModelBuilder
from ._authoring import _add_custom_frequency_row, _ensure_mujoco_attributes


def _vector(builder: ModelBuilder, key: str, values: Sequence[float], length: int) -> Any:
    components = [float(value) for value in values]
    if len(components) != length:
        raise ValueError(f"{key} requires exactly {length} values, got {len(components)}.")
    return builder.custom_attributes[key].dtype(*components)


def add_contact_pair(
    builder: ModelBuilder,
    shape0: int,
    shape1: int,
    *,
    condim: int = 3,
    solref: Sequence[float] = (0.02, 1.0),
    solreffriction: Sequence[float] = (0.02, 1.0),
    solimp: Sequence[float] = (0.9, 0.95, 0.001, 0.5, 2.0),
    margin: float = 0.0,
    gap: float = 0.0,
    friction: Sequence[float] = (1.0, 1.0, 0.005, 0.0001, 0.0001),
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add an explicit MuJoCo contact pair.

    Args:
        builder: Model builder receiving the pair.
        shape0: First Newton shape index.
        shape1: Second Newton shape index.
        condim: MuJoCo contact dimensionality.
        solref: Normal constraint reference parameters.
        solreffriction: Friction constraint reference parameters.
        solimp: Constraint impedance parameters.
        margin: Contact inclusion margin [m].
        gap: Inactive contact gap [m].
        friction: Sliding, torsional, and rolling friction coefficients.
        custom_attributes: Additional registered ``mujoco:pair`` attributes.

    Returns:
        The ``mujoco:pair`` row index.
    """
    _ensure_mujoco_attributes(builder, "mujoco:pair_geom1")
    shape_count = len(builder.shape_body)
    for name, shape in (("shape0", shape0), ("shape1", shape1)):
        if shape < 0 or shape >= shape_count:
            raise IndexError(f"{name} index {shape} is outside [0, {shape_count}).")
    if shape0 == shape1:
        raise ValueError("A MuJoCo contact pair requires two distinct shapes.")
    if condim not in (1, 3, 4, 6):
        raise ValueError("condim must be one of 1, 3, 4, or 6.")

    values = {
        "mujoco:pair_world": builder.current_world,
        "mujoco:pair_geom1": int(shape0),
        "mujoco:pair_geom2": int(shape1),
        "mujoco:pair_condim": int(condim),
        "mujoco:pair_solref": _vector(builder, "mujoco:pair_solref", solref, 2),
        "mujoco:pair_solreffriction": _vector(builder, "mujoco:pair_solreffriction", solreffriction, 2),
        "mujoco:pair_solimp": _vector(builder, "mujoco:pair_solimp", solimp, 5),
        "mujoco:pair_margin": float(margin),
        "mujoco:pair_gap": float(gap),
        "mujoco:pair_friction": _vector(builder, "mujoco:pair_friction", friction, 5),
    }
    return _add_custom_frequency_row(
        builder,
        "mujoco:pair",
        values,
        index_key="mujoco:pair_geom1",
        custom_attributes=custom_attributes,
    )


__all__ = ["add_contact_pair"]
