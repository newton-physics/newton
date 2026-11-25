# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Geom ``condim`` from shape friction modes, and compatibility for the deprecated ``mujoco:condim``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...sim.enums import FRICTION_MODE_CONDIM, ShapeFrictionMode, friction_mode_from_condim
from ...sim.model import Model
from ...utils.deprecation import warn_deprecation

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder

_FRICTION_MODE_CONDIM = np.array(FRICTION_MODE_CONDIM, dtype=np.int32)

_DEPRECATION_MSG = (
    "The mujoco:condim custom attribute is deprecated in Newton 1.9; set ModelBuilder.ShapeConfig.friction_mode "
    "and use Model.shape_material_friction_mode instead. Model.mujoco.condim is a copy of the friction modes; "
    "values written to it are applied when it is next read or a SolverMuJoCo is constructed."
)


def skip_imported_condim(_value: Any, _context: dict[str, Any] | None = None) -> None:
    """Leave the deprecated ``mujoco:condim`` unset; importers set the shape friction mode."""
    return None


class _DeprecatedCondim:
    """Getter of the deprecated ``model.mujoco.condim``, which keeps the array it returns writable.

    The friction modes are encoded differently, so the array is a copy of them, and values written
    to it are folded back by :meth:`apply_edits`.
    """

    def __init__(self, model: Model):
        self._model = model
        self._array: wp.array | None = None
        self._synced: np.ndarray | None = None
        """Values of ``_array`` when it was last derived from the friction modes."""

    def apply_edits(self) -> None:
        """Set the friction modes of the shapes whose condim was edited since the last read."""
        if self._array is None:
            return
        values = self._array.numpy()
        edited = np.flatnonzero(values != self._synced)
        if len(edited):
            modes = self._model.shape_material_friction_mode.numpy()
            modes[edited] = [friction_mode_from_condim(int(condim)) for condim in values[edited]]
            self._model.shape_material_friction_mode.assign(modes)
            self._synced = values.copy()

    def __call__(self) -> wp.array:
        self.apply_edits()
        self._synced = _FRICTION_MODE_CONDIM[self._model.shape_material_friction_mode.numpy()]
        if self._array is None:
            self._array = wp.array(self._synced, dtype=wp.int32, device=self._model.device)
        else:
            self._array.assign(self._synced)
        return self._array


def finalize_condim(builder: ModelBuilder, model: Model, attr: ModelBuilder.CustomAttribute) -> None:
    """Apply builder-authored ``mujoco:condim`` values to the friction modes and alias them on the model."""
    modes = model.shape_material_friction_mode.numpy()
    sliding = _FRICTION_MODE_CONDIM[ShapeFrictionMode.SLIDING]
    # The values the attribute would hold: authored ones override the mode, and a changed default
    # applies to the shapes left at the sliding mode.
    condim = np.full(builder.shape_count, attr.default, dtype=np.int32)
    authored = condim != sliding
    authored[: len(modes)] &= modes == int(ShapeFrictionMode.SLIDING)
    for shape, value in (attr.values or {}).items():
        if value is not None:
            condim[shape] = value
            authored[shape] = True
    # Body variants fill in the default for their rows, which keep their mode, and for their
    # shapes, which are stored past the model's shapes.
    variant = np.zeros(builder.shape_count, dtype=bool)
    variant[len(modes) :] = True
    for variants in builder._body_variants.values():
        variant[variants.row_start : variants.row_start + variants.variant_set.row_count] = True
    if np.any(variant & (condim != sliding)):
        raise ValueError(
            "mujoco:condim is not supported on body variant shapes; set ModelBuilder.ShapeConfig.friction_mode instead."
        )
    authored &= ~variant
    if authored.any():
        warn_deprecation(_DEPRECATION_MSG)
        modes[authored[: len(modes)]] = [friction_mode_from_condim(int(value)) for value in condim[authored]]
        model.shape_material_friction_mode.assign(modes)

    if not hasattr(model, "mujoco"):
        model.mujoco = Model.AttributeNamespace("mujoco")
    model.mujoco._add_deprecated_alias(attr.name, _DeprecatedCondim(model), _DEPRECATION_MSG)
    model._set_attribute_spec(attr.key, Model.AttributeSpec(Model.AttributeFrequency.SHAPE, deprecated=True))


def resolve_shape_condim(model: Model) -> np.ndarray:
    """Return each shape's MuJoCo ``condim`` from :attr:`~newton.Model.shape_material_friction_mode`."""
    if model.shape_material_friction_mode is None:
        return np.full(model.shape_count, 3, dtype=np.int32)
    namespace = getattr(model, "mujoco", None)
    alias = namespace._deprecated_aliases.get("condim") if namespace is not None else None
    if alias is not None:
        alias[0].apply_edits()
    return _FRICTION_MODE_CONDIM[model.shape_material_friction_mode.numpy()]


def shapes_rolling_without_torsion(model: Model) -> np.ndarray:
    """Return which shapes are ``SLIDING_ROLLING``, MuJoCo ``condim`` 6 without torsional friction."""
    if model.shape_material_friction_mode is None:
        return np.zeros(model.shape_count, dtype=bool)
    return model.shape_material_friction_mode.numpy() == int(ShapeFrictionMode.SLIDING_ROLLING)
