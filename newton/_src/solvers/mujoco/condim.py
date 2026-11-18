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
            self._synced = values

    def __call__(self) -> wp.array:
        self.apply_edits()
        self._synced = _FRICTION_MODE_CONDIM[self._model.shape_material_friction_mode.numpy()]
        if self._array is None:
            self._array = wp.array(self._synced, dtype=wp.int32, device=self._model.device)
        else:
            self._array.assign(self._synced)
        return self._array


def finalize_condim(builder: ModelBuilder, model: Model, attr: ModelBuilder.CustomAttribute) -> None:
    """Fold builder-authored ``mujoco:condim`` values into the friction modes and alias them on the model."""
    values = attr.values or {}
    modes = model.shape_material_friction_mode.numpy()
    # Body variants fill in the default for their shapes, which are stored past the model's shapes.
    authored = [
        (int(shape), int(value))
        for shape, value in (values.items() if isinstance(values, dict) else enumerate(values))
        if value is not None and value != attr.default
    ]
    if any(shape >= len(modes) for shape, _ in authored):
        # A variant row keeps its friction mode, which the shapes it can hold must share.
        raise ValueError(
            "mujoco:condim is not supported on body variant shapes; set ModelBuilder.ShapeConfig.friction_mode instead."
        )
    if authored:
        warn_deprecation(_DEPRECATION_MSG)
        for shape, condim in authored:
            mode = int(friction_mode_from_condim(condim))
            if modes[shape] != mode and modes[shape] != int(ShapeFrictionMode.SLIDING):
                raise ValueError(
                    f"mujoco:condim {condim} of shape {builder.shape_label[shape]!r} conflicts with its friction mode "
                    f"{ShapeFrictionMode(modes[shape]).name}."
                )
            modes[shape] = mode
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
