# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resolve inspection components without owning the simulation loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sim.collide import CollisionPipeline
from ..sim.model import Model

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from ..sim.control import Control
    from ..solvers.solver import SolverBase


class InspectionBindings:
    """Expose the components an application supplied through the viewer's public setters.

    References belong to one viewer layer and remain application-owned: Control
    from :meth:`~newton.viewer.ViewerBase.set_control`, the solver from
    :meth:`~newton.viewer.ViewerBase.set_solver`, and the collision pipeline from
    :meth:`~newton.viewer.ViewerBase.set_collision_pipeline`. State and Contacts
    come from frame logging. This object does not step, reset, render, record,
    or migrate state across topology changes.
    """

    def __init__(self, *, components: Mapping[str, Any] | None = None) -> None:
        """Retain a snapshot of the active layer's supplied references.

        Args:
            components: Control, solver, and collision_pipeline entries supplied
                by the viewer setters. The caller replaces rather than mutates
                this mapping when a component changes.
        """
        self._components = components if components is not None else {}

    @property
    def control(self) -> Control | None:
        """Return the inspected Control, or None when the application supplied none."""
        return self._components.get("control")

    @property
    def solver(self) -> SolverBase | None:
        """Return the inspected solver, or None when the application supplied none."""
        return self._components.get("solver")

    def owns_control(self, owner: object) -> bool:
        """Include registered namespace containers when authorizing Control edits."""
        control = self.control
        return control is not None and (
            owner is control
            or any(owner is value for value in vars(control).values() if isinstance(value, Model.AttributeNamespace))
        )

    def collision_pipelines(self, model: Model) -> list[tuple[str, CollisionPipeline]]:
        """Return the supplied pipeline, or pipelines held directly by the solver, for the given Model.

        Without an explicitly supplied pipeline, solvers that run collision
        detection themselves expose theirs as direct attributes. Inspect only
        those attributes, without traversing other objects or reading device data.
        """
        if "collision_pipeline" in self._components:
            pipeline = self._components["collision_pipeline"]
            return [("Collision pipeline", pipeline)] if pipeline is not None and pipeline.model is model else []
        result: list[tuple[str, CollisionPipeline]] = []
        solver = self.solver
        for name, value in vars(solver).items() if solver is not None else ():
            if isinstance(value, CollisionPipeline) and value.model is model and all(value is not p for _, p in result):
                result.append((name.lstrip("_"), value))
        return result
