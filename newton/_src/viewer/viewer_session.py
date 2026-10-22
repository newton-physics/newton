# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resolve inspection components without owning the simulation loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sim.collide import CollisionPipeline
from ..sim.control import Control
from ..sim.model import Model
from ..sim.state import State

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from ..sim.contacts import Contacts
    from ..solvers.solver import SolverBase
    from .viewer_types import ViewerExample


class ViewerSession:
    """Combine independently supplied components with a conventional example.

    References belong to one viewer layer and remain application-owned. State
    and Contacts come from frame logging; the example adapter can additionally
    resolve live buffers during historical display. This object does not step,
    reset, render, record, or migrate state across topology changes.
    """

    def __init__(self, *, components: Mapping[str, Any] | None = None) -> None:
        """Retain a snapshot of the active layer's explicitly supplied references.

        Args:
            components: Control, solver, and collision_pipeline entries supplied
                by the viewer setters. Missing keys allow example discovery;
                explicit None values suppress it. The caller replaces rather
                than mutates this mapping when a component changes.
        """
        self._components = components if components is not None else {}
        self._example: ViewerExample | None = None

    @classmethod
    def from_example(
        cls, example: ViewerExample | None, *, components: Mapping[str, Any] | None = None
    ) -> ViewerSession:
        """Adapt the example runner's storage convention at one boundary."""
        session = cls(components=components)
        session._example = example
        return session

    def live_state(self, fallback: State | None = None) -> State | None:
        """Resolve the current live state without caching a swapped-out buffer.

        The example adapter prefers ``state_0`` to ``state``. Custom loops use
        the last State supplied through log_state, passed as the fallback.
        Historical display-state substitution belongs to the caller.

        Raises:
            TypeError: The example supplies a non-State object other than None.
        """
        state = getattr(self._example, "state_0", getattr(self._example, "state", None))
        if state is not None and not isinstance(state, State):
            raise TypeError("The example state must be a Newton State or None.")
        return state if state is not None else fallback

    @property
    def control(self) -> Control | None:
        """Resolve the explicit Control override, including None, or the example fallback."""
        return self._components.get("control", getattr(self._example, "control", None))

    @property
    def solver(self) -> SolverBase | None:
        """Resolve the explicit solver override, including None, or the example fallback."""
        return self._components.get("solver", getattr(self._example, "solver", None))

    @property
    def contacts(self) -> Contacts | None:
        """Return the example contact buffer as a fallback when no frame buffer was logged."""
        return getattr(self._example, "contacts", None)

    def owns_control(self, owner: object) -> bool:
        """Include registered namespace containers when authorizing Control edits."""
        control = self.control
        return control is not None and (
            owner is control
            or any(owner is value for value in vars(control).values() if isinstance(value, Model.AttributeNamespace))
        )

    def matches_model(self, model: Model) -> bool:
        """Reject conventional example bindings from another active viewer layer.

        Some examples store their Model inside a simulation object. For those,
        validate the core state layout without transferring array contents.
        """
        for owner in (self._example, self.solver, self._components.get("collision_pipeline")):
            bound = getattr(owner, "model", None)
            if isinstance(bound, Model) and bound is not model:
                return False
        state = self.live_state()
        for name, count in (
            ("body_q", model.body_count),
            ("joint_q", model.joint_coord_count),
            ("joint_qd", model.joint_dof_count),
            ("particle_q", model.particle_count),
        ):
            array = getattr(state, name, None)
            if array is not None and len(array) != count:
                return False
        return True

    def collision_pipelines(self, model: Model) -> list[tuple[str, CollisionPipeline]]:
        """Return named pipelines belonging to the given Model, deduplicated by identity.

        Inspect only direct attributes of the example, Model, and solver. Avoid
        recursively traversing application objects or transferring device arrays;
        applications with deeper ownership can provide an explicit pipeline.
        """
        if "collision_pipeline" in self._components:
            pipeline = self._components["collision_pipeline"]
            return [("Collision pipeline", pipeline)] if pipeline is not None and pipeline.model is model else []
        result: list[tuple[str, CollisionPipeline]] = []
        seen: set[int] = set()
        for owner in (self._example, model, self.solver):
            for name, value in vars(owner).items() if hasattr(owner, "__dict__") else ():
                if isinstance(value, CollisionPipeline) and value.model is model and id(value) not in seen:
                    result.append((name.lstrip("_"), value))
                    seen.add(id(value))
        return result
