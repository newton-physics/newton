# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulation bindings shared by viewer frontends without owning the run loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sim.collide import CollisionPipeline
from ..sim.control import Control
from ..sim.model import Model
from ..sim.state import State

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..sim.contacts import Contacts
    from ..solvers.solver import SolverBase
    from .viewer_types import ViewerExample


class ViewerSession:
    """Resolve explicit simulation bindings or a conventional example adapter.

    References remain owned by the application. A state provider resolves the
    current input buffer after a simulation swaps its double-buffered states.
    This adapter does not step, reset, render, or record the simulation.

    Bindings describe one viewer layer. Replacing a Model or its solver requires
    rebinding the session; this object does not migrate state across topology
    changes. Explicit bindings are separate from the example-runner convention.
    """

    def __init__(
        self,
        *,
        state: State | Callable[[], State | None] | None = None,
        control: Control | None = None,
        solver: SolverBase | None = None,
        collision_pipeline: CollisionPipeline | None = None,
        contacts: Contacts | None = None,
    ) -> None:
        """Bind existing simulation objects without copying their arrays.

        Args:
            state: Live State, or a provider evaluated on each lookup. Use a
                provider when the simulation swaps State objects after stepping.
            control: Live inputs that the inspector may edit in place.
            solver: Solver used to discover settings and inspection resources.
            collision_pipeline: Optional pipeline for collision diagnostics.
            contacts: Contact buffer produced by the application. Binding it
                does not run collision detection or populate the buffer.
        """
        self._state = state
        self._control = control
        self._solver = solver
        self._pipeline = collision_pipeline
        self._contacts = contacts
        self._example: ViewerExample | None = None

    @classmethod
    def from_example(cls, example: ViewerExample | None) -> ViewerSession:
        """Adapt the example runner's storage convention at one boundary."""
        session = cls()
        session._example = example
        return session

    def live_state(self, fallback: State | None = None) -> State | None:
        """Resolve the current live state without caching a swapped-out buffer.

        The example adapter prefers ``state_0`` to ``state``. If the binding
        resolves to None, return the caller's fallback, usually the last logged
        State. Historical display-state substitution belongs to the caller.

        Raises:
            TypeError: The binding resolves to a non-State object other than None.
        """
        state = self._state
        if self._example is not None:
            state = getattr(self._example, "state_0", getattr(self._example, "state", None))
        elif callable(state):
            state = state()
        if state is not None and not isinstance(state, State):
            raise TypeError("The viewer state provider must return a Newton State or None.")
        return state if state is not None else fallback

    @property
    def control(self) -> Control | None:
        return getattr(self._example, "control", None) if self._example is not None else self._control

    @property
    def solver(self) -> SolverBase | None:
        return getattr(self._example, "solver", None) if self._example is not None else self._solver

    @property
    def contacts(self) -> Contacts | None:
        return getattr(self._example, "contacts", None) if self._example is not None else self._contacts

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
        for owner in (self._example, self.solver, self._pipeline):
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
        result: list[tuple[str, CollisionPipeline]] = []
        seen: set[int] = set()
        if self._pipeline is not None and self._pipeline.model is model:
            result.append(("Collision pipeline", self._pipeline))
            seen.add(id(self._pipeline))
        for owner in (self._example, model, self.solver):
            for name, value in vars(owner).items() if hasattr(owner, "__dict__") else ():
                if isinstance(value, CollisionPipeline) and value.model is model and id(value) not in seen:
                    result.append((name.lstrip("_"), value))
                    seen.add(id(value))
        return result
