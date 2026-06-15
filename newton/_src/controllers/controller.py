# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Abstract base for Newton controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class Controller(ABC):
    """Abstract interface for a single Newton control law.

    Every concrete control law (PID, differential IK, gravity compensation,
    …) subclasses :class:`Controller` directly. There is no framework-level
    composition: users who want to combine multiple control laws call each
    one's :meth:`compute` in sequence themselves.

    Subclasses are responsible for:

    - :meth:`is_graphable`, :meth:`is_stateful`: predicates the user can
      query to decide CUDA-graph capture and double-buffered state setup.
    - :meth:`state`: allocate a fresh :class:`State`, or return ``None``
      for stateless laws.
    - :meth:`input_struct`, :meth:`output_struct`: allocate fresh
      duck-typed containers (auto-generated dataclasses) holding only the
      live ports the user declared at construction. Baked-in arrays (gain
      ports given a :class:`wp.array` rather than an attr name) are stored
      on the controller and do **not** appear on the input struct.
    - :meth:`compute`: read the input struct's live arrays, run kernels,
      write the output struct's live arrays. Writes are slot-replacing
      (``=``, not ``+=``); composing laws is the user's job, and the user
      controls whether the output is zeroed beforehand.
    """

    @dataclass
    class State:
        """Pure data container. Subclasses declare their fields."""

    @abstractmethod
    def is_graphable(self) -> bool:
        """Whether :meth:`compute` is safe to capture in a CUDA graph."""

    @abstractmethod
    def is_stateful(self) -> bool:
        """Whether this controller maintains internal state between steps."""

    @abstractmethod
    def state(self) -> Controller.State | None:
        """Allocate a fresh per-step :class:`State`, or ``None`` if stateless."""

    @abstractmethod
    def input_struct(self) -> Any:
        """Allocate a fresh input container with one field per input port.

        The returned object has attributes with
        names that match the ``*_attr`` strings the user passed at
        construction. Each field is a freshly :func:`wp.zeros`-allocated
        array of the right dtype and size. The user typically reassigns
        these fields to point at live data buffers.
        """

    @abstractmethod
    def output_struct(self) -> Any:
        """Allocate a fresh output container with one field per output port."""

    @abstractmethod
    def compute(
        self,
        input_struct: Any,
        output_struct: Any,
        controller_state_now: Controller.State | None,
        controller_state_next: Controller.State | None,
        time_step: float,
    ) -> None:
        """Run one control step.

        Args:
            input_struct: Object whose attributes hold the live read ports.
                Resolved via ``getattr(input_struct, attr_name)``. Any
                duck-typed object works — :meth:`input_struct` returns a
                pre-allocated one; the user may swap fields or pass an
                entirely custom container (e.g. a :class:`SimpleNamespace`
                wrapping :class:`newton.State` fields directly).
            output_struct: Same contract as ``input_struct`` for write
                ports. The kernel performs slot-replacing writes
                (``arr[idx[i]] = value``) — slots outside the declared
                port indices are left untouched.
            controller_state_now: Current state (``None`` if stateless).
            controller_state_next: Next-step state to populate (``None``
                if stateless). Double-buffered against
                ``controller_state_now``.
            time_step: Step duration [s].
        """
