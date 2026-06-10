# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Base class for Newton control laws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from .utils import ControlSignal, HardwareInterface


class ControlLaw:
    """Abstract base for a single control law.

    A `ControlLaw` is bound to :class:`ControlSignal`s at construction —
    it knows *what kind* of array each of its ports needs, but nothing
    about attribute names. The composing :class:`Controller` provides a
    :class:`HardwareInterface` which resolves each signal to the
    attribute name on the runtime ``input`` / ``output`` object via
    :meth:`_resolve`, and the law stashes the resolved name per port so
    its :meth:`compute` can do plain ``getattr(input, attr_name)``.

    Subclasses declare:

    - :attr:`INPUT_PORTS` — the set of constructor kwargs that are read
      ports.
    - :attr:`OUTPUT_PORTS` — the set of constructor kwargs that are
      write ports.

    Every constructor kwarg listed in these sets accepts a
    ``(ControlSignal, wp.array[wp.uint32])`` tuple. Subclasses are free
    to take additional kwargs that aren't ports (e.g.,
    :class:`ControlLawDifferentialIK` takes ``model_builder``).
    """

    INPUT_PORTS: ClassVar[frozenset[str]] = frozenset()
    OUTPUT_PORTS: ClassVar[frozenset[str]] = frozenset()

    @dataclass
    class State:
        """Pure data container. Subclasses declare their fields."""

    # Populated by subclasses as they normalize their port kwargs.
    _used_inputs: frozenset[ControlSignal]
    _used_outputs: frozenset[ControlSignal]

    def finalize(self, device: wp.Device, requires_grad: bool = False) -> None:
        """Allocate device-side private buffers.

        Called by :class:`Controller` after construction and after
        :meth:`_resolve`. The law's signal-to-attribute-name resolution is
        already in place by this point, so ``finalize`` can size internal
        buffers using ``num_outputs`` derived from the output port's
        ``port_indices``.

        Args:
            device: Warp device to allocate on.
            requires_grad: Propagated from :class:`Controller`. When
                ``True``, internal buffers are allocated with gradient
                support so the law's kernels are transparent to
                :class:`wp.Tape`.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement finalize().")

    def state(self, device: wp.Device, requires_grad: bool = False) -> ControlLaw.State | None:
        """Allocate a fresh state, or return ``None`` if stateless."""
        return None

    def is_stateful(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__} must implement is_stateful().")

    def is_graphable(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__} must implement is_graphable().")

    def inputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        """Return the resolved read-port bindings as
        ``(attr_name, port_indices)`` pairs.

        Called by :class:`Controller` after :meth:`_resolve` has filled
        in the attribute names. The Controller uses these to size its
        ``input()`` factory; otherwise read ports just appear in
        :meth:`compute` via ``getattr(input, self._<port>_attr)``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement inputs().")

    def outputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        """Return the resolved write-port bindings as
        ``(attr_name, port_indices)`` pairs.

        Called by :class:`Controller` after :meth:`_resolve` has filled
        in the attribute names. The Controller uses these to zero the
        declared output slots at the start of each :meth:`Controller.step`
        call.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement outputs().")

    def _resolve(self, hw: HardwareInterface) -> None:
        """Convert the law's stashed ``(signal, port_indices)`` bindings
        into ``(attr_name, port_indices)`` pairs using ``hw``.

        Called once by :class:`Controller` at composition time. The law
        is expected to look up every read-port signal in ``hw.inputs``
        and every write-port signal in ``hw.outputs`` and stash the
        resolved attribute names on ``self`` for fast access in
        :meth:`compute`. Raises if any signal is not present in the
        appropriate direction of ``hw``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _resolve().")

    def compute(
        self,
        input: Any,
        output: Any,
        state: ControlLaw.State | None,
        next_state: ControlLaw.State | None,
        dt: float,
    ) -> None:
        """Run the per-step compute.

        By the time this is called, :meth:`_resolve` has been invoked, so
        the law can fetch each port's live array via
        ``getattr(input, self._<port>_attr)`` /
        ``getattr(output, self._<port>_attr)``.

        Args:
            input: User-supplied object whose attributes hold the read
                ports' live arrays.
            output: User-supplied object whose attributes hold the write
                ports' live arrays.
            state: Current law state (``None`` if stateless).
            next_state: Next-step state to populate (``None`` if
                stateless).
            dt: Timestep [s].
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute().")
