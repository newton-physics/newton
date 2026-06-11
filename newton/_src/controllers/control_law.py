# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Base class for Newton controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp


class ControlLaw:
    """Abstract base for a single control law.

    Subclasses implement :meth:`compute`, declare which output arrays they
    write to via :meth:`outputs`, and expose a nested :class:`State`
    dataclass holding their internal buffers.

    Every ``ControlLaw`` carries a unique-within-its-:class:`Controller`
    ``label`` set at construction; the composing :class:`Controller` uses
    that label as the key into its composed state's
    ``control_law_states`` dict.
    """

    @dataclass
    class State:
        """Pure data container. Subclasses declare their fields."""

    label: str
    """Unique-within-Controller string identifier. Set by subclasses in ``__init__``."""

    def finalize(self, device: wp.Device, num_outputs: int, requires_grad: bool = False) -> None:
        """Allocate device-side private buffers.

        Called by :class:`Controller` after construction.

        Args:
            device: Warp device to allocate on.
            num_outputs: The shared outer length of every binding returned
                by :meth:`outputs` — derived by the law from its output
                port indices at ``__init__``.
            requires_grad: Propagated from :class:`Controller`. If True, all
                internal buffers are allocated with gradient support so the
                controller is transparent to :class:`wp.Tape` — Isaac Lab
                and other autograd consumers can differentiate through
                ``compute()`` end-to-end. Kernels are autograd-able by
                default; this flag only controls allocations.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement finalize().")

    def state(self, num_outputs: int, device: wp.Device, requires_grad: bool = False) -> ControlLaw.State | None:
        """Allocate a fresh state, or return ``None`` if stateless.

        Args:
            num_outputs: The law's ``num_outputs``.
            device: Warp device for allocation.
            requires_grad: If True, allocate ``State`` fields with gradient support.
        """
        return None

    def is_stateful(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__} must implement is_stateful().")

    def is_graphable(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__} must implement is_graphable().")

    def outputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        """Return ``(output_attr_name, output_port_indices)`` bindings.

        :class:`Controller` collects these from every control law, resolves
        each ``output_attr_name`` against the ``output`` object passed to
        :meth:`Controller.step` once per step, and zeros the listed slots
        before any law's :meth:`compute` runs. Most laws return a single
        binding; multi-output laws return more than one.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement outputs().")

    def compute(
        self,
        input: Any,
        output: Any,
        state: ControlLaw.State | None,
        next_state: ControlLaw.State | None,
        dt: float,
    ) -> None:
        """Fetch port arrays from ``input``/``output``, then run kernels.

        Called by :meth:`Controller.step`. The device is fixed at
        :meth:`finalize` time, so this method does not take one. Every port
        is resolved here via ``getattr(input, attr_name)`` /
        ``getattr(output, attr_name)`` — nothing is bound at construction.

        Args:
            input: User-supplied object holding the read ports declared at
                ``__init__`` (e.g. ``input.joint_q``, ``input.kp``). Can be
                any duck-typed object — ``SimpleNamespace``, a dataclass,
                a :class:`newton.State` if its fields happen to match, etc.
            output: User-supplied object holding the write ports. Same
                duck-typed contract as ``input``.
            state: Current controller state (``None`` if stateless).
            next_state: Next-step state to populate (``None`` if stateless).
            dt: Timestep [s].
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute().")
