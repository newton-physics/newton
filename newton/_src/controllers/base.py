# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Base class for Newton controllers."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


class Controller:
    """Abstract base for a single control law.

    Subclasses implement :meth:`compute` (and optionally :meth:`reset`),
    declare which output arrays they write to via :meth:`outputs`, and
    expose a nested :class:`State` dataclass holding their internal
    buffers.
    """

    @dataclass
    class State:
        """Pure data container. Subclasses declare their fields."""

    indices: wp.array[wp.uint32]
    """Global DOF indices this controller writes to (set by subclasses in ``__init__``)."""

    reset_state: Controller.State | None = None
    """Per-controller reset target. Allocated by :meth:`finalize` to zeros; user-mutable."""

    def finalize(self, device: wp.Device, num_outputs: int, requires_grad: bool = False) -> None:
        """Allocate device-side private buffers and :attr:`reset_state`.

        Called by :class:`ControlGroup` after construction.

        Args:
            device: Warp device to allocate on.
            num_outputs: Equal to ``len(self.indices)``.
            requires_grad: Propagated from :class:`ControlGroup`. If True, all
                internal buffers (including ``reset_state``) are allocated with
                gradient support so the controller is transparent to
                :class:`wp.Tape` — Isaac Lab and other autograd consumers can
                differentiate through ``compute()`` end-to-end. Kernels are
                autograd-able by default; this flag only controls allocations.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement finalize().")

    def state(self, num_outputs: int, device: wp.Device, requires_grad: bool = False) -> Controller.State | None:
        """Allocate a fresh state, or return ``None`` if stateless.

        Args:
            num_outputs: Equal to ``len(self.indices)``.
            device: Warp device for allocation.
            requires_grad: If True, allocate ``State`` fields with gradient support.
        """
        return None

    def is_stateful(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__} must implement is_stateful().")

    def is_graphable(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__} must implement is_graphable().")

    def outputs(self) -> list[tuple[wp.array, wp.array[wp.uint32]]]:
        """Return ``(output_array, output_port_indices)`` bindings.

        :class:`ControlGroup` collects these from every controller and zeros
        the listed slots at the start of each :meth:`ControlGroup.step` call.
        Most controllers return a single binding; multi-output controllers
        return more than one.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement outputs().")

    def compute(
        self,
        state: Controller.State | None,
        next_state: Controller.State | None,
        dt: float,
    ) -> None:
        """Read bound inputs, write ``+=`` into bound outputs, write ``next_state``.

        Called by :meth:`ControlGroup.step`. The device is fixed at
        :meth:`finalize` time, so this method does not take one.

        Args:
            state: Current controller state (``None`` if stateless).
            next_state: Next-step state to populate (``None`` if stateless).
            dt: Timestep [s].
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute().")

    def reset(self, state: Controller.State, mask: wp.array[wp.bool]) -> None:
        """Update ``state`` from :attr:`reset_state` where ``mask`` is True.

        ``mask`` is a bool array of length equal to this controller's
        ``num_outputs`` (the shared outer length of all output bindings
        returned by :meth:`outputs`). ``mask[i] = True`` means "reset slot
        ``i``." Slot ``i`` corresponds to whichever portion of the state
        the controller associates with output slot ``i`` — for a simple
        per-DOF controller like :class:`ControllerPID`, that's a single
        state element; for a controller with multi-component-per-output
        state, it may be a wider chunk.

        The default implementation is a no-op (suitable for stateless
        controllers).
        """
        return
