# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Controller — composer of multiple ControlLaws."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import warp as wp

from .control_law import ControlLaw
from .utils import _resolve_input_array


@wp.kernel
def _zero_at_indices_kernel(
    array: wp.array[float],
    indices: wp.array[wp.uint32],
):
    i = wp.tid()
    array[indices[i]] = 0.0


class Controller:
    """Composes one or more :class:`ControlLaw` instances.

    Owns the per-step zero / compute sequence and routes the user-supplied
    ``input`` / ``output`` through every law. Laws run serially in
    registration order; their contributions accumulate via ``+=`` into the
    output arrays after a single upfront zero pass.

    Args:
        control_laws: One or more :class:`ControlLaw` instances. Every
            law's ``label`` must be unique within the controller and every
            law must agree on ``num_outputs`` (the shared outer length of
            every ``outputs()`` binding).
        requires_grad: Single source of truth for gradient support across
            the controller. Propagated to each ControlLaw's
            :meth:`ControlLaw.finalize` and :meth:`ControlLaw.state`. When
            ``True``, every internally-allocated buffer is created with
            ``requires_grad=True`` so the ControlLaws' kernel launches are
            transparent to :class:`wp.Tape`. Users with mixed-grad needs
            split into multiple Controllers.
        device: Device for internal allocations. Defaults to
            :func:`wp.get_device`.
    """

    @dataclass
    class State:
        """Composed state, keyed by each :class:`ControlLaw`'s ``label``.

        ``control_law_states[label]`` is the per-law sub-state, or
        ``None`` for stateless laws. Dict insertion order matches the
        order laws were passed to the :class:`Controller`.
        """

        control_law_states: dict[str, ControlLaw.State | None] = field(default_factory=dict)

    def __init__(
        self,
        control_laws: list[ControlLaw],
        requires_grad: bool = False,
        device: wp.Device | None = None,
    ):
        if not control_laws:
            raise ValueError("Controller requires at least one ControlLaw.")
        self._control_laws = list(control_laws)
        self._requires_grad = requires_grad

        # Labels must be unique within a Controller — they're the dict
        # keys into the composed state. Catching collisions at construction
        # makes the failure mode loud instead of silently overwriting.
        seen: set[str] = set()
        for c in self._control_laws:
            if not hasattr(c, "label") or not isinstance(c.label, str):
                raise TypeError(f"Controller: {type(c).__name__} is missing a `label: str` attribute.")
            if c.label in seen:
                raise ValueError(f"Controller: duplicate ControlLaw label '{c.label}'.")
            seen.add(c.label)

        # All laws must share num_outputs (the outer length of every
        # outputs() binding). Each law derives num_outputs from its own
        # port_indices at construction; the Controller cross-checks the
        # group-wide invariant here. Single shared value lets a single
        # Controller-wide concept (e.g. a future reset mask) apply across
        # the group.
        num_outputs = None
        for c in self._control_laws:
            bindings = c.outputs()
            if not bindings:
                raise ValueError(f"Controller: {type(c).__name__} returned no output bindings.")
            for _attr_name, out_indices in bindings:
                n = len(out_indices)
                if num_outputs is None:
                    num_outputs = n
                elif n != num_outputs:
                    raise ValueError(
                        f"Controller: all ControlLaws must share num_outputs (the outer length "
                        f"of their output bindings); got {num_outputs} and {n}."
                    )
        self._num_outputs = num_outputs
        self._device = device if device is not None else wp.get_device()

        for c in self._control_laws:
            c.finalize(self._device, self._num_outputs, requires_grad=self._requires_grad)

        self._output_bindings: list[tuple[str, wp.array[wp.uint32]]] = []
        for c in self._control_laws:
            self._output_bindings.extend(c.outputs())

    @property
    def device(self) -> wp.Device:
        return self._device

    @property
    def num_outputs(self) -> int:
        """Shared outer length of every ControlLaw's output bindings."""
        return self._num_outputs

    @property
    def requires_grad(self) -> bool:
        """Whether internal buffers were allocated with gradient support."""
        return self._requires_grad

    def is_stateful(self) -> bool:
        return any(c.is_stateful() for c in self._control_laws)

    def is_graphable(self) -> bool:
        return all(c.is_graphable() for c in self._control_laws)

    def state(self) -> Controller.State:
        """Allocate composed state, keyed by each ControlLaw's ``label``."""
        return Controller.State(
            control_law_states={
                c.label: c.state(self._num_outputs, self._device, requires_grad=self._requires_grad)
                for c in self._control_laws
            }
        )

    def step(
        self,
        input: Any,
        output: Any,
        current_state: Controller.State,
        next_state: Controller.State,
        dt: float,
    ) -> None:
        """Zero all output slots, then run each ControlLaw's :meth:`compute`.

        Args:
            input: User-supplied object whose attributes hold the read ports
                each ControlLaw declared (e.g. ``input.joint_q``,
                ``input.kp``). Duck-typed — any object on which
                ``getattr(input, name)`` returns a :class:`wp.array` works.
            output: User-supplied object whose attributes hold the write
                ports each ControlLaw declared. The slots indicated by the
                laws' ``outputs()`` bindings are zeroed before any
                :meth:`compute` runs, then each law writes via ``+=``.
            current_state: Current composed state (per-law sub-states for
                stateful laws; ``None`` entries for stateless ones).
            next_state: Next composed state. ``compute()`` populates the
                per-law sub-states here.
            dt: Timestep [s].
        """
        for attr_name, out_indices in self._output_bindings:
            out_array = _resolve_input_array(output, attr_name, name=attr_name)
            wp.launch(
                _zero_at_indices_kernel,
                dim=len(out_indices),
                inputs=[out_array, out_indices],
                device=self._device,
            )
        for c in self._control_laws:
            cur = current_state.control_law_states[c.label]
            nxt = next_state.control_law_states[c.label]
            c.compute(input, output, cur, nxt, dt)
