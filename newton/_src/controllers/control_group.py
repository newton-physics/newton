# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControlGroup — composer of multiple Controllers."""

from __future__ import annotations

from dataclasses import dataclass, field

import warp as wp

from .base import Controller


@wp.kernel
def _zero_at_indices_kernel(
    array: wp.array[float],
    indices: wp.array[wp.uint32],
):
    i = wp.tid()
    array[indices[i]] = 0.0


class ControlGroup:
    """Composes one or more :class:`Controller` instances.

    Owns the per-step zero / compute sequence and the reset fan-out.
    Controllers run serially in registration order; their contributions
    accumulate via ``+=`` directly into their bound output arrays after a
    single upfront zero pass.
    """

    @dataclass
    class State:
        """Composed state: one entry per controller (``None`` for stateless)."""

        controller_states: list = field(default_factory=list)

    def __init__(self, controllers: list[Controller]):
        if not controllers:
            raise ValueError("ControlGroup requires at least one controller.")
        self._controllers = list(controllers)

        # Walk every output binding once: pick the device, enforce device
        # agreement, and enforce that all bindings share an outer length
        # (the group-wide num_outputs / reset-mask length).
        device = None
        num_outputs = None
        for c in self._controllers:
            bindings = c.outputs()
            if not bindings:
                raise ValueError(f"ControlGroup: {type(c).__name__} returned no output bindings.")
            for out_array, out_indices in bindings:
                if device is None:
                    device = out_array.device
                elif out_array.device != device:
                    raise ValueError(
                        f"ControlGroup: controllers' output arrays are on different devices "
                        f"({device} vs {out_array.device})."
                    )
                n = len(out_indices)
                if num_outputs is None:
                    num_outputs = n
                elif n != num_outputs:
                    raise ValueError(
                        f"ControlGroup: all controllers must share num_outputs (the outer length "
                        f"of their output bindings); got {num_outputs} and {n}."
                    )
        self._device = device
        self._num_outputs = num_outputs

        for c in self._controllers:
            c.finalize(self._device, len(c.indices))

        self._output_bindings: list[tuple[wp.array, wp.array[wp.uint32]]] = []
        for c in self._controllers:
            self._output_bindings.extend(c.outputs())

    @property
    def device(self) -> wp.Device:
        return self._device

    @property
    def num_outputs(self) -> int:
        """Shared outer length of every controller's output bindings; also the
        required length of the ``mask`` passed to :meth:`reset`."""
        return self._num_outputs

    def is_stateful(self) -> bool:
        return any(c.is_stateful() for c in self._controllers)

    def is_graphable(self) -> bool:
        return all(c.is_graphable() for c in self._controllers)

    def state(self) -> ControlGroup.State:
        """Allocate composed state with one entry per controller."""
        return ControlGroup.State(controller_states=[c.state(len(c.indices), self._device) for c in self._controllers])

    def step(
        self,
        current_state: ControlGroup.State,
        next_state: ControlGroup.State,
        dt: float,
    ) -> None:
        """Zero all outputs, then run each controller's :meth:`compute`."""
        for out_array, out_indices in self._output_bindings:
            wp.launch(
                _zero_at_indices_kernel,
                dim=len(out_indices),
                inputs=[out_array, out_indices],
                device=self._device,
            )
        for c, cur_s, nxt_s in zip(
            self._controllers,
            current_state.controller_states,
            next_state.controller_states,
            strict=True,
        ):
            c.compute(cur_s, nxt_s, dt)

    def reset(self, state: ControlGroup.State, mask: wp.array[wp.bool]) -> None:
        """Reset slots flagged by ``mask`` in every stateful controller.

        ``mask`` is a bool array of length :attr:`num_outputs` (the
        group-wide shared output length validated at construction).
        ``controller.reset(sub_state, mask)`` is called for every controller
        whose sub-state is not ``None``.
        """
        if len(mask) != self._num_outputs:
            raise ValueError(f"ControlGroup.reset: mask length {len(mask)} must equal num_outputs={self._num_outputs}.")
        for c, sub_state in zip(self._controllers, state.controller_states, strict=True):
            if sub_state is not None:
                c.reset(sub_state, mask)
