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

        # Pick the device from the first bound output array; validate agreement.
        device = None
        for c in self._controllers:
            for out_array, _ in c.outputs():
                if device is None:
                    device = out_array.device
                elif out_array.device != device:
                    raise ValueError(
                        f"ControlGroup: controllers' output arrays are on different devices "
                        f"({device} vs {out_array.device})."
                    )
        if device is None:
            raise ValueError("ControlGroup: no output arrays found on any controller.")
        self._device = device

        # Finalize each controller (allocates private buffers + reset_state).
        for c in self._controllers:
            c.finalize(self._device, len(c.indices))

        # Collect bindings to be zeroed at the start of every step.
        self._output_bindings: list[tuple[wp.array, wp.array[wp.uint32]]] = []
        for c in self._controllers:
            self._output_bindings.extend(c.outputs())

    @property
    def device(self) -> wp.Device:
        return self._device

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
        """Reset masked DOFs in ``state``.

        Calls ``controller.reset(sub_state, mask)`` for every controller whose
        sub-state is not ``None``. No framework-level interpretation of
        ``mask`` — each controller handles it according to its own layout.
        """
        for c, sub_state in zip(self._controllers, state.controller_states, strict=True):
            if sub_state is not None:
                c.reset(sub_state, mask)
