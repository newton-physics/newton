# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Controller — composer of multiple ControlLaws."""

from __future__ import annotations

from dataclasses import dataclass, field

import warp as wp

from .control_law import ControlLaw


@wp.kernel
def _zero_at_indices_kernel(
    array: wp.array[float],
    indices: wp.array[wp.uint32],
):
    i = wp.tid()
    array[indices[i]] = 0.0


class Controller:
    """Composes one or more :class:`ControlLaw` instances.

    Owns the per-step zero / compute sequence and the reset fan-out.
    ControlLaws run serially in registration order; their contributions
    accumulate via ``+=`` directly into their bound output arrays after a
    single upfront zero pass.

    Args:
        control_laws: One or more :class:`ControlLaw` instances. All must
            share the same device and the same per-binding ``num_outputs``
            (the outer length of every ``outputs()`` binding).
        requires_grad: Single source of truth for gradient support across the
            controller. Propagated to each ControlLaw's :meth:`ControlLaw.finalize`
            and :meth:`ControlLaw.state`. When ``True``, every internally-allocated
            buffer is created with ``requires_grad=True`` so the ControlLaws'
            kernel launches are transparent to :class:`wp.Tape` — Isaac Lab
            and other autograd consumers can differentiate through the
            controller end-to-end. Users with mixed-grad needs split into
            multiple Controllers.
    """

    @dataclass
    class State:
        """Composed state: one entry per ControlLaw (``None`` for stateless)."""

        control_law_states: list = field(default_factory=list)

    def __init__(self, control_laws: list[ControlLaw], requires_grad: bool = False):
        if not control_laws:
            raise ValueError("Controller requires at least one ControlLaw.")
        self._control_laws = list(control_laws)
        self._requires_grad = requires_grad

        # Walk every output binding once: pick the device, enforce device
        # agreement, and enforce that all bindings share an outer length
        # (the group-wide num_outputs / reset-mask length).
        device = None
        num_outputs = None
        for c in self._control_laws:
            bindings = c.outputs()
            if not bindings:
                raise ValueError(f"Controller: {type(c).__name__} returned no output bindings.")
            for out_array, out_indices in bindings:
                if device is None:
                    device = out_array.device
                elif out_array.device != device:
                    raise ValueError(
                        f"Controller: ControlLaws' output arrays are on different devices "
                        f"({device} vs {out_array.device})."
                    )
                n = len(out_indices)
                if num_outputs is None:
                    num_outputs = n
                elif n != num_outputs:
                    raise ValueError(
                        f"Controller: all ControlLaws must share num_outputs (the outer length "
                        f"of their output bindings); got {num_outputs} and {n}."
                    )
        self._device = device
        self._num_outputs = num_outputs

        for c in self._control_laws:
            c.finalize(self._device, len(c.indices), requires_grad=self._requires_grad)

        self._output_bindings: list[tuple[wp.array, wp.array[wp.uint32]]] = []
        for c in self._control_laws:
            self._output_bindings.extend(c.outputs())

    @property
    def device(self) -> wp.Device:
        return self._device

    @property
    def num_outputs(self) -> int:
        """Shared outer length of every ControlLaw's output bindings; also the
        required length of the ``mask`` passed to :meth:`reset`."""
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
        """Allocate composed state with one entry per ControlLaw."""
        return Controller.State(
            control_law_states=[
                c.state(len(c.indices), self._device, requires_grad=self._requires_grad) for c in self._control_laws
            ]
        )

    def step(
        self,
        current_state: Controller.State,
        next_state: Controller.State,
        dt: float,
    ) -> None:
        """Zero all outputs, then run each ControlLaw's :meth:`compute`."""
        for out_array, out_indices in self._output_bindings:
            wp.launch(
                _zero_at_indices_kernel,
                dim=len(out_indices),
                inputs=[out_array, out_indices],
                device=self._device,
            )
        for c, cur_s, nxt_s in zip(
            self._control_laws,
            current_state.control_law_states,
            next_state.control_law_states,
            strict=True,
        ):
            c.compute(cur_s, nxt_s, dt)

    def reset(self, state: Controller.State, mask: wp.array[wp.bool]) -> None:
        """Reset slots flagged by ``mask`` in every stateful ControlLaw.

        ``mask`` is a bool array of length :attr:`num_outputs` (the
        controller-wide shared output length validated at construction).
        ``control_law.reset(sub_state, mask)`` is called for every ControlLaw
        whose sub-state is not ``None``.
        """
        if len(mask) != self._num_outputs:
            raise ValueError(f"Controller.reset: mask length {len(mask)} must equal num_outputs={self._num_outputs}.")
        for c, sub_state in zip(self._control_laws, state.control_law_states, strict=True):
            if sub_state is not None:
                c.reset(sub_state, mask)
