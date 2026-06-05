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

    Owns the per-step zero / compute sequence and the reset fan-out.
    ControlLaws run serially in registration order; their contributions
    accumulate via ``+=`` directly into the output arrays after a single
    upfront zero pass.

    Args:
        control_laws: One or more :class:`ControlLaw` instances. All must
            agree on ``num_outputs`` (the outer length of every
            ``outputs()`` binding).
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

    def __init__(self, control_laws: list[ControlLaw], requires_grad: bool = False, device: wp.Device | None = None):
        """Args:
        control_laws: One or more :class:`ControlLaw`.
        requires_grad: see class docstring.
        device: Device for internal allocations. Defaults to
            :func:`wp.get_device`.
        """
        if not control_laws:
            raise ValueError("Controller requires at least one ControlLaw.")
        self._control_laws = list(control_laws)
        self._requires_grad = requires_grad

        # All output bindings must agree on ``num_outputs`` — the outer length
        # of every ``outputs()`` binding — so a single reset mask covers all
        # of them. Output *arrays* are looked up at step time (no longer
        # stored on the laws), so we can only validate the indices length
        # here, not the array shapes; that's deferred to step.
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
            c.finalize(self._device, len(c.indices), requires_grad=self._requires_grad)

        self._output_bindings: list[tuple[str, wp.array[wp.uint32]]] = []
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
        input: Any,
        output: Any,
        current_state: Controller.State,
        next_state: Controller.State,
        dt: float,
    ) -> None:
        """Zero all output slots, then run each ControlLaw's :meth:`compute`.

        Args:
            input: User-supplied object whose attributes hold the read ports
                each ControlLaw declared (e.g. ``input.joint_q``, ``input.kp``).
                Duck-typed — any object on which ``getattr(input, name)``
                returns a :class:`wp.array` is acceptable.
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
        # Resolve every output array against ``output`` *once* per step, then
        # zero its declared slots. If the user mutates ``output.<attr>`` to a
        # different array between steps, the next step picks up the new one.
        for attr_name, out_indices in self._output_bindings:
            out_array = _resolve_input_array(output, attr_name, name=attr_name)
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
            c.compute(input, output, cur_s, nxt_s, dt)

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
