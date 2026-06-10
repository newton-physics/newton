# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Controller — composer of multiple :class:`ControlLaw`s.

The Controller is the place where the abstract :class:`ControlSignal`s
that laws are bound to get tied to concrete attribute names on a
deployment's runtime ``input`` / ``output`` objects, via a
:class:`HardwareInterface`. Once composed, the Controller routes the
per-step zero / compute sequence and lets every law write into the
shared output arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import warp as wp

from .control_law import ControlLaw
from .utils import ControlSignal, HardwareInterface, _resolve_input_array


@wp.kernel
def _zero_at_indices_kernel(
    array: wp.array[float],
    indices: wp.array[wp.uint32],
):
    i = wp.tid()
    array[indices[i]] = 0.0


class Controller:
    """Composes one or more :class:`ControlLaw`s under a single
    :class:`HardwareInterface`.

    The Controller validates that every law's used signals are covered
    by the interface in the appropriate direction (inputs / outputs),
    asks each law to resolve its signal bindings against the interface
    (so step-time lookups are plain ``getattr``), and owns the per-step
    "zero outputs, then accumulate via ``+=``" pattern.

    Args:
        hw: The :class:`HardwareInterface` for this deployment. Every
            signal any of ``control_laws`` reads must appear in
            ``hw.inputs``; every signal any of them writes must appear in
            ``hw.outputs``. Two laws binding the same output signal
            accumulate via ``+=`` into the shared output array.
        control_laws: The laws to compose. Run in registration order.
        requires_grad: Propagated to each law's :meth:`finalize` /
            :meth:`state` so internally-allocated buffers carry gradient
            support. Useful for differentiable-control workflows.
        device: Device for internal allocations. Defaults to
            :func:`wp.get_device`.
    """

    @dataclass
    class State:
        """Composed state: one entry per :class:`ControlLaw`, keyed by
        the law's index in the controller's registration order."""

        control_law_states: list[ControlLaw.State | None] = field(default_factory=list)

    def __init__(
        self,
        hw: HardwareInterface,
        control_laws: list[ControlLaw],
        requires_grad: bool = False,
        device: wp.Device | None = None,
    ):
        if not control_laws:
            raise ValueError("Controller requires at least one ControlLaw.")
        if not isinstance(hw, HardwareInterface):
            raise TypeError(f"Controller: hw must be a HardwareInterface, got {type(hw).__name__}.")

        self._hw = hw
        self._control_laws = list(control_laws)
        self._requires_grad = requires_grad
        self._device = device if device is not None else wp.get_device()

        # Validate every law's signal usage is covered by the interface
        # in the correct direction, then ask each law to resolve its
        # bindings (so step-time can do plain getattr).
        for law in self._control_laws:
            missing_in = law._used_inputs - hw.inputs.keys()
            if missing_in:
                raise ValueError(
                    f"Controller: {type(law).__name__} reads signals not in hw.inputs: "
                    f"{[s.description for s in missing_in]}."
                )
            missing_out = law._used_outputs - hw.outputs.keys()
            if missing_out:
                raise ValueError(
                    f"Controller: {type(law).__name__} writes signals not in hw.outputs: "
                    f"{[s.description for s in missing_out]}."
                )
            law._resolve(hw)
            law.finalize(self._device, requires_grad=self._requires_grad)

        # Collect every law's resolved (attr_name, port_indices) output
        # binding into a flat list. The Controller zeros these slots at
        # the start of each step before any law runs, then each law
        # writes via += into the shared arrays.
        self._output_bindings: list[tuple[str, wp.array[wp.uint32]]] = []
        for law in self._control_laws:
            self._output_bindings.extend(law.outputs())

    @property
    def hw(self) -> HardwareInterface:
        return self._hw

    @property
    def device(self) -> wp.Device:
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def is_stateful(self) -> bool:
        return any(c.is_stateful() for c in self._control_laws)

    def is_graphable(self) -> bool:
        return all(c.is_graphable() for c in self._control_laws)

    def state(self) -> Controller.State:
        """Allocate composed state with one entry per :class:`ControlLaw`
        (``None`` for stateless laws), in registration order."""
        return Controller.State(
            control_law_states=[
                law.state(self._device, requires_grad=self._requires_grad) for law in self._control_laws
            ]
        )

    def input(self) -> SimpleNamespace:
        """Allocate a fresh ``input`` object — a :class:`SimpleNamespace`
        with one ``wp.array`` per signal the composed laws actually read.

        Each field is zero-initialized to size ``max(port_indices) + 1``
        across the laws that bind that signal. The user is expected to
        either use this as-is or rebind individual fields to share
        arrays with sim state.
        """
        return self._make_namespace(direction="inputs")

    def output(self) -> SimpleNamespace:
        """Allocate a fresh ``output`` object. See :meth:`input`."""
        return self._make_namespace(direction="outputs")

    def step(
        self,
        input: Any,
        output: Any,
        current_state: Controller.State,
        next_state: Controller.State,
        dt: float,
    ) -> None:
        """Zero declared output slots, then run each :class:`ControlLaw`'s
        :meth:`compute` in registration order.

        Args:
            input: Object whose attributes hold the read ports' live
                arrays (per the :class:`HardwareInterface`).
            output: Object whose attributes hold the write ports' live
                arrays.
            current_state: Composed state to read from.
            next_state: Composed state to write to.
            dt: Timestep [s].
        """
        for attr_name, port_indices in self._output_bindings:
            out_array = _resolve_input_array(output, attr_name, name=attr_name)
            wp.launch(
                _zero_at_indices_kernel,
                dim=len(port_indices),
                inputs=[out_array, port_indices],
                device=self._device,
            )
        for law, cur_s, nxt_s in zip(
            self._control_laws,
            current_state.control_law_states,
            next_state.control_law_states,
            strict=True,
        ):
            law.compute(input, output, cur_s, nxt_s, dt)

    # --- internals ---

    def _make_namespace(self, *, direction: str) -> SimpleNamespace:
        # Walk every law's resolved bindings in `direction`. For each
        # (attr_name, port_indices) pair, the array we allocate must be
        # at least max(port_indices) + 1 long. If two laws share an
        # attribute name with different port_indices, take the max
        # across both. Dtype is the signal's dtype (looked up by attr
        # name on hw).
        mapping = self._hw.inputs if direction == "inputs" else self._hw.outputs
        # Invert signal->name to name->signal for the dtype lookup.
        name_to_signal: dict[str, ControlSignal] = {name: sig for sig, name in mapping.items()}

        per_law = (law.inputs() if direction == "inputs" else law.outputs() for law in self._control_laws)

        sizes: dict[str, int] = {}
        for law_bindings in per_law:
            for attr_name, port_indices in law_bindings:
                length = int(port_indices.numpy().max()) + 1
                sizes[attr_name] = max(sizes.get(attr_name, 0), length)

        fields = {
            attr_name: wp.zeros(
                size,
                dtype=name_to_signal[attr_name].dtype,
                device=self._device,
                requires_grad=self._requires_grad,
            )
            for attr_name, size in sizes.items()
        }
        return SimpleNamespace(**fields)
