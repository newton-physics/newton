# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Signal vocabulary, hardware-interface wiring, and per-port helpers for
:mod:`newton.controllers`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import warp as wp


@dataclass(frozen=True, eq=False)  # identity equality — module-level constants are canonical
class ControlSignal:
    """A slot type — what *kind* of array fills this slot.

    Carries metadata only (dtype, ndim, description). Has no attribute
    name; the attribute name on the runtime ``input`` / ``output`` object
    is supplied per-deployment by a :class:`HardwareInterface`.

    Equality is by identity. The canonical signals are the module-level
    constants Newton (and user code) defines once and imports everywhere.

    Args:
        dtype: Warp element dtype of the array that fills this slot
            (``wp.float32``, ``wp.vec3``, ``wp.quat``, …).
        ndim: Rank of the array. 1 for the common case.
        description: Human-readable description; surfaces in error
            messages and docstrings.
    """

    dtype: type
    ndim: int
    description: str


@dataclass(frozen=True)
class HardwareInterface:
    """Per-deployment wiring: which attribute on the runtime ``input`` /
    ``output`` object holds each signal's live array.

    A :class:`HardwareInterface` is a flat record of two mappings,
    ``inputs`` and ``outputs``, each from :class:`ControlSignal` to the
    attribute-name string. A signal may appear in either or both
    directions if the deployment legitimately reads and writes it.

    Newton ships a small set of canonical signals as module-level
    constants but does **not** ship a pre-built interface; every user
    assembles their own from Newton's signals plus their own.
    """

    inputs: Mapping[ControlSignal, str] = field(default_factory=dict)
    outputs: Mapping[ControlSignal, str] = field(default_factory=dict)

    def __post_init__(self):
        # Freeze the mappings so callers can't mutate them after the
        # Controller has validated coverage. MappingProxyType is the
        # standard immutable-dict wrapper.
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

        # Two signals colliding on the same attribute name in the same
        # direction would silently mean "writes from law A and law B land
        # on the same backing array under different signal identities" —
        # almost never what the user wants. Raise.
        for direction, mapping in (("inputs", self.inputs), ("outputs", self.outputs)):
            seen: dict[str, ControlSignal] = {}
            for signal, attr in mapping.items():
                if attr in seen and seen[attr] is not signal:
                    raise ValueError(
                        f"HardwareInterface.{direction}: two distinct signals map to attribute "
                        f"'{attr}' (descriptions: '{seen[attr].description}' and '{signal.description}')."
                    )
                seen[attr] = signal

    def covers_inputs(self, signals: set[ControlSignal]) -> bool:
        return signals <= self.inputs.keys()

    def covers_outputs(self, signals: set[ControlSignal]) -> bool:
        return signals <= self.outputs.keys()


def _normalize_port(
    spec: Any,
    *,
    name: str,
) -> tuple[ControlSignal, wp.array[wp.uint32]]:
    """Validate a port spec at :meth:`ControlLaw.__init__` time.

    Every port spec is a 2-tuple ``(signal, port_indices)``:

    - ``signal`` (:class:`ControlSignal`): the slot type. Identifies
      which hardware-interface entry this port reads from / writes to.
    - ``port_indices`` (``wp.array[wp.uint32]``): per-element kernel
      lookup; ``arr[port_indices[i]]`` for per-DOF ports,
      ``arr[port_indices[r]]`` for per-robot ports.

    Args:
        spec: ``(ControlSignal, wp.array[wp.uint32])``.
        name: Port name used in error messages.

    Returns:
        ``(signal, port_indices)`` ready to stash on the ControlLaw.
    """
    if not isinstance(spec, tuple) or len(spec) != 2:
        raise TypeError(f"Port '{name}': expected a 2-tuple (ControlSignal, port_indices), got {type(spec).__name__}.")
    signal, port_indices = spec
    if not isinstance(signal, ControlSignal):
        raise TypeError(f"Port '{name}': first tuple element must be a ControlSignal, got {type(signal).__name__}.")
    if not isinstance(port_indices, wp.array):
        raise TypeError(f"Port '{name}': second tuple element must be wp.array, got {type(port_indices).__name__}.")
    return signal, port_indices


def _resolve_input_array(
    source: Any,
    attr_name: str,
    *,
    name: str,
) -> wp.array:
    """Step-time: fetch a port array from ``source`` (an ``input`` or
    ``output`` object) and validate it's a :class:`wp.array`.

    The attribute name is the one the law stashed via
    :meth:`ControlLaw._resolve` against the Controller's
    :class:`HardwareInterface`. No dtype/shape validation here — kernel
    launches surface mismatches with precise Warp diagnostics. The cheap
    type check catches typos (``input.joint_q = some_list``) early.
    """
    try:
        arr = getattr(source, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Port '{name}': source object has no attribute '{attr_name}'.") from e
    if not isinstance(arr, wp.array):
        raise TypeError(f"Port '{name}': source.{attr_name} must be wp.array, got {type(arr).__name__}.")
    return arr
