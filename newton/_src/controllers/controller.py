# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Abstract base for Newton controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import warp as wp


class Controller(ABC):
    """Abstract interface for a single Newton control law.

    Every concrete control law (joint impedance, differential IK, …) subclasses
    :class:`Controller` directly. There is no framework-level composition: users
    who want to combine multiple control laws call each one's :meth:`compute`
    in sequence themselves.

    Subclasses are responsible for:

    - :meth:`is_graphable`: predicate the user can query to decide CUDA-graph
      capture. Check ``ctrl.state() is not None`` to determine whether
      double-buffered state setup is needed.
    - :meth:`state`: allocate a fresh :class:`State`, or return ``None`` for
      stateless laws.
    - :meth:`input`, :meth:`output`: allocate fresh duck-typed containers
      holding only the live ports declared at construction. Baked-in arrays
      (gain ports given a :class:`wp.array` rather than an attr name) are
      stored on the controller and do **not** appear on the input struct.
    - :meth:`compute`: read the input struct's live arrays, run kernels, write
      the output struct's live arrays. Writes are slot-replacing (``=``, not
      ``+=``); composing laws is the user's job.
    """

    @abstractmethod
    def is_graphable(self) -> bool:
        """Whether :meth:`compute` is safe to capture in a CUDA graph."""

    @abstractmethod
    def input(self) -> Any:
        """Allocate a fresh input container with one field per input port.

        The returned object has attributes whose names match the ``*_attr``
        strings passed at construction. Each field is a freshly
        :func:`wp.zeros`-allocated array of the right dtype and size. The
        user typically reassigns these fields to point at live data buffers.
        """

    @abstractmethod
    def output(self) -> Any:
        """Allocate a fresh output container with one field per output port."""

    @abstractmethod
    def compute(
        self,
        inputs: Any,
        outputs: Any,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Run one control step.

        Args:
            inputs: Object whose attributes hold the live read ports.
                Resolved via ``getattr(inputs, attr_name)``. Any
                duck-typed object works; :meth:`input` returns a
                pre-allocated one.
            outputs: Same contract as ``inputs`` for write ports. The
                kernel performs slot-replacing writes — slots outside the
                declared port indices are left untouched.
            dt: Step duration [s]. Either a plain ``float`` or a
                ``wp.array`` of shape ``(1,)`` dtype ``wp.float32``.
        """
