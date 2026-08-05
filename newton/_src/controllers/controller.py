# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Abstract base for Newton controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import warp as wp

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ControllerBase(ABC, Generic[InputT, OutputT]):
    """Abstract interface for a single Newton control law.

    Every concrete control law (joint impedance, differential IK, …) subclasses
    :class:`ControllerBase` directly. There is no framework-level composition: users
    who want to combine multiple control laws call each one's :meth:`step`
    in sequence themselves.

    Subclasses are responsible for:

    - :meth:`is_graphable`: predicate the user can query to decide whether
      graph capture is possible.
    - :meth:`input`, :meth:`output`: allocate fresh typed input/output structs.
      Baked-in arrays (gains passed as a ``wp.array`` at construction) are
      stored on the controller and do **not** appear on the input struct.
    - :meth:`step`: read the input struct's live arrays, run kernels, write
      the output struct's live arrays. Writes are slot-replacing (``=``, not
      ``+=``); composing laws is the user's job.
    """

    @abstractmethod
    def is_graphable(self) -> bool:
        """Whether :meth:`step` is safe to capture in a graph."""

    @abstractmethod
    def input(self) -> InputT:
        """Allocate a fresh input struct with zero-initialised arrays.

        The user typically reassigns fields to point at live data buffers.
        Fields for disabled features are ``None``.
        """

    @abstractmethod
    def output(self) -> OutputT:
        """Allocate a fresh output struct."""

    @abstractmethod
    def step(
        self,
        *,
        inputs: InputT,
        outputs: OutputT,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Run one control step.

        Args:
            inputs: Populated input struct (see :meth:`input`).
            outputs: Output struct to write into (see :meth:`output`).
            dt: Step duration [s].
        """
