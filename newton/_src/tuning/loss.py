# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""What a tuning objective has to provide."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp


class TuningLoss(ABC):
    """Abstract interface for one tuning objective.

    .. experimental::

    A loss is split into a one-time goal preparation and a per-candidate score, so
    any work that depends only on the observation is done once rather than for every
    candidate in every generation.

    Subclasses must implement :meth:`prepare` and :meth:`score`. A subclass that
    also scores the whole population on-device sets :attr:`supports_accum` and
    implements :meth:`make_accum` and :meth:`accum`; the class definition fails if
    it sets the flag without both methods.
    """

    wants_geometry: bool = False
    """The loss needs the cable's image-space geometry, not only the render.

    The caller then projects the capsule chain for that frame and passes it as
    ``geom``. A loss that works from pixels alone leaves this ``False``.
    """

    wants_render: bool = True
    """The loss needs the rendered binary mask (uint8, 0 or 255).

    Setting ``False`` lets the caller skip rendering entirely, which is the largest
    phase after stepping.
    """

    supports_accum: bool = False
    """The loss implements :meth:`make_accum` and :meth:`accum`.

    The caller uses the device path only when this is ``True`` and the render is
    on CUDA; otherwise it reads frames back and calls :meth:`score`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.supports_accum and (cls.make_accum is TuningLoss.make_accum or cls.accum is TuningLoss.accum):
            raise TypeError(f"{cls.__name__} sets supports_accum but does not implement make_accum and accum.")

    @abstractmethod
    def prepare(self, goal_mask: np.ndarray) -> Any:
        """Precompute the goal-side work for one observed frame.

        Args:
            goal_mask: Binary uint8 goal mask, cropped to the scoring region.

        Returns:
            An opaque goal representation, passed back to :meth:`score` or
            :meth:`accum` for every candidate.
        """

    @abstractmethod
    def score(
        self,
        goal: Any,
        sim_mask: np.ndarray | None,
        crop: Sequence[int],
        *,
        geom: np.ndarray | None = None,
    ) -> float:
        """Score one world's simulated frame against a prepared goal.

        Args:
            goal: The representation :meth:`prepare` returned.
            sim_mask: The world's rendered binary mask, or ``None`` when
                :attr:`wants_render` is ``False`` and nothing was rendered.
            crop: ``[x0, y0, x1, y1]`` pixel crop the goal was prepared in.
            geom: The world's projected cable, shape ``(n_nodes, 2)``, when
                :attr:`wants_geometry` is ``True``; otherwise ``None``.

        Returns:
            The frame's loss.
        """

    def make_accum(self, n_worlds: int, device: wp.DeviceLike) -> Any:
        """Allocate a per-world device accumulator for :meth:`accum`.

        Args:
            n_worlds: Number of worlds scored together.
            device: Warp device the accumulator lives on.

        Returns:
            An accumulator exposing ``totals()``, the per-world summed loss.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support on-device accumulation.")

    def accum(
        self,
        goal: Any,
        sim_mask: wp.array | None,
        crop: Sequence[int],
        accumulator: Any,
        *,
        geom: wp.array3d[wp.float32] | None = None,
    ) -> None:
        """Score every world's frame on-device, adding into ``accumulator``.

        Args:
            goal: The representation :meth:`prepare` returned.
            sim_mask: The rendered shape-ID mask for all worlds, or ``None`` when
                :attr:`wants_render` is ``False``.
            crop: ``[x0, y0, x1, y1]`` pixel crop the goal was prepared in.
            accumulator: The object :meth:`make_accum` returned.
            geom: The projected cable for all worlds, shape
                ``(n_worlds, n_nodes, 2)``, when :attr:`wants_geometry` is
                ``True``; otherwise ``None``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support on-device accumulation.")
