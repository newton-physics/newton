# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched numerical searches, independent of the model and the evidence."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import is_real as _is_real

logger = logging.getLogger(__name__)


@dataclass
class SearchOutcome:
    """The result of a search."""

    vector: list[float]
    """The best decision vector found."""

    loss: float
    """The objective value of :attr:`vector`."""

    iterations: int
    """Number of iterations done, not counting the scored start."""

    interrupted: bool
    """Whether an interrupt stopped the search. Then :attr:`vector` is the best so far, not a converged value."""

    history: list[tuple[float, list[float]]]
    """The best ``(loss, vector)`` so far after each iteration. Entry 0 is the scored start."""

    stop_reason: str
    """Why the search stopped, for example ``"maxiter"``, a convergence criterion, ``"interrupted"``,
    ``"all candidates invalid"``, ``"error"`` or ``"stopped by the caller"``."""

    invalid: int
    """Number of evaluated candidates whose objective value was not finite."""

    error: Exception | None
    """The error that stopped the search, or ``None``. Then :attr:`vector` is the best of the
    iterations completed before the error."""


class CMASettings:
    """Settings of a CMA-ES search over the full decision vector.

    Requires the ``cma`` package (``newton[calibration]``).

    Args:
        seed: Random seed in ``[0, 2**32 - 1]``. ``None`` uses a new seed, so
            the run cannot be repeated.
        sigma0: Initial step size.
        maxiter: Maximum number of iterations. The convergence criteria of
            CMA-ES can stop the search earlier.
        popsize: Number of candidates per iteration, evaluated in one batch.

    Raises:
        ValueError: If ``seed`` is not ``None`` or an integer in range,
            ``sigma0`` is not a finite positive number, ``maxiter`` is not an
            integer of at least 1, or ``popsize`` is not an integer of at least 2.
    """

    def __init__(self, *, seed: int | None = None, sigma0: float = 0.1, maxiter: int = 500, popsize: int = 128):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError(f"CMA seed must be None or an integer in [0, 2**32 - 1], got {seed!r}.")
        if not _is_real(sigma0) or not math.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError(f"CMA sigma0 must be a finite number > 0, got {sigma0!r}.")
        if type(maxiter) is not int or maxiter < 1:
            raise ValueError(f"CMA maxiter must be an integer >= 1, got {maxiter!r}.")
        if type(popsize) is not int or popsize < 2:
            raise ValueError(f"CMA popsize must be an integer >= 2, got {popsize!r}.")
        self.seed = seed
        self.sigma0 = sigma0
        self.maxiter = maxiter
        self.popsize = popsize

    def to_dict(self) -> dict[str, Any]:
        """Return the ``kind`` and the settings of the optimizer."""
        return {"kind": "cma", **vars(self)}

    def start(self, space) -> Search:
        """Start a search of ``space``.

        Args:
            space: Object with ``x0()``, the search start, ``bounds()``, the
                lower and upper bound of each decision variable (infinite for no
                bound), and ``describe(vector)``, which returns the searched
                values as text. The search stays within the bounds.

        Returns:
            The search. Drive it with :meth:`Search.ask` and :meth:`Search.tell`.

        Raises:
            ImportError: If the ``cma`` package is not installed.
        """
        try:
            import cma  # noqa: PLC0415 -- optional dependency
        except ImportError as exc:
            raise ImportError("CMASettings requires the 'cma' package; install newton[calibration].") from exc

        lower, upper = space.bounds()
        opts = {
            "bounds": [list(lower), list(upper)],
            "maxiter": self.maxiter,
            "popsize": self.popsize,
            "tolconditioncov": 1e12,
            "verbose": -9,
            "verb_log": 0,
            "randn": np.random.RandomState(self.seed).randn,
        }
        if self.seed is not None:
            opts["seed"] = self.seed
        # The strategy object of the cma package has the ask(), tell() and stop() that Search needs.
        return Search(cma.CMAEvolutionStrategy(list(space.x0()), self.sigma0, opts), space)


class Search:
    """A running search, driven by the caller.

    Get the decision vectors with :meth:`ask`, evaluate them, and give the
    values to :meth:`tell`. Repeat until :attr:`stop_reason` is not ``None``,
    or stop earlier. Then get the result with :meth:`outcome`.

    The first :meth:`ask` returns only the start, so the best vector is never
    worse than the start. After each :meth:`tell`, the best loss and
    ``space.describe(vector)`` are logged at ``INFO`` level. The log record also
    has the attributes ``iteration``, ``loss`` and ``vector`` (the best decision
    vector so far), so a log handler can follow the search.

    A value that is not finite marks an invalid candidate: the engine gets an
    infinite value for it, so it ranks below every valid candidate, and it is
    never the best. The search stops with the stop reason
    ``"all candidates invalid"`` when a whole iteration is invalid.

    Create it with the ``start(space)`` of the optimizer settings, for example
    :meth:`CMASettings.start`.

    Args:
        engine: The optimizer. It has ``ask()``, which returns the decision
            vectors of the next iteration, ``tell(vectors, values)``, and
            ``stop()``, which returns the names of the stop criteria that were
            met, empty while the search runs. It must accept infinite values.
        space: Object with ``x0()``, the search start, and ``describe(vector)``,
            which returns the searched values as text.
    """

    def __init__(self, engine, space):
        self._engine = engine
        self._x0 = list(space.x0())
        self._space = space
        self._asked = None
        self.history: list[tuple[float, list[float]]] = []
        """The best ``(loss, vector)`` so far after each iteration. Entry 0 is the scored start."""
        self.invalid = 0
        """Number of evaluated candidates whose value was not finite."""
        self.stop_reason: str | None = None
        """Why the search stopped: the names of the engine's stop criteria that were met, or
        ``"all candidates invalid"``. ``None`` while the search runs."""

    @property
    def iteration(self) -> int:
        """Number of iterations told, not counting the start. -1 before the start is told."""
        return len(self.history) - 1

    @property
    def best_loss(self) -> float:
        """The lowest valid value told so far. Infinite before the start is told."""
        return self.history[-1][0] if self.history else math.inf

    def ask(self) -> list[list[float]]:
        """Return the decision vectors to evaluate next.

        Raises:
            RuntimeError: If the search has stopped, or the values of the last
                :meth:`ask` were not told.
        """
        if self.stop_reason is not None:
            raise RuntimeError(f"the search has stopped: {self.stop_reason}.")
        if self._asked is not None:
            raise RuntimeError("tell() the values of the last ask() first.")
        # The engine samples around the start but does not score it.
        self._asked = [self._x0] if not self.history else [list(x) for x in self._engine.ask()]
        return self._asked

    def tell(self, values) -> None:
        """Give the values of the vectors from the last :meth:`ask`, one per vector, in order.

        Raises:
            RuntimeError: If :meth:`ask` was not called first.
            ValueError: If there is not one value per vector, or the start is invalid.
        """
        if self._asked is None:
            raise RuntimeError("ask() for vectors before tell().")
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self._asked),):
            raise ValueError("tell() needs one value per decision vector.")
        vectors, self._asked = self._asked, None
        valid = np.isfinite(values)
        n_invalid = int(np.count_nonzero(~valid))
        if not self.history:
            if n_invalid:
                raise ValueError("the search start is invalid: its objective value is not finite.")
            self._record(float(values[0]), vectors[0])
        else:
            self.invalid += n_invalid
            loss, vector = self.history[-1]
            if valid.any():
                ranked = np.where(valid, values, math.inf)
                self._engine.tell(vectors, ranked)
                i = int(np.argmin(ranked))
                if ranked[i] < loss:
                    loss, vector = float(ranked[i]), vectors[i]
            self._record(loss, vector, n_invalid)
            if not valid.any():
                self.stop_reason = "all candidates invalid"
        if self.stop_reason is None and self._engine.stop():
            self.stop_reason = ", ".join(self._engine.stop())

    def _record(self, loss: float, vector: list[float], n_invalid: int = 0) -> None:
        """Add the best so far to the history and log it."""
        self.history.append((loss, list(vector)))
        note = f", {n_invalid} invalid" if n_invalid else ""
        logger.info(
            "iteration %d: loss %.6g, %s%s",
            self.iteration,
            loss,
            self._space.describe(vector),
            note,
            extra={"iteration": self.iteration, "loss": loss, "vector": list(vector)},
        )

    def outcome(self, *, interrupted: bool = False, error: Exception | None = None) -> SearchOutcome:
        """Return the best vector found and log the stop reason.

        Args:
            interrupted: Whether an interrupt stopped the search.
            error: The error that stopped the search, or ``None``. The outcome
                then holds the best of the iterations completed before it.

        Raises:
            RuntimeError: If the start was not told.
        """
        if not self.history:
            raise RuntimeError("the search stopped before it scored the start.")
        if interrupted:
            stop_reason = "interrupted"
        elif error is not None:
            stop_reason = "error"
        else:
            stop_reason = self.stop_reason or "stopped by the caller"
        logger.info("search stopped: %s", stop_reason if error is None else f"error ({type(error).__name__}: {error})")
        loss, vector = self.history[-1]
        return SearchOutcome(
            list(vector), loss, self.iteration, interrupted, list(self.history), stop_reason, self.invalid, error
        )
