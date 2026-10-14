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
    """Why the search stopped, for example ``"maxiter"``, a convergence criterion, ``"interrupted"`` or
    ``"all candidates invalid"``."""

    invalid: int
    """Number of evaluated candidates whose objective value was not finite."""


class OptimizerCMA:
    """CMA-ES over the full decision vector.

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

    def minimize(self, objective, space) -> SearchOutcome:
        """Search ``space`` and return the best decision vector found.

        The search scores the start first, so the best vector is never worse
        than the start. After each iteration, the best loss and
        ``space.describe(vector)`` are logged at ``INFO`` level. The stop
        reason is the names of the CMA-ES stop criteria that were met.

        A candidate whose objective value is not finite is invalid: CMA-ES ranks
        it below every valid candidate, and it is never the best. The search
        stops with the stop reason ``"all candidates invalid"`` when a whole
        iteration is invalid.

        Args:
            objective: Object with ``evaluate(vectors)``, which returns one
                value per decision vector. A value that is not finite marks an
                invalid candidate.
            space: Object with ``dim``, ``x0()``, the search start,
                ``bounds()``, the lower and upper bound of each decision
                variable (infinite for no bound), and ``describe(vector)``,
                which returns the searched values as text. The search stays
                within the bounds.

        Returns:
            The search outcome.

        Raises:
            ImportError: If the ``cma`` package is not installed.
            ValueError: If the objective does not return one value per vector,
                or the start is invalid.
            RuntimeError: If the search stops before it scores the start.
        """
        try:
            import cma  # noqa: PLC0415 -- optional dependency
        except ImportError as exc:
            raise ImportError("OptimizerCMA requires the 'cma' package; install newton[calibration].") from exc

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
        x0 = list(space.x0())
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)

        best_x, best_loss, history, interrupted, invalid, stop_reason = None, math.inf, [], False, 0, None

        def record(n_invalid=0):
            history.append((best_loss, list(best_x)))
            note = f", {n_invalid} invalid" if n_invalid else ""
            logger.info("iteration %d: loss %.6g, %s%s", len(history) - 1, best_loss, space.describe(best_x), note)

        try:
            # CMA-ES samples around the start but does not score it.
            start_loss = float(_evaluate(objective, [x0])[0])
            if not math.isfinite(start_loss):
                raise ValueError("the search start is invalid: its objective value is not finite.")
            best_x, best_loss = x0, start_loss
            record()
            while not es.stop():
                solutions = es.ask()
                losses = _evaluate(objective, solutions)
                valid = np.isfinite(losses)
                n_invalid = int(np.count_nonzero(~valid))
                invalid += n_invalid
                if not valid.any():
                    record(n_invalid)
                    stop_reason = "all candidates invalid"
                    break
                # CMA-ES uses only the ranking, so an invalid candidate ranks below every valid one.
                ranked = np.where(valid, losses, math.inf)
                es.tell(solutions, ranked)
                i = int(np.argmin(ranked))
                if ranked[i] < best_loss:
                    best_x, best_loss = list(solutions[i]), float(ranked[i])
                record(n_invalid)
        except KeyboardInterrupt:
            interrupted = True

        if not history:
            raise RuntimeError("the search stopped before it scored the start.")
        if interrupted:
            stop_reason = "interrupted"
        elif stop_reason is None:
            stop_reason = ", ".join(es.stop())
        logger.info("search stopped: %s", stop_reason)
        best_loss, best_x = history[-1]
        return SearchOutcome(list(best_x), best_loss, len(history) - 1, interrupted, history, stop_reason, invalid)


def _evaluate(objective, vectors) -> np.ndarray:
    """Evaluate ``vectors`` and require one value per vector. A value may be non-finite."""
    values = np.asarray(objective.evaluate(vectors), dtype=float)
    if values.shape != (len(vectors),):
        raise ValueError("objective must return one scalar per decision vector.")
    return values
