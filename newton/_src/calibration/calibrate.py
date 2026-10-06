# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a search on a calibration problem."""

from __future__ import annotations

import os


def calibrate(
    problem,
    *,
    optimizer,
    output_dir: str | os.PathLike | None = None,
    trace_every: int = 0,
):
    """Search a calibration problem and return its result.

    The problem supplies ``search_space``, ``objective.evaluate(vectors)``,
    ``candidate(vector)``, ``build_result(outcome, optimizer=...)``,
    ``create_writer(output_dir, trace_every=...)`` and ``close()``. This
    function does not depend on the model type.

    The optimizer supplies ``to_dict()``, which returns its ``kind`` and
    settings, and ``minimize(objective, space)``, which returns a
    :class:`~.optimizer.SearchOutcome`. See
    :class:`~.optimizer.OptimizerCMA`. The optimizer settings, including the
    seed, come only from ``optimizer``.

    Args:
        problem: The calibration problem, for example a cable calibration
            problem.
        optimizer: The search, for example from
            :func:`~.optimizer.optimizer_from_spec`.
        output_dir: Artifact directory. It must be absent or empty. ``None``
            writes no files.
        trace_every: Record a rendered rollout every N iterations. 0 disables
            traces. Traces require ``output_dir``. The rollouts are rendered
            after the search, from :attr:`~.optimizer.SearchOutcome.history`.

    Returns:
        The result of the problem, a :class:`~.result.CableCalibrationResult` for a
        cable problem. Artifact paths are relative to ``output_dir``.

    Raises:
        ValueError: If a setting is not valid or the objective is not finite.
        RuntimeError: If the search stops before it scores the start.

    On ``KeyboardInterrupt`` after the start is scored, the optimizer returns
    the best candidate so far and the result status is ``"interrupted"``. The
    problem is closed when the search completes or fails.
    """
    writer = None
    try:
        # The result needs the optimizer settings, so get them before the search.
        optimizer.to_dict()
        if type(trace_every) is not int or trace_every < 0:
            raise ValueError("trace_every must be a nonnegative integer.")
        if trace_every and output_dir is None:
            raise ValueError("traces require output_dir.")
        if output_dir is not None:
            writer = problem.create_writer(output_dir, trace_every=trace_every)

        outcome = optimizer.minimize(problem.objective, problem.search_space)
        result = problem.build_result(outcome, optimizer=optimizer)
        if writer is not None:
            for iteration, (loss, vector) in enumerate(outcome.history):
                writer.record_iteration(iteration, problem.candidate(vector), loss)
            writer.finish(result, problem.candidate(outcome.vector), outcome.iterations)
        return result
    finally:
        problem.close()
