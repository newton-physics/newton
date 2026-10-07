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
    ``build_result(outcome, optimizer=..., artifacts=...)``,
    ``create_writer(output_dir, trace_every=...)`` and ``close()``. The writer
    supplies ``artifacts``, ``write_result(result, history)`` and
    ``write_traces(history)``. This function does not depend on the model type.

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
        output_dir: Artifact directory. It must be absent or empty. The
            result and the search history are written to it before any trace.
            ``None`` writes no files.
        trace_every: Render a rollout every N iterations, and of the last
            iteration. 0 disables traces. Traces require ``output_dir``. The
            rollouts are rendered after the search, from
            :attr:`~.optimizer.SearchOutcome.history`.

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
        artifacts = writer.artifacts if writer is not None else {}
        result = problem.build_result(outcome, optimizer=optimizer, artifacts=artifacts)
        if writer is not None:
            # Write the result before the traces, so a failed trace keeps it.
            writer.write_result(result, outcome.history)
            writer.write_traces(outcome.history)
        return result
    finally:
        problem.close()
