# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run artifacts: the result files, the search history and optional rendered traces."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .trace import CableTraceWriter


class CableRunWriter:
    """Write the artifacts of one run into an empty directory.

    Write the result and the search history with :meth:`write_result` before
    :meth:`write_traces`, so they are kept if a trace fails. Traces go to the
    ``trace`` subdirectory.

    Args:
        output_dir: Artifact directory. It must be absent or empty.
        trace_every: Render a rollout every N iterations, and of the last
            iteration. 0 disables traces.
        rollout: Rolls out the candidate of one decision vector with its frames
            kept and returns its :class:`~.evaluate.CableTraceView` entries.
        bundle: The :class:`~.evidence.CableEvidenceBundle` of the run, which
            supplies the reference images of the traces.
        bundle_dir: Directory that the bundle's relative paths resolve against.

    Raises:
        ValueError: If ``output_dir`` is not empty.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike,
        *,
        trace_every: int,
        rollout: Callable[[list[float]], list],
        bundle,
        bundle_dir: str | os.PathLike,
    ):
        self.directory = Path(output_dir)
        self.trace_every = trace_every
        self.rollout = rollout
        self.trace = None
        # Check now, before the search, but create the directory only when writing.
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError(f"output directory must be empty: {self.directory}")
        if trace_every:
            self.trace = CableTraceWriter(str(self.directory / "trace"), bundle, str(bundle_dir))

    @property
    def artifacts(self) -> dict[str, str]:
        """Paths that this writer writes, relative to the output directory."""
        paths = {"result": "result.json", "history": "history.json"}
        if self.trace is not None:
            paths["trace"] = "trace"
        return paths

    def write_result(self, result, history: list[tuple[float, list[float]]]) -> None:
        """Write ``result.json`` and ``history.json``.

        Args:
            result: The result of the run. Its output settings are set here.
            history: The best ``(loss, vector)`` after each iteration. Entry 0
                is the scored start.
        """
        result.provenance["run_spec"]["output"] = {
            "directory": str(self.directory.resolve()),
            "trace_every": self.trace_every,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write(
            "history.json",
            [{"iteration": i, "loss": loss, "vector": list(vector)} for i, (loss, vector) in enumerate(history)],
        )
        self._write("result.json", result.to_dict())

    def write_traces(self, history: list[tuple[float, list[float]]]) -> None:
        """Render the best vector of every ``trace_every``-th iteration and of the last one.

        Args:
            history: The best ``(loss, vector)`` after each iteration. Entry 0
                is the scored start.
        """
        if self.trace is None:
            return
        last = len(history) - 1
        for iteration, (loss, vector) in enumerate(history):
            if iteration % self.trace_every == 0 or iteration == last:
                self.trace.write(iteration, self.rollout(vector), loss)

    def _write(self, name: str, value: Any) -> None:
        with (self.directory / name).open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
