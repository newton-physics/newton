# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable calibration problem: evidence, search space, objective and result."""

from __future__ import annotations

import copy
import datetime
import os
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

import warp as wp

import newton

from .artifacts import CableRunWriter
from .evaluate import CableCandidate, CableEvaluator, CableTraceView
from .evidence import CableEvidenceBundle
from .loss_chamfer import CHAMFER
from .optimizer import SearchOutcome
from .rehydrate import bundle_to_goals
from .result import STATUS_INTERRUPTED, STATUS_OK, CableCalibrationResult
from .run_spec import CableRunSpec
from .search_space import CableSearchSpace


class CableCalibrationProblem:
    """A cable experiment and the parameter space to search.

    Create the problem with :meth:`from_bundle` and give it to
    :func:`~.calibrate.calibrate`. Only the ``"chamfer"`` loss is supported. Each
    recording has unit weight, shared equally among its camera views.

    The problem keeps copies of the bundle and the run spec, so changes that
    the caller makes later do not change the problem. It decodes the masks
    once. It creates the simulation worlds only during an evaluation.
    """

    @classmethod
    def from_bundle(
        cls, bundle: CableEvidenceBundle, bundle_dir: str | os.PathLike, run_spec: CableRunSpec
    ) -> CableCalibrationProblem:
        """Validate a bundle and a run spec, and read the evidence into goals.

        Args:
            bundle: The cable evidence.
            bundle_dir: Directory that the bundle's relative paths resolve against.
            run_spec: Setup and fit settings. The optimizer and the output
                directory are given to :func:`~.calibrate.calibrate`, not read from here.

        Returns:
            The prepared problem.

        Raises:
            ValueError: If the bundle or the run spec is not valid, or a
                referenced file is missing.
        """
        self = cls.__new__(cls)
        self.bundle = copy.deepcopy(bundle)
        self.bundle_dir = Path(bundle_dir).resolve()
        self.run_spec = copy.deepcopy(run_spec)
        self.bundle.validate()
        self.run_spec.validate_problem()
        self.search_space = CableSearchSpace(self.run_spec)
        setup = self.run_spec.setup
        self.goals = bundle_to_goals(
            self.bundle,
            self.bundle_dir,
            attachment_transform=setup.attachment_transform,
            clamp_position=setup.clamp_position,
            cable_axis=setup.cable_axis,
        )
        self.objective = self
        self._evaluator = None
        return self

    def _prepare_evaluator(self):
        """Create the evaluator on first use and return it."""
        if self._evaluator is None:
            spec = self.run_spec
            setup = spec.setup
            self._evaluator = CableEvaluator(
                self.goals,
                CHAMFER,
                settle_frames=spec.settle_frames,
                settle_mode=spec.settle_mode,
                sim_iterations=spec.sim_iterations,
                stretch_stiffness=setup.stretch_stiffness,
                num_elements=setup.num_elements,
                segment_length=setup.segment_length,
                cable_radius=setup.cable_radius,
                cable_mass=setup.cable_mass,
                angle_parametrization=setup.angle_parametrization,
            )
        return self._evaluator

    def evaluate(self, vectors: Sequence[Sequence[float]]) -> list[float]:
        """Return the objective value of each decision vector."""
        candidates = [self.search_space.decode(x) for x in vectors]
        return self._prepare_evaluator().evaluate(candidates)

    def candidate(self, vector: Sequence[float]) -> CableCandidate:
        """Return the cable parameters that a decision vector describes."""
        return self.search_space.decode(vector)

    def build_result(self, outcome: SearchOutcome, *, optimizer, artifacts: dict[str, str]) -> CableCalibrationResult:
        """Build the result of a search.

        Args:
            outcome: The best vector and loss that the search found.
            optimizer: The optimizer that ran the search, for the provenance.
            artifacts: Paths that the run writes, relative to the output
                directory. Empty when the run writes no files.

        Returns:
            The validated result. Its ``fit`` holds only the fitted values. The
            setup needed to use them is in ``provenance["run_spec"]``.
        """
        candidate = self.candidate(outcome.vector)
        spec = self.run_spec
        optimizer_settings = optimizer.to_dict()
        effective_spec = spec.to_dict()
        effective_spec["optimizer"] = optimizer_settings
        effective_spec["output"] = {"directory": None, "trace_every": 0}
        result = CableCalibrationResult(
            status=STATUS_INTERRUPTED if outcome.interrupted else STATUS_OK,
            fit={
                "bend_angles": [list(a) for a in candidate.angles],
                "bend_stiffness": candidate.bend_stiffness,
                "twist_stiffness": candidate.twist_stiffness,
                "bend_damping": candidate.bend_damping,
                "twist_damping": candidate.twist_damping,
            },
            metrics={"loss": outcome.loss, "iterations": outcome.iterations},
            diagnostics={
                "searched_groups": spec.searched_groups(),
                "search_dimension": self.search_space.dim,
                "stop_reason": outcome.stop_reason,
            },
            provenance={
                "run_id": uuid.uuid4().hex,
                "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "bundle_path": str(self.bundle_dir),
                "run_spec": effective_spec,
                "seed": optimizer_settings.get("seed"),
                "package_versions": {
                    "python": sys.version.split()[0],
                    "newton": newton.__version__,
                    "warp": wp.__version__,
                },
            },
            artifacts=dict(artifacts),
        )
        result.validate()
        return result

    def create_writer(self, output_dir: str | os.PathLike, *, trace_every: int = 0) -> CableRunWriter:
        """Create the writer of the run artifacts.

        Args:
            output_dir: Artifact directory. It must be absent or empty.
            trace_every: Render a rollout every N iterations, and of the last
                iteration. 0 disables traces.
        """
        return CableRunWriter(
            output_dir,
            trace_every=trace_every,
            rollout=self._rollout,
            bundle=self.bundle,
            bundle_dir=self.bundle_dir,
        )

    def _rollout(self, vector: Sequence[float]) -> list[CableTraceView]:
        """Roll out the candidate of one decision vector with its frames kept."""
        return self._prepare_evaluator().record(self.candidate(vector))

    def close(self) -> None:
        """Release the evaluator. The decoded evidence stays available."""
        self._evaluator = None
