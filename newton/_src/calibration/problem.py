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
from .cable_setup import CableSetup
from .evaluate import CableEvaluator, CableParameters, CableTraceView
from .evidence import CableEvidenceBundle
from .loss_chamfer import CHAMFER
from .model import SETTLE_MODES
from .optimizer import SearchOutcome
from .rehydrate import bundle_to_goals
from .result import STATUS_FAILED, STATUS_INTERRUPTED, STATUS_OK, CableCalibrationResult
from .search_space import CableSearchSpace


class CableCalibrationProblem:
    """A cable experiment and the parameter space to search.

    Create the problem with :meth:`from_bundle` and give it to
    :func:`~.calibrate.calibrate`. The objective is the Chamfer loss. Each
    recording has unit weight, shared equally among its camera views.

    The problem keeps copies of the bundle, the setup and the search space, so
    changes that the caller makes later do not change the problem. It decodes
    the masks once. It creates the simulation worlds only during an evaluation.
    """

    @classmethod
    def from_bundle(
        cls,
        bundle: CableEvidenceBundle,
        bundle_dir: str | os.PathLike,
        setup: CableSetup,
        search_space: CableSearchSpace,
        *,
        settle_mode: str,
        settle_frames: int,
        sim_iterations: int,
    ) -> CableCalibrationProblem:
        """Validate the evidence and the settings, and read the evidence into goals.

        Args:
            bundle: The cable evidence.
            bundle_dir: Directory that the bundle's relative paths resolve against.
            setup: The physical cable setup.
            search_space: The start values, the searched fields and their bounds.
            settle_mode: How to reach the initial equilibrium. One of
                :data:`~.model.SETTLE_MODES`.
            settle_frames: Maximum number of frames spent settling before frame 0.
                0 does not settle.
            sim_iterations: Solver iterations per substep.

        Returns:
            The prepared problem.

        Raises:
            ValueError: If the bundle, the setup or a setting is not valid, the
                search space does not match the number of elements, or a
                referenced file is missing.
        """
        if not isinstance(setup, CableSetup):
            raise ValueError(f"setup must be a CableSetup, got {type(setup).__name__}.")
        if not isinstance(search_space, CableSearchSpace):
            raise ValueError(f"search_space must be a CableSearchSpace, got {type(search_space).__name__}.")
        if settle_mode not in SETTLE_MODES:
            raise ValueError(f"settle_mode {settle_mode!r} is not one of {list(SETTLE_MODES)}.")
        if type(settle_frames) is not int or settle_frames < 0:
            raise ValueError(f"settle_frames must be a nonnegative integer, got {settle_frames!r}.")
        if type(sim_iterations) is not int or sim_iterations <= 0:
            raise ValueError(f"sim_iterations must be a positive integer, got {sim_iterations!r}.")
        setup.validate()
        if len(search_space.start.angles) != setup.num_elements:
            raise ValueError(
                f"the search start has {len(search_space.start.angles)} rest-angle pair(s) "
                f"for {setup.num_elements} element(s)."
            )
        self = cls.__new__(cls)
        self.bundle = copy.deepcopy(bundle)
        self.bundle_dir = Path(bundle_dir).resolve()
        self.setup = copy.deepcopy(setup)
        self.search_space = copy.deepcopy(search_space)
        self.simulation = {"settle_mode": settle_mode, "settle_frames": settle_frames, "sim_iterations": sim_iterations}
        self.bundle.validate()
        self.goals = bundle_to_goals(
            self.bundle,
            self.bundle_dir,
            attachment_transform=self.setup.attachment_transform,
            clamp_position=self.setup.clamp_position,
            cable_axis=self.setup.cable_axis,
        )
        self.objective = self
        self._evaluator = None
        return self

    def _prepare_evaluator(self):
        """Create the evaluator on first use and return it."""
        if self._evaluator is None:
            setup = self.setup
            self._evaluator = CableEvaluator(
                self.goals,
                CHAMFER,
                **self.simulation,
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

    def candidate(self, vector: Sequence[float]) -> CableParameters:
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
            provenance holds what ran: the setup needed to use the values, the
            search space, the simulation settings and the optimizer settings.
        """
        candidate = self.candidate(outcome.vector)
        optimizer_settings = optimizer.to_dict()
        if outcome.error is not None:
            status = STATUS_FAILED
        elif outcome.interrupted:
            status = STATUS_INTERRUPTED
        else:
            status = STATUS_OK
        diagnostics = {
            "search_dimension": self.search_space.dim,
            "stop_reason": outcome.stop_reason,
            "invalid_candidates": outcome.invalid,
        }
        if outcome.error is not None:
            diagnostics["error"] = f"{type(outcome.error).__name__}: {outcome.error}"
        result = CableCalibrationResult(
            status=status,
            fit={
                "bend_angles": [list(a) for a in candidate.angles],
                "bend_stiffness": candidate.bend_stiffness,
                "twist_stiffness": candidate.twist_stiffness,
                "bend_damping": candidate.bend_damping,
                "twist_damping": candidate.twist_damping,
            },
            metrics={"loss": outcome.loss, "iterations": outcome.iterations},
            diagnostics=diagnostics,
            provenance={
                "run_id": uuid.uuid4().hex,
                "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "bundle_path": str(self.bundle_dir),
                "setup": self.setup.to_dict(),
                "search_space": self.search_space.to_dict(),
                "simulation": dict(self.simulation),
                "optimizer": optimizer_settings,
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
