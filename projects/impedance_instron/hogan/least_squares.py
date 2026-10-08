# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Levenberg-Marquardt identification of shared runner parameters.

The cost is the sample part of :func:`~projects.impedance_instron.hogan.identify.score`
(coordinate and GRF mean squares, averaged over trials) plus the CEM offset
regularization, written as a residual vector. Finite-difference Jacobians and
the parallel damping ladder each run as one batched rollout; only the small
damped normal-equation solve runs on the host.

:class:`newton.ik.IKOptimizerLM` is not used: it optimizes articulation joint
coordinates against FK objectives and tiles the full Jacobian per row, whereas
here parameters drive a contact rollout and the Jacobian has ~10^4-10^5 rows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .identify import Parameterization, Trial, evaluate, predict_many, score
from .runner import RolloutConfig, Runner

_HIP_SCALE_M = 0.02
_ANGLE_SCALE_RAD = 0.05
_FORCE_SCALE_N = 100.0


@dataclass(frozen=True)
class LMConfig:
    """Levenberg-Marquardt settings; selection uses training trials only."""

    iterations: int = 15
    step: float = 0.01
    """Finite-difference step in offset units."""
    central: bool = False
    """Use central instead of forward differences (twice the rollouts)."""
    damping: float = 1e-2
    """Initial Marquardt damping relative to diag(J^T J)."""
    ladder: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    """Damping multipliers evaluated in parallel each iteration."""
    bound: float = 1.5
    regularization: float = 0.01
    tolerance: float = 1e-4
    """Stop when an accepted step improves the cost by less than this fraction."""
    chunk: int = 128
    """Candidates per batched rollout."""

    def __post_init__(self):
        if self.iterations < 1 or self.chunk < 1 or not self.ladder:
            raise ValueError("iterations, chunk, and ladder must be positive and nonempty")
        values = (self.step, self.damping, self.bound, self.regularization, self.tolerance, *self.ladder)
        if not np.isfinite(values).all() or min(self.step, self.damping, self.bound, *self.ladder) <= 0:
            raise ValueError("LM step, damping, bound, and ladder must be finite and positive")
        if self.regularization < 0 or self.tolerance < 0:
            raise ValueError("regularization and tolerance must be nonnegative")


def residuals(trace: dict, summary: dict, trial: Trial) -> np.ndarray | None:
    """Return weighted sample residuals whose squared sum equals the score's sample terms.

    Matches the coordinate and GRF mean-square terms of :func:`identify.score`;
    peak, impulse, contact-duration and effort terms are excluded. Returns
    ``None`` for failed rollouts.
    """
    if summary["status"] != "completed":
        return None
    time = trace["time_s"]
    observed = (trial.time_s > 0) & (trial.time_s <= time[-1] + 1e-12)
    parts = []
    count = int(observed.sum())
    if count:
        simulated = np.column_stack([np.interp(trial.time_s[observed], time, trace["state"][:, c]) for c in range(6)])
        error = simulated - trial.q[observed]
        parts.append((error[:, :2] / (_HIP_SCALE_M * math.sqrt(2 * count))).ravel())
        parts.append((error[:, 2:] / (_ANGLE_SCALE_RAD * math.sqrt(4 * count))).ravel())
    steps = len(trace["grf_n"])
    if steps:
        measured = np.column_stack([np.interp(time[:-1], trial.force_time_s, trial.grf_n[:, c]) for c in range(2)])
        parts.append(((trace["grf_n"] - measured) / (_FORCE_SCALE_N * math.sqrt(2 * steps))).ravel())
    return np.concatenate(parts) if parts else np.zeros(0)


def motion_metrics(scores: list[dict]) -> dict:
    """Average per-stance errors in physical units over trials."""
    tracking = np.array([s["tracking_rmse"] for s in scores])
    force = np.array([s["grf_rmse_n"] for s in scores])
    return {
        "hip_rmse_m": tracking[:, :2].mean(0).tolist(),
        "joint_rmse_rad": float(tracking[:, 2:].mean()),
        "grf_rmse_n": force.mean(0).tolist(),
        "peak_fz_error_n": float(np.mean([s["peak_fz_error_n"] for s in scores])),
        "failed": sum(s["status"] != "completed" for s in scores),
    }


def _describe(metrics: dict) -> str:
    hip, force = metrics["hip_rmse_m"], metrics["grf_rmse_n"]
    return (
        f"hip x/y {1e3 * hip[0]:.0f}/{1e3 * hip[1]:.0f} mm, joints {math.degrees(metrics['joint_rmse_rad']):.1f} deg,"
        f" Fx/Fz {force[0]:.0f}/{force[1]:.0f} N, peak Fz {metrics['peak_fz_error_n']:+.0f} N"
    )


class _Rollouts:
    """Evaluate stacked residual vectors, reusing GPU batches across calls."""

    def __init__(self, parameters: Parameterization, trials: list[Trial], config: RolloutConfig, device: str):
        self.parameters, self.trials, self.config, self.device = parameters, trials, config, device
        self.batches = {}
        self.rollouts = 0

    def _predict(self, models: list[Runner]) -> list:
        if self.device == "cpu":
            return predict_many(models, self.trials, self.config, device="cpu")
        from .gpu_runner import GpuBatch  # noqa: PLC0415 - optional execution backend

        size = len(models)
        if size not in self.batches:
            t = self.trials
            self.batches[size] = GpuBatch(
                [x.chain for x in t],
                [x.shoe for x in t],
                [x.initial for x in t],
                [x.task for x in t],
                [x.duration_s for x in t],
                candidates=size,
                config=self.config,
                device=self.device,
            )
        return self.batches[size].evaluate(models)

    def __call__(self, offsets: np.ndarray, chunk: int, *, metrics: bool = False) -> list[np.ndarray | None]:
        """Return one residual vector per offset row, ``None`` if any trial fails.

        With ``metrics``, :attr:`metrics` holds :func:`motion_metrics` per row.
        """
        weight = 1.0 / math.sqrt(len(self.trials))
        result = []
        self.metrics = []
        for start in range(0, len(offsets), chunk):
            block = offsets[start : start + chunk]
            # Pad to a fixed width so each persistent batch keeps its captured graph.
            width = chunk if len(offsets) > chunk else len(block)
            padded = np.vstack((block, np.repeat(block[-1:], width - len(block), axis=0)))
            models = [self.parameters.model(x) for x in padded]
            rows = self._predict(models)[: len(block)]
            self.rollouts += len(padded) * len(self.trials)
            for model, row in zip(models, rows, strict=False):
                if metrics:
                    self.metrics.append(
                        motion_metrics(
                            [score(*pair, trial, model) for trial, pair in zip(self.trials, row, strict=True)]
                        )
                    )
                values = [
                    residuals(trace, summary, trial) for trial, (trace, summary) in zip(self.trials, row, strict=True)
                ]
                result.append(None if any(v is None for v in values) else weight * np.concatenate(values))
        return result


def _augment(r: np.ndarray, x: np.ndarray, regularization: float) -> np.ndarray:
    return np.concatenate((r, math.sqrt(regularization / len(x)) * x))


def fit_lm(
    baseline: Runner,
    trials: list[Trial],
    *,
    config: RolloutConfig | None = None,
    search: LMConfig | None = None,
    allow_incompatible: bool = False,
    device: str = "cpu",
) -> tuple[Runner, dict]:
    """Fit on training trials with Levenberg-Marquardt; evaluate held-out trials afterwards.

    Each iteration evaluates the Jacobian and all damping-ladder proposals as
    batched rollouts, accepts the lowest-cost completed proposal if it improves
    the cost, and otherwise increases damping. Failed perturbations fall back to
    a one-sided difference or a zero column for that iteration.
    """
    cfg, search = config or RolloutConfig(), search or LMConfig()
    train = [trial for trial in trials if trial.split == "train"]
    held_out = [trial for trial in trials if trial.split == "eval"]
    if not train:
        raise ValueError("Identification requires training trials")
    incompatible = [trial.id for trial in trials if not trial.provenance["compatibility"]["passed"]]
    if incompatible and not allow_incompatible:
        raise ValueError(f"Input compatibility failed for {len(incompatible)} trials; run inspect before fitting")
    parameters = Parameterization(baseline, [trial.task.speed_m_s for trial in train])
    size, reg = parameters.size, search.regularization
    rollouts = _Rollouts(parameters, train, cfg, device)
    x = np.zeros(size)
    r0 = rollouts(x[None], 1, metrics=True)[0]
    if r0 is None:
        raise ValueError("The initial model fails a training trial; LM needs a completed starting point")
    r = _augment(r0, x, reg)
    cost = float(r @ r)
    initial_cost = cost
    motion = initial_motion = rollouts.metrics[0]
    print(f"lm 0/{search.iterations}: cost {cost:.4g} | {_describe(motion)}", flush=True)
    damping = search.damping
    history = []
    started = perf_counter()
    for iteration in range(search.iterations):
        tick = perf_counter()
        eye = np.eye(size) * search.step
        perturbed = np.vstack((x + eye, x - eye)) if search.central else x + eye
        values = rollouts(perturbed, search.chunk)
        jacobian = np.zeros((len(r0), size))
        failed_columns = 0
        for i in range(size):
            plus = values[i]
            minus = values[size + i] if search.central else r0
            if plus is not None and minus is not None:
                jacobian[:, i] = (plus - minus) / (2 * search.step if search.central else search.step)
            elif search.central and (plus is not None or minus is not None):
                jacobian[:, i] = (plus - r0) / search.step if plus is not None else (r0 - minus) / search.step
            else:
                failed_columns += 1
        jacobian = np.vstack((jacobian, math.sqrt(reg / size) * np.eye(size)))
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ r
        scale = np.maximum(np.diag(normal), 1e-12 * max(np.diag(normal).max(), 1e-300))
        proposals = []
        for multiplier in search.ladder:
            delta = np.linalg.solve(normal + damping * multiplier * np.diag(scale), -gradient)
            proposals.append(np.clip(x + delta, -search.bound, search.bound))
        proposals = np.asarray(proposals)
        costs = []
        candidate_residuals = rollouts(proposals, len(proposals), metrics=True)
        for p, value in zip(proposals, candidate_residuals, strict=True):
            costs.append(math.inf if value is None else float(_augment(value, p, reg) @ _augment(value, p, reg)))
        best = int(np.argmin(costs))
        accepted = costs[best] < cost
        improvement = (cost - costs[best]) / cost if accepted else 0.0
        if accepted:
            x, r0 = proposals[best], candidate_residuals[best]
            r, cost = _augment(r0, x, reg), costs[best]
            motion = rollouts.metrics[best]
            damping = max(damping * search.ladder[best] / 3.0, 1e-12)
        else:
            damping = min(damping * max(search.ladder) * 10.0, 1e12)
        history.append(
            {
                "iteration": iteration,
                "cost": cost,
                "motion": motion,
                "accepted": accepted,
                "damping": damping,
                "ladder_costs": [c if math.isfinite(c) else None for c in costs],
                "failed_jacobian_columns": failed_columns,
                "wall_s": perf_counter() - tick,
            }
        )
        print(
            f"lm {iteration + 1}/{search.iterations}: cost {cost:.4g} ({'accepted' if accepted else 'rejected'})"
            f" | {_describe(motion)} | damping {damping:.3g}, failed columns {failed_columns},"
            f" {perf_counter() - tick:.0f} s",
            flush=True,
        )
        if accepted and improvement < search.tolerance:
            break
    learned = parameters.model(x)
    results = {
        "train": {
            "baseline": evaluate(baseline, train, cfg, device=device),
            "learned": evaluate(learned, train, cfg, device=device),
        }
    }
    if held_out:
        results["eval"] = {
            "baseline": evaluate(baseline, held_out, cfg, device=device),
            "learned": evaluate(learned, held_out, cfg, device=device),
        }
    return learned, {
        "schema": "generative_runner_identification_lm_1",
        "validated": False,
        "reference_inputs_used": False,
        "device": device,
        "selection_split": "train",
        "method": "levenberg_marquardt",
        "objective": "coordinate and GRF mean squares of identify.score plus offset regularization",
        "initial_model": baseline.to_dict(),
        "search": asdict(search),
        "rollout": asdict(cfg),
        "parameters": size,
        "initial_cost": initial_cost,
        "final_cost": cost,
        "initial_motion": initial_motion,
        "final_motion": motion,
        "rollouts": rollouts.rollouts,
        "wall_s": perf_counter() - started,
        "history": history,
        "incompatible_trials": incompatible,
        "allow_incompatible": allow_incompatible,
        "splits": results,
        "trials": [{"id": trial.id, "split": trial.split, **trial.provenance} for trial in trials],
    }
