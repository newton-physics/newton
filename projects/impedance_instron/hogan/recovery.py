# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check whether shared runner parameters can be recovered from synthetic data.

A known "truth" runner generates free rollouts from real trial initial states,
chains, and shoes. Gaussian noise is added on the measured clocks, then
:func:`~projects.impedance_instron.hogan.identify.fit` starts from the seed and
tries to recover the truth. Measured motion and force are never used, so this
test does not depend on the contact compatibility of the real references.

Two independent answers are reported:

- **Search recovery:** whether the fit reaches the truth's loss and, if so,
  whether its impedance (equilibrium, K, D along the truth trajectories) matches.
- **Local identifiability:** a finite-difference Fisher information analysis at
  the truth, which does not depend on the optimizer.

Run ``python -m projects.impedance_instron.hogan.recovery --help`` for options.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .identify import FitConfig, Parameterization, Trial, evaluate, fit, load_trials, predict_many
from .least_squares import LMConfig, fit_lm, residuals
from .runner import FEATURE_NAMES, OUTPUT_NAMES, RolloutConfig, Runner, State

JOINT_NAMES = ("hip", "knee", "ankle")


@dataclass(frozen=True)
class Noise:
    """Additive Gaussian measurement noise standard deviations."""

    hip_m: float = 0.002
    """Hip x/z position noise [m]."""
    angle_rad: float = 0.01
    """Pelvis and joint angle noise [rad]."""
    force_n: float = 20.0
    """Horizontal and vertical GRF noise [N]."""

    def __post_init__(self):
        values = (self.hip_m, self.angle_rad, self.force_n)
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Noise levels must be finite and nonnegative")

    @property
    def q(self) -> np.ndarray:
        return np.array([self.hip_m, self.hip_m] + [self.angle_rad] * 4)


@dataclass(frozen=True)
class Criteria:
    """Engineering thresholds for the recovery verdict, not statistical tests."""

    loss_tolerance: float = 0.1
    """Learned training loss may exceed the truth's by this fraction."""
    impedance_error: float = 0.05
    """Maximum mean normalized impedance error for recovery."""
    improvement: float = 0.25
    """Maximum learned/seed impedance error ratio for recovery."""
    well_determined_std: float = 0.1
    """Cramér-Rao standard deviation [offset units] below which a parameter is well determined."""
    poorly_determined_std: float = 1.0
    """Cramér-Rao standard deviation [offset units] above which a parameter is poorly determined."""


def parameter_names(parameters: Parameterization) -> list[str]:
    """Name every offset in a parameterization vector."""
    outputs, joints, features = np.unravel_index(parameters.indices, parameters.baseline.weights.shape)
    names = [
        f"{OUTPUT_NAMES[o]}.{JOINT_NAMES[j]}.{FEATURE_NAMES[f]}"
        for o, j, f in zip(outputs, joints, features, strict=True)
    ]
    names += ["log_frequency", "log_response_time"]
    if parameters.variable_speed:
        names.append("cadence_speed_gain")
    return names


def offsets_of(parameters: Parameterization, model: Runner) -> np.ndarray:
    """Invert :meth:`Parameterization.model` for a model it produced."""
    n = len(parameters.indices)
    base = parameters.baseline
    x = np.zeros(parameters.size)
    x[:n] = (model.weights - base.weights).ravel()[parameters.indices]
    x[n] = math.log(model.frequency_hz / base.frequency_hz)
    x[n + 1] = math.log(model.response_time_s / base.response_time_s)
    if parameters.variable_speed:
        x[-1] = model.cadence_speed_gain - base.cadence_speed_gain
    return x


def _sample(trace: dict, times_s: np.ndarray, force_times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = trace["time_s"]
    q = np.column_stack([np.interp(times_s, time, trace["state"][:, c]) for c in range(6)])
    grf = np.column_stack([np.interp(force_times_s, time[:-1], trace["grf_n"][:, c]) for c in range(2)])
    return q, grf


def draw_truth(
    parameters: Parameterization,
    trials: list[Trial],
    config: RolloutConfig,
    *,
    scale: float,
    bound: float,
    attempts: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[np.ndarray, Runner, list]:
    """Draw seed offsets until one completes every trial with simulated contact."""
    draws = np.clip(scale * rng.standard_normal((attempts, parameters.size)), -bound, bound)
    models = [parameters.model(x) for x in draws]
    for x, model, row in zip(draws, models, predict_many(models, trials, config, device=device), strict=True):
        if all(summary["status"] == "completed" and summary["contact_duration_s"] > 0 for _, summary in row):
            return x, model, row
    raise ValueError(f"None of {attempts} truth draws completed every trial with contact; reduce truth scale")


def synthesize(trials: list[Trial], predictions: list, noise: Noise, rng: np.random.Generator) -> list[Trial]:
    """Replace measured targets by noisy truth predictions on the trial clocks.

    Predictive inputs (chain, shoe, initial state, task, horizon) are unchanged.
    Forces are sampled on the measured clock clipped to the prediction horizon.
    """
    result = []
    for trial, (trace, summary) in zip(trials, predictions, strict=True):
        if summary["status"] != "completed":
            raise ValueError(f"{trial.id}: truth prediction failed")
        inside = (trial.force_time_s > 0) & (trial.force_time_s < trial.duration_s)
        force_time = np.concatenate(([0.0], trial.force_time_s[inside], [trial.duration_s]))
        q, grf = _sample(trace, trial.time_s, force_time)
        q = q + rng.standard_normal(q.shape) * noise.q
        grf = grf + rng.standard_normal(grf.shape) * noise.force_n
        provenance = {key: value for key, value in trial.provenance.items() if key != "compatibility"}
        provenance["source_compatibility"] = trial.provenance.get("compatibility")
        provenance["compatibility"] = {
            "passed": True,
            "basis": "synthetic targets generated by the same chain, shoe, and initial state",
        }
        provenance["synthetic"] = {"source_trial": trial.id, "noise": asdict(noise)}
        result.append(
            Trial(
                trial.id,
                trial.split,
                trial.chain,
                trial.shoe,
                trial.task,
                trial.initial,
                trial.time_s,
                q,
                force_time,
                grf,
                provenance,
            )
        )
    return result


def impedance_error(model: Runner, truth: Runner, trials: list[Trial], predictions: list, *, stride: int = 10) -> dict:
    """Compare equilibrium, K, and D along the truth trajectories.

    Errors are normalized by the equilibrium span and by the stiffness and
    damping maxima, so all nine channels are fractions of their bounded range.
    """
    differences = []
    for trial, (trace, _) in zip(trials, predictions, strict=True):
        for k in range(0, len(trace["load"]), stride):
            state = State(
                trace["state"][k],
                trace["velocity"][k],
                float(trace["phase_rad"][k]),
                max(float(trace["normal_load_bw"][k]), 0.0),
            )
            a = np.array(truth.impedance(state, trial.chain, trial.task))
            b = np.array(model.impedance(state, trial.chain, trial.task))
            differences.append(b - a)
    bounds = truth.bounds
    span = np.array(
        [
            np.subtract(bounds.equilibrium_upper_rad, bounds.equilibrium_lower_rad),
            bounds.stiffness_max_nm_rad,
            bounds.damping_max_nms_rad,
        ]
    )
    rms = np.sqrt(np.mean(np.square(differences), axis=0))
    normalized = rms / span
    return {
        "samples": len(differences),
        "rms": {
            name: dict(zip(JOINT_NAMES, row.tolist(), strict=True))
            for name, row in zip(("equilibrium_rad", "stiffness_nm_rad", "damping_nms_rad"), rms, strict=True)
        },
        "normalized": {
            name: dict(zip(JOINT_NAMES, row.tolist(), strict=True))
            for name, row in zip(OUTPUT_NAMES, normalized, strict=True)
        },
        "mean_normalized": float(normalized.mean()),
    }


def sensitivity(
    parameters: Parameterization,
    center: np.ndarray,
    trials: list[Trial],
    config: RolloutConfig,
    noise: Noise,
    *,
    step: float = 0.01,
    chunk: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Return the noise-whitened central-difference Jacobian of sampled observations.

    Rows are the trial observations (coordinates after the initial frame, then
    GRF) divided by their noise level; columns are parameter offsets. Columns
    whose perturbed rollouts fail are NaN.
    """
    if step <= 0 or chunk < 1:
        raise ValueError("step and chunk must be positive")
    if np.any(noise.q <= 0) or noise.force_n <= 0:
        raise ValueError("Sensitivity whitening needs positive noise levels")
    eye = np.eye(parameters.size) * step
    candidates = np.vstack((center + eye, center - eye))
    models = [parameters.model(x) for x in candidates]
    observations = []
    for start in range(0, len(models), chunk):
        for row in predict_many(models[start : start + chunk], trials, config, device=device):
            values = []
            for trial, (trace, summary) in zip(trials, row, strict=True):
                if summary["status"] != "completed":
                    values = None
                    break
                q, grf = _sample(trace, trial.time_s[1:], trial.force_time_s)
                values.append(np.concatenate(((q / noise.q).ravel(), (grf / noise.force_n).ravel())))
            observations.append(None if values is None else np.concatenate(values))
    size = next((len(o) for o in observations if o is not None), None)
    if size is None:
        raise ValueError("Every perturbed rollout failed; reduce the sensitivity step")
    jacobian = np.full((size, parameters.size), np.nan)
    for i in range(parameters.size):
        plus, minus = observations[i], observations[parameters.size + i]
        if plus is not None and minus is not None:
            jacobian[:, i] = (plus - minus) / (2 * step)
    return jacobian


def identifiability(jacobian: np.ndarray, names: list[str], criteria: Criteria) -> dict:
    """Summarize local identifiability from a whitened Jacobian.

    The Cramér-Rao bound assumes the declared Gaussian noise and a locally
    linear model; it is a lower bound on achievable parameter uncertainty.
    """
    valid = np.isfinite(jacobian).all(axis=0)
    j = jacobian[:, valid]
    kept = [name for name, ok in zip(names, valid, strict=True) if ok]
    _, singular, vt = np.linalg.svd(j, full_matrices=False)
    relative = singular / singular[0]
    # Floor tiny eigenvalues so unobservable directions report large, finite deviations.
    eigen = np.maximum(singular**2, singular[0] ** 2 * 1e-16)
    std = np.sqrt(np.sum(vt**2 / eigen[:, None], axis=0))
    weakest = []
    for k in range(len(singular) - 1, max(len(singular) - 4, -1), -1):
        top = np.argsort(-np.abs(vt[k]))[:5]
        weakest.append(
            {
                "relative_singular_value": float(relative[k]),
                "loadings": {kept[i]: float(vt[k, i]) for i in top},
            }
        )
    order = np.argsort(std)
    return {
        "observations": int(j.shape[0]),
        "parameters": int(j.shape[1]),
        "failed_columns": [name for name, ok in zip(names, valid, strict=True) if not ok],
        "relative_singular_values": relative.tolist(),
        "rank_relative_1e-3": int(np.sum(relative > 1e-3)),
        "rank_relative_1e-6": int(np.sum(relative > 1e-6)),
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
        "cramer_rao_std": {kept[i]: float(std[i]) for i in order},
        "well_determined": [kept[i] for i in order if std[i] < criteria.well_determined_std],
        "poorly_determined": [kept[i] for i in order if std[i] > criteria.poorly_determined_std],
        "weakest_directions": weakest,
    }


def verdict(truth_loss: float, learned_loss: float, learned_error: float, seed_error: float, criteria: Criteria) -> str:
    """Classify the search outcome as recovered, not identifiable, or inconclusive."""
    if learned_loss > truth_loss * (1 + criteria.loss_tolerance):
        return "inconclusive_search"
    if learned_error <= criteria.impedance_error and learned_error <= criteria.improvement * seed_error:
        return "recovered"
    return "not_identifiable"


def recover(
    trials: list[Trial],
    *,
    config: RolloutConfig | None = None,
    search: FitConfig | LMConfig | None = None,
    noise: Noise | None = None,
    criteria: Criteria | None = None,
    truth_scale: float = 0.3,
    truth_attempts: int = 16,
    seed: int = 0,
    device: str = "cpu",
    sensitivity_step: float | None = 0.01,
    sensitivity_chunk: int = 64,
) -> tuple[Runner, Runner, dict]:
    """Generate synthetic targets from a known truth and try to recover it.

    Returns the truth runner, the learned runner, and a JSON-serializable report.
    ``search`` selects the optimizer: :class:`LMConfig` (default) or CEM
    :class:`FitConfig`. Pass ``sensitivity_step=None`` to skip the Fisher analysis.
    """
    cfg, search = config or RolloutConfig(), search or LMConfig()
    noise, criteria = noise or Noise(), criteria or Criteria()
    train_speeds = [trial.task.speed_m_s for trial in trials if trial.split == "train"]
    if not train_speeds:
        raise ValueError("Recovery requires training trials")
    baseline = Runner.seed(reference_speed_m_s=float(np.mean(train_speeds)))
    parameters = Parameterization(baseline, train_speeds)
    names = parameter_names(parameters)
    rng = np.random.default_rng(seed)
    timings = {}
    started = perf_counter()
    truth_x, truth, predictions = draw_truth(
        parameters, trials, cfg, scale=truth_scale, bound=search.bound, attempts=truth_attempts, rng=rng, device=device
    )
    synthetic = synthesize(trials, predictions, noise, rng)
    timings["truth_s"] = perf_counter() - started
    started = perf_counter()
    if isinstance(search, LMConfig):
        learned, fit_report = fit_lm(baseline, synthetic, config=cfg, search=search, device=device)
    else:
        learned, fit_report = fit(baseline, synthetic, config=cfg, search=search, device=device)
    timings["fit_s"] = perf_counter() - started
    learned_x = offsets_of(parameters, learned)
    splits = {}
    for split in ("train", "eval"):
        subset = [i for i, trial in enumerate(synthetic) if trial.split == split]
        if not subset:
            continue
        chosen = [synthetic[i] for i in subset]
        truth_predictions = [predictions[i] for i in subset]
        splits[split] = {
            "truth_loss": evaluate(truth, chosen, cfg, device=device)["mean_loss"],
            "seed_loss": fit_report["splits"][split]["baseline"]["mean_loss"],
            "learned_loss": fit_report["splits"][split]["learned"]["mean_loss"],
            "learned_failed": fit_report["splits"][split]["learned"]["failed"],
            "seed_impedance_error": impedance_error(baseline, truth, chosen, truth_predictions),
            "learned_impedance_error": impedance_error(learned, truth, chosen, truth_predictions),
        }
    train = splits["train"]
    distance = float(np.linalg.norm(truth_x))
    report = {
        "schema": "generative_runner_recovery_1",
        "validated": False,
        "measured_targets_used": False,
        "device": device,
        "seed": seed,
        "truth_scale": truth_scale,
        "noise": asdict(noise),
        "criteria": asdict(criteria),
        "search": asdict(search),
        "method": "levenberg_marquardt" if isinstance(search, LMConfig) else "cem",
        "rollout": asdict(cfg),
        "parameters": parameters.size,
        "verdict": verdict(
            train["truth_loss"],
            train["learned_loss"],
            train["learned_impedance_error"]["mean_normalized"],
            train["seed_impedance_error"]["mean_normalized"],
            criteria,
        ),
        "splits": splits,
        "offsets": {
            "truth_distance_from_seed": distance,
            "learned_distance_from_truth": float(np.linalg.norm(learned_x - truth_x)),
            "relative_error": float(np.linalg.norm(learned_x - truth_x) / distance),
            "correlation": float(np.corrcoef(truth_x, learned_x)[0, 1]) if np.ptp(learned_x) > 0 else 0.0,
            "truth": dict(zip(names, truth_x.tolist(), strict=True)),
            "learned": dict(zip(names, learned_x.tolist(), strict=True)),
        },
        "fit_history": fit_report["history"],
        "timings_s": timings,
        "trials": [{"id": trial.id, "split": trial.split, **trial.provenance} for trial in synthetic],
    }
    if isinstance(search, LMConfig):
        train_rows = [(p, t) for p, t in zip(predictions, synthetic, strict=True) if t.split == "train"]
        data = np.mean([np.sum(residuals(*p, t) ** 2) for p, t in train_rows])
        report["lm_cost"] = {
            "truth": float(data + search.regularization * np.mean(truth_x**2)),
            "learned": fit_report["final_cost"],
            "rollouts": fit_report["rollouts"],
        }
    if sensitivity_step is not None:
        started = perf_counter()
        jacobian = sensitivity(
            parameters,
            truth_x,
            [trial for trial in synthetic if trial.split == "train"],
            cfg,
            noise,
            step=sensitivity_step,
            chunk=sensitivity_chunk,
            device=device,
        )
        report["identifiability"] = identifiability(jacobian, names, criteria) | {"step": sensitivity_step}
        timings["sensitivity_s"] = perf_counter() - started
    return truth, learned, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="Source of chains, shoes, initial states, clocks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mount", type=float, nargs=3)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--speed", type=float, help="Known task speed [m/s]; members may override")
    parser.add_argument("--height-offset", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1.25e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--method", choices=("lm", "cem"), default="lm")
    parser.add_argument("--iterations", type=int, default=LMConfig.iterations, help="LM iterations")
    parser.add_argument("--central", action="store_true", help="LM central-difference Jacobian")
    parser.add_argument("--chunk", type=int, default=LMConfig.chunk, help="LM candidates per batched rollout")
    parser.add_argument("--population", type=int, default=64, help="CEM population")
    parser.add_argument("--generations", type=int, default=40, help="CEM generations")
    parser.add_argument("--seed", type=int, default=0, help="Truth and noise seed; search uses --search-seed")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--truth-scale", type=float, default=0.3, help="Truth offset standard deviation")
    parser.add_argument("--truth-attempts", type=int, default=16)
    parser.add_argument("--noise-hip", type=float, default=Noise.hip_m)
    parser.add_argument("--noise-angle", type=float, default=Noise.angle_rad)
    parser.add_argument("--noise-force", type=float, default=Noise.force_n)
    parser.add_argument("--sensitivity-step", type=float, default=0.01)
    parser.add_argument("--sensitivity-chunk", type=int, default=64)
    parser.add_argument("--no-sensitivity", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a synthetic parameter-recovery experiment and write its report."""
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    trials = load_trials(
        args.dataset,
        mount_m=args.mount,
        pitch_rad=args.pitch,
        speed_m_s=args.speed,
        height_offset_m=args.height_offset,
        limit_per_split=args.limit_per_split,
    )
    if args.method == "lm":
        search = LMConfig(iterations=args.iterations, central=args.central, chunk=args.chunk)
    else:
        search = FitConfig(population=args.population, generations=args.generations, seed=args.search_seed)
    truth, learned, report = recover(
        trials,
        config=RolloutConfig(dt_s=args.dt),
        search=search,
        noise=Noise(args.noise_hip, args.noise_angle, args.noise_force),
        truth_scale=args.truth_scale,
        truth_attempts=args.truth_attempts,
        seed=args.seed,
        device=args.device,
        sensitivity_step=None if args.no_sensitivity else args.sensitivity_step,
        sensitivity_chunk=args.sensitivity_chunk,
    )
    report["command"] = vars(args) | {
        key: str(value.resolve()) for key, value in vars(args).items() if isinstance(value, Path)
    }
    args.output.mkdir(parents=True)
    truth.save(args.output / "truth.json")
    learned.save(args.output / "learned.json")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    ident = report.get("identifiability")
    extra = "" if ident is None else f"; local rank {ident['rank_relative_1e-3']}/{ident['parameters']} at 1e-3"
    print(f"recovery: {report['verdict']}{extra}; wrote {args.output}")


if __name__ == "__main__":
    main()
