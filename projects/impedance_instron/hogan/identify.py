# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Identify shared generative runner dynamics from offline measurements.

Measurement arrays belong to this module, never to the runner. CUDA executes
candidate dynamics and objective reductions; CPU remains the reference backend.
Run ``python -m projects.impedance_instron.hogan.identify --help`` for commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..cartesian import data as reference_data
from ..cartesian import profile as profile_data
from ..cartesian.shoe import Shoe
from .mechanics import Chain, RestOfBody, chain_from_profile, from_leg_coordinates
from .runner import RolloutConfig, Runner, State, Task, simulate


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinates(reference: dict, chain: Chain | None = None, *, height_offset_m: float = 0.0) -> np.ndarray:
    """Convert measured coordinates without GRF integration or filtering.

    With ``chain`` and a measured ``ankle_target_m``, hip and knee angles are
    re-solved per frame so fixed-length FK reaches the measured hip and ankle
    centers while the foot's absolute angle is kept. This uses positions only,
    so it is causal and independent of measured force. The exported knee angle
    is otherwise inconsistent with the exported joint centers.

    The upstream prepared motion may already be filtered. Missing pelvis
    orientation is rejected rather than treating upright posture as measured.
    """
    if "pelvis_target_rad" not in reference:
        raise ValueError("Generative identification requires measured pelvis_target_rad")
    if not np.isfinite(height_offset_m):
        raise ValueError("height_offset_m must be finite")
    q, _ = from_leg_coordinates(
        reference["state"], np.zeros_like(reference["state"]), reference["pelvis_target_rad"], 0.0
    )
    q[:, 1] += height_offset_m
    if chain is not None and "ankle_target_m" in reference:
        q = chain.reach(q, reference["ankle_target_m"] + np.array([0.0, height_offset_m]))
    return q


def initialize(time_s, q, *, com_velocity_m_s=None, chain: Chain | None = None) -> State:
    """Initialize at the END of a supplied three-frame observation prefix.

    Quadratic backward differentiation uses only these three observed positions,
    not exported velocities, measured forces, a plan, or future stance events.
    Phase uses current hip angle/rate as a fixed causal engineering convention.
    Torque and load memory start at zero, assuming a relaxed shoe in flight.

    Args:
        com_velocity_m_s: Flight whole-body COM velocity [m/s] estimated from data
            up to the prefix end. When given, the hip velocity is chosen so the
            ``chain`` COM moves at this velocity with the prefix joint rates; the
            single modeled leg otherwise carries unbalanced swing momentum.
        chain: Model chain; required with ``com_velocity_m_s``.
    """
    time = np.asarray(time_s, dtype=float)
    q = np.asarray(q, dtype=float)
    if time.shape != (3,) or q.shape != (3, 6) or not np.isfinite(time).all() or not np.isfinite(q).all():
        raise ValueError("Initialization needs three finite times and three six-coordinate positions")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Initialization times must increase strictly")
    t = time - time[-1]
    coefficients = np.linalg.solve(np.column_stack((np.ones(3), t, t * t)), q)
    velocity = coefficients[1]
    if com_velocity_m_s is not None:
        com_velocity = np.asarray(com_velocity_m_s, dtype=float)
        if com_velocity.shape != (2,) or not np.isfinite(com_velocity).all():
            raise ValueError("com_velocity_m_s must be two finite values")
        if chain is None:
            raise ValueError("com_velocity_m_s needs the model chain")
        velocity[:2] = com_velocity - chain.com_jacobian(q[-1])[:, 2:] @ velocity[2:]
    phase = math.atan2(-velocity[3] / 8.0, q[-1, 3]) % (2 * math.pi)
    return State(q[-1], velocity, phase_rad=phase)


@dataclass
class Trial:
    """Offline trial boundary separating predictive inputs from fitting targets."""

    id: str
    split: str
    chain: Chain
    shoe: Shoe
    task: Task
    initial: State
    time_s: np.ndarray
    """Observation times relative to the end of the initialization prefix [s]."""
    q: np.ndarray
    """Observed coordinates [m, m, rad, rad, rad, rad], shape [frames, 6]."""
    force_time_s: np.ndarray
    """Native force target clock [s], covering the prediction horizon."""
    grf_n: np.ndarray
    """Measured horizontal/vertical GRF [N], shape [force_frames, 2]."""
    provenance: dict

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, dtype=float).copy()
        self.q = np.asarray(self.q, dtype=float).copy()
        self.force_time_s = np.asarray(self.force_time_s, dtype=float).copy()
        self.grf_n = np.asarray(self.grf_n, dtype=float).copy()
        if self.split not in ("train", "eval"):
            raise ValueError("Trial split must be train or eval")
        if self.time_s.ndim != 1 or len(self.time_s) < 2 or self.time_s[0] != 0:
            raise ValueError("Trial times must start at zero and contain at least two samples")
        if self.q.shape != (len(self.time_s), 6):
            raise ValueError("Trial coordinate shape must match its clock")
        if self.force_time_s.ndim != 1 or len(self.force_time_s) < 2:
            raise ValueError("Trial force clock must contain at least two samples")
        if self.grf_n.shape != (len(self.force_time_s), 2):
            raise ValueError("Trial GRF shape must match its clock")
        if not all(np.isfinite(a).all() for a in (self.time_s, self.q, self.force_time_s, self.grf_n)):
            raise ValueError("Trial observations must be finite")
        if np.any(np.diff(self.time_s) <= 0) or np.any(np.diff(self.force_time_s) <= 0):
            raise ValueError("Trial clocks must increase strictly")
        if self.force_time_s[0] > 0 or self.force_time_s[-1] < self.time_s[-1]:
            raise ValueError("Trial forces must cover the prediction horizon")

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1])


def compatibility(reference: dict, q: np.ndarray, chain: Chain, shoe: Shoe) -> dict:
    """Check observed ankle/shoe/load compatibility; never alter rollout inputs.

    Only fixed-length FK disagreement with the measured ankle (10 mm) fails the
    screen. Loaded clearance and COP footprint conflicts are reported but not
    gated: marker-based ankle and shoe registration are only good to several
    millimeters, and treadmill COP is unreliable at low force.
    """
    times = reference["time_s"]
    forces = np.column_stack(
        [np.interp(times, reference["grf_time_s"], reference["grf_target_n"][:, axis]) for axis in range(2)]
    )
    loaded = forces[:, 1] > 50.0
    gaps, model_gaps, ankle_errors, cop_excess = [], [], [], []
    cop = None
    if "cop_target_m" in reference:
        cop = np.interp(times, reference["grf_time_s"], reference["cop_target_m"])
    for i, row in enumerate(q):
        ankle = chain.point(row, 3, np.zeros(2))[0]
        # Diagnose contact against measured ankle centers even if fixed-length FK disagrees.
        if "ankle_target_m" in reference:
            measured = reference["ankle_target_m"][i].copy()
            measured[1] += row[1] - reference["state"][i, 1]
            ankle_errors.append(float(np.linalg.norm(ankle - measured)))
        else:
            measured = ankle
        outline = shoe.outline(measured, chain.angle(row, 3))
        gaps.append(float(outline[:, 1].min()))
        model_gaps.append(float(shoe.outline(ankle, chain.angle(row, 3))[:, 1].min()))
        if cop is not None:
            cop_excess.append(float(max(outline[:, 0].min() - cop[i], cop[i] - outline[:, 0].max(), 0.0)))
    gaps = np.asarray(gaps)
    conflict = loaded & (gaps > 0.002)
    model_conflict = loaded & (np.asarray(model_gaps) > 0.002)
    errors = np.asarray(ankle_errors)
    cop_bad = np.asarray(cop_excess) > 0.002 if cop is not None else np.zeros(len(q), dtype=bool)
    return {
        "passed": not np.any(errors > 0.01),
        "clearance_gating": False,
        "model_loaded_clearance_conflict_frames": int(model_conflict.sum()),
        "model_loaded_clearance_peak_m": float(np.max(np.asarray(model_gaps)[loaded])) if loaded.any() else None,
        "clearance_basis": "measured ankle and independently checked model FK; all nominal bottom points",
        "loaded_clearance_conflict_frames": int(conflict.sum()),
        "loaded_clearance_peak_m": float(np.max(gaps[loaded])) if loaded.any() else None,
        "force_during_clearance_peak_n": float(forces[conflict, 1].max()) if conflict.any() else 0.0,
        "ankle_fk_error_peak_m": float(errors.max()) if len(errors) else None,
        "loaded_cop_outside_frames": int(np.count_nonzero(loaded & cop_bad)),
        "cop_checked": cop is not None,
        "cop_gating": False,
        "clearance_tolerance_m": 0.002,
        "ankle_fk_tolerance_m": 0.01,
    }


def load_trials(
    dataset: str | Path,
    *,
    mount_m=None,
    pitch_rad: float | None = None,
    speed_m_s: float | None = None,
    height_offset_m: float = 0.0,
    friction_model: str = "elastic_coulomb",
    limit_per_split: int | None = None,
) -> list[Trial]:
    """Load existing or multi-condition datasets without cached tracking plans.

    A member may specify ``profile``, ``shoe_artifact``, ``mount_m``, ``pitch_rad``,
    ``speed_m_s`` and ``height_offset_m``. Paths are dataset-relative; profile/shoe
    may instead come from existing ``shared_assets``. All trials describe one
    runner; optional ``subject_id`` values must agree. Speed is explicit task
    metadata, never inferred from future movement. Existing source files are
    fingerprinted in the output; declared shared-asset hashes are checked.
    """
    root = Path(dataset).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in ("peak_hip_stance_dataset_1", "generative_runner_dataset_1"):
        raise ValueError("Unsupported dataset schema")
    members = manifest["members"]
    ids = [member["id"] for member in members]
    if len(ids) != len(set(ids)) or any(member["split"] not in ("train", "eval") for member in members):
        raise ValueError("Members need unique IDs and train/eval splits")
    subjects = {member.get("subject_id", manifest.get("subject_id", "unspecified")) for member in members}
    if len(subjects) != 1:
        raise ValueError("A shared runner fit must contain one subject")
    if limit_per_split is not None and limit_per_split < 1:
        raise ValueError("limit_per_split must be positive")
    assets = manifest.get("shared_assets", {})
    for entry in assets.values():
        if "sha256" in entry and _hash(root / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Shared asset hash mismatch: {entry['file']}")
    trials, shoes, counts = [], {}, {"train": 0, "eval": 0}
    for member in members:
        split = member["split"]
        if limit_per_split is not None and counts[split] >= limit_per_split:
            continue
        reference_path = root / member["reference"]
        reference = reference_data.load(reference_path)
        profile_path = root / (member.get("profile") or assets.get("profile.json", {}).get("file", ""))
        shoe_path = root / (member.get("shoe_artifact") or assets.get("digital_shoe.json", {}).get("file", ""))
        if not profile_path.is_file() or not shoe_path.is_file():
            raise ValueError("Each member needs a profile and shoe artifact, directly or via shared_assets")
        mount = member.get("mount_m", mount_m)
        if mount is None:
            raise ValueError("Supply the fixed ankle/shoe mount; it is not learned from force")
        pitch = member.get("pitch_rad", pitch_rad)
        if pitch is None:
            pitch = float(reference.get("shoe_static_pitch_rad", reference["static_pitch_rad"]))
        speed = member.get("speed_m_s", speed_m_s)
        if speed is None:
            raise ValueError("Supply speed_m_s task metadata per member or --speed")
        offset = float(member.get("height_offset_m", height_offset_m))
        rest = RestOfBody(**manifest.get("rest_of_body", {}))
        chain = chain_from_profile(reference, profile_data.load(profile_path), rest)
        key = (str(shoe_path), tuple(mount), float(pitch), friction_model)
        if key not in shoes:
            shoes[key] = Shoe(shoe_path, mount, pitch, friction_model=friction_model)
        shoe = shoes[key]
        q = coordinates(reference, chain, height_offset_m=offset)
        if len(q) < 4:
            raise ValueError("A trial needs a three-frame prefix and at least one future observation")
        flight_velocity = member.get("flight_velocity_m_s")
        if flight_velocity is not None and member.get("flight_velocity_frame") != 2:
            raise ValueError("flight_velocity_m_s must be given at the three-frame prefix end")
        initial = initialize(reference["time_s"][:3], q[:3], com_velocity_m_s=flight_velocity, chain=chain)
        origin = float(reference["time_s"][2])
        ankle = chain.point(initial.q, 3, np.zeros(2))[0]
        clearance = float(shoe.outline(ankle, chain.angle(initial.q, 3))[:, 1].min())
        qc = compatibility(reference, q, chain, shoe)
        # The relaxed-shoe initialization assumes flight at the prediction origin.
        qc["initial_clearance_m"] = clearance
        qc["passed"] = qc["passed"] and clearance > 0
        trials.append(
            Trial(
                member["id"],
                split,
                chain,
                shoe,
                Task(float(speed)),
                initial,
                reference["time_s"][2:] - origin,
                q[2:].copy(),
                reference["grf_time_s"] - origin,
                reference["grf_target_n"].copy(),
                {
                    "reference": str(reference_path),
                    "reference_sha256": _hash(reference_path),
                    "profile": str(profile_path),
                    "profile_sha256": _hash(profile_path),
                    "shoe_artifact": str(shoe_path),
                    "shoe_sha256": _hash(shoe_path),
                    "manifest_sha256": _hash(manifest_path),
                    "subject_id": next(iter(subjects)),
                    "mount_m": list(mount),
                    "pitch_rad": pitch,
                    "height_offset_m": offset,
                    "speed_m_s": speed,
                    "initialization_prefix_frames": 3,
                    "prediction_origin_s": origin,
                    "initialization": (
                        "backward quadratic positions; model COM velocity from preceding-stride plate force"
                        if flight_velocity is not None
                        else "backward quadratic positions only"
                    )
                    + "; fixed kinematic phase; zero torque/load memory",
                    "target_processing": (
                        "hip/knee re-solved to measured hip and ankle centers when available, foot angle kept; "
                        "no Hogan filter or GRF COM correction"
                    ),
                    "rest_of_body": asdict(rest),
                    "friction_model": friction_model,
                    "compatibility": qc,
                },
            )
        )
        counts[split] += 1
    return trials


def predict(runner: Runner, trial: Trial, config: RolloutConfig, *, device: str = "cpu") -> tuple[dict, dict]:
    """Cross the rollout boundary without passing any observation arrays."""
    if device != "cpu":
        return predict_many([runner], [trial], config, device=device)[0][0]
    return simulate(
        runner, trial.chain, trial.shoe, trial.initial, trial.task, duration_s=trial.duration_s, config=config
    )


def predict_many(models: list[Runner], trials: list[Trial], config: RolloutConfig, *, device: str) -> list:
    """Predict concurrent candidates, streaming trial groups to bound trace memory."""
    if device == "cpu":
        return [[predict(model, trial, config) for trial in trials] for model in models]
    from .gpu_runner import GpuBatch  # noqa: PLC0415 - keep CPU inspection independent of CUDA modules

    result = [[] for _ in models]
    for start in range(0, len(trials), 4):
        group = trials[start : start + 4]
        batch = GpuBatch(
            [t.chain for t in group],
            [t.shoe for t in group],
            [t.initial for t in group],
            [t.task for t in group],
            [t.duration_s for t in group],
            candidates=len(models),
            config=config,
            device=device,
        )
        values = batch.evaluate(models)
        for candidate, row in enumerate(values):
            result[candidate].extend(row)
        del batch
    return result


def scenario(trial: Trial, config: RolloutConfig) -> dict:
    """Export a predictive scenario without measurements or fitting metadata."""
    return {
        "schema": "generative_runner_scenario_1",
        "chain": {
            name: getattr(trial.chain, name).tolist()
            for name in ("lengths_m", "endpoint_local_m", "masses_kg", "com_local_m", "inertias_kg_m2")
        },
        "shoe": {
            "artifact": trial.provenance["shoe_artifact"],
            "sha256": trial.provenance["shoe_sha256"],
            "mount_m": trial.provenance["mount_m"],
            "pitch_rad": trial.provenance["pitch_rad"],
            "friction_model": trial.provenance["friction_model"],
        },
        "initial": {
            "q": trial.initial.q.tolist(),
            "v": trial.initial.v.tolist(),
            "phase_rad": trial.initial.phase_rad,
            "normal_load_bw": trial.initial.normal_load_bw,
            "torque_nm": trial.initial.torque_nm.tolist(),
        },
        "task": asdict(trial.task),
        "duration_s": trial.duration_s,
        "config": asdict(config),
    }


def score(trace: dict, summary: dict, trial: Trial, runner: Runner) -> dict:
    """Compare a completed prediction to native-clock targets, entirely offline.

    Failed predictions receive an explicit unfinished-window penalty. Selection
    additionally ranks failure count before numeric loss, so early failure cannot
    win by avoiding difficult late samples. Physical metrics remain diagnostics,
    not physiological acceptance claims.
    """
    time = trace["time_s"]
    observed = (trial.time_s > 0) & (trial.time_s <= time[-1] + 1e-12)
    tracking = np.zeros(6)
    if observed.any():
        simulated = np.column_stack([np.interp(trial.time_s[observed], time, trace["state"][:, c]) for c in range(6)])
        tracking = np.sqrt(np.mean((simulated - trial.q[observed]) ** 2, axis=0))
    force = np.zeros(2)
    if len(trace["grf_n"]):
        measured = np.column_stack([np.interp(time[:-1], trial.force_time_s, trial.grf_n[:, c]) for c in range(2)])
        force = np.sqrt(np.mean((trace["grf_n"] - measured) ** 2, axis=0))
    # Integrate the target on a horizon-clipped native clock, retaining endpoints.
    interior = (trial.force_time_s > 0) & (trial.force_time_s < trial.duration_s)
    clock = np.concatenate(([0.0], trial.force_time_s[interior], [trial.duration_s]))
    measured = np.column_stack([np.interp(clock, trial.force_time_s, trial.grf_n[:, c]) for c in range(2)])
    impulse = np.sum(0.5 * (measured[1:] + measured[:-1]) * np.diff(clock)[:, None], axis=0)
    peak_error = float(summary["peak_grf_n"][1] - measured[:, 1].max())
    impulse_error = np.asarray(summary["grf_impulse_ns"]) - impulse
    contact = measured[:, 1] > summary["contact_threshold_n"]
    contact_time = float(np.sum(np.diff(clock) * contact[:-1]))
    contact_error = summary["contact_duration_s"] - contact_time
    effort = 0.0
    if len(trace["load"]):
        effort = float(np.mean((trace["load"][:, 3:] / runner.bounds.torque_max_nm) ** 2))
    loss = (
        float(np.mean((tracking[:2] / 0.02) ** 2))
        + float(np.mean((tracking[2:] / 0.05) ** 2))
        + float(np.mean((force / 100.0) ** 2))
        + (peak_error / 100.0) ** 2
        + float(np.mean((impulse_error / 20.0) ** 2))
        + (contact_error / 0.02) ** 2
        + 0.01 * effort
    )
    if summary["status"] != "completed":
        loss += 1000.0 * (2.0 - time[-1] / trial.duration_s)
    return {
        "loss": loss,
        "tracking_rmse": tracking.tolist(),
        "grf_rmse_n": force.tolist(),
        "peak_fz_error_n": peak_error,
        "impulse_error_ns": impulse_error.tolist(),
        "contact_duration_error_s": contact_error,
        "id": trial.id,
        "split": trial.split,
        **summary,
    }


def evaluate(runner: Runner, trials: list[Trial], config: RolloutConfig, *, device: str = "cpu") -> dict:
    """Evaluate independent free predictions with a frozen shared model."""
    if not trials:
        raise ValueError("Evaluation needs at least one trial")
    rows = []
    for trial, (trace, summary) in zip(trials, predict_many([runner], trials, config, device=device)[0], strict=True):
        rows.append(score(trace, summary, trial, runner))
    return {
        "mean_loss": float(np.mean([row["loss"] for row in rows])),
        "failed": sum(row["status"] != "completed" for row in rows),
        "trials": rows,
    }


@dataclass(frozen=True)
class FitConfig:
    """Cross-entropy search settings; no held-out selection."""

    population: int = 8
    generations: int = 10
    sigma: float = 0.2
    bound: float = 1.5
    regularization: float = 0.01
    seed: int = 0

    def __post_init__(self):
        for name in ("population", "generations", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.population < 4 or self.generations < 1:
            raise ValueError("population must be at least four and generations at least one")
        if not np.isfinite([self.sigma, self.bound, self.regularization]).all():
            raise ValueError("Search settings must be finite")
        if min(self.sigma, self.bound) <= 0 or self.regularization < 0:
            raise ValueError("Search sigma/bound must be positive and regularization nonnegative")


class Parameterization:
    """Fit shared impedance weights, stride frequency, and torque response time.

    Task-speed weights and cadence-speed gain are frozen unless training data
    contain multiple distinct speeds. There are no stance-specific coefficients.
    """

    def __init__(self, baseline: Runner, speeds):
        speeds = np.asarray(speeds, dtype=float)
        if speeds.ndim != 1 or speeds.size == 0 or not np.isfinite(speeds).all() or np.any(speeds < 0):
            raise ValueError("Training speeds must be finite and nonempty")
        self.baseline = baseline
        if not 0.5 <= baseline.frequency_hz <= 4.0 or not 0.005 <= baseline.response_time_s <= 0.15:
            raise ValueError("Identification requires frequency in [0.5, 4] Hz and response time in [0.005, 0.15] s")
        if not -1 <= baseline.cadence_speed_gain <= 1:
            raise ValueError("Identification requires cadence_speed_gain in [-1, 1]")
        self.variable_speed = bool(np.ptp(speeds) > 1e-6)
        mask = np.ones(baseline.weights.shape, dtype=bool)
        if not self.variable_speed:
            mask[:, :, -1] = False
        self.indices = np.flatnonzero(mask.ravel())
        self.size = len(self.indices) + 2 + int(self.variable_speed)

    def model(self, offsets) -> Runner:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (self.size,) or not np.isfinite(offsets).all():
            raise ValueError("Parameter offsets have incorrect shape or nonfinite values")
        n = len(self.indices)
        data = self.baseline.to_dict()
        weights = self.baseline.weights.copy().ravel()
        weights[self.indices] += offsets[:n]
        data["weights"] = weights.reshape(self.baseline.weights.shape).tolist()
        data["frequency_hz"] = float(
            np.clip(self.baseline.frequency_hz * np.exp(np.clip(offsets[n], -20, 20)), 0.5, 4.0)
        )
        data["response_time_s"] = float(
            np.clip(self.baseline.response_time_s * np.exp(np.clip(offsets[n + 1], -20, 20)), 0.005, 0.15)
        )
        if self.variable_speed:
            data["cadence_speed_gain"] = float(np.clip(self.baseline.cadence_speed_gain + offsets[-1], -1.0, 1.0))
        return Runner.from_dict(data)


def fit(
    baseline: Runner,
    trials: list[Trial],
    *,
    config: RolloutConfig | None = None,
    search: FitConfig | None = None,
    allow_incompatible: bool = False,
    device: str = "cpu",
) -> tuple[Runner, dict]:
    """Fit on training trials only; evaluate held-out trials after selection.

    By default incompatible references stop identification. An explicit override
    permits implementation experiments, but never marks the result validated.
    """
    cfg, search = config or RolloutConfig(), search or FitConfig()
    train = [trial for trial in trials if trial.split == "train"]
    held_out = [trial for trial in trials if trial.split == "eval"]
    if not train:
        raise ValueError("Identification requires training trials")
    incompatible = [trial.id for trial in trials if not trial.provenance["compatibility"]["passed"]]
    if incompatible and not allow_incompatible:
        raise ValueError(f"Input compatibility failed for {len(incompatible)} trials; run inspect before fitting")
    parameters = Parameterization(baseline, [trial.task.speed_m_s for trial in train])
    evaluator = None
    if device != "cpu":
        from .gpu_objective import GpuEvaluator  # noqa: PLC0415 - optional execution backend

        evaluator = GpuEvaluator(train, candidates=search.population, config=cfg, device=device)

    def evaluations(models):
        if evaluator is None:
            return [evaluate(model, train, cfg) for model in models]
        count = len(models)
        return evaluator.evaluate(models + [models[-1]] * (search.population - count))[:count]

    rng = np.random.default_rng(search.seed)
    mean, sigma = np.zeros(parameters.size), np.full(parameters.size, search.sigma)
    best = mean.copy()
    initial_evaluation = evaluations([baseline])[0]
    best_rank = (initial_evaluation["failed"], initial_evaluation["mean_loss"])
    history = []
    for generation in range(search.generations):
        samples = np.vstack(
            (
                mean,
                best,
                np.clip(
                    mean + sigma * rng.standard_normal((search.population - 2, parameters.size)),
                    -search.bound,
                    search.bound,
                ),
            )
        )
        ranks = []
        for offset, result in zip(samples, evaluations([parameters.model(x) for x in samples]), strict=True):
            ranks.append((result["failed"], result["mean_loss"] + search.regularization * float(np.mean(offset**2))))
        order = sorted(range(len(samples)), key=lambda i: ranks[i])
        if ranks[order[0]] < best_rank:
            best, best_rank = samples[order[0]].copy(), ranks[order[0]]
        elites = samples[order[: max(2, search.population // 4)]]
        mean = 0.3 * mean + 0.7 * elites.mean(0)
        sigma = np.maximum(0.3 * sigma + 0.7 * elites.std(0), 0.03)
        history.append(
            {
                "generation": generation,
                "best_failed": best_rank[0],
                "best_score": best_rank[1],
                "sigma_mean": float(sigma.mean()),
            }
        )
    # The final mean is a candidate too, but only training observations choose it.
    mean_result = evaluations([parameters.model(mean)])[0]
    mean_rank = (mean_result["failed"], mean_result["mean_loss"] + search.regularization * float(np.mean(mean**2)))
    if mean_rank < best_rank:
        best = mean
    learned = parameters.model(best)
    del evaluator
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
        "schema": "generative_runner_identification_1",
        "validated": False,
        "reference_inputs_used": False,
        "device": device,
        "objective_backend": "numpy" if device == "cpu" else "cuda",
        "selection_split": "train",
        "initial_model": baseline.to_dict(),
        "search": asdict(search),
        "rollout": asdict(cfg),
        "history": history,
        "parameters": parameters.size,
        "task_speed_parameters_learned": parameters.variable_speed,
        "incompatible_trials": incompatible,
        "allow_incompatible": allow_incompatible,
        "initialization": "three-frame observed position prefix; prediction starts at prefix end",
        "shoe_adaptation": "shared law responds to simulated load; no shoe-ID-specific adaptation fitted",
        "scope": "single-leg independent windows; effective actuation, not identified physiology or sustained running",
        "loss_scales": {
            "hip_m": 0.02,
            "angle_rad": 0.05,
            "force_n": 100.0,
            "impulse_ns": 20.0,
            "contact_s": 0.02,
            "effort_weight": 0.01,
        },
        "splits": results,
        "trials": [{"id": trial.id, "split": trial.split, **trial.provenance} for trial in trials],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("inspect", "fit", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Frozen model for evaluate, or initialization for fit")
    parser.add_argument("--mount", type=float, nargs=3)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--speed", type=float, help="Known task speed [m/s]; members may override")
    parser.add_argument("--height-offset", type=float, default=0.0)
    parser.add_argument(
        "--friction-model",
        choices=("elastic_coulomb", "column_maxwell", "maxwell", "legacy"),
        default="elastic_coulomb",
    )
    parser.add_argument("--dt", type=float, default=1.25e-4)
    parser.add_argument(
        "--compression-limit",
        type=float,
        default=RolloutConfig.compression_limit,
        help="Driven-shoe compression fraction that ends a rollout as failed",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA backend by default; cpu selects the reference")
    parser.add_argument("--limit-per-split", type=int, help="Explicit small subset for implementation checks")
    parser.add_argument("--method", choices=("cem", "lm"), default="lm", help="Fit optimizer")
    parser.add_argument("--population", type=int, default=8, help="CEM population")
    parser.add_argument("--generations", type=int, default=10, help="CEM generations")
    parser.add_argument("--seed", type=int, default=0, help="CEM seed")
    parser.add_argument("--iterations", type=int, default=15, help="LM iterations")
    parser.add_argument("--chunk", type=int, default=16, help="LM candidates per batched GPU rollout")
    parser.add_argument("--central", action="store_true", help="LM central-difference Jacobian")
    parser.add_argument(
        "--intrinsic-damping",
        type=float,
        nargs=3,
        help="Fixed lag-free hip, knee, ankle damping [N m s/rad] set on the initial model for fit",
    )
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Permit diagnostic fitting of incompatible inputs; never a validation claim",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Inspect, identify, or evaluate generative models without changing the tracker."""
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.command == "evaluate" and args.model is None:
        raise ValueError("evaluate requires --model")
    trials = load_trials(
        args.dataset,
        mount_m=args.mount,
        pitch_rad=args.pitch,
        speed_m_s=args.speed,
        height_offset_m=args.height_offset,
        friction_model=args.friction_model,
        limit_per_split=args.limit_per_split,
    )
    if not trials:
        raise ValueError("Dataset is empty")
    cfg = RolloutConfig(dt_s=args.dt, compression_limit=args.compression_limit)
    report = {
        "validated": False,
        "trials": [{"id": trial.id, "split": trial.split, **trial.provenance} for trial in trials],
    }
    if args.command == "fit":
        training_speeds = [trial.task.speed_m_s for trial in trials if trial.split == "train"]
        if not training_speeds:
            raise ValueError("Identification requires training trials")
        baseline = (
            Runner.load(args.model) if args.model else Runner.seed(reference_speed_m_s=float(np.mean(training_speeds)))
        )
        if args.intrinsic_damping is not None:
            baseline = Runner.from_dict({**baseline.to_dict(), "intrinsic_damping_nms_rad": args.intrinsic_damping})
        if args.method == "lm":
            from .least_squares import LMConfig, fit_lm  # noqa: PLC0415 - least_squares imports this module

            optimizer = fit_lm
            search = LMConfig(iterations=args.iterations, chunk=args.chunk, central=args.central)
        else:
            optimizer = fit
            search = FitConfig(population=args.population, generations=args.generations, seed=args.seed)
        model, report = optimizer(
            baseline,
            trials,
            config=cfg,
            search=search,
            allow_incompatible=args.allow_incompatible,
            device=args.device,
        )
    elif args.command == "evaluate":
        model = Runner.load(args.model)
        report.update(
            rollout=asdict(cfg),
            splits={
                name: evaluate(model, subset, cfg, device=args.device)
                for name in ("train", "eval")
                if (subset := [trial for trial in trials if trial.split == name])
            },
        )
    report["command"] = vars(args) | {
        key: str(value.resolve()) for key, value in vars(args).items() if isinstance(value, Path)
    }
    project_root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        Path(__file__).with_name("runner.py"),
        Path(__file__).with_name("generate.py"),
        Path(__file__).with_name("mechanics.py"),
        Path(__file__).with_name("gpu_runner.py"),
        Path(__file__).with_name("gpu_objective.py"),
        Path(__file__).with_name("least_squares.py"),
        project_root / "cartesian/shoe.py",
        project_root / "cartesian/data.py",
        project_root / "cartesian/profile.py",
    ]
    sources.extend((project_root.parent / "digital_shoe").glob("*.py"))
    report["source_sha256"] = {str(path.relative_to(project_root.parent)): _hash(path) for path in sources}
    if args.model is not None:
        report["input_model_sha256"] = _hash(args.model)
    args.output.mkdir(parents=True)
    if args.command != "inspect":
        model.save(args.output / "runner.json")
        # Save every prediction without letting visualization re-run a different model.
        predictions = predict_many([model], trials, cfg, device=args.device)[0]
        for index, (trial, (trace, _)) in enumerate(zip(trials, predictions, strict=True)):
            np.savez_compressed(args.output / f"trace_{index:03d}.npz", **trace)
            report["trials"][index]["trace"] = f"trace_{index:03d}.npz"
            name = f"scenario_{index:03d}.json"
            (args.output / name).write_text(
                json.dumps(scenario(trial, cfg), indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            report["trials"][index]["scenario"] = name
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    bad = sum(not trial.provenance["compatibility"]["passed"] for trial in trials)
    print(f"{args.command}: {len(trials)} trials; {bad} incompatible; not validated; wrote {args.output}")


if __name__ == "__main__":
    main()
