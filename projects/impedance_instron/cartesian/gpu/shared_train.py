# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit one stance-conditioned shared residual controller with antithetic CEM search.

This is an episodic black-box optimizer over one common 12-by-6 residual. Each
stance supplies a deterministic six-channel PD-compensated nominal spline. It
is not PPO or an independently fitted controller per stance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..data import load as load_reference
from ..fit import FitConfig
from ..profile import load as load_profile
from ..run import Config
from ..trajectory import Spline
from .engine import Engine
from .shared_controller import nominal_six_channel, project_shared_residual

_DEFAULT_DATASET = Path("/home/jkuzm/projects/newton/outputs/impedance_instron/stance_dataset_peak_hip")
_DEFAULT_FIT = Path("/home/jkuzm/projects/newton/outputs/impedance_instron/fr3_2_ankle_xy_refit_20260929/fit_run2")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fit(directory: Path):
    """Load explicit six-channel physics and the saved controller bundle."""
    metadata = _json(directory / "optimization_inputs.json")
    summary = _json(directory / "summary.json")
    profile = load_profile(directory / "profile.json")
    if len(profile["equilibrium_lower"]) != 6:
        raise ValueError("Saved fit profile must have six equilibrium channels")
    reference = load_reference(directory / "reference.npz")
    with np.load(directory / "equilibrium.npz", allow_pickle=False) as archive:
        coefficients = archive["coefficients"].copy()
        duration = float(archive["duration_s"])
    if coefficients.shape != (12, 6) or not np.isclose(duration, reference["time_s"][-1], rtol=0, atol=1e-12):
        raise ValueError("Saved fit must contain a matching twelve-control six-channel spline")
    shoe = metadata["shoe"]
    shoe_path = Path(shoe["path"])
    if not shoe_path.is_file() or hashlib.sha256(shoe_path.read_bytes()).hexdigest() != shoe["sha256"]:
        raise ValueError("Saved shoe asset is missing or differs from the frozen fit metadata")
    config = Config(**metadata["simulation_config"])
    settings = FitConfig(**metadata["fit_config"])
    return reference, profile, coefficients, config, settings, shoe, summary


def _load_residual_checkpoint(path: Path) -> np.ndarray:
    """Load and validate one finite shared residual checkpoint [12, 6]."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if "residual" not in archive.files:
            raise ValueError("Initial controller checkpoint must contain a residual array")
        residual = np.asarray(archive["residual"], dtype=np.float64).copy()
    if residual.shape != (12, 6):
        raise ValueError("Initial controller residual must have shape (12, 6)")
    if not np.isfinite(residual).all():
        raise ValueError("Initial controller residual must contain only finite values")
    return residual


def _load_dataset(root: Path, train_limit: int | None, score_limit: int | None):
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    members = manifest["members"]
    if len({item["id"] for item in members}) != len(members):
        raise ValueError("Stance IDs must be unique across training and evaluation")
    if len({item["reference"] for item in members}) != len(members):
        raise ValueError("A reference cannot occur in both training and evaluation")
    if manifest.get("policy", {}).get("selected_contact_side") != "right":
        raise ValueError("This experiment requires a right-foot-contact dataset")
    for item in members:
        if item["trial"] not in ("FR3_1", "FR3_2") or item["split"] not in ("train", "eval"):
            raise ValueError("Dataset must contain only FR3_1/FR3_2 training and evaluation members")
        if item.get("contact_assignment", {}).get("side") != "right":
            raise ValueError("Every stance must have an explicit right-foot contact assignment")
    train = [item for item in manifest["members"] if item["split"] == "train"]
    evaluation = [item for item in manifest["members"] if item["split"] == "eval"]
    if train_limit is not None:
        train = train[:train_limit]
    if score_limit is not None:
        evaluation = evaluation[:score_limit]
    if not train:
        raise ValueError("The selected training set is empty")
    members = []
    for item in train + evaluation:
        reference = load_reference(root / item["reference"])
        members.append({**item, "reference_data": reference, "duration_s": float(reference["time_s"][-1])})
    return manifest, train, evaluation, members


class _EnginePool:
    """Keep a small LRU of fixed-size engines for distinct stance durations."""

    def __init__(self, capacity, profile, shoe, config, settings, friction_model):
        self.capacity = capacity
        self.profile, self.shoe = profile, shoe
        self.config, self.settings = config, settings
        self.friction_model = friction_model
        self.engines = OrderedDict()
        self.setup_s = 0.0

    def get(self, item, world_count):
        key = (item["id"], world_count)
        if key in self.engines:
            self.engines.move_to_end(key)
            return self.engines[key]
        started = time.perf_counter()
        engine = Engine(
            item["reference_data"],
            self.profile,
            self.shoe["path"],
            self.shoe["mount_m"],
            self.shoe["static_pitch_rad"],
            config=self.config,
            settings=self.settings,
            world_count=world_count,
            friction_model=self.friction_model,
        )
        engine.capture(np.repeat(item["nominal"][None, :, :], world_count, axis=0))
        self.setup_s += time.perf_counter() - started
        self.engines[key] = engine
        while len(self.engines) > self.capacity:
            self.engines.popitem(last=False)
            gc.collect()
        return engine


def _balanced_batches(items, batch_size, iterations, seed):
    """Yield seeded groups that balance the two source trials when possible."""
    rng = np.random.default_rng(seed)
    by_trial = {
        trial: [item for item in items if item["trial"] == trial] for trial in sorted({x["trial"] for x in items})
    }
    orders = {trial: rng.permutation(len(group)).tolist() for trial, group in by_trial.items()}
    cursor = dict.fromkeys(by_trial, 0)
    for _ in range(iterations):
        selected = []
        selected_ids = set()
        trials = sorted(by_trial)
        rng.shuffle(trials)
        while len(selected) < min(batch_size, len(items)):
            changed = False
            for trial in trials:
                group = by_trial[trial]
                if not group:
                    continue
                if cursor[trial] == len(orders[trial]):
                    orders[trial] = rng.permutation(len(group)).tolist()
                    cursor[trial] = 0
                candidate = group[orders[trial][cursor[trial]]]
                cursor[trial] += 1
                if candidate["id"] not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(candidate["id"])
                    changed = True
                    if len(selected) >= batch_size:
                        break
            if not changed:
                break
        yield selected


def _project(residual, items, profile):
    nominals = np.stack([item["nominal"] for item in items])
    durations = np.asarray([item["duration_s"] for item in items])
    return project_shared_residual(nominals, residual, durations, profile)


def _score(coefficients, items, profile, pool, world_count, failure_penalty, settings):
    """Average equally weighted stance losses; any failed rollout invalidates the proposal."""
    per_stance, failed, metrics = [], [], []
    completed_rollouts = 0
    for item, coeff in zip(items, coefficients, strict=True):
        engine = pool.get(item, world_count)
        worlds = np.repeat(coeff[None, :, :], world_count, axis=0)
        result = engine.evaluate(worlds)
        valid = (
            (result["failure_code"] == 0) & (result["integrated_steps"] == engine.steps) & np.isfinite(result["loss"])
        )
        completed_rollouts += int(np.count_nonzero(valid))
        if not bool(np.all(valid)):
            failed.append(item["id"])
            per_stance.append(None)
            metrics.append(
                {
                    "stance_id": item["id"],
                    "valid": False,
                    "rmse": None,
                    "maximum_error": None,
                    "within_measured_tolerances": False,
                }
            )
        else:
            per_stance.append(float(np.mean(result["loss"])))
            rmse = np.mean(result["rmse"], axis=0)
            maximum_error = np.max(result["maximum_error"], axis=0)
            tolerances = np.asarray(
                [settings.hip_tolerance_m] * 2 + [settings.joint_tolerance_rad] * 2 + [settings.force_tolerance_n] * 2
            )
            metrics.append(
                {
                    "stance_id": item["id"],
                    "valid": True,
                    "rmse": rmse.tolist(),
                    "maximum_error": maximum_error.tolist(),
                    "within_measured_tolerances": bool(np.all(rmse <= tolerances)),
                }
            )
    if failed:
        return failure_penalty + float(len(failed)), {
            "valid": False,
            "failed_stances": failed,
            "completed_rollouts": completed_rollouts,
            "total_rollouts": len(items) * world_count,
            "stance_losses": per_stance,
            "stance_metrics": metrics,
            "accepted_by_measured_tolerances": False,
        }
    return float(np.mean(per_stance)), {
        "valid": True,
        "failed_stances": [],
        "completed_rollouts": completed_rollouts,
        "total_rollouts": len(items) * world_count,
        "stance_losses": per_stance,
        "stance_metrics": metrics,
        "accepted_by_measured_tolerances": all(item["within_measured_tolerances"] for item in metrics),
    }


def _score_population(coefficients_by_stance, items, pool, failure_penalty, settings):
    """Evaluate distinct candidates together and invalidate any with a failed stance."""
    population = len(coefficients_by_stance[0])
    losses = np.zeros(population, dtype=np.float64)
    failed = [[] for _ in range(population)]
    completed = np.zeros(population, dtype=np.int64)
    stance_metrics = [[] for _ in range(population)]
    for item, coefficients in zip(items, coefficients_by_stance, strict=True):
        engine = pool.get(item, population)
        result = engine.evaluate(coefficients)
        valid = (
            (result["failure_code"] == 0) & (result["integrated_steps"] == engine.steps) & np.isfinite(result["loss"])
        )
        completed += valid.astype(np.int64)
        losses += np.where(valid, result["loss"] / len(items), 0.0)
        tolerances = np.asarray(
            [settings.hip_tolerance_m] * 2 + [settings.joint_tolerance_rad] * 2 + [settings.force_tolerance_n] * 2
        )
        for index in range(population):
            if valid[index]:
                max_error = result["maximum_error"][index]
                stance_metrics[index].append(
                    {
                        "stance_id": item["id"],
                        "valid": True,
                        "rmse": result["rmse"][index].tolist(),
                        "maximum_error": max_error.tolist(),
                        "within_measured_tolerances": bool(np.all(result["rmse"][index] <= tolerances)),
                    }
                )
            else:
                stance_metrics[index].append(
                    {
                        "stance_id": item["id"],
                        "valid": False,
                        "rmse": None,
                        "maximum_error": None,
                        "within_measured_tolerances": False,
                    }
                )
        for index in np.flatnonzero(~valid):
            failed[int(index)].append(item["id"])
    details = []
    for index in range(population):
        count = len(failed[index])
        if count:
            losses[index] = failure_penalty + float(count)
        details.append(
            {
                "valid": count == 0,
                "failed_stances": failed[index],
                "completed_rollouts": int(completed[index]),
                "total_rollouts": len(items),
                "stance_metrics": stance_metrics[index],
                "accepted_by_measured_tolerances": bool(
                    count == 0 and all(m["within_measured_tolerances"] for m in stance_metrics[index])
                ),
            }
        )
    return losses, details


def run(args):
    command_started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint_path = args.output.with_name(args.output.name + ".checkpoint.npz")
    if args.checkpoint_every and checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fit_dir = args.saved_fit
    saved_ref, profile, starting_coeff, config, settings, shoe, _saved_summary = _load_fit(fit_dir)
    if args.benchmark_single:
        manifest = {"schema": "matched_single_stance_benchmark", "members": [{"split": "train"}]}
        train_records, _eval_records = manifest["members"], []
        all_items = [
            {
                "id": "saved_FR3_2_reference",
                "trial": "saved_FR3_2",
                "split": "train",
                "reference_data": saved_ref,
                "duration_s": float(saved_ref["time_s"][-1]),
            }
        ]
    else:
        manifest, train_records, _eval_records, all_items = _load_dataset(
            args.dataset, args.train_limit, args.score_limit
        )
    train_n = len(train_records)
    train = all_items[:train_n]
    evaluation = all_items[train_n:]
    for item in train + evaluation:
        item["nominal"] = nominal_six_channel(item["reference_data"], profile)
    nominal_saved = nominal_six_channel(saved_ref, profile)
    saved_fit_residual = starting_coeff - nominal_saved
    if not Spline(float(saved_ref["time_s"][-1]), starting_coeff).bounds(
        profile["equilibrium_lower"],
        profile["equilibrium_upper"],
        profile["equilibrium_rate_limit"],
        profile["equilibrium_acceleration_limit"],
    ):
        raise ValueError("Saved fitted controller violates the declared six-channel spline bounds")
    start_roundtrip, start_shrink = project_shared_residual(
        nominal_saved[None],
        saved_fit_residual,
        np.asarray([saved_ref["time_s"][-1]]),
        profile,
    )
    if start_shrink[0] != 1.0 or not np.allclose(start_roundtrip[0], starting_coeff, rtol=0.0, atol=1e-14):
        raise ValueError("Saved controller could not be reconstructed exactly from its shared residual")
    if args.initial_controller is not None:
        starting_residual = _load_residual_checkpoint(args.initial_controller)
        initialization = {
            "kind": "residual_checkpoint",
            "source": str(args.initial_controller.resolve()),
            "sha256": hashlib.sha256(args.initial_controller.read_bytes()).hexdigest(),
        }
    elif args.fresh_controller:
        starting_residual = np.zeros((12, 6), dtype=np.float64)
        initialization = {"kind": "fresh_zero_residual", "source": None, "sha256": None}
    else:
        starting_residual = saved_fit_residual.copy()
        initialization = {
            "kind": "saved_fit_transfer",
            "source": str((fit_dir / "equilibrium.npz").resolve()),
            "sha256": hashlib.sha256((fit_dir / "equilibrium.npz").read_bytes()).hexdigest(),
        }
    settings = FitConfig(**{**asdict(settings), "control_count": 12})
    population = 1 + 2 * args.population_pairs
    pool = _EnginePool(4, profile, shoe, config, settings, shoe["friction_model"])
    score_pool = _EnginePool(4, profile, shoe, config, settings, shoe["friction_model"])
    train_all_start = time.perf_counter()
    start_train_coeff, start_train_factors = _project(starting_residual, train, profile)
    start_train_loss, start_train = _score(
        start_train_coeff,
        train,
        profile,
        score_pool,
        1,
        args.failure_penalty,
        settings,
    )
    initial_full_train_wall = time.perf_counter() - train_all_start
    eval_start = time.perf_counter()
    start_eval_loss, start_eval = (None, {"valid": None, "omitted": True})
    if evaluation:
        start_eval_coeff, start_eval_factors = _project(starting_residual, evaluation, profile)
        start_eval_loss, start_eval = _score(
            start_eval_coeff,
            evaluation,
            profile,
            score_pool,
            1,
            args.failure_penalty,
            settings,
        )
    initial_eval_wall = time.perf_counter() - eval_start
    print(
        json.dumps(
            {
                "stage": "initial_scores",
                "train_loss": start_train_loss,
                "eval_loss": start_eval_loss,
                "train_failures": len(start_train["failed_stances"]),
                "eval_failures": len(start_eval.get("failed_stances", [])),
            }
        ),
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    scales = np.asarray(settings.parameter_scale, dtype=np.float64)
    if scales.shape != (6,):
        raise ValueError("Shared six-channel fitting requires six parameter scales")
    coordinate_scale = np.broadcast_to(scales, (12, 6))
    accepted_residual = starting_residual.copy()
    sigma = args.sigma
    history = []
    coverage = set()
    search_started = time.perf_counter()
    batches = _balanced_batches(train, args.batch_size, args.iterations, args.seed)
    for iteration, batch in enumerate(batches, start=1):
        iteration_started = time.perf_counter()
        setup_before = pool.setup_s
        coverage.update(item["id"] for item in batch)
        candidates = [accepted_residual.copy()]
        noise_pairs = rng.normal(size=(args.population_pairs, 12, 6)) * (sigma * coordinate_scale)
        for noise in noise_pairs:
            candidates.extend((accepted_residual + noise, accepted_residual - noise))
        candidate_coefficients_by_stance = []
        shrink_records = []
        for candidate in candidates:
            projected, shrink = _project(candidate, batch, profile)
            candidate_coefficients_by_stance.append(projected)
            shrink_records.append(shrink)
        rollout_started = time.perf_counter()
        candidate_losses, candidate_meta = _score_population(
            np.stack(candidate_coefficients_by_stance, axis=1),
            batch,
            pool,
            args.failure_penalty,
            settings,
        )
        order = np.argsort(candidate_losses)
        elite_count = min(len(candidates), max(2, int(math.ceil(len(candidates) * args.elite_fraction))))
        elite_indices = [
            int(index)
            for index in order[:elite_count]
            if np.isfinite(candidate_losses[index]) and candidate_meta[index]["valid"]
        ]
        proposed_mean = (
            np.mean(np.stack([candidates[index] for index in elite_indices]), axis=0)
            if elite_indices
            else accepted_residual.copy()
        )
        # Evaluate the CEM mean itself; an elite average is never accepted by interpolation alone.
        mean_projection, mean_shrink = _project(proposed_mean, batch, profile)
        mean_loss, mean_detail = _score(mean_projection, batch, profile, pool, 1, args.failure_penalty, settings)
        rollout_wall = time.perf_counter() - rollout_started
        best_candidate = int(order[0])
        batch_baseline = candidate_losses[0]
        accepted_kind = "incumbent"
        accepted_detail = candidate_meta[0]
        accepted_loss = batch_baseline
        if mean_detail["valid"] and mean_loss < batch_baseline:
            accepted_residual = proposed_mean.copy()
            accepted_kind = "cem_mean"
            accepted_detail = mean_detail
            accepted_loss = mean_loss
        elif candidate_meta[best_candidate]["valid"] and candidate_losses[best_candidate] < batch_baseline:
            accepted_residual = candidates[best_candidate].copy()
            accepted_kind = "sample"
            accepted_detail = candidate_meta[best_candidate]
            accepted_loss = candidate_losses[best_candidate]
        history.append(
            {
                "iteration": iteration,
                "stance_ids": [item["id"] for item in batch],
                "batch_incumbent_loss": float(batch_baseline),
                "best_sample_loss": float(candidate_losses[best_candidate]),
                "proposed_mean_loss": float(mean_loss),
                "accepted_kind": accepted_kind,
                "accepted_batch_loss": float(accepted_loss),
                "accepted_by_measured_tolerances": accepted_detail["accepted_by_measured_tolerances"],
                "best_sample_stance_metrics": candidate_meta[best_candidate]["stance_metrics"],
                "proposed_mean_stance_metrics": mean_detail["stance_metrics"],
                "elite_count": elite_count,
                "population_worlds": population,
                "completed_rollouts": int(
                    sum(detail["completed_rollouts"] for detail in candidate_meta) + mean_detail["completed_rollouts"]
                ),
                "expected_rollouts": int(population * len(batch) + len(batch)),
                "failed_candidate_count": int(
                    sum(not detail["valid"] for detail in candidate_meta) + (not mean_detail["valid"])
                ),
                "minimum_projection_factor": float(min(np.min(x) for x in [*shrink_records, mean_shrink])),
                "sigma_dimensionless": sigma,
                "rollout_wall_s": rollout_wall,
                "engine_setup_capture_wall_s": pool.setup_s - setup_before,
                "iteration_wall_s": time.perf_counter() - iteration_started,
            }
        )
        if args.checkpoint_every and iteration % args.checkpoint_every == 0:
            temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp.npz")
            np.savez_compressed(temporary, residual=accepted_residual, iteration=iteration, seed=args.seed)
            temporary.replace(checkpoint_path)
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "batch_loss": batch_baseline,
                    "best_sample": candidate_losses[best_candidate],
                    "cem_mean": mean_loss,
                    "accepted": accepted_kind,
                }
            ),
            flush=True,
        )
    search_wall = time.perf_counter() - search_started

    final_train_coeff, final_train_scales = _project(accepted_residual, train, profile)
    final_eval_coeff, final_eval_scales = (
        _project(accepted_residual, evaluation, profile) if evaluation else (np.empty((0, 12, 6)), np.empty(0))
    )
    final_train_start = time.perf_counter()
    final_train_loss, final_train = _score(
        final_train_coeff, train, profile, score_pool, 1, args.failure_penalty, settings
    )
    final_train_wall = time.perf_counter() - final_train_start
    final_eval_loss, final_eval = None, {"valid": None, "omitted": True}
    final_eval_wall = 0.0
    if evaluation:
        final_eval_start = time.perf_counter()
        final_eval_loss, final_eval = _score(
            final_eval_coeff, evaluation, profile, score_pool, 1, args.failure_penalty, settings
        )
        final_eval_wall = time.perf_counter() - final_eval_start

    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "shared_controller.npz",
        residual=accepted_residual,
        train_coefficients=final_train_coeff,
        train_projection_factors=final_train_scales,
        eval_coefficients=final_eval_coeff,
        eval_projection_factors=final_eval_scales,
        parameter_scale=scales,
        train_ids=np.asarray([item["id"] for item in train]),
        eval_ids=np.asarray([item["id"] for item in evaluation]),
    )
    dataset_hash = hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
    fit_hash = hashlib.sha256((fit_dir / "equilibrium.npz").read_bytes()).hexdigest()
    last_checkpoint_iteration = (
        (history[-1]["iteration"] // args.checkpoint_every) * args.checkpoint_every
        if args.checkpoint_every and history
        else 0
    )
    report = {
        "schema": "cartesian_shared_residual_search_1",
        "method": "antithetic Gaussian population search with evaluated elite-mean CEM update",
        "policy": "One common 12x6 residual added to a deterministic stance-conditioned six-channel nominal spline.",
        "optimizer_scope": "episodic shared-controller black-box fitting; not PPO",
        "initialization": {
            **initialization,
            "saved_fit_roundtrip_projection_factor": float(start_shrink[0]),
            "saved_fit_roundtrip_max_abs_error": float(np.max(np.abs(start_roundtrip[0] - starting_coeff))),
            "training_projection_factors": start_train_factors.tolist(),
            "evaluation_projection_factors": start_eval_factors.tolist() if evaluation else [],
        },
        "parameters": {
            "coefficients": 72,
            "worlds_per_population_batch": population,
            "population_pairs": args.population_pairs,
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "sigma_dimensionless": args.sigma,
            "failure_penalty": args.failure_penalty,
            "seed": args.seed,
        },
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "dataset_manifest_sha256": dataset_hash,
            "saved_fit": str(fit_dir.resolve()),
            "saved_controller_sha256": fit_hash,
            "profile_source": str((fit_dir / "profile.json").resolve()),
            "profile_sha256": hashlib.sha256((fit_dir / "profile.json").read_bytes()).hexdigest(),
            "dataset_profile": str((args.dataset / "assets" / "profile.json").resolve()),
            "dataset_profile_sha256": hashlib.sha256(
                (args.dataset / "assets" / "profile.json").read_bytes()
            ).hexdigest(),
            "profile_policy": "The saved fit's explicit six-channel profile supplies all simulation gains, bounds, and inertias; dataset reference-specific lengths and initial states remain stance-specific. The dataset's four-channel profile is recorded for provenance and is not substituted.",
            "shoe": shoe,
        },
        "selection": {
            "mode": "single_stance_benchmark" if args.benchmark_single else "multi_stance_training",
            "train_members_available": len([m for m in manifest["members"] if m["split"] == "train"]),
            "train_members_used": len(train),
            "eval_members_available": len([m for m in manifest["members"] if m["split"] == "eval"]),
            "eval_members_scored": len(evaluation),
            "train_coverage_unique_during_search": len(coverage),
            "complete_train_coverage": len(train) == 100 and len(coverage) == len(train),
            "train_limit": args.train_limit,
            "score_limit": args.score_limit,
            "seed": args.seed,
            "all_stance_aggregation": "equal mean of each complete stance loss; any failed stance invalidates a candidate",
        },
        "initial": {
            "train_loss": start_train_loss,
            "train": start_train,
            "eval_loss": start_eval_loss,
            "eval": start_eval,
        },
        "final": {
            "train_loss": final_train_loss,
            "train": final_train,
            "eval_loss": final_eval_loss,
            "eval": final_eval,
        },
        "timings_s": {
            "search": search_wall,
            "initial_full_train": initial_full_train_wall,
            "initial_eval": initial_eval_wall,
            "final_full_train": final_train_wall,
            "final_eval": final_eval_wall,
            "engine_construction_and_capture": pool.setup_s + score_pool.setup_s,
            "total_command": time.perf_counter() - command_started,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()) if last_checkpoint_iteration else None,
            "every_iterations": args.checkpoint_every,
            "final_checkpoint_iteration": last_checkpoint_iteration,
            "seed": args.seed,
            "resume_semantics": "A checkpoint initializes a new run; optimizer RNG/CEM history are not resumed.",
        },
        "history": history,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "initial_train_loss": start_train_loss,
                "final_train_loss": final_train_loss,
                "initial_eval_loss": start_eval_loss,
                "final_eval_loss": final_eval_loss,
                "search_wall_s": search_wall,
                "train_coverage": len(coverage),
            }
        ),
        flush=True,
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-fit", type=Path, default=_DEFAULT_FIT)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--population-pairs", type=int, default=12)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--score-limit", type=int)
    parser.add_argument("--failure-penalty", type=float, default=1.0e12)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write a resumable sibling residual checkpoint every N iterations; zero disables",
    )
    initial_group = parser.add_mutually_exclusive_group()
    initial_group.add_argument(
        "--initial-controller", type=Path, help="NPZ checkpoint containing a finite shared residual [12, 6]"
    )
    initial_group.add_argument(
        "--fresh-controller",
        action="store_true",
        help="Start from the deterministic nominal controller with zero residual",
    )
    parser.add_argument(
        "--benchmark-single",
        action="store_true",
        help="Use the saved FR3_2 reference for a matched single-stance timing run",
    )
    parser.set_defaults(func=run)
    return parser


def main():
    args = build_parser().parse_args()
    if args.iterations < 1 or args.batch_size < 1 or args.population_pairs < 1:
        raise SystemExit("iterations, batch-size, and population-pairs must be positive")
    if not 0 < args.elite_fraction <= 1 or not np.isfinite(args.sigma) or args.sigma <= 0:
        raise SystemExit("elite-fraction and sigma must be positive (elite-fraction <= 1)")
    if args.train_limit is not None and args.train_limit < 1:
        raise SystemExit("train-limit must be positive")
    if args.score_limit is not None and args.score_limit < 1:
        raise SystemExit("score-limit must be positive")
    if args.checkpoint_every < 0:
        raise SystemExit("checkpoint-every must be nonnegative")
    args.func(args)


if __name__ == "__main__":
    main()
