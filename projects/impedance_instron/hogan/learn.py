# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learn one phase-scheduled stiffness and damping over many measured stances.

The schedule is shared by every stance on the normalized gait phase of
:mod:`.batch`. Pelvis, hip, knee, and ankle gains are searched in log space
about the fixed baseline (``K`` of :data:`.__main__.DEFAULT_STIFFNESS`,
critically damped on the mean initial mass-matrix diagonal); the hip x/z
channels stay at zero so the body is carried by the shoe alone. The search is a
cross-entropy method over the training stances; held-out stances are only
evaluated.

Example::

    uv run python -m projects.impedance_instron.hogan.learn \\
        --dataset outputs/impedance_instron/hogan_stance_dataset \\
        --mount -0.0319 0.0 0.1094 --output runs/hogan_learn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..cartesian import data as reference_data
from ..cartesian import profile as profile_data
from ..cartesian.shoe import Shoe
from .__main__ import DEFAULT_STIFFNESS
from .batch import FAILURE_REASONS, Batch, reference_timing
from .mechanics import COORDINATE_NAMES, RestOfBody, chain_from_profile
from .plan import Plan, build_plan
from .registration import register_static_height
from .rollout import PHASE_MODES, Config

LEARNED = slice(2, 6)
DEFAULT_KNOTS = (0.0, 0.5, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.5, 3.0)
# Error scales that make one unit of loss per channel group.
JOINT_SCALE_RAD = 0.05
HIP_SCALE_M = 0.02
FORCE_SCALE_N = 100.0
PEAK_SCALE_N = 100.0
FAILURE_PENALTY = 20.0


def _plan_arrays(plan: Plan) -> dict:
    return {
        "time_s": plan.time_s,
        "q": plan.q,
        "v": plan.v,
        "a": plan.a,
        "feedforward": plan.feedforward,
        "grf_n": plan.grf_n,
        "touchdown_s": np.asarray(plan.touchdown_s),
        "contact_source": np.asarray(plan.contact_source),
        "diagnostics_json": np.asarray(json.dumps(plan.diagnostics)),
    }


def _load_plan(path: Path) -> Plan:
    with np.load(path, allow_pickle=False) as a:
        return Plan(
            a["time_s"],
            a["q"],
            a["v"],
            a["a"],
            a["feedforward"],
            a["grf_n"],
            float(a["touchdown_s"]),
            str(a["contact_source"]),
            json.loads(str(a["diagnostics_json"])),
        )


def load_stances(dataset: Path, args, cache: Path) -> tuple[list[dict], list, list[Plan], dict]:
    """Build or reload each member's chain and plan; registration is shared per trial."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    profile = profile_data.load(dataset / manifest["shared_assets"]["profile.json"]["file"])
    cache.mkdir(parents=True, exist_ok=True)
    members, chains, plans, registrations = [], [], [], {}
    shoe = None
    for member in manifest["members"]:
        reference = reference_data.load(dataset / member["reference"])
        chain = chain_from_profile(reference, profile, RestOfBody())
        path = cache / f"{member['id']}.npz"
        if path.exists():
            plan = _load_plan(path)
        else:
            if shoe is None:
                shoe = Shoe(args.shoe_artifact, args.mount, args.pitch, friction_model=args.friction_model)
            trial = member["trial"]
            if trial not in registrations:
                load = args.static_load_fraction * float(reference["subject_mass_kg"]) * 9.81
                registrations[trial] = register_static_height(reference, shoe, static_load_n=load).to_dict()
            plan = build_plan(
                reference,
                chain,
                dt_s=args.dt,
                contact="measured",
                height_offset_m=registrations[trial]["height_offset_m"],
                cutoff_hz=args.cutoff or None,
                pelvis_cutoff_hz=args.pelvis_cutoff or None,
            )
            np.savez(path, **_plan_arrays(plan))
        members.append(member)
        chains.append(chain)
        plans.append(plan)
    return members, chains, plans, registrations


class Schedule:
    """Map log-gain offsets, shape [knots, 4, 2], to stiffness and damping tables.

    Args:
        knots_phase: Normalized gait phases of the knots, shape [knots].
        stiffness: Baseline stiffness, shape (6,).
        damping: Baseline damping, shape (6,).
    """

    def __init__(self, knots_phase, stiffness, damping):
        self.knots_phase = np.asarray(knots_phase, dtype=float)
        self.stiffness = np.asarray(stiffness, dtype=float)
        self.damping = np.asarray(damping, dtype=float)
        self.shape = (len(self.knots_phase), LEARNED.stop - LEARNED.start, 2)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def tables(self, theta) -> tuple[np.ndarray, np.ndarray]:
        """Return stiffness and damping, each shape [candidates, knots, 6]."""
        theta = np.asarray(theta, dtype=float).reshape(-1, *self.shape)
        k = np.broadcast_to(self.stiffness, (len(theta), self.shape[0], 6)).copy()
        d = np.broadcast_to(self.damping, (len(theta), self.shape[0], 6)).copy()
        k[:, :, LEARNED] *= np.exp(theta[..., 0])
        d[:, :, LEARNED] *= np.exp(theta[..., 1])
        return k, d


def stance_loss(result: dict, steps: np.ndarray, reference_peak_fz: np.ndarray) -> np.ndarray:
    """Return the per-world loss, shape [candidates, stances]."""
    tracking = result["tracking_rmse"]
    joints = np.mean(np.square(tracking[..., 2:] / JOINT_SCALE_RAD), axis=-1)
    hip = np.mean(np.square(tracking[..., :2] / HIP_SCALE_M), axis=-1)
    force = np.mean(np.square(result["grf_rmse_n"] / FORCE_SCALE_N), axis=-1)
    peak = np.square((result["peak_grf_n"][..., 1] - reference_peak_fz[None, :]) / PEAK_SCALE_N)
    loss = joints + hip + force + peak
    failed = result["status"] != 1
    unfinished = 1.0 - result["recorded"] / steps[None, :]
    return np.where(failed, loss + FAILURE_PENALTY * (1.0 + unfinished), loss)


def _summaries(result: dict, members: list[dict], candidate: int, loss: np.ndarray) -> list[dict]:
    rows = []
    for s, member in enumerate(members):
        status = int(result["status"][candidate, s])
        rows.append(
            {
                "id": member["id"],
                "trial": member["trial"],
                "status": "completed" if status == 1 else "failed",
                "failure": [text for bit, text in FAILURE_REASONS.items() if status & bit] if status != 1 else [],
                "loss": float(loss[candidate, s]),
                "tracking_rmse": dict(
                    zip(COORDINATE_NAMES, result["tracking_rmse"][candidate, s].tolist(), strict=True)
                ),
                "grf_rmse_n": result["grf_rmse_n"][candidate, s].tolist(),
                "peak_grf_n": result["peak_grf_n"][candidate, s].tolist(),
                "contact_duration_s": float(result["contact_duration_s"][candidate, s]),
                "maximum_compression_fraction": float(result["maximum_compression_fraction"][candidate, s]),
                "touchdown_s": float(result["touchdown_s"][candidate, s]),
                "toeoff_s": float(result["toeoff_s"][candidate, s]),
            }
        )
    return rows


def _aggregate(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["status"] == "completed"]
    if not completed:
        return {"completed": 0, "stances": len(rows)}
    timing = {}
    for event in ("touchdown", "toeoff"):
        errors = [row[f"{event}_s"] - row[f"reference_{event}_s"] for row in completed if row[f"{event}_s"] >= 0.0]
        if errors:
            timing[f"{event}_error_s"] = float(np.mean(errors))
    return {
        **timing,
        "completed": len(completed),
        "stances": len(rows),
        "mean_loss": float(np.mean([row["loss"] for row in rows])),
        "tracking_rmse": {
            name: float(np.mean([row["tracking_rmse"][name] for row in completed])) for name in COORDINATE_NAMES
        },
        "grf_rmse_n": np.mean([row["grf_rmse_n"] for row in completed], axis=0).tolist(),
        "peak_fz_error_n": float(
            np.mean([abs(row["peak_grf_n"][1] - row["reference_peak_fz_n"]) for row in completed])
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mount", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shoe-artifact", type=Path, help="Defaults to the dataset's shared shoe")
    parser.add_argument("--pitch", type=float, help="Static shoe pitch [rad]; defaults to the first reference")
    parser.add_argument("--phase", choices=PHASE_MODES, default="time")
    parser.add_argument("--dt", type=float, default=1.25e-4)
    parser.add_argument("--cutoff", type=float, default=12.0)
    parser.add_argument("--pelvis-cutoff", type=float, default=4.0)
    parser.add_argument("--static-load-fraction", type=float, default=0.5)
    parser.add_argument("--friction-model", default="elastic_coulomb")
    parser.add_argument("--knots", type=float, nargs="+", default=list(DEFAULT_KNOTS))
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.5, help="Initial log-gain standard deviation")
    parser.add_argument("--sigma-floor", type=float, default=0.03)
    parser.add_argument("--smoothing", type=float, default=0.7, help="CEM update weight of the elite statistics")
    parser.add_argument("--bound", type=float, default=3.0, help="Log-gain offset bound")
    parser.add_argument("--regularization", type=float, default=0.01, help="Weight on the mean squared log offset")
    parser.add_argument(
        "--roughness", type=float, default=0.1, help="Weight on the mean squared second difference of log offsets"
    )
    parser.add_argument("--stances", type=int, help="Use only the first N training stances")
    parser.add_argument(
        "--minibatches",
        type=int,
        default=1,
        help="Split training stances into this many interleaved groups and score one group per generation",
    )
    parser.add_argument("--init", type=Path, help="Previous learn output whose final mean starts the search")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if args.shoe_artifact is None:
        args.shoe_artifact = args.dataset / manifest["shared_assets"]["digital_shoe.json"]["file"]
    if args.pitch is None:
        first = reference_data.load(args.dataset / manifest["members"][0]["reference"])
        args.pitch = float(first.get("shoe_static_pitch_rad", first["static_pitch_rad"]))
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    members, chains, plans, registrations = load_stances(args.dataset, args, args.output / "plans")
    print(f"loaded {len(plans)} stance plans in {perf_counter() - started:.1f} s")
    config = Config(phase=args.phase)
    for member, plan in zip(members, plans, strict=True):
        member["reference_peak_fz_n"] = float(np.max(plan.grf_n[:, 1]))
    split = {name: [i for i, member in enumerate(members) if member["split"] == name] for name in ("train", "eval")}
    if args.stances:
        split["train"] = split["train"][: args.stances]

    inertia = np.mean([np.diag(chains[i].dynamics(plans[i].q[0], plans[i].v[0])[0]) for i in split["train"]], axis=0)
    base_k = np.asarray(DEFAULT_STIFFNESS, dtype=float)
    schedule = Schedule(args.knots, base_k, 2.0 * np.sqrt(base_k * inertia))

    def batch(indices, candidates):
        return Batch(
            [plans[i] for i in indices],
            [chains[i] for i in indices],
            args.shoe_artifact,
            args.mount,
            args.pitch,
            args.knots,
            candidates=candidates,
            config=config,
            device=args.device,
            friction_model=args.friction_model,
        )

    groups = [split["train"][g :: args.minibatches] for g in range(args.minibatches)]
    trains = [batch(group, args.population) for group in groups]
    group_steps = [train.steps.astype(float) for train in trains]
    group_peaks = [np.array([members[i]["reference_peak_fz_n"] for i in group]) for group in groups]
    print(
        f"{sum(train.world_count for train in trains)} worlds ({len(split['train'])} stances x {args.population}"
        f" in {args.minibatches} minibatches), setup {sum(train.setup_wall_s for train in trains):.1f} s"
    )

    rng = np.random.default_rng(args.seed)
    mean = np.zeros(schedule.size)
    if args.init is not None:
        with np.load(args.init / "schedule.npz", allow_pickle=False) as archive:
            labels = [str(label) for label in archive["labels"]]
            mean = archive["theta"][labels.index("final_mean")].copy()
        if mean.shape != (schedule.size,):
            raise ValueError("--init schedule must use the same knots")
    sigma = np.full(schedule.size, args.sigma)
    elite_count = max(2, round(args.elite_fraction * args.population))
    # With minibatches, best is ranked on its own group; the final evaluation re-scores it on all stances.
    best = {"score": np.inf, "theta": mean.copy(), "generation": -1}
    history = []
    for generation in range(args.generations):
        part = generation % args.minibatches
        train = trains[part]
        noise = rng.standard_normal((args.population - 1, schedule.size))
        theta = np.vstack((mean, np.clip(mean + sigma * noise, -args.bound, args.bound)))
        result = train.evaluate(*schedule.tables(theta))
        loss = stance_loss(result, group_steps[part], group_peaks[part])
        score = loss.mean(axis=1) + args.regularization * np.mean(np.square(theta), axis=1)
        if len(args.knots) > 2:
            curvature = np.diff(theta.reshape(-1, *schedule.shape), n=2, axis=1)
            score += args.roughness * np.mean(np.square(curvature), axis=(1, 2, 3))
        order = np.argsort(score)
        elite = theta[order[:elite_count]]
        if score[order[0]] < best["score"]:
            best = {"score": float(score[order[0]]), "theta": theta[order[0]].copy(), "generation": generation}
        mean = (1.0 - args.smoothing) * mean + args.smoothing * elite.mean(axis=0)
        sigma = np.maximum((1.0 - args.smoothing) * sigma + args.smoothing * elite.std(axis=0), args.sigma_floor)
        failures = int(np.count_nonzero(result["status"] != 1))
        history.append(
            {
                "generation": generation,
                "minibatch": part,
                "mean_score": float(score[0]),
                "best_score": float(score[order[0]]),
                "median_score": float(np.median(score)),
                "failed_worlds": failures,
                "sigma_mean": float(sigma.mean()),
                "wall_s": train.last_wall_s,
            }
        )
        print(
            f"gen {generation:3d}  mean {score[0]:8.3f}  best {score[order[0]]:8.3f}  "
            f"median {np.median(score):8.3f}  failed {failures:5d}  sigma {sigma.mean():.3f}  {train.last_wall_s:.1f} s"
        )

    # Re-evaluate the final mean too; the best sample may owe part of its score to its own noise.
    final = np.vstack((np.zeros(schedule.size), best["theta"], mean))
    labels = ("baseline", "best", "final_mean")
    report = {
        "dataset": str(args.dataset),
        "shoe_artifact": str(args.shoe_artifact),
        "shoe_mount_m": list(args.mount),
        "shoe_static_pitch_rad": args.pitch,
        "phase": args.phase,
        "friction_model": args.friction_model,
        "registration": registrations,
        "knots_phase": list(args.knots),
        "interpolation": "pchip",
        "baseline": {"stiffness": schedule.stiffness.tolist(), "damping": schedule.damping.tolist()},
        "loss_scales": {
            "joint_rad": JOINT_SCALE_RAD,
            "hip_m": HIP_SCALE_M,
            "force_n": FORCE_SCALE_N,
            "peak_fz_n": PEAK_SCALE_N,
            "failure_penalty": FAILURE_PENALTY,
        },
        "search": {
            "method": "cross-entropy",
            "population": args.population,
            "generations": args.generations,
            "elite_count": elite_count,
            "minibatches": args.minibatches,
            "init": None if args.init is None else str(args.init),
            "regularization": args.regularization,
            "roughness": args.roughness,
            "seed": args.seed,
            "best_generation": best["generation"],
        },
        "history": history,
        "splits": {},
    }
    k, d = schedule.tables(final)
    np.savez(
        args.output / "schedule.npz",
        knots_phase=np.asarray(args.knots),
        labels=np.asarray(labels),
        stiffness=k,
        damping=d,
        theta=final,
    )
    for name, indices in split.items():
        if not indices:
            continue
        evaluation = batch(indices, len(final))
        result = evaluation.evaluate(k, d)
        loss = stance_loss(
            result,
            evaluation.steps.astype(float),
            np.array([members[i]["reference_peak_fz_n"] for i in indices]),
        )
        subset = [members[i] for i in indices]
        report["splits"][name] = {}
        for c, label in enumerate(labels):
            rows = _summaries(result, subset, c, loss)
            for row, i in zip(rows, indices, strict=True):
                timing = reference_timing(plans[i], config.contact_threshold_n)
                row["reference_peak_fz_n"] = members[i]["reference_peak_fz_n"]
                row["reference_touchdown_s"] = float(timing[0])
                row["reference_toeoff_s"] = float(timing[1])
                row["reference_contact_duration_s"] = float(timing[1] - timing[0])
            report["splits"][name][label] = {"aggregate": _aggregate(rows), "stances": rows}
        for label in labels:
            agg = report["splits"][name][label]["aggregate"]
            print(
                f"{name:<6} {label:<11} loss {agg.get('mean_loss', np.nan):7.3f}  "
                f"completed {agg['completed']}/{agg['stances']}  "
                f"Fz RMSE {agg.get('grf_rmse_n', [np.nan, np.nan])[1]:6.1f} N  "
                f"ankle {agg.get('tracking_rmse', {}).get('ankle', np.nan):.4f} rad"
            )
    report["phase_map"] = "phi = 0..1 window start to touchdown, 1..2 contact, 2..3 toe-off to window end"
    report["wall_s"] = perf_counter() - started
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output} in {report['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
