# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a fixed-budget, train-only constant-versus-variable impedance experiment."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .identify import Parameterization, evaluate, load_trials, predict_many, scenario, score
from .runner import RolloutConfig, Runner


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def active_parameters(parameters: Parameterization, variable: bool) -> np.ndarray:
    """Keep equilibrium dynamics trainable in both arms; fix only nonconstant K/D."""
    active = np.ones(parameters.size, dtype=bool)
    if not variable:
        outputs, _, features = np.unravel_index(parameters.indices, parameters.baseline.weights.shape)
        active[: len(parameters.indices)] = (outputs == 0) | (features == 0)
    return active


def search(baseline: Runner, trials, *, variable: bool, output: Path, protocol: dict) -> Runner:
    """Use matched candidate budgets, common random draws, and training data only."""
    parameters = Parameterization(baseline, [trial.task.speed_m_s for trial in trials])
    active = active_parameters(parameters, variable)
    config = RolloutConfig(dt_s=protocol["search_dt_s"])
    rng = np.random.default_rng(protocol["seed"])
    mean = np.zeros(parameters.size)
    sigma = np.full(parameters.size, protocol["sigma"])
    best = mean.copy()
    best_rank = (len(trials) + 1, float("inf"))
    history = []
    start = perf_counter()
    population = protocol["population"]
    evaluator = None
    if protocol.get("device", "cpu") != "cpu":
        from .gpu_objective import GpuEvaluator  # noqa: PLC0415 - optional execution backend

        evaluator = GpuEvaluator(trials, candidates=population, config=config, device=protocol["device"])

    def evaluations(models):
        if evaluator is None:
            return [evaluate(model, trials, config) for model in models]
        count = len(models)
        return evaluator.evaluate(models + [models[-1]] * (population - count))[:count]

    for generation in range(protocol["generations"]):
        samples = np.vstack(
            (
                mean,
                best,
                np.clip(
                    mean + sigma * rng.standard_normal((population - 2, parameters.size)),
                    -protocol["bound"],
                    protocol["bound"],
                ),
            )
        )
        samples[:, ~active] = 0.0
        ranks, losses = [], []
        for offset, result in zip(samples, evaluations([parameters.model(x) for x in samples]), strict=True):
            # Same normalization for both arms; absent variable-gain coefficients are zero.
            penalty = protocol["regularization"] * float(np.mean(offset**2))
            ranks.append((result["failed"], result["mean_loss"] + penalty))
            losses.append(result["mean_loss"])
        order = sorted(range(population), key=lambda i: ranks[i])
        if ranks[order[0]] < best_rank:
            best, best_rank = samples[order[0]].copy(), ranks[order[0]]
        elite = samples[order[:2]]
        mean = 0.3 * mean + 0.7 * elite.mean(0)
        sigma = np.maximum(0.3 * sigma + 0.7 * elite.std(0), 0.03)
        history.append(
            {
                "generation": generation,
                "best_score": best_rank[1],
                "best_failed": best_rank[0],
                "candidate_losses": losses,
                "wall_s": perf_counter() - start,
            }
        )
        parameters.model(best).save(output / "runner.json")
        _write(output / "search.json", {"active_parameters": int(active.sum()), "history": history})
        print(
            f"{output.name} generation {generation + 1}: train score {best_rank[1]:.3f}, failed {best_rank[0]}",
            flush=True,
        )
    # Keep this final-mean test equal in both arms and record it in the budget.
    result = evaluations([parameters.model(mean)])[0]
    rank = (result["failed"], result["mean_loss"] + protocol["regularization"] * float(np.mean(mean**2)))
    if rank < best_rank:
        best = mean
    model = parameters.model(best)
    model.save(output / "runner.json")
    _write(
        output / "search.json",
        {
            "active_parameters": int(active.sum()),
            "candidate_evaluations": population * protocol["generations"] + 1,
            "stance_rollouts": (population * protocol["generations"] + 1) * len(trials),
            "wall_s": perf_counter() - start,
            "history": history,
        },
    )
    return model


def aggregate(rows: list[dict]) -> dict:
    """Report mean per-stance metrics without hiding failures or worsening channels."""
    return {
        "count": len(rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "mean_loss": float(np.mean([row["loss"] for row in rows])),
        "grf_rmse_n": np.mean([row["grf_rmse_n"] for row in rows], axis=0).tolist(),
        "tracking_rmse": np.mean([row["tracking_rmse"] for row in rows], axis=0).tolist(),
        "peak_fz_mae_n": float(np.mean([abs(row["peak_fz_error_n"]) for row in rows])),
        "contact_mae_ms": 1000 * float(np.mean([abs(row["contact_duration_error_s"]) for row in rows])),
        "impulse_mae_ns": np.mean([np.abs(row["impulse_error_ns"]) for row in rows], axis=0).tolist(),
    }


def plot(series, ylabel: str) -> str:
    """Create a dependency-free inline SVG curve with common axes."""
    colors = ("#475569", "#d97706", "#2563eb", "#059669")
    series = [(label, t, y) for label, t, y in series if len(t) and len(y)]
    xmin, xmax = 0.0, max(float(np.max(t)) for _, t, _ in series)
    ymin = min(float(np.min(y)) for _, _, y in series)
    ymax = max(float(np.max(y)) for _, _, y in series)
    pad = max((ymax - ymin) * 0.08, 1e-6)
    ymin, ymax = ymin - pad, ymax + pad
    parts = [f'<svg viewBox="0 0 660 320" role="img" aria-label="{html.escape(ylabel)}">']
    for i in range(5):
        y = 250 - i * 50
        value = ymin + (ymax - ymin) * i / 4
        parts.append(f'<path d="M60 {y}H630" stroke="#e2e8f0"/><text x="4" y="{y + 4}">{value:.2f}</text>')
    for index, (label, t, values) in enumerate(series):
        stride = max(1, len(t) // 600)
        points = " ".join(
            f"{60 + 570 * (x - xmin) / max(xmax - xmin, 1e-9):.2f},{250 - 200 * (y - ymin) / (ymax - ymin):.2f}"
            for x, y in zip(t[::stride], values[::stride], strict=True)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[index]}" stroke-width="2"/>')
        parts.append(f'<text x="{60 + index * 145}" y="290" fill="{colors[index]}">{html.escape(label)}</text>')
    parts.append(
        f'<text x="60" y="22">{html.escape(ylabel)}</text><text x="500" y="272">time [s], end {xmax:.3f}</text></svg>'
    )
    return "".join(parts)


def report(output: Path, results: dict, trials, traces: dict) -> None:
    """Show all held-out examples, rather than selecting a favorable trajectory."""
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Generative runner quick fit</title>
<style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#172033;background:#f8fafc}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;text-align:right;border-bottom:1px solid #ddd}
td:first-child,th:first-child{text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
svg{background:white;border-radius:10px}svg text{font:12px system-ui}p{max-width:950px}</style>
<h1>Generative runner: matched-budget quick fit</h1><p><b>Diagnostic only—not validated.</b>
Two training stances, ten held-out stances. Same seed model, search draws, bounds, loss, and budget.
Only the availability of variable K/D differs. Equilibrium dynamics are trainable in both arms.
Models are frozen before held-out scoring. Known reference/geometry incompatibilities were explicitly allowed.</p>"""
    ]
    for split in ("train", "eval"):
        parts.append(
            f"<h2>{split}: full-resolution free predictions</h2><table><tr><th>Model</th><th>Completed</th>"
            "<th>Loss</th><th>Fx RMS [N]</th><th>Fz RMS [N]</th><th>Hip z RMS [mm]</th>"
            "<th>Knee RMS [deg]</th><th>Ankle RMS [deg]</th><th>Contact MAE [ms]</th></tr>"
        )
        for label in ("seed", "constant", "variable"):
            a = results["models"][label][split]["aggregate"]
            parts.append(
                f"<tr><td>{label}</td><td>{a['completed']}/{a['count']}</td><td>{a['mean_loss']:.2f}</td>"
                f"<td>{a['grf_rmse_n'][0]:.1f}</td><td>{a['grf_rmse_n'][1]:.1f}</td>"
                f"<td>{a['tracking_rmse'][1] * 1000:.1f}</td><td>{np.rad2deg(a['tracking_rmse'][4]):.2f}</td>"
                f"<td>{np.rad2deg(a['tracking_rmse'][5]):.2f}</td><td>{a['contact_mae_ms']:.1f}</td></tr>"
            )
        parts.append("</table>")
    parts.append("<h2>All held-out stances</h2><p>Figures use original time alignment, not phase warping.</p>")
    for trial in trials:
        if trial.split != "eval":
            continue
        parts.append(f"<h3>{html.escape(trial.id)}</h3><div class=grid>")
        mask = (trial.force_time_s >= 0) & (trial.force_time_s <= trial.duration_s)
        series = [("measured", trial.force_time_s[mask], trial.grf_n[mask, 1])]
        series.extend(
            (label, traces[(label, trial.id)]["time_s"][:-1], traces[(label, trial.id)]["grf_n"][:, 1])
            for label in ("seed", "constant", "variable")
        )
        parts.append(plot(series, "Vertical ground force [N]"))
        series = [("measured", trial.time_s, np.rad2deg(trial.q[:, 5]))]
        series.extend(
            (label, traces[(label, trial.id)]["time_s"], np.rad2deg(traces[(label, trial.id)]["state"][:, 5]))
            for label in ("seed", "constant", "variable")
        )
        parts.append(plot(series, "Ankle angle [deg]"))
        parts.append("</div>")
    parts.append(
        "<p>These are repeat stances at one speed/shoe, not independent speed/footwear validation. "
        "The old tracker receives future reference and measured-force feedforward; its published errors "
        "are not an equal-information benchmark for these free predictions.</p></html>"
    )
    (output / "report.html").write_text("\n".join(parts), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Freeze the protocol, fit on two stances, then evaluate all held-out cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-incompatible", action="store_true", help="Diagnostic-only explicit override")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.dataset.resolve()
    original = json.loads((root / "manifest.json").read_text())
    train_ids = ("FR3_1_train_000", "FR3_2_train_000")
    selected = [m.copy() for m in original["members"] if m["id"] in train_ids or m["split"] == "eval"]
    if sum(m["split"] == "train" for m in selected) != 2 or sum(m["split"] == "eval" for m in selected) != 10:
        raise ValueError("This fixed experiment requires the two named training stances and ten held-out stances")
    protocol = {
        "train_ids": list(train_ids),
        "eval_ids": [m["id"] for m in selected if m["split"] == "eval"],
        "population": 6,
        "generations": 6,
        "seed": 17,
        "sigma": 0.2,
        "bound": 1.5,
        "regularization": 0.01,
        "search_dt_s": 5e-4,
        "evaluation_dt_s": 1.25e-4,
        "candidate_evaluations_per_arm": 37,
        "selection": "training failures then penalized loss only",
        "primary_comparison": "variable versus constant held-out mean loss at evaluation_dt_s",
        "allow_incompatible": args.allow_incompatible,
        "validated": False,
        "device": args.device,
        "shared_initialization": "Runner.seed with all nonconstant K/D weights zeroed for both arms",
        "original_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("quick_fit.py", "identify.py", "runner.py", "mechanics.py")
        },
    }
    args.output.mkdir(parents=True)
    _write(args.output / "protocol.json", protocol)
    subset = args.output / "dataset"
    subset.mkdir()
    assets = {name: {**entry, "file": str(root / entry["file"])} for name, entry in original["shared_assets"].items()}
    for member in selected:
        member["reference"] = str(root / member["reference"])
    _write(subset / "manifest.json", {**original, "members": selected, "shared_assets": assets})
    trials = load_trials(subset, mount_m=[-0.03186147427106201, 0.0, 0.10943209684347802], speed_m_s=3.7)
    train = [trial for trial in trials if trial.split == "train"]
    incompatible = [trial.id for trial in trials if not trial.provenance["compatibility"]["passed"]]
    if incompatible and not args.allow_incompatible:
        _write(args.output / "blocked.json", {"reason": "contact compatibility failed", "trials": incompatible})
        raise ValueError("Contact compatibility failed; no fit launched. Inspect physical inputs first.")
    seed_data = Runner.seed().to_dict()
    weights = np.array(seed_data["weights"])
    weights[1:, :, 1:] = 0.0
    seed_data["weights"] = weights.tolist()
    baseline = Runner.from_dict(seed_data)
    baseline.save(args.output / "seed.json")
    _write(args.output / "inputs.json", {"trials": [{"id": t.id, **t.provenance} for t in trials]})
    start = perf_counter()
    models = {"seed": baseline}
    for label in ("constant", "variable"):
        destination = args.output / label
        destination.mkdir()
        models[label] = search(baseline, train, variable=label == "variable", output=destination, protocol=protocol)
    print("Both models frozen. Beginning full-resolution held-out evaluation.", flush=True)
    results = {"protocol": protocol, "models": {}, "evaluation_config": asdict(RolloutConfig())}
    traces = {}
    for label, model in models.items():
        rows = []
        destination = args.output / label
        destination.mkdir(exist_ok=True)
        model.save(destination / "runner.json")
        for trial, (trace, summary) in zip(
            trials, predict_many([model], trials, RolloutConfig(), device=args.device)[0], strict=True
        ):
            row = score(trace, summary, trial, model)
            row["maximum_external_actuator_load"] = (
                float(np.max(np.abs(trace["load"][:, :3]))) if len(trace["load"]) else 0.0
            )
            row["stiffness_range_nm_rad"] = (
                [trace["stiffness_nm_rad"].min(0).tolist(), trace["stiffness_nm_rad"].max(0).tolist()]
                if len(trace["load"])
                else None
            )
            rows.append(row)
            traces[(label, trial.id)] = trace
            np.savez_compressed(destination / f"{trial.id}.npz", **trace)
            _write(destination / f"{trial.id}.scenario.json", scenario(trial, RolloutConfig()))
            print(f"evaluate {label} {trial.id}: {row['status']}, loss {row['loss']:.3f}", flush=True)
        results["models"][label] = {
            split: {"aggregate": aggregate(subset_rows), "trials": subset_rows}
            for split in ("train", "eval")
            if (subset_rows := [row for row in rows if row["split"] == split])
        }
        results["wall_s"] = perf_counter() - start
        _write(args.output / "summary.json", results)
    report(args.output, results, trials, traces)
    print(f"Finished in {results['wall_s']:.1f} s: {args.output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
