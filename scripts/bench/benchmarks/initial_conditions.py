# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Initial conditions benchmark: epsilon sweep + N scaling comparison.

Two experiments:
1. Epsilon sweep at fixed N: how does perturbation magnitude affect K and cost?
2. N-scaling comparison: identical vs perturbed ICs across world counts.

Standalone:
    uv run python -m scripts.bench.benchmarks.initial_conditions
    uv run python -m scripts.bench.benchmarks.initial_conditions --epsilons 0 1e-6 1e-4 1e-2 1e-1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.bench.infra import MeasureResult, measure, power_law_exponent
from scripts.bench.plotting import STYLES, SeriesData, log_log_plot, save_fig
from scripts.scenes.contact_objects import (
    DT_OUTER,
    build_model,
    build_model_perturbed,
    build_model_randomized,
    make_solver,
)

IC_MODES = ["identical", "randomized"]

# Colors for epsilon sweep (perceptually ordered, light to dark).
_EPSILON_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]


def _measure_ic_mode(
    mode: str, n: int, steps: int, warmup: int, trials: int = 1,
    epsilon: float = 1e-4,
) -> MeasureResult:
    """Measure CENIC at one N with one IC mode."""
    if mode == "identical":
        build_fn = build_model
    elif mode == "randomized":
        build_fn = build_model_randomized
    else:
        def build_fn(n_worlds):
            return build_model_perturbed(n_worlds, epsilon=epsilon)

    def _single_trial() -> MeasureResult:
        solver_cache = {}

        def _get_solver(model, cache):
            key = id(model)
            if key not in cache:
                cache[key] = make_solver(model)
            return cache[key]

        def step_fn(model, s0, s1, ctrl):
            solver = _get_solver(model, solver_cache)
            return solver.step_dt(DT_OUTER, s0, s1, ctrl)

        def get_k():
            solver = next(iter(solver_cache.values()))
            return int(solver.iteration_count.numpy()[0])

        return measure(build_fn, step_fn, n, steps, warmup, get_k=get_k)

    best = _single_trial()
    for _ in range(trials - 1):
        result = _single_trial()
        if result.median < best.median:
            best = result
    return best


def _collect_error_trace(mode: str, steps: int, epsilon: float = 1e-4) -> dict:
    """Run N=1 and record accepted error + dt at each outer step."""
    if mode == "identical":
        model = build_model(1)
    elif mode == "randomized":
        model = build_model_randomized(1)
    else:
        model = build_model_perturbed(1, epsilon=epsilon)
    solver = make_solver(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()

    sim_times, errors, dts = [], [], []
    for _ in range(steps):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        sim_times.append(float(solver.sim_time.numpy()[0]))
        errors.append(float(solver.last_error.numpy()[0]))
        dts.append(float(solver.dt.numpy()[0]))

    return {"sim_times": sim_times, "errors": errors, "dts": dts, "tol": solver._tol}


def _run_epsilon_sweep(
    epsilons: list[float], n: int, steps: int, warmup: int, trials: int = 1,
) -> dict:
    """Sweep epsilon at fixed N, measuring K and timing for each."""
    sweep: dict = {"epsilons": epsilons, "n": n, "results": []}

    for eps in epsilons:
        mode = "identical" if eps == 0 else "perturbed"
        result = _measure_ic_mode(mode, n, steps, warmup, trials, epsilon=eps)
        entry = {
            "epsilon": eps,
            "median": result.median,
            "p25": result.p25,
            "p75": result.p75,
            "k_mean": result.k_mean,
            "k_max": result.k_max,
            "k_p25": result.k_p25,
            "k_p75": result.k_p75,
            "per_iter_median": result.per_iter_median,
        }
        sweep["results"].append(entry)
        label = "eps=0" if eps == 0 else f"eps={eps:.0e}"
        print(
            f"  N={n:>5}  {label:>12}  "
            f"median={result.median * 1e3:7.2f} ms  "
            f"per_iter={result.per_iter_median * 1e3:7.2f} ms  "
            f"K_mean={result.k_mean:.1f}  K_max={result.k_max}",
            flush=True,
        )

    return sweep


def run(
    ns: list[int], steps: int, warmup: int, trials: int = 1,
    epsilons: list[float] | None = None,
    epsilon_sweep_n: int = 64,
) -> dict:
    """Run IC experiments: N-scaling comparison + epsilon sweep."""
    if epsilons is None:
        epsilons = [0, 1e-6, 1e-4, 1e-2, 1e-1]

    data: dict = {
        "ns": ns, "steps": steps, "warmup": warmup,
        "trials": trials, "modes": {},
    }

    # Experiment 1: N-scaling comparison (identical vs perturbed at default epsilon).
    for mode in IC_MODES:
        mode_data: dict = {
            "medians": [], "p25": [], "p75": [],
            "k_means": [], "k_maxes": [], "k_p25s": [], "k_p75s": [],
            "per_iter_medians": [],
        }
        for n in ns:
            result = _measure_ic_mode(mode, n, steps, warmup, trials)
            mode_data["medians"].append(result.median)
            mode_data["p25"].append(result.p25)
            mode_data["p75"].append(result.p75)
            mode_data["k_means"].append(result.k_mean)
            mode_data["k_maxes"].append(result.k_max)
            mode_data["k_p25s"].append(result.k_p25)
            mode_data["k_p75s"].append(result.k_p75)
            mode_data["per_iter_medians"].append(result.per_iter_median)
            print(
                f"  N={n:>5}  {mode:>12}  "
                f"median={result.median * 1e3:7.2f} ms  "
                f"per_iter={result.per_iter_median * 1e3:7.2f} ms  "
                f"K_mean={result.k_mean:.1f}  K_max={result.k_max}",
                flush=True,
            )
        data["modes"][mode] = mode_data

    data["exponents"] = {}
    for mode in IC_MODES:
        data["exponents"][mode] = power_law_exponent(
            ns, data["modes"][mode]["medians"],
        )

    # Experiment 2: Epsilon sweep at fixed N.
    print(f"\n  Epsilon sweep at N={epsilon_sweep_n}:", flush=True)
    data["epsilon_sweep"] = _run_epsilon_sweep(
        epsilons, epsilon_sweep_n, steps, warmup, trials,
    )

    # Error traces (N=1, both modes).
    print("  Collecting error traces (N=1)...", flush=True)
    data["error_traces"] = {}
    for mode in IC_MODES:
        data["error_traces"][mode] = _collect_error_trace(mode, steps + warmup)

    return data


def plot(data: dict, out_dir: Path) -> None:
    """Generate comparison plots from results dict."""
    ns = data["ns"]
    modes_data = data["modes"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Wall time per outer step vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    series = {}
    for mode in IC_MODES:
        md = modes_data[mode]
        series[mode] = SeriesData(
            medians=[m * 1e3 for m in md["medians"]],
            p25=[m * 1e3 for m in md["p25"]],
            p75=[m * 1e3 for m in md["p75"]],
        )
    log_log_plot(
        ax, ns, series,
        ylabel="Wall time per outer step [ms]",
        title=f"IC comparison: wall time vs N  (DT_outer={DT_OUTER * 1e3:.0f} ms, tol=1e-3)",
    )
    save_fig(fig, out_dir / "ic_wall_time.png")

    # Plot 2: Per-iteration cost vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    per_iter_series = {}
    for mode in IC_MODES:
        md = modes_data[mode]
        per_iter_series[mode] = SeriesData(
            medians=[p * 1e3 for p in md["per_iter_medians"]],
        )
    log_log_plot(
        ax, ns, per_iter_series,
        ylabel="Wall time per iteration [ms]",
        title="IC comparison: per-iteration GPU cost vs N",
    )
    save_fig(fig, out_dir / "ic_per_iter.png")

    # Plot 3: Iteration count K vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in IC_MODES:
        style = STYLES[mode]
        md = modes_data[mode]
        ax.plot(
            ns, md["k_means"], color=style.color, marker=style.marker,
            ls="-", lw=2, ms=5, label=f'{style.label}  $K_{{mean}}$',
        )
        if "k_p25s" in md and "k_p75s" in md:
            ax.fill_between(ns, md["k_p25s"], md["k_p75s"],
                            color=style.color, alpha=0.10)
        ax.plot(
            ns, md["k_maxes"], color=style.color, marker=style.marker,
            ls=":", lw=1, ms=3, alpha=0.5, label=f'{style.label}  $K_{{max}}$',
        )
    ax.axhline(1, color="grey", ls=":", lw=1, label="K = 1 (ideal)")
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Iterations per step_dt call", fontsize=11)
    ax.set_title("IC comparison: iteration count K vs N  (tol=1e-3)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir / "ic_iterations.png")

    # Plot 4: Cost per world vs N (amortization)
    fig, ax = plt.subplots(figsize=(10, 6))
    amort_series = {}
    for mode in IC_MODES:
        md = modes_data[mode]
        amort = [m / n_val * 1e3 for m, n_val in zip(md["medians"], ns)]
        amort_series[mode] = SeriesData(medians=amort)
    log_log_plot(
        ax, ns, amort_series,
        ylabel="Wall time per world per outer step [ms]",
        title="IC comparison: GPU amortization cost per world vs N",
        show_exponents=False,
    )
    save_fig(fig, out_dir / "ic_amortization.png")

    # Plot 5: Error vs simulation time (N=1, both modes)
    if "error_traces" in data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode in IC_MODES:
            trace = data["error_traces"][mode]
            style = STYLES[mode]
            ax.plot(
                trace["sim_times"], trace["errors"],
                color=style.color, lw=1, alpha=0.8,
                label=f'{style.label} error',
            )
        tol = data["error_traces"][IC_MODES[0]]["tol"]
        ax.axhline(tol, color="red", ls="--", lw=1.5, label=f'tol = {tol:.0e}')
        ax.set_xlabel("Simulation time [s]", fontsize=11)
        ax.set_ylabel("Error (inf-norm on q)", fontsize=11)
        ax.set_title(f"IC comparison: error vs time  (N=1, tol={tol:.0e})", fontsize=11)
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        save_fig(fig, out_dir / "ic_error_trace.png")

    # Plot 6+7: Epsilon sweep (K and wall time vs epsilon).
    if "epsilon_sweep" in data:
        sweep = data["epsilon_sweep"]
        epsilons = sweep["epsilons"]
        results = sweep["results"]
        n_sweep = sweep["n"]

        # Use non-zero epsilons for log x-axis; mark eps=0 at a small position.
        eps_plot = []
        for eps in epsilons:
            eps_plot.append(eps if eps > 0 else min(e for e in epsilons if e > 0) / 10)

        k_means = [r["k_mean"] for r in results]
        k_maxes = [r["k_max"] for r in results]
        k_p25s = [r["k_p25"] for r in results]
        k_p75s = [r["k_p75"] for r in results]
        medians_ms = [r["median"] * 1e3 for r in results]
        per_iter_ms = [r["per_iter_median"] * 1e3 for r in results]

        # Plot 6: K vs epsilon
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eps_plot, k_means, color="#1f77b4", marker="o", ls="-", lw=2, ms=6,
                label="$K_{mean}$")
        ax.fill_between(eps_plot, k_p25s, k_p75s, color="#1f77b4", alpha=0.10)
        ax.plot(eps_plot, k_maxes, color="#1f77b4", marker="o", ls=":", lw=1, ms=3,
                alpha=0.5, label="$K_{max}$")
        ax.axhline(1, color="grey", ls=":", lw=1, label="K = 1 (ideal)")
        ax.set_xlabel("Epsilon (IC perturbation) [m]", fontsize=11)
        ax.set_ylabel("Iterations per step_dt call", fontsize=11)
        ax.set_title(f"Epsilon sweep: iteration count K  (N={n_sweep}, tol=1e-3)", fontsize=11)
        ax.set_xscale("log")
        # Label eps=0 tick if present.
        if epsilons[0] == 0:
            ax.annotate("eps=0", xy=(eps_plot[0], k_means[0]),
                        xytext=(eps_plot[0] * 3, k_means[0] * 1.15),
                        fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
        save_fig(fig, out_dir / "ic_epsilon_k.png")

        # Plot 7: Wall time and per-iter cost vs epsilon
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eps_plot, medians_ms, color="#1f77b4", marker="o", ls="-", lw=2, ms=6,
                label="Wall time per outer step")
        ax.plot(eps_plot, per_iter_ms, color="#ff7f0e", marker="D", ls="--", lw=2, ms=5,
                label="Per-iteration cost")
        ax.set_xlabel("Epsilon (IC perturbation) [m]", fontsize=11)
        ax.set_ylabel("Time [ms]", fontsize=11)
        ax.set_title(f"Epsilon sweep: cost vs perturbation  (N={n_sweep}, tol=1e-3)", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if epsilons[0] == 0:
            ax.annotate("eps=0", xy=(eps_plot[0], medians_ms[0]),
                        xytext=(eps_plot[0] * 3, medians_ms[0] * 1.3),
                        fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
        save_fig(fig, out_dir / "ic_epsilon_cost.png")

    # Summary table.
    print(f"\n{'=' * 78}")
    print("INITIAL CONDITIONS SUMMARY")
    print(f"{'=' * 78}")
    hdr = (
        f"{'mode':>12}  {'exponent':>10}  {'N=1':>10}  "
        f"{'N=' + str(ns[-1]):>10}  {'ratio':>6}  {'K_mean':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for mode in IC_MODES:
        md = modes_data[mode]
        exp = data["exponents"][mode]
        k = md["k_means"][-1]
        t1 = md["medians"][0] * 1e3
        tN = md["medians"][-1] * 1e3
        print(
            f"{mode:>12}  N^{exp:<7.3f}   {t1:9.2f}  {tN:10.2f}  "
            f"{tN / t1:5.1f}x  {k:6.1f}  (wall time)"
        )

    if "epsilon_sweep" in data:
        sweep = data["epsilon_sweep"]
        print(f"\nEPSILON SWEEP (N={sweep['n']})")
        print(f"{'epsilon':>12}  {'wall_ms':>10}  {'per_iter_ms':>12}  {'K_mean':>8}  {'K_max':>6}")
        for r in sweep["results"]:
            eps_str = "0" if r["epsilon"] == 0 else f"{r['epsilon']:.0e}"
            print(
                f"{eps_str:>12}  {r['median'] * 1e3:10.2f}  "
                f"{r['per_iter_median'] * 1e3:12.2f}  "
                f"{r['k_mean']:8.1f}  {r['k_max']:6}"
            )


def main():
    parser = argparse.ArgumentParser(description="Initial conditions benchmark")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0, 1e-6, 1e-4, 1e-2, 1e-1],
                        help="Epsilon values for IC perturbation sweep")
    parser.add_argument("--epsilon-sweep-n", type=int, default=64,
                        help="Fixed N for epsilon sweep experiment")
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run(
        sorted(args.ns), args.steps, args.warmup, args.trials,
        epsilons=sorted(args.epsilons),
        epsilon_sweep_n=args.epsilon_sweep_n,
    )

    with open(out_dir / "initial_conditions.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'initial_conditions.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
