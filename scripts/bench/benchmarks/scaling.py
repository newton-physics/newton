# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""N-scaling benchmark: wall time vs world count for cenic/fixed/single_iter.

Standalone:
    uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 16 64 256

Produces 5 plots: wall_time, per_iter, iterations, amortization, error_trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import warp as wp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.bench.infra import MeasureResult, measure, power_law_exponent
from scripts.bench.plotting import STYLES, SeriesData, log_log_plot, save_fig
from scripts.scenes.contact_objects import DT_OUTER, build_model, make_fixed_solver, make_solver

MODES = ["cenic", "fixed", "single_iter"]


def _get_solver(model, _state, factory):
    """Get or create solver for a model, caching by model id."""
    key = id(model)
    if key not in _state:
        _state[key] = factory(model)
    return _state[key]


def _measure_mode(mode: str, n: int, steps: int, warmup: int, trials: int = 1) -> MeasureResult:
    """Measure one mode at one N with a completely fresh solver.

    When trials > 1, runs the full measurement multiple times and returns
    the trial with the lowest median (least system interference).
    """
    def _single_trial() -> MeasureResult:
        if mode == "cenic":
            solver_cache = {}
            def step_fn(model, s0, s1, ctrl):
                solver = _get_solver(model, solver_cache, make_solver)
                return solver.step_dt(DT_OUTER, s0, s1, ctrl)
            def get_k():
                solver = next(iter(solver_cache.values()))
                return int(solver.iteration_count.numpy()[0])
            return measure(build_model, step_fn, n, steps, warmup, get_k=get_k)

        elif mode == "fixed":
            state_cache = {}
            def step_fn(model, s0, s1, ctrl):
                key = id(model)
                if key not in state_cache:
                    state_cache[key] = (make_fixed_solver(model), model.contacts())
                solver, contacts = state_cache[key]
                s1 = solver.step(s0, s1, ctrl, contacts, DT_OUTER)
                return s1, s0
            return measure(build_model, step_fn, n, steps, warmup)

        elif mode == "single_iter":
            from scripts.bench.benchmarks._single_iter import measure_single_iter
            return measure_single_iter(n, steps, warmup)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    best = _single_trial()
    for _ in range(trials - 1):
        result = _single_trial()
        if result.median < best.median:
            best = result
    return best


def _collect_error_trace(steps: int) -> dict:
    """Run N=1 cenic mode and record accepted error + dt at each outer step."""
    model = build_model(1)
    solver = make_solver(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()

    sim_times, errors, dts = [], [], []
    for _ in range(steps):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        sim_times.append(float(solver.sim_time.numpy()[0]))
        errors.append(float(solver.last_error.numpy()[0]))
        dts.append(float(solver.dt.numpy()[0]))

    return {
        "sim_times": sim_times,
        "errors": errors,
        "dts": dts,
        "tol": solver._tol,
    }


def run(ns: list[int], steps: int, warmup: int, trials: int = 1) -> dict:
    """Run all modes at all N values. Returns JSON-serializable dict."""
    data: dict = {"ns": ns, "steps": steps, "warmup": warmup, "trials": trials, "modes": {}}

    for mode in MODES:
        mode_data: dict = {
            "medians": [], "p25": [], "p75": [],
            "k_means": [], "k_maxes": [], "k_p25s": [], "k_p75s": [],
            "per_iter_medians": [],
        }
        for n in ns:
            result = _measure_mode(mode, n, steps, warmup, trials)
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

    # Exponents: per-iteration cost for single_iter and fixed,
    # wall time for cenic (per_iter is biased by K correlation with N).
    data["exponents"] = {}
    for mode in MODES:
        if mode == "cenic":
            data["exponents"][mode] = power_law_exponent(
                ns, data["modes"][mode]["medians"])
        else:
            data["exponents"][mode] = power_law_exponent(
                ns, data["modes"][mode]["per_iter_medians"])

    # Error trace (N=1, cenic mode).
    print("  Collecting error trace (N=1)...", flush=True)
    data["error_trace"] = _collect_error_trace(steps + warmup)

    return data


def plot(data: dict, out_dir: Path) -> None:
    """Generate 5 scaling plots from results dict."""
    ns = data["ns"]
    modes_data = data["modes"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Wall time per outer step vs N (cenic + fixed)
    fig, ax = plt.subplots(figsize=(10, 6))
    wall_modes = ["cenic", "fixed"]
    series = {}
    for mode in wall_modes:
        if mode not in modes_data:
            continue
        md = modes_data[mode]
        series[mode] = SeriesData(
            medians=[m * 1e3 for m in md["medians"]],
            p25=[m * 1e3 for m in md["p25"]],
            p75=[m * 1e3 for m in md["p75"]],
        )
    log_log_plot(
        ax, ns, series,
        ylabel="Wall time per outer step [ms]",
        title=f"Scaling: wall time vs N  (DT_outer={DT_OUTER * 1e3:.0f} ms, tol=1e-3)",
    )
    save_fig(fig, out_dir / "scaling_wall_time.png")

    # Plot 2: Per-iteration cost vs N (cenic + single_iter + fixed)
    fig, ax = plt.subplots(figsize=(10, 6))
    iter_modes = ["cenic", "single_iter", "fixed"]
    per_iter_series = {}
    for mode in iter_modes:
        if mode not in modes_data:
            continue
        md = modes_data[mode]
        per_iter_series[mode] = SeriesData(
            medians=[p * 1e3 for p in md["per_iter_medians"]],
            p25=[p * 1e3 for p in md["p25"]] if mode == "single_iter" else None,
            p75=[p * 1e3 for p in md["p75"]] if mode == "single_iter" else None,
        )
    log_log_plot(
        ax, ns, per_iter_series,
        ylabel="Wall time per iteration [ms]",
        title="Per-iteration GPU cost vs N  (single iteration, sync-to-sync)",
    )
    save_fig(fig, out_dir / "scaling_per_iter.png")

    # Plot 3: Iteration count K vs N (cenic only)
    if "cenic" in modes_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        style = STYLES["cenic"]
        md = modes_data["cenic"]
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
        ax.set_title("Adaptive iteration count K vs N  (tol=1e-3)", fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
        save_fig(fig, out_dir / "scaling_iterations.png")

    # Plot 4: Cost per world vs N (cenic + fixed)
    fig, ax = plt.subplots(figsize=(10, 6))
    amort_series = {}
    for mode in wall_modes:
        if mode not in modes_data:
            continue
        md = modes_data[mode]
        amort = [m / n_val * 1e3 for m, n_val in zip(md["medians"], ns)]
        amort_series[mode] = SeriesData(medians=amort)
    log_log_plot(
        ax, ns, amort_series,
        ylabel="Wall time per world per outer step [ms]",
        title="GPU amortization: cost per world vs N",
        show_exponents=False,
    )
    save_fig(fig, out_dir / "scaling_amortization.png")

    # Plot 5: Error vs simulation time (N=1, cenic)
    if "error_trace" in data:
        trace = data["error_trace"]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            trace["sim_times"], trace["errors"],
            color="#1f77b4", lw=1, alpha=0.8,
            label="accepted error (inf-norm on q)",
        )
        ax.axhline(
            trace["tol"], color="red", ls="--", lw=1.5,
            label=f'tol = {trace["tol"]:.0e}',
        )
        ax.set_xlabel("Simulation time [s]", fontsize=11)
        ax.set_ylabel("Error (inf-norm on q)", fontsize=11)
        ax.set_title(
            f"Step doubling error vs simulation time  (N=1, tol={trace['tol']:.0e})",
            fontsize=11,
        )
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        save_fig(fig, out_dir / "scaling_error_trace.png")

    # Summary table
    print(f"\n{'=' * 78}")
    print("SCALING SUMMARY")
    print(f"{'=' * 78}")
    hdr = (
        f"{'mode':>12}  {'exponent':>10}  {'N=1':>10}  "
        f"{'N=' + str(ns[-1]):>10}  {'ratio':>6}  {'K_mean':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for mode in MODES:
        md = modes_data[mode]
        exp = data["exponents"][mode]
        k = md["k_means"][-1]
        if mode == "cenic":
            t1 = md["medians"][0] * 1e3
            tN = md["medians"][-1] * 1e3
            label = "wall time"
        else:
            t1 = md["per_iter_medians"][0] * 1e3
            tN = md["per_iter_medians"][-1] * 1e3
            label = "per iter"
        print(
            f"{mode:>12}  N^{exp:<7.3f}   {t1:9.2f}  {tN:10.2f}  "
            f"{tN / t1:5.1f}x  {k:6.1f}  ({label})"
        )


def main():
    parser = argparse.ArgumentParser(description="N-scaling benchmark")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1,
                        help="Repeat each (N, mode) measurement and keep the best median.")
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run(sorted(args.ns), args.steps, args.warmup, args.trials)

    with open(out_dir / "scaling.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'scaling.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
