# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""N-scaling benchmark: wall time vs world count for graph/loop/fixed/manual.

Standalone:
    uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 16 64 256

Produces 4 plots: wall_time, iterations, per_iter, amortization.
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

MODES = ["graph", "loop", "fixed", "manual"]


def _step_graph(model, s0, s1, ctrl, _state={}):
    """step_dt via capture_while. Solver created on first call."""
    key = id(model)
    if key not in _state:
        _state[key] = make_solver(model)
    solver = _state[key]
    s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    return s0, s1


def _step_loop(model, s0, s1, ctrl, _state={}):
    """step_dt_loop via Python loop. Solver created on first call."""
    key = id(model)
    if key not in _state:
        _state[key] = make_solver(model)
    solver = _state[key]
    s0, s1 = solver.step_dt_loop(DT_OUTER, s0, s1, ctrl)
    return s0, s1


def _step_fixed(model, s0, s1, ctrl, _state={}):
    """Fixed-step SolverMuJoCo. Solver + contacts created on first call."""
    key = id(model)
    if key not in _state:
        solver = make_fixed_solver(model)
        contacts = model.contacts()
        _state[key] = (solver, contacts)
    solver, contacts = _state[key]
    s1 = solver.step(s0, s1, ctrl, contacts, DT_OUTER)
    return s1, s0


def _get_solver(model, _state, factory):
    """Get or create solver for a model, caching by model id."""
    key = id(model)
    if key not in _state:
        _state[key] = factory(model)
    return _state[key]


def _measure_mode(mode: str, n: int, steps: int, warmup: int) -> MeasureResult:
    """Measure one mode at one N with a completely fresh solver."""
    if mode == "graph":
        solver_cache = {}
        def step_fn(model, s0, s1, ctrl):
            solver = _get_solver(model, solver_cache, make_solver)
            return solver.step_dt(DT_OUTER, s0, s1, ctrl)
        def get_k():
            solver = next(iter(solver_cache.values()))
            return int(solver.iteration_count.numpy()[0])
        return measure(build_model, step_fn, n, steps, warmup, get_k=get_k)

    elif mode == "loop":
        solver_cache = {}
        def step_fn(model, s0, s1, ctrl):
            solver = _get_solver(model, solver_cache, make_solver)
            return solver.step_dt_loop(DT_OUTER, s0, s1, ctrl)
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

    elif mode == "manual":
        from scripts.bench.benchmarks._manual_step import measure_manual
        return measure_manual(n, steps, warmup)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run(ns: list[int], steps: int, warmup: int) -> dict:
    """Run all modes at all N values. Returns JSON-serializable dict."""
    data: dict = {"ns": ns, "steps": steps, "warmup": warmup, "modes": {}}

    for mode in MODES:
        mode_data: dict = {
            "medians": [], "p25": [], "p75": [],
            "k_means": [], "k_maxes": [],
        }
        for n in ns:
            result = _measure_mode(mode, n, steps, warmup)
            mode_data["medians"].append(result.median)
            mode_data["p25"].append(result.p25)
            mode_data["p75"].append(result.p75)
            mode_data["k_means"].append(result.k_mean)
            mode_data["k_maxes"].append(result.k_max)
            print(
                f"  N={n:>5}  {mode:>7}  "
                f"median={result.median * 1e3:7.2f} ms  "
                f"K_mean={result.k_mean:.1f}  K_max={result.k_max}",
                flush=True,
            )
        data["modes"][mode] = mode_data

    # Compute exponents.
    data["exponents"] = {
        mode: power_law_exponent(ns, data["modes"][mode]["medians"])
        for mode in MODES
    }
    return data


def plot(data: dict, out_dir: Path) -> None:
    """Generate 4 scaling plots from results dict."""
    ns = data["ns"]
    modes_data = data["modes"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Wall time per outer step vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    series = {}
    for mode in MODES:
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

    # Plot 2: Iteration count K vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in ["graph", "loop"]:
        style = STYLES[mode]
        md = modes_data[mode]
        ax.plot(
            ns, md["k_means"], color=style.color, marker=style.marker,
            ls="-", lw=2, ms=5, label=f'{style.label}  $K_{{mean}}$',
        )
        ax.plot(
            ns, md["k_maxes"], color=style.color, marker=style.marker,
            ls="--", lw=1.5, ms=4, alpha=0.6,
            label=f'{style.label}  $K_{{max}}$',
        )
    ax.axhline(1, color="grey", ls=":", lw=1, label="K = 1 (ideal)")
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Iterations per step_dt call", fontsize=11)
    ax.set_title("Adaptive iteration count K vs N  (tol=1e-3)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir / "scaling_iterations.png")

    # Plot 3: Per-iteration wall time vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    per_iter_series = {}
    for mode in MODES:
        md = modes_data[mode]
        per_iter = [m / max(k, 1) for m, k in zip(md["medians"], md["k_means"])]
        per_iter_series[mode] = SeriesData(medians=[p * 1e3 for p in per_iter])
    log_log_plot(
        ax, ns, per_iter_series,
        ylabel="Wall time per iteration [ms]",
        title="Per-iteration wall time vs N  (wall_time / K)",
        show_iqr=False,
    )
    save_fig(fig, out_dir / "scaling_per_iter.png")

    # Plot 4: Cost per world vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    amort_series = {}
    for mode in MODES:
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

    # Summary table
    print(f"\n{'=' * 72}")
    print("SCALING SUMMARY")
    print(f"{'=' * 72}")
    hdr = f"{'mode':>10}  {'exponent':>8}  {'N=1 (ms)':>10}  {'N=' + str(ns[-1]) + ' (ms)':>12}  {'ratio':>6}  {'K_mean':>6}"
    print(hdr)
    print("-" * len(hdr))
    for mode in MODES:
        md = modes_data[mode]
        t1 = md["medians"][0] * 1e3
        tN = md["medians"][-1] * 1e3
        exp = data["exponents"][mode]
        k = md["k_means"][-1]
        print(
            f"{mode:>10}  N^{exp:<6.3f}  {t1:10.2f}  {tN:12.2f}  "
            f"{tN / t1:5.1f}x  {k:6.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="N-scaling benchmark")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run(sorted(args.ns), args.steps, args.warmup)

    with open(out_dir / "scaling.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'scaling.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
