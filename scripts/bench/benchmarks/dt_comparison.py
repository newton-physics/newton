# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""dt-mode comparison: per-world vs global adaptive vs fixed-step.

Standalone:
    uv run python -m scripts.bench.benchmarks.dt_comparison --ns 1 4 16 64 256

Produces 4 plots:
  - dt_comparison_wall_time: wall time per outer step vs N (all 3 modes)
  - dt_comparison_iterations: iteration count K vs N (per_world vs global)
  - dt_comparison_amortization: cost per world vs N (all 3 modes)
  - dt_comparison_speedup: speedup of per_world over global/fixed vs N
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from scripts.bench.infra import MeasureResult, measure, power_law_exponent
from scripts.bench.plotting import SeriesData, log_log_plot, save_fig
from scripts.scenes.contact_objects import DT_OUTER, build_model_randomized, make_fixed_solver, make_solver

MODES = ["cenic_per_world", "cenic_global", "fixed"]

_STYLES = {
    "cenic_per_world": {"color": "#1f77b4", "marker": "o", "ls": "-", "label": "CENIC per-world dt"},
    "cenic_global": {"color": "#ff7f0e", "marker": "s", "ls": "-", "label": "CENIC global dt"},
    "fixed": {"color": "#2ca02c", "marker": "D", "ls": "--", "label": "Fixed-step (dt=10 ms)"},
}


def _measure_mode(mode: str, n: int, steps: int, warmup: int) -> MeasureResult:
    if mode == "cenic_per_world":
        solver_cache = {}

        def step_fn(model, s0, s1, ctrl):
            key = id(model)
            if key not in solver_cache:
                solver_cache[key] = make_solver(model, dt_mode="per_world")
            return solver_cache[key].step_dt(DT_OUTER, s0, s1, ctrl)

        def get_k():
            return int(next(iter(solver_cache.values())).iteration_count.numpy()[0])

        return measure(build_model_randomized, step_fn, n, steps, warmup, get_k=get_k)

    elif mode == "cenic_global":
        solver_cache = {}

        def step_fn(model, s0, s1, ctrl):
            key = id(model)
            if key not in solver_cache:
                solver_cache[key] = make_solver(model, dt_mode="global")
            return solver_cache[key].step_dt(DT_OUTER, s0, s1, ctrl)

        def get_k():
            return int(next(iter(solver_cache.values())).iteration_count.numpy()[0])

        return measure(build_model_randomized, step_fn, n, steps, warmup, get_k=get_k)

    elif mode == "fixed":
        state_cache = {}

        def step_fn(model, s0, s1, ctrl):
            key = id(model)
            if key not in state_cache:
                state_cache[key] = (make_fixed_solver(model), model.contacts())
            solver, contacts = state_cache[key]
            s1 = solver.step(s0, s1, ctrl, contacts, DT_OUTER)
            return s1, s0

        return measure(build_model_randomized, step_fn, n, steps, warmup)

    raise ValueError(f"Unknown mode: {mode}")


def run(ns: list[int], steps: int, warmup: int) -> dict:
    data: dict = {"ns": ns, "steps": steps, "warmup": warmup, "modes": {}}

    for mode in MODES:
        mode_data: dict = {
            "medians": [], "p25": [], "p75": [],
            "k_means": [], "k_maxes": [], "k_p25s": [], "k_p75s": [],
            "per_iter_medians": [],
        }
        for n in ns:
            result = _measure_mode(mode, n, steps, warmup)
            mode_data["medians"].append(result.median)
            mode_data["p25"].append(result.p25)
            mode_data["p75"].append(result.p75)
            mode_data["k_means"].append(result.k_mean)
            mode_data["k_maxes"].append(result.k_max)
            mode_data["k_p25s"].append(result.k_p25)
            mode_data["k_p75s"].append(result.k_p75)
            mode_data["per_iter_medians"].append(result.per_iter_median)
            print(
                f"  N={n:>5}  {mode:>18}  "
                f"median={result.median * 1e3:7.2f} ms  "
                f"K_mean={result.k_mean:.1f}  K_max={result.k_max}",
                flush=True,
            )
        data["modes"][mode] = mode_data

    data["exponents"] = {}
    for mode in MODES:
        vals = data["modes"][mode]["medians"]
        data["exponents"][mode] = power_law_exponent(ns, vals)

    return data


def plot(data: dict, out_dir: Path) -> None:
    ns = np.array(data["ns"])
    modes_data = data["modes"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def _style(mode):
        return _STYLES[mode]

    # --- Plot 1: Wall time vs N (all 3 modes) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in MODES:
        if mode not in modes_data:
            continue
        md = modes_data[mode]
        s = _style(mode)
        exp = data["exponents"].get(mode, float("nan"))
        label = f'{s["label"]}  $N^{{{exp:.2f}}}$'
        meds = [m * 1e3 for m in md["medians"]]
        ax.plot(ns, meds, color=s["color"], marker=s["marker"], ls=s["ls"],
                lw=2, ms=5, label=label)
        if md["p25"] and md["p75"]:
            ax.fill_between(
                ns,
                [m * 1e3 for m in md["p25"]],
                [m * 1e3 for m in md["p75"]],
                color=s["color"], alpha=0.10,
            )
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Wall time per outer step [ms]", fontsize=11)
    ax.set_title(
        f"Wall time vs N: per-world dt vs global dt vs fixed-step"
        f"  (DT_outer={DT_OUTER * 1e3:.0f} ms)",
        fontsize=11,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir / "dt_comparison_wall_time.png")

    # --- Plot 2: Iteration count K vs N (per_world vs global) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in ["cenic_per_world", "cenic_global"]:
        if mode not in modes_data:
            continue
        md = modes_data[mode]
        s = _style(mode)
        ax.plot(ns, md["k_means"], color=s["color"], marker=s["marker"],
                ls="-", lw=2, ms=5, label=f'{s["label"]}  $K_{{mean}}$')
        if md.get("k_p25s") and md.get("k_p75s"):
            ax.fill_between(ns, md["k_p25s"], md["k_p75s"],
                            color=s["color"], alpha=0.10)
        ax.plot(ns, md["k_maxes"], color=s["color"], marker=s["marker"],
                ls=":", lw=1, ms=3, alpha=0.5, label=f'{s["label"]}  $K_{{max}}$')
    ax.axhline(1, color="grey", ls=":", lw=1, label="K = 1 (ideal)")
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Iterations per step_dt call", fontsize=11)
    ax.set_title("Adaptive iteration count: per-world dt vs global dt", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir / "dt_comparison_iterations.png")

    # --- Plot 3: Cost per world vs N (amortization) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in MODES:
        if mode not in modes_data:
            continue
        md = modes_data[mode]
        s = _style(mode)
        amort = [m / n_val * 1e3 for m, n_val in zip(md["medians"], ns)]
        ax.plot(ns, amort, color=s["color"], marker=s["marker"], ls=s["ls"],
                lw=2, ms=5, label=s["label"])
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Wall time per world per outer step [ms]", fontsize=11)
    ax.set_title("GPU amortization: cost per world vs N", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir / "dt_comparison_amortization.png")

    # --- Plot 4: Speedup of adaptive over fixed, and per-world over global ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: adaptive (per_world) vs fixed
    ax = axes[0]
    if "cenic_per_world" in modes_data and "fixed" in modes_data:
        pw = modes_data["cenic_per_world"]["medians"]
        fx = modes_data["fixed"]["medians"]
        ratio = [f / p for p, f in zip(pw, fx)]
        ax.plot(ns, ratio, color=_STYLES["cenic_per_world"]["color"],
                marker="o", ls="-", lw=2, ms=5)
        ax.axhline(1, color="grey", ls=":", lw=1)
        ax.set_xlabel("N worlds", fontsize=11)
        ax.set_ylabel("Speedup (fixed / CENIC per-world)", fontsize=11)
        ax.set_title("Fixed-step vs CENIC per-world dt", fontsize=11)
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)

    # Right: per-world vs global
    ax = axes[1]
    if "cenic_per_world" in modes_data and "cenic_global" in modes_data:
        pw = modes_data["cenic_per_world"]["medians"]
        gl = modes_data["cenic_global"]["medians"]
        ratio = [g / p for p, g in zip(pw, gl)]
        ax.plot(ns, ratio, color=_STYLES["cenic_per_world"]["color"],
                marker="o", ls="-", lw=2, ms=5)
        ax.axhline(1, color="grey", ls=":", lw=1)
        ax.set_xlabel("N worlds", fontsize=11)
        ax.set_ylabel("Speedup (global / per-world)", fontsize=11)
        ax.set_title("Per-world dt vs global dt", fontsize=11)
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, out_dir / "dt_comparison_speedup.png")

    # Summary table
    print(f"\n{'=' * 78}")
    print("DT COMPARISON SUMMARY")
    print(f"{'=' * 78}")
    hdr = f"{'mode':>18}  {'N^exp':>8}  {'N=1 [ms]':>10}  {'N=' + str(ns[-1]):>10}  {'K_mean':>6}"
    print(hdr)
    print("-" * len(hdr))
    for mode in MODES:
        md = modes_data[mode]
        exp = data["exponents"][mode]
        t1 = md["medians"][0] * 1e3
        tN = md["medians"][-1] * 1e3
        k = md["k_means"][-1]
        print(f"{mode:>18}  N^{exp:<5.3f}   {t1:9.2f}  {tN:10.2f}  {k:6.1f}")


def main():
    parser = argparse.ArgumentParser(description="dt-mode comparison benchmark")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run(sorted(args.ns), args.steps, args.warmup)

    with open(out_dir / "dt_comparison.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'dt_comparison.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
