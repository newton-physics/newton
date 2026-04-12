# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Initial conditions benchmark: randomized chaotic ICs, tolerance × N sweep.

Each world starts with a fully randomized object layout (fixed seed, so the
first K worlds are the same chaotic layouts across runs).  Sweeps world count
N at several tolerances and reports wall time, iteration count K, per-iteration
GPU cost, and amortization.

Standalone:
    uv run python -m scripts.bench.benchmarks.initial_conditions
    uv run python -m scripts.bench.benchmarks.initial_conditions --ns 1 4 16 64 256 --tols 1e-2 1e-3 1e-4
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
from scripts.bench.plotting import save_fig
from scripts.scenes.contact_objects import DT_OUTER, build_model_randomized, make_solver

SEED = 42

# Perceptually ordered colors for ascending tolerance (tight → loose).
_TOL_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]


def _tol_label(tol: float) -> str:
    return f"tol={tol:.0e}"


def _measure_tol_n(
    tol: float, n: int, steps: int, warmup: int, trials: int = 1,
    seed: int = SEED,
) -> MeasureResult:
    """Measure CENIC on randomized ICs at one (tol, N) with best-of-`trials`."""

    def build_fn(n_worlds):
        return build_model_randomized(n_worlds, seed=seed)

    def _single_trial() -> MeasureResult:
        solver_cache: dict = {}

        def _get_solver(model):
            key = id(model)
            if key not in solver_cache:
                solver_cache[key] = make_solver(model, tol=tol)
            return solver_cache[key]

        def step_fn(model, s0, s1, ctrl):
            solver = _get_solver(model)
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


def run(
    ns: list[int],
    tols: list[float],
    steps: int,
    warmup: int,
    trials: int = 1,
    seed: int = SEED,
) -> dict:
    """Sweep (tol, N) on randomized-IC chaotic scenes."""
    data: dict = {
        "ns": ns,
        "tols": tols,
        "steps": steps,
        "warmup": warmup,
        "trials": trials,
        "seed": seed,
        "curves": {},
        "exponents": {},
    }

    for tol in tols:
        key = f"{tol:.0e}"
        curve: dict = {
            "medians": [], "p25": [], "p75": [],
            "k_means": [], "k_maxes": [], "k_p25s": [], "k_p75s": [],
            "per_iter_medians": [],
        }
        print(f"\n  tol={tol:.0e}", flush=True)
        for n in ns:
            r = _measure_tol_n(tol, n, steps, warmup, trials, seed=seed)
            curve["medians"].append(r.median)
            curve["p25"].append(r.p25)
            curve["p75"].append(r.p75)
            curve["k_means"].append(r.k_mean)
            curve["k_maxes"].append(r.k_max)
            curve["k_p25s"].append(r.k_p25)
            curve["k_p75s"].append(r.k_p75)
            curve["per_iter_medians"].append(r.per_iter_median)
            print(
                f"    N={n:>5}  median={r.median * 1e3:7.2f} ms  "
                f"per_iter={r.per_iter_median * 1e3:7.2f} ms  "
                f"K_mean={r.k_mean:.1f}  K_max={r.k_max}",
                flush=True,
            )
        data["curves"][key] = curve
        data["exponents"][key] = power_law_exponent(ns, curve["medians"])

    return data


def _tol_color(i: int) -> str:
    return _TOL_COLORS[i % len(_TOL_COLORS)]


def _plot_lines_vs_n(
    ax, ns, tols, curves, y_key, *, scale: float = 1.0, ls: str = "-",
    fill_lo: str | None = None, fill_hi: str | None = None,
) -> None:
    """Draw one line per tol on (ax) using ``curves[tol_key][y_key]`` for y."""
    for i, tol in enumerate(tols):
        key = f"{tol:.0e}"
        c = _tol_color(i)
        curve = curves[key]
        ys = [v * scale for v in curve[y_key]]
        ax.plot(ns, ys, color=c, marker="o", ls=ls, lw=1.8, ms=5, label=_tol_label(tol))
        if fill_lo and fill_hi and fill_lo in curve and fill_hi in curve:
            lo = [v * scale for v in curve[fill_lo]]
            hi = [v * scale for v in curve[fill_hi]]
            ax.fill_between(ns, lo, hi, color=c, alpha=0.12)


def plot(data: dict, out_dir: Path) -> None:
    """Generate randomized-IC scaling plots, one curve per tolerance."""
    ns = data["ns"]
    tols = data["tols"]
    curves = data["curves"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Wall time per outer step vs N.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_lines_vs_n(ax, ns, tols, curves, "medians", scale=1e3,
                     fill_lo="p25", fill_hi="p75")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Wall time per outer step [ms]", fontsize=11)
    ax.set_title(
        f"Randomized ICs: wall time vs N  (DT_outer={DT_OUTER * 1e3:.0f} ms, seed={data['seed']})",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", title="tolerance")
    save_fig(fig, out_dir / "ic_wall_time.png")

    # Plot 2: Per-iteration cost vs N.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_lines_vs_n(ax, ns, tols, curves, "per_iter_medians", scale=1e3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Wall time per iteration [ms]", fontsize=11)
    ax.set_title("Randomized ICs: per-iteration GPU cost vs N", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", title="tolerance")
    save_fig(fig, out_dir / "ic_per_iter.png")

    # Plot 3: Iteration count K vs N.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, tol in enumerate(tols):
        key = f"{tol:.0e}"
        c = _tol_color(i)
        curve = curves[key]
        ax.plot(ns, curve["k_means"], color=c, marker="o", ls="-", lw=2, ms=5,
                label=f"{_tol_label(tol)}  $K_{{mean}}$")
        ax.fill_between(ns, curve["k_p25s"], curve["k_p75s"], color=c, alpha=0.10)
        ax.plot(ns, curve["k_maxes"], color=c, marker="o", ls=":", lw=1, ms=3,
                alpha=0.55, label=f"{_tol_label(tol)}  $K_{{max}}$")
    ax.axhline(1, color="grey", ls=":", lw=1, label="K = 1 (ideal)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Iterations per step_dt call", fontsize=11)
    ax.set_title("Randomized ICs: iteration count K vs N", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    save_fig(fig, out_dir / "ic_iterations.png")

    # Plot 4: Amortization — wall time per world per outer step.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, tol in enumerate(tols):
        key = f"{tol:.0e}"
        c = _tol_color(i)
        curve = curves[key]
        ys = [m / n_val * 1e3 for m, n_val in zip(curve["medians"], ns)]
        ax.plot(ns, ys, color=c, marker="o", ls="-", lw=1.8, ms=5, label=_tol_label(tol))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Wall time per world per outer step [ms]", fontsize=11)
    ax.set_title("Randomized ICs: GPU amortization cost per world vs N", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right", title="tolerance")
    save_fig(fig, out_dir / "ic_amortization.png")

    # Summary table.
    print(f"\n{'=' * 72}")
    print("RANDOMIZED-IC TOLERANCE × N SUMMARY")
    print(f"{'=' * 72}")
    hdr = (
        f"{'tol':>10}  {'exponent':>10}  "
        f"{'N=' + str(ns[0]):>10}  {'N=' + str(ns[-1]):>10}  {'ratio':>6}  {'K(N_max)':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tol in tols:
        key = f"{tol:.0e}"
        curve = curves[key]
        exp = data["exponents"][key]
        t1 = curve["medians"][0] * 1e3
        tN = curve["medians"][-1] * 1e3
        k = curve["k_means"][-1]
        print(f"{key:>10}  N^{exp:<7.3f}   {t1:9.2f}  {tN:10.2f}  {tN / t1:5.1f}x  {k:8.1f}")


def main():
    parser = argparse.ArgumentParser(description="Randomized-IC tolerance × N benchmark")
    parser.add_argument("--ns", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--tols", type=float, nargs="+",
                        default=[1e-2, 1e-3, 1e-4])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run(
        ns=sorted(args.ns),
        tols=sorted(args.tols, reverse=True),  # loose → tight
        steps=args.steps,
        warmup=args.warmup,
        trials=args.trials,
        seed=args.seed,
    )

    with open(out_dir / "initial_conditions.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'initial_conditions.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
