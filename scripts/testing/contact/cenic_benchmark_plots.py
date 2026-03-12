"""CENIC benchmark plots.

Generates 6 figures:
  1. Wall time vs error tolerance  (fixed N)
  2. Wall time vs N worlds         (fixed tol)
  3. Cost per world vs N worlds    (GPU amortization curve)
  4. RMS error over sim time       (fixed N, several tol values)
  5. Adaptive dt over sim time     (fixed N, several tol values)
  6. GPU component breakdown       (mujoco_warp vs device-transfer overhead vs N)

Usage:
    uv run python scripts/testing/contact/cenic_benchmark_plots.py
    uv run python scripts/testing/contact/cenic_benchmark_plots.py --out-dir /tmp/cenic_plots
    uv run python scripts/testing/contact/cenic_benchmark_plots.py --quick
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from scripts.testing.contact.cenic_contact_objects import DT_OUTER, DT_INNER_MIN, build_model, make_solver
from scripts.testing.contact.cenic_scaling_diag import measure_n as _measure_component_n

_build_model = build_model
_make_solver = lambda model, tol=1e-3: make_solver(model, tol=tol)


def bench_wall_vs_tol(tols, n_worlds=1, outer_steps=100, warmup=20):
    """Returns list of mean wall-time-per-step [s] for each tol."""
    print(f"\n[Fig 1] Wall time vs tol  N={n_worlds}", flush=True)
    model = _build_model(n_worlds)
    results = []
    for tol in tols:
        solver  = _make_solver(model, tol=tol)
        s0, s1  = model.state(), model.state()
        ctrl    = model.control()
        for _ in range(warmup):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(outer_steps):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        mean_ms = elapsed / outer_steps * 1e3
        print(f"  tol={tol:.0e}  {mean_ms:.2f} ms/step", flush=True)
        results.append(mean_ms)
    return results


def bench_wall_vs_n(ns, tol=1e-3, outer_steps=50, warmup=15):
    """Returns list of mean wall-time-per-step [s] for each N."""
    print(f"\n[Fig 2] Wall time vs N worlds  tol={tol:.0e}", flush=True)
    results = []
    for n in ns:
        model  = _build_model(n)
        solver = _make_solver(model, tol=tol)
        s0, s1 = model.state(), model.state()
        ctrl   = model.control()
        for _ in range(warmup):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(outer_steps):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        mean_ms = elapsed / outer_steps * 1e3
        print(f"  N={n:>5}  {mean_ms:.2f} ms/step", flush=True)
        results.append(mean_ms)
    return results


def trace_error_and_dt(tols, n_worlds=1, sim_duration=1.0, warmup=20):
    """Returns dict tol → (sim_times, errors, dts) for world 0."""
    print(f"\n[Fig 4/5] Error & dt traces  N={n_worlds}  sim={sim_duration}s", flush=True)
    traces = {}
    for tol in tols:
        model  = _build_model(n_worlds)
        solver = _make_solver(model, tol=tol)
        s0, s1 = model.state(), model.state()
        ctrl   = model.control()

        for _ in range(warmup):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)

        sim_times, errors, dts = [], [], []
        t = solver.sim_time.numpy()[0]
        while t < sim_duration:
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
            t = solver.sim_time.numpy()[0]
            sim_times.append(t)
            errors.append(solver.last_error.numpy()[0])
            dts.append(solver.dt.numpy()[0])

        traces[tol] = (np.array(sim_times), np.array(errors), np.array(dts))
        print(f"  tol={tol:.0e}  {len(sim_times)} outer steps", flush=True)
    return traces


STYLE = {"linewidth": 1.8, "marker": "o", "markersize": 4}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved → {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".", help="Directory to write PNG files")
    parser.add_argument("--quick",   action="store_true", help="Fewer points for fast iteration")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.quick:
        tols         = [1e-1, 1e-2, 1e-3]
        ns           = [1, 4, 16, 64, 256, 512, 1000]
        ns_diag      = [1, 4, 16, 64, 256, 512, 1000]
        outer_steps  = 20
        warmup       = 10
        sim_duration = 3.0
    else:
        tols         = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
        ns           = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000]
        ns_diag      = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000]
        outer_steps  = 50
        warmup       = 20
        sim_duration = 3.0

    tol_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tols)))

    wall_vs_tol = bench_wall_vs_tol(tols, n_worlds=1, outer_steps=outer_steps, warmup=warmup)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([str(f"{t:.0e}") for t in tols], wall_vs_tol, **STYLE, color="tab:blue")
    ax.set_xlabel("Error tolerance")
    ax.set_ylabel("Wall time per step_dt [ms]")
    ax.set_title("Wall time vs tolerance  (N=1)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    _save(fig, out / "fig1_wall_vs_tol.png")

    wall_vs_n = bench_wall_vs_n(ns, tol=1e-3, outer_steps=outer_steps, warmup=warmup)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, wall_vs_n, **STYLE, color="black")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Wall time per step_dt [ms]")
    ax.set_title("Wall time vs N worlds  (tol=1e-3)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig2_wall_vs_n.png")

    # Cost per world falls as the GPU fills (amortization), then rises past saturation.
    cost_per_world = [w / n for w, n in zip(wall_vs_n, ns)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, cost_per_world, **STYLE, color="tab:orange")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Wall time per world per step_dt [ms]")
    ax.set_title("GPU amortization: cost per world vs N  (tol=1e-3)")
    ax.set_xscale("log", base=2)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.3)
    opt_idx = int(np.argmin(cost_per_world))
    ax.axvline(ns[opt_idx], color="grey", linestyle="--", linewidth=1.0,
               label=f"optimal N ≈ {ns[opt_idx]}")
    ax.legend(fontsize=8)
    _save(fig, out / "fig3_cost_per_world.png")

    traces = trace_error_and_dt(tols, n_worlds=1, sim_duration=sim_duration, warmup=warmup)

    fig, ax = plt.subplots(figsize=(9, 4))
    for (tol, (ts, errs, _)), color in zip(traces.items(), tol_colors):
        ax.plot(ts, errs, label=f"tol={tol:.0e}", color=color, linewidth=1.2, alpha=0.85)
        ax.axhline(tol, color=color, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("RMS error (world 0)")
    ax.set_title("Integration error over time  (N=1)  [dashed = tolerance target]")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig4_error_over_time.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    for (tol, (ts, _, dts)), color in zip(traces.items(), tol_colors):
        ax.plot(ts, np.array(dts) * 1e3, label=f"tol={tol:.0e}", color=color, linewidth=1.2, alpha=0.85)
    ax.axhline(DT_INNER_MIN * 1e3, color="grey", linestyle=":", linewidth=1.0, label=f"dt_min={DT_INNER_MIN*1e3:.1f} ms")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Adaptive dt [ms]")
    ax.set_title("Step size over time  (N=1)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig5_dt_over_time.png")

    print("\n[Fig 6] GPU component breakdown  ns_diag=" + str(ns_diag), flush=True)
    diag_results = [_measure_component_n(n, steps=outer_steps, warmup=warmup) for n in ns_diag]

    _COMPONENTS = [
        ("3x_substep",      "3× MuJoCo step (physics)",         "tab:blue",   "-",  2.0),
        ("mujoco_warp",     "  mujoco_warp kernel (×1)",        "tab:cyan",   "--", 1.2),
        ("error_control",   "error control + state select",     "tab:orange", "-",  1.5),
        ("update_mjc_data", "update_mjc_data (×1)",             "tab:green",  "--", 1.2),
        ("update_newton",   "update_newton_state (×1)",         "tab:red",    "--", 1.2),
        ("boundary_numpy",  "boundary_flag.numpy() — 1 int32", "tab:purple", ":",  1.8),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color, ls, lw in _COMPONENTS:
        ys = [r[key] * 1e3 for r in diag_results]
        ax.plot(ns_diag, ys, label=label, color=color, linestyle=ls, linewidth=lw,
                marker="o", markersize=4)

    ax.set_xlabel("N parallel worlds")
    ax.set_ylabel("Wall time per step_dt call [ms]")
    ax.set_title("GPU component breakdown: MuJoCo physics kernel vs device-transfer overhead")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig6_component_breakdown.png")

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
