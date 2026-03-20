"""CENIC benchmark plots.

Generates 6 figures: wall time vs tol, wall time vs N, GPU amortization,
L2 error trace, adaptive dt trace, and GPU component breakdown.

Usage:
    uv run python scripts/testing/contact/cenic_benchmark_plots.py
    uv run python scripts/testing/contact/cenic_benchmark_plots.py --out-dir /tmp/cenic_plots
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


def _compile_kernels():
    """Run a few steps with N=1 so all Warp kernels are JIT-compiled before timing."""
    model  = _build_model(1)
    solver = _make_solver(model)
    s0, s1 = model.state(), model.state()
    ctrl   = model.control()
    for _ in range(5):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    wp.synchronize()


def _time_sim(n, tol, sim_duration):
    """Build a fresh solver, run exactly sim_duration/DT_OUTER steps, return ms/sim-s."""
    model   = _build_model(n)
    solver  = _make_solver(model, tol=tol)
    s0, s1  = model.state(), model.state()
    ctrl    = model.control()
    n_steps = round(sim_duration / DT_OUTER)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    wp.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / sim_duration * 1e3, n_steps


def bench_wall_vs_tol(tols, n_worlds=1, sim_duration=3.0):
    """Returns list of mean ms/sim-second for each tol (3 s simulation, includes contact)."""
    print(f"\n[Fig 1] Wall time vs tol  N={n_worlds}  sim={sim_duration}s", flush=True)
    _compile_kernels()
    results = []
    for tol in tols:
        ms, steps = _time_sim(n_worlds, tol, sim_duration)
        print(f"  tol={tol:.0e}  {ms:.1f} ms/sim-s  ({steps} steps)", flush=True)
        results.append(ms)
    return results


def bench_wall_vs_n(ns, tol=1e-3, sim_duration=3.0):
    """Returns list of mean ms/sim-second for each N (3 s simulation, includes contact)."""
    print(f"\n[Fig 2] Wall time vs N worlds  tol={tol:.0e}  sim={sim_duration}s", flush=True)
    _compile_kernels()
    results = []
    for n in ns:
        ms, steps = _time_sim(n, tol, sim_duration)
        print(f"  N={n:>5}  {ms:.1f} ms/sim-s  ({steps} steps)", flush=True)
        results.append(ms)
    return results


def trace_error_and_dt(tol=1e-5, n_worlds=1, sim_duration=3.0):
    """Returns (sim_times, errors, dts) for world 0 over a 3 s simulation."""
    print(f"\n[Fig 4/5] Error & dt traces  N={n_worlds}  tol={tol:.0e}  sim={sim_duration}s", flush=True)
    model  = _build_model(n_worlds)
    solver = _make_solver(model, tol=tol)
    s0, s1 = model.state(), model.state()
    ctrl   = model.control()

    sim_times, errors, dts = [], [], []
    for _ in range(round(sim_duration / DT_OUTER)):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        sim_times.append(solver.sim_time.numpy()[0])
        errors.append(solver.last_error.numpy()[0])
        dts.append(solver.dt.numpy()[0])

    print(f"  {len(sim_times)} outer steps", flush=True)
    return np.array(sim_times), np.array(errors), np.array(dts)


STYLE = {"linewidth": 1.8, "marker": "o", "markersize": 4}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved → {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".", help="Directory to write PNG files")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tols         = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    ns           = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000]
    ns_diag      = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000]
    outer_steps  = 200
    sim_duration = 2.0

    wall_vs_tol = bench_wall_vs_tol(tols, n_worlds=1, sim_duration=sim_duration)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tols, wall_vs_tol, **STYLE, color="tab:blue")
    ax.set_xlabel("Error tolerance")
    ax.set_ylabel("Wall time per sim-second [ms/sim-s]")
    ax.set_title("Wall time vs tolerance  (N=1, 2 s sim including contact)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig1_wall_vs_tol.png")

    wall_vs_n = bench_wall_vs_n(ns, tol=1e-3, sim_duration=sim_duration)

    # Power-law fit in the pre-saturation regime (N <= 128).
    sat = 128
    mask = [i for i, n in enumerate(ns) if n <= sat]
    ns_fit = np.array([ns[i] for i in mask])
    ws_fit = np.array([wall_vs_n[i] for i in mask])
    slope, intercept = np.polyfit(np.log(ns_fit), np.log(ws_fit), 1)
    fit_x = np.geomspace(ns_fit[0], ns_fit[-1], 100)
    fit_y = np.exp(intercept) * fit_x ** slope

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, wall_vs_n, **STYLE, color="black")
    ax.plot(fit_x, fit_y, color="tab:red", linestyle="--", linewidth=1.2,
            label=f"fit N <= {sat}: t ~ N^{{{slope:.2f}}}")
    ax.axvline(sat, color="grey", linestyle=":", linewidth=0.8, label=f"saturation N={sat}")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Wall time per sim-second [ms/sim-s]")
    ax.set_title("Wall time vs N worlds  (tol=1e-3, 2 s sim including contact)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig2_wall_vs_n.png")

    cost_per_world = [w / n for w, n in zip(wall_vs_n, ns)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, cost_per_world, **STYLE, color="tab:orange")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Wall time per world per sim-second [ms/sim-s]")
    ax.set_title("GPU amortization: cost per world vs N  (tol=1e-3)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    opt_idx = int(np.argmin(cost_per_world))
    ax.axvline(ns[opt_idx], color="grey", linestyle="--", linewidth=1.0,
               label=f"optimal N ≈ {ns[opt_idx]}")
    ax.legend(fontsize=8)
    _save(fig, out / "fig3_cost_per_world.png")

    trace_tol = 1e-3
    ts, errs, dts = trace_error_and_dt(tol=trace_tol, n_worlds=1, sim_duration=sim_duration)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, errs, color="tab:blue", linewidth=1.2)
    ax.axhline(trace_tol, color="tab:blue", linestyle="--", linewidth=0.8,
               label=f"tolerance = {trace_tol:.0e}")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("L2 error (world 0)")
    ax.set_title(f"Integration error over time  (N=1, tol={trace_tol:.0e})")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig4_error_over_time.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, np.array(dts) * 1e3, color="tab:orange", linewidth=1.2)
    ax.axhline(DT_INNER_MIN * 1e3, color="grey", linestyle=":", linewidth=1.0,
               label=f"dt_inner_min = {DT_INNER_MIN*1e3:.2f} ms")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("dt_inner [ms]")
    ax.set_title(f"Adaptive inner step size over time  (N=1, tol={trace_tol:.0e})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig5_dt_over_time.png")

    print("\n[Fig 6] GPU component breakdown  ns_diag=" + str(ns_diag), flush=True)
    diag_results = [_measure_component_n(n, steps=outer_steps, warmup=20) for n in ns_diag]

    _COMPONENTS = [
        ("3x_substep",      "3× MuJoCo step (uncaptured)",         "tab:blue",   "-",  2.0),
        ("mujoco_warp",     "  mujoco_warp kernel (×1)",           "tab:cyan",   "--", 1.2),
        ("graph_replay",    "CUDA graph replay (full inner step)",  "tab:orange", "-",  1.5),
        ("update_mjc_data", "update_mjc_data (×1)",                "tab:green",  "--", 1.2),
        ("update_newton",   "update_newton_state (×1)",            "tab:red",    "--", 1.2),
        ("boundary_numpy",  "boundary_flag.numpy() — 1 int32",    "tab:purple", ":",  1.8),
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
